"""Round stage R-4a: the teacher (Orpheus 3B, voice-cloned) speaks the round's texts, time-boxed.

Each text of runs/sft/<stage>/texts.jsonl gets a voice round-robin (by its position in the file)
from the non-dev EARS speakers in sorted order. Orpheus clones the speaker from their real
rainbow_01_regular reading (inventory path + transcript), with the prompt format, special tokens
and SNAC encode/decode of benchmark/generate_orpheus.py (imported, not copied):

  [SOH] ref text [EOT EOH SOA SOS] ref audio codes [EOS EOA] [SOH] text [EOT EOH SOA SOS] -> codes

Batched HF generation: prompts are left-padded with EOS plus an attention mask (HF Llama derives
position ids from the mask). Pending texts are shuffled with a fixed seed, then sorted by length
inside windows of 8 batches, so batches are length-homogeneous while any time-boxed prefix keeps
the category mix. max_new_tokens per batch = max over the batch of
(words / 2.6 + 1) s x 82 tokens/s x 1.6 + 100, capped at 2400. Each output is cut at its first EOS
(hit_eos records whether it had one), decoded with SNAC and written at 24 kHz. A decode error
only drops that sample (logged in failed.jsonl). Sampling as generate_orpheus.py: T 0.6, top-p
0.95, repetition penalty 1.1; the seed is set per batch.

Time box: no new batch starts once used + mean batch time would exceed --gpu-hours, where used =
gen_seconds already in generated.jsonl/failed.jsonl (earlier runs) + this run's wall time.

Writes <out-dir>/wav/<id>.wav, <out-dir>/generated.jsonl (INTERFACES §2.7; gen_seconds = the
batch's wall time / batch size), <out-dir>/failed.jsonl and <out-dir>/generation_config.json, and
logs throughput (audio s per GPU s) to runs/sft/timings.json.
Env: TTS (.venv-tts). Resumable: ids already in generated.jsonl or failed.jsonl are skipped.

  python sft/teacher.py --texts runs/sft/pilot/texts.jsonl --out-dir runs/sft/pilot/teacher --gpu-hours 4
"""
import argparse
import os
import random
import sys
import time
import zlib
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

TOKENS_PER_S = 82        # Orpheus: 7 SNAC tokens per ~12 frames/s
WORDS_PER_S = 2.6
LENGTH_MARGIN = 1.6
TOKEN_SLACK = 100
MAX_NEW_TOKENS = 2400
WINDOW_BATCHES = 8


# ---- pure planning logic (unit-tested without torch) ---------------------------------------

def est_new_tokens(text):
    seconds = len(text.split()) / WORDS_PER_S + 1
    return min(MAX_NEW_TOKENS, int(seconds * TOKENS_PER_S * LENGTH_MARGIN + TOKEN_SLACK))


def batch_max_new_tokens(batch):
    return max(est_new_tokens(it["text"]) for it in batch)


def plan_batches(items, batch_size, seed, window_batches=WINDOW_BATCHES):
    """Shuffle (fixed seed), then sort by length inside windows -> list of batches."""
    items = list(items)
    random.Random(seed).shuffle(items)
    window = batch_size * window_batches
    batches = []
    for i in range(0, len(items), window):
        chunk = sorted(items[i:i + window], key=lambda it: (len(it["text"]), it["id"]))
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    return batches


def load_voices(inventory_path):
    """Non-dev EARS speakers' rainbow_01_regular rows, sorted by speaker."""
    voices = {}
    for r in sftlib.read_jsonl(inventory_path):
        if r.get("name") == sftlib.PROMPT_READING and not r.get("dev_speaker"):
            voices[r["speaker"]] = r
    return [voices[s] for s in sorted(voices)]


def assign_voices(texts, voices):
    """Round-robin over the sorted voices by position in texts.jsonl (stable across resumes)."""
    out = []
    for i, t in enumerate(texts):
        v = voices[i % len(voices)]
        out.append({**t, "speaker": v["speaker"], "ref_wav": v["path"], "ref_text": v["transcript"]})
    return out


def left_pad(seqs, pad_id):
    """-> (input_ids, attention_mask) as lists of equal-length rows, padded on the left."""
    width = max(len(s) for s in seqs)
    ids = [[pad_id] * (width - len(s)) + list(s) for s in seqs]
    mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
    return ids, mask


def cut_at_eos(tokens, eos):
    """Generated tokens up to (not including) the first EOS, and whether there was one."""
    if eos in tokens:
        return tokens[:tokens.index(eos)], True
    return tokens, False


def budget_exhausted(used_s, batch_times, budget_s):
    """Stop starting batches when used + mean batch time would pass the budget."""
    expected = mean(batch_times) if batch_times else 0.0
    return used_s + expected > budget_s


# ---- generation ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--texts", required=True, help="runs/sft/<stage>/texts.jsonl")
    p.add_argument("--out-dir", required=True, help="runs/sft/<stage>/teacher")
    p.add_argument("--gpu-hours", type=float, required=True, help="time box for this round's generation")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--inventory", default=str(sftlib.DATA / "ears_inventory.jsonl"))
    p.add_argument("--voices", default="", help="voices jsonl ({speaker, path, transcript} per line) instead of "
                                                "the inventory's non-dev EARS speakers (turing-tester CLI)")
    p.add_argument("--model", default="unsloth/orpheus-3b-0.1-pretrained")
    # same values as generate_orpheus.py's argparse defaults (they are not importable constants)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="Orpheus needs >= 1.1 to stay stable")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    t_start = time.time()

    out_dir = os.path.abspath(args.out_dir)
    wav_dir = os.path.join(out_dir, "wav")
    gen_path = os.path.join(out_dir, "generated.jsonl")
    fail_path = os.path.join(out_dir, "failed.jsonl")
    os.makedirs(wav_dir, exist_ok=True)

    texts = sftlib.read_jsonl(args.texts)
    if not texts:
        sys.exit(f"no texts in {args.texts}")
    stage = texts[0].get("stage", "")
    if args.voices:
        voices = sorted(sftlib.read_jsonl(args.voices), key=lambda v: v["speaker"])
    else:
        voices = load_voices(args.inventory)
    if not voices:
        sys.exit(f"no non-dev {sftlib.PROMPT_READING} rows in {args.inventory} (run sft/ears_inventory.py)")
    missing = [v["path"] for v in voices if not os.path.exists(v["path"])]
    if missing:
        sys.exit(f"{len(missing)} reference wavs missing, e.g. {missing[0]}: the EARS audio lives on the "
                 f"scratch NVMe and is wiped when the worker stops; re-run sft/ears_inventory.py")

    generated = sftlib.read_jsonl(gen_path)
    failed = sftlib.read_jsonl(fail_path)
    done = {r["id"] for r in generated} | {r["id"] for r in failed}
    used_prior = sum(r.get("gen_seconds", 0) for r in generated + failed)
    todo = [t for t in assign_voices(texts, voices) if t["id"] not in done]
    budget_s = args.gpu_hours * 3600
    print(f"{len(done)} texts already done ({used_prior / 3600:.2f} GPU-h used), {len(todo)} pending; "
          f"budget {args.gpu_hours} GPU-h; {len(voices)} voices", flush=True)
    if not todo or used_prior >= budget_s:
        print("nothing to do", flush=True)
        return

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sftlib.add_benchmark_to_path()
    from common import resolve_attn
    from generate_orpheus import EOA, EOH, EOS, EOT, SOA, SOH, SOS, SR, decode, encode
    from snac import SNAC

    attn = resolve_attn(args.attn)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device, attn_implementation=attn
    ).eval()
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(args.device)

    def turn(text):
        # tok(text).input_ids includes the BOS, exactly as in generate_orpheus.py
        return [SOH] + tok(text).input_ids + [EOT, EOH, SOA, SOS]

    voice_prompt = {}

    def prompt_of(item):
        if item["speaker"] not in voice_prompt:
            voice_prompt[item["speaker"]] = (turn(item["ref_text"]) + encode(snac, item["ref_wav"], args.device)
                                             + [EOS, EOA])
        return voice_prompt[item["speaker"]] + turn(item["text"])

    sftlib.write_json(os.path.join(out_dir, "generation_config.json"), {
        "model": args.model, "codec": "hubertsiuzdak/snac_24khz", "batch_size": args.batch_size,
        "temperature": args.temperature, "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
        "seed": args.seed, "attn_implementation": attn, "torch_version": torch.__version__,
        "max_new_tokens": f"(words/{WORDS_PER_S}+1)*{TOKENS_PER_S}*{LENGTH_MARGIN}+{TOKEN_SLACK}, cap {MAX_NEW_TOKENS}",
        "gpu_hours": args.gpu_hours, "texts": args.texts,
    })

    batches = plan_batches(todo, args.batch_size, args.seed)
    batch_times, n_clips, n_eos, audio_s, n_failed = [], 0, 0, 0.0, 0
    gen_start = time.time()
    stopped_by_budget = False
    for b, batch in enumerate(batches, 1):
        used = used_prior + (time.time() - t_start)
        if budget_exhausted(used, batch_times, budget_s):
            stopped_by_budget = True
            est = mean(batch_times) if batch_times else 0.0
            print(f"time box reached: {used / 3600:.2f} GPU-h used + ~{est:.0f} s/batch "
                  f"> {args.gpu_hours} GPU-h; {len(batches) - b + 1} batches left unstarted", flush=True)
            break
        t0 = time.time()
        ids, mask = left_pad([prompt_of(it) for it in batch], EOS)
        input_ids = torch.tensor(ids, device=args.device)
        attention_mask = torch.tensor(mask, device=args.device)
        max_new = batch_max_new_tokens(batch)
        torch.manual_seed(args.seed + zlib.crc32(batch[0]["id"].encode()) % 100_000)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, eos_token_id=EOS, pad_token_id=EOS,
            )
        new_tokens = out[:, input_ids.shape[1]:].tolist()
        results = []
        for it, toks in zip(batch, new_tokens):
            codes, hit = cut_at_eos(toks, EOS)
            try:
                audio = decode(snac, codes, args.device)
            except Exception as exc:  # noqa: BLE001  (one bad sample must not stop the round)
                results.append((it, None, hit, len(codes), f"{type(exc).__name__}: {exc}"))
                continue
            results.append((it, audio, hit, len(codes), None))
        dt = time.time() - t0
        batch_times.append(dt)
        share = dt / len(batch)
        for it, audio, hit, n_tok, err in results:
            if audio is None:
                n_failed += 1
                sftlib.append_jsonl(fail_path, {"id": it["id"], "error": err[:300], "hit_eos": hit,
                                                "n_tokens": n_tok, "gen_seconds": round(share, 3)})
                print(f"  [warn] {it['id']}: decode failed ({err[:120]})", flush=True)
                continue
            wav = os.path.join(wav_dir, f"{it['id']}.wav")
            sf.write(wav + ".part", audio, SR, format="WAV")
            os.replace(wav + ".part", wav)
            dur = len(audio) / SR
            sftlib.append_jsonl(gen_path, {
                "id": it["id"], "text": it["text"], "category": it["category"], "speaker": it["speaker"],
                "ref_wav": it["ref_wav"], "ref_text": it["ref_text"], "wav": wav, "duration": round(dur, 3),
                "hit_eos": hit, "gen_seconds": round(share, 3), "stage": it.get("stage", stage),
                "n_tokens": n_tok, "batch_size": len(batch),
            })
            n_clips += 1
            n_eos += hit
            audio_s += dur
        gen_s = time.time() - gen_start
        elapsed = time.time() - t_start
        left = min(len(batches) - b, max(0, (budget_s - used_prior - elapsed)) / mean(batch_times))
        print(f"[batch {b}/{len(batches)}] {len(batch)} clips, {max_new} max tokens, {dt:.0f} s; "
              f"{n_clips} clips ({n_eos} with EOS), {audio_s / 60:.1f} min audio, "
              f"{audio_s / gen_s:.2f} audio s/GPU s; {elapsed / 60:.1f} min elapsed, "
              f"~{left * mean(batch_times) / 60:.1f} min left", flush=True)

    elapsed = time.time() - t_start
    gen_s = sum(batch_times)
    if not stopped_by_budget:
        frac = (used_prior + elapsed) / budget_s
        print(f"[note] all texts rendered after {frac:.0%} of the {args.gpu_hours} GPU-h budget; "
              f"more texts would have used the rest", flush=True)
    print(f"done: {n_clips} clips ({n_failed} decode failures), {audio_s / 3600:.2f} h audio in "
          f"{elapsed / 3600:.2f} h", flush=True)
    if n_clips:
        sftlib.log_timing(f"{stage}_teacher_throughput", elapsed, n_clips=n_clips, n_failed=n_failed,
                          audio_s=round(audio_s, 1), gen_s=round(gen_s, 1), batch_size=args.batch_size,
                          audio_per_gpu_s=round(audio_s / elapsed, 4),
                          audio_per_gen_s=round(audio_s / gen_s, 4) if gen_s else None,
                          eos_rate=round(n_eos / n_clips, 4))


if __name__ == "__main__":
    main()
