"""Stage 2 (Orpheus-TTS 3B): zero-shot voice-clone TTS with the pretrained Orpheus model.

Orpheus is Llama-3.2-3B with SNAC (24 kHz) audio codes added as tokens, 7 per frame.
The finetuned release only speaks 8 built-in voices; cloning needs the pretrained model,
prompted with a text-speech pair in the target voice (canopyai/Orpheus-TTS README):

  [SOH] ref text [EOT EOH SOA SOS] ref audio codes [EOS EOA] [SOH] text [EOT EOH SOA SOS] -> audio codes

The model was not trained for cloning explicitly, so this tests its in-context ability.
Canopy's official repo is gated, so the default is unsloth's ungated copy of the weights.

Writes the same outputs as generate_tts.py: <run-dir>/tts/<sentence_id>.wav,
<run-dir>/manifest.csv (input rows + tts_wav) and <run-dir>/generation_config.json.
Runs in the TTS environment (.venv-tts). Resumable: existing output wavs are skipped.
"""
import argparse
import json
import os
import time
import zlib

from common import read_manifest, resolve_attn, write_manifest

# Orpheus special tokens (orpheus_tts_pypi/engine_class.py, decoder.py)
SOH, EOT, EOH, SOA, SOS, EOS, EOA = 128259, 128009, 128260, 128261, 128257, 128258, 128262
AUDIO = 128266   # first audio token; slot k of a 7-token frame adds k * 4096
SR = 24000


def encode(snac, wav_path, device):
    import librosa
    import torch

    wav, _ = librosa.load(wav_path, sr=SR, mono=True)
    with torch.inference_mode():
        c0, c1, c2 = (c[0].tolist() for c in snac.encode(torch.from_numpy(wav)[None, None].to(device)))
    ids = []
    for i in range(len(c0)):
        frame = [c0[i], c1[2 * i], c2[4 * i], c2[4 * i + 1], c1[2 * i + 1], c2[4 * i + 2], c2[4 * i + 3]]
        ids += [AUDIO + k * 4096 + code for k, code in enumerate(frame)]
    return ids


def decode(snac, ids, device):
    import torch

    codes = [t - AUDIO for t in ids if t >= AUDIO]
    codes = codes[: len(codes) // 7 * 7]
    c0, c1, c2 = [], [], []
    for i in range(0, len(codes), 7):
        f = [c - k * 4096 for k, c in enumerate(codes[i : i + 7])]
        c0.append(f[0])
        c1 += [f[1], f[4]]
        c2 += [f[2], f[3], f[5], f[6]]
    if not c0 or not all(0 <= c < 4096 for c in c0 + c1 + c2):
        raise ValueError("generated tokens are not a valid SNAC frame sequence")
    with torch.inference_mode():
        audio = snac.decode([torch.tensor(c, device=device)[None] for c in (c0, c1, c2)])
    return audio[0, 0].float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default="unsloth/orpheus-3b-0.1-pretrained")
    p.add_argument("--max-new-tokens", type=int, default=2400, help="~82 tokens per second of audio")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="Orpheus needs >= 1.1 to stay stable")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (smoke test)")
    args = p.parse_args()

    import soundfile as sf
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    tts_dir = os.path.join(args.run_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    for r in rows:
        r["tts_wav"] = os.path.join(tts_dir, f"{r['sentence_id']}.wav")

    todo = [r for r in rows if not os.path.exists(r["tts_wav"])]
    print(f"{len(rows) - len(todo)} already generated, {len(todo)} to generate")

    attn = resolve_attn(args.attn)
    if todo:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=args.device, attn_implementation=attn
        ).eval()
        snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(args.device)

        def turn(text):
            return [SOH] + tok(text).input_ids + [EOT, EOH, SOA, SOS]

        prompts = {}
        start = time.time()
        for i, r in enumerate(todo, 1):
            if r["prompt_wav"] not in prompts:
                prompts[r["prompt_wav"]] = turn(r["prompt_text"]) + encode(snac, r["prompt_wav"], args.device) + [EOS, EOA]
            input_ids = torch.tensor([prompts[r["prompt_wav"]] + turn(r["text"])], device=args.device)
            # same per-sentence seed scheme as generate_tts.py
            torch.manual_seed(args.seed + zlib.crc32(r["sentence_id"].encode()) % 100_000)
            with torch.inference_mode():
                out = model.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty, eos_token_id=EOS, pad_token_id=EOS,
                )
            new = out[0, input_ids.shape[1] :].tolist()
            if EOS not in new:
                print(f"  warning: {r['sentence_id']} hit max_new_tokens without end-of-speech", flush=True)
            sf.write(r["tts_wav"], decode(snac, new, args.device), SR)
            elapsed = time.time() - start
            print(f"[{i}/{len(todo)}] {r['sentence_id']}  {elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "codec": "hubertsiuzdak/snac_24khz",
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "attn_implementation": attn,
            "torch_version": torch.__version__,
            "source_manifest": args.manifest,
            "n_sentences": len(rows),
        }, f, indent=2)


if __name__ == "__main__":
    main()
