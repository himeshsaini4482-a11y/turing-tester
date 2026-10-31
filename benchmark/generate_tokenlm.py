"""Stage 2 (LLM-style TTS with turing-tester.yaml): voice-clone generation, one sentence at a time.

The model folder holds a Hugging Face causal LM + tokenizer and a turing-tester.yaml that names the
audio codec and the prompt layout (benchmark/tokenlm.py, docs/adding_a_model.md). Same loop as
generate_orpheus.py: prompt = the yaml's prompt with the row's reference clip and text, sample
until stop_token or max_new_tokens, decode the audio tokens with the codec.

Writes the same outputs as generate_tts.py. Resumable: existing wavs are skipped.
"""
import argparse
import json
import os
import time
import zlib

from common import read_manifest, resolve_attn, write_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--attn", default="auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM

    from tokenlm import TokenLM

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    tts_dir = os.path.join(args.run_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    for r in rows:
        r["tts_wav"] = os.path.join(tts_dir, f"{r['sentence_id']}.wav")
    todo = [r for r in rows if not os.path.exists(r["tts_wav"])]
    print(f"{len(rows) - len(todo)} already generated, {len(todo)} to generate", flush=True)

    lm = TokenLM(args.model, args.device)
    attn = resolve_attn(args.attn)
    if todo:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=attn
                                                     ).to(args.device).eval()
        sampling = lm.sampling()
        start = time.time()
        for i, r in enumerate(todo, 1):
            ids = lm.prompt(r["text"], r["prompt_wav"] if lm.clones() else None, r["prompt_text"])
            input_ids = torch.tensor([ids], device=args.device)
            wav = None
            for attempt in range(3):     # an undecodable sample is drawn again with the next seed
                torch.manual_seed(args.seed + zlib.crc32(r["sentence_id"].encode()) % 100_000 + attempt)
                with torch.inference_mode():
                    out = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                         max_new_tokens=lm.max_new_tokens(), do_sample=True,
                                         eos_token_id=lm.stop, pad_token_id=lm.pad, **sampling)
                tokens, ended = lm.cut(out[0, input_ids.shape[1]:].tolist())
                if not ended:
                    print(f"  warning: {r['sentence_id']} hit max_new_tokens without the stop token", flush=True)
                try:
                    wav = lm.decode(tokens)
                    break
                except ValueError as e:
                    print(f"  warning: {r['sentence_id']} attempt {attempt + 1}: {e}", flush=True)
            if wav is None:
                raise SystemExit(f"{r['sentence_id']}: no decodable sample in 3 attempts")
            sf.write(r["tts_wav"] + ".part.wav", wav, lm.codec.sample_rate)
            os.replace(r["tts_wav"] + ".part.wav", r["tts_wav"])
            elapsed = time.time() - start
            print(f"[{i}/{len(todo)}] {r['sentence_id']}  {elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": os.path.abspath(args.model), "generator": "token-lm (turing-tester.yaml)",
                   "codec": lm.spec["codec"], "sampling": lm.sampling(), "max_new_tokens": lm.max_new_tokens(),
                   "seed": args.seed, "attn_implementation": attn, "torch_version": torch.__version__,
                   "source_manifest": args.manifest, "n_sentences": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
