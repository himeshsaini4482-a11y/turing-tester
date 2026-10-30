"""Stage 2: generate cloned-voice TTS for every benchmark sentence.

Each sentence is spoken in the voice of its LibriTTS speaker, cloned from that row's
prompt clip (same speaker and chapter, never the sentence itself).

Writes <run-dir>/tts/<sentence_id>.wav, <run-dir>/manifest.csv (input rows + tts_wav)
and <run-dir>/generation_config.json.

Runs in the TTS environment (.venv-tts). Resumable: existing output wavs are skipped.
"""
import argparse
import json
import os
import time
import zlib
from collections import defaultdict
from importlib.metadata import version

from common import read_manifest, resolve_attn, write_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                   help="HF model id or local checkpoint dir (e.g. a trained checkpoint)")
    p.add_argument("--language", default="English")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (smoke test)")
    p.add_argument("--speaker", default="",
                   help="a fine-tuned (custom_voice) checkpoint: speak as this built-in speaker instead of cloning")
    args = p.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

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
        model = Qwen3TTSModel.from_pretrained(
            args.model, device_map=args.device, dtype=torch.bfloat16, attn_implementation=attn
        )
        by_prompt = defaultdict(list)
        for r in todo:
            by_prompt[r["prompt_wav"]].append(r)

        start, done = time.time(), 0
        for prompt_wav, group in by_prompt.items():
            if not args.speaker:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=prompt_wav, ref_text=group[0]["prompt_text"], x_vector_only_mode=False
                )
            for i in range(0, len(group), args.batch_size):
                batch = group[i : i + args.batch_size]
                # Seed from the batch's first sentence so base and trained runs sample identically
                torch.manual_seed(args.seed + zlib.crc32(batch[0]["sentence_id"].encode()) % 100_000)
                if args.speaker:   # Qwen3-TTS fine-tuned on one voice (sft/train_qwen3.py)
                    wavs, sr = model.generate_custom_voice(
                        text=[r["text"] for r in batch],
                        language=[args.language] * len(batch),
                        speaker=[args.speaker] * len(batch),
                        max_new_tokens=args.max_new_tokens,
                    )
                    for r, wav in zip(batch, wavs):
                        sf.write(r["tts_wav"], wav, sr)
                    done += len(batch)
                    continue
                wavs, sr = model.generate_voice_clone(
                    text=[r["text"] for r in batch],
                    language=[args.language] * len(batch),
                    voice_clone_prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                for r, wav in zip(batch, wavs):
                    sf.write(r["tts_wav"], wav, sr)
                done += len(batch)
                elapsed = time.time() - start
                eta = elapsed / done * (len(todo) - done)
                print(f"[{done}/{len(todo)}] {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left", flush=True)

    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "language": args.language,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "speaker": args.speaker or None,
            "attn_implementation": attn,
            "qwen_tts_version": version("qwen-tts"),
            "torch_version": torch.__version__,
            "source_manifest": args.manifest,
            "n_sentences": len(rows),
        }, f, indent=2)


if __name__ == "__main__":
    main()
