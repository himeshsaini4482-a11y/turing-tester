"""Stage 2 (MOSS-TTS-Nano): zero-shot voice-clone TTS with OpenMOSS's 0.1B MOSS-TTS-Nano.

MOSS-TTS-Nano pins its own torch/transformers (2.7.0 / 4.57.1), so scripts/run_moss_nano.sh
gives it a separate venv and a checkout of github.com/OpenMOSS/MOSS-TTS-Nano (--repo-dir),
whose text normalization this script reuses. Generation follows the repo's infer.py:
voice_clone mode (prompt audio + target text; the prompt transcript is not used), its
default sampling settings, robust text cleanup on, WeTextProcessing off (needs pynini).

The model writes 48 kHz stereo; it is downmixed to mono here like the other models' output.

Writes the same outputs as generate_tts.py: <run-dir>/tts/<sentence_id>.wav,
<run-dir>/manifest.csv (input rows + tts_wav) and <run-dir>/generation_config.json.
Resumable: existing output wavs are skipped.
"""
import argparse
import json
import os
import sys
import time
import zlib

from common import read_manifest, write_manifest

# infer.py's defaults (resolve_sampling_kwargs)
SAMPLING = {
    "text_temperature": 1.0, "text_top_p": 1.0, "text_top_k": 50,
    "audio_temperature": 0.8, "audio_top_p": 0.95, "audio_top_k": 25, "audio_repetition_penalty": 1.2,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--repo-dir", required=True, help="checkout of github.com/OpenMOSS/MOSS-TTS-Nano")
    p.add_argument("--model", default="OpenMOSS-Team/MOSS-TTS-Nano-100M")
    p.add_argument("--audio-tokenizer", default="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
    p.add_argument("--max-new-frames", type=int, default=375, help="infer.py default")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (smoke test)")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo_dir))
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM
    from text_normalization_pipeline import prepare_tts_request_texts

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    tts_dir = os.path.join(args.run_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    for r in rows:
        r["tts_wav"] = os.path.join(tts_dir, f"{r['sentence_id']}.wav")

    todo = [r for r in rows if not os.path.exists(r["tts_wav"])]
    print(f"{len(rows) - len(todo)} already generated, {len(todo)} to generate")

    device = torch.device(args.device)
    if todo:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
        model.to(device=device, dtype=torch.bfloat16)
        model._set_attention_implementation("sdpa")
        model.eval()
        # load the tokenizers once instead of on every inference() call
        text_tokenizer = model._load_text_tokenizer(text_tokenizer=None, text_tokenizer_path=None)
        audio_tokenizer = model._load_audio_tokenizer(
            audio_tokenizer=None, audio_tokenizer_type="moss-audio-tokenizer-nano",
            audio_tokenizer_pretrained_name_or_path=args.audio_tokenizer, device=device,
        )

        start = time.time()
        for i, r in enumerate(todo, 1):
            text = str(prepare_tts_request_texts(
                text=r["text"], enable_wetext=False, enable_normalize_tts_text=True, text_normalizer_manager=None,
            )["text"])
            # same per-sentence seed scheme as generate_tts.py
            seed = args.seed + zlib.crc32(r["sentence_id"].encode()) % 100_000
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            part = r["tts_wav"] + ".part.wav"
            with torch.inference_mode():
                result = model.inference(
                    text=text, output_audio_path=part, mode="voice_clone", prompt_audio_path=r["prompt_wav"],
                    text_tokenizer=text_tokenizer, audio_tokenizer=audio_tokenizer, device=device,
                    max_new_frames=args.max_new_frames, do_sample=True, use_kv_cache=True, **SAMPLING,
                )
            wav, sr = sf.read(result["audio_path"])
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            sf.write(r["tts_wav"], wav, sr)
            os.remove(result["audio_path"])
            elapsed = time.time() - start
            print(f"[{i}/{len(todo)}] {r['sentence_id']}  {len(wav) / sr:.1f} s audio, {elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "audio_tokenizer": args.audio_tokenizer,
            "mode": "voice_clone",
            "max_new_frames": args.max_new_frames,
            "sampling": SAMPLING,
            "text_normalization": "normalize_tts_text (WeTextProcessing off)",
            "seed": args.seed,
            "output": "48 kHz stereo downmixed to mono",
            "torch_version": torch.__version__,
            "source_manifest": args.manifest,
            "n_sentences": len(rows),
        }, f, indent=2)


if __name__ == "__main__":
    main()
