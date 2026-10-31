"""Stage 2 (your own model): call the model folder's synthesize.py for every benchmark sentence.

synthesize.py (docs/adding_a_model.md) defines
  load(device)                                  called once (device is "cuda:0", "mps" or "cpu")
  synthesize(text, ref_audio, ref_text)         -> (mono float numpy array, sample rate)
and optionally CLONES_VOICE = False when the model cannot clone (ref_audio is then None).

The folder is put on sys.path, so synthesize.py can import its own helper modules.
Writes the same outputs as generate_tts.py: <run-dir>/tts/<sentence_id>.wav, <run-dir>/manifest.csv
and <run-dir>/generation_config.json. Resumable: existing wavs are skipped.
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import zlib

from common import read_manifest, write_manifest


def load_module(folder):
    folder = os.path.abspath(folder)
    sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location("synthesize", os.path.join(folder, "synthesize.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "synthesize"):
        sys.exit(f"{folder}/synthesize.py must define synthesize(text, ref_audio, ref_text)")
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", required=True, help="folder with synthesize.py")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import numpy as np
    import soundfile as sf

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    tts_dir = os.path.join(args.run_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    for r in rows:
        r["tts_wav"] = os.path.join(tts_dir, f"{r['sentence_id']}.wav")
    todo = [r for r in rows if not os.path.exists(r["tts_wav"])]
    print(f"{len(rows) - len(todo)} already generated, {len(todo)} to generate", flush=True)

    mod = load_module(args.model)
    clones = getattr(mod, "CLONES_VOICE", True)
    if todo:
        if hasattr(mod, "load"):
            mod.load(args.device) if inspect.signature(mod.load).parameters else mod.load()
        start = time.time()
        for i, r in enumerate(todo, 1):
            seed = args.seed + zlib.crc32(r["sentence_id"].encode()) % 100_000
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass
            np.random.seed(seed)
            ref = (r["prompt_wav"], r["prompt_text"]) if clones else (None, None)
            wav, sr = mod.synthesize(r["text"], *ref)
            wav = np.asarray(wav, dtype="float32")
            if wav.ndim == 2:
                wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
            sf.write(r["tts_wav"] + ".part.wav", wav, int(sr))
            os.replace(r["tts_wav"] + ".part.wav", r["tts_wav"])
            elapsed = time.time() - start
            print(f"[{i}/{len(todo)}] {r['sentence_id']}  {len(wav) / sr:.1f} s audio, {elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": os.path.abspath(args.model), "generator": "synthesize.py", "clones_voice": clones,
                   "seed": args.seed, "source_manifest": args.manifest, "n_sentences": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
