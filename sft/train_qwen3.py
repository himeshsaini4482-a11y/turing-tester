"""Qwen3-TTS (12Hz Base) fine-tuning on one voice, with Qwen's official finetuning code.

Not part of our experiments. Qwen's script trains a single-speaker "custom_voice" model: afterwards
the model speaks that voice by name (generate_tts.py --speaker), it no longer clones others. So this
path exists for `voicegym sft --model <Qwen3-TTS Base> --voice voices/<name>` only.

Steps (third_party/Qwen3-TTS/finetuning, pinned commit, flash-attn swapped for SDPA):
  prepare_data.py  audio codes for every training clip (Qwen/Qwen3-TTS-Tokenizer-12Hz)
  sft_12hz.py      the script's own defaults: batch 2 (x 4 accumulation), lr 2e-5, 3 epochs
                   (Qwen's README suggests the same reference clip for every record, as here)
Output: <out>/checkpoint-last (the last epoch's checkpoint).

Env: qwen3-train.

  python sft/train_qwen3.py --model models/qwen3-tts --records data/voices/me/train.jsonl \
      --ref-audio data/voices/me/clips/x.wav --speaker me --out runs/sft/me/ckpt
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FT = REPO / "third_party" / "Qwen3-TTS" / "finetuning"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--ref-audio", required=True, help="one reference clip for every record")
    p.add_argument("--speaker", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint-last" / "config.json").exists():
        print(f"{out}/checkpoint-last exists, skipping", flush=True)
        return
    init = Path(args.model)
    if not init.is_dir():   # sft_12hz.py copies the init folder, so it must be local
        from huggingface_hub import snapshot_download
        init = Path(snapshot_download(args.model, local_dir=str(REPO / "pretrained" / args.model.split("/")[-1])))
    raw = out / "train_raw.jsonl"
    with open(args.records, encoding="utf-8") as f, open(raw, "w", encoding="utf-8") as o:
        for line in f:
            if line.strip():
                r = json.loads(line)
                o.write(json.dumps({"audio": str(Path(r["audio"]).resolve()), "text": r["text"],
                                    "ref_audio": str(Path(args.ref_audio).resolve())}, ensure_ascii=False) + "\n")
    codes = out / "train_with_codes.jsonl"
    env = dict(os.environ)
    if not codes.exists():
        subprocess.run([sys.executable, "prepare_data.py", "--device", args.device, "--tokenizer_model_path",
                        "Qwen/Qwen3-TTS-Tokenizer-12Hz", "--input_jsonl", str(raw), "--output_jsonl",
                        str(codes) + ".part"], cwd=FT, check=True, env=env)
        os.replace(str(codes) + ".part", codes)
    work = out / "train"
    subprocess.run([sys.executable, "sft_12hz.py", "--init_model_path", str(init.resolve()), "--output_model_path",
                    str(work), "--train_jsonl", str(codes), "--batch_size", str(args.batch), "--lr", str(args.lr),
                    "--num_epochs", str(args.epochs), "--speaker_name", args.speaker], cwd=FT, check=True, env=env)
    last = work / f"checkpoint-epoch-{args.epochs - 1}"
    if not last.is_dir():
        sys.exit(f"no checkpoint at {last}")
    shutil.rmtree(out / "checkpoint-last", ignore_errors=True)
    last.rename(out / "checkpoint-last")
    shutil.rmtree(work, ignore_errors=True)
    print(f"done: {out / 'checkpoint-last'} (speaker {args.speaker!r})", flush=True)


if __name__ == "__main__":
    main()
