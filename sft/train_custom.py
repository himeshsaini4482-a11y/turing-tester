"""SFT for a synthesize.py model: call the model folder's own train.py (docs/adding_a_model.md).

train.py must define
  train(records_jsonl, output_dir, device)
where records_jsonl has one {"audio", "text", "ref_audio", "speaker", "duration"} per line, and
output_dir must end up as a complete model folder (with synthesize.py) that VoiceGym can benchmark.

  python sft/train_custom.py --model models/my-tts --records train.jsonl --out runs/sft/x/ckpt
"""
import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    out = Path(args.out) / "checkpoint-last"
    if (out / "synthesize.py").exists():
        print(f"{out} exists, skipping", flush=True)
        return
    folder = os.path.abspath(args.model)
    sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location("vg_train", os.path.join(folder, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "train"):
        sys.exit(f"{folder}/train.py must define train(records_jsonl, output_dir, device)")
    tmp = Path(args.out) / "checkpoint-last.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    kwargs = {"device": args.device} if "device" in inspect.signature(mod.train).parameters else {}
    mod.train(os.path.abspath(args.records), str(tmp.resolve()), **kwargs)
    if not (tmp / "synthesize.py").exists():
        sys.exit(f"train() finished but {tmp} has no synthesize.py: the output must be a complete model folder")
    tmp.rename(out)
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
