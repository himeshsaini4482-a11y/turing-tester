"""Patch the pinned Qwen3-TTS checkout's finetuning/sft_12hz.py (idempotent).

It loads the model with attn_implementation="flash_attention_2", which needs the flash-attn package
(no Windows/macOS wheels, slow to build). PyTorch SDPA gives the same results, so use it.

  python sft/patch_qwen3_ft.py --repo-dir third_party/Qwen3-TTS
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    args = p.parse_args()
    path = Path(args.repo_dir) / "finetuning" / "sft_12hz.py"
    s = path.read_text(encoding="utf-8")
    new = s.replace('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
    if new != s:
        path.write_text(new, encoding="utf-8")
        print(f"patched {path}: flash_attention_2 -> sdpa")
    else:
        print(f"{path}: already patched")


if __name__ == "__main__":
    main()
