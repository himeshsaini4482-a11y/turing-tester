"""Patch the pinned MOSS-TTS-Nano checkout's finetuning code (idempotent).

1. finetuning/dataset.py builds the training prompt differently from inference
   (modeling_moss_tts_nano.build_inference_input_ids + prompting.py on the HF model):
   - it omits <im_end> between "</user_inst>" and the assistant turn;
   - it encodes "user\\n" + "<user_inst>..." and the field block + text as one string each,
     where inference encodes them separately (sentencepiece can merge across the seam).
   The patch makes training tokens identical to inference. Gate G0a (sft/check_prompt.py)
   verifies the result on real records.
2. finetuning/sft.py loads the model in the training dtype, so --mixed-precision bf16 means
   pure-bf16 weights and AdamW updates; at lr 1e-5 most updates are smaller than bf16's
   resolution and are rounded away. The patch adds --fp32-master-weights: fp32 weights with
   bf16 autocast (Accelerate's mixed precision), the standard recipe.

  python sft/patch_moss.py --repo-dir /opt/dlami/nvme/moss_nano/MOSS-TTS-Nano
"""
import argparse
from pathlib import Path

MARK = "[voicegym patch]"

DATASET_EDITS = [
    (
        """        prefix_ids = [self.model_config.im_start_token_id] + encode_text(
            self.tokenizer,
            USER_ROLE_PREFIX + USER_TEMPLATE_REFERENCE_PREFIX,
        )
        suffix_text = self._build_suffix_text(record)
        assistant_prefix_ids = encode_text(self.tokenizer, USER_TEMPLATE_SUFFIX + ASSISTANT_TURN_PREFIX) + [
            self.model_config.im_start_token_id
        ] + encode_text(self.tokenizer, ASSISTANT_ROLE_PREFIX)
""",
        f"""        # {MARK} token-for-token the same as prompting.build_user_prompt_prefix() and
        # build_assistant_prompt_prefix() used at inference (separate encodes, <im_end>).
        prefix_ids = (
            [self.model_config.im_start_token_id]
            + encode_text(self.tokenizer, USER_ROLE_PREFIX)
            + encode_text(self.tokenizer, USER_TEMPLATE_REFERENCE_PREFIX)
        )
        suffix_text = self._build_suffix_text(record)
        assistant_prefix_ids = (
            encode_text(self.tokenizer, USER_TEMPLATE_SUFFIX)
            + [self.model_config.im_end_token_id]
            + encode_text(self.tokenizer, ASSISTANT_TURN_PREFIX)
            + [self.model_config.im_start_token_id]
            + encode_text(self.tokenizer, ASSISTANT_ROLE_PREFIX)
        )
""",
    ),
    (
        """            sections.append(
                self._build_text_rows([self.model_config.audio_end_token_id] + encode_text(self.tokenizer, suffix_text))
            )
""",
        f"""            # {MARK} inference encodes the field block and the target text separately
            fields_text = suffix_text[: len(suffix_text) - len(str(record["text"]))]
            sections.append(
                self._build_text_rows(
                    [self.model_config.audio_end_token_id]
                    + encode_text(self.tokenizer, fields_text)
                    + encode_text(self.tokenizer, str(record["text"]))
                )
            )
""",
    ),
]

SFT_EDITS = [
    (
        """    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
""",
        f"""    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    # {MARK} fp32 weights + bf16 autocast instead of pure-bf16 weights
    parser.add_argument("--fp32-master-weights", action="store_true")
""",
    ),
    (
        """    model_dtype = resolve_torch_dtype(args.mixed_precision)
""",
        f"""    model_dtype = torch.float32 if args.fp32_master_weights else resolve_torch_dtype(args.mixed_precision)  # {MARK}
""",
    ),
]


def apply(path, edits):
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        print(f"already patched: {path}")
        return
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: expected exactly one match for:\n{old}\n(upstream code changed? "
                             "check the pinned commit in scripts/sft/env.sh)")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    args = p.parse_args()
    root = Path(args.repo_dir)
    apply(root / "finetuning" / "dataset.py", DATASET_EDITS)
    apply(root / "finetuning" / "sft.py", SFT_EDITS)


if __name__ == "__main__":
    main()
