"""SpeechJudge-BTRM pointwise naturalness rewards for TTS and human clips.

BTRM (RMSnow/SpeechJudge-BTRM) is a Bradley-Terry reward model: Qwen2.5-Omni-7B thinker +
LoRA + a linear head that maps the last hidden state to one unbounded scalar per clip.
Each clip is scored on its own (no A/B position, no sampling), so the result is
deterministic. Rewards are only comparable between clips of the same text; under the
Bradley-Terry model P(TTS more natural than human) = sigmoid(r_tts - r_human).

Caveat (SpeechJudge paper, Table 11): on real human recordings vs TTS clones BTRM is
near chance (50.4%); on TTS vs TTS it reaches 72.7%.

Also reports the ratio score: per sentence (r_tts / r_human) x 100, averaged over the
sentences. Rewards are unbounded and can be <= 0, where the ratio stops meaning "percent of
human": sentences with r_human <= 0 are left out of the average (ratio undefined) and those
with r_tts <= 0 are counted as they come but flagged; the summary gives both counts.

Writes <out-dir>/btrm.jsonl and <out-dir>/btrm_summary.json.
Runs in the judge environment (.venv-judge). Resumable: sentences already in btrm.jsonl are skipped.

  python benchmark/btrm_score.py --manifest runs/ears_smoke/manifest.csv --out-dir runs/ears_smoke --limit 5
"""
import argparse
import json
import math
import os
import sys
from statistics import mean

from common import read_manifest, resolve_attn


def load_btrm(speechjudge_dir, base_path, rm_path, attn):
    # SpeechJudge's infer/ modules import each other as top-level modules
    sys.path.insert(0, os.path.join(speechjudge_dir, "infer"))
    import torch
    from btrm_pipeline import RewardModel
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from utils import build_qwen_omni_inputs, build_rm_conversation

    # Same as SpeechJudge's RewardModelInferencePipeline, but with a configurable attention backend
    snapshot_download("Qwen/Qwen2.5-Omni-7B", local_dir=base_path, max_workers=16)
    snapshot_download("RMSnow/SpeechJudge-BTRM", local_dir=rm_path)
    processor = Qwen2_5OmniProcessor.from_pretrained(base_path)
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation=resolve_attn(attn)
    )
    model = RewardModel(PeftModel.from_pretrained(base, os.path.join(rm_path, "lora")))
    model.to_reward.load_state_dict(torch.load(os.path.join(rm_path, "reward_head.pt"), map_location="cuda"))
    model.to("cuda").eval()

    @torch.no_grad()
    def score(text, wav):
        inputs = build_qwen_omni_inputs(processor, build_rm_conversation(wav, text))
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        return float(model(inputs)[0][0])

    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="run manifest with tts_wav (from generate_tts.py)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--speechjudge-dir", default="third_party/SpeechJudge")
    p.add_argument("--base-path", default="pretrained/Qwen2.5-Omni-7B")
    p.add_argument("--rm-path", default="pretrained/SpeechJudge-BTRM")
    p.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    out = os.path.join(args.out_dir, "btrm.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    done = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["sentence_id"]] = r
    todo = [r for r in rows if r["sentence_id"] not in done]
    print(f"{len(rows) - len(todo)} sentences already scored, {len(todo)} to score", flush=True)

    if todo:
        score = load_btrm(args.speechjudge_dir, args.base_path, args.rm_path, args.attn)
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            tts, human = score(r["text"], r["tts_wav"]), score(r["text"], r["human_wav"])
            res = {
                "speaker_id": r["speaker_id"],
                "sentence_id": r["sentence_id"],
                "tts_reward": tts,
                "human_reward": human,
                "p_tts_better": 1 / (1 + math.exp(human - tts)),
                "ratio": 100 * tts / human if human > 0 else None,
            }
            f.write(json.dumps(res) + "\n")
            f.flush()
            done[r["sentence_id"]] = res
            ratio = "n/a" if res["ratio"] is None else f"{res['ratio']:.1f}%"
            print(f"[{i}/{len(todo)}] {r['sentence_id']}: tts {tts:+.3f}  human {human:+.3f}  "
                  f"P(tts better) {res['p_tts_better']:.2f}  ratio {ratio}", flush=True)

    results = [done[r["sentence_id"]] for r in rows]
    for r in results:   # btrm.jsonl files written before the ratio score existed
        r.setdefault("ratio", 100 * r["tts_reward"] / r["human_reward"] if r["human_reward"] > 0 else None)
    ratios = [r["ratio"] for r in results if r["ratio"] is not None]
    speakers = sorted({r["speaker_id"] for r in results})
    mean_p = mean(r["p_tts_better"] for r in results)
    summary = {
        "n_sentences": len(results),
        "mean_tts_reward": mean(r["tts_reward"] for r in results),
        "mean_human_reward": mean(r["human_reward"] for r in results),
        "mean_p_tts_better": mean_p,
        # 100 = as natural as the human recording (P = 0.5); above 100 = preferred over it
        "human_parity_score": 200 * mean_p,
        "tts_wins": sum(r["tts_reward"] > r["human_reward"] for r in results),
        # mean over sentences of (r_tts / r_human) x 100; 100 = same reward as the human
        "ratio_score": mean(ratios) if ratios else None,
        "ratio_n_sentences": len(ratios),
        "ratio_skipped_human_reward_le_0": len(results) - len(ratios),
        "ratio_n_tts_reward_le_0": sum(r["tts_reward"] <= 0 for r in results if r["ratio"] is not None),
        "per_speaker": {
            spk: {
                "ratio_score": mean(v) if (v := [r["ratio"] for r in results
                                                 if r["speaker_id"] == spk and r["ratio"] is not None]) else None,
                "human_parity_score": 200 * mean(r["p_tts_better"] for r in results if r["speaker_id"] == spk),
                "n_sentences": sum(r["speaker_id"] == spk for r in results),
            }
            for spk in speakers
        },
        "model": "RMSnow/SpeechJudge-BTRM",
    }
    with open(os.path.join(args.out_dir, "btrm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
