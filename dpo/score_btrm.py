"""DPO round stage 2: SpeechJudge-BTRM reward for every rollout candidate (Alg. 1 line 4).

The reward model and its loading are benchmark/btrm_score.py's load_btrm, unchanged, so a DPO
reward is the same number the SFT pipeline and the benchmark report. Unlike btrm_score.py, only
the TTS clip is scored: a candidate's text was written by the 9B, so there is no human recording
of it, and pairs are compared within one prompt (same text, same voice), where BTRM's rewards are
comparable and where it is strongest (TTS vs TTS 72.7% in the SpeechJudge paper, Table 11).

With --round, candidates that already failed a gate (pairs.py) are not scored: they cannot be in a
pair and the sigma_r^2 estimate uses valid candidates only.

Writes <out-dir>/btrm.jsonl: {"sentence_id": cid, "speaker_id", "tts_reward"}.
Env: JUDGE (.venv-judge). Resumable.

  python dpo/score_btrm.py --manifest runs/dpo/r01/manifest.csv --out-dir runs/dpo/r01 \
      --base-path $BTRM_DIR/Qwen2.5-Omni-7B --rm-path $BTRM_DIR/SpeechJudge-BTRM
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmark"))
from btrm_score import load_btrm  # noqa: E402
from common import read_manifest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--speechjudge-dir", default="third_party/SpeechJudge")
    p.add_argument("--base-path", default="pretrained/Qwen2.5-Omni-7B")
    p.add_argument("--rm-path", default="pretrained/SpeechJudge-BTRM")
    p.add_argument("--attn", default="auto")
    p.add_argument("--round", type=int, default=0,
                   help="score only candidates that can still form a pair (pairs.scorable: every gate but the "
                        "reward passes, >= 2 per prompt); needs that round's metrics.jsonl")
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    if args.round:
        sys.path.insert(0, os.path.join(REPO, "dpo"))
        import dpolib
        import pairs
        rdir = dpolib.round_dir(args.round)
        metrics = {r["sentence_id"]: r for r in dpolib.read_jsonl(rdir / "metrics.jsonl")}
        keep = pairs.scorable(dpolib.read_jsonl(rdir / "candidates.jsonl"), metrics,
                              dpolib.reward_cfg(dpolib.load_hparams()))
        print(f"{len(keep)}/{len(rows)} candidates can still form a pair; scoring only those", flush=True)
        rows = [r for r in rows if r["sentence_id"] in keep]
    out = os.path.join(args.out_dir, "btrm.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["sentence_id"] for line in f if line.strip()}
    todo = [r for r in rows if r["sentence_id"] not in done]
    print(f"{len(done)} candidates already scored, {len(todo)} to score", flush=True)
    if not todo:
        return
    score = load_btrm(args.speechjudge_dir, args.base_path, args.rm_path, args.attn)
    start = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            f.write(json.dumps({"sentence_id": r["sentence_id"], "speaker_id": r["speaker_id"],
                                "tts_reward": score(r["text"], r["tts_wav"])}) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - start
                print(f"[{i}/{len(todo)}] {el / i:.2f} s/clip, {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(todo) - i) / 60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
