"""Score TTS clips with SpeechJudge-BTRM while they are still being generated.

Appends to the same <run-dir>/btrm.jsonl as btrm_score.py (same record format), so the
btrm_score.py run that follows generation only scores what is left and writes the summary.
Each pass scores the clips not yet in btrm.jsonl, skipping files modified in the last
--settle seconds (possibly still being written). A run is finished once its manifest.csv
exists (written by the generator at the end). This script exits as soon as any run is, so
that run's btrm_score.py (started right after generation) is the only BTRM model on the GPU;
scripts/btrm_follow.sh restarts it for the remaining runs once that scoring is done.

  python benchmark/btrm_follow.py --manifest benchmark/manifests/ears_regular_26x19.csv \
      --runs runs/moss_nano_full runs/orpheus_full
"""
import argparse
import json
import math
import os
import time

from btrm_score import load_btrm
from common import read_manifest


def scored(run):
    path = os.path.join(run, "btrm.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["sentence_id"] for line in f if line.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="the source manifest the runs were generated from")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--speechjudge-dir", default="third_party/SpeechJudge")
    p.add_argument("--base-path", default="pretrained/Qwen2.5-Omni-7B")
    p.add_argument("--rm-path", default="pretrained/SpeechJudge-BTRM")
    p.add_argument("--attn", default="auto")
    p.add_argument("--settle", type=int, default=120, help="seconds a clip must be unmodified before scoring")
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    score = load_btrm(args.speechjudge_dir, args.base_path, args.rm_path, args.attn)
    n = 0
    def finished():
        return any(os.path.exists(os.path.join(run, "manifest.csv")) for run in args.runs)

    while not finished():
        did = 0
        for run in args.runs:
            done = scored(run)
            for r in rows:
                wav = os.path.join(run, "tts", f"{r['sentence_id']}.wav")
                if r["sentence_id"] in done or not os.path.exists(wav) or time.time() - os.path.getmtime(wav) < args.settle:
                    continue
                if finished():
                    break
                tts, human = score(r["text"], wav), score(r["text"], r["human_wav"])
                res = {
                    "speaker_id": r["speaker_id"],
                    "sentence_id": r["sentence_id"],
                    "tts_reward": tts,
                    "human_reward": human,
                    "p_tts_better": 1 / (1 + math.exp(human - tts)),
                    "ratio": 100 * tts / human if human > 0 else None,
                }
                with open(os.path.join(run, "btrm.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                did += 1
                n += 1
                if n % 25 == 0:
                    print(f"{n} clips scored ({run}: {len(done) + did}+ done)", flush=True)
        if not did:
            time.sleep(60)
    print(f"A run finished generating; {n} clips scored here, the rest go to btrm_score.py", flush=True)


if __name__ == "__main__":
    main()
