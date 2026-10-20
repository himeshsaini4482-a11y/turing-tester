#!/usr/bin/env bash
# Judge each human clip against itself. Closeness should come out near 100%;
# a large gap means SpeechJudge favours one A/B position or is too noisy to trust.
#
#   bash scripts/run_sanity_check.sh            # first 50 sentences
#   LIMIT=100 bash scripts/run_sanity_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=${MANIFEST:-benchmark/manifests/libritts_r_test_clean_10x50.csv}
LIMIT=${LIMIT:-50}

if [ ! -f "$MANIFEST" ]; then
  .venv-tts/bin/python benchmark/prepare_data.py --corpus libritts-r --speakers 10 --sentences-per-speaker 50 --out "$MANIFEST"
fi

.venv-judge/bin/python benchmark/judge.py --manifest "$MANIFEST" --out-dir runs/sanity_human_vs_human \
  --self-check --limit "$LIMIT"
