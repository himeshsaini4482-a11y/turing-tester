#!/usr/bin/env bash
# Judge each human clip against itself. Closeness should come out near 100%;
# a large gap means SpeechJudge favours one A/B position or is too noisy to trust.
#
#   bash scripts/run_sanity_check.sh            # first 50 sentences
#   LIMIT=100 bash scripts/run_sanity_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# pip torch bundles its own CUDA/cuDNN; a system CUDA on LD_LIBRARY_PATH (e.g. the AWS
# Deep Learning AMI) shadows it and breaks cuDNN with SUBLIBRARY_LOADING_FAILED
unset LD_LIBRARY_PATH

CORPUS=${CORPUS:-ears}
LIMIT=${LIMIT:-50}

TTS_PY=.venv-tts/bin/python
source scripts/corpus.sh

JUDGE_MODEL=$(bash scripts/stage_judge.sh)
.venv-judge/bin/python benchmark/judge.py --manifest "$MANIFEST" --out-dir "runs/sanity_${CORPUS}" \
  --self-check --limit "$LIMIT" --model-path "$JUDGE_MODEL"
