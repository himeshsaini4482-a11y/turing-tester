#!/usr/bin/env bash
# SpeechJudge-BTRM pointwise rewards (TTS clip and human clip each scored on its own)
# for an existing run, i.e. after run_benchmark.sh stage 2 has generated the TTS audio.
#
#   RUN_NAME=ears_smoke LIMIT=5 bash scripts/run_btrm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# pip torch bundles its own CUDA/cuDNN; a system CUDA on LD_LIBRARY_PATH (e.g. the AWS
# Deep Learning AMI) shadows it and breaks cuDNN with SUBLIBRARY_LOADING_FAILED
unset LD_LIBRARY_PATH

RUN_NAME=${RUN_NAME:-base}
CORPUS=${CORPUS:-ears}
LIMIT=${LIMIT:-0}
TTS_PY=${TTS_PY:-.venv-tts/bin/python}
source scripts/corpus.sh   # makes sure the human audio is on this machine

# local NVMe on the AWS worker loads far faster than the EBS disk (see stage_judge.sh)
DIR=pretrained
[ -d /opt/dlami/nvme ] && { sudo mkdir -p /opt/dlami/nvme/btrm && sudo chown "$(id -u):$(id -g)" /opt/dlami/nvme/btrm; DIR=/opt/dlami/nvme/btrm; }

"${JUDGE_PY:-.venv-judge/bin/python}" benchmark/btrm_score.py --manifest "runs/$RUN_NAME/manifest.csv" --out-dir "runs/$RUN_NAME" \
  --limit "$LIMIT" --base-path "$DIR/Qwen2.5-Omni-7B" --rm-path "$DIR/SpeechJudge-BTRM"
