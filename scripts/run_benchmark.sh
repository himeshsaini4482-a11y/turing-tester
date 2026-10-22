#!/usr/bin/env bash
# Full human-closeness benchmark for one TTS checkpoint:
#   1. select the locked set (EARS by default; also fetches its audio on a fresh worker)
#   2. generate cloned-voice TTS          (.venv-tts)
#   3. SpeechJudge TTS-vs-human scoring   (.venv-judge)
#   4. WER / speaker similarity / UTMOS   (.venv-tts)
# Every stage is resumable: re-run the same command after a crash.
#
#   bash scripts/run_benchmark.sh                                   # base Qwen3-TTS
#   RUN_NAME=trained TTS_MODEL=/path/to/checkpoint bash scripts/run_benchmark.sh
#   RUN_NAME=smoke LIMIT=5 bash scripts/run_benchmark.sh            # quick end-to-end test
#
# Env vars: RUN_NAME, CORPUS (ears | libritts-r), TTS_MODEL, MANIFEST, VOTES, LIMIT, SKIP_METRICS=1
set -euo pipefail
cd "$(dirname "$0")/.."
# pip torch bundles its own CUDA/cuDNN; a system CUDA on LD_LIBRARY_PATH (e.g. the AWS
# Deep Learning AMI) shadows it and breaks cuDNN with SUBLIBRARY_LOADING_FAILED
unset LD_LIBRARY_PATH

RUN_NAME=${RUN_NAME:-base}
TTS_MODEL=${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}
CORPUS=${CORPUS:-ears}
VOTES=${VOTES:-1}
LIMIT=${LIMIT:-0}
RUN_DIR=runs/$RUN_NAME

TTS_PY=.venv-tts/bin/python
JUDGE_PY=.venv-judge/bin/python

source scripts/corpus.sh

echo "== Stage 2: generating TTS ($TTS_MODEL -> $RUN_DIR)"
$TTS_PY benchmark/generate_tts.py --manifest "$MANIFEST" --model "$TTS_MODEL" --run-dir "$RUN_DIR" --limit "$LIMIT"

echo "== Stage 3: SpeechJudge human-closeness"
JUDGE_MODEL=$(bash scripts/stage_judge.sh)
$JUDGE_PY benchmark/judge.py --manifest "$RUN_DIR/manifest.csv" --out-dir "$RUN_DIR" --votes "$VOTES" --limit "$LIMIT" \
  --model-path "$JUDGE_MODEL"

if [ "${SKIP_METRICS:-0}" != 1 ]; then
  echo "== Stage 4: WER / speaker similarity / UTMOS"
  $TTS_PY benchmark/metrics.py --manifest "$RUN_DIR/manifest.csv" --out-dir "$RUN_DIR" --limit "$LIMIT"
fi

echo "Done. Results: $RUN_DIR/summary.json and $RUN_DIR/metrics_summary.json"
