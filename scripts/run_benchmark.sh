#!/usr/bin/env bash
# Full human-closeness benchmark for one TTS checkpoint:
#   1. select the locked LibriTTS-R set (only if the manifest doesn't exist yet)
#   2. generate cloned-voice TTS          (.venv-tts)
#   3. SpeechJudge TTS-vs-human scoring   (.venv-judge)
#   4. WER / speaker similarity / UTMOS   (.venv-tts)
# Every stage is resumable: re-run the same command after a crash.
#
#   bash scripts/run_benchmark.sh                                   # base Qwen3-TTS
#   RUN_NAME=trained TTS_MODEL=/path/to/checkpoint bash scripts/run_benchmark.sh
#   RUN_NAME=smoke LIMIT=5 bash scripts/run_benchmark.sh            # quick end-to-end test
#
# Env vars: RUN_NAME, TTS_MODEL, MANIFEST, VOTES, LIMIT, SKIP_METRICS=1
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME=${RUN_NAME:-base}
TTS_MODEL=${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}
MANIFEST=${MANIFEST:-benchmark/manifests/libritts_r_test_clean_10x50.csv}
VOTES=${VOTES:-1}
LIMIT=${LIMIT:-0}
RUN_DIR=runs/$RUN_NAME

TTS_PY=.venv-tts/bin/python
JUDGE_PY=.venv-judge/bin/python

if [ ! -f "$MANIFEST" ]; then
  echo "== Stage 1: selecting benchmark sentences"
  $TTS_PY benchmark/prepare_data.py --corpus libritts-r --speakers 10 --sentences-per-speaker 50 --out "$MANIFEST"
fi

echo "== Stage 2: generating TTS ($TTS_MODEL -> $RUN_DIR)"
$TTS_PY benchmark/generate_tts.py --manifest "$MANIFEST" --model "$TTS_MODEL" --run-dir "$RUN_DIR" --limit "$LIMIT"

echo "== Stage 3: SpeechJudge human-closeness"
$JUDGE_PY benchmark/judge.py --manifest "$RUN_DIR/manifest.csv" --out-dir "$RUN_DIR" --votes "$VOTES" --limit "$LIMIT"

if [ "${SKIP_METRICS:-0}" != 1 ]; then
  echo "== Stage 4: WER / speaker similarity / UTMOS"
  $TTS_PY benchmark/metrics.py --manifest "$RUN_DIR/manifest.csv" --out-dir "$RUN_DIR" --limit "$LIMIT"
fi

echo "Done. Results: $RUN_DIR/summary.json and $RUN_DIR/metrics_summary.json"
