#!/usr/bin/env bash
# MOSS-TTS-Nano (0.1B) voice-clone TTS, then the usual scoring:
#   generate -> SpeechJudge-GRM closeness -> WER / speaker sim / UTMOS -> SpeechJudge-BTRM
#
#   RUN_NAME=moss_nano_smoke LIMIT=5 bash scripts/run_moss_nano.sh
#   KEEP_UP=1 bash tools/aws/gpu_run.sh "RUN_NAME=moss_nano_smoke LIMIT=5 bash scripts/run_moss_nano.sh"
#   GEN_ONLY=1 ...   # stop after generating the TTS clips
#   SKIP_GRM=1 ...   # skip the (slow) SpeechJudge-GRM stage
#   MOSS_MAX_TEXT_TOKENS=100 ...  # MOSS voice-clone chunk limit (default 50; 100 = no split)
#   MOSS_MODEL=/path/to/checkpoint ...  # a fine-tuned checkpoint instead of the HF model
#
# MOSS-TTS-Nano pins torch 2.7.0 / transformers 4.57.1, so it gets its own venv (and a
# checkout of its repo, for the text normalization) on the local NVMe.
set -euo pipefail
cd "$(dirname "$0")/.."
# pip torch bundles its own CUDA/cuDNN; a system CUDA on LD_LIBRARY_PATH (e.g. the AWS
# Deep Learning AMI) shadows it and breaks cuDNN with SUBLIBRARY_LOADING_FAILED
unset LD_LIBRARY_PATH

RUN_NAME=${RUN_NAME:-moss_nano}
CORPUS=${CORPUS:-ears}
LIMIT=${LIMIT:-0}
VOTES=${VOTES:-1}
MOSS_MODEL=${MOSS_MODEL:-OpenMOSS-Team/MOSS-TTS-Nano-100M}
RUN_DIR=runs/$RUN_NAME
TTS_PY=${TTS_PY:-.venv-tts/bin/python}
JUDGE_PY=${JUDGE_PY:-.venv-judge/bin/python}
source scripts/corpus.sh   # makes sure the human audio is on this machine

# local NVMe on the AWS worker (fast, wiped on stop); the repo dir elsewhere
DIR=$PWD/pretrained/moss_nano
[ -d /opt/dlami/nvme ] && { sudo mkdir -p /opt/dlami/nvme/moss_nano && sudo chown "$(id -u):$(id -g)" /opt/dlami/nvme/moss_nano; DIR=/opt/dlami/nvme/moss_nano; }
mkdir -p "$DIR" "$RUN_DIR"
export UV_CACHE_DIR=$DIR/uv-cache UV_PYTHON_INSTALL_DIR=$DIR/python
REPO=${MOSS_REPO:-$DIR/MOSS-TTS-Nano}
VENV=$DIR/venv
MOSS_GEN_PY=${MOSS_PY:-$VENV/bin/python}   # the voicegym CLI passes its pinned, patched environment

if [ ! -x "$MOSS_GEN_PY" ]; then
  echo "== MOSS-TTS-Nano: installing"
  [ -d "$REPO" ] || git clone -q --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$REPO"
  command -v uv >/dev/null || [ -x "$DIR/bin/uv" ] ||
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DIR/bin" UV_NO_MODIFY_PATH=1 sh
  UV=$(command -v uv || echo "$DIR/bin/uv")
  "$UV" venv -q --python 3.12 --python-preference only-managed "$VENV"
  # the inference subset of its requirements.txt (no web app, ONNX or WeTextProcessing/pynini)
  "$UV" pip install -q --python "$VENV/bin/python" torch==2.7.0 torchaudio==2.7.0 transformers==4.57.1 \
    "numpy>=1.24" "sentencepiece>=0.1.99" soundfile
fi

echo "== Stage 2: generating TTS ($MOSS_MODEL -> $RUN_DIR)"
"$MOSS_GEN_PY" benchmark/generate_moss_nano.py --manifest "$MANIFEST" --run-dir "$RUN_DIR" \
  --repo-dir "$REPO" --model "$MOSS_MODEL" --limit "$LIMIT" \
  --voice-clone-max-text-tokens "${MOSS_MAX_TEXT_TOKENS:-50}"

[ "${GEN_ONLY:-0}" = 1 ] && exit 0

if [ "${SKIP_GRM:-0}" != 1 ]; then
  echo "== Stage 3: SpeechJudge-GRM human-closeness"
  JUDGE_MODEL=$(bash scripts/stage_judge.sh)
  $JUDGE_PY benchmark/judge.py --manifest "$RUN_DIR/manifest.csv" --out-dir "$RUN_DIR" --votes "$VOTES" --limit "$LIMIT" \
    --model-path "$JUDGE_MODEL"
fi

echo "== Stage 4: WER / speaker similarity / UTMOS"
$TTS_PY benchmark/metrics.py --manifest "$RUN_DIR/manifest.csv" --out-dir "$RUN_DIR" --limit "$LIMIT"

echo "== SpeechJudge-BTRM"
RUN_NAME=$RUN_NAME CORPUS=$CORPUS LIMIT=$LIMIT MANIFEST=$MANIFEST bash scripts/run_btrm.sh

echo "Done. Results: $RUN_DIR/{summary.json (unless SKIP_GRM),metrics_summary.json,btrm_summary.json}"
