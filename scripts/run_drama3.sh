#!/usr/bin/env bash
# Fish Audio Drama 3 (int4) voice-clone TTS, then SpeechJudge-BTRM scoring, on one GPU:
#   serve Drama 3 -> generate the TTS clips -> stop the server (frees the GPU) -> BTRM
#
#   RUN_NAME=drama3_smoke LIMIT=5 bash scripts/run_drama3.sh
#   bash scripts/gpu_run.sh "RUN_NAME=drama3_smoke LIMIT=5 bash scripts/run_drama3.sh"
#
# Drama 3 needs its own runtime (SGLang-Omni, Python 3.12, torch 2.14), so it gets a venv
# on the local NVMe next to its weights (fishaudio/drama-3-int4, ~22 GB, 20.8 GB on GPU).
# The full FP8 checkpoint needs >= 48 GB, too tight for one L40S with the vocoder.
set -euo pipefail
cd "$(dirname "$0")/.."
# pip torch bundles its own CUDA/cuDNN; a system CUDA on LD_LIBRARY_PATH (e.g. the AWS
# Deep Learning AMI) shadows it and breaks cuDNN with SUBLIBRARY_LOADING_FAILED
unset LD_LIBRARY_PATH

RUN_NAME=${RUN_NAME:-drama3}
CORPUS=${CORPUS:-ears}
LIMIT=${LIMIT:-0}
DRAMA3_MODEL=${DRAMA3_MODEL:-fishaudio/drama-3-int4}
SGLANG_OMNI=${SGLANG_OMNI:-0.1.9}
PORT=8000
TTS_PY=.venv-tts/bin/python
source scripts/corpus.sh   # makes sure the human audio is on this machine

# local NVMe on the AWS worker (fast, wiped on stop); the repo dir elsewhere
DIR=$PWD/pretrained/drama3
[ -d /opt/dlami/nvme ] && { sudo mkdir -p /opt/dlami/nvme/drama3 && sudo chown "$(id -u):$(id -g)" /opt/dlami/nvme/drama3; DIR=/opt/dlami/nvme/drama3; }
mkdir -p "$DIR" "runs/$RUN_NAME"
export UV_CACHE_DIR=$DIR/uv-cache UV_PYTHON_INSTALL_DIR=$DIR/python
CKPT=$DIR/$(basename "$DRAMA3_MODEL")
VENV=$DIR/venv

echo "== Drama 3: downloading $DRAMA3_MODEL and installing sglang-omni $SGLANG_OMNI (in parallel)"
start=$(date +%s)
.venv-judge/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$DRAMA3_MODEL', local_dir='$CKPT', max_workers=16,
                  allow_patterns=['*.safetensors', '*.json', '*.jinja', 'codec.pth', 'LICENSE'])
" &
download=$!
if [ ! -x "$VENV/bin/sgl-omni" ]; then
  command -v uv >/dev/null || [ -x "$DIR/bin/uv" ] ||
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DIR/bin" UV_NO_MODIFY_PATH=1 sh
  UV=$(command -v uv || echo "$DIR/bin/uv")
  "$UV" venv -q --python 3.12 --python-preference only-managed "$VENV"
  # descript-audiotools pins protobuf<3.20, sglang needs >=4.21; sglang-omni's pyproject
  # overrides it (tool.uv.override-dependencies), but that only applies to a source install
  echo "protobuf>=6.31.1,<7.0.0" > "$DIR/overrides.txt"
  "$UV" pip install -q --python "$VENV/bin/python" --prerelease=allow --overrides "$DIR/overrides.txt" \
    "sglang-omni[fishaudio]==$SGLANG_OMNI"
fi
wait $download
echo "Ready in $(( $(date +%s) - start )) s."

echo "== Drama 3: starting the server (log: runs/$RUN_NAME/drama3_server.log)"
export PATH=$VENV/bin:$PATH
# The defaults are sized for 80-141 GB GPUs: on a 48 GB L40S the vocoder's 10% share fits
# only ~1240 frames per decode, fewer than max_new_tokens=2048. 1024 frames is ~47 s of
# audio (the longest EARS sentence is 18 s) and fits with a 12% vocoder share.
# On a fresh worker Triton compiles the L40S (sm_89) kernels during CUDA-graph capture
# (~10-12 min, single core); the vocoder waits on the GPU startup lock meanwhile and the
# default 600 s no-progress limit can kill it. The wait loop below still caps at 30 min.
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
setsid sgl-omni serve --model-path "$CKPT" --port $PORT \
  --tts_engine.factory.max_new_tokens 1024 --vocoder.gpu_memory_fraction 0.12 \
  > "runs/$RUN_NAME/drama3_server.log" 2>&1 &
server=$!
stop_server() { kill -- -"$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }
trap stop_server EXIT
start=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  if ! kill -0 "$server" 2>/dev/null; then
    echo "Drama 3 server exited during startup:" >&2; tail -n 40 "runs/$RUN_NAME/drama3_server.log" >&2; exit 1
  fi
  [ $(( $(date +%s) - start )) -lt 1800 ] || { echo "Drama 3 server not up after 30 min" >&2; exit 1; }
  sleep 5
done
echo "Server up in $(( $(date +%s) - start )) s."

echo "== Drama 3: generating TTS"
$TTS_PY benchmark/generate_drama3.py --manifest "$MANIFEST" --run-dir "runs/$RUN_NAME" \
  --server "http://127.0.0.1:$PORT" --model "$DRAMA3_MODEL" --limit "$LIMIT"

echo "== Drama 3: stopping the server to free the GPU"
stop_server
trap - EXIT
for _ in $(seq 60); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 2
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "== SpeechJudge-BTRM"
RUN_NAME=$RUN_NAME CORPUS=$CORPUS LIMIT=$LIMIT MANIFEST=$MANIFEST bash scripts/run_btrm.sh
