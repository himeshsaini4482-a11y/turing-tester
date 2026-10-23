# Sourced by every scripts/sft/*.sh: paths, venvs and the resumable `step` helper.
#
#   source "$(dirname "$0")/env.sh"
#
# Durable outputs go to runs/sft/ (worker EBS, copied back by gpu_run.sh). Bulky re-creatable
# files go to $SCR on the local NVMe (wiped when the worker stops).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# pip torch bundles its own CUDA/cuDNN; a system CUDA on LD_LIBRARY_PATH (e.g. the AWS
# Deep Learning AMI) shadows it and breaks cuDNN with SUBLIBRARY_LOADING_FAILED
unset LD_LIBRARY_PATH

REPO_DIR=$PWD
RUNS=$REPO_DIR/runs/sft
if [ -d /opt/dlami/nvme ]; then NV=/opt/dlami/nvme; else NV=$REPO_DIR/data/nvme; fi

# mkdir that also works on the worker's root-owned NVMe mount (as run_moss_nano.sh does)
nv_dir() {
  mkdir -p "$1" 2>/dev/null || { sudo mkdir -p "$1" && sudo chown "$(id -u):$(id -g)" "$1"; }
}

export SFT_SCRATCH=$NV/sft
SCR=$SFT_SCRATCH
nv_dir "$SCR"
if [ -d /opt/dlami/nvme ]; then   # model downloads on the fast local NVMe
  nv_dir "$NV/hf"
  export HF_HUB_CACHE=$NV/hf
fi

TTS_PY=$REPO_DIR/.venv-tts/bin/python
JUDGE_PY=$REPO_DIR/.venv-judge/bin/python

# MOSS-TTS-Nano: own venv + pinned repo checkout (same location as scripts/run_moss_nano.sh)
MOSS_DIR=$NV/moss_nano
nv_dir "$MOSS_DIR"
MOSS_REPO=$MOSS_DIR/MOSS-TTS-Nano
MOSS_COMMIT=8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb
MOSS_PY=$MOSS_DIR/venv/bin/python
MOSS_ACCEL=$MOSS_DIR/venv/bin/accelerate
MOSS_MODEL_DIR=$MOSS_DIR/models/MOSS-TTS-Nano-100M
MOSS_CODEC_DIR=$MOSS_DIR/models/MOSS-Audio-Tokenizer-Nano
MOSS_HF_MODEL=OpenMOSS-Team/MOSS-TTS-Nano-100M
MOSS_HF_CODEC=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano
# Python stages (sft/gates.py, make_mix.py, ...) read these from the environment
export MOSS_DIR MOSS_REPO MOSS_PY MOSS_ACCEL MOSS_MODEL_DIR MOSS_CODEC_DIR

# SpeechJudge-BTRM weights (same layout as scripts/run_btrm.sh)
BTRM_DIR=$REPO_DIR/pretrained
[ -d /opt/dlami/nvme ] && BTRM_DIR=$NV/btrm
nv_dir "$BTRM_DIR"

mkdir -p "$RUNS"

# step NAME CMD...: run CMD once. A marker in runs/sft/.done/ makes re-runs skip it, so every
# pipeline script can simply be launched again after a crash or a worker restart.
step() {
  local name=$1; shift
  local marker=$RUNS/.done/$name
  if [ -f "$marker" ]; then echo "== [skip] $name (done $(cat "$marker")s)"; return 0; fi
  echo "== [$(date +%H:%M:%S)] $name"
  local t0; t0=$(date +%s)
  "$@"
  local dt=$(( $(date +%s) - t0 ))
  mkdir -p "$RUNS/.done"
  echo "$dt" > "$marker"
  "$TTS_PY" -c "import sys; sys.path.insert(0, 'sft'); import sftlib; sftlib.log_timing('$name', $dt)"
  echo "== [done] $name in ${dt}s"
}

# Inference chunk limit chosen in the pilot (runs/sft/decoding.json); 50 = MOSS default.
dec_tokens() {
  "$TTS_PY" -c "import sys; sys.path.insert(0, 'sft'); import sftlib; \
print(sftlib.read_json(sftlib.RUNS / 'decoding.json', {}).get('voice_clone_max_text_tokens', 50))"
}

# Max text tokens per *training* chunk that matches that decoding (50 chunked, 75 whole).
train_max_tokens() {
  "$TTS_PY" -c "import sys; sys.path.insert(0, 'sft'); import sftlib; \
print(sftlib.read_json(sftlib.RUNS / 'decoding.json', {}).get('train_max_text_tokens', 50))"
}

judge_model_path() { bash scripts/stage_judge.sh; }
