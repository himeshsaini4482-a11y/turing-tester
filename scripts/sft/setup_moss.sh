#!/usr/bin/env bash
# MOSS-TTS-Nano training setup (idempotent):
#   - MOSS venv (same recipe as scripts/run_moss_nano.sh) + accelerate/tqdm for finetuning
#   - MOSS-TTS-Nano repo checked out at the pinned commit, finetuning code patched (sft/patch_moss.py)
#   - model + audio codec downloaded to local dirs (training reads them; checkpoints copy the code)
#
#   bash scripts/sft/setup_moss.sh
source "$(dirname "$0")/env.sh"
mkdir -p "$MOSS_DIR"
export UV_CACHE_DIR=$MOSS_DIR/uv-cache UV_PYTHON_INSTALL_DIR=$MOSS_DIR/python

if [ ! -d "$MOSS_REPO/.git" ]; then
  git clone -q https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$MOSS_REPO"
fi
if [ "$(git -C "$MOSS_REPO" rev-parse HEAD)" != "$MOSS_COMMIT" ]; then
  git -C "$MOSS_REPO" fetch -q --depth 1 origin "$MOSS_COMMIT" 2>/dev/null || git -C "$MOSS_REPO" fetch -q origin
  git -C "$MOSS_REPO" checkout -q "$MOSS_COMMIT"
fi

command -v uv >/dev/null || [ -x "$MOSS_DIR/bin/uv" ] ||
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$MOSS_DIR/bin" UV_NO_MODIFY_PATH=1 sh
UV=$(command -v uv || echo "$MOSS_DIR/bin/uv")
if [ ! -x "$MOSS_PY" ]; then
  "$UV" venv -q --python 3.12 --python-preference only-managed "$MOSS_DIR/venv"
fi
# the inference subset of MOSS's requirements.txt (as run_moss_nano.sh) + what finetuning/ imports
"$UV" pip install -q --python "$MOSS_PY" torch==2.7.0 torchaudio==2.7.0 transformers==4.57.1 \
  "numpy>=1.24" "sentencepiece>=0.1.99" soundfile "accelerate>=1.0.0" "tqdm>=4.66.0"

"$MOSS_PY" sft/patch_moss.py --repo-dir "$MOSS_REPO"

"$MOSS_PY" - <<EOF
from huggingface_hub import snapshot_download
for repo, path in (("$MOSS_HF_MODEL", "$MOSS_MODEL_DIR"), ("$MOSS_HF_CODEC", "$MOSS_CODEC_DIR")):
    snapshot_download(repo, local_dir=path)
    print("ready:", repo, "->", path)
EOF
"$MOSS_PY" -c "import torch, transformers, accelerate; print('MOSS venv: torch', torch.__version__, \
'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'cuda', torch.cuda.is_available())"
