#!/usr/bin/env bash
# One-time VM setup: system packages, three Python venvs, pinned SpeechJudge code.
#
#   bash scripts/setup_vm.sh
#
# Env vars:
#   PYTHON=python3.12        interpreter used to create the venvs (default: python3; .venv-dpo needs >= 3.11)
#   WITH_TRL_VENV=1          also build .venv-dpo (trl; only docs/moss_tts_nano_online_dpo.md's design study
#                            uses it, the DPO pipeline in dpo/ does not)
#   INSTALL_FLASH_ATTN=1     also build flash-attn (slow; without it SDPA attention is used)
#   SKIP_APT=1               skip apt-get (no sudo, or packages already installed)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
SPEECHJUDGE_COMMIT=1fbd1c883b03f3f7437506b421f7cf57bad78071

if [ "${SKIP_APT:-0}" != 1 ] && command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y git sox libsox-dev ffmpeg python3-venv
fi

ENVS="tts judge"
[ "${WITH_TRL_VENV:-0}" = 1 ] && ENVS="tts judge dpo"
if [ "${WITH_TRL_VENV:-0}" = 1 ] && ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "[error] .venv-dpo needs Python >= 3.11 (trl 1.16.0); set PYTHON=python3.11 or newer" >&2
  exit 1
fi

for env in $ENVS; do
  if [ ! -d ".venv-$env" ]; then
    "$PYTHON" -m venv ".venv-$env"
  fi
  ".venv-$env/bin/pip" install -U pip wheel
  ".venv-$env/bin/pip" install -r "requirements-$env.txt"
done

if [ ! -d third_party/SpeechJudge ]; then
  git clone https://github.com/AmphionTeam/SpeechJudge.git third_party/SpeechJudge
fi
git -C third_party/SpeechJudge fetch --quiet origin
git -C third_party/SpeechJudge checkout --quiet "$SPEECHJUDGE_COMMIT"

if [ "${INSTALL_FLASH_ATTN:-0}" = 1 ]; then
  for env in $ENVS; do
    MAX_JOBS=4 ".venv-$env/bin/pip" install flash-attn --no-build-isolation \
      || echo "[warn] flash-attn failed to install in .venv-$env; SDPA attention will be used"
  done
fi

.venv-tts/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
.venv-judge/bin/python -c "import transformers; print('judge transformers', transformers.__version__)"
[ -x .venv-dpo/bin/python ] && .venv-dpo/bin/python -c "import transformers, trl; print('dpo transformers', transformers.__version__, '| trl', trl.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "Setup done."
