#!/usr/bin/env bash
# DPO environment for the hparams model (idempotent): its venv + tensorboard, model and codec.
#   moss_nano: the SFT MOSS venv (scripts/sft/setup_moss.sh) + tensorboard
#   orpheus:   .venv-dpo-orpheus from requirements-dpo-orpheus.txt (peft, snac, tensorboard)
#
#   bash scripts/dpo/setup.sh
source "$(dirname "$0")/env.sh"
case "$MODEL" in
  moss_nano)
    bash scripts/sft/setup_moss.sh
    UV=$(command -v uv || echo "$MOSS_DIR/bin/uv")
    "$UV" pip install -q --python "$MOSS_PY" "tensorboard>=2.17"
    ;;
  orpheus)
    if [ ! -x "$DPO_PY" ]; then
      python3 -m venv "$REPO_DIR/.venv-dpo-orpheus"
    fi
    "$DPO_PY" -m pip install -q -r requirements-dpo-orpheus.txt
    ;;
esac
"$DPO_PY" -c "import torch, transformers; from torch.utils.tensorboard import SummaryWriter; \
print('DPO venv ($MODEL): torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
