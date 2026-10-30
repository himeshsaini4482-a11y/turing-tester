#!/usr/bin/env bash
# Prints the SpeechJudge-GRM directory judge.py should load from.
#
# On the AWS worker (Deep Learning AMI) the 22 GB checkpoint loads very slowly from the EBS
# root disk: the loader's small reads get ~16 MB/s, and on a worker freshly launched from
# the image the blocks are first pulled lazily from the snapshot. So once per boot it is
# downloaded from Hugging Face to the local NVMe instance store (wiped on stop), which
# loads fast. Elsewhere it prints the normal path and judge.py downloads there as before.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=pretrained/SpeechJudge-GRM
NVME=/opt/dlami/nvme
DST=$NVME/SpeechJudge-GRM

if [ ! -d "$NVME" ]; then
  echo "$SRC"; exit 0
fi
if [ ! -f "$DST/.staged" ]; then
  echo "Downloading SpeechJudge-GRM to local NVMe $DST ..." >&2
  sudo mkdir -p "$DST" && sudo chown "$(id -u):$(id -g)" "$DST"
  start=$(date +%s)
  "${JUDGE_PY:-.venv-judge/bin/python}" -c "
from huggingface_hub import snapshot_download
snapshot_download('RMSnow/SpeechJudge-GRM', local_dir='$DST', max_workers=16)
" >&2
  touch "$DST/.staged"
  echo "Staged in $(( $(date +%s) - start )) s." >&2
fi
echo "$DST"
