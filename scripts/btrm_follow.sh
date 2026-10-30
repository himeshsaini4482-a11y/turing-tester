#!/usr/bin/env bash
# Score run_btrm_full.sh's clips with BTRM while they are generated (benchmark/btrm_follow.py),
# so little BTRM work is left when generation ends. Start it next to the running job:
#
#   source tools/aws/config.sh
#   gpu_ssh 'cd voicegym && tmux new-session -d -s btrm "bash scripts/btrm_follow.sh > runs/btrm_follow.log 2>&1"'
#
# Never puts a second BTRM model on the GPU: btrm_follow.py exits when a run finishes
# generating, and is restarted for the other runs only after that run's own BTRM is done.
set -euo pipefail
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH   # see run_btrm.sh
MANIFEST=benchmark/manifests/ears_regular_26x19.csv
N=$(($(wc -l < "$MANIFEST") - 1))
DIR=pretrained   # same weights as run_btrm.sh: the AWS worker's local NVMe when present
[ -d /opt/dlami/nvme ] && DIR=/opt/dlami/nvme/btrm
RUNS=${RUNS:-"runs/moss_nano_full runs/orpheus_full"}

while :; do
  active=() busy=0
  for run in $RUNS; do
    if [ ! -f "$run/manifest.csv" ]; then
      active+=("$run")
    elif ! python3 -c "import json,sys; sys.exit(json.load(open('$run/btrm_summary.json'))['n_sentences'] != $N)" 2>/dev/null; then
      busy=1   # generation done, its btrm_score.py still running
    fi
  done
  [ ${#active[@]} -gt 0 ] || { echo "All runs generated; nothing left to follow."; exit 0; }
  if [ $busy = 1 ]; then sleep 60; continue; fi
  echo "== following ${active[*]}"
  .venv-judge/bin/python benchmark/btrm_follow.py --manifest "$MANIFEST" --runs "${active[@]}" \
    --base-path "$DIR/Qwen2.5-Omni-7B" --rm-path "$DIR/SpeechJudge-BTRM"
done
