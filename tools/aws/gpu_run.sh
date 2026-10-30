#!/usr/bin/env bash
# Run a command on a GPU worker, then get rid of it:
#   launch worker -> copy code -> run command in tmux -> copy small runs/ reports back (FULL_PULL=1: all) -> terminate worker
#
#   bash tools/aws/gpu_run.sh "RUN_NAME=smoke LIMIT=5 bash scripts/run_benchmark.sh"
#   KEEP_UP=1 bash tools/aws/gpu_run.sh "..."     # leave the worker running (no idle stop) until gpu_down.sh
#   PUSH_RUNS=ears_smoke bash tools/aws/gpu_run.sh "RUN_NAME=ears_smoke LIMIT=5 bash scripts/run_btrm.sh"
#
# The job runs in a tmux session on the worker, so it survives this terminal closing.
# If this script dies mid-job, the worker's idle timer stops it 30 min after the job ends
# (stopped, not terminated, so results survive); gpu_up.sh then says how to recover them.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 1 ] || { echo "usage: $0 \"<command>\"" >&2; exit 1; }
CMD=$1

bash tools/aws/gpu_up.sh
source tools/aws/config.sh
if [ "${KEEP_UP:-0}" = 1 ]; then
  # also keep the idle timer (user-data.sh) from powering it off; gpu_down.sh ends it
  gpu_ssh 'touch ~/.keep-awake'
else
  trap 'bash tools/aws/gpu_down.sh' EXIT
fi

RSYNC_SSH="ssh ${SSH_OPTS[*]}"
rsync -az -e "$RSYNC_SSH" --exclude '.venv-*' --exclude '/data/' --exclude '/pretrained/' \
  --exclude '/third_party/' --exclude '/runs/' --exclude '__pycache__' --exclude 'tools/aws/state.env' \
  ./ "ubuntu@$GPU_HOST:$GPU_DIR/"
# PUSH_RUNS="a b": copy existing runs/a, runs/b to the worker (e.g. to score TTS audio made earlier)
for run in ${PUSH_RUNS:-}; do
  rsync -az -e "$RSYNC_SSH" "runs/$run" "ubuntu@$GPU_HOST:$GPU_DIR/runs/"
done
# pick up requirement changes made since the image was built (no-op when nothing changed)
gpu_ssh "cd $GPU_DIR && for e in tts judge; do .venv-\$e/bin/pip install -q -r requirements-\$e.txt; done"

if gpu_ssh 'tmux has-session -t vg 2>/dev/null'; then
  echo "A job is already running on the worker; waiting for it instead of starting a new one."
else
  printf -v QCMD '%q' "$CMD"
  # keep the previous job's log instead of truncating it
  gpu_ssh "cd $GPU_DIR && mkdir -p runs/logs && if [ -s runs/job.log ]; then mv runs/job.log runs/logs/job.\$(date -u +%Y%m%dT%H%M%S).log; fi; rm -f runs/job.exit"
  gpu_ssh "cd $GPU_DIR && mkdir -p runs && tmux new-session -d -s vg \"bash -c $QCMD > runs/job.log 2>&1; echo \\\$? > runs/job.exit\""
  echo "Job started on worker: $CMD"
fi

gpu_ssh "cd $GPU_DIR && touch runs/job.log && tail -n +1 -f runs/job.log --pid=\$(tmux display -p -t vg '#{pane_pid}' 2>/dev/null || echo 1)" || true
while gpu_ssh 'tmux has-session -t vg 2>/dev/null'; do sleep 10; done

mkdir -p runs
if [ "${FULL_PULL:-0}" = 1 ]; then
  rsync -az -e "$RSYNC_SSH" "ubuntu@$GPU_HOST:$GPU_DIR/runs/" runs/
else
  # default: only small reports (audio, checkpoints and rollouts stay on the worker; this host's disk is small)
  rsync -az -e "$RSYNC_SSH" --prune-empty-dirs --max-size=20m \
    --include='*/' --include='*.json' --include='*.md' --include='*.log' --include='*.csv' --include='*.txt' \
    --include='job.exit' --include='.done/*' --exclude='*' \
    "ubuntu@$GPU_HOST:$GPU_DIR/runs/" runs/
fi
rsync -az -e "$RSYNC_SSH" "ubuntu@$GPU_HOST:$GPU_DIR/benchmark/manifests/" benchmark/manifests/ 2>/dev/null || true
status=$(cat runs/job.exit 2>/dev/null || echo "?")
echo "Job finished with exit code $status. Results copied to $(pwd)/runs/"
[ "$status" = 0 ]
