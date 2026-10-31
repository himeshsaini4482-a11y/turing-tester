#!/usr/bin/env bash
# Copy runs/ back from the GPU worker, then terminate it (billing stops, nothing is left).
# If results cannot be copied, the worker is only stopped so they are not lost.
#   bash tools/aws/gpu_down.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/aws/config.sh

state=$(gpu_state)
if [ "$state" = none ] || [ "$state" = terminated ] || [ "$state" = shutting-down ]; then
  echo "No GPU worker running."; gpu_forget; exit 0
fi

if [ "$state" = running ] && gpu_ssh "tmux kill-session -t vg 2>/dev/null; true" \
   && rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$GPU_HOST:$GPU_DIR/runs/" runs/; then
  aws ec2 terminate-instances --region "$GPU_REGION" --instance-ids "$GPU_INSTANCE_ID" >/dev/null
  echo "Results copied to $(pwd)/runs/. Terminating $GPU_INSTANCE_ID ..."
  aws ec2 wait instance-terminated --region "$GPU_REGION" --instance-ids "$GPU_INSTANCE_ID"
  gpu_forget
  echo "GPU worker terminated."
else
  echo "Could not copy results from $GPU_INSTANCE_ID; stopping it instead of terminating." >&2
  aws ec2 stop-instances --region "$GPU_REGION" --instance-ids "$GPU_INSTANCE_ID" >/dev/null
  exit 1
fi
