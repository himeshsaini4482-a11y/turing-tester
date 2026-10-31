#!/usr/bin/env bash
# Show whether a GPU worker exists (i.e. costs money) and what it is doing.
set -euo pipefail
source "$(dirname "$0")/config.sh"

for region in ap-south-2 ap-south-1; do
  aws ec2 describe-instances --region "$region" \
    --filters Name=tag:Project,Values=voicegym Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,LaunchTime]' \
    --output text
done | { grep . || echo "No Turing Tester GPU workers exist."; }

if [ "$(gpu_state)" = running ] && gpu_ssh true 2>/dev/null; then
  gpu_ssh 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; tmux ls 2>/dev/null || echo "no job running"'
fi
