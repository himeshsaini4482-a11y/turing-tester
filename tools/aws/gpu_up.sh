#!/usr/bin/env bash
# Launch the GPU worker (billing starts) in the first zone with capacity, wait for SSH.
# Reuses the current worker if it is already running.
#   bash tools/aws/gpu_up.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

wait_for_ssh() {
  for _ in $(seq 1 60); do
    gpu_ssh true 2>/dev/null && { echo "GPU worker up: $GPU_INSTANCE_ID in $GPU_REGION ($GPU_HOST)"; return 0; }
    sleep 5
  done
  echo "Worker is running but SSH did not answer within 5 min" >&2
  return 1
}

state=$(gpu_state)
if [ "$state" = running ] || [ "$state" = pending ]; then
  wait_for_ssh; exit
fi
if [ "$state" = stopped ] || [ "$state" = stopping ]; then
  # Left by the idle auto-stop: its disk may hold results that were never copied back
  echo "Previous worker $GPU_INSTANCE_ID ($GPU_REGION) is stopped and may hold unsynced runs/." >&2
  echo "Restart it with:  aws ec2 start-instances --region $GPU_REGION --instance-ids $GPU_INSTANCE_ID" >&2
  echo "then run gpu_down.sh to fetch results; or delete it: aws ec2 terminate-instances ..." >&2
  exit 1
fi

my_ip=$(curl -s -m 5 https://checkip.amazonaws.com || true)
for round in $(seq 1 "$GPU_ROUNDS"); do
  for type in $GPU_TYPES; do
    while read -r region zone subnet sg ami via; do
      [ -n "$region" ] || continue
      [[ "$ami" == ami-* ]] || continue   # image not copied to this region yet
      if [ "$via" = public ] && [ -n "$my_ip" ]; then
        # this server's public IP can change if it is ever stopped; keep the SSH rule current
        aws ec2 authorize-security-group-ingress --region "$region" --group-id "$sg" \
          --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$my_ip/32,Description=uai-server}]" \
          >/dev/null 2>&1 || true
      fi
      if out=$(aws ec2 run-instances --region "$region" --image-id "$ami" --instance-type "$type" \
          --key-name voicegym-gpu --subnet-id "$subnet" --security-group-ids "$sg" \
          --instance-initiated-shutdown-behavior stop --metadata-options HttpTokens=required \
          --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=voicegym-gpu},{Key=Project,Value=voicegym}]' \
            'ResourceType=volume,Tags=[{Key=Name,Value=voicegym-gpu},{Key=Project,Value=voicegym}]' \
          --query 'Instances[0].InstanceId' --output text 2>&1); then
        GPU_REGION=$region GPU_INSTANCE_ID=$out
        echo "Launched $type in $zone: $GPU_INSTANCE_ID (billing starts now)"
        aws ec2 wait instance-running --region "$region" --instance-ids "$GPU_INSTANCE_ID"
        field=$([ "$via" = public ] && echo PublicIpAddress || echo PrivateIpAddress)
        GPU_HOST=$(aws ec2 describe-instances --region "$region" --instance-ids "$GPU_INSTANCE_ID" \
          --query "Reservations[0].Instances[0].$field" --output text)
        printf 'GPU_REGION=%s\nGPU_INSTANCE_ID=%s\nGPU_HOST=%s\n' "$GPU_REGION" "$GPU_INSTANCE_ID" "$GPU_HOST" > "$GPU_STATE"
        wait_for_ssh; exit
      fi
      case "$out" in
        *InsufficientInstanceCapacity*|*Unsupported*|*VcpuLimitExceeded*|*InstanceLimitExceeded*)
          echo "  no $type in $zone: $(grep -o '[A-Za-z]*\(Capacity\|Unsupported\|Exceeded\)' <<<"$out" | head -1)" ;;
        *) echo "$out" >&2; exit 1 ;;
      esac
    done <<<"$GPU_ZONES"
  done
  [ "$round" -lt "$GPU_ROUNDS" ] && { echo "No capacity anywhere (round $round/$GPU_ROUNDS), retrying in 60 s ..."; sleep 60; }
done
echo "No L40S capacity in any zone after $GPU_ROUNDS rounds" >&2
exit 1
