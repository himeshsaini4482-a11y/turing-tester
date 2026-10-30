# Shared settings for the on-demand VoiceGym GPU worker (sourced by the gpu_*.sh scripts).
#
# The worker is disposable: gpu_up.sh launches it from the voicegym-gpu AMI in the first
# zone/type with capacity, gpu_down.sh copies runs/ back and terminates it.

GPU_KEY=~/.ssh/voicegym-gpu.pem
GPU_DIR=voicegym                       # code dir on the worker, relative to /home/ubuntu
GPU_TYPES=${GPU_TYPES:-g6e.xlarge}      # 1x L40S 48 GB, cheapest; GPU_TYPES="g6e.xlarge g6e.2xlarge" also allows the 2xlarge (+CPU, ~$0.46/hr more)
GPU_ROUNDS=${GPU_ROUNDS:-10}           # passes over all zones before giving up, 60 s apart

# region  zone  subnet  security-group  ami  ssh-via
# Hyderabad shares the UAI server's VPC (private IP); Mumbai is reached over its public IP.
GPU_ZONES="
ap-south-2 ap-south-2a subnet-045778026cf8869d8 sg-02eab69a9b284c9fe ami-0e52ac5004c754c36 private
ap-south-2 ap-south-2b subnet-0daf259cbc028a749 sg-02eab69a9b284c9fe ami-0e52ac5004c754c36 private
ap-south-1 ap-south-1a subnet-04c7b96e2aa6ddeae sg-018399f005815382e ami-0061b2572e9464b94 public
ap-south-1 ap-south-1b subnet-003cdb8aa3f83d3cb sg-018399f005815382e ami-0061b2572e9464b94 public
"

# Current worker, written by gpu_up.sh: GPU_REGION, GPU_INSTANCE_ID, GPU_HOST
GPU_STATE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/state.env
[ -f "$GPU_STATE" ] && source "$GPU_STATE"

SSH_OPTS=(-i "$GPU_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ConnectTimeout=5 -o ServerAliveInterval=30)

gpu_ssh() { ssh "${SSH_OPTS[@]}" "ubuntu@$GPU_HOST" "$@"; }

# Forget the worker, unless state.env already names a newer one (another gpu_up ran meanwhile)
gpu_forget() {
  grep -q "^GPU_INSTANCE_ID=${GPU_INSTANCE_ID:-}$" "$GPU_STATE" 2>/dev/null && rm -f "$GPU_STATE" || true
}

gpu_state() {   # prints the worker's EC2 state, or "none"
  [ -n "${GPU_INSTANCE_ID:-}" ] || { echo none; return; }
  aws ec2 describe-instances --region "$GPU_REGION" --instance-ids "$GPU_INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo none
}
