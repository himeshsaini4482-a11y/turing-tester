# Sourced by every scripts/dpo/*.sh: the SFT paths and venvs (scripts/sft/env.sh), plus the DPO
# model's venv and the resumable `dstep` helper.
#
#   source "$(dirname "$0")/env.sh"
#
# Durable outputs go to runs/dpo/ (worker EBS). Candidate wavs go to the NVMe scratch.
source "$(dirname "${BASH_SOURCE[0]}")/../sft/env.sh"
DRUNS=$REPO_DIR/runs/dpo
mkdir -p "$DRUNS"
export DPO_SCRATCH=$NV/dpo
nv_dir "$DPO_SCRATCH"

# hp KEY.PATH: one value from dpo/hparams.json (lists/dicts as JSON)
hp() {
  "$TTS_PY" -c "import json, sys; v = json.load(open('dpo/hparams.json'))
for k in sys.argv[1].split('.'): v = v[k]
print(json.dumps(v) if isinstance(v, (dict, list)) else v)" "$1"
}
MODEL=$(hp model)
case "$(hp "models.$MODEL.venv")" in
  MOSS) DPO_PY=$MOSS_PY ;;
  ORPHEUS) DPO_PY=$REPO_DIR/.venv-dpo-orpheus/bin/python ;;
  *) echo "dpo env.sh: unknown venv for model $MODEL" >&2; exit 1 ;;
esac
export MODEL DPO_PY DRUNS

# dstep NAME CMD...: run CMD once; a marker in runs/dpo/.done/ makes re-runs skip it
# Every step also goes to runs/dpo/logs/stages.tsv (start, end, exit code, name; failures included),
# in the same timestamp format as gpu.csv, so GPU use can be attributed to stages.
dstep() {
  local name=$1; shift
  local marker=$DRUNS/.done/$name
  if [ -f "$marker" ]; then echo "== [skip] $name (done $(cat "$marker")s)"; return 0; fi
  echo "== [$(date +%H:%M:%S)] $name"
  rm -f "$DRUNS/STATUS.json"
  local t0 ts rc=0; t0=$(date +%s); ts=$(date '+%Y/%m/%d %H:%M:%S')
  "$@" || rc=$?
  mkdir -p "$DRUNS/logs"
  printf '%s\t%s\t%s\t%s\n' "$ts" "$(date '+%Y/%m/%d %H:%M:%S')" "$rc" "$name" >> "$DRUNS/logs/stages.tsv"
  [ "$rc" = 0 ] || return "$rc"
  local dt=$(( $(date +%s) - t0 ))
  mkdir -p "$DRUNS/.done"
  echo "$dt" > "$marker"
  "$TTS_PY" -c "import sys; sys.path.insert(0, 'dpo'); import dpolib; dpolib.log_timing('$name', $dt)"
  echo "== [done] $name in ${dt}s"
}

# current policy (runs/dpo/current.json, written by train.py); round 0 = models.<model>.policy_init
current_policy() {
  "$TTS_PY" -c "import json, os; f = 'runs/dpo/current.json'
print(json.load(open(f))['policy'] if os.path.exists(f) else '$(hp "models.$MODEL.policy_init")')"
}

# GPU telemetry the observability agent reads (1 line / 10 s); stopped by the caller's EXIT trap
gpu_logger_start() {
  mkdir -p "$DRUNS/logs"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader -l 10 >> "$DRUNS/logs/gpu.csv" 2>/dev/null &
  SMI=$!
  trap 'kill $SMI 2>/dev/null || true' EXIT
}

# pause_for_user STATE DETAIL: record why the run stopped for the user and let the worker's 30-min
# idle stop (tools/aws/user-data.sh) act, so a paused run does not bill while you decide.
# Stopping keeps runs/ (EBS); relaunch the same command afterwards.
pause_for_user() {
  "$TTS_PY" - "$1" "$2" <<'PY'
import json, sys, time
json.dump({"state": sys.argv[1], "detail": sys.argv[2], "at": time.strftime("%Y-%m-%d %H:%M:%S")},
          open("runs/dpo/STATUS.json", "w"), indent=2)
PY
  rm -f ~/.keep-awake
  echo "dpo: PAUSED ($1): $2"
  echo "dpo: the worker stops itself after 30 idle minutes; relaunch the same command when ready."
}
