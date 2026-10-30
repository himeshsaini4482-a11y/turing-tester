#!/usr/bin/env bash
# Optimizer / beta sweep on round 1's pairs (each setting: train from pi_ref, then the dev eval).
# Settings come from dpo/sweep.json, which you must approve first ("approved": true). Results are
# ranked in runs/dpo/sweep/SWEEP_REPORT.md; you then copy the chosen values into dpo/hparams.json.
#
#   KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/dpo/sweep.sh"
#
# Needs round 1 up to its pairs (scripts/dpo/run.sh does that before it pauses).
source "$(dirname "$0")/env.sh"
APPROVED=$("$TTS_PY" -c "import json; print(json.load(open('dpo/sweep.json'))['approved'])")
[ "$APPROVED" = True ] || { pause_for_user sweep_unapproved "check dpo/sweep.json and set \"approved\": true"; exit 3; }
[ -s "$DRUNS/r01/pairs.jsonl" ] || { echo "sweep.sh: no round-1 pairs; run scripts/dpo/run.sh first" >&2; exit 1; }
bash scripts/dpo/setup.sh
gpu_logger_start
TAGS=$("$TTS_PY" -c "import json; print(' '.join(s['tag'] for s in json.load(open('dpo/sweep.json'))['settings']))")
mkdir -p "$DRUNS/sweep"
for tag in $TAGS; do
  ov=$DRUNS/sweep/$tag.overlay.json
  "$TTS_PY" - "$tag" "$ov" <<'PY'
import json, sys
tag, out = sys.argv[1], sys.argv[2]
s = next(x for x in json.load(open("dpo/sweep.json"))["settings"] if x["tag"] == tag)
json.dump({k: s[k] for k in ("beta", "kl", "optimizer") if k in s}, open(out, "w"), indent=2)
PY
  dstep "sweep_${tag}_train" "$DPO_PY" dpo/train.py --round 1 --tag "$tag" --overlay "$ov"
  dstep "sweep_${tag}_eval" bash scripts/dpo/eval.sh "sweep_$tag" "$DRUNS/sweep/$tag/ckpt" r00
done
"$TTS_PY" dpo/rounds.py sweep
