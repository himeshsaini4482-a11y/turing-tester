#!/usr/bin/env bash
# Round stage R-1/R-2: SpeechJudge-GRM diagnosis of a student checkpoint, then 9B weakness tags.
#
#   bash scripts/sft/diagnose.sh STAGE EVAL_NAME
#
# Takes the seed-0 dev-mirror rows (set == mirror, sentence_id ending __s0) of an existing eval
# run $RUNS/eval/EVAL_NAME (eval_ckpt.sh), judges student vs human in both A/B orders (1 vote)
# and tags the critiques:
#   $RUNS/STAGE/diag/{manifest.csv, judgments.jsonl, summary.json}           benchmark/judge.py
#   $RUNS/STAGE/diag/{tags.jsonl, weakness_report.json, weakness_report.md}  sft/tag_weaknesses.py
# Both steps are resumable and marked done in $RUNS/.done/.
source "$(dirname "$0")/env.sh"

if [ $# -ne 2 ]; then
  echo "usage: $0 STAGE EVAL_NAME" >&2; exit 2
fi
STAGE=$1
EVAL_NAME=$2
EVAL_DIR=$RUNS/eval/$EVAL_NAME
DIAG=$RUNS/$STAGE/diag

[ -f "$EVAL_DIR/manifest.csv" ] || { echo "missing $EVAL_DIR/manifest.csv (run eval_ckpt.sh first)" >&2; exit 1; }
mkdir -p "$DIAG"

diag_manifest() {
  "$TTS_PY" - "$EVAL_DIR/manifest.csv" "$DIAG/manifest.csv" <<'EOF'
import sys
sys.path.insert(0, "sft")
import sftlib
src, dst = sys.argv[1], sys.argv[2]
rows = [r for r in sftlib.read_csv(src)
        if r.get("set") == "mirror" and r["sentence_id"].endswith("__s0")]
if not rows:
    sys.exit(f"no seed-0 mirror rows in {src}")
sftlib.write_csv(dst, rows)
print(f"{len(rows)} seed-0 mirror rows -> {dst}")
EOF
}

diag_judge() {
  diag_manifest
  "$JUDGE_PY" benchmark/judge.py --manifest "$DIAG/manifest.csv" --out-dir "$DIAG" \
    --model-path "$(judge_model_path)" --votes 1
}

diag_tag() {
  "$TTS_PY" sft/tag_weaknesses.py --diag-dir "$DIAG" --eval-dir "$EVAL_DIR" --stage "$STAGE"
}

step "${STAGE}_diag_judge" diag_judge
step "${STAGE}_diag_tag" diag_tag
