#!/usr/bin/env bash
# The test round (pilot): settle every open design question before the main training
# (INTERFACES §4), then write $RUNS/pilot/recipe.json for scripts/sft/main.sh.
#
#   KEEP_UP=1 bash tools/aws/gpu_run.sh "bash scripts/sft/pilot.sh"
#   ARMS="B A CU" TEACHER_GPU_HOURS=3 PILOT_TEXTS=1000 F1_POLICY=exclude bash scripts/sft/pilot.sh
#
# Order: setup -> gates -> EARS inventory -> dev-mirror -> base eval under both decodings ->
# decoding decision -> base 494 under both decodings (no GRM) -> align -> build EARS chunks ->
# dev-fresh -> encode -> base full dev eval -> GRM diagnosis -> teacher round -> arms
# (mix, train, eval) -> decide.py pilot.
#
# Resumable: every step leaves a marker in $RUNS/.done/ (see `step` in env.sh), so after a
# crash or worker restart just launch it again. Exit 3 = STOP (no pilot arm beats base).
#
# If the teacher round accepted nothing, arms A, B and D are skipped (CU, CT, F still run).
#
# Env: ARMS (default "B A CU CT D F"; B before A because A matches B's teacher hours),
#      TEACHER_GPU_HOURS (default 5), PILOT_TEXTS (default 4000), F1_POLICY (auto|include|exclude).
source "$(dirname "$0")/env.sh"

ARMS=${ARMS:-B A CU CT D F}
TEACHER_GPU_HOURS=${TEACHER_GPU_HOURS:-5}
PILOT_TEXTS=${PILOT_TEXTS:-4000}   # > what 5 GPU-h renders: the teacher is time-boxed
TEACHER_CODES=$RUNS/pilot/teacher/codes
TEACHER_ACCEPTED=$RUNS/pilot/teacher/accepted_raw.jsonl
LR=2e-5
SAVE_EVERY=100   # > epochs: only checkpoint-last is saved and evaluated

# Decoding mode chosen at step 6 (read at call time, never before decoding.json exists).
dec_mode() {
  "$TTS_PY" -c "import sys; sys.path.insert(0, 'sft'); import sftlib; \
print(sftlib.read_json(sftlib.RUNS / 'decoding.json', {}).get('mode', 'chunked'))"
}

# Gate G0c (9B reachable) is not fatal: R-2/R-3 can run on the UAI server instead.
gate_qwen() {
  mkdir -p "$RUNS/pilot"
  if "$TTS_PY" sft/gates.py qwen; then
    echo pass > "$RUNS/pilot/gate_qwen.txt"
  else
    echo fail > "$RUNS/pilot/gate_qwen.txt"
    echo "WARNING: gate qwen failed (9B unreachable from the worker); continuing." \
         "Tagging/text generation must then run on the UAI server (RUNBOOK S0)."
  fi
}

bench_base() {   # $1 = chunked|whole, $2 = text tokens
  MOSS_MODEL="$MOSS_MODEL_DIR" RUN_NAME="${BENCH_PREFIX}moss_base_$1" MOSS_MAX_TEXT_TOKENS="$2" SKIP_GRM=1 \
    bash scripts/run_moss_nano.sh
}

base_eval_full() {
  bash scripts/sft/eval_ckpt.sh "base_$(dec_mode)" "$MOSS_MODEL_DIR" $RUNS/dev/eval_full.csv "$(dec_tokens)"
}

dev_fresh() {
  "$TTS_PY" sft/dev_sets.py fresh
  "$TTS_PY" sft/dev_sets.py eval-manifest --sets mirror,fresh --seeds 3 --out $RUNS/dev/eval_full.csv
}

dev_mirror() {
  "$TTS_PY" sft/dev_sets.py mirror
  "$TTS_PY" sft/dev_sets.py eval-manifest --sets mirror --seeds 3 --out $RUNS/dev/eval_mirror.csv
}

# make_mix.py config per arm (INTERFACES §2.9 / §4). No category re-weighting in the pilot, so
# the arms stay comparable; f1_include is only true in arm F (with the dev-leak probe).
arm_config() {
  local ears teacher
  case $1 in
    A)  ears='{"use": true, "freeform_tag": "U", "f1_include": false, "leak_probe": false, "match_teacher": "'"$TEACHER_ACCEPTED"'", "category_weights": {}}'
        teacher='{"codes": [], "oversample": 1.0}' ;;
    B)  ears='{"use": false}'
        teacher='{"codes": ["'"$TEACHER_CODES"'"], "oversample": 1.0}' ;;
    CU) ears='{"use": true, "freeform_tag": "U", "f1_include": false, "leak_probe": false, "match_teacher": null, "category_weights": {}}'
        teacher='{"codes": [], "oversample": 1.0}' ;;
    CT) ears='{"use": true, "freeform_tag": "T", "f1_include": false, "leak_probe": false, "match_teacher": null, "category_weights": {}}'
        teacher='{"codes": [], "oversample": 1.0}' ;;
    D)  ears='{"use": true, "freeform_tag": "U", "f1_include": false, "leak_probe": false, "match_teacher": null, "category_weights": {}}'
        teacher='{"codes": ["'"$TEACHER_CODES"'"], "oversample": 3.0}' ;;
    F)  ears='{"use": true, "freeform_tag": "U", "f1_include": true, "leak_probe": true, "match_teacher": null, "category_weights": {}}'
        teacher='{"codes": [], "oversample": 1.0}' ;;
    *)  echo "unknown pilot arm: $1" >&2; return 1 ;;
  esac
  echo '{"ears": '"$ears"', "teacher": '"$teacher"', "seed": 0}'
}

# Zero teacher yield is possible (voice filter 0.88 vs Orpheus ~0.85): then A, B, D are skipped.
teacher_ok() { ls "$TEACHER_CODES"/*.jsonl >/dev/null 2>&1; }

arm_epochs() { case $1 in A|B) echo 6 ;; *) echo 2 ;; esac; }

# Mix and train in one step: the mix lives on NVMe ($SCR), which is wiped when the worker stops.
train_arm() {
  local arm=$1 mix
  if [ -s "$RUNS/ckpt/pilot/$arm/checkpoint-last/pytorch_model.bin" ]; then
    echo "pilot arm $arm: checkpoint-last exists, skipping mix + train"
    return 0
  fi
  # a (re)trained model invalidates any earlier evaluation of this arm
  rm -f "$RUNS/.done/pilot_15_eval_$arm"
  mix=$("$MOSS_PY" sft/make_mix.py --stage pilot --arm "$arm" --config "$(arm_config "$arm")" | tail -1)
  [ -s "$mix" ] || { echo "make_mix for arm $arm gave no mix file: '$mix'" >&2; return 1; }
  bash scripts/sft/train.sh pilot "$arm" "$mix" "$LR" "$(arm_epochs "$arm")" "$SAVE_EVERY"
}

eval_arm() {
  local arm=$1
  bash scripts/sft/eval_ckpt.sh "pilot_$arm" "$RUNS/ckpt/pilot/$arm/checkpoint-last" \
    $RUNS/dev/eval_full.csv "$(dec_tokens)" "base_$(dec_mode)"
}

decide_pilot() {
  local rc=0
  "$TTS_PY" sft/decide.py pilot || rc=$?
  if [ "$rc" -eq 3 ]; then
    echo
    echo "=================================================================="
    echo " PILOT STOP: no pilot arm beats base on dev-mirror (CI lo > 0)."
    echo " Do not start the main workflow. Read $RUNS/pilot/pilot_report.md"
    echo " and $RUNS/DECISIONS.md, then report to the user (PLAN §8)."
    echo "=================================================================="
    exit 3
  fi
  return "$rc"
}

for arm in $ARMS; do arm_config "$arm" >/dev/null; done   # fail fast on a bad ARMS

# not a step: the MOSS venv, pinned repo and model live on the NVMe, which a worker stop wipes;
# setup_moss.sh is idempotent and quick when everything is already there
bash scripts/sft/setup_moss.sh
step pilot_02a_gate_prompt     "$MOSS_PY" sft/gates.py prompt
step pilot_02b_gate_roundtrip  "$MOSS_PY" sft/gates.py roundtrip
step pilot_02c_gate_qwen       gate_qwen
step pilot_03_ears_inventory   "$TTS_PY" sft/ears_inventory.py
step pilot_04_dev_mirror       dev_mirror
step pilot_05a_base_eval_chunked bash scripts/sft/eval_ckpt.sh base_chunked "$MOSS_MODEL_DIR" $RUNS/dev/eval_mirror.csv 50
step pilot_05b_base_eval_whole   bash scripts/sft/eval_ckpt.sh base_whole "$MOSS_MODEL_DIR" $RUNS/dev/eval_mirror.csv 100
step pilot_06_decoding         "$TTS_PY" sft/decide.py decoding --chunked $RUNS/eval/base_chunked --whole $RUNS/eval/base_whole
step pilot_07a_bench_chunked   bench_base chunked 50
step pilot_07b_bench_whole     bench_base whole 100
step pilot_08_align            "$TTS_PY" sft/align.py
step pilot_09_build_ears       "$MOSS_PY" sft/build_ears.py --max-tokens "$(train_max_tokens)" --moss-repo "$MOSS_REPO" --model-dir "$MOSS_MODEL_DIR"
step pilot_10_dev_fresh        dev_fresh
step pilot_11_encode_ears      bash scripts/sft/encode.sh $RUNS/data/ears_raw.jsonl $RUNS/data/ears_codes
step pilot_12_base_eval_full   base_eval_full
step pilot_13_diagnose         bash scripts/sft/diagnose.sh pilot "base_$(dec_mode)"
step pilot_14_teacher          bash scripts/sft/teacher_round.sh pilot "$MOSS_MODEL_DIR" "$TEACHER_GPU_HOURS" "$PILOT_TEXTS"
for arm in $ARMS; do
  if [[ " A B D " == *" $arm "* ]] && ! teacher_ok; then
    echo "== [skip] pilot arm $arm: the pilot teacher round accepted no clips" \
         "($TEACHER_CODES has no *.jsonl); decide.py falls back to recipe H"
    continue
  fi
  step "pilot_15_train_$arm"   train_arm "$arm"
  step "pilot_15_eval_$arm"    eval_arm "$arm"
done
step pilot_16_decide           decide_pilot
echo "== pilot done: $RUNS/pilot/recipe.json, $RUNS/pilot/pilot_report.md"
