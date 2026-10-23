#!/usr/bin/env bash
# Main workflow, run after the pilot (scripts/sft/pilot.sh) has written runs/sft/pilot/recipe.json.
# Each round:
#   GRM diagnoses the current best checkpoint on dev-mirror -> the 9B writes texts for the confirmed
#   weaknesses -> Orpheus speaks them (GPU-hour budget from the recipe) -> filter -> training mix
#   (EARS per recipe, re-weighted to the weak categories, + all teacher data so far) -> SFT from
#   the base model -> dev eval of every epoch -> keep the best; stop early if it is not better.
#   Each round also trains an "alt" arm with the smaller source's share of the mix raised (same
#   size and lr); if it beats the main arm, later rounds use its share (sft/reweight.py).
# Then scripts/sft/final.sh runs the full 494 benchmark (unless SKIP_FINAL=1).
#
#   KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/main.sh"
#
# Env overrides: MAIN_ROUNDS, TEACHER_GPU_HOURS, MAIN_EPOCHS (3), SWEEP (1: last round also
# trains at half the learning rate), REWEIGHT (1: train the alt arm each round), FORCE=1 (run
# even if the pilot said stop), SKIP_FINAL=1.
# Resumable: finished steps are skipped (runs/sft/.done/).
source "$(dirname "$0")/env.sh"
RECIPE=$RUNS/pilot/recipe.json
[ -f "$RECIPE" ] || { echo "main.sh: $RECIPE is missing; run scripts/sft/pilot.sh first" >&2; exit 1; }
rj() { "$TTS_PY" -c "import json; r = json.load(open('$RECIPE')); print($1)"; }

if [ "$(rj "r['status']")" != go ] && [ "${FORCE:-0}" != 1 ]; then
  echo "main.sh: the pilot decided STOP (see runs/sft/pilot/pilot_report.md). FORCE=1 overrides." >&2
  exit 3
fi
# MOSS venv/repo/model live on the NVMe (wiped if the worker stopped since the pilot); idempotent
bash scripts/sft/setup_moss.sh
ROUNDS=${MAIN_ROUNDS:-$(rj "r['main_rounds']")}
LR=$(rj "r['lr']")
TEACHER_H=${TEACHER_GPU_HOURS:-$(rj "r['teacher_gpu_hours_per_round']")}
EPOCHS=${MAIN_EPOCHS:-3}
TOK=$(dec_tokens)
BASE_EVAL=base_$("$TTS_PY" -c "import json; print(json.load(open('$RUNS/decoding.json'))['mode'])")
EVAL_MAN=$RUNS/dev/eval_full.csv
echo "main.sh: recipe $(rj "r['recipe']"), $ROUNDS round(s), lr $LR, teacher ${TEACHER_H} GPU-h/round, \
decoding chunk limit $TOK, base eval $BASE_EVAL"

"$TTS_PY" -c "import sys; sys.path.insert(0, 'sft'); import round_select; round_select.current_best()"
best() { "$TTS_PY" -c "import json; print(json.load(open('$RUNS/main/best.json'))['$1'])"; }

for r in $(seq 1 "$ROUNDS"); do
  S=main_r$r
  if [ -f "$RUNS/$S/round_summary.json" ]; then
    stop=$("$TTS_PY" -c "import json; print(json.load(open('$RUNS/$S/round_summary.json'))['early_stop'])")
    echo "== round $r already decided (early_stop=$stop)"
    [ "$stop" = True ] && break
    continue
  fi
  BEST_EVAL=$(best eval)
  BEST_CKPT=$(best ckpt)
  echo "== round $r: improving on $BEST_EVAL ($BEST_CKPT)"

  step "${S}_diagnose" bash scripts/sft/diagnose.sh "$S" "$BEST_EVAL"
  if [ "$TEACHER_H" != 0 ]; then
    step "${S}_teacher" bash scripts/sft/teacher_round.sh "$S" "$BEST_CKPT" "$TEACHER_H" auto
  fi

  LRS=$LR
  if [ "$r" = "$ROUNDS" ] && [ "${SWEEP:-1}" = 1 ]; then
    LRS="$LR $("$TTS_PY" -c "print('%g' % ($LR / 2))")"
  fi
  # arms "name|mix arm|lr": the main mix at each lr, then the alt (re-weighted) mix at the main lr.
  # The main mix is built first: the alt mix is sized from its mix_summary.json.
  specs=()
  for lr in $LRS; do specs+=("lr$lr|mix|$lr"); done
  [ "${REWEIGHT:-1}" = 1 ] && specs+=("alt_lr$LR|alt|$LR")
  mkdir -p "$RUNS/$S"
  evals=()
  ckpts=()
  for spec in "${specs[@]}"; do
    IFS='|' read -r ARM MIX_ARM lr <<< "$spec"
    MIX=$SCR/mixes/${S}_$MIX_ARM.jsonl
    if [ ! -s "$RUNS/ckpt/$S/$ARM/checkpoint-last/pytorch_model.bin" ]; then
      # a (re)trained model invalidates any earlier evaluation of its epochs
      rm -f "$RUNS/.done/${S}_train_$ARM" "$RUNS/.done/${S}_eval_${ARM}_e"*
    fi
    if [ ! -s "$RUNS/ckpt/$S/$ARM/checkpoint-last/pytorch_model.bin" ] && [ ! -s "$MIX" ]; then
      CFG=$RUNS/$S/${MIX_ARM}_config.json
      [ "$MIX_ARM" = mix ] && variant=main || variant=alt
      "$TTS_PY" sft/reweight.py config --stage "$S" --variant "$variant" > "$CFG"
      if [ ! -s "$CFG" ]; then
        echo "== round $r: no re-weighting arm this round (runs/sft/$S/alt_skip.json)"
        continue
      fi
      "$MOSS_PY" sft/make_mix.py --stage "$S" --arm "$MIX_ARM" --config "@$CFG" | tail -1
    fi
    step "${S}_train_$ARM" bash scripts/sft/train.sh "$S" "$ARM" "$MIX" "$lr" "$EPOCHS" 1
    for e in $(seq 1 "$EPOCHS"); do
      CK=$RUNS/ckpt/$S/$ARM/checkpoint-epoch-$e
      NAME=${S}_${ARM}_e$e
      step "${S}_eval_${ARM}_e$e" bash scripts/sft/eval_ckpt.sh "$NAME" "$CK" "$EVAL_MAN" "$TOK" "$BASE_EVAL"
      evals+=("$NAME")
      ckpts+=("$CK")
    done
  done

  verdict=$("$TTS_PY" sft/round_select.py --stage "$S" --round "$r" --evals "${evals[@]}" --ckpts "${ckpts[@]}" | tail -1)

  # keep this round's best and the overall best; drop the other checkpoints (0.4 GB each)
  keep_round=$(realpath -m "$("$TTS_PY" -c "import json; print(json.load(open('$RUNS/$S/round_summary.json'))['best_ckpt'] or '')")")
  keep_best=$(realpath -m "$(best ckpt)")
  for ck in "$RUNS/ckpt/$S"/*/checkpoint-*; do
    c=$(realpath -m "$ck")
    [ "$c" = "$keep_round" ] || [ "$c" = "$keep_best" ] || rm -rf "$ck"
  done
  [ "$verdict" = stop ] && { echo "== round $r: not better than the previous best, stopping early"; break; }
done

echo "main.sh: best checkpoint $(best ckpt) ($(best eval), dev-mirror BTRM $(best btrm_pct)%)"
[ "${SKIP_FINAL:-0}" = 1 ] || bash scripts/sft/final.sh
