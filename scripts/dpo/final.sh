#!/usr/bin/env bash
# Final benchmark of the best DPO round (runs/dpo/best.json) on the locked 494-sentence EARS set,
# and of pi_ref with the same generator, for the report. GRM unless eval.final_skip_grm.
# The benchmark is used only here, never to choose anything.
source "$(dirname "$0")/env.sh"
BEST=$("$TTS_PY" -c "import json; print(json.load(open('runs/dpo/best.json'))['policy'])")
REF=$(hp "models.$MODEL.policy_init")
SKIP=$([ "$(hp eval.final_skip_grm)" = True ] && echo 1 || echo 0)
MAN=benchmark/manifests/ears_regular_26x19.csv
# pi_ref = the SFT deliverable, already benchmarked on the 494 by SFT final.sh with this generator
# (runs/sft_final); re-running it costs ~4.5 h for the same numbers. FINAL_REDO_REF=1 forces it.
REF_DONE=0
[ "$MODEL" = moss_nano ] && [ "$REF" = runs/sft/best/final ] && [ -s runs/sft_final/btrm_summary.json ] && \
  [ "${FINAL_REDO_REF:-0}" != 1 ] && REF_DONE=1
if [ "$BEST" = "$REF" ] && [ "$REF_DONE" = 1 ]; then
  echo "final.sh: no DPO round beat pi_ref on dev; pi_ref's 494 benchmark is runs/sft_final, nothing to run"
  exit 0
fi
[ "$BEST" = "$REF" ] && echo "final.sh: no DPO round beat pi_ref on dev; benchmarking pi_ref only"
for pair in "dpo_final_$MODEL:$BEST" "dpo_final_${MODEL}_ref:$REF"; do
  RUN=${pair%%:*}; CK=${pair#*:}
  [ "$RUN" != "dpo_final_$MODEL" ] && [ "$BEST" = "$REF" ] && continue
  [ "$RUN" != "dpo_final_$MODEL" ] && [ "$REF_DONE" = 1 ] && { echo "final.sh: pi_ref = runs/sft_final (SFT benchmark, same generator)"; continue; }
  case "$MODEL" in
    moss_nano)
      MOSS_MODEL="$(cd "$REPO_DIR" && realpath -m "$CK")" RUN_NAME=$RUN MOSS_MAX_TEXT_TOKENS="$(dec_tokens)" \
        SKIP_GRM=$SKIP bash scripts/run_moss_nano.sh ;;
    orpheus)
      D=runs/$RUN
      CORPUS=ears MANIFEST=$MAN LIMIT=0 source scripts/corpus.sh   # the human audio for the 494
      "$TTS_PY" benchmark/generate_orpheus.py --manifest "$MAN" --run-dir "$D" --model "$CK"
      if [ "$SKIP" != 1 ]; then
        "$JUDGE_PY" benchmark/judge.py --manifest "$D/manifest.csv" --out-dir "$D" --votes 1 \
          --model-path "$(bash scripts/stage_judge.sh)"
      fi
      "$TTS_PY" benchmark/metrics.py --manifest "$D/manifest.csv" --out-dir "$D"
      "$JUDGE_PY" benchmark/btrm_score.py --manifest "$D/manifest.csv" --out-dir "$D" \
        --base-path "$BTRM_DIR/Qwen2.5-Omni-7B" --rm-path "$BTRM_DIR/SpeechJudge-BTRM" ;;
  esac
done
