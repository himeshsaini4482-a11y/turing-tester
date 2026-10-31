#!/usr/bin/env bash
# Final benchmark: the full locked 494-sentence EARS set, base vs the chosen checkpoint, with the
# decoding setting chosen in the pilot. Each run: MOSS generation, SpeechJudge-GRM (988 calls),
# WER / speaker sim / UTMOS, SpeechJudge-BTRM. Then the final report ($RUNS/REPORT.md).
#
#   KEEP_UP=1 bash tools/aws/gpu_run.sh "bash scripts/sft/final.sh"
#   CKPT=$RUNS/ckpt/... bash scripts/sft/final.sh   # a specific checkpoint instead of best.json
#   RUNNER_UP=$RUNS/ckpt/... ...                    # also benchmark a second checkpoint
#   FINAL_SKIP_GRM=1 ...                               # budget lever: skip the SpeechJudge-GRM pass
#                                                      # (BTRM + WER/sim/UTMOS still run)
#
# The benchmark is used only here, never to choose anything.
source "$(dirname "$0")/env.sh"
# MOSS venv/repo/model live on the NVMe (wiped if the worker stopped); idempotent
bash scripts/sft/setup_moss.sh
TOK=$(dec_tokens)
CKPT=${CKPT:-$("$TTS_PY" -c "import json; print(json.load(open('$RUNS/main/best.json'))['ckpt'])" 2>/dev/null || \
  "$TTS_PY" -c "import json; print(json.load(open('$RUNS/pilot/recipe.json'))['pilot_best']['ckpt'])")}
[ -s "$CKPT/pytorch_model.bin" ] || { echo "final.sh: no checkpoint at $CKPT" >&2; exit 1; }
echo "final.sh: base vs $CKPT on the 494, chunk limit $TOK"

# deliverable copy of the chosen checkpoint (copied back to the UAI server with runs/). If the
# chosen checkpoint changed since an earlier final.sh (e.g. main.sh found a better one later),
# re-copy and redo its benchmark from scratch: the old wavs/scores belong to another model.
if [ ! -s "$RUNS/best/final/pytorch_model.bin" ] || [ "$(cat "$RUNS/best/final/SOURCE" 2>/dev/null)" != "$CKPT" ]; then
  echo "final.sh: (re)copying $CKPT to $RUNS/best/final and resetting its benchmark"
  mkdir -p "$RUNS/best" && rm -rf "$RUNS/best/final" && cp -r "$CKPT" "$RUNS/best/final"
  echo "$CKPT" > "$RUNS/best/final/SOURCE"
  rm -rf "runs/${BENCH_PREFIX}sft_final" "$RUNS/.done/final_best"
fi

GRM=${FINAL_SKIP_GRM:-0}
[ "$GRM" = 1 ] && echo "final.sh: FINAL_SKIP_GRM=1: no GRM pass; the GRM closeness column will be missing from the report"
step final_base env MOSS_MODEL="$MOSS_MODEL_DIR" RUN_NAME=${BENCH_PREFIX}sft_final_base MOSS_MAX_TEXT_TOKENS="$TOK" SKIP_GRM="$GRM" \
  bash scripts/run_moss_nano.sh
step final_best env MOSS_MODEL="$RUNS/best/final" RUN_NAME=${BENCH_PREFIX}sft_final MOSS_MAX_TEXT_TOKENS="$TOK" SKIP_GRM="$GRM" \
  bash scripts/run_moss_nano.sh
if [ -n "${RUNNER_UP:-}" ]; then
  if [ "$(cat runs/${BENCH_PREFIX}sft_final_runner_up/.model 2>/dev/null)" != "$RUNNER_UP" ]; then
    rm -rf runs/${BENCH_PREFIX}sft_final_runner_up "$RUNS/.done/final_runner_up"
    mkdir -p runs/${BENCH_PREFIX}sft_final_runner_up && echo "$RUNNER_UP" > runs/${BENCH_PREFIX}sft_final_runner_up/.model
  fi
  step final_runner_up env MOSS_MODEL="$RUNNER_UP" RUN_NAME=${BENCH_PREFIX}sft_final_runner_up MOSS_MAX_TEXT_TOKENS="$TOK" SKIP_GRM="$GRM" \
    bash scripts/run_moss_nano.sh
fi
"$TTS_PY" sft/report.py
echo "final.sh: report at $RUNS/REPORT.md"
