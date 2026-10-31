#!/usr/bin/env bash
# The whole online-DPO run for the hparams model, resumable at every step (markers in runs/dpo/.done/):
#   setup -> 9B texts -> prompts -> gates -> pi_ref dev eval -> for each round:
#   rollout -> BTRM -> WER/sim/UTMOS -> pairs (+ beta* diagnostic) -> train -> dev eval -> select
#   -> final 494 benchmark -> report
#
#   KEEP_UP=1 bash tools/aws/gpu_run.sh "bash scripts/dpo/run.sh"
#
# Exit 3 = paused for the user: dpo/hparams.json USER_REVIEW (beta, kl, optimizer) is not filled
# in yet. Round 1's rollout, scores, pairs and beta_diagnostic.md are done by then; set the values
# (or run scripts/dpo/sweep.sh first), then launch the same command again.
source "$(dirname "$0")/env.sh"
echo "dpo run.sh: model $MODEL, pi_ref $(hp "models.$MODEL.policy_init")"
bash scripts/dpo/setup.sh
[ "$MODEL" = moss_nano ] && REPO_ARG=(--repo-dir "$MOSS_REPO") || REPO_ARG=()

gpu_logger_start

dstep texts "$TTS_PY" dpo/prompts.py texts --n "$(hp texts.n)" --report auto
dstep prompts "$TTS_PY" dpo/prompts.py build
dstep gates "$DPO_PY" dpo/gates.py "${REPO_ARG[@]}"
dstep eval_r00 bash scripts/dpo/eval.sh r00 "$(current_policy)"

ROUNDS=$(hp rounds)
for r in $(seq 1 "$ROUNDS"); do
  R=$(printf 'r%02d' "$r")
  RD=$DRUNS/$R
  if [ -f "$RD/round_summary.json" ]; then   # a round that stopped the run (select exit 10) stays the last one
    "$TTS_PY" -c "import json, sys; sys.exit(10 if json.load(open('$RD/round_summary.json'))['stop'] else 0)" ||
      { echo "== [skip] round $r (done; it stopped the run)"; break; }
    echo "== [skip] round $r (done)"; continue
  fi
  POLICY=$(current_policy)
  dstep "${R}_rollout" "$DPO_PY" dpo/rollout.py --round "$r" --policy "$POLICY" "${REPO_ARG[@]}"
  dstep "${R}_metrics" "$TTS_PY" benchmark/metrics.py --manifest "$RD/manifest.csv" --out-dir "$RD" --tts-only
  dstep "${R}_btrm" "$JUDGE_PY" dpo/score_btrm.py --manifest "$RD/manifest.csv" --out-dir "$RD" --round "$r" \
    --base-path "$BTRM_DIR/Qwen2.5-Omni-7B" --rm-path "$BTRM_DIR/SpeechJudge-BTRM"
  dstep "${R}_pairs" "$TTS_PY" dpo/pairs.py --round "$r"
  set +e
  "$DPO_PY" dpo/train.py --round "$r"
  rc=$?
  set -e
  if [ "$rc" = 3 ]; then
    pause_for_user paused_for_user "set beta / kl / optimizer in dpo/hparams.json (see $R/beta_diagnostic.md), or approve dpo/sweep.json and run scripts/dpo/sweep.sh"
    exit 3
  fi
  [ "$rc" = 0 ] || exit "$rc"
  dstep "${R}_eval" bash scripts/dpo/eval.sh "$R" "$(current_policy)" r00
  set +e
  "$TTS_PY" dpo/rounds.py select --round "$r"
  rc=$?
  set -e
  [ "$rc" = 10 ] && break
  [ "$rc" = 0 ] || exit "$rc"
done

dstep final bash scripts/dpo/final.sh
"$TTS_PY" dpo/rounds.py report
echo "dpo run.sh: done, report at runs/dpo/REPORT.md"
