#!/usr/bin/env bash
# Full EARS set (26 speakers x 19 sentences = 494) for Orpheus 3B and MOSS-TTS-Nano,
# voice-cloned per speaker, scored with SpeechJudge-BTRM only (ratio score + parity score).
#
#   KEEP_UP=1 bash tools/aws/gpu_run.sh "bash scripts/run_btrm_full.sh"
#
# Orpheus (~30 s/clip) is the long pole, so MOSS generation and its BTRM scoring run
# alongside it on the same GPU (~8 + ~2 + ~22 GB of 48 GB). Every step is resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH   # see run_btrm.sh
# model downloads on the AWS worker's fast local NVMe (elsewhere: the default Hugging Face cache)
[ -d /opt/dlami/nvme ] && export HF_HUB_CACHE=/opt/dlami/nvme/hf

CORPUS=ears
TTS_PY=.venv-tts/bin/python
source scripts/corpus.sh   # fetches all 26 speakers' audio once, before the jobs below start

ORPHEUS_RUN=${ORPHEUS_RUN:-orpheus_full}
MOSS_RUN=${MOSS_RUN:-moss_nano_full}
mkdir -p "runs/$ORPHEUS_RUN" "runs/$MOSS_RUN"

$TTS_PY benchmark/generate_orpheus.py --manifest "$MANIFEST" --run-dir "runs/$ORPHEUS_RUN" \
  > "runs/$ORPHEUS_RUN/generate.log" 2>&1 &
orpheus=$!

(
  RUN_NAME=$MOSS_RUN GEN_ONLY=1 bash scripts/run_moss_nano.sh
  RUN_NAME=$MOSS_RUN CORPUS=$CORPUS MANIFEST=$MANIFEST bash scripts/run_btrm.sh
) > "runs/$MOSS_RUN/job.log" 2>&1 &
moss=$!

echo "== Orpheus generating (runs/$ORPHEUS_RUN/generate.log), MOSS generating + BTRM (runs/$MOSS_RUN/job.log)"
wait $moss && echo "== MOSS-TTS-Nano done" || { echo "MOSS-TTS-Nano failed:" >&2; tail -n 30 "runs/$MOSS_RUN/job.log" >&2; }
wait $orpheus || { echo "Orpheus generation failed:" >&2; tail -n 30 "runs/$ORPHEUS_RUN/generate.log" >&2; exit 1; }
echo "== Orpheus generated; BTRM"
RUN_NAME=$ORPHEUS_RUN CORPUS=$CORPUS MANIFEST=$MANIFEST bash scripts/run_btrm.sh

for run in $MOSS_RUN $ORPHEUS_RUN; do
  [ -f "runs/$run/btrm_summary.json" ] || continue
  python3 -c "
import json; s = json.load(open('runs/$run/btrm_summary.json'))
print('$run: ratio score %.1f%% over %d sentences (%d with TTS reward <= 0, %d skipped), parity %.1f/100' % (
    s['ratio_score'], s['ratio_n_sentences'], s['ratio_n_tts_reward_le_0'], s['ratio_skipped_human_reward_le_0'],
    s['human_parity_score']))"
done
