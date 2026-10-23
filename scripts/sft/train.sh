#!/usr/bin/env bash
# One SFT run of MOSS-TTS-Nano (always from the base model) with MOSS's patched finetuning/sft.py.
#
#   bash scripts/sft/train.sh STAGE ARM MIX_JSONL LR EPOCHS SAVE_EVERY
#   -> runs/sft/ckpt/STAGE/ARM/{checkpoint-epoch-K, checkpoint-last, train.log}
#
# Skipped if checkpoint-last already exists. Env: BATCH (16) x ACCUM (2) = global batch 32,
# MAX_LEN (640; records longer than this were already dropped by make_mix.py).
source "$(dirname "$0")/env.sh"
STAGE=$1
ARM=$2
MIX=$3
LR=$4
EPOCHS=$5
SAVE_EVERY=$6
OUT=$RUNS/ckpt/$STAGE/$ARM

if [ -s "$OUT/checkpoint-last/pytorch_model.bin" ]; then
  echo "train.sh: $OUT/checkpoint-last exists, skipping"
  exit 0
fi
rm -rf "$OUT" && mkdir -p "$OUT"
echo "train.sh: $STAGE/$ARM lr=$LR epochs=$EPOCHS mix=$MIX ($(grep -c . "$MIX") records)"
"$MOSS_ACCEL" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  "$MOSS_REPO/finetuning/sft.py" \
  --model-path "$MOSS_MODEL_DIR" --codec-path "$MOSS_CODEC_DIR" \
  --train-jsonl "$MIX" --output-dir "$OUT" \
  --per-device-batch-size "${BATCH:-16}" --gradient-accumulation-steps "${ACCUM:-2}" \
  --learning-rate "$LR" --warmup-ratio 0.03 --lr-scheduler-type cosine \
  --num-epochs "$EPOCHS" --save-every-epochs "$SAVE_EVERY" \
  --mixed-precision bf16 --fp32-master-weights --max-length "${MAX_LEN:-640}" \
  --channelwise-loss-weight 1,32 --seed 42 --logging-steps 20 \
  2>&1 | tee "$OUT/train.log"
[ -s "$OUT/checkpoint-last/pytorch_model.bin" ] || { echo "train.sh: no checkpoint written" >&2; exit 1; }
# sft.py also saves checkpoint-epoch-N at the final step; when only the last checkpoint was
# wanted (SAVE_EVERY > EPOCHS) that copy is a 0.4 GB duplicate that rsync would carry around
if [ "$SAVE_EVERY" -gt "$EPOCHS" ]; then
  rm -rf "$OUT"/checkpoint-epoch-*
fi
