#!/usr/bin/env bash
# Dev-set eval of one DPO policy, the same measurement the SFT uses for its checkpoints:
# the model's benchmark generator on the SFT dev mirror -> SpeechJudge-BTRM -> WER / sim / UTMOS
# -> sft/compare.py summary (BTRM % with CI, vs the round-0 = pi_ref eval, SFT guard rails).
#
#   bash scripts/dpo/eval.sh NAME CKPT [BASE_NAME]      -> runs/dpo/eval/NAME/eval_summary.json
source "$(dirname "$0")/env.sh"
NAME=$1
CKPT=$2
BASE=${3:-}
MANIFEST=$(hp eval.manifest)
DIR=$DRUNS/eval/$NAME
mkdir -p "$DIR"
[ -e "$CKPT" ] || [ "$MODEL" = orpheus ] || { echo "eval.sh: no checkpoint at $CKPT" >&2; exit 1; }

# results belong to these exact weights: a different checkpoint under the same NAME resets the dir
if [ -d "$CKPT" ]; then
  fp=$(find "$CKPT" -maxdepth 1 \( -name 'pytorch_model*.bin' -o -name '*.safetensors' \) -print0 | sort -z |
       xargs -0 cat | md5sum | cut -c1-16)
else
  fp=$CKPT    # an HF model id (Orpheus pi_ref)
fi
if [ -f "$DIR/.model" ] && [ "$(cat "$DIR/.model")" != "$fp" ]; then
  echo "== eval $NAME: different weights than last time; discarding its old results"
  rm -rf "$DIR/tts" "$DIR"/*.jsonl "$DIR"/*.json "$DIR/manifest.csv"
fi
echo "$fp" > "$DIR/.model"

echo "== eval $NAME ($MODEL): generating with $CKPT"
case "$MODEL" in
  moss_nano)
    "$MOSS_PY" benchmark/generate_moss_nano.py --manifest "$MANIFEST" --run-dir "$DIR" --repo-dir "$MOSS_REPO" \
      --model "$CKPT" --voice-clone-max-text-tokens "$(dec_tokens)" ;;
  orpheus)
    "$TTS_PY" benchmark/generate_orpheus.py --manifest "$MANIFEST" --run-dir "$DIR" --model "$CKPT" ;;
esac
echo "== eval $NAME: SpeechJudge-BTRM"
"$JUDGE_PY" benchmark/btrm_score.py --manifest "$DIR/manifest.csv" --out-dir "$DIR" \
  --base-path "$BTRM_DIR/Qwen2.5-Omni-7B" --rm-path "$BTRM_DIR/SpeechJudge-BTRM"
echo "== eval $NAME: WER / speaker similarity / UTMOS"
"$TTS_PY" benchmark/metrics.py --manifest "$DIR/manifest.csv" --out-dir "$DIR"
if [ -n "$BASE" ] && [ "$BASE" != "$NAME" ]; then
  "$TTS_PY" sft/compare.py summarize --eval "$DIR" --base "$DRUNS/eval/$BASE"
else
  "$TTS_PY" sft/compare.py summarize --eval "$DIR"
fi
