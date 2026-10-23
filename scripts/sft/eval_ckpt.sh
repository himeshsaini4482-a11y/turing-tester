#!/usr/bin/env bash
# Dev-set evaluation of one model: MOSS generation -> SpeechJudge-BTRM -> WER / speaker sim / UTMOS
# -> eval_summary.json (BTRM % with CI, vs base, guard rails). Incremental: wavs and BTRM scores
# that already exist are kept, so a base eval can later be extended with new rows.
#
#   bash scripts/sft/eval_ckpt.sh NAME MODEL MANIFEST TOKENS [BASE_NAME]
#     NAME       -> runs/sft/eval/NAME/
#     MODEL      checkpoint dir, or "$MOSS_MODEL_DIR" for the base model
#     MANIFEST   runs/sft/dev/eval_mirror.csv or eval_full.csv
#     TOKENS     MOSS voice-clone chunk limit (50 chunked, 100 whole; see `dec_tokens`)
#     BASE_NAME  eval to compare against (omit for base evals)
source "$(dirname "$0")/env.sh"
NAME=$1
MODEL=$2
MANIFEST=$3
TOKENS=$4
BASE=${5:-}
DIR=$RUNS/eval/$NAME
mkdir -p "$DIR"

# Results are reused per sentence_id, so they must belong to this exact model and chunk limit.
# A retrained checkpoint at the same path (new weights) or another TOKENS resets the dir. Content
# hash, not mtime: a base model re-downloaded after a worker restart is still the same model.
fingerprint="$(md5sum < "$MODEL/pytorch_model.bin" | cut -c1-16) tokens=$TOKENS"
if [ -f "$DIR/.model" ] && [ "$(cat "$DIR/.model")" != "$fingerprint" ]; then
  echo "== eval $NAME: model or settings changed since the last eval; discarding its old results"
  rm -rf "$DIR/tts" "$DIR"/*.jsonl "$DIR"/*.json "$DIR/manifest.csv"
fi
echo "$fingerprint" > "$DIR/.model"

echo "== eval $NAME: generating ($MODEL, chunk limit $TOKENS)"
"$MOSS_PY" benchmark/generate_moss_nano.py --manifest "$MANIFEST" --run-dir "$DIR" --repo-dir "$MOSS_REPO" \
  --model "$MODEL" --voice-clone-max-text-tokens "$TOKENS"

echo "== eval $NAME: SpeechJudge-BTRM"
"$JUDGE_PY" benchmark/btrm_score.py --manifest "$DIR/manifest.csv" --out-dir "$DIR" \
  --base-path "$BTRM_DIR/Qwen2.5-Omni-7B" --rm-path "$BTRM_DIR/SpeechJudge-BTRM"

echo "== eval $NAME: WER / speaker similarity / UTMOS"
"$TTS_PY" benchmark/metrics.py --manifest "$DIR/manifest.csv" --out-dir "$DIR"

if [ -n "$BASE" ] && [ "$BASE" != "$NAME" ]; then
  "$TTS_PY" sft/compare.py summarize --eval "$DIR" --base "$RUNS/eval/$BASE"
else
  "$TTS_PY" sft/compare.py summarize --eval "$DIR"
fi
