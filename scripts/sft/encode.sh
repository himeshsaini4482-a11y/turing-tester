#!/usr/bin/env bash
# Encode a raw MOSS training JSONL into audio codes with MOSS's finetuning/prepare_data.py,
# in shards so a crash only loses one shard (resumable). Shards are rebuilt if the raw file changes.
#
#   bash scripts/sft/encode.sh runs/sft/data/ears_raw.jsonl runs/sft/data/ears_codes
#   SHARD_SIZE=2000 (default)
source "$(dirname "$0")/env.sh"
RAW=$1
OUT=$2
SHARD_SIZE=${SHARD_SIZE:-2000}

[ -s "$RAW" ] || { echo "encode.sh: $RAW is missing or empty" >&2; exit 1; }
mkdir -p "$OUT"
src=$(md5sum "$RAW" | cut -c1-16)
if [ -f "$OUT/.source" ] && [ "$(cat "$OUT/.source")" != "$src" ]; then
  echo "encode.sh: $RAW changed since the last encode; removing old shards"
  rm -f "$OUT"/codes.rank*.jsonl
fi
echo "$src" > "$OUT/.source"

n=$(grep -c . "$RAW")
shards=$(( (n + SHARD_SIZE - 1) / SHARD_SIZE ))
echo "encode.sh: $n records -> $shards shard(s) in $OUT"
for k in $(seq 0 $((shards - 1))); do
  out=$(printf "%s/codes.rank%05d-of-%05d.jsonl" "$OUT" "$k" "$shards")
  [ -s "$out" ] && continue
  tmp=$OUT/.tmp
  rm -rf "$tmp" && mkdir -p "$tmp"
  "$MOSS_PY" "$MOSS_REPO/finetuning/prepare_data.py" --codec-path "$MOSS_CODEC_DIR" --device cuda \
    --input-jsonl "$RAW" --output-jsonl "$tmp/codes.jsonl" --num-shards "$shards" --shard-rank "$k" \
    --save-shard-suffix --batch-size "${ENCODE_BATCH:-8}"
  mv "$tmp"/codes.rank*.jsonl "$out"   # only a finished shard lands in OUT
  echo "encode.sh: shard $((k + 1))/$shards done"
done
rm -rf "$OUT/.tmp"
