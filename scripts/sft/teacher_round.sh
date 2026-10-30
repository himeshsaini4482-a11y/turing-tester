#!/usr/bin/env bash
# Round stage R-3/R-4: weakness-targeted texts -> teacher audio -> filter -> MOSS codes.
#
#   bash scripts/sft/teacher_round.sh STAGE STUDENT_MODEL GPU_HOURS N_TEXTS
#
#   STAGE          pilot | main_r<k>; reads $RUNS/STAGE/diag/weakness_report.json (diagnose.sh)
#   STUDENT_MODEL  current student: a checkpoint dir or $MOSS_MODEL_DIR (base)
#   GPU_HOURS      time box for teacher generation
#   N_TEXTS        number of texts, or "auto": budget mode from the most recent
#                  $RUNS/*/teacher/filter_summary.json (fallback when there is none:
#                  TEACHER_FALLBACK_N, default 1500 texts per GPU-hour)
#
# Steps (each marked done in $RUNS/.done/${STAGE}_*, so a re-run resumes):
#   texts -> teacher (Orpheus, time-boxed) -> prescore (teacher WER + voice match, cheap)
#   -> prefilter (eos, rate, WER, sim) -> student gen (MOSS, candidates only) -> BTRM teacher vs
#   student (candidates only) -> filter -> encode.
# The cheap rules run first so the student generation and the BTRM scoring, the expensive part,
# only see clips that can still be accepted.
# Everything durable lives in $RUNS/STAGE/ (teacher wavs in $RUNS/STAGE/teacher/wav/).
# Env: TEACHER_SIM_MIN overrides the 0.88 voice-match threshold (recorded in the summaries).
source "$(dirname "$0")/env.sh"

if [ $# -ne 4 ]; then
  echo "usage: $0 STAGE STUDENT_MODEL GPU_HOURS N_TEXTS|auto" >&2; exit 2
fi
STAGE=$1
STUDENT_MODEL=$2
GPU_HOURS=$3
N_TEXTS=$4
D=$RUNS/$STAGE
T=$D/teacher
REPORT=$D/diag/weakness_report.json

[ -f "$REPORT" ] || { echo "missing $REPORT (run scripts/sft/diagnose.sh $STAGE <eval> first)" >&2; exit 1; }
mkdir -p "$T"

# The voice prompts are raw EARS wavs on the NVMe, which is wiped if the worker stops. Re-fetch
# them (ears_inventory.py is resumable per speaker) before any step that reads them.
missing_refs=$("$TTS_PY" - <<'EOF'
import json, os
rows = [json.loads(l) for l in open(os.path.join(os.environ.get("SFT_RUNS", "runs/sft"), "data", "ears_inventory.jsonl"), encoding="utf-8") if l.strip()]
print(sum(1 for r in rows if r["name"] == "rainbow_01_regular" and not os.path.exists(r["path"])))
EOF
)
if [ "$missing_refs" != 0 ]; then
  echo "$missing_refs voice-prompt wavs are missing on this worker (NVMe wiped?); re-fetching EARS"
  "$TTS_PY" sft/ears_inventory.py
fi

# 1. targeted texts from the 9B
round_texts() {
  if [ "$N_TEXTS" = auto ]; then
    local fallback summary
    fallback=${TEACHER_FALLBACK_N:-$("$TTS_PY" -c "import math; print(math.ceil(1500 * $GPU_HOURS))")}
    summary=$(ls -t $RUNS/*/teacher/filter_summary.json 2>/dev/null | head -n 1 || true)
    if [ -n "$summary" ]; then
      echo "budget mode from $summary (fallback --n $fallback)"
      "$TTS_PY" sft/gen_texts.py --report "$REPORT" --out "$D/texts.jsonl" --stage "$STAGE" \
        --budget-gpu-hours "$GPU_HOURS" --throughput "$summary" --n "$fallback"
    else
      echo "no earlier filter_summary.json; using --n $fallback"
      "$TTS_PY" sft/gen_texts.py --report "$REPORT" --out "$D/texts.jsonl" --stage "$STAGE" --n "$fallback"
    fi
  else
    "$TTS_PY" sft/gen_texts.py --report "$REPORT" --out "$D/texts.jsonl" --stage "$STAGE" --n "$N_TEXTS"
  fi
}

# 2. teacher audio (time-boxed batched Orpheus)
round_teacher() {
  "$TTS_PY" sft/teacher.py --texts "$D/texts.jsonl" --out-dir "$T" --gpu-hours "$GPU_HOURS"
}

# manifest for a scoring run: $1 = source jsonl (generated / candidates), $2 = out csv,
# $3 = tts side, $4 = human side (each "teacher" | "ref" | "none")
score_manifest() {
  "$TTS_PY" - "$1" "$2" "$3" "$4" <<'EOF'
import sys
sys.path.insert(0, "sft")
import sftlib
src, dst, tts, human = sys.argv[1:5]
pick = {"teacher": lambda g: g["wav"], "ref": lambda g: g["ref_wav"], "none": lambda g: ""}
rows = [{"speaker_id": g["speaker"], "sentence_id": g["id"], "text": g["text"],
         "tts_wav": pick[tts](g), "human_wav": pick[human](g),
         "prompt_wav": g["ref_wav"], "prompt_text": g["ref_text"]} for g in sftlib.read_jsonl(src)]
if not rows:
    sys.exit(f"no clips in {src}")
sftlib.write_csv(dst, rows)
print(f"{len(rows)} rows -> {dst}")
EOF
}

# 3. cheap teacher-side scores: Whisper WER of the teacher clip (tts_wer) and WavLM voice match
#    against the reference reading (speaker_sim). --tts-only: no Whisper pass over the refs.
round_prescore() {
  score_manifest "$T/generated.jsonl" "$T/pre/manifest_in.csv" teacher ref
  "$TTS_PY" benchmark/metrics.py --manifest "$T/pre/manifest_in.csv" --out-dir "$T/pre" --tts-only --skip-utmos
}

# 4. cheap rules -> candidates.jsonl (the only clips the student and BTRM see)
round_prefilter() {
  "$TTS_PY" sft/filter_teacher.py --teacher-dir "$T" --prefilter
}

have_candidates() { [ -s "$T/candidates.jsonl" ]; }

# 5. the current student on the candidates' texts and voices (human_wav = teacher wav, so the
#    BTRM run below scores teacher as "human" and student as "tts")
round_student() {
  if ! have_candidates; then
    echo "[warn] no candidate passed the cheap rules (see $T/prefilter_summary.json); skipping the student"
    return 0
  fi
  score_manifest "$T/candidates.jsonl" "$T/student/manifest_in.csv" none teacher
  "$MOSS_PY" benchmark/generate_moss_nano.py --manifest "$T/student/manifest_in.csv" --run-dir "$T/student" \
    --repo-dir "$MOSS_REPO" --model "$STUDENT_MODEL" --voice-clone-max-text-tokens "$(dec_tokens)"
}

# 6. BTRM on the candidates: human_reward = teacher, tts_reward = student (weights on NVMe, as run_btrm.sh)
round_btrm() {
  if ! have_candidates; then
    echo "[warn] no candidates; skipping BTRM"
    return 0
  fi
  if [ -d /opt/dlami/nvme ]; then
    sudo mkdir -p "$BTRM_DIR" && sudo chown "$(id -u):$(id -g)" "$BTRM_DIR"
  fi
  "$JUDGE_PY" benchmark/btrm_score.py --manifest "$T/student/manifest.csv" --out-dir "$T/student" \
    --base-path "$BTRM_DIR/Qwen2.5-Omni-7B" --rm-path "$BTRM_DIR/SpeechJudge-BTRM"
}

# 7. acceptance filter (all rules; non-candidates fail at their cheap rule)
round_filter() {
  "$TTS_PY" sft/filter_teacher.py --teacher-dir "$T"
}

# 8. MOSS codes for training
round_encode() {
  if [ ! -s "$T/accepted_raw.jsonl" ]; then
    # zero yield is a result, not a crash: the mix simply has no teacher data from this round
    echo "[warn] no teacher clip passed the filter (see $T/filter_summary.json); nothing to encode"
    return 0
  fi
  bash scripts/sft/encode.sh "$T/accepted_raw.jsonl" "$T/codes"
}

step "${STAGE}_texts" round_texts
step "${STAGE}_teacher_gen" round_teacher   # not ${STAGE}_teacher: main.sh wraps this script in that marker
step "${STAGE}_teacher_prescore" round_prescore
step "${STAGE}_teacher_prefilter" round_prefilter
step "${STAGE}_teacher_student" round_student
step "${STAGE}_teacher_btrm" round_btrm
step "${STAGE}_teacher_filter" round_filter
step "${STAGE}_teacher_encode" round_encode
