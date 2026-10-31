# SFT pipeline: interface contract

This is the binding contract between the pipeline's parts. Every script reads and writes exactly
the files, columns and keys below. The reasons behind the choices are in
`docs/sft_experiment/`. If you
need to change a contract, change it here first.

Nothing here runs on the dev machine (Windows, no GPU, no torch). It runs on the AWS GPU worker
(Ubuntu, 1× L40S 48 GB), driven by the shell scripts in `scripts/sft/`.

## 0. Conventions

- **Venvs** (all on the worker):
  - `TTS` = `.venv-tts`: transformers 4.57.3, torch, librosa, soundfile, jiwer, snac, qwen-tts.
  - `JUDGE` = `.venv-judge`: transformers 4.52.3, peft, qwen-omni-utils.
  - `MOSS` = `$MOSS_PY`: torch 2.7.0, torchaudio, transformers 4.57.1, sentencepiece, soundfile,
    numpy, accelerate.
  - `STD` = standard library only; runs in any venv.
- **Shell:** every `scripts/sft/*.sh` starts with `source "$(dirname "$0")/env.sh"`. That gives
  `REPO_DIR RUNS SCR TTS_PY JUDGE_PY MOSS_PY MOSS_ACCEL MOSS_REPO MOSS_MODEL_DIR MOSS_CODEC_DIR
  BTRM_DIR` and the functions `step NAME cmd...`, `dec_tokens`, `train_max_tokens` and
  `judge_model_path`.
- **Python stages** import shared helpers with
  `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sftlib`. Use
  `sftlib.read_jsonl/write_jsonl/append_jsonl/read_json/write_json/read_csv/write_csv`
  (writes are atomic), `sftlib.RUNS`, `sftlib.scratch_dir()`, `sftlib.log_timing`,
  `sftlib.log_decision`, `sftlib.btrm_pct`, `sftlib.bootstrap_pct`, `sftlib.bootstrap_diff`,
  `sftlib.item_id`, `sftlib.classify_ears`, `sftlib.normalize_words`,
  `sftlib.STYLE_INSTRUCTION`, `sftlib.emotion_instruction`, `sftlib.FREEFORM_TAG`,
  `sftlib.WEAKNESS_TAXONOMY`, `sftlib.WEAKNESS_TEXT_HINT` and `sftlib.matches_category`.
  Reuse benchmark code via `sftlib.add_benchmark_to_path()` (e.g. `from generate_orpheus import
  encode, decode, SOH, ...`, `from btrm_score import load_btrm`).
- **Style:** match `benchmark/*.py`. Each file opens with a docstring saying what the stage does,
  its env and its outputs. Use argparse. Every stage is **resumable**: it skips work whose output
  already exists, and progress lines are printed with `flush=True` and an ETA.
- **Paths:** `RUNS = runs/sft` is durable (worker EBS, copied back to the UAI server). `SCR =
  sftlib.scratch_dir()` is the local NVMe (wiped when the worker stops), for bulky re-creatable
  files only.
- **Text normalisation:** the MOSS inference normaliser is
  `from text_normalization_pipeline import prepare_tts_request_texts` (with `$MOSS_REPO` on
  `sys.path`), called as `prepare_tts_request_texts(text=t, enable_wetext=False,
  enable_normalize_tts_text=True, text_normalizer_manager=None)["text"]`. Only MOSS-venv code
  calls it.
- **Text tokens:** MOSS's chunker counts tokens with the MOSS tokenizer:
  `AutoTokenizer.from_pretrained($MOSS_MODEL_DIR, trust_remote_code=True)`,
  `len(tok.encode(text, add_special_tokens=False))`. MOSS venv only.

## 1. Files and owners

| Path | Producer (owner) | Format |
|---|---|---|
| `sft/data/dev_speakers.json` | committed | `{"speakers": {"p015": "female", "p016": "male", "p046": "male", "p061": "female"}, ...}` |
| `$SCR/ears/<spk>/<name>.wav` | `sft/ears_inventory.py` (EARS agent) | raw EARS 48 kHz |
| `runs/sft/data/ears_inventory.jsonl` | `ears_inventory.py` | §2.1 |
| `runs/sft/data/inventory_summary.json` | `ears_inventory.py` | counts and hours per class, per exclusion reason |
| `runs/sft/data/units.jsonl` | `sft/align.py` (EARS agent) | §2.2 |
| `$SCR/ears_chunks/<spk>/<id>.wav` + `runs/sft/data/ears_raw.jsonl` + `runs/sft/data/build_summary.json` | `sft/build_ears.py` (EARS agent) | §2.3 |
| `runs/sft/data/ears_codes/*.jsonl` | `scripts/sft/encode.sh` (core) | §2.3 records + `audio_codes`, `ref_audio_codes` |
| `runs/sft/dev/{dev_mirror.csv, dev_fresh.csv, eval_mirror.csv, eval_full.csv}`, `runs/sft/dev/wav/` | `sft/dev_sets.py` (EARS agent) | §2.4 |
| `runs/sft/decoding.json` | `sft/decide.py decoding` (pilot agent) | §2.8 |
| `runs/sft/eval/<name>/{manifest.csv, tts/, btrm.jsonl, btrm_summary.json, metrics.jsonl, metrics_summary.json}` | `scripts/sft/eval_ckpt.sh` (core) | existing benchmark formats |
| `runs/sft/eval/<name>/eval_summary.json` | `sft/compare.py summarize` (pilot agent) | §2.6 |
| `runs/sft/<stage>/diag/{manifest.csv, judgments.jsonl, summary.json}` | `scripts/sft/diagnose.sh` (teacher agent) via `benchmark/judge.py` | existing judge formats |
| `runs/sft/<stage>/diag/{tags.jsonl, weakness_report.json, weakness_report.md}` | `sft/tag_weaknesses.py` (teacher agent) | §2.5 |
| `runs/sft/<stage>/texts.jsonl` | `sft/gen_texts.py` (teacher agent) | §2.7 |
| `runs/sft/<stage>/teacher/...` | `teacher.py`, `filter_teacher.py`, `teacher_round.sh` (teacher agent) | §2.7 |
| `$SCR/mixes/<stage>_<arm>.jsonl` + `runs/sft/<stage>/<arm>/mix_summary.json` | `sft/make_mix.py` (core) | MOSS training records |
| `runs/sft/ckpt/<stage>/<arm>/checkpoint-*` | `scripts/sft/train.sh` (core) | MOSS checkpoints |
| `runs/sft/pilot/{recipe.json, pilot_report.md}`, `runs/sft/DECISIONS.md` | `sft/decide.py pilot` (pilot agent) | §2.8 |
| `runs/sft/REPORT.md` | `sft/report.py` (pilot agent) | final report |
| `runs/sft/main/mix_state.json`, `runs/sft/main_r<k>/{mix_config.json, alt_config.json, alt_skip.json}` | `sft/reweight.py` (core) | §4.1 |
| `runs/sft/main_r<k>/round_summary.json`, `runs/sft/main/best.json` | `sft/round_select.py` (core) | round result incl. `reweight` (§4.1) |

`<stage>` is `pilot` or `main_r<k>` (k = 1, 2, ...).

## 2. Record formats

### 2.1 `ears_inventory.jsonl`, one line per EARS wav (all 107 speakers)
```json
{"speaker": "p003", "gender": "female", "name": "rainbow_02_fast", "path": "/opt/dlami/nvme/sft/ears/p003/rainbow_02_fast.wav",
 "class": "read-styled", "style": "fast", "emotion": null, "transcript": "These take the shape ...",
 "duration": 9.8, "is_bench_text": true, "bench_file": false, "f1_styled_bench": false,
 "dev_speaker": false, "dev_leak_probe": false, "excluded": null}
```
- `class`, `style`, `emotion` come from `sftlib.classify_ears(name)`.
- `transcript` is from EARS `transcripts.json`, or null for freeform/interjection.
- `is_bench_text`: the transcript is one of the 19 benchmark texts.
- `bench_file`: (speaker, name) is one of the 494 benchmark rows.
- `f1_styled_bench`: a benchmark speaker, `is_bench_text`, style ≠ regular. These are the 1,404 files.
- `dev_speaker`: speaker in `sft/data/dev_speakers.json`.
- `dev_leak_probe`: `dev_speaker` and `is_bench_text` and style ≠ regular. This is pilot arm F's
  simulation of F1 on the dev speakers.
- `excluded`: `"benchmark_file"` | `"nonverbal"` | `"other"` | null. Dev speakers are **not**
  marked excluded here; downstream code filters them with `dev_speaker`.

### 2.2 `units.jsonl`, one line per processed wav (all non-excluded files, dev speakers included)
```json
{"speaker": "p003", "name": "freeform_speech_01", "path": "...", "status": "ok", "align_wer": null,
 "units": [{"text": "So I grew up in Ohio.", "start": 1.02, "end": 2.71, "cut_start": 0.90, "cut_end": 2.86,
            "left_ok": true, "right_ok": true, "bad": false}]}
```
- One unit per sentence.
  - For read classes: sentence text from the transcript, times from Whisper word alignment;
    `align_wer` = WER(transcript, ASR).
  - For freeform/interjection: Whisper's own text.
- `status`: `"ok"` | `"misread"` (align_wer > 0.10) | `"align_fail"` | `"asr_fail"`.
- `cut_start` / `cut_end` sit in the silence around the unit.
- `left_ok` / `right_ok`: true when the cut point is in a low-energy gap of at least 0.2 s (or the
  file edge).
- `bad`: the unit contains untranscribed speech (a word gap longer than 0.35 s that has speech
  energy, e.g. a dropped filler).

### 2.3 `ears_raw.jsonl`, one line per training chunk (MOSS raw format + metadata)
```json
{"id": "p003__freeform_speech_01__004", "audio": "/opt/dlami/nvme/sft/ears_chunks/p003/p003__freeform_speech_01__004.wav",
 "text": "<normalized>", "ref_audio": "/opt/dlami/nvme/sft/ears/p003/rainbow_01_regular.wav",
 "speaker": "p003", "gender": "female", "src": "freeform_speech_01", "class": "freeform", "style": null,
 "emotion": null, "duration": 6.4, "n_text_tokens": 31, "is_bench_text": false, "f1_styled_bench": false,
 "dev_speaker": false, "dev_leak_probe": false}
```
- A chunk is consecutive good units of one file, merged greedily while the normalised text is
  ≤ `--max-tokens` MOSS tokens (as in MOSS's `_split_text_into_best_sentences`). The first unit
  must have `left_ok` and the last must have `right_ok`. Duration is 1.5–20 s.
- The audio is cut at `[cut_start(first), cut_end(last)]`, edge silence trimmed to ≤ 0.15 s, and
  written as 48 kHz mono 16-bit wav.
- **No `instruction`, no `language` keys.** `make_mix.py` sets `instruction` per arm from
  class/style/emotion.
- `ref_audio`: the speaker's `rainbow_01_regular` raw wav in 70% of records. Otherwise, and
  always when the chunk's own source is `rainbow_01_regular`, use another chunk (same speaker,
  class read-regular or freeform, 4–10 s, different source file). The choice is deterministic
  per id.
- Dev speakers' chunks **are** included, flagged `dev_speaker`, because dev-fresh is drawn from them.

### 2.4 Dev and eval manifests (benchmark manifest columns)
Columns: `speaker_id, gender, chapter_id, sentence_id, text, duration, human_wav, prompt_id,
prompt_wav, prompt_text` plus `set` and `item_id`.
- `dev_mirror.csv`: the 4 dev speakers × the 19 benchmark reading names (`*_regular`),
  prompt = their `rainbow_01_regular`. 76 rows, `set=mirror`.
- `dev_fresh.csv`: 10 chunks per dev speaker from `ears_raw.jsonl` (class freeform, 4–12 s,
  seed 4321). 40 rows, `set=fresh`. The text is the chunk text.
- All `human_wav` / `prompt_wav` of dev rows are **copied** into `runs/sft/dev/wav/`, so they
  are durable.
- `eval_mirror.csv` / `eval_full.csv`: mirror (or mirror + fresh) rows repeated 3×, with
  `sentence_id = f"{item_id}__s{k}"` for k = 0, 1, 2. The seed suffix changes the generation seed,
  because `generate_moss_nano.py` seeds from crc32(sentence_id).

### 2.5 `weakness_report.json`
```json
{"stage": "pilot", "eval": "base_chunked", "n_judgments": 152, "n_human_preferred": 131,
 "categories": {"unnatural_pauses": {"count": 58, "share": 0.44, "n_items": 33, "confirmed": true,
                                     "evidence": ["the pause before 'and' sounds unnatural", "..."]}},
 "confirmed": ["unnatural_pauses", "flat_prosody"],
 "checkpoint_metrics": {"btrm_pct": 36.1, "wer": 0.006, "sim": 0.96}}
```
- `share` = count / n_human_preferred.
- `confirmed` means share ≥ 0.15 and n_items ≥ 8, ordered by share.
- Category keys are exactly `sftlib.WEAKNESS_TAXONOMY`.

### 2.6 `eval_summary.json`
```json
{"name": "pilot_A", "model": "<ckpt path>",
 "sets": {"mirror": {"btrm_pct": 41.2, "lo": 39.0, "hi": 43.5, "n_items": 76, "n_rows": 228,
                     "wer": 0.008, "sim": 0.955, "utmos": 3.71},
          "fresh": {...}},
 "vs_base": {"base": "base_chunked", "mirror": {"diff": 5.1, "lo": 2.9, "hi": 7.2, "n_items": 76}, "fresh": {...}},
 "guards": {"wer_ok": true, "sim_ok": true, "utmos_ok": true, "pass": true}}
```
- `btrm_pct` is `sftlib.btrm_pct` (mean r_tts / mean r_human × 100).
- `wer` / `sim` / `utmos` are means of `tts_wer` / `speaker_sim` / `tts_utmos`.
- The guards compare mirror + fresh rows against the base eval:
  - `wer` ≤ base + 0.01;
  - `sim` ≥ base − 0.02;
  - `utmos` ≥ base − 0.10.

### 2.7 Teacher round (`runs/sft/<stage>/`)
- `texts.jsonl`: `{"id": "pilot_t00042", "text": "...", "category": "unnatural_pauses" | "general", "stage": "pilot"}`
- `teacher/generated.jsonl`: `{"id", "text", "category", "speaker", "ref_wav", "ref_text", "wav", "duration", "hit_eos", "gen_seconds"}`
  - `wav` is in `teacher/wav/<id>.wav`, 24 kHz.
  - `ref_wav` is the speaker's `rainbow_01_regular` raw wav.
  - Speakers are all non-dev EARS speakers, assigned round-robin in sorted order.
- `teacher/pre/`: `metrics.py --tts-only --skip-utmos` on a manifest with `tts_wav` = teacher wav
  and `human_wav` = ref wav (one Whisper pass over the teacher clips, no human-side scores):
  - `tts_wer` = teacher WER;
  - `speaker_sim` = teacher-vs-reference voice match.
- `teacher/candidates.jsonl` + `teacher/prefilter_summary.json` (`filter_teacher.py --prefilter`):
  the `generated.jsonl` rows that pass the four cheap rules below (`no_eos`, `rate`, `wer`,
  `sim`). Only these go through the student and BTRM, which is where the scoring time goes.
- `teacher/student/`: a `generate_moss_nano.py` run dir for the *candidates'* texts and voices,
  generated with the *current student*. Its `manifest.csv` has `human_wav` = teacher wav and
  `tts_wav` = student wav; `btrm_score.py` is run on it, so:
  - `human_reward` = teacher reward;
  - `tts_reward` = student reward.
- `teacher/accepted_raw.jsonl` (MOSS raw):
  `{"id", "audio": <teacher wav>, "text": <raw text>, "ref_audio": <ref wav>, "speaker", "category", "source": "teacher", "stage", "duration", "r_teacher", "r_student", "teacher_wer", "sim"}`.
  Accept only if all hold:
  - `hit_eos`;
  - 8 ≤ chars/s ≤ 22;
  - `teacher_wer` ≤ 0.05;
  - `sim` ≥ 0.88 (env `TEACHER_SIM_MIN` overrides it explicitly; the override is recorded under
    `thresholds.sim_override` in both summaries);
  - `r_teacher > r_student`.
  Rejections are counted under the first failing rule in that order, so a clip that never
  reached BTRM is counted under its cheap rule.
- `teacher/filter_summary.json`:
  `{"n_generated", "n_candidates", "n_accepted", "yield", "rejected": {"no_eos": n, "rate": n, "wer": n, "sim": n, "btrm": n}, "accepted_audio_s", "gen_gpu_s", "accepted_audio_per_gpu_s", "mean_accepted_duration", "thresholds"}`
- `teacher/codes/*.jsonl`: `accepted_raw.jsonl` after `bash scripts/sft/encode.sh teacher/accepted_raw.jsonl teacher/codes`.

### 2.8 Decisions
- `runs/sft/decoding.json`:
  `{"mode": "chunked" | "whole", "voice_clone_max_text_tokens": 50 | 100, "train_max_text_tokens": 50 | 75, "evidence": {...}}`
- `runs/sft/pilot/recipe.json`:
```json
{"status": "go" | "stop", "decoding": "chunked", "freeform_tag": "U" | "T", "f1_include": false,
 "recipe": "H" | "M" | "T", "ears_in_mix": true, "teacher_gpu_hours_per_round": 4, "teacher_oversample": 1.0,
 "lr": 2e-5, "main_rounds": 2, "pilot_best": {"arm": "CU", "eval": "pilot_CU", "ckpt": "runs/sft/ckpt/pilot/CU/checkpoint-last", "btrm_pct": 44.0},
 "evidence": {"delta1": {...}, "delta2": {...}, "tag": {...}, "f1": {...}}}
```

### 2.9 Training-mix config (`make_mix.py --config '<json>'`)
```json
{"ears": {"use": true, "freeform_tag": "U", "f1_include": false, "leak_probe": false,
          "match_teacher": null, "category_weights": {"unnatural_pauses": 2.0}, "weight": 1.0},
 "teacher": {"codes": ["runs/sft/pilot/teacher/codes"], "oversample": 3.0, "max_oversample": 3.0},
 "teacher_share": null, "total_records": null,
 "seed": 0}
```
- `ears.weight`: a global EARS repeat factor, multiplied with the category weight
  (`weight × max(category weights)`). Below 1 it is a seeded subsample.
- Share mode (`teacher_share` set, used by the main rounds' re-weighting, §4.1):
  - **Base counts:** one extra pass sums each source's base hours and record count: EARS at
    weight 1 with category weights applied, teacher at oversample 1.
  - **Fixed record count:** batches are a fixed number of records, so equal hours would give
    the teacher-heavier mix (shorter clips) more optimizer steps. The count is `total_records`,
    or by default `E_n × ears.weight + T_n × oversample`, where E_n and T_n are the base record
    counts. `ears.weight: 0` therefore sizes the mix by the teacher alone.
  - **Weights:** `ears.weight` and `oversample` are solved in closed form so the teacher gets
    `teacher_share` of the hours at that record count.
  - **Cap:** the teacher's repeat factor is capped at `max_oversample`. At the cap, EARS fills
    the remaining records and the achieved share falls short of the target.
  - **Empty source:** if either source is empty, share mode is ignored with a warning.
  - **Outputs:** `mix_summary.json` gets `share` (target, achieved, weights, base counts) and
    `teacher_share` (achieved share of hours).
- `ears.use: false` → no EARS records.
- Dev-speaker records are always dropped, except `dev_leak_probe` records when `leak_probe` is
  true. `f1_styled_bench` records are dropped unless `f1_include` is true.
- `match_teacher`: path of a teacher `accepted_raw.jsonl`. The EARS part is then a random subset
  restricted to that file's speakers, with total duration equal to its accepted duration (arm A).
- `category_weights`: EARS records for which `sftlib.matches_category` is true for a listed
  category are repeated that many times. Fractions are kept with probability.
- `teacher.codes`: list of code dirs or files, cumulative across rounds. `oversample` repeats
  teacher records the same way.
- `instruction` is set per record:
  - styled → `sftlib.STYLE_INSTRUCTION[style]`;
  - emotion classes → `sftlib.emotion_instruction(emotion)`;
  - freeform → `sftlib.FREEFORM_TAG` if `freeform_tag == "T"`, else unset;
  - everything else, and all teacher records → unset.
- Texts are re-normalised with the MOSS normaliser. Records whose packed length is > 640 rows
  are dropped, never truncated.
- Output: `$SCR/mixes/<stage>_<arm>.jsonl` (path printed on the last stdout line) and
  `runs/sft/<stage>/<arm>/mix_summary.json` (records, hours, by class, by source, dropped).

### 2.10 Core shell contracts (for orchestration)
- `eval_ckpt.sh NAME MODEL MANIFEST TOKENS [BASE_NAME]` generates into `runs/sft/eval/NAME`
  (incremental: existing wavs are kept), runs BTRM and metrics, then
  `compare.py summarize --eval runs/sft/eval/NAME --base runs/sft/eval/BASE_NAME` (skipped when
  BASE_NAME is empty or equals NAME). `MODEL` = a checkpoint dir or `$MOSS_MODEL_DIR` for base.
- `train.sh STAGE ARM MIX LR EPOCHS SAVE_EVERY` writes `runs/sft/ckpt/STAGE/ARM/checkpoint-epoch-K`
  and `checkpoint-last` (skipped if `checkpoint-last` exists).
- `MOSS_MODEL=... RUN_NAME=... MOSS_MAX_TEXT_TOKENS=50|100 SKIP_GRM=1 bash scripts/run_moss_nano.sh`
  runs the full 494 benchmark of a model (GRM skipped with SKIP_GRM=1).
- `generate_moss_nano.py` accepts `--voice-clone-max-text-tokens N` (default 50).

## 3. Command-line contracts

| Command | Env | Notes |
|---|---|---|
| `sft/ears_inventory.py [--speakers p001,p002] [--workers 8]` | TTS | download → extract → inventory (resumable per speaker: `$SCR/ears/<spk>/.complete`) |
| `sft/align.py [--batch-size 16] [--limit N]` | TTS | inventory → units.jsonl (append, skips done files) |
| `sft/build_ears.py --max-tokens N --moss-repo DIR --model-dir DIR` | MOSS | units → chunks + ears_raw.jsonl |
| `sft/dev_sets.py mirror` / `fresh` / `eval-manifest --sets mirror[,fresh] --seeds 3 --out PATH` | TTS | §2.4 |
| `sft/compare.py summarize --eval DIR --base DIR [--out PATH]` | STD | §2.6, default out `DIR/eval_summary.json` |
| `sft/compare.py diff --a DIR --b DIR --set mirror [--items ID,ID]` | STD | prints JSON `bootstrap_diff` |
| `sft/decide.py decoding --chunked DIR --whole DIR` | STD | §2.8 decoding.json |
| `sft/decide.py pilot` | STD | reads `runs/sft/eval/pilot_*` + teacher summaries → recipe.json, DECISIONS.md, pilot_report.md; exit 3 if status stop |
| `sft/report.py` | STD | REPORT.md |
| `scripts/sft/diagnose.sh STAGE EVAL_NAME` | sh | GRM on seed-0 mirror rows of `runs/sft/eval/EVAL_NAME` → `runs/sft/STAGE/diag/` + tags |
| `sft/tag_weaknesses.py --diag-dir DIR --eval-dir DIR --stage S` | STD | §2.5 (uses `scripts/qwen9b_client.chat`) |
| `sft/gen_texts.py --report PATH --out PATH --stage S (--n N \| --budget-gpu-hours H --throughput PATH)` | STD | §2.7 texts |
| `sft/teacher.py --texts PATH --out-dir DIR --gpu-hours H [--batch-size 16]` | TTS | time-boxed batched Orpheus |
| `sft/filter_teacher.py --teacher-dir DIR` | STD | §2.7 accepted + summary |
| `scripts/sft/teacher_round.sh STAGE STUDENT_MODEL GPU_HOURS N_TEXTS` | sh | texts → teacher → student gen → metrics ×2 → btrm → filter → encode |
| `scripts/sft/eval_ckpt.sh NAME MODEL MANIFEST TOKENS` | sh | gen + btrm + metrics + `compare.py summarize` vs base (core) |
| `scripts/sft/encode.sh RAW_JSONL OUT_DIR` | sh | sharded `prepare_data.py` (core) |
| `scripts/sft/train.sh STAGE ARM MIX LR EPOCHS SAVE_EVERY` | sh | core |
| `sft/make_mix.py --stage S --arm A --config JSON` | MOSS | core |
| `scripts/sft/pilot.sh` / `main.sh` / `final.sh` | sh | orchestration |

## 4. The test round (pilot): what it decides and how

| Question | Experiment | Decision rule (decide.py) |
|---|---|---|
| Q1 Decoding: chunked (50 tok) or whole text (100 tok)? | base MOSS on `eval_mirror.csv` under both | `whole` if its mirror BTRM % > chunked's and mean WER ≤ chunked + 0.005; else `chunked`. Sets `train_max_text_tokens` 75 / 50 |
| Q2 Freeform tag | arms CU vs CT (full EARS, freeform `instruction` unset vs `sftlib.FREEFORM_TAG`) | `T` only if diff(CT − CU) CI lo > 0, else `U` |
| Q3 Teacher vs EARS per hour | arm B (teacher accepted) vs arm A (EARS subset of equal hours, B's speakers) | Δ1 = diff(B − A) |
| Q4 Teacher on top of EARS | arm D (full EARS U + teacher ×3) vs CU | Δ2 = diff(D − CU) |
| Q5 F1 (benchmark speakers' styled readings of their own test texts) | arm F = CU data + all `f1_styled_bench` + `dev_leak_probe` records; compare with CU on dev-mirror (the dev-leak-probe simulates the benchmark speakers' situation on the dev speakers) | env `F1_POLICY` = `exclude` / `include` / `auto` (default `auto`: include iff the **memorisation effect** = diff(F − CU) on the probe items (the dev-mirror items whose texts the dev-leak-probe records cover) − diff(F − CU) on the other dev-mirror items has CI lo > 0. Arm F also heard the dev voices, which raises every dev item and would not carry over to the benchmark, so the whole-set diff is not used). Keep F out of best-checkpoint selection (its dev score is contaminated by design) |
| Recipe | Δ1, Δ2 (Δ1 is ignored if the pilot accepted < 1 h of teacher audio) | Δ1 CI lo > 0 → T (EARS kept iff pct(D) ≥ pct(B)); else Δ1 CI contains 0 or Δ2 CI lo > 0 → M; else H. Teacher GPU-h per main round: H 4, M 10, T 20 |
| Go / stop | every non-F arm vs base | stop (exit 3) if no non-F arm has diff vs base with CI lo > 0 |

Pilot training settings: lr 2e-5 cosine, global batch 32, `--max-length 640`, fp32 master
weights. Arms A and B run 6 epochs; arms CU, CT, D and F run 2 epochs. Only `checkpoint-last` is
saved and evaluated.

### 4.1 Main rounds: source re-weighting check (`sft/reweight.py`, `REWEIGHT=1` default)

Each main round trains, besides the main arm(s) `lr<lr>`, one arm `alt_lr<lr>` at the main lr
(not in the half-lr sweep), same epochs. Its mix is the main mix with the **smaller source's
share of the hours raised**, at the same record count, and so the same optimizer steps (share
mode, §2.9). Only the share differs. Both record counts are logged in the decision.

| Main mix (teacher share s of hours) | Alt arm |
|---|---|
| no teacher hours | not trained (`alt_skip.json`) |
| EARS not in mix (recipe T without EARS) | `ears_up`: EARS added at 25% of the hours |
| s ≤ 0.5 | `teacher_up`: s → min(2s, s + 0.25); not trained if the ×`max_oversample` cap (max(3, recipe oversample); 3 = arm D's tested ×3) allows < +0.02 |
| s > 0.5 | `ears_up`: EARS share e = 1 − s → min(2e, e + 0.25) |

Both arms' checkpoints are candidates for the round's best (round_select). Then, before
`round_summary.json` is written, `reweight.decide` compares the best guard-passing alt checkpoint
with the best guard-passing main checkpoint (paired bootstrap, dev-mirror):
- alt mean higher (or only alt passes the guards) → **adopt**: later rounds' main mix uses alt's
  achieved teacher share (share mode; size = the recipe mix's natural size, or the teacher-only
  size if the recipe had no EARS);
- otherwise the main mix stays.

The state is a teacher *share*, not a repeat factor, because the teacher pool grows every round.
The teacher GPU-hour budget is never changed: if the diff CI lo > 0, DECISIONS.md gets a suggestion
only. `mix_state.json` = `{"teacher_share", "ears_in_mix", "ears_weight", "teacher_oversample",
"max_oversample", "decided": {stage: result}}`. It is initialised from `recipe.json`. Each
`result` keeps `state_before` (for redoing a round by hand) and is also stored as
`round_summary.json["reweight"]`. `decide` is idempotent per stage.
