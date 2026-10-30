# VoiceGym SFT plan: MOSS-TTS-Nano-100M, 4 days on one L40S

Written 2026-10-22. This is the build-and-run spec for the SFT pipeline. The project background
(SpeechJudge, the benchmark design, the human-closeness formula) is in
`VoiceGym_SpeechJudge_Project_Plan.md`. This document supersedes its §7.2 / §10 choice of
online DPO as the main path: the main path is now **SFT**. How to *run* it on the GPU worker
is in `sft/RUNBOOK.md`.

> **Status (2026-10-22): implemented.** Three entry points, each resumable:
> `scripts/sft/pilot.sh` (the test round: gates, data, baselines, every open question
> answered by experiment, ending in `runs/sft/pilot/recipe.json`), then `scripts/sft/main.sh`
> (rounds driven by that recipe), then `scripts/sft/final.sh` (494 benchmark + report).
> **`sft/INTERFACES.md` is the binding contract for files, formats and decision rules.** Where
> this plan and INTERFACES disagree, INTERFACES wins. Script names in §6/§9 below are from the
> first draft; the actual file map is INTERFACES §1. How to run it: `sft/RUNBOOK.md`.

---

## 1. Goal and success criteria

**Goal:** raise the **SpeechJudge-BTRM** score of `OpenMOSS-Team/MOSS-TTS-Nano-100M` on the
locked EARS benchmark (`benchmark/manifests/ears_regular_26x19.csv`, 26 speakers × 19 neutral
readings = 494 pairs, voice prompt = each speaker's `rainbow_01_regular`) within 4 days of
compute on one 48 GB GPU.

**Baseline** = base MOSS-TTS-Nano on those 494 sentences, generated and scored exactly as
`scripts/run_moss_nano.sh` does today (same seeds, same sampling, same voice prompts).
One decoding setting (chunked at 50 text tokens as today, or whole-text; see §3 and S1 in §6) is
chosen on the *base* model, then frozen for every checkpoint. Base is reported under both.

**The score (user's definition):**

```
BTRM % = mean BTRM reward of the TTS clips / mean BTRM reward of the human clips × 100
```

over the same sentence set. The human recordings are 100%. 5-sentence smoke values: human
mean reward 17.34; MOSS 6.21 → **35.8%**; Orpheus 8.96 → 51.7%; Qwen3-TTS-1.7B 9.24 → 53.3%.
(`human_parity_score`, e.g. MOSS 1.17 / Orpheus 10.83, is a *different* number, 200 × mean
sigmoid(r_tts − r_human). It must not be divided by the human reward.)

**Numbers reported for every checkpoint, base and final:**

| Metric | Source | Role |
|---|---|---|
| **BTRM %** (above), with 95% paired-bootstrap CI over sentences (10,000 resamples) | `btrm.jsonl` | **Headline and decision metric**: teacher choice, pilot arms, checkpoint choice |
| BTRM `human_parity_score`, `ratio_score` (mean of per-sentence ratios), `tts_wins` | `btrm_summary.json` | Secondary, reported |
| GRM human-closeness | `summary.json` | Second judge (base and final only, it is slow) |
| WER, speaker similarity, UTMOS | `metrics_summary.json` | Guard rails: must not get worse |

Within one sentence set the human rewards are fixed, so ranking checkpoints by BTRM % is the
same as ranking them by mean TTS reward. **Throughout this plan `gap(X)` means BTRM % of X on the
named set, and "Δ" is a difference in BTRM % points with its paired CI.**

**Success** = final checkpoint has a higher BTRM % than base on the 494 with the 95% CI of the
difference excluding 0, AND WER ≤ base WER + 1.0 point (absolute), speaker similarity ≥ base − 0.02,
UTMOS ≥ base − 0.10.

---

## 2. Decisions already made (do not re-litigate)

| # | Decision | Made by |
|---|---|---|
| D1 | Student = `OpenMOSS-Team/MOSS-TTS-Nano-100M`, trained with its official `finetuning/sft.py` | user |
| D2 | Teacher for weakness-targeted texts = **`unsloth/orpheus-3b-0.1-pretrained`** (in-context voice cloning). Confirmed by the user 2026-10-22 after seeing the Qwen3-TTS comparison; no teacher bake-off | user |
| D11 | **No self-distillation / best-of-N data in SFT.** Selecting among the student's own samples is the DPO stage that follows this project; keep the two stages separate so their contributions can be measured | user |
| D3 | Weakness diagnosis = SpeechJudge-GRM; targeted texts = Qwen3.5-9B via `scripts/qwen9b_client.py` | user |
| D4 | Main human data = **all of EARS** (48 kHz studio) except the **26 × 19 benchmark files** | user |
| D5 | Other speakers' readings of the 19 benchmark texts **are allowed** in training | user |
| D6 | The 26 benchmark speakers' non-benchmark audio (freeform, emotional, styled, prompt clip) **is allowed** | user |
| D7 | Styles: **everything except non-verbal** (keep all reading styles + all 22 emotions + freeform + interjections; drop `nonverbal_*`, `vegetative_*`, `melodic_*`) | user |
| D8 | LibriTTS-R: **not used** (audio quality) | user |
| D9 | **Neither data source is assumed to be better.** The Round-1 pilot measures EARS vs the 9B→Orpheus teacher loop. The winner gets most of the round-2/3 compute: EARS-led, mixed, or teacher-led (§7.2, §8) | user + plan |
| D10 | One GPU: AWS g6e.xlarge (1× L40S 48 GB) via `tools/aws/gpu_run.sh`, `KEEP_UP=1` | user |

| D12 | **Test round first** (user, 2026-10-22). Every open choice is settled by experiment in the pilot before the main workflow starts: Q1 decoding, Q2 freeform tag, Q3/Q4 teacher vs EARS, Q5 F1. Rules in INTERFACES §4 | user |
| D13 | MOSS training uses **fp32 master weights with bf16 autocast** (`--fp32-master-weights`, our patch). Stock `sft.py` trains pure-bf16 weights, which rounds most lr-1e-5 updates away | plan |

**F1 (tested, not assumed):** the 26 benchmark speakers also read 9 of their own benchmark
texts in 6 other styles (`rainbow_02..08_{fast,slow,loud,whisper,highpitch,lowpitch}`,
`sentences_01..02_{same}`): 9 × 6 × 26 = **1,404 files** of the *same voice saying the same test
sentence*. Pilot arm F simulates including them on the dev speakers (their own styled readings
of the dev-mirror texts) and measures the effect. `F1_POLICY=auto` (default) includes them in
the main workflow only if arm F beats arm CU with the CI excluding 0. The report always states
the setting used and the measured memorisation effect.

**Disclosure (goes into the final report automatically, §11):** because of D5/D6, the benchmark
measures *seen texts (spoken by other people) and seen speakers (on other texts)*. Scores are
therefore an in-domain result, not a held-out generalisation result. The dev-fresh set (§5)
gives the one clean, unseen-speaker-unseen-text number.

---

## 3. Facts the design depends on (verified 2026-10-22)

**MOSS-TTS-Nano** (github.com/OpenMOSS/MOSS-TTS-Nano):
- Official finetuning: `finetuning/prepare_data.py` (encodes `audio` → `audio_codes` and
  `ref_audio` → `ref_audio_codes`), `finetuning/dataset.py`, `finetuning/sft.py` (accelerate),
  `finetuning/verify.py`. Checkpoints load directly with `from_pretrained(..., trust_remote_code=True)`.
- Codec MOSS-Audio-Tokenizer-Nano: 48 kHz stereo in/out, **12.5 frames/s**, RVQ 16 codebooks.
  `prepare_data.py` resamples input audio to the codec rate.
- Defaults: lr 1e-5, warmup 3%, 3 epochs, bf16, `--max-length 1024`,
  `--channelwise-loss-weight 1,32`, AdamW (0.9, 0.95), wd 0.1. ~3.2 GB peak at batch 1 / len 1024.
- `dataset.py` packs: `[user prompt with reference codes + fields] + [target codes] + [audio_end]`.
  **Sequences longer than `--max-length` are silently truncated** and the end-of-audio token is
  lost, which would teach the model not to stop. We must drop over-length records ourselves.
- Prompt fields printed for every record: `Instruction, Tokens, Quality, Sound Event, Ambient
  Sound, Language` (each `None` if missing), then `Text`. The benchmark inference passes **no**
  instruction and **no** language. Training records must leave those unset (→ `None`) to match,
  except where we deliberately tag styles (§4.4).
- `benchmark/generate_moss_nano.py` normalises text with
  `prepare_tts_request_texts(enable_wetext=False, enable_normalize_tts_text=True)` before
  inference, and uses `voice_clone` mode (reference audio only, no reference transcript).
  **Training text must go through the same normaliser**, and training must use `ref_audio`
  (voice-clone) conditioning.
- Needs its own venv: torch 2.7.0, transformers 4.57.1 (see `scripts/run_moss_nano.sh`).
  SFT additionally needs `accelerate>=1.0`, `tqdm`, `torchaudio==2.7.0`.
- **Inference chunks the text** (`modeling_moss_tts_nano.py`, HF): voice-clone mode splits the
  text into sentences, merges them greedily up to `voice_clone_max_text_tokens` = **50** tokens,
  generates each chunk *independently* (no acoustic context from the previous chunk), and joins
  the chunks with **digital silence** (0.24 s; 0.40 s after a chunk of ≤ 4 words). The 19
  benchmark texts are 37–75 tokens (median 50), all 2–5 sentences; **9 of 19 exceed 50 tokens**,
  so ~half the benchmark clips contain a hard-coded silent gap and a prosodic reset mid-utterance.
  No fine-tuning changes this. Passing `voice_clone_max_text_tokens=100` makes every benchmark
  text one chunk. Either way, a chunk is ≤ 50 (or ≤ 75) text tokens, so **training targets must be
  the same shape: 1–3 sentences, ≤ 50 text tokens** (or ≤ 75 under whole-text decoding).
- **The official `finetuning/dataset.py` does not match inference exactly.** Inference
  (`prompting.py`) puts `<im_end>` after `</user_inst>` before the assistant turn; `dataset.py`
  omits it. Everything else (template text, tokenizer boundaries, `None` fields, reference in the
  user slot) is identical: checked with the MOSS sentencepiece model. We apply a one-line patch
  (`sft/patches/dataset_im_end.patch`) at setup, and gate G0a verifies the packed sequence is
  identical to the inference prompt.
- **Every judge hears 16 kHz audio.** SpeechJudge-GRM and BTRM load clips through
  `qwen_omni_utils` at 16 kHz; UTMOS and WavLM also run at 16 kHz. So Orpheus's 24 kHz output
  costs it nothing on any metric. Bandwidth only matters for actual listening.

**EARS** (facebookresearch/ears_dataset, CC BY-NC 4.0): 107 speakers, ~100 h, 48 kHz anechoic.
Per speaker (measured on p001's zip): freeform monologues ~24 min (6 files, untranscribed),
emotional freeform ~8 min (23 files, untranscribed), emotional reading ~5 min (23 files,
transcribed), regular reading ~5.5 min (20 files), styled reading ~17 min (fast/slow/loud/
whisper/high/low pitch), non-verbal/vegetative/interjection/melodic ~4 min.
`transcripts.json` has 115 reading scripts, identical for every speaker. 74 reading files carry
one of the 19 benchmark texts (all `rainbow_02..08_*`, `sentences_01..02_*`,
`sentences_15..24_regular`).
Speakers: 26 benchmark; of the other 81, 71 are native American English (24 M, 45 F, 2 other).

**Orpheus 3B pretrained** (`benchmark/generate_orpheus.py`): HF generate, batch 1, ~30 s per
~12 s clip (≈ 2.5× slower than real time), 24 kHz SNAC output, ~8 GB. In-context cloning,
not trained for it. 5-sentence smoke: speaker sim 0.85 (MOSS 0.97), WER 1.3% (MOSS 0.4%),
BTRM mean reward 8.96 (MOSS 6.21, human 17.34).

**5-sentence smoke baselines** (`runs/*_smoke`, too small to trust):

| | **BTRM %** | mean r_tts | GRM closeness | BTRM parity | spk sim | WER | UTMOS |
|---|---|---|---|---|---|---|---|
| human (EARS) | **100** | 17.34 | — | — | — | 0.9% | 3.83 |
| Qwen3-TTS 1.7B | 53.3 | 9.24 | 93.9 | 4.53 | 0.963 | 0.0% | 4.38 |
| Orpheus 3B | 51.7 | 8.96 | 94.9 | 10.83 | **0.850** | 1.3% | 3.89 |
| Qwen3-TTS 0.6B | 51.2 | 8.88 | 109.0 | 5.12 | 0.957 | 0.4% | 4.35 |
| Fish Drama-3 | 50.3 | 8.72 | 114.8 | 2.57 | 0.967 | 0.4% | 4.20 |
| **MOSS-Nano (student)** | **35.8** | 6.21 | 82.4 | 1.17 | 0.967 | 0.4% | 3.63 |

What follows from this, given the sample is only 5 sentences:
1. **Every TTS teacher scores about half of human (50–53%).** Teacher SFT aims MOSS at ~52%;
   EARS SFT aims it at 100%. That is the main reason EARS is expected to matter, but the pilot decides.
2. **Orpheus (the chosen teacher) has two training-side risks to manage:**
   - voice match 0.85 (all 5 smoke sentences are one speaker, p002, so this is one voice):
     the student can learn the teacher's voice errors. The ≥ 0.88 voice filter (§4.5) guards
     against it; its yield is logged per round.
   - speed: ~30 s per ~12 s clip at batch 1, because Orpheus emits ~82 tokens per audio
     second. Batched generation, and vLLM if batching gives < 2× (§4.5), are what make
     teacher-heavy recipes feasible.

---

## 4. Data

### 4.1 Splits

| Split | Speakers | Content | Use |
|---|---|---|---|
| **benchmark** | 26 benchmark speakers | their 19 `*_regular` benchmark readings (494) | final eval only, never for tuning |
| **dev-mirror** | 4 held-out speakers (2 M + 2 F, native American English, not benchmark) | their 19 `*_regular` benchmark-text readings, prompt = their `rainbow_01_regular` → 76 items | checkpoint choice, GRM diagnosis. Mirrors the benchmark exactly, but with unseen voices |
| **dev-fresh** | same 4 held-out speakers | 10 freeform segments each (5–15 s) → 40 items | clean generalisation check (unseen voice, unseen text) |
| **train-ears** | the other 103 speakers (26 benchmark + 77) | everything allowed by D4–D7 and F1 | SFT |
| **train-teacher** | voices of train-ears speakers | Orpheus audio of 9B-generated targeted texts | SFT (if the pilot supports it) |

The 4 dev speakers are picked with `random.Random(4321)` from the 71 non-benchmark native
American English speakers (2 male, 2 female) and written to `sft/data/dev_speakers.json`,
which is committed so it never changes. They are excluded from **all** training data.

### 4.2 EARS inventory and exclusion rules (`sft/ears_inventory.py`)

Download all 107 speaker zips (~600 MB each, ~64 GB) to the worker's local NVMe
(`/opt/dlami/nvme/ears`, falls back to `data/ears_full`), keeping every wav. Classify each
file by name:

| Class | File pattern | Transcript | Kept? |
|---|---|---|---|
| read-regular | `rainbow_*_regular`, `sentences_*_regular` | `transcripts.json` | yes |
| read-styled | `rainbow_*_{fast,slow,loud,whisper,highpitch,lowpitch}`, `sentences_*_{same}` | `transcripts.json` | yes |
| read-emotion | `emo_*_sentences` | `transcripts.json` | yes |
| freeform | `freeform_speech_*` | Whisper (§4.3) | yes |
| freeform-emotion | `emo_*_freeform` | Whisper (§4.3) | yes |
| interjection | `interjection_*` | Whisper (§4.3), strict filter | yes |
| non-verbal | `nonverbal_*`, `vegetative_*`, `melodic_*` | — | **no** (D7) |

Exclusions, applied in this order, each logged with a count in `inventory_summary.json`:
1. Every file in `ears_regular_26x19.csv` (speaker + reading name): 494 files.
2. If F1=1: benchmark speaker × reading whose text equals one of *that speaker's* benchmark
   texts, in any style: 1,404 files.
3. All files of the 4 dev speakers.
4. Non-verbal classes.

Output: `runs/sft/data/ears_inventory.jsonl`, one line per kept file:
`{speaker, gender, file, path, class, style, emotion, transcript|null, duration_s, is_bench_text}`.

### 4.3 Transcribing and segmenting untranscribed audio (`sft/segment.py`, `.venv-tts`)

For freeform, freeform-emotion and interjection files:
1. **ASR pass 1:** `openai/whisper-large-v3` (transformers pipeline, bf16, sdpa,
   `chunk_length_s=30`, batch 16, `return_timestamps="word"`, language en). To stop Whisper
   silently dropping fillers, which would make the text not match the audio, pass a
   disfluent prompt via `prompt_ids`: *"Um, so, I was like, uh, thinking about it, you know,
   and, hmm, yeah."*
2. **Segment to match inference chunks** (§3): cut at sentence-final punctuation, then merge
   consecutive sentences greedily up to **50 MOSS text tokens**, exactly like
   `_split_text_into_best_sentences`. Cut only at word gaps ≥ 0.25 s. Keep segments of 2–12 s and
   ≥ 4 words. Pad each cut 0.10 s, then trim leading/trailing silence (−40 dBFS) to ≤ 0.15 s.
   If whole-text decoding is chosen at S1, also build a 25% share of 2–3-sentence segments up
   to 75 tokens / 18 s, which is the benchmark's shape under that setting.
3. **ASR pass 2:** re-transcribe every segment alone with `openai/whisper-large-v3-turbo`.
   Drop it if WER(pass-1 text, pass-2 text) > 15%, or speaking rate is outside 8–22 characters/s.
   Interjections: WER > 0% → drop.
4. Write segments as 48 kHz mono 24-bit wavs to NVMe plus `runs/sft/data/segments.jsonl`
   `{seg_id, speaker, src_file, start_s, end_s, text, class, style, emotion, asr_wer}`.
   The start/end times are enough to rebuild the wavs from raw EARS on a new worker without ASR.

For read classes (transcript known): the scripts are 2–5 sentences (7–18 s), so they are
**split into sentence chunks the same way**, using Whisper word timestamps aligned to the known
transcript (word-level alignment of the ASR output to the script; drop the file if alignment
WER > 10%, which means a misread). Each chunk's text is the script's own sentence text, not the
ASR text.

### 4.4 Building SFT records (`sft/build_ears.py`)

For every kept segment/file:
- `text` = `prepare_tts_request_texts(text, enable_wetext=False, enable_normalize_tts_text=True)["text"]`
  (same call as `generate_moss_nano.py`; import from the MOSS repo checkout).
- `ref_audio`: 70% of records use the speaker's `rainbow_01_regular` (= the benchmark's prompt
  condition); 30% use a random other neutral segment of the same speaker, 5–10 s. Never the
  target itself. For the `rainbow_01_regular` target, always use a random other segment.
- `instruction` (MOSS prompt field):
  - read-regular → unset (`None`, the benchmark condition)
  - read-styled → `"Speak fast."`, `"Speak slowly."`, `"Speak loudly."`, `"Whisper."`,
    `"Speak in a high pitch."`, `"Speak in a low pitch."`
  - read-emotion / freeform-emotion → `"Speak with <emotion>."` (EARS emotion name;
    `neutral` → unset)
  - freeform → **ablation in the pilot (§7.2):** variant U = unset, variant T = `"Speak spontaneously."`
  - interjection → unset
- `language` unset.
- Length check: `packed_len = prompt_tokens(text, fields) + ref_frames + target_frames + 2`, where frames =
  ceil(duration × 12.5). Drop if `packed_len > MAX_LEN` (640, see §6). Never let `dataset.py`
  truncate.

Output: `runs/sft/data/ears_raw_{U,T}.jsonl` (MOSS raw format plus our metadata keys, which
`dataset.py` ignores), then `prepare_data.py` → `runs/sft/data/ears_codes_{U,T}.jsonl`.

**Expected yield (estimate):** ~103 speakers × ~50 usable min × ~0.85 after filters ≈
**65–80 h**, ~25–30k records.

### 4.5 Teacher data (`sft/teacher_orpheus.py`, `sft/filter_teacher.py`)

- **Texts:** from the 9B (§6, stage R-3), English, 8–40 words, tagged with a weakness category.
- **Voices:** each text is assigned round-robin to a train-ears speaker; Orpheus clones from
  that speaker's real `rainbow_01_regular` (EARS, 48 kHz).
- **Generation:** batched HF generate (left padding, batch 16, same sampling as
  `generate_orpheus.py`: T 0.6, top-p 0.95, repetition penalty 1.1, seed per text).
  Fallback if batching misbehaves: 4 parallel batch-1 processes. Generation is
  **time-boxed** (stop when the round's GPU-hour budget is spent), not count-boxed.
- **Filter.** A clip is accepted only if all of these hold:
  1. it emitted end-of-speech (not cut by `max_new_tokens`);
  2. its speaking rate is 8–22 characters/s;
  3. Whisper-large-v3 WER ≤ 5%;
  4. WavLM speaker similarity to the reference ≥ 0.88 (log the yield; Orpheus averages 0.85);
  5. BTRM `r_teacher > r_student`, where `r_student` is the current MOSS checkpoint on the same text and voice.
- **SFT record:** `ref_audio` = the same real EARS `rainbow_01_regular`, `audio` = Orpheus clip
  (24 kHz, upsampled by `prepare_data.py`), `instruction` unset.
- **Teacher check in S1** (`sft/teacher_check.py`): Orpheus generates the 76 dev-mirror items
  (4 unseen speakers, timed), scored with BTRM %, WER and voice match, and run through the
  acceptance filter below with base MOSS as the student. This gives the real voice match over 4
  voices (the smoke run had one), the acceptance yield, and accepted audio per GPU-hour, which sets
  the teacher budgets in §7.2. Output: `runs/sft/teacher_check.json`. If yield < 20% because of
  the voice filter, report it to the user before round 1; do not silently lower the threshold.
- **Caveat:** teacher voice match is weaker than MOSS's own (Orpheus 0.85, Qwen 0.96 vs MOSS
  0.97). A large teacher share can pull the student's voice away from the reference. There is no
  fixed cap. The pilot measures this (arms B vs A, D vs C), the teacher share in rounds 2–3 is set at
  G1 (§8), anywhere from 0% to teacher-led, and the speaker-similarity guard rail (§1) catches
  drift. Orpheus's 24 kHz output does not affect any metric (§3); it only limits listening quality.
- **Teacher volume depends on Orpheus speed.** That is the one thing that limits a teacher-led
  run. At the measured batch-1 speed (~2.5 GPU-h per audio hour) with ~50% filter yield, 20 GPU-h
  gives ~4 h of accepted audio. With batched generation at ~5× (to be measured in S0), the same
  20 GPU-h gives ~20 h. If the S0 probe shows batching gives < 2×, add a vLLM serving path for Orpheus
  before round 2. Orpheus is a Llama-3.2-3B, which vLLM supports.

---

## 5. Evaluation protocol

- **Dev eval** (`scripts/sft/eval_dev.sh CKPT NAME`): MOSS generates dev-mirror (76 items) with
  **3 seeds each** (228 clips) and dev-fresh (40 × 3), then BTRM + WER + speaker sim + UTMOS. Report the
  BTRM % with a paired bootstrap CI vs base (resampling items, 10,000 draws). Cheap
  (estimate ≤ 30 min).
- **GRM diagnosis** (`sft/diagnose.py`): dev-mirror, seed 0 only, MOSS vs human, both orders →
  152 GRM calls. Each GRM analysis is tagged by the 9B into a fixed taxonomy (§6 R-2).
- **Full benchmark** (unchanged scripts): `MOSS_MODEL=<ckpt> RUN_NAME=<name> bash
  scripts/run_moss_nano.sh` → 494 clips, GRM (988 calls), metrics, BTRM. Only for **base**,
  the **final** checkpoint and, if time allows, the runner-up. Never used to choose anything.
- Every checkpoint is compared with the **base model evaluated the same way** (base dev eval is
  run once in stage S1).

---

## 6. Pipeline stages

Environments: `TTS` = `.venv-tts`, `JUDGE` = `.venv-judge`, `MOSS` = MOSS venv (+accelerate),
`STD` = any Python (standard library only).

### S0 — Setup and hard gates (`scripts/sft/s0_setup.sh`), ~1 h
1. `scripts/sft/setup_moss.sh`: create the MOSS venv (same recipe as `run_moss_nano.sh` +
   `accelerate>=1.0 tqdm`). Clone MOSS-TTS-Nano at a **pinned commit** (record the hash on first
   run; same commit for training and inference). Download `MOSS-TTS-Nano-100M` and
   `MOSS-Audio-Tokenizer-Nano`. `run_moss_nano.sh` is refactored to call this script.
2. **Gate G0a, prompt parity** (`sft/check_prompt.py`, MOSS): for 5 texts, build the voice-clone
   prompt the way `model.inference(mode="voice_clone")` does, and the way the *patched*
   `dataset.py` does for a record with the same `ref_audio` / no instruction / no language.
   Assert the token-id sequences are **identical** up to the target audio. Print a diff on failure.
   The stock `dataset.py` is known to fail this (missing `<im_end>`, §3); `setup_moss.sh` applies
   the patch first. **Stop the project if it still fails.**
3. **Gate G0b, round-trip:** run `sft.py` for 1 step at lr 1e-12 on 8 records, then generate
   5 benchmark sentences with `generate_moss_nano.py --model <that ckpt>` and with the base
   model, same seeds. The waveforms must be identical, or almost (max abs diff < 1e-3).
   This proves the checkpoint loads tokenizer, codec and dtype the same way.
4. **Gate G0c, 9B reachable from the worker:** `python scripts/qwen9b_client.py "Reply with
   exactly: QWEN_OK"`. If unreachable, run R-2 tagging and R-3 text generation on the UAI server instead
   (both are STD).
5. **Throughput probes** (written to `runs/sft/timings.json`): MOSS SFT samples/s at batch 8 /
   len 640; MOSS generation s/clip; Orpheus s per audio-second at batch 1 and batch 16;
   BTRM s/clip.

### S1 — Baselines (`scripts/sft/s1_baseline.sh`), ~6–8 h wall, overlaps S2
- **Decoding setting.** Generate the 494 with base MOSS twice: `moss_base` (today's default,
  chunked at 50 tokens) and `moss_base_whole` (`--voice-clone-max-text-tokens 100`, one chunk per
  text). Score both with BTRM + metrics (cheap). GRM runs on `moss_base` and, if it is chosen, on
  `moss_base_whole`. **Rule:** use whole-text decoding for everything if its base BTRM % is
  higher and its WER is not worse by > 0.5 pt; otherwise keep chunked. Write the choice to
  `runs/sft/decoding.json`; every later generation call reads it. Both base rows go in the report.
  This is a decoding change applied identically to base and trained models, not a training change.
- `RUN_NAME=moss_base bash scripts/run_moss_nano.sh` (494 full: gen, GRM, metrics, BTRM).
  **Record GRM seconds/call** in `timings.json`. Everything else in the plan is re-budgeted from it.
- Build `sft/data/dev_speakers.json`, `runs/sft/dev/dev_mirror.csv`, `runs/sft/dev/dev_fresh.csv`
  (needs the S2 segments for dev-fresh). Run base dev eval and base GRM diagnosis → `runs/sft/base/`.
- **Teacher check** (§4.5) on dev-mirror → `runs/sft/teacher_check.json`.

### S2 — EARS data (`scripts/sft/s2_ears.sh`), ~4–6 h
`ears_inventory.py` → `segment.py` → `build_ears.py` (U and T variants) → `prepare_data.py`.
DoD: `ears_codes_{U,T}.jsonl` exist, `inventory_summary.json` shows the exclusion counts,
total hours ≥ 50 (else investigate before training).

S1 and S2 run inside **one** job (the worker runs one tmux job at a time). VRAM plan:
GRM (~26 GB) + Whisper (~6 GB) + codec (~2 GB) together; BTRM only after GRM has finished.

### Round r (r = 1, 2, 3): `scripts/sft/round.sh R=<r>`

**R-1 Diagnose.** `diagnose.py --ckpt <best so far, base for r=1>` → 152 GRM calls on dev-mirror.

**R-2 Tag and aggregate.** Each GRM analysis goes to the 9B with a fixed taxonomy. The 9B returns
JSON `{"weaknesses": [category, ...], "evidence": "<quote>"}` for the *TTS* clip only:
`flat_prosody, unnatural_pauses, pacing_too_fast, pacing_too_slow, sentence_final_intonation,
stress_emphasis, mispronunciation, robotic_or_metallic, audio_artifacts_noise, muffled_bandwidth,
breathiness_voice_quality, emotion_mismatch, speaker_mismatch, truncation_or_omission, other`.
Python validates the JSON (retries ×2, then `other`) and counts per category as a share of TTS-side analyses where
the judge preferred human. Joins in WER, speaker sim and BTRM per item.
→ `runs/sft/round<r>/weakness_report.json`, plus a markdown table.
A category is **confirmed** if share ≥ 15% and ≥ 8 items.

**R-3 Targeted texts.** `gen_texts.py`: for each confirmed category, prompt the 9B with the
structured summary (category, share, 3 evidence quotes, 3 example items) and ask for 25
sentences per call, JSON `[{text, category, why}]`. Mix 75% targeted (split across confirmed categories by share) and 25%
general/retention. Validate: 8–40 words, English letters/punctuation (digits allowed),
no duplicates (normalised exact match plus 5-gram Jaccard ≥ 0.6 against all earlier rounds).
Request 1.6× what the teacher budget can render.
→ `runs/sft/round<r>/texts.jsonl`.

**R-4 Teacher audio.** `teacher_orpheus.py` (time-boxed) → `filter_teacher.py` (TTS for WER/sim, then JUDGE for BTRM vs current student).
→ `runs/sft/round<r>/teacher/{wav/, accepted.jsonl, filter_summary.json}` (yield per filter).

**R-5 EARS re-weighting.** Map confirmed categories to EARS records with Python features
(e.g. `sentence_final_intonation` → texts containing `?`; `pacing_*` → fast/slow styles and long
texts; `emotion_mismatch` → emotional classes; `flat_prosody` → emotional + freeform). Weight
those records ×2 in the mix. No new EARS audio is created.

**R-6 Mix and train** (`make_mix.py` → `scripts/sft/train.sh`). **Always retrain from the base
model** on the cumulative mix. Training is cheap, and starting from base avoids compounding drift.
→ checkpoints under `runs/sft/round<r>/<arm>/ckpt-epoch*`.

**R-7 Evaluate** every arm/epoch with dev eval. Pick by the rule in §8. Copy the winner to
`runs/sft/best/` and record it in `runs/sft/DECISIONS.md`.

### Training config (`scripts/sft/train.sh`, MOSS venv)
```
accelerate launch --num_processes 1 <MOSS repo>/finetuning/sft.py \
  --model-path <MOSS-TTS-Nano-100M> --codec-path <MOSS-Audio-Tokenizer-Nano> \
  --train-jsonl <mix_codes.jsonl> --output-dir <out> \
  --per-device-batch-size 8 --gradient-accumulation-steps 4   # global batch 32
  --learning-rate 1e-5 --warmup-ratio 0.03 --lr-scheduler-type cosine \
  --num-epochs 3 --save-every-epochs 1 --mixed-precision bf16 \
  --max-length 640 --channelwise-loss-weight 1,32 --seed 42
```
`--max-length 640`: 25 s target (313 frames) + 7.7 s ref (97) + ≤ 180 prompt/text tokens fits.
Over-length records are already dropped in `make_mix.py`. Raise to 768 if more than 1% are dropped.
Hold out 2% of the mix as a loss-only validation file; log validation loss each epoch.

---

## 7. Schedule (96 h wall clock; estimates, re-plan from `timings.json`)

| Hours | Stage | Notes |
|---|---|---|
| 0–1 | S0 | gates G0a/b/c, throughput probes |
| 1–8 | S1 ∥ S2 | base 494 benchmark (GRM in background) + EARS download/ASR/segments/codes |
| 8–9 | base dev eval + R1-1/R1-2 | 152 GRM calls on dev-mirror |
| 9–15 | R1-3/R1-4 pilot teacher | time box **5 GPU-h** of Orpheus, aim ≥ 3 h accepted |
| 15–22 | R1-6/R1-7 pilot training + eval | arms below |
| 22 | **Gate G1** | choose the round-2 recipe (§8) |
| 22–46 | Round 2 | recipe H, M or T from G1 (§7.2); teacher budget 4–20 GPU-h |
| 46–70 | Round 3 | diagnose round-2 best, new targeted texts + teacher audio for the categories still weak, small lr/epoch sweep |
| 70–90 | Final | full 494 benchmark of final (+ runner-up), report |
| 90–96 | Buffer | copy back, `gpu_down.sh` |

### 7.1 Round 1 pilot arms
Equal **hours** for A and B, the same speakers' voices, same steps/epochs (3), lr 1e-5:

| Arm | Data | Question it answers |
|---|---|---|
| A | EARS U-variant, random subset matched to B's accepted hours (only speakers used by B; same style mix as the full set) | human data, hour for hour |
| B | Teacher accepted clips (R1-4) | teacher data, hour for hour |
| C-U | Full EARS, freeform `instruction` unset | does scale help? |
| C-T | Full EARS, freeform `instruction="Speak spontaneously."` | does tagging freeform keep `None` = neutral read speech? |
| D | Full EARS (better of C-U/C-T's tagging; use U if not yet known) + all of B, teacher oversampled ×3 | does the targeted teacher data add something EARS doesn't have? |

If a full-EARS epoch takes > 2 h (from the S0 probe), run C-U/C-T/D for 1 epoch in the pilot.
A–B is a per-hour comparison at small scale. D–C asks whether the targeted loop helps on top of
the full human set. Both feed G1.
If the user keeps Orpheus *and* wants Qwen compared, B splits into B-o and B-q on the same texts.

### 7.2 Round 2 / 3 recipes (chosen at G1, §8)

| Recipe | When | Training data | Orpheus GPU-h per round |
|---|---|---|---|
| **H**: EARS-led | teacher loses per hour and adds nothing on top | all EARS (~65–80 h), re-weighted to the weak categories; teacher only for categories EARS cannot cover | 4 |
| **M**: mixed | teacher about equal per hour, or it adds on top of EARS | all EARS + all accepted teacher audio, teacher oversampled to ~30–40% of samples | 10 |
| **T**: teacher-led | teacher beats EARS per hour | all accepted teacher audio (cumulative over rounds), with EARS added only if the pilot shows it helps (§8) | 20 (most of the round) |
In every recipe each round runs the full loop: GRM diagnoses the current best → the 9B writes
new texts for the categories still weak → Orpheus speaks them → filter → retrain from base on the
cumulative data. In recipe T this is the user's original design, where the loop *is* the
training set. Recipes M and T need more Orpheus throughput (see §4.5).

Round 3 sweep (only on the round-2 recipe): lr ∈ {1e-5, 2e-5} × epochs ∈ {2, 3}, 4 runs, pick
by dev eval.

---

## 8. Decision rules (the runbook applies these and logs them in `DECISIONS.md`)

**Checkpoint choice:** among all checkpoints of the round, maximise dev-mirror BTRM % (3
seeds), subject to the guard rails against base: WER ≤ base + 1.0 pt, speaker sim ≥ base − 0.02,
UTMOS ≥ base − 0.10. A checkpoint failing any guard is ineligible.

**Gate G1 (after the pilot):**
- Freeform tag: pick C-T over C-U only if its BTRM % CI lower bound is above C-U's mean,
  else C-U (simpler).
- Recipe. Let C = the better of C-U/C-T, and use paired CIs:
  Δ1 = gap(B) − gap(A) (teacher vs EARS, hour for hour), Δ2 = gap(D) − gap(C) (teacher added on top of full EARS).
  - Δ1 CI entirely > 0 → **recipe T**. Keep EARS in the mix only if gap(D) ≥ gap(B).
  - else Δ1 CI contains 0, **or** Δ2 CI entirely > 0 → **recipe M**.
  - else → **recipe H**.
  - Write the recipe, budgets and the numbers that chose it to `runs/sft/round1/recipe.json`
    and `DECISIONS.md`.
- At the end of round 2, re-check with the same logic: if the round-2 data shows the other source
  doing better (e.g. teacher categories improved more than EARS-matched ones), round 3 may
  switch recipe. Log why.
- If **no** pilot arm beats base (every BTRM % CI includes or is below base): stop and
  report to the user before spending more compute. Likely suspects in order: prompt mismatch (G0a),
  lr too high/low, data errors in a sample of records.

**Stop early** if round r's best checkpoint is not better than round r−1's best (CI overlaps
and mean lower). Go to Final with round r−1's best.

---

## 9. Code to write (dev machine), file by file

| File | Env | Purpose |
|---|---|---|
| `scripts/sft/setup_moss.sh` | sh | MOSS venv + accelerate, pinned MOSS repo, model download; reused by `run_moss_nano.sh` |
| `scripts/sft/s0_setup.sh` | sh | S0: setup + gates + probes |
| `scripts/sft/s1_s2.sh` | sh | S1 ∥ S2 in one job with the VRAM plan |
| `scripts/sft/round.sh` | sh | R-1..R-7 for `R=<r>`, `ARMS=...`, `TEACHER_GPU_HOURS=...` |
| `scripts/sft/train.sh` | sh | `sft.py` wrapper (config above) |
| `scripts/sft/eval_dev.sh` | sh | dev eval of a checkpoint |
| `scripts/sft/final.sh` | sh | full 494 benchmark of named checkpoints + report |
| `sft/common.py` | STD | paths (NVMe/EBS), split constants, JSONL helpers, `packed_len`, bootstrap |
| `sft/ears_inventory.py` | TTS | §4.2 |
| `sft/segment.py` | TTS | §4.3 |
| `sft/build_ears.py` | MOSS | §4.4 (MOSS for the text normaliser + tokenizer for `packed_len`) |
| `sft/dev_sets.py` | STD | dev-mirror / dev-fresh manifests (same columns as benchmark manifests) |
| `sft/check_prompt.py` | MOSS | gate G0a |
| `sft/diagnose.py` | JUDGE + STD | R-1 (reuses `benchmark/judge.py`'s `load_judge`) and R-2 |
| `sft/gen_texts.py` | STD | R-3 (imports `scripts/qwen9b_client.py`) |
| `sft/teacher.py` | TTS | R-4 Orpheus generation, batched with left padding; reuses `encode`/`decode` from `generate_orpheus.py` |
| `sft/teacher_check.py` | TTS + JUDGE | S1 teacher check (§4.5) |
| `sft/filter_teacher.py` | TTS + JUDGE | R-4 filters (two passes, two venvs) |
| `sft/patches/dataset_im_end.patch` | — | adds `<im_end>` to `finetuning/dataset.py`'s assistant prefix (§3); applied by `setup_moss.sh` |
| `sft/make_mix.py` | STD | R-5/R-6 mixing, weights, caps, length filter, 2% val split |
| `sft/compare.py` | STD | BTRM % with paired bootstrap CIs, guard rails, checkpoint choice → `DECISIONS.md` |
| `sft/report.py` | STD | final `runs/sft/REPORT.md` incl. disclosure |
| `benchmark/generate_moss_nano.py` | — | add `--seeds` / output suffix for multi-seed dev eval and `--voice-clone-max-text-tokens` (default 50 = today's behaviour); otherwise unchanged |
| `.gitignore` | — | allow small results: `runs/sft/**/*.md`, `**/weakness_report.json`, `**/*summary.json`, `timings.json` |

Conventions: match the repo style (module docstring explaining the stage, argparse, resumable
outputs that skip finished work, `flush=True` progress lines with ETA, standard library where
possible). Every stage is resumable and idempotent.

**Build order:** common → setup_moss/s0 + check_prompt → dev_sets → ears_inventory/segment/
build_ears → train.sh/make_mix → eval_dev/compare → diagnose → gen_texts → teacher_orpheus/
filter_teacher → round.sh → final/report. Before pushing, test locally what can run without a
GPU: `python -m compileall -q benchmark scripts sft`, the 9B client, text generation on a fake
weakness report, `make_mix` on synthetic records, and `compare` on fake BTRM files.

---

## 10. Storage and what survives

| Location | Survives stop | Survives `gpu_down` (terminate) | Contents |
|---|---|---|---|
| NVMe `/opt/dlami/nvme/...` | no | no | raw EARS zips/wavs, segment wavs, HF caches, MOSS venv, judge checkpoints |
| Worker EBS `~/voicegym/data/` | yes | **no** | nothing that matters |
| Worker EBS `~/voicegym/runs/` | yes | copied back to the UAI server by `gpu_run.sh` / `gpu_down.sh` | **everything that cost GPU time**: `runs/sft/**` (segments.jsonl with timestamps, codes JSONL, teacher wavs, accepted lists, best checkpoints, eval outputs) |

Keep only the best 2 checkpoints per round under `runs/sft/` (≈ 0.4 GB each); delete the rest.
A replacement worker gets prior work back with `PUSH_RUNS=sft` (see runbook). The raw EARS
audio is re-downloaded, and segments are rebuilt from `segments.jsonl` timestamps without re-running ASR.

---

## 11. Report (`sft/report.py` → `runs/sft/REPORT.md`)

Base vs final (and runner-up) on the 494: BTRM % with CI, parity, ratio, wins, GRM
closeness, WER, speaker sim, UTMOS, per-speaker table; dev-mirror and dev-fresh results per
round; weakness-category shares per round (did targeted categories shrink?); data table
(hours per class, teacher hours, yields); timings and cost. **Disclosure section (always printed):**
D5, D6, F1 setting, "BTRM was used for teacher filtering and checkpoint choice on the dev
sets", judge licences (SpeechJudge, EARS: CC BY-NC 4.0).

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Training prompt ≠ inference prompt (normaliser, `None` fields, ref mode, missing `<im_end>` in stock `dataset.py`) | patch + gate G0a, normalise every text with the inference normaliser |
| Training targets longer than inference chunks (model never sees multi-sentence context at test time) | segment training data to ≤ 50 tokens like the inference chunker (§4.3) |
| Hard-coded silent gaps between chunks drag BTRM down and show up as "unnatural pauses" that SFT cannot fix | S1 decoding decision (whole-text vs chunked), frozen for base and trained || Silent truncation drops end-of-audio → model rambles | `packed_len` filter in `build_ears`/`make_mix`; `max_new_frames` stays 375 |
| Styled/emotional data leaks into neutral output | `instruction` tags; pilot C-T vs C-U; dev UTMOS/WER guards |
| Whisper drops fillers in freeform → text/audio mismatch | disfluent prompt + 2-pass WER filter |
| Orpheus voice drift / 24 kHz dullness | sim ≥ 0.88, BTRM teacher > student, pilot arms B and D measure it, UTMOS/sim guard rails |
| Teacher-led recipe starved by slow Orpheus | S0 batching probe; vLLM path if batching < 2× |
| Worker stops/terminates mid-run | resumable stages; durable outputs in `runs/`; `PUSH_RUNS=sft` |
| GRM slower than assumed | GRM only on base/final full runs and 152-call diagnoses; re-plan from `timings.json` |
| Dev noise hides real differences | 3 seeds × 76 items, paired bootstrap |
| Overfitting to the BTRM judge | dev-fresh, GRM and human listening on 20 pairs at the end |
