# Turing Tester results

Every experiment run so far, with numbers taken from the runs' summary files (`summary.json`,
`btrm_summary.json`, `metrics_summary.json`). Those files are no longer kept in the repo (they are in
its git history); this page is the record. The detailed SFT write-ups are in
[`docs/sft_experiment/`](docs/sft_experiment/).
Audio, per-call judge logs and the DPO round logs stayed on the GPU worker and are not in the repo.

**All scores below are automatic judges (SpeechJudge-GRM, SpeechJudge-BTRM, Whisper, WavLM,
UTMOS). No human listening test has been run, so none of these numbers is a human preference.**

## 1. How to read the numbers

| Metric | Definition | Human reference |
|---|---|---|
| **GRM closeness** | SpeechJudge-GRM scores the TTS clip and the human clip of the same sentence (1–10, both A/B orders, 1 vote each); per sentence `TTS score / human score × 100`, then the mean over sentences | 100 |
| **BTRM %** | SpeechJudge-BTRM reward per clip; `mean TTS reward / mean human reward × 100` over the same sentences (the SFT decision metric) | 100 |
| **BTRM parity** | `200 × mean sigmoid(r_tts − r_human)`; 100 = the reward model can't separate TTS from human | 100 |
| **TTS wins** | sentences where `r_tts > r_human` | half |
| **WER** | Whisper-large-v3 transcript vs target text | 2.3% (EARS 494) |
| **Spk sim** | WavLM x-vector cosine, TTS clip vs the human clip | 1.0 |
| **UTMOS** | predicted naturalness MOS, 1–5 | 4.00 (EARS 494) |

CIs are 95% bootstrap intervals over sentences. `btrm_summary.json` also has `ratio_score`
(mean of per-sentence BTRM ratios). It is not reported here: 99 of the 494 human clips get a
BTRM reward ≤ 0, so per-sentence ratios are undefined or explode (values from −129 to +392).

## 2. Benchmark sets

| Set | Source | Size | Prompt (voice to clone) | Used by |
|---|---|---|---|---|
| **EARS regular 26×19** (default) | `benchmark/manifests/ears_regular_26x19.csv`, EARS studio 48 kHz, CC BY-NC 4.0 | 26 native US-English speakers (13 M / 13 F) × 19 neutral readings = **494** | each speaker's `rainbow_01_regular` | every full run |
| LibriTTS-R 10×50 | `benchmark/manifests/libritts_r_test_clean_10x50.csv`, CC BY 4.0 | 10 speakers × 50 = 500 | other clip, same speaker and chapter | first smoke test only |

## 3. Full 494-sentence benchmark (EARS)

| Run | Model / checkpoint | BTRM % | BTRM parity | TTS wins | GRM closeness [CI] | WER | Spk sim | UTMOS |
|---|---|---|---|---|---|---|---|---|
| `sft_final_base` | MOSS-TTS-Nano-100M base, chunked decoding (50 tokens) | 54.3 | 73.6 | 179/494 | 92.6 [89.9, 95.4] | 5.5% | 0.955 | 3.71 |
| `moss_base_whole` | MOSS-TTS-Nano-100M base, whole-sentence decoding (100 tokens) | 51.5 | 66.9 | 165/494 | — | 5.7% | 0.954 | 3.75 |
| `orpheus_full` | Orpheus 3B pretrained (`unsloth/orpheus-3b-0.1-pretrained`) | 73.7 | 84.4 | 207/494 | — | — | — | — |
| `sft_final` | **MOSS SFT final** (pilot arm B) | 94.7 | 96.2 | 240/494 | 101.5 [99.0, 104.2] | 3.3% | 0.946 | 4.06 |
| `sft_final_runner_up` | MOSS SFT main round 1, lr 1e-5, epoch 3 | 99.2 | 97.9 | 242/494 | — | 4.3% | 0.969 | 3.95 |
| `dpo_r01_beta0.1` | DPO from SFT final, β 0.1 fixed, round 1 | 97.4 | 99.6 | 249/494 | 103.3 [100.4, 106.2] | 3.4% | 0.945 | 4.04 |
| `dpo_t2_pilotB_r03` | DPO from SFT final, dual-ascent β, round 3 | 91.1 | 90.1 | 222/494 | **105.1** [101.9, 108.2] | **3.0%** | 0.944 | **4.08** |
| `dpo_ru_r01_beta0.1` | DPO from SFT runner-up, β 0.1 fixed, round 1 | **104.9** | **104.2** | **257/494** | 100.6 [98.2, 103.0] | 5.1% | **0.970** | 3.97 |
| `dpo_t2_runnerup_r01` | DPO from SFT runner-up, dual-ascent β, round 1 | 96.2 | 95.8 | 238/494 | 99.9 [97.5, 102.3] | 5.1% | 0.969 | 3.95 |

`moss_base_chunked` and `moss_nano_full` are the same model and decoding as `sft_final_base`,
with identical BTRM numbers, so they are not listed separately. "—" = not run (GRM was skipped
on some runs to save GPU time; `orpheus_full` was BTRM-only).

### Paired GRM differences (same 494 sentences)

| Run | vs MOSS base | vs SFT final |
|---|---|---|
| `sft_final` | **+8.9** [+5.4, +12.2] | — |
| `dpo_r01_beta0.1` | +10.6 [+7.1, +14.2] | +1.8 [−1.4, +5.0] |
| `dpo_t2_pilotB_r03` | +12.4 [+8.8, +16.0] | +3.5 [−0.2, +7.5] |
| `dpo_ru_r01_beta0.1` | +8.0 [+4.5, +11.4] | −0.9 [−4.0, +2.2] |
| `dpo_t2_runnerup_r01` | +7.3 [+3.9, +10.7] | −1.6 [−4.9, +1.6] |

From [`docs/sft_experiment/REPORT.md`](docs/sft_experiment/REPORT.md) (BTRM, paired over the 494): SFT final vs base **+40.4 [+29.0, +52.7]**,
runner-up vs base +44.9 [+33.7, +56.8].

## 4. What the results say

1. **SFT works, by every automatic metric.** MOSS-TTS-Nano went from 54% to 95% of the human
   BTRM reward, GRM closeness rose 92.6 → 101.5 (paired +8.9, CI excludes 0), WER fell 5.5% → 3.3%
   and UTMOS rose 3.71 → 4.06, with speaker similarity within guard (0.955 → 0.946).
   The 100M student ended above the 3B Orpheus teacher (73.7%) on BTRM.
2. **DPO on top of SFT is not a clear win yet.** No DPO run beats its own SFT starting point on
   GRM with a CI that excludes 0 (best: +3.5 [−0.2, +7.5]).
3. **The two judges disagree on DPO.** `dpo_t2_pilotB_r03` has the best GRM (105.1), WER and
   UTMOS but the lowest BTRM (91.1, down from 94.7). `dpo_ru_r01_beta0.1` has the best BTRM
   (104.9, the only run above 100) but worse WER (5.1%) and flat GRM. BTRM was the training
   reward, so a BTRM gain without a GRM gain is the reward-hacking pattern the plan warned about.
4. **Above-100 scores do not mean "better than human".** GRM and BTRM are pair-relative
   judges, both have high variance, and SpeechJudge reports BTRM near chance on human-vs-TTS pairs.

## 5. SFT experiments (MOSS-TTS-Nano-100M)

Full report: [`docs/sft_experiment/REPORT.md`](docs/sft_experiment/REPORT.md); every decision: [`docs/sft_experiment/DECISIONS.md`](docs/sft_experiment/DECISIONS.md).

**Data.** EARS: 107 speakers, 100 h; 47.6 h of aligned chunks after filtering (Whisper
large-v3-turbo alignment). The 494 benchmark files are excluded. Teacher: Orpheus 3B cloning
EARS voices on texts written by Qwen 3.5 9B for the GRM-diagnosed weaknesses, filtered by
WER, speaker sim and BTRM (teacher reward > student reward).

**Decoding (Q1).** Chunked (≤ 50 text tokens) vs whole sentence, base MOSS on dev-mirror:
59.7% vs 57.3% BTRM, WER 4.2% vs 5.8% → chunked.

**Pilot arms** (dev-mirror = 4 held-out dev speakers reading benchmark-style texts, 76 items × 3 seeds;
guards: WER ≤ base + 0.01, sim ≥ base − 0.02, UTMOS ≥ base − 0.10, on dev-mirror and dev-fresh):

| Arm | Training data | BTRM % [CI] | Δ vs base [paired CI] | Guards |
|---|---|---|---|---|
| base | — | 59.7 [41.2, 84.2] | — | — |
| A | EARS subset, hour-matched to B (2.8 h) | 69.3 [46.2, 98.7] | +9.5 [−10.4, +29.1] | fail (WER) |
| **B** | Orpheus teacher clips only (2.8 h) | 86.2 [66.8, 117.3] | **+26.5 [+12.2, +46.0]** | **pass** |
| CU | full EARS (41 h) | 99.6 [78.7, 135.1] | +39.9 [+23.5, +64.8] | fail (WER) |
| CT | full EARS, freeform tagged | 100.9 [79.6, 136.9] | +41.2 [+24.4, +67.1] | fail (WER) |
| D | full EARS + teacher ×3 | 94.1 [72.7, 129.7] | +34.4 [+19.7, +57.5] | fail (WER) |
| F | CU + benchmark speakers' styled readings (leak probe) | 105.9 [83.8, 143.7] | +46.1 [+28.9, +73.0] | not selectable |

Full-EARS arms scored highest on read speech but failed the WER guard on spontaneous
(dev-fresh) speech (e.g. CU 15.0% vs base 10.3%), so B was the only arm that passed.
Teacher vs EARS hour for hour: +16.9 [−2.7, +43.0] (not significant) → mixed recipe M.

**Main round 1** (90.6 h mix, lr 1e-5 and 2e-5, plus a teacher-up-weighted arm, 3 epochs each):
the 9 checkpoints reached 54.1–104.7% dev-mirror BTRM, but all failed the dev-fresh WER guard
(e.g. 12.9% vs base 10.3%). Pilot B stayed the official final; the lr 1e-5 epoch 3 checkpoint
was benchmarked as the runner-up by user decision.

**Weaknesses GRM diagnosed** (share of human-preferred judgments, base → after pilot B):
flat prosody 0.98 → 0.99, robotic/metallic 0.81 → 0.90, unnatural pauses 0.83 → 0.87,
pacing too slow 0.70 → 0.62, stress/emphasis 0.58 → 0.49, emotion mismatch 0.31 → 0.41.

**Teacher yield.** 43% (pilot, 1,719 of 4,000 accepted) and 37% (main round, 4,827 of 13,040);
speaker sim and BTRM are the main rejection reasons.

**Cost.** 76.6 h of worker time, about **$172** at $2.24/h (1× L40S, g6e.xlarge).

## 6. DPO experiments (online DPO, MOSS-TTS-Nano)

Pipeline: `dpo/README.md`. Each round the policy samples K = 4 clips for 1,000 fresh prompts,
BTRM ranks them, best vs worst valid sample form a pair (invalid: WER > 0.2, sim < 0.85,
0.15–1.0 s/word violated, no stop), and the policy trains against a frozen π_ref. AdamW, lr 1e-6,
16 pairs per step, 1 epoch per round, up to 3 rounds, patience 1.

| Trial (archived hparams) | π_ref | β | Benchmarked round |
|---|---|---|---|
| `dpo/archive/hparams_trial_01_beta0.1_adamw.json` | SFT final (pilot B) | fixed 0.1 | r01 → `dpo_r01_beta0.1` |
| `dpo/archive/hparams_trial_02_pilotB.json` | SFT final (pilot B) | dual ascent, β0 1.057, ε 10 | r03 → `dpo_t2_pilotB_r03` |
| `dpo/archive/hparams_runner_up_beta0.1_adamw.json` | SFT runner-up | fixed 0.1 | r01 → `dpo_ru_r01_beta0.1` |
| `dpo/archive/hparams_trial_02_runnerup.json` | SFT runner-up | dual ascent, β0 1.018, ε 10 | r01 → `dpo_t2_runnerup_r01` |

Results are in section 3. The run-to-trial mapping is from the run names and archived settings;
the per-round dev reports (`runs/dpo/`) are on the worker, not in the repo.
`dpo_final_moss_nano` has only a generation config: that final run was not scored.

## 7. 5-sentence smoke tests (too small to trust)

| Run | Model | Set | BTRM % | GRM closeness | WER | Spk sim | UTMOS |
|---|---|---|---|---|---|---|---|
| `smoke` | Qwen3-TTS 0.6B Base | LibriTTS-R | — | 113.5 | 4.2% | 0.951 | 4.49 |
| `ears_smoke` | Qwen3-TTS 0.6B Base | EARS | 51.2 | 109.0 | 0.4% | 0.957 | 4.35 |
| `qwen17_smoke` | Qwen3-TTS 1.7B Base | EARS | 53.3 | 93.9 | 0.0% | 0.963 | 4.38 |
| `orpheus_smoke` | Orpheus 3B pretrained | EARS | 51.7 | 94.9 | 1.3% | 0.850 | 3.89 |
| `drama3_smoke` | Fish Drama-3 int4 (local SGLang-Omni server) | EARS | 50.3 | 114.8 | 0.4% | 0.967 | 4.20 |
| `moss_nano_smoke` | MOSS-TTS-Nano-100M | EARS | 35.8 | 82.4 | 0.4% | 0.967 | 3.63 |

All 5 EARS smoke sentences are one speaker (p002). On these 5 the human clips got an unusually
high BTRM reward (17.3 vs 7.9 over the full 494), so smoke BTRM % is not comparable with section 3.

## 8. Limits and disclosures

- **In-domain result.** Other speakers' readings of the 19 benchmark texts and the 26 benchmark
  speakers' non-benchmark audio are in the SFT data (decisions D5/D6). The 494 benchmark
  recordings themselves never were. The dev-fresh set (unseen speakers and texts) is the only
  clean generalisation check.
- **BTRM was in the loop.** It filtered teacher data, picked SFT checkpoints on the dev sets
  and was the DPO reward, so BTRM gains are partly judge-specific. GRM never entered training
  and is the independent check.
- **No human listening test.** That is the missing evidence for any "sounds more human" claim.
- **Not run yet:** Qwen3-TTS on the full 494, a GRM human-vs-human sanity check on EARS, GRM for
  the SFT runner-up and the base whole-decoding run, and the scored DPO final.
- **Licences:** SpeechJudge models and data and EARS are CC BY-NC 4.0 (non-commercial);
  LibriTTS-R is CC BY 4.0.
