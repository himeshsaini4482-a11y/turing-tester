# VoiceGym SFT: MOSS-TTS-Nano-100M report

Generated 2026-10-26 13:05. Decoding: **chunked** (voice_clone_max_text_tokens 50).

## Headline: the 494-sentence benchmark

Paired differences are vs `runs/sft_final_base` (base under the chosen decoding).

| run | n | BTRM % [95% CI] | Δ vs base [paired CI] | BTRM parity | ratio | TTS wins | GRM closeness | WER | spk sim | UTMOS |
|---|---|---|---|---|---|---|---|---|---|---|
| base, chunked (50) | 494 | 54.3 [44.2, 64.2] | +0.0 [+0.0, +0.0] | 73.6 | -115.2 | 179/494 | n/a (GRM skipped) | 0.055 | 0.955 | 3.709 |
| base, whole (100) | 494 | 51.5 [41.7, 61.3] | -2.8 [-9.7, +4.0] | 66.9 | -129.4 | 165/494 | n/a (GRM skipped) | 0.057 | 0.954 | 3.754 |
| base, chosen decoding | 494 | 54.3 [44.2, 64.2] | ref | 73.6 | -115.2 | 179/494 | 92.6 | 0.055 | 0.955 | 3.709 |
| **SFT final** | 494 | 94.7 [86.2, 103.9] | +40.4 [+29.0, +52.7] | 96.2 | 103.0 | 240/494 | 101.5 | 0.033 | 0.946 | 4.061 |
| SFT runner-up | 494 | 99.2 [90.6, 108.8] | +44.9 [+33.7, +56.8] | 97.9 | 125.7 | 242/494 | missing | 0.043 | 0.969 | 3.953 |

BTRM % = mean TTS reward / mean human reward × 100 over the same sentences (human = 100). Parity = 200 × mean sigmoid(r_tts − r_human): a different scale, never divided by the human reward. CIs: 95%, 10000 bootstrap draws over sentences.

**Success criteria (PLAN §1): MET.** BTRM % higher, CI excludes 0: yes; WER ≤ base + 0.01: yes; speaker sim ≥ base − 0.02: yes; UTMOS ≥ base − 0.10: yes.

### Per-speaker BTRM %

| speaker | base, chunked (50) | base, whole (100) | base, chosen decoding | **SFT final** | SFT runner-up |
|---|---|---|---|---|---|
| p002 | 42.4 | 46.7 | 42.4 | 73.3 | 79.2 |
| p004 | 65.5 | 61.7 | 65.5 | 102.0 | 86.3 |
| p009 | 7.0 | 14.2 | 7.0 | 62.8 | 69.6 |
| p010 | 71.4 | 28.8 | 71.4 | 104.1 | 114.1 |
| p014 | 70.7 | 63.6 | 70.7 | 199.9 | 181.8 |
| p017 | 26.2 | 26.3 | 26.2 | 134.6 | 69.7 |
| p018 | 129.3 | 114.6 | 129.3 | 112.3 | 136.6 |
| p022 | 49.9 | 63.8 | 49.9 | 75.9 | 81.7 |
| p024 | 112.8 | 90.6 | 112.8 | 94.2 | 113.2 |
| p027 | 38.4 | 56.4 | 38.4 | 112.4 | 126.6 |
| p038 | 15.5 | 40.8 | 15.5 | 82.1 | 88.4 |
| p043 | -4.2 | -19.3 | -4.2 | 68.4 | 84.5 |
| p052 | 46.9 | 36.1 | 46.9 | 75.9 | 92.6 |
| p053 | 71.1 | 55.8 | 71.1 | 71.6 | 116.1 |
| p056 | 27.0 | -16.5 | 27.0 | 293.1 | 221.3 |
| p066 | 58.8 | 77.9 | 58.8 | 98.4 | 65.0 |
| p069 | 118.1 | 97.3 | 118.1 | 156.0 | 182.4 |
| p079 | 81.6 | 73.0 | 81.6 | 64.3 | 87.3 |
| p081 | 39.5 | 28.8 | 39.5 | 66.2 | 54.1 |
| p085 | 98.3 | 101.2 | 98.3 | 115.4 | 133.0 |
| p086 | 5.0 | 36.6 | 5.0 | 243.2 | 203.1 |
| p088 | -4.3 | -10.2 | -4.3 | 91.1 | 74.5 |
| p097 | 64.2 | 64.5 | 64.2 | 87.1 | 90.7 |
| p100 | 85.4 | 76.1 | 85.4 | 118.5 | 125.5 |
| p101 | 44.1 | 47.1 | 44.1 | 52.6 | 94.2 |
| p102 | 63.9 | 57.9 | 63.9 | 78.9 | 74.7 |

## Pilot (test round)

Status **go**; decoding chunked; recipe **M**; freeform tag U; F1 include False; EARS in mix True; teacher 10 GPU-h/round, oversample 1.14; pilot best B (86.2).

Dev sets (unseen speakers), base = `base_chunked`.

| arm | mirror BTRM % [CI] | Δ vs base | fresh BTRM % [CI] | Δ vs base | guards |
|---|---|---|---|---|---|
| base | 59.7 [41.2, 84.2] | — | — | — | — |
| A | 69.3 [46.2, 98.7] | +9.5 [-10.4, +29.1] | — | — | FAIL |
| B | 86.2 [66.8, 117.3] | +26.5 [+12.2, +46.0] | — | — | pass |
| CU | 99.6 [78.7, 135.1] | +39.9 [+23.5, +64.8] | — | — | FAIL |
| CT | 100.9 [79.6, 136.9] | +41.2 [+24.4, +67.1] | — | — | FAIL |
| D | 94.1 [72.7, 129.7] | +34.4 [+19.7, +57.5] | — | — | FAIL |
| F (not selectable) | 105.9 [83.8, 143.7] | +46.1 [+28.9, +73.0] | — | — | FAIL |

- Δ1 teacher vs EARS hour for hour, diff(B − A): +16.9 [-2.7, +43.0]
- Δ2 teacher on top of EARS, diff(D − CU): -5.5 [-19.5, +7.7]
- freeform tag, diff(CT − CU): +1.3 [-10.3, +12.8]
- F1, diff(F − CU): +6.2 [-7.0, +19.8]; on the 36 leak-probe items +4.2 [-4.6, +13.0], on the others — (memorisation estimate)

Full table and reasoning: `runs/sft/pilot/pilot_report.md`.

## Main rounds

| round | best eval | mirror BTRM % [CI] | fresh BTRM % [CI] | vs previous best [paired CI] | confirmed weaknesses | teacher yield | early stop |
|---|---|---|---|---|---|---|---|
| 1 | None | — | — | — | flat_prosody, robotic_or_metallic, unnatural_pauses, pacing_too_slow, stress_emphasis, emotion_mismatch, pacing_too_fast, sentence_final_intonation | 0.370 | True |

Source re-weighting check (alt arm: the smaller source's share of the mix raised, same size and learning rate; `runs/sft/main/mix_state.json`):

- main_r1: teacher_up: teacher share 0.137 → 0.274; no alt checkpoint passed the guard rails — not adopted

Overall best checkpoint: `runs/sft/ckpt/pilot/B/checkpoint-last` (eval pilot_B, stage pilot, dev-mirror BTRM 86.2).

## Weakness shares across stages (GRM diagnosis, dev-mirror seed 0)

| category | pilot (base_chunked) | main_r1 (pilot_B) |
|---|---|---|
| flat_prosody | 0.98 ✓ | 0.99 ✓ |
| unnatural_pauses | 0.83 ✓ | 0.87 ✓ |
| pacing_too_fast | 0.16 ✓ | 0.25 ✓ |
| pacing_too_slow | 0.70 ✓ | 0.62 ✓ |
| sentence_final_intonation | 0.22 ✓ | 0.22 ✓ |
| stress_emphasis | 0.58 ✓ | 0.49 ✓ |
| mispronunciation | 0.06 | 0.04 |
| robotic_or_metallic | 0.81 ✓ | 0.90 ✓ |
| audio_artifacts_noise | 0.13 | 0.06 |
| muffled_bandwidth | 0.05 | 0.01 |
| emotion_mismatch | 0.31 ✓ | 0.41 ✓ |
| truncation_or_omission | 0.01 | 0.00 |
| other | 0.01 | 0.00 |
| n human-preferred / judgments | 83 / 152 | 68 / 152 |

Share = count / judgments where GRM preferred the human clip; ✓ = confirmed (share ≥ 0.15 and ≥ 8 items).

## Data

### `build_summary.json`

| key | value |
|---|---|
| max_tokens | 50 |
| n_chunks | 18981 |
| hours | 47.563 |
| by_class.freeform.chunks | 2163 |
| by_class.freeform.hours | 4.807 |
| by_class.freeform-emotion.chunks | 2868 |
| by_class.freeform-emotion.hours | 6.973 |
| by_class.interjection.chunks | 414 |
| by_class.interjection.hours | 1.121 |
| by_class.read-emotion.chunks | 2089 |
| by_class.read-emotion.hours | 4.346 |
| by_class.read-regular.chunks | 2083 |
| by_class.read-regular.hours | 5.564 |
| by_class.read-styled.chunks | 9364 |
| by_class.read-styled.hours | 24.752 |
| by_flag.dev_leak_probe.chunks | 267 |
| by_flag.dev_leak_probe.hours | 0.677 |
| by_flag.dev_speaker.chunks | 732 |
| by_flag.dev_speaker.hours | 1.813 |
| by_flag.f1_styled_bench.chunks | 1819 |
| by_flag.f1_styled_bench.hours | 4.495 |
| by_flag.is_bench_text.chunks | 9226 |
| by_flag.is_bench_text.hours | 23.715 |
| trainable_non_dev.with_f1.chunks | 18249 |
| trainable_non_dev.with_f1.hours | 45.749 |
| trainable_non_dev.without_f1.chunks | 16430 |
| trainable_non_dev.without_f1.hours | 41.255 |
| ref_audio_prompt_share | 0.694 |
| mean_duration | 9.020 |
| mean_text_tokens | 33.700 |
| drops.bad_unit.n | 1603 |
| drops.bad_unit.hours | 8.547 |
| drops.file_misread.n | 1220 |
| drops.file_misread.hours | 4.905 |
| drops.no_left_boundary.n | 10039 |
| drops.no_left_boundary.hours | 16.976 |
| drops.no_right_boundary.n | 3461 |
| drops.no_right_boundary.hours | 5.326 |
| drops.too_long.n | 72 |
| drops.too_long.hours | 0.464 |
| drops.too_many_tokens.n | 1064 |
| drops.too_many_tokens.hours | 6.679 |
| drops.too_short.n | 165 |
| drops.too_short.hours | 0.038 |
| drops.too_short_after_trim.n | 26 |
| drops.too_short_after_trim.hours | 0.009 |

### `inventory_summary.json`

| key | value |
|---|---|
| n_files | 17227 |
| n_speakers | 107 |
| hours | 100.060 |
| by_class.freeform.files | 642 |
| by_class.freeform.hours | 32.517 |
| by_class.freeform-emotion.files | 2461 |
| by_class.freeform-emotion.hours | 10.116 |
| by_class.interjection.files | 535 |
| by_class.interjection.hours | 2.415 |
| by_class.nonverbal.files | 1284 |
| by_class.nonverbal.hours | 3.712 |
| by_class.read-emotion.files | 2461 |
| by_class.read-emotion.hours | 7.830 |
| by_class.read-regular.files | 2140 |
| by_class.read-regular.hours | 8.970 |
| by_class.read-styled.files | 7704 |
| by_class.read-styled.hours | 34.500 |
| by_flag.is_bench_text.files | 7811 |
| by_flag.is_bench_text.hours | 35.132 |
| by_flag.f1_styled_bench.files | 1404 |
| by_flag.f1_styled_bench.hours | 6.185 |
| by_flag.bench_file.files | 494 |
| by_flag.bench_file.hours | 2.034 |
| by_flag.dev_speaker.files | 644 |
| by_flag.dev_speaker.hours | 3.788 |
| by_flag.dev_leak_probe.files | 216 |
| by_flag.dev_leak_probe.hours | 1.067 |
| by_exclusion.None.files | 15449 |
| by_exclusion.None.hours | 94.313 |
| by_exclusion.benchmark_file.files | 494 |
| by_exclusion.benchmark_file.hours | 2.034 |
| by_exclusion.nonverbal.files | 1284 |
| by_exclusion.nonverbal.hours | 3.712 |
| trainable_non_dev.with_f1.files | 14853 |
| trainable_non_dev.with_f1.hours | 90.662 |
| trainable_non_dev.without_f1.files | 13449 |
| trainable_non_dev.without_f1.hours | 84.476 |
| expected.bench_file | 494 |
| expected.f1_styled_bench | 1404 |
| expected.dev_leak_probe | 216 |

### Training mixes

| stage / arm | records | hours |
|---|---|---|
| pilot / B | 1719 | 2.8 |
| pilot / F | 18516 | 46.4 |
| pilot / A | 1103 | 2.8 |
| pilot / CT | 16430 | 41.2 |
| pilot / CU | 16430 | 41.2 |
| pilot / D | 21587 | 49.5 |
| main_r1 / alt | 37605 | 84.5 |
| main_r1 / mix | 37682 | 90.6 |

## Teacher (Orpheus 3B) yields

| stage | generated | accepted | yield | rejected (eos / rate / wer / sim / btrm) | accepted h | GPU-h | audio s per GPU-s |
|---|---|---|---|---|---|---|---|
| pilot | 4000 | 1719 | 0.430 | 2 / 65 / 208 / 883 / 1123 | 2.7 | 2.8 | 0.970 |
| main_r1 | 13040 | 4827 | 0.370 | 10 / 225 / 685 / 2894 / 4399 | 8.1 | 10.0 | 0.816 |

## Worker time and cost

- Wall span of logged stages: **76.6 h** → **$171.6** at $2.24/h (g6e.xlarge). Idle worker time outside logged stages is not included.
- Sum of logged stage times: 117.9 h (may double-count nested stages that log their own timings).

| stage group | hours |
|---|---|
| pilot | 23.8 |
| main_r1_teacher | 21.4 |
| main_r1_teacher_gen | 10.0 |
| main_r1_teacher_throughput | 10.0 |
| main_r1_teacher_student | 8.7 |
| align | 6.9 |
| final_base | 4.8 |
| final_best | 4.7 |
| pilot_teacher_gen | 2.9 |
| pilot_teacher_throughput | 2.9 |
| pilot_teacher_student | 2.7 |
| main_r1_train_lr1e-05 | 1.6 |
| main_r1_train_lr2e-05 | 1.6 |
| main_r1_train_alt_lr2e-05 | 1.6 |
| final_runner_up | 1.1 |
| main_r1_teacher_prescore | 0.9 |
| main_r1_gen_texts | 0.9 |
| main_r1_texts | 0.9 |
| main_r1_eval_lr1e-05_e1 | 0.9 |
| main_r1_teacher_btrm | 0.8 |
| main_r1_eval_alt_lr2e-05_e1 | 0.8 |
| main_r1_eval_lr2e-05_e1 | 0.8 |
| main_r1_eval_alt_lr2e-05_e2 | 0.8 |
| main_r1_eval_lr1e-05_e2 | 0.7 |
| main_r1_eval_alt_lr2e-05_e3 | 0.7 |
| main_r1_eval_lr2e-05_e2 | 0.7 |
| main_r1_eval_lr2e-05_e3 | 0.7 |
| main_r1_eval_lr1e-05_e3 | 0.7 |
| pilot_diag_judge | 0.6 |
| main_r1_diag_judge | 0.6 |
| ears_inventory | 0.5 |
| pilot_teacher_prescore | 0.3 |
| pilot_teacher_btrm | 0.3 |
| pilot_texts | 0.2 |
| pilot_gen_texts | 0.2 |
| build_ears | 0.1 |
| main_r1_teacher_encode | 0.0 |
| pilot_teacher_encode | 0.0 |
| make_mix_main_r1_alt | 0.0 |
| make_mix_main_r1_mix | 0.0 |
| make_mix_pilot_A | 0.0 |
| make_mix_pilot_D | 0.0 |
| make_mix_pilot_CU | 0.0 |
| make_mix_pilot_F | 0.0 |
| make_mix_pilot_CT | 0.0 |
| pilot_diag_tag | 0.0 |
| make_mix_pilot_B | 0.0 |
| main_r1_diag_tag | 0.0 |
| main_r1_diagnose | 0.0 |
| main_r1_teacher_filter | 0.0 |
| pilot_teacher_prefilter | 0.0 |
| pilot_teacher_filter | 0.0 |
| main_r1_teacher_prefilter | 0.0 |

## Disclosure

- **Seen texts:** other speakers' readings of the 19 benchmark texts are in the training data (decision D5).
- **Seen speakers:** the 26 benchmark speakers' non-benchmark audio (freeform, emotional, styled, their `rainbow_01_regular` prompt clip) is in the training data (D6). The 494 benchmark recordings themselves are not.
- F1 = excluded: the benchmark speakers' styled readings of their own benchmark texts (1,404 files) are not in training. (F1_POLICY auto)
- The benchmark result is therefore an **in-domain** result (seen texts, seen speakers), not held-out generalisation. Dev-fresh (unseen speakers, unseen text) is the one clean generalisation number.
- **SpeechJudge-BTRM was used in the loop:** to filter teacher data (r_teacher > r_student) and to choose checkpoints and pilot decisions on the dev sets (never on the 494). The headline metric is the same judge, so part of any gain may be judge-specific.
- **Licences:** SpeechJudge (models and data) and EARS are CC BY-NC 4.0 (non-commercial).
- **BTRM is not human preference.** It is an automatic reward model; SpeechJudge reports it near chance on real human vs TTS pairs. No human listening test is reported here.

Every decision with its numbers: `runs/sft/DECISIONS.md`.
