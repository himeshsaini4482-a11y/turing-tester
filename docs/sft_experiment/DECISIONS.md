
## 2026-10-23 08:28 — Manual override: gate G0a (user-approved)

Gate G0a_prompt failed only on its codec batch check; it was overridden by the operator with the user's approval.

- Prompt parity (the gate's main check): PASS. Training prompt == inference prompt token for token, 3 references x 12 texts.
- Codec batch8 == batch1: FAIL on synthetic_ref_2.wav only. 1 of 1392 codes differs (frame 14, codebook 13 of 16).
  The difference is deterministic and does not come from padding: ref_2 is the longest clip in the batch, and ref_2
  encoded alone at bs=8 equals bs=1. ref_0 and ref_1 match exactly.
- Mitigation: every job runs with ENCODE_BATCH=1, so training codes are encoded exactly like inference (bs=1 == inference, verified by the gate).
- Action: touched runs/sft/.done/pilot_02a_gate_prompt; no code changed.

## 2026-10-23 10:05 — Q1 decoding

**Decoding: chunked** (voice_clone_max_text_tokens 50, train_max_text_tokens 50).

| decoding | BTRM % [95% CI] | WER | sim | UTMOS |
|---|---|---|---|---|
| chunked (50) | 59.7 [41.2, 84.2] | 0.042 | 0.955 | 3.660 |
| whole (100) | 57.3 [39.5, 79.7] | 0.058 | 0.954 | 3.725 |

Paired diff whole − chunked: -2.5 [-13.4, +7.1] (228 rows, base MOSS on dev-mirror).

Rule (INTERFACES §4 Q1): whole iff BTRM% whole > chunked (57.3 vs 59.7: False) and WER whole <= chunked + 0.005 (0.058 vs 0.042: False).

## 2026-10-23 22:40 — Operator fix: EARS alignment re-run (Whisper turbo, energy-based chunk boundaries)

The first build_ears gave 23.9 h of EARS chunks; the runbook expects 60-80 h and says not to trust the pilot under 40 h. Freeform kept 0.76 of ~32 h, mostly dropped as no_right_boundary, and dev_fresh failed because p016 had 0 freeform chunks of 4-12 s.

Causes, measured on the worker:
- transformers Whisper word timestamps are contiguous: the median word gap is 0.000 s, and only ~1% of gaps are >= 0.2 s, with both large-v3 and turbo. So the "gap >= 0.2 s" boundary test almost never passed.
- The energy threshold p95 - 35 dB sat below the EARS noise floor (p95 -32 vs p5 -58 dB), so no pause counted as quiet.

Fix (sft/align.py, commit 29c5f1b on sft-vm-fixes): the pause is now measured from frame energy (quiet_gap), with threshold = max(p95 - 35 dB, p5 + 12 dB). The 12 dB was chosen by a sweep over 19 EARS files: right_ok went from 6/36/28/30/28% to 48/54/50/76/57% (freeform/freeform-emotion/read-regular/read-styled/read-emotion), and bad-unit counts were unchanged.

Alignment model: openai/whisper-large-v3-turbo (align.py --model). large-v3 was ~3x slower in an A/B test and timed out the schedule. Its word timestamps are no better for boundaries (also a median word gap of 0.000 s). Two speed fixes are in the same file (attentions dropped after word timestamps, 6.3x; OOM halving, 87a91c8).

The previous units.jsonl is kept as runs/sft/data/units.v1_gapbug.jsonl, and build_summary.json as build_summary.v1_gapbug.json.

## 2026-10-24 02:15 — Operator fix: dev-fresh with 36 items (p046 7, p061 9 freeform chunks)

After the alignment boundary fix, build_ears gives 47.6 h of EARS chunks (23.9 h before): freeform 4.81, freeform-emotion 6.97, interjection 1.12, read-emotion 4.35, read-regular 5.56, read-styled 24.75. That is above the runbook's 40 h bar but below the 60-80 h expected. The main remaining drops are too_many_tokens (6.68 h, chunks over the 50-token limit with no pause inside) and no_right_boundary (5.33 h).

The dev speakers have 12/27/7/9 freeform chunks of 4-12 s (p015/p016/p046/p061), and dev_sets.fresh_rows required 10 each. Changed (sft/dev_sets.py, sft-vm-fixes): use all eligible chunks when a speaker has 5-9, with a warning. Dev-fresh = 36 items (x3 seeds), dev-mirror is unchanged, and all decisions still use dev-mirror.

## 2026-10-24 02:48 — Operator fix: dev-fresh BTRM % undefined (human mean reward < 0)

On dev-fresh (EARS freeform chunks of the 4 dev speakers), the human recordings get a negative mean BTRM reward: -2.44, with 60/108 clips <= 0. So BTRM % (mean TTS / mean human) is undefined there, and compare.summary_of crashed. It now records btrm_pct = null with an 'undefined' note and mean_reward_gap (TTS - human; base: -2.12).

Dev-fresh BTRM % and its diffs will show as '—' throughout. All decisions use dev-mirror, so they are unaffected. Base on dev-mirror, chunked decoding: BTRM 59.7% [41.4, 84.2], 76 items x 3 seeds.

Observation for the user/Daksh: BTRM scores spontaneous (freeform) human speech below 0 on average, so BTRM % is not a usable metric for freeform speech.

## 2026-10-24 16:40 — Pilot: Q2 freeform tag

**freeform_tag = U.** diff(CT − CU) on dev-mirror = +1.3 [-10.3, +12.8]; CI lo ≤ 0 → U (simpler). dev-fresh diff(CT − CU) = — (secondary). Rule: INTERFACES §4 Q2 (T only if diff CI lo > 0; this replaces PLAN §8's 'CT lo above CU mean').

## 2026-10-24 16:40 — Pilot: Recipe

**recipe = M** (Δ1 CI contains 0 → M); teacher GPU-h per main round 10, teacher_oversample 1.14, ears_in_mix True.

- Δ1 = diff(B − A), teacher vs EARS hour for hour, dev-mirror: +16.9 [-2.7, +43.0] (dev-fresh —)
- Δ2 = diff(D − CU), teacher on top of full EARS, dev-mirror: -5.5 [-19.5, +7.7] (dev-fresh —)
- Rule: INTERFACES §4 recipe row.
- teacher_oversample for M: {"n_ears_records": 16430.0, "n_teacher_pilot": 1719.0, "pilot_teacher_gpu_h": 2.833833333333333, "projected_teacher_after_round1": 7784.988354996178, "target_share": 0.35, "raw": 1.1364080039047688, "clamped_to": [1.0, 5.0]}

## 2026-10-24 16:40 — Pilot: Q5 F1 (styled readings of own test texts)

**f1_include = False.** F1_POLICY=auto but F, CU or the probe items are missing: safe default exclude.

- Memorisation estimate, diff(F − CU) on the 36 dev-mirror items whose text the dev-leak-probe records cover: +4.2 [-4.6, +13.0]
- … on the other 40 mirror items: —
- dev-fresh diff(F − CU): —
- F is excluded from best-checkpoint selection (its dev-mirror score is contaminated by design).

## 2026-10-24 16:40 — Pilot: Go / stop and best pilot checkpoint

**status = go** (arms beating base (diff CI lo > 0): B, CU, CT, D). Diffs vs base_chunked on dev-mirror: A +9.5 [-10.4, +29.1]; B +26.5 [+12.2, +46.0]; CU +39.9 [+23.5, +64.8]; CT +41.2 [+24.4, +67.1]; D +34.4 [+19.7, +57.5].

best = **B** (mirror BTRM 86.2) among guard-passing non-F arms B; guard failures: A, CU, CT, D.

## 2026-10-24 16:41 — Schedule (user decision): 1 main round, full final

The user set a deadline of Oct 26, so the main job runs with MAIN_ROUNDS=1 (env override; recipe.json
is unchanged), and final.sh runs in full, including GRM. Recipe T is not allowed; if the pilot picks it,
it is replaced by M (see the T -> M entry, if any). SWEEP and REWEIGHT stay at their defaults, so
round 1 (the last round) trains the main arm, the half-lr arm and the alt arm. ENCODE_BATCH=1 for
every job (see the G0a entry).

## 2026-10-26 02:29 — main_r1: source re-weighting (round 1)

Alt arm (teacher_up): teacher share of the mix hours 0.1367 (main) → 0.274 (alt), same learning rate; records main 37682 vs alt 37605.
no alt checkpoint passed the guard rails.
Not adopted: the main mix stays.

## 2026-10-26 02:29 — main_r1: checkpoint choice

no checkpoint of this round passed the guard rails

- main_r1_lr2e-05_e1: 74.8% guards FAIL
- main_r1_lr2e-05_e2: 102.1% guards FAIL
- main_r1_lr2e-05_e3: 98.3% guards FAIL
- main_r1_lr1e-05_e1: 54.1% guards FAIL
- main_r1_lr1e-05_e2: 93.1% guards FAIL
- main_r1_lr1e-05_e3: 103.1% guards FAIL
- main_r1_alt_lr2e-05_e1: 74.7% guards FAIL
- main_r1_alt_lr2e-05_e2: 104.7% guards FAIL
- main_r1_alt_lr2e-05_e3: 96.8% guards FAIL

## 2026-10-26 07:01 — User decision: benchmark a runner-up (main_r1 lr1e-05 epoch 3, no GRM)

Main round 1: all 9 checkpoints failed the WER guard, so round_select kept pilot B (86.2% dev-mirror) as the final, unchanged. The failures come from dev-fresh (freeform) WER, e.g. main_r1_lr1e-05_e3: 0.129, base 0.103, alt e2 0.351. On dev-mirror (read speech, like the 494) their WER is within the guard (0.046-0.053 vs base 0.042) and BTRM is 102-105%.

The user chose to also benchmark main_r1 lr1e-05 checkpoint-epoch-3 (dev-mirror 103.1%, +43.4 [+28.9, +65.5]; best balance of fresh WER) on the 494 as RUNNER_UP, with FINAL_SKIP_GRM=1 for that run only, to fit the Oct 26 deadline. The official final checkpoint remains pilot B. The runner-up is reported side by side for the DPO-stage decision.

Note for Daksh: main.sh exits 1 at `realpath -m ""` when a round has no guard-passing checkpoint (round_summary best_ckpt = null). The supervisor's retry continued into final.sh, because the round was already decided. Its checkpoint cleanup never ran, so all main_r1 checkpoints are still on disk.
