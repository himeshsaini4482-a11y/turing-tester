# Pilot report

Status: **GO**. Decoding: chunked. Recipe **M**, freeform tag **U**, F1 include **False**, EARS in mix True, teacher 10 GPU-h per round, oversample 1.14.

BTRM % = mean TTS reward / mean human reward × 100 on the same rows (human = 100). CIs: 95%, items resampled (10000 draws), paired for differences. Base = `base_chunked`.

## Arms vs base

| arm | set | BTRM % [CI] | Δ vs base [paired CI] | WER | sim | UTMOS | guards |
|---|---|---|---|---|---|---|---|
| base | mirror | 59.7 [41.2, 84.2] | — | 0.042 | 0.955 | 3.660 | — |
| base | fresh | — | — | 0.103 | 0.920 | 3.418 | — |
| A | mirror | 69.3 [46.2, 98.7] | +9.5 [-10.4, +29.1] | 0.036 | 0.954 | 3.846 | FAIL wer✗ |
| A | fresh | — | — | 0.234 | 0.896 | 3.257 | FAIL wer✗ |
| B | mirror | 86.2 [66.8, 117.3] | +26.5 [+12.2, +46.0] | 0.021 | 0.945 | 4.039 | pass |
| B | fresh | — | — | 0.044 | 0.910 | 3.844 | pass |
| CU | mirror | 99.6 [78.7, 135.1] | +39.9 [+23.5, +64.8] | 0.060 | 0.959 | 3.984 | FAIL wer✗ |
| CU | fresh | — | — | 0.150 | 0.905 | 3.383 | FAIL wer✗ |
| CT | mirror | 100.9 [79.6, 136.9] | +41.2 [+24.4, +67.1] | 0.047 | 0.960 | 3.966 | FAIL wer✗ |
| CT | fresh | — | — | 0.201 | 0.905 | 3.364 | FAIL wer✗ |
| D | mirror | 94.1 [72.7, 129.7] | +34.4 [+19.7, +57.5] | 0.036 | 0.959 | 3.996 | FAIL wer✗ |
| D | fresh | — | — | 0.175 | 0.910 | 3.385 | FAIL wer✗ |
| F | mirror | 105.9 [83.8, 143.7] | +46.1 [+28.9, +73.0] | 0.052 | 0.969 | 3.938 | FAIL wer✗ (not selectable) |
| F | fresh | — | — | 0.158 | 0.903 | 3.327 | FAIL wer✗ (not selectable) |

Guards (mirror + fresh rows vs base): WER ≤ base + 0.01, sim ≥ base − 0.02, UTMOS ≥ base − 0.10. `?` = metric missing (unchecked).

## Pilot best checkpoint

`runs/sft/ckpt/pilot/B/checkpoint-last` (arm B, mirror BTRM 86.2)

## Decisions

### Q2 freeform tag

**freeform_tag = U.** diff(CT − CU) on dev-mirror = +1.3 [-10.3, +12.8]; CI lo ≤ 0 → U (simpler). dev-fresh diff(CT − CU) = — (secondary). Rule: INTERFACES §4 Q2 (T only if diff CI lo > 0; this replaces PLAN §8's 'CT lo above CU mean').

### Recipe

**recipe = M** (Δ1 CI contains 0 → M); teacher GPU-h per main round 10, teacher_oversample 1.14, ears_in_mix True.

- Δ1 = diff(B − A), teacher vs EARS hour for hour, dev-mirror: +16.9 [-2.7, +43.0] (dev-fresh —)
- Δ2 = diff(D − CU), teacher on top of full EARS, dev-mirror: -5.5 [-19.5, +7.7] (dev-fresh —)
- Rule: INTERFACES §4 recipe row.
- teacher_oversample for M: {"n_ears_records": 16430.0, "n_teacher_pilot": 1719.0, "pilot_teacher_gpu_h": 2.833833333333333, "projected_teacher_after_round1": 7784.988354996178, "target_share": 0.35, "raw": 1.1364080039047688, "clamped_to": [1.0, 5.0]}

### Q5 F1 (styled readings of own test texts)

**f1_include = False.** F1_POLICY=auto but F, CU or the probe items are missing: safe default exclude.

- Memorisation estimate, diff(F − CU) on the 36 dev-mirror items whose text the dev-leak-probe records cover: +4.2 [-4.6, +13.0]
- … on the other 40 mirror items: —
- dev-fresh diff(F − CU): —
- F is excluded from best-checkpoint selection (its dev-mirror score is contaminated by design).

### Go / stop and best pilot checkpoint

**status = go** (arms beating base (diff CI lo > 0): B, CU, CT, D). Diffs vs base_chunked on dev-mirror: A +9.5 [-10.4, +29.1]; B +26.5 [+12.2, +46.0]; CU +39.9 [+23.5, +64.8]; CT +41.2 [+24.4, +67.1]; D +34.4 [+19.7, +57.5].

best = **B** (mirror BTRM 86.2) among guard-passing non-F arms B; guard failures: A, CU, CT, D.

## Pilot teacher round

Generated 4000, accepted 1719 (yield 0.430); rejected: no_eos 2, rate 65, wer 208, sim 883, btrm 1123. Accepted audio 2.7 h from 2.8 GPU-h.
