# VoiceGym online DPO (after SFT)

Online, iterative DPO for TTS following the KL paper's Algorithm 1 (`docs/KL_divergence_paper.pdf`):
each round, the current policy speaks fresh prompts K times, SpeechJudge-BTRM ranks the samples, the
best and the worst valid sample form a pair, and the policy takes Eq. 2 steps against a frozen π_ref,
while the paper's KL quantities (u_w, u_l, M, k̂, D̂, β) are tracked every step.

Models (one adapter each, `dpo/adapters/`; the contract is `adapters/base.py`):

| model | π_ref | training | status |
|---|---|---|---|
| `moss_nano` | the SFT deliverable `runs/sft/best/final` | full fine-tune, fp32 master + bf16 autocast, frozen fp32 copy as π_ref | written, untested on GPU |
| `orpheus` | `unsloth/orpheus-3b-0.1-pretrained` | LoRA; π_ref = adapters disabled | written, untested on GPU |

Qwen3-TTS is out of scope (user decision, 2026-10-23); Fish Drama-3 int4 has no training path.

## 1. Flow

```
scripts/dpo/run.sh
  setup        model venv + tensorboard                                   scripts/dpo/setup.sh
  texts        Qwen 3.5 9B writes texts.n sentences                       dpo/prompts.py texts
  prompts      text x voice (non-dev EARS speakers), shuffled, per round  dpo/prompts.py build
  gates        D1-D7 on the GPU, stop on failure                          dpo/gates.py
  eval r00     dev eval of pi_ref                                         scripts/dpo/eval.sh
  round r = 1..rounds:
    rollout    K samples per prompt, batched, per-batch length cap        dpo/rollout.py
    metrics    WER / speaker sim / UTMOS per sample                       benchmark/metrics.py --tts-only
    btrm       BTRM reward, only for samples that can still form a pair   dpo/score_btrm.py --round
    pairs      gates, best-vs-worst, ties skipped, sigma_r^2 and beta*    dpo/pairs.py
    train      Eq. 2 + Alg. 1 lines 8-12        (exit 3 = paused for you) dpo/train.py
    eval       dev eval vs r00, SFT guard rails                           scripts/dpo/eval.sh
    select     best round, stop rule            (exit 10 = stop)          dpo/rounds.py select
  final        494 benchmark of the best round and of pi_ref              scripts/dpo/final.sh
  report       runs/dpo/REPORT.md                                         dpo/rounds.py report
```

Every step is resumable (markers in `runs/dpo/.done/`); after any crash or worker stop, launch the
same command again.

## 2. How the 9B is connected

`dpo/prompts.py texts` calls `sft/gen_texts.generate` unchanged, which calls the 9B through
`sft/tag_weaknesses.llm` → `scripts/qwen9b_client.py chat()` (`POST /chat/completions` on
`VOICEGYM_QWEN_BASE_URL`, bearer token from `.env`). So the DPO texts get the SFT rules for free:
8–40 words, 1–3 sentences, no digits, deduplicated against every SFT text and the 19 benchmark
texts, and no benchmark sentence copied verbatim. The weakness mix comes from the newest SFT main-round
diagnosis on the dev speakers (`--report auto`), never from the 494. The endpoint answered from the
UAI host and from the worker (SFT gate G0c) on 2026-10-23; if the worker cannot reach it, run the
`texts` step on the host (standard library only) and copy `runs/dpo/texts.jsonl` over.

## 3. When to do what

| when | what | who |
|---|---|---|
| SFT running (until ~Oct 26) | nothing on the GPU; code review and CPU tests only | agents |
| SFT `final.sh` done, `runs/sft/best/final` exists | `KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/dpo/run.sh"` | you / Claude |
| `gates` fails | debugger agent reads `runs/dpo/gates.json`, fixes, re-run | debugger |
| run.sh exits 3 (paused) | read `runs/dpo/r01/beta_diagnostic.md`; either fill the USER_REVIEW values in `hparams.json` or check `dpo/sweep.json`, set `"approved": true` and run `scripts/dpo/sweep.sh`; stop the worker while you decide | **you** |
| sweep done | read `runs/dpo/sweep/SWEEP_REPORT.md`, copy the chosen values into `hparams.json`, re-run run.sh | **you** |
| each round | observability agent checks TensorBoard / logs; inference agent checks throughput | agents |
| run.sh done | `runs/dpo/REPORT.md`; stop the worker | Claude |

## 4. Watching a run

Live (only the UAI host can SSH to the worker, so two tunnels):

```bash
# on the UAI host, in the repo: start TensorBoard on the worker (not in tmux: tmux keeps it awake)
source scripts/gpu/config.sh
gpu_ssh 'cd voicegym && setsid nohup /opt/dlami/nvme/moss_nano/venv/bin/tensorboard --logdir runs/dpo/tb \
  --host 127.0.0.1 --port 6006 --reload_interval 30 >/tmp/tb.log 2>&1 </dev/null &'
ssh "${SSH_OPTS[@]}" -f -N -L 127.0.0.1:16006:127.0.0.1:6006 ubuntu@$GPU_HOST
# on your laptop
ssh -N -L 6006:127.0.0.1:16006 ubuntu@<UAI host>     # then open http://localhost:6006
```

After a run (or a stop), pull only the small files and view them anywhere tensorboard is installed:
`rsync -az -e "ssh ${SSH_OPTS[*]}" --prune-empty-dirs --include='*/' --include='events.out.tfevents.*'
--include='*.json' --include='*.jsonl' --include='*.md' --include='*.csv' --include='*.tsv' --include='*.log'
--exclude='*' ubuntu@$GPU_HOST:voicegym/runs/dpo/ runs/dpo/`.

TensorBoard event files are in `runs/dpo/tb/` on the worker (`run/` for the rounds, `sweep_<tag>/`
per sweep setting). Tags: `paper/{u_w,u_l,margin,k_hat,d_hat,beta,logratio_var}` and
`train/{loss,accuracy,logp_chosen,logp_rejected,grad_norm,lr,entropy_mean,entropy_first}`; the same
rows are in `runs/dpo/r<rr>/train_log.jsonl` (plus step time, peak memory, per-frame log-probs,
min log π(y_l), max |margin|, `entropy/<codebook>`, and `paper/epsilon` reference lines). GPU telemetry:
`runs/dpo/logs/gpu.csv` (every 10 s); stage start/end/exit code: `runs/dpo/logs/stages.tsv`; per rollout
batch: `r<rr>/rollout_log.jsonl`; stage times: `runs/dpo/timings.json`; decisions: `runs/dpo/DECISIONS.md`;
paused state: `runs/dpo/STATUS.json`. `pairs_summary.json` compares chosen vs rejected WER / sim / UTMOS /
seconds per word (a reward-hacking signal); a round whose guard rails could not be checked counts as failed.
Why TensorBoard and not SageMaker: `.claude/agents/observability.md`.

## 5. Pipeline defaults (not USER_REVIEW; change them in `hparams.json` if needed)

1. **Reward = BTRM.** Its rewards are Bradley–Terry logits, the scale the paper's σ̄²_r assumes, and
   it is strongest on TTS-vs-TTS comparisons (72.7%, SpeechJudge Table 11), which is what every pair
   is. GRM stays out of training and checks the final model independently (BTRM gaming guard).
2. **Decoding during rollouts = the benchmark's** (`rollout.sampling = deployment`), so the policy
   improves on the outputs the benchmark measures. `on_policy` samples π_θ exactly (Alg. 1 line 3)
   but at different settings than the benchmark decodes. The log-probs are always at temperature 1
   over the sampler's support (gates.py D1 checks this).
3. **K = 4, best vs worst, tie gap 0.5** (σ(0.5) = 0.62 preference), invalid = no stop token,
   undecodable, WER > 0.20, speaker sim < 0.85, or 0.15–1.0 s per word violated. Base MOSS on dev:
   WER 0.042, sim 0.955.
4. **Rounds = 3, 1,000 fresh prompts each, patience 1.** Round 1 samples from π_θ1 = π_ref, so its
   σ̄²_r is the paper's Thm 3.2 quantity.
5. **Batched update.** R pairs per round, `batch_pairs` per step, which departs from Alg. 1's
   one pair per step (`batch_pairs = 1, grad_accum = 1` restores it).

## 6. Your values (USER_REVIEW in `hparams.json`)

`train.py` exits 3 until every required field is set; `dpolib.check_user_review` lists the missing
ones. Claude's proposal, all of it in `dpo/sweep.json` with `"approved": false`:

| setting | proposal | why |
|---|---|---|
| optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0 | MOSS's own SFT betas |
| alternatives | RMSprop (alpha 0.99, 150 warmup steps); SGD + momentum 0.9 | RMSprop is what the original DPO paper used; SGD is the paper's Alg. 1 line 11 and §4 regret analysis. The SGD lr 1e-4 is a guess, unmeasured |
| lr | 1e-6 (try 5e-7 if unstable) | 10× below the SFT lr 1e-5 |
| clipping | max_grad_norm 1.0 | §4 assumes bounded gradients (‖g‖ ≤ 2βG) |
| batch | 16 pairs, 1 epoch per round, constant lr, 20 warmup steps | each fresh pair used once (online) |
| β fixed | 0.05, 0.1, 0.3 | brackets the usual 0.1 |
| β dual ascent | β₀ = β* = √(σ̄²_r / 2ε) from round 1, range [0.01, 1], η_β 0.01 | the paper's H2 + Alg. 1 line 12 |
| KL | ε from the β* table, EMA α 0.1, k̂ per frame, on_violation none | per-sequence log-ratios grow with clip length |

Sweep plan: stage 1 = AdamW × {0.05, 0.1, 0.3, dual} (4 runs), stage 2 = RMSprop and SGD at the
stage-1 winner (2 runs). Each run ≈ training (~10–20 min, estimate) + dev eval (~40 min, measured
for the SFT base eval). The pick is yours.

## 7. Files

| path | what |
|---|---|
| `dpo/hparams.json` | all settings; USER_REVIEW sections are yours |
| `dpo/sweep.json` | proposed sweep settings (needs `"approved": true`) |
| `dpo/dpolib.py` | paths, I/O, hparams check, paper diagnostics |
| `dpo/adapters/` | `base.py` contract, `moss_nano.py` (+ `dpo/nano.py` internals), `orpheus.py` |
| `dpo/{prompts,rollout,score_btrm,pairs,train,gates,rounds}.py` | the stages above |
| `scripts/dpo/*.sh` | `run.sh`, `sweep.sh`, `eval.sh`, `final.sh`, `setup.sh`, `env.sh` |
| `requirements-dpo-orpheus.txt` | `.venv-dpo-orpheus` (MOSS uses the SFT MOSS venv) |
| `.claude/agents/{coder,debugger,observability,inference}.md` | the four agent specs |
| `runs/dpo/` | outputs (worker EBS); candidate wavs on the NVMe |

`docs/moss_tts_nano_online_dpo.md` is the earlier design study (trl-based, β never set). This
pipeline replaces its skeleton: a plain PyTorch loop, because MOSS/Orpheus would override almost all
of trl's DPOTrainer, and trl has no hook for the paper's adaptive β. `requirements-dpo.txt` (trl) is
not used by this pipeline.

## 8. Review status (2026-10-23) and open items

Reviewed by the debugger, observability and inference agents (specs in `.claude/agents/`). CPU tests
with tiny random models proved: training log-prob = the sampler's own log-prob = the frame-by-frame
recomputation; padding invariance; u = 0 and loss = log 2 at round start; Alg. 1 D̂/β bookkeeping in
both β modes; resume, rollback, stop_round, sweep isolation; Orpheus LoRA log-probs, π_ref and adapter
reload; the vectorised repetition penalty is bit-identical to the pinned one. Not provable without the
GPU: real-weight bf16 gaps, codec/BTRM behaviour, memory and speed (gates D1-D7 on the worker).

Open, in order (none blocks the first run):
1. `scripts/gpu_run.sh` copies all of `runs/` back to the host (tens of GB) and truncates `runs/job.log`
   on every launch. It is shared with the running SFT, so it is left unchanged until SFT ends; until then
   launch DPO the way the SFT supervisor does (pull only small files).
2. MOSS rollout speed beyond the repetition penalty: remove the remaining per-frame host syncs, then CUDA
   graphs with a static KV cache (est. several x more; needs new equivalence gates). Measure D6 first.
3. Orpheus: vLLM for rollouts (est. ~10x vs HF); a gate for the bf16-merge gap of the rollout/eval
   checkpoint vs base + adapter; run D1 with fp32 base weights if bf16 D1 fails by < 1 nat.
4. BTRM batching (est. 2-3x; check |Δr| and identical pair decisions first).
5. Optional, needs your OK: a CloudWatch alarm that stops the worker after 60 min of < 5 % CPU (~$0.10/month).
