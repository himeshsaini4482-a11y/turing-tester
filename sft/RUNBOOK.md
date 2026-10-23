# Runbook for Claude on the VM: running the VoiceGym SFT pipeline

You are Claude Code running on the **UAI server**: the machine that holds this repo, the AWS
CLI credentials and `~/.ssh/voicegym-gpu.pem`. You do not train anything on this machine. You
drive a disposable AWS GPU worker (1× L40S 48 GB, g6e.xlarge, ~$2.24/h) with the `scripts/gpu_*.sh`
scripts, check results between stages, and report to the user.

Read `sft/PLAN.md` (why) and `sft/INTERFACES.md` (the binding contract: files, formats,
decision rules) first. This file is the *how*.

---

## 0. What we are doing (one paragraph you can repeat to the user)

We fine-tune the 100M-parameter **MOSS-TTS-Nano** voice-cloning TTS so that it scores higher on
**SpeechJudge-BTRM** on the locked EARS benchmark (26 speakers × 19 sentences = 494). There are
two candidate training-data sources:
- **Real studio speech from EARS:** everything except the 494 benchmark recordings and non-verbal sounds.
- **Teacher audio from Orpheus-3B:** recordings of sentences that **Qwen3.5-9B** writes to
  target weaknesses found by **SpeechJudge-GRM**.

First a **test round** (the pilot) settles every open choice by experiment:
- decoding: whole text or MOSS's 50-token chunks;
- whether freeform speech is tagged;
- teacher vs EARS, hour for hour;
- teacher on top of EARS;
- whether to train on the benchmark speakers' styled readings of their own test sentences (F1).

The pilot writes `runs/sft/pilot/recipe.json`. The **main workflow** then runs rounds of
*diagnose → targeted texts → teacher audio → SFT → evaluate*, as the recipe says: EARS-led (H),
mixed (M) or teacher-led (T). Checkpoints are chosen on held-out dev speakers. The full
benchmark runs once at the end. Selecting among MOSS's own samples is deliberately *not* part of
this; that is the DPO stage that follows. Budget: 4 days of one GPU.

**The score:** BTRM % = mean BTRM reward of the TTS clips ÷ mean BTRM reward of the human clips
× 100, on the same sentences (human = 100%). Always quote it with its CI. Never divide
`human_parity_score` by the human reward: they are different scales.

---

## 1. Hard rules

1. **Do not run `scripts/gpu_down.sh` until the project is finished** or the user says so. It
   terminates the worker and destroys everything on it except what was copied back into `runs/`.
   To pause and save money, *stop* the instance instead (§5). Stopping keeps its disk.
2. **Never tune on the benchmark.** The 494 set (`benchmark/manifests/ears_regular_26x19.csv`)
   is only measured (base in the pilot, base and final in `final.sh`), never used to choose
   anything. All choices use dev-mirror/dev-fresh. Never edit the locked manifest.
3. **Never commit `.env`**, tokens, audio, checkpoints or `runs/**` files beyond what
   `.gitignore` allows. Never paste the 9B API key into logs or chat.
4. **Code fixes:** the user writes code on another machine and pushes to `main`. If you must fix
   something here, commit it to branch `sft-vm-fixes` (never `main`), push that branch, and tell
   the user exactly what changed and why. Keep fixes minimal.
5. **Report honestly.** Give numbers with their CIs, and say when a step failed or was skipped.
   Never present dev results as benchmark results, or BTRM as human preference.
6. Decisions are made by the code (`sft/decide.py`, `sft/round_select.py`) and logged in
   `runs/sft/DECISIONS.md` with their numbers. If you override one (by editing `recipe.json`),
   add your own entry there saying what and why, and tell the user.

---

## 2. Preconditions (check before anything else)

```bash
cd ~/voicegym            # or wherever the repo lives on this server
git checkout main && git pull origin main
ls scripts/sft/pilot.sh scripts/sft/main.sh scripts/sft/final.sh sft/INTERFACES.md
python3 -m compileall -q benchmark scripts sft && for f in scripts/sft/*.sh; do bash -n "$f"; done
```
- If the pipeline files are missing, **stop and tell the user** the code has not been pushed.
  Do not write the pipeline yourself.
- `.env` exists with `VOICEGYM_QWEN_API_KEY` set (`grep -c VOICEGYM_QWEN_API_KEY= .env`; do not
  print it). `gpu_run.sh` rsyncs it to the worker. Smoke-test it here:
  `python3 scripts/qwen9b_client.py "Reply with exactly: QWEN_OK"`.
- `aws sts get-caller-identity` works, and `~/.ssh/voicegym-gpu.pem` exists.
- `bash scripts/gpu_status.sh`: no worker should already be running or stopped. If one is
  stopped, it may hold results: follow the message `gpu_up.sh` prints, don't delete it blindly.
- Optional knobs, set as env vars in the job command: `F1_POLICY=auto|include|exclude` (default
  auto), `TEACHER_GPU_HOURS` (pilot default 5), `PILOT_TEXTS` (default 4000), `ARMS` (default
  `B A CU CT D F`).

---

## 3. How to launch and watch a job

The worker runs **one** job at a time, in a tmux session named `vg`. `gpu_run.sh` launches (or
reuses) the worker, rsyncs the code, starts the job, streams its log, waits, then rsyncs
`runs/` back. Jobs take many hours, longer than your 10-minute tool timeout, so always launch in
the background and poll:

```bash
mkdir -p runs/sft/logs
nohup env KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/pilot.sh" \
  > runs/sft/logs/pilot.server.log 2>&1 &
```

**Watching:**
- `tail -n 50 runs/sft/logs/<job>.server.log` shows the job log streamed from the worker. Each
  pipeline step prints `== [HH:MM:SS] <step>` when it starts and `== [done] <step> in Ns` when
  it ends.
- `bash scripts/gpu_status.sh` shows whether the worker is alive, GPU memory, and the current job.
- `source scripts/gpu/config.sh && gpu_ssh 'tail -n 30 voicegym/runs/job.log; ls voicegym/runs/sft/.done | tail; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv'`
- A job is finished when the server-side `gpu_run.sh` process exits. Its last line says
  `Job finished with exit code N`, and `runs/` on this server has been refreshed.
- Poll every 15–30 min. Use the scheduling or monitor tools you have rather than tight sleep loops.

**Resuming:** every pipeline script is resumable. Finished steps leave a marker in
`runs/sft/.done/<step>`, and re-running the same command skips them. To redo one step, delete its
marker (and its outputs if they are corrupt), then re-run.

`KEEP_UP=1` keeps the worker running between jobs (it also disables the 30-min idle
auto-stop). That is intended: the local NVMe caches (EARS audio, judge checkpoints, MOSS venv)
survive between jobs, and re-creating them costs ~1 h.

---

## 4. The three jobs, in order

Durations are estimates. After the first hours, re-estimate from `runs/sft/timings.json` (one
entry per finished step) and tell the user if the 4-day schedule no longer fits.

### Job 1 — test round: `scripts/sft/pilot.sh` (~20–30 h)
```bash
nohup env KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/pilot.sh" > runs/sft/logs/pilot.server.log 2>&1 &
```
What it does, in order (INTERFACES §4):
1. **Setup and gates.** MOSS venv, pinned repo + patch, then gates:
   - G0a: training prompt = inference prompt, token for token;
   - G0b: checkpoint round-trip;
   - G0c: 9B reachable.
2. **EARS download + inventory**, dev-mirror set.
3. **Q1 decoding:** base MOSS on dev-mirror, chunked vs whole text → `runs/sft/decoding.json`.
   Base 494 benchmark under both settings (BTRM + metrics, no GRM) → `runs/moss_base_{chunked,whole}/`.
4. **EARS data:** Whisper alignment, chunking to the chosen size, dev-fresh set, codec
   encoding. Base full dev eval.
5. **Diagnosis:** GRM on base, 9B tags the weaknesses, 9B writes ~4000 targeted texts, Orpheus
   speaks as many as fit in 5 GPU-h (time box). Then the cheap filters (end-of-speech, rate,
   Whisper WER, voice match) pick the candidates, and only those go through the student and
   BTRM (`teacher/prefilter_summary.json`, then `teacher/filter_summary.json`).
6. **Six training arms** (B, A, CU, CT, D, F), each trained and evaluated on the dev sets.
7. `sft/decide.py pilot` → `runs/sft/pilot/recipe.json`, `pilot_report.md`, `DECISIONS.md`.

**Check as it goes** (read the files after the job, or over ssh while it runs):
- `runs/sft/gates/*.json`:
  - **G0a or G0b fail → the job stops. Report the gate's `details` to the user;** nothing
    downstream is valid.
  - G0c failing is non-fatal. It means the worker can't reach the 9B; see §5.
- `runs/sft/data/inventory_summary.json`: 494 benchmark files and 1,404 F1 files flagged, 4 dev
  speakers.
- `runs/sft/data/build_summary.json`: total training hours (expect ~60–80 h). Under 40 h →
  look at the drop reasons before trusting the pilot.
- `runs/sft/pilot/teacher/prefilter_summary.json` (available right after the Orpheus step) and
  `filter_summary.json`: yield per filter rule. Under 20% → tell the user (voice match is the
  likely cause; Orpheus averaged 0.85 on one speaker). If the user then wants a lower voice-match
  bar, they set `TEACHER_SIM_MIN=<value>` on the job command; it is recorded in the summaries.
  Never change the threshold yourself.
- `runs/sft/pilot/*/mix_summary.json`: hours per arm; arm A ≈ arm B's hours.
- `runs/sft/ckpt/pilot/*/train.log`: loss going down, samples/s (training throughput).

**When it ends, report to the user:**
- base 494 numbers under both decodings, and the decoding chosen;
- the weakness table;
- every arm's dev-mirror BTRM % with CI vs base, and the guard rails;
- the five decisions with their numbers (`pilot_report.md`), including the F1 memorisation
  estimate;
- the recipe.

Exit code 3 means the pilot decided **STOP** (no arm beat base). Do not start the main job. Report
and wait for the user.

### Job 2 — main workflow: `scripts/sft/main.sh` (~40–60 h, plus one extra training run and its evals per round for the re-weighting check)
```bash
nohup env KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/main.sh" > runs/sft/logs/main.server.log 2>&1 &
```
Runs `main_rounds` rounds (from the recipe, default 2). Each round:
1. GRM diagnoses the current best;
2. the 9B writes texts for what is still weak, and Orpheus speaks them (recipe budget: H 4,
   M 10, T 20 GPU-h);
3. builds the mix and retrains from base;
4. **re-weighting check:** also trains an `alt` arm whose mix gives the smaller source (teacher or
   EARS) a larger share of the hours, at the same record count and lr (INTERFACES §4.1). If alt
   beats the main arm, later rounds use its share (`runs/sft/main/mix_state.json`). Cost: about
   one more main-arm training run plus its 3 dev evals per round. Use the pilot CU arm's
   `timings.json` entries as the first estimate, and re-check after round 1;
5. evaluates every epoch of every arm;
6. `sft/round_select.py` keeps the best (main or alt) and stops early if it isn't better than the
   previous best.

The last round also trains at half the learning rate (`SWEEP=1`). When the rounds finish, it calls
`final.sh` (job 3) itself unless `SKIP_FINAL=1`.

After each round (`runs/sft/main_r<k>/round_summary.json` + `DECISIONS.md`), tell the user:
- the best checkpoint vs the previous best vs base, on dev-mirror with CI;
- the re-weighting result (`round_summary.json["reweight"]`): teacher share main → alt, alt − main
  with CI, adopted or not. If DECISIONS.md suggests a different teacher budget, pass the
  suggestion on and let the user decide. Don't change `TEACHER_GPU_HOURS` yourself;
- how the confirmed weaknesses changed;
- the teacher yield;
- the time left.

### Job 3 — final benchmark: `scripts/sft/final.sh` (~8–20 h, mostly GRM)
Runs automatically at the end of job 2. To run it alone:
```bash
nohup env KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/final.sh" > runs/sft/logs/final.server.log 2>&1 &
```
Steps:
1. Copies the chosen checkpoint to `runs/sft/best/final/`.
2. Runs the full 494 (GRM + metrics + BTRM) for base (`runs/sft_final_base/`) and the chosen
   checkpoint (`runs/sft_final/`), under the pilot's decoding setting.
3. Writes `runs/sft/REPORT.md`.

Add `RUNNER_UP=<ckpt dir>` to also benchmark a second checkpoint if time allows.

**Done when** `runs/sft/REPORT.md` exists. Then:
1. Read `REPORT.md` fully and check its numbers against the summary JSONs.
2. Send the user the headline table (BTRM % + CI, the diff vs base with CI, parity, ratio, wins,
   GRM closeness, WER, sim, UTMOS: base vs final) and the **Disclosure** section verbatim.
3. Ask the user before shutting down. With their OK: `bash scripts/gpu_down.sh`, then confirm
   `bash scripts/gpu_status.sh` shows no worker.

### Budget guard
- Start time = first `gpu_up`. The 4-day budget is 96 h of worker time (~$215).
- The teacher GPU-hour budget covers **Orpheus rendering only**. After it, every rendered clip
  goes through Whisper and WavLM, and the clips that pass the cheap rules also through MOSS
  student generation and BTRM. For recipe T (20 h) that can add up to a similar amount per
  round, so budget a round as roughly 1.5–2× its teacher hours and re-check after the pilot
  (`timings.json` has every teacher step).
- `runs/` (checkpoints, eval wavs, teacher wavs) grows to tens of GB and is rsynced back after
  every job. Check `df -h ~` on the worker and the server after the pilot.
- If more than 6 h behind schedule, shrink in this order:
  1. `MAIN_ROUNDS=1`;
  2. `SWEEP=0`;
  3. `REWEIGHT=0` (drops the alt arm);
  4. a lower `TEACHER_GPU_HOURS`;
  5. no `RUNNER_UP`;
  6. `FINAL_SKIP_GRM=1` (the final 494 runs still score BTRM, WER, sim and UTMOS; only the
     slow GRM closeness column is dropped).

  In the pilot itself the lever is `ARMS` (e.g. `ARMS="B A CU D F"` drops the freeform-tag arm
  CT; the tag then defaults to U).

  Set these as env vars in the job command; the recipe is not changed.
- Never drop the final 494 run of the chosen checkpoint. That is the deliverable.

---

## 5. Recovery

| Situation | What to do |
|---|---|
| `gpu_run.sh` on this server died, job still running on the worker | Re-run the same `gpu_run.sh` command. It sees the tmux session and waits for it instead of starting a new one. |
| A job failed | Read the tail of `runs/job.log` (copied back) to find the failing step. Fix the cause and re-run the same command; finished steps are skipped. If the fix is code, see Hard rule 4. |
| Worker *stopped* (idle stop or you paused it) | `source scripts/gpu/config.sh; aws ec2 start-instances --region $GPU_REGION --instance-ids $GPU_INSTANCE_ID`, wait for running, then re-run the job. The NVMe was wiped. Steps whose outputs lived only there are redone automatically or fail loudly; delete those steps' markers in `runs/sft/.done/` if they fail. Durable data (codes, teacher audio, checkpoints, evals) is in `runs/sft/` on the EBS disk. |
| Worker *terminated* / lost | Launch the job again with `PUSH_RUNS="sft moss_base_chunked moss_base_whole"` so prior results are pushed to the new worker. The EARS audio is re-downloaded; delete the `.done` markers of steps that need it (inventory, align, build) only if they fail. |
| Need to pause (e.g. waiting > 2 h for the user) | Make sure no job runs (`gpu_status.sh`), then `aws ec2 stop-instances ...`. Resume as above. |
| 9B unreachable from the worker (gate G0c failed) | The 9B steps are stdlib-only and run fine here. When the job stops at the tagging or text step, run that step's command here: `python3 sft/tag_weaknesses.py ...` / `python3 sft/gen_texts.py ...` with the arguments shown in the job log. Then `touch runs/sft/.done/<step>` and re-launch with `PUSH_RUNS=sft`. |
| Need to redo main round k (e.g. after a code fix; only the last finished round) | In this order: (1) write `prev_best` from `runs/sft/main_r<k>/round_summary.json` into `runs/sft/main/best.json`. (2) In `runs/sft/main/mix_state.json`, set `teacher_share`, `ears_in_mix` and `ears_weight` back to `decided["main_r<k>"]["state_before"]`, then delete that `decided` entry. (3) Delete the round's `round_summary.json`, its `.done` markers `main_r<k>_train_*` and `main_r<k>_eval_*` (keep `_diagnose`/`_teacher` unless they must be redone too; the teacher costs GPU-hours), its mixes in `$SCR/mixes/` and `runs/sft/ckpt/main_r<k>/`. (4) Re-run `main.sh`. |
| CUDA OOM | Steps run one model at a time. Lower batch sizes: `BATCH=8 ACCUM=4` for training; `--batch-size` for the teacher. |
| cuDNN `SUBLIBRARY_LOADING_FAILED` | Something left `LD_LIBRARY_PATH` set. `scripts/sft/env.sh` unsets it; check any command you added. |
| 9B endpoint 401 | Check `.env` was synced. Rotate the token with the admin if it leaked. |

---

## 6. What to tell the user, and when

| When | Message contents |
|---|---|
| Gates done | gate results; G0a/G0b details if they failed |
| Pilot done | base 494 under both decodings + choice; weakness table; arm table with CIs; decisions incl. F1 memorisation estimate; recipe; teacher yield; time used |
| Each main round | best vs previous best vs base on dev (CI); weakness shares over rounds; teacher yield; time left |
| Final | headline table base vs final on the 494; Disclosure section; checkpoint and report locations; ask before `gpu_down` |
| Any time | anything blocked, any gate failed, schedule slipping > 6 h |
