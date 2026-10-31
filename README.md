<div align="center">

# 🏋️ VoiceGym

### The gym where voice models put in their reps.

*Built for the Bolna AI Hackathon*

</div>

---

Okay, real talk.

Most hackathon projects this weekend will look something like this: take a voice API, call it, put a nice UI on top, ship it. And honestly? That's fine. We love a good wrapper.

But nobody is asking the obvious question. **Who makes the model behind that API better?**

Frontier labs and voice AI teams spend months trying to figure out *why* their TTS model still sounds like a robot reading out a parking ticket. They guess. They collect a random pile of data. They retrain. Then they guess again.

We built VoiceGym so they can stop guessing.

VoiceGym is a gym for voice models. Your model walks in, gets tested against real human voices, gets told exactly which muscles are weak, trains on exactly those, and comes back for another round. Every round the model gets better, and every round the next batch of training data gets smarter, because it's aimed at whatever is still broken. The loop feeds itself. It's recursive self-improvement, but for how your model *sounds*.

> 🗣️ **Coach says:** "Bro. You don't fix a weak voice with random reps. You find the weak spot and you hit it. Again. And again. And again."

## The 30 second version

VoiceGym does two things:

1. **🏟️ The Benchmark Arena.** Plug in your TTS model. It reads the same sentences a real human read, in that human's cloned voice, and a judge model tells you how close you got to sounding human. Change the test cases whenever you want, or buy ready-made benchmark packs from us.

2. **💪 The Training Gym.** A training environment that finds what your model is bad at, makes training data aimed at exactly that, trains, tests again, and repeats. You pick where the data comes from: your own human recordings, a bigger "teacher" model, or sentences written by an LLM.

And yes, it actually works. Here's what happened when we put a tiny model through it.

## The scoreboard 🏆

We took **MOSS-TTS-Nano**, a 100 million parameter TTS model. Tiny. The kind of model that skips leg day. We ran it through VoiceGym on **one GPU in about four days**, then benchmarked it on 494 sentences against real studio recordings of 26 human speakers.

| Model | Size | How human it sounds (% of a real human) | Independent judge (100 = human level) | Word error rate |
|---|---|---|---|---|
| MOSS-TTS-Nano, straight out of the box | 100M | 54.3% | 92.6 | 5.5% |
| Orpheus, the big teacher model | 3B | 73.7% | not run | not measured |
| **MOSS-TTS-Nano after VoiceGym SFT** | 100M | **94.7%** | **101.5** | **3.3%** |
| **...then after our DPO rounds** | 100M | 91.1% | **105.1** | **3.0%** |

What that means in normal words:

- The tiny model went from **54% to 95% of human** on the main judge. Its wins against the real human recording went from 179 to 240 out of 494.
- **It beat its own teacher.** A 100M model taught by a 3B model ended up sounding more human than the 3B model itself (94.7% vs 73.7%). The student is now bullying the teacher. Respectfully.
- A second, independent judge rates it **above the actual human recordings** on average (101.5 after SFT, 105.1 after DPO).
- It also got **easier to understand**. Word error rate went down from 5.5% to 3.0%.

Small models matter a lot for voice agents, by the way. A 100M model is way cheaper and faster to run than a 3B one, and in a live phone call every millisecond counts.

## 🏟️ Part 1: The Benchmark Arena

Here's how the arena works, minus the jargon.

We have real people reading sentences in a studio. Your model gets a short clip of one person's voice and then has to read the same sentences in that voice. A judge model listens to both, the real human and your model, and scores them.

Your score is basically **"how close did you get to the real human?"** 100% means the judge thinks you sound as natural as an actual person. Above 100 means the judge liked you *more* than the person. (Yes, we've seen that happen. More on that later, it's a fun story.)

What you can do in the arena:

- **Bring any TTS model.** We've already wired up Qwen3-TTS, Orpheus 3B, MOSS-TTS-Nano and Fish Drama-3.
- **Change the test cases.** It's your benchmark. Swap in the sentences and speakers that match what your product actually says.
- **Buy benchmark packs from us.** No good human reference recordings? We'll give you a curated question set built for your use case.
- **Get more than one number.** You also get word error rate (did it say the right words?), speaker similarity (does it still sound like that person?) and UTMOS (overall audio quality). So a model can't win by sounding pretty while saying the wrong thing.

### The judge: SpeechJudge 🧑‍⚖️

Our judge comes from **SpeechJudge** (Zhang et al., ICLR 2026). It's a research project that trained AI models to rate how natural speech sounds the way human listeners do, using a big dataset of human preference labels. It comes with two judges: **SpeechJudge-GRM**, which writes out *why* one clip sounds better than another (prosody, pacing, pronunciation and so on), and **SpeechJudge-BTRM**, a fast scorer that just gives you a number.

**Then we made it a bit better.** 😤

- **We anchored it to real humans.** The paper compares TTS models with each other. We compare your model with a real person saying the exact same sentence in the exact same voice. So "100" actually means something: human level.
- **We fixed the "first clip wins" problem.** Judges like this can favor whichever clip they hear first. We play every pair both ways, A then B and B then A, and average the two, so the order can't tip the score.
- **We actually read the judge's notes.** The SpeechJudge authors only used the final score. But GRM writes a full critique every single time, so we turn those critiques into a weakness report: flat prosody, unnatural pauses, robotic sound, pacing, missed emphasis, emotion mismatch and more. That report is what powers the Training Gym.
- **Two judges keep each other honest.** We train with one judge and double check with the other, so a model can't quietly learn to game a single scorer.

## 💪 Part 2: The Training Gym

This is the part we're really proud of.

Normal fine-tuning goes: grab a pile of data, train, hope for the best. VoiceGym goes like this:

```
  train on a batch  -->  test against real humans  -->  find the weak spots  -->  make data for those spots
         ^                                                                                  |
         |__________________________________________________________________________________|
```

After every training round, VoiceGym takes a fresh batch of roughly 100 to 200 clips and has the judge compare your model's voice with natural human voices. Then it tells you, in plain words, what's still off. This is what it said about our 100M model before any training, out of all the clips where the human won:

| Weakness | Showed up in |
|---|---|
| Flat prosody (monotone) | 98% |
| Unnatural pauses | 83% |
| Robotic or metallic sound | 81% |
| Pacing too slow | 70% |
| Wrong stress and emphasis | 58% |

Ouch. But now we know exactly where to train.

### Where the new data comes from (you pick)

Once VoiceGym knows where your model is struggling, there are three ways to fix it:

**🎙️ Option 1: Bring real human voices.** Got a voice dataset or a recording team? Add your human voice packs to the environment and the model trains on them. After every round VoiceGym tells you exactly what kind of recordings would help most, like "more questions with natural rising intonation" or "more expressive speech with real pauses". So your team records the stuff that fixes the model instead of reading random text for hours.

**🧑‍🏫 Option 2: Bring a bigger teacher model.** This is RL from AI feedback (RLAIF), and it's what our current pipeline runs on. No humans needed in the loop. A big model speaks the training sentences and the small model learns from it. We used **Orpheus 3B** as the teacher, which is 30 times bigger than our 100M student.

**✍️ Option 3: Let the LLM write targeted sentences.** The LLM loaded into the environment (we use Qwen 3.5 9B) reads the weakness report and writes brand new sentences built to hit those weak spots. Weak at question intonation? Here come the questions. Pauses feel off? Here come long sentences full of commas. Pair this with option 1 or 2: the LLM writes the script, and a human or the teacher reads it.

### The teacher trick (and why we wrote a paper on it) 📄

Our default pipeline isn't just "distill a big model into a small one". Here's what actually happens every round:

1. After a round of SFT, SpeechJudge takes a small batch from both the big model and the small model, and finds where the small one still falls short.
2. An LLM reads the judge's critiques, keeps track of which weaknesses keep coming back, and writes new sentences that target them.
3. The teacher (Orpheus 3B) speaks those sentences.
4. **The gate.** A teacher clip only makes it into training if the judge says it beats the student *on that exact sentence*. If the teacher isn't actually better at that line, we throw the clip out. In our run only about 37% of teacher clips survived. Turns out bigger doesn't mean better at everything.
5. The student trains on what survived, and the loop goes again.

The result: the student passed its teacher. And in our pilot, hour for hour, the gated teacher clips trained the model better than real studio recordings did (86.2% vs 69.3% of human on speakers the model had never heard).

We dug through the research, and no published speech paper runs this whole loop. So we wrote one. **Our paper on this weakness-targeted teacher loop is under review right now**, and should be out around January or February 2027 once the review is done.

## 🎯 After SFT: DPO, the finishing set

SFT got the model most of the way there. For the last few reps we used **DPO** (Direct Preference Optimization).

Here's the idea in plain English. Each round, the model reads the same sentence several times. The judge picks its best take and its worst take. The model learns "be more like the good one, less like the bad one". Then it does it all again with fresh sentences. Nobody has to label anything, the judge is the coach.

The tricky part of DPO is a knob called **beta**. It controls how far the model is allowed to drift from where it started. Too loose and the model forgets how to talk. Too tight and nothing changes. Most people just set it to 0.1 and pray. 🙏

So we wrote a paper about it: **"On the Optimality of Global Improvement Regularization over Local KL Maximization in Incremental Direct Preference Optimization"**. Instead of guessing beta, it works out the right value from the data and keeps adjusting it every step, so the model stays inside a safe zone while still improving. Our DPO pipeline implements that paper's algorithm directly.

What we got:

- **Three rounds, three improvements.** On speakers it had never heard, the model went from 86.2% to 89.4%, then 93.4%, then 96.5% of human, with word error rate holding at around 2% the whole time.
- **Best scores of any model we trained** on the full 494 benchmark: the independent judge gave it 105.1, word error rate fell to 3.0%, and audio quality (UTMOS) hit 4.08.
- **Honest moment:** on the main judge (BTRM) it dipped a little on the 494, from 94.7% to 91.1%. The two judges disagree here and it's probably within noise, but we're not going to hide it from you.

Now the fun story we promised. We also ran DPO the "set beta to 0.1 and pray" way for comparison. One of those runs scored **104.9% on BTRM, meaning the judge rated it more human than the actual humans.** Amazing, right? Except its word error rate went up and the independent judge didn't buy it. That's a model learning to please the judge instead of actually getting better. Classic reward hacking. It's exactly why VoiceGym always checks with two judges plus word accuracy, and why we trust the paper's way of setting beta.

## 📚 Built on three research papers (two are ours)

| Paper | What it gives VoiceGym | Who |
|---|---|---|
| **SpeechJudge** (Zhang et al., ICLR 2026) | The judges, GRM and BTRM | The SpeechJudge authors. We built on it and made it better. |
| **On the Optimality of Global Improvement Regularization over Local KL Maximization in Incremental Direct Preference Optimization** (Ayosh Singh Yadav, IIT Delhi) | How to set beta in DPO, round by round | Us |
| **Weakness-targeted teacher data for TTS** | The diagnose, target, teach and gate loop | Us, under review |

## The fine print (we're nice people)

- All scores come from automated judges (SpeechJudge BTRM and GRM), not a human listening test. A proper listening test is next on our list.
- BTRM also picks which teacher clips get used, so we always report the independent GRM judge and word error rate next to it.
- A few comparisons, like teacher clips vs real recordings in the pilot, have confidence intervals that cross zero. Every number, with confidence intervals, is in [`runs/sft/REPORT.md`](runs/sft/REPORT.md) and the `runs/dpo*/REPORT.md` files.

> 🗣️ **Coach says:** "Alright champ, that's the warm-up. Below is the heavy lifting. Hydrate."

---

# 🔧 Technical details

## What's in the repo

```
benchmark/      the arena: data prep, TTS generation, SpeechJudge scoring, metrics
sft/            the training gym: weakness diagnosis, targeted texts, teacher + gate, data mix, training
dpo/            online DPO with the KL paper's adaptive beta (MOSS-TTS-Nano and Orpheus adapters)
scripts/        entry points for every stage, plus the on-demand AWS GPU worker
runs/           small result files from our runs (reports, summaries, no audio)
docs/           the KL paper and the DPO design doc
```

Deeper docs: [`VoiceGym_SpeechJudge_Project_Plan.md`](VoiceGym_SpeechJudge_Project_Plan.md) (background and benchmark design), [`sft/PLAN.md`](sft/PLAN.md), [`sft/INTERFACES.md`](sft/INTERFACES.md) (file formats and decision rules), [`sft/RUNBOOK.md`](sft/RUNBOOK.md), [`dpo/README.md`](dpo/README.md), [`docs/moss_tts_nano_online_dpo.md`](docs/moss_tts_nano_online_dpo.md).

## How the scores are computed

Each benchmark sentence has a real human recording. The TTS speaks the same sentence in that speaker's cloned voice, and both clips get scored.

```
BTRM %          = mean TTS reward / mean human reward x 100        (SpeechJudge-BTRM, the headline number)
GRM closeness   = per sentence TTS score / human score x 100,      (SpeechJudge-GRM, the independent check)
                  averaged over sentences
```

GRM judges every pair in both A/B orders and averages them before taking the ratio. Every report also has WER (Whisper), speaker similarity, UTMOS and a BTRM parity score (200 x mean sigmoid(r_tts - r_human)), with 95% bootstrap confidence intervals over sentences.

## Benchmark set

Default (`CORPUS=ears`): **EARS** (facebookresearch/ears_dataset, CC BY-NC 4.0), anechoic studio recordings at 48 kHz, picked over LibriTTS-R for cleaner human reference audio.

- 26 native American English speakers (13 M / 13 F, `--seed 1234`) x 19 readings = 494 pairs
- Neutral "regular" style readings, 7 to 18 s each, every speaker reads the same texts. Emotional, whisper, fast and slow styles are left out, so the benchmark tests naturalness and not style control.
- Voice prompt: the same speaker's `rainbow_01_regular` (7.7 s), never a test sentence
- Locked set: `benchmark/manifests/ears_regular_26x19.csv`. A fresh worker re-downloads the 26 speakers' audio (about 15 GB of zips, only the regular readings are kept).

Previous set (`CORPUS=libritts-r`): LibriTTS-R `test-clean` (CC BY 4.0), 10 speakers x 50 sentences, `benchmark/manifests/libritts_r_test_clean_10x50.csv`.

## Requirements

- Linux and an NVIDIA GPU. SpeechJudge-GRM is a ~22 GB bf16 checkpoint, so 48 GB+ VRAM is recommended (A6000, L40S, A100, H100). Stages run one at a time, so the TTS and the judge never share the GPU.
- Python 3.10 to 3.12 (3.11+ for DPO), internet access to Hugging Face, OpenSLR and GitHub
- Separate venvs, because the packages conflict: `qwen-tts` pins `transformers==4.57.3`, SpeechJudge needs `transformers==4.52.3` (code pinned to commit `1fbd1c8`), MOSS-TTS-Nano pins torch 2.7.0 / transformers 4.57.1, and `.venv-dpo` uses `trl==1.16.0` (`requirements-dpo.txt`).

## Run the benchmark

```bash
git clone https://github.com/himeshsaini4482-a11y/voicegym.git && cd voicegym
bash scripts/setup_vm.sh                                # once; INSTALL_FLASH_ATTN=1 to also build flash-attn

RUN_NAME=smoke LIMIT=5 bash scripts/run_benchmark.sh    # 5-sentence end-to-end test first
bash scripts/run_sanity_check.sh                        # human vs itself, should be ~100%
bash scripts/run_benchmark.sh                           # full 494 benchmark (Qwen3-TTS) -> runs/base/
RUN_NAME=trained TTS_MODEL=/path/to/checkpoint bash scripts/run_benchmark.sh

# MOSS-TTS-Nano (generate -> GRM -> metrics -> BTRM); MOSS_MODEL=/path for a fine-tuned checkpoint
RUN_NAME=moss_nano bash scripts/run_moss_nano.sh
RUN_NAME=<run> bash scripts/run_btrm.sh                 # BTRM rewards for an existing run
```

Options (env vars): `VOTES=3` (judge calls per A/B order), `LIMIT=N`, `SKIP_METRICS=1`, `SKIP_GRM=1`. Every stage is resumable, so if the VM drops, just run the same command again. Long runs should go in `tmux` or `nohup`.

Re-compute a summary without the GPU: `.venv-judge/bin/python benchmark/judge.py --out-dir runs/base --summarize-only`

## Run the training gym (SFT)

Three resumable jobs, run in order on the GPU worker:

```bash
KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/pilot.sh"   # test round: gates, data, baselines, picks the recipe
KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/main.sh"    # diagnose -> targeted texts -> teacher + gate -> train, per round
KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/sft/final.sh"   # 494 benchmark + runs/sft/REPORT.md
```

The pieces: `sft/tag_weaknesses.py` turns GRM critiques into the weakness report, `sft/gen_texts.py` has the LLM write targeted sentences, `sft/teacher.py` and `sft/filter_teacher.py` run the teacher and the "teacher must beat student" gate, and `sft/make_mix.py` / `sft/reweight.py` build the training mix. Every decision (recipe, checkpoint, guard rails on WER, speaker similarity and UTMOS) is made on held-out dev speakers, never on the 494. Details in `sft/RUNBOOK.md`.

## Run DPO

```bash
KEEP_UP=1 bash scripts/gpu_run.sh "bash scripts/dpo/run.sh"
```

Each round: the LLM writes fresh texts, the policy speaks each prompt K times, BTRM ranks the samples, best vs worst become a pair, and the policy trains against a frozen reference with the KL paper's adaptive beta. Hyperparameters live in `dpo/hparams.json`. The script pauses (exit 3) when it needs a human to approve beta or optimizer settings. Results go to `runs/dpo/REPORT.md`. See `dpo/README.md` for the full flow and how to watch training in TensorBoard.

## On-demand GPU worker (AWS)

Everything runs on a disposable EC2 GPU worker (1x L40S 48 GB) that only exists while a job needs it. `gpu_up.sh` launches it from the `voicegym-gpu-v1` image in the first zone with capacity (ap-south-2a, 2b, ap-south-1a, 1b; g6e.xlarge, then g6e.2xlarge). Cost is about $2.24/hr while it's up.

```bash
bash scripts/gpu_run.sh "RUN_NAME=smoke LIMIT=5 bash scripts/run_benchmark.sh"
# launch worker -> copy code -> run in tmux -> copy small results back -> terminate worker

bash scripts/gpu_up.sh       # launch manually (ssh details end up in scripts/gpu/state.env)
bash scripts/gpu_down.sh     # copy runs/ back, terminate
bash scripts/gpu_status.sh   # any workers alive (= costing money), GPU memory, current job
```

Each job downloads the judge checkpoint to the worker's local NVMe (`scripts/stage_judge.sh`), because loading it from EBS takes 35+ minutes. Safety net: the worker powers itself off after 30 minutes with no GPU process, SSH login or tmux session (`touch ~/.keep-awake` on the worker to prevent that). It's *stopped*, not terminated, so unsynced results survive. Settings: `scripts/gpu/config.sh`.

## Outputs (`runs/<RUN_NAME>/`)

| File | Contents |
|---|---|
| `summary.json` | GRM human-closeness: overall, per speaker, per sentence |
| `btrm_summary.json` | BTRM rewards, BTRM %, parity, wins vs human |
| `judgments.jsonl` | Every SpeechJudge call: order, scores, full written analysis |
| `metrics_summary.json` | WER (TTS and human), speaker similarity, UTMOS (TTS and human) |
| `metrics.jsonl` | Per-sentence metrics and Whisper transcripts |
| `generation_config.json` | TTS model, seed, settings |
| `tts/*.wav` | Generated audio (kept on the worker, not in git) |

## LLM client

The LLM steps (weakness tagging, targeted sentences) call an OpenAI-compatible endpoint through `scripts/qwen9b_client.py`, which uses only the Python standard library.

```bash
cp .env.example .env    # set VOICEGYM_QWEN_BASE_URL, VOICEGYM_QWEN_API_KEY, VOICEGYM_QWEN_MODEL
python scripts/qwen9b_client.py "Write five varied sentences for a TTS model to speak."
```

Never commit `.env`. Rotate a leaked token right away.

## Licences

SpeechJudge-GRM and EARS are CC BY-NC 4.0 (non-commercial). LibriTTS-R is CC BY 4.0. Qwen3-TTS is Apache-2.0.
