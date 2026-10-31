<div align="center">

# 🎙️ Turing Tester

### Can your voice model pass for human? We test it. Then we train it until it does.

*Built for the Bolna AI Hackathon*

</div>

---

Okay, real talk.

Most hackathon projects this weekend will look something like this: take a voice API, call it, put a nice UI on top, ship it. And honestly? That's fine. We love a good wrapper.

But nobody is asking the obvious question. **Who makes the model behind that API better?**

Frontier labs and voice AI teams spend months trying to figure out *why* their TTS model still sounds like a robot reading out a parking ticket. They guess. They collect a random pile of data. They retrain. Then they guess again.

We built Turing Tester so they can stop guessing.

Think of it as a Turing test for voices, with a gym attached. Your model walks in, gets tested against real human voices, gets told exactly which muscles are weak, trains on exactly those, and comes back for another round. Every round the model gets better, and every round the next batch of training data gets smarter, because it's aimed at whatever is still broken. The loop feeds itself. It's recursive self-improvement, but for how your model *sounds*.

> 🗣️ **Coach says:** "Bro. You don't fix a weak voice with random reps. You find the weak spot and you hit it. Again. And again. And again."

## The 30 second version

Turing Tester does two things:

1. **🏟️ The Benchmark Arena.** Plug in your TTS model. It reads the same sentences a real human read, in that human's cloned voice, and a judge model tells you how close you got to sounding human. Change the test cases whenever you want, or buy ready-made benchmark packs from us.

2. **💪 The Training Gym.** A training environment that finds what your model is bad at, makes training data aimed at exactly that, trains, tests again, and repeats. You pick where the data comes from: your own human recordings, a bigger "teacher" model, or sentences written by an LLM. For labs that want more, we provide custom training and RL environments, expert multilingual voice data, and forward deployed engineers who work inside your lab to keep improving your models.

And yes, it actually works. Here's what happened when we put a tiny model through it.

## The scoreboard 🏆

We took **MOSS-TTS-Nano**, a 100 million parameter TTS model. Tiny. The kind of model that skips leg day. We trained it on **one GPU for about 77 hours, roughly $172 of compute**, then benchmarked it on 494 sentences against real studio recordings of 26 human speakers.

| Model | Size | How human it sounds (% of a real human) | Independent judge (100 = human level) | Word error rate |
|---|---|---|---|---|
| MOSS-TTS-Nano, straight out of the box | 100M | 54.3% | 92.6 | 5.5% |
| Orpheus, the big teacher model | 3B | 73.7% | not run | not measured |
| **MOSS-TTS-Nano after our SFT** | 100M | **94.7%** | **101.5** | **3.3%** |
| **...then after our DPO rounds** | 100M | 91.1% | **105.1** | **3.0%** |

What that means in normal words:

- The tiny model went from **54% to 95% of human** on the main judge. Its wins against the real human recording went from 179 to 240 out of 494.
- **It beat its own teacher.** A 100M model taught by a 3B model ended up scoring higher than the 3B model itself (94.7% vs 73.7%). The student is now bullying the teacher. Respectfully.
- A second, independent judge, one that never touched training, scored it **above the real human recordings** on average (101.5 after SFT, 105.1 after DPO).
- It also got **easier to understand**. Word error rate went down from 5.5% to 3.0%.

Small models matter a lot for voice agents, by the way. A 100M model is way cheaper and faster to run than a 3B one, and in a live phone call every millisecond counts.

## 🏟️ Part 1: The Benchmark Arena

Here's how the arena works, minus the jargon.

We have real people reading sentences in a studio. Your model gets a short clip of one person's voice and then has to read the same sentences in that voice. A judge model listens to both, the real human and your model, and scores them.

Your score is basically **"how close did you get to the real human?"** 100% means the judge thinks you sound as natural as an actual person. Above 100 means the judge liked you *more* than the person. (Yes, we've seen that happen. More on that later, it's a fun story.)

What you can do in the arena:

- **Bring any TTS model.** Drop your weights in a folder, give a Hugging Face id, or point it at a TTS API like ElevenLabs. Qwen3-TTS, MOSS-TTS-Nano and Orpheus are recognised out of the box.
- **Change the test cases.** It's your benchmark. Use your own sentences, or even your own voice recordings as the human reference.
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

Normal fine-tuning goes: grab a pile of data, train, hope for the best. Turing Tester goes like this:

```
  train on a batch  -->  test against real humans  -->  find the weak spots  -->  make data for those spots
         ^                                                                                  |
         |__________________________________________________________________________________|
```

After every training round, Turing Tester takes a fresh batch of roughly 100 to 200 clips and has the judge compare your model's voice with natural human voices. Then it tells you, in plain words, what's still off. This is what it said about our 100M model before any training, out of all the clips where the human won:

| Weakness | Showed up in |
|---|---|
| Flat prosody (monotone) | 98% |
| Unnatural pauses | 83% |
| Robotic or metallic sound | 81% |
| Pacing too slow | 70% |
| Wrong stress and emphasis | 58% |

Ouch. But now we know exactly where to train.

The judge tags every critique into six pillars: intonation, rhythm, stress, emotion, sound quality, and accuracy (right words, right voice). The report goes to your researchers, caveats included, and they pick what the next batch should be.

### Where the new data comes from (you pick)

Once Turing Tester knows where your model is struggling, there are three ways to fix it:

**🎙️ Option 1: Bring real human voices.** Got a voice dataset or a recording team? Add your human voice packs to the environment and the model trains on them. After every round the weakness report tells you what kind of recordings would help most, like "more questions with natural rising intonation" or "more expressive speech with real pauses". So your team records the stuff that fixes the model instead of reading random text for hours. No recording team, or need languages you don't have speakers for? We provide expert multilingual voice data, recorded to target exactly what the report says is broken.

**🧑‍🏫 Option 2: Bring a bigger teacher model.** This is RL from AI feedback (RLAIF), and it's what our current pipeline runs on. No humans needed in the loop. A big model speaks the training sentences and the small model learns from it. We used **Orpheus 3B** as the teacher, which is 30 times bigger than our 100M student.

**✍️ Option 3: Let the LLM write targeted sentences.** The LLM loaded into the environment (we used Qwen 3.5 9B, but any OpenAI-compatible model works, even a local one) reads the weakness report and writes brand new sentences built to hit those weak spots. Weak at question intonation? Here come the questions. Pauses feel off? Here come long sentences full of commas. Pair this with option 1 or 2: the LLM writes the script, and a human or the teacher reads it. No LLM? We ship 2,396 hand-written sentences grouped by the weakness they train.

### The teacher trick (and why we wrote a paper on it) 📄

Our default pipeline isn't just "distill a big model into a small one". Here's what actually happens every round:

1. After a round of SFT, SpeechJudge takes a small batch from both the big model and the small model, and finds where the small one still falls short.
2. An LLM reads the judge's critiques, keeps track of which weaknesses keep coming back, and writes new sentences that target them.
3. The teacher (Orpheus 3B) speaks those sentences.
4. **The gate.** A teacher clip only makes it into training if the judge says it beats the student *on that exact sentence* (and it has to say the right words in the right voice). If the teacher isn't actually better at that line, we throw the clip out. Only 37 to 43% of teacher clips survived. Turns out bigger doesn't mean better at everything.
5. The student trains on what survived, and the loop goes again.

The result: the student passed its teacher. And in our pilot, hour for hour, the gated teacher clips trained the model better than real studio recordings did (86.2% vs 69.3% of human on speakers the model had never heard). The teacher-trained model was also the only one that kept its word accuracy on spontaneous speech.

We dug through the research, and no published speech paper runs this whole loop. So we wrote one. **Our paper on this loop, "Diagnose, Target, Teach: Critique-Guided, Judge-Gated Distillation for Small Text-to-Speech Models" by Daksh Choudhary (IIT Delhi), is under review right now**, and should be out around January or February 2027 once the review is done.

## 🎯 After SFT: DPO, the finishing set

SFT got the model most of the way there. For the last few reps we used **DPO** (Direct Preference Optimization).

Here's the idea in plain English. Each round, the model reads the same sentence several times. The judge picks its best take and its worst take. The model learns "be more like the good one, less like the bad one". Then it does it all again with fresh sentences. Nobody has to label anything, the judge is the coach.

The tricky part of DPO is a knob called **beta**. It controls how far the model is allowed to drift from where it started. Too loose and the model forgets how to talk. Too tight and nothing changes. Most people just set it to 0.1 and pray. 🙏

So we wrote a paper about it: **"On the Optimality of Global Improvement Regularization over Local KL Maximization in Incremental Direct Preference Optimization"**. Instead of guessing beta, it works out the right value from the data and keeps adjusting it every step, so the model stays inside a safe zone while still improving. The DPO pipeline in [`dpo/`](dpo/README.md) implements that paper's algorithm.

What we got:

- **Three rounds, three improvements.** On speakers it had never heard, the model went from 86.2% to 89.4%, then 93.4%, then 96.5% of human, with word error rate holding at around 2% the whole time.
- **Best scores of any model we trained** on the full 494 benchmark: the independent judge gave it 105.1, word error rate fell to 3.0%, and audio quality (UTMOS) hit 4.08.
- **Honest moment:** on the main judge (BTRM) it dipped a little on the 494, from 94.7% to 91.1%. The two judges disagree here, and the gain over SFT on the independent judge (+3.5) isn't statistically solid yet. DPO is the experimental part of the gym, and we're not going to pretend otherwise.

Now the fun story we promised. We also ran DPO the "set beta to 0.1 and pray" way for comparison. One of those runs scored **104.9% on BTRM, meaning the judge rated it more human than the actual humans.** Amazing, right? Except its word error rate went up and the independent judge didn't buy it. That's a model learning to please the judge instead of actually getting better. Classic reward hacking. It's exactly why Turing Tester always checks with two judges plus word accuracy.

## 📚 Built on three research papers (two are ours)

| Paper | What it gives Turing Tester | Who |
|---|---|---|
| **SpeechJudge** (Zhang et al., ICLR 2026) | The judges, GRM and BTRM | The SpeechJudge authors. We built on it and made it better. |
| **On the Optimality of Global Improvement Regularization over Local KL Maximization in Incremental Direct Preference Optimization** (Ayosh Singh Yadav, IIT Delhi) | How to set beta in DPO, round by round | Us |
| **Diagnose, Target, Teach: Critique-Guided, Judge-Gated Distillation for Small Text-to-Speech Models** (Daksh Choudhary, IIT Delhi) | The diagnose, target, teach and gate loop | Us, under review |

## The fine print (we're nice people)

- All scores come from automated judges (SpeechJudge BTRM and GRM), not a human listening test. A score above 100 means the judge preferred the TTS clip, not that people would. A proper listening test is next on our list.
- BTRM also picks which teacher clips get used, so we always report the independent GRM judge and word error rate next to it.
- A few comparisons, like teacher clips vs real recordings in the pilot, have confidence intervals that cross zero. Every number, with confidence intervals and limits, is in [`RESULTS.md`](RESULTS.md) and [`docs/sft_experiment/`](docs/sft_experiment/).

> 🗣️ **Coach says:** "Alright champ, that's the warm-up. Below is the heavy lifting. Hydrate."

---

# 🔧 Technical details: using Turing Tester

Measure how close a text-to-speech (TTS) model sounds to real human speech, then make it
closer with SFT and DPO, all on your own machine.

> **Status: new.** Each command wraps the code that produced [`RESULTS.md`](RESULTS.md), which
> ran on an NVIDIA L40S. The commands themselves have so far been tested only without a GPU
> (`tests/`). The first full GPU runs are still to come, so please report anything that breaks.
> The scripts the experiments used are documented in [`docs/internal_pipeline.md`](docs/internal_pipeline.md).

```
your TTS model ──► /benchmark ──► score vs real human recordings
      │
      ├──► /sft ──► learns from human recordings, your voice, or a bigger model
      │
      └──► /dpo ──► learns from its own best and worst samples
                    └──► every run ends with a before/after benchmark
```

## What you need

| | |
|---|---|
| **GPU** | NVIDIA with **24 GB+** of memory (Windows or Linux), or an Apple Silicon Mac with **32 GB+** of memory (best effort, much slower) |
| **Disk** | about 100 GB free for benchmarking, about 150 GB for SFT with a teacher or your voice, about 250 GB for SFT on the EARS recordings. Keep the repo outside OneDrive/Dropbox-style synced folders |
| **Software** | `git` and an up-to-date NVIDIA driver. Everything else (Python, libraries, models, datasets) is installed automatically the first time a command needs it |
| **Internet** | for the first run of each command (downloads: ~45 GB of judge weights; a teacher or DPO run also fetches one reference clip per EARS speaker, which means downloading each speaker's ~600 MB archive once); after that, runs work offline except API benchmarks |

## Start

```bash
git clone https://github.com/himeshsaini4482-a11y/turing-tester.git
cd turing-tester
./turing-tester            # Linux / macOS / Git Bash
.\turing-tester            # Windows (PowerShell or cmd)
```

This opens the Turing Tester interface in your terminal (the title is drawn in shaded block letters; in a terminal narrower than 52 columns it is plain text). You work in it by typing `/` commands:

```
 ████████╗██╗   ██╗██████╗ ██╗███╗   ██╗ ██████╗
 ╚══██╔══╝██║   ██║██╔══██╗██║████╗  ██║██╔════╝
    ██║   ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
    ██║   ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
    ██║   ╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝

 ████████╗███████╗███████╗████████╗███████╗██████╗
 ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝██╔══██╗
    ██║   █████╗  ███████╗   ██║   █████╗  ██████╔╝
    ██║   ██╔══╝  ╚════██║   ██║   ██╔══╝  ██╔══██╗
    ██║   ███████╗███████║   ██║   ███████╗██║  ██║
    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
╭───────────────────────────────────────────────────────────────────────╮
│                                                                       │
│  How close does your TTS model sound to a real person?                │
│                                                                       │
│  GPU     NVIDIA GeForce RTX 4090 · 24 GB                              │
│  found   2 models in models/ · 1 API in apis/ · 1 voice in voices/    │
│                                                                       │
│  /doctor to check this machine · /help for all commands               │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯

> /be
  /benchmark   score a model or API against real human recordings
```

The first start downloads [uv](https://docs.astral.sh/uv/) and Python 3.12 into `.turing-tester/`
(about a minute). Nothing is installed outside the repo except the usual Hugging Face download
cache. Run `/doctor` first: it reports the GPU, free memory, free disk and what is already
installed, and tells you which commands your machine can run.

### Commands

| Command | What it does |
|---|---|
| `/doctor` | check this machine: GPU, memory, disk, what is installed, which commands can run |
| `/models` | list the models in `models/` and the API configs in `apis/`, with the type detected for each and whether it can be benchmarked, SFT'd or DPO'd |
| `/voices` | list your recordings in `voices/` with clip counts and total minutes |
| `/benchmark` | score a model or API against real human recordings ([section 1](#1-benchmark-a-model)) |
| `/sft` | fine-tune a model on good examples ([section 2](#2-sft-train-on-good-examples)) |
| `/dpo` | improve a model on its own best and worst samples ([section 3](#3-dpo-learn-from-its-own-best-and-worst-samples)) |
| `/runs` | list past runs with their headline scores |
| `/report <run>` | show a run's results in the terminal |
| `/setup [tts judge …]` | build environments now instead of on first use |
| `/help [command]` | list the commands, or show one command's options |
| `/clear` | clear the screen |
| `/exit` | leave (also Ctrl+D) |

**Guided or direct.** Type a command on its own and it asks what it needs, one question at a time,
with choices you pick using the arrow keys (for `/benchmark`: which model, which sentence set,
quick check or full run). Before starting it shows a summary and the equivalent direct command, so
you learn the short form. Or give everything at once:

```
> /benchmark models/my-tts --limit 10
```

The first argument can be a model folder, a Hugging Face id or an API config; the type is detected
automatically. The options are the ones listed in each section below.

**Keys**

| Key | Does |
|---|---|
| `/` | opens the command menu; keep typing and it shows the closest commands first (see below) |
| Tab | completes commands, options and file paths (`models/`, `voices/`, `apis/`, `runs/`) |
| ↑ / ↓ | earlier commands (history is kept in `.turing-tester/history`) |
| Esc | cancels a guided question |
| Ctrl+C | stops the running job and returns to the prompt; at an empty prompt, press twice to leave |
| Ctrl+D | leaves |

**Command suggestions.** As soon as you type `/`, a menu under the prompt lists every command with a
one-line description. Each letter you type re-sorts it so the closest matches come first:

```
> /r                                     > /bnch                          > /benchmrk models/my-tts
  /report    show a run's results          /benchmark   score a model…      unknown command /benchmrk
  /runs      list past runs …                                               did you mean /benchmark? (Enter runs it)
  /benchmark score a model …
```

- Commands that start with what you typed come first (`/r` → `/report`, `/runs`), then commands that contain
  it, then commands with the same letters in order (`/bnch` → `/benchmark`), then near-misses
  with a typo or two (`/benchmrk` → `/benchmark`).
- ↑ / ↓ moves through the menu, Tab or Enter picks the highlighted command, Esc closes the menu.
- Short forms work when only one command matches: `/bench`, `/doc`, `/rep`.
- If you press Enter on a command that does not exist, nothing runs; it suggests the closest one
  and Enter accepts it.
- After the command name, the menu switches to that command's options (`--limit`, `--set`, …) and
  then to matching paths in `models/`, `voices/`, `apis/` and `runs/`.

**Running jobs.** A running job's output streams into the interface, with a line under it showing
the current step and the elapsed time. One job runs at a time. A job that was stopped, or cut off
by a closed window, continues where it left off when you run the same command again. For runs that
take hours, start Turing Tester inside `tmux` or `screen` so a dropped connection does not stop it.

Text that does not start with `/` is not sent anywhere; the interface reminds you to use a command.

### Without the interface

Every command also works straight from your shell, without the `/`, for scripts and servers:

```bash
./turing-tester doctor
./turing-tester benchmark --model models/my-tts --limit 10
```

## Folders you use

```
models/        drop model weights here, one folder per model   (never committed)
voices/        your own voice recordings                        (never committed)
apis/          one config file per TTS API                      (keys stay in environment variables)
texts/         your own sentence lists (optional)
runs/          every result, one folder per run
```

---

## 1. Benchmark a model

Every benchmark has the model speak the same locked set of sentences in the voices of real
speakers, then compares each clip with the real person's recording of that sentence.

### Local model weights

Copy the model folder (the one with `config.json` and the weights) into `models/` and run:

```
> /benchmark models/my-tts
```

Turing Tester detects the model type from the folder:

| Model in the folder | What happens |
|---|---|
| Qwen3-TTS, MOSS-TTS-Nano, Orpheus | recognised automatically (a Hugging Face id such as `OpenMOSS-Team/MOSS-TTS-Nano-100M` also works) |
| another LLM-style TTS (a language-model backbone that outputs audio-codec tokens) | add a short `turing-tester.yaml` in the folder that names its codec and prompt format ([guide](docs/adding_a_model.md)) |
| anything else (VITS, Kokoro, F5-TTS, …) | add a `synthesize.py` with two functions, `load(device)` and `synthesize(text, ref_audio, ref_text)` ([guide](docs/adding_a_model.md)) |

A `requirements.txt` in the model folder gets its own environment.

### A TTS API

Describe the API in one small file in `apis/` and keep the key in an environment variable, set
before you start Turing Tester:

```bash
export ELEVENLABS_API_KEY=...            # Windows: $env:ELEVENLABS_API_KEY="..."
./turing-tester
```

```
> /benchmark apis/elevenlabs.yaml
```

Ready-made configs:

- `apis/elevenlabs.yaml`: an instant voice clone per speaker, which needs a paid plan. The
  clones are deleted at the end of the run.
- `apis/openai-compatible.yaml`: any server with `/v1/audio/speech`. It has no voice cloning.

For any other provider, copy `apis/template.yaml`:

```yaml
name: my-provider
url: https://api.example.com/v1/tts
headers:
  Authorization: "Bearer ${MY_TTS_API_KEY}"   # read from the environment, never written to logs
body:
  text: "{text}"
  reference_audio: "{ref_audio_base64}"       # the voice to clone
  reference_text: "{ref_text}"
  format: wav
response:
  audio: binary                              # or a JSON path, e.g. data.audio_base64
  format: wav                                # wav | mp3 | flac | ogg | pcm_s16le
voice_cloning: per_request                   # per_request | create_voice | none
requests_per_minute: 60
```

Requests are rate-limited and retried on errors. A stopped run continues where it left off.

If an API cannot clone voices (`voice_cloning: none`), its clips are compared with a different
speaker's recording. That comparison is less fair, so the report marks speaker similarity as not
available and adds a warning.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--set` | `ears` | sentence set: `ears` (494 clips, 26 speakers), `libritts-r` (500 clips, 10 speakers), or `voices/<name>` (your own recordings) |
| `--judges` | `grm,btrm` | which SpeechJudge judges to run |
| `--limit N` | all | first N sentences only, for a quick test |
| `--name` | from the model | run folder name under `runs/` |

### What you get

`runs/<name>/report.md`, plus the raw numbers in `report.json` and, per sentence, `judgments.jsonl`
(GRM, including the judge's written analysis), `btrm.jsonl`, `metrics.jsonl` and the audio in `tts/`:

| Score | Meaning | Real human |
|---|---|---|
| **GRM closeness** | SpeechJudge-GRM scores the TTS and human clip of each sentence; TTS ÷ human × 100, averaged | 100 |
| **BTRM %** | SpeechJudge-BTRM reward, mean TTS ÷ mean human × 100 | 100 |
| WER | words the model got wrong (Whisper) | ~2% |
| Speaker similarity | does it sound like the target speaker (0-1) | 1.0 |
| UTMOS | predicted naturalness, 1-5 | ~4.0 |

These are automatic judges, not human listeners. A score above 100 means the judge preferred
the TTS clip, not that people would.

---

## 2. SFT: train on good examples

```
> /sft models/my-tts                              # learn from EARS human recordings
> /sft models/my-tts --voice voices/my-voice      # learn your own voice
> /sft models/my-tts --teacher models/big-tts     # learn from a bigger model
```

`/sft` on its own asks for the model and the data sources instead.

You can combine sources, e.g. `--voice voices/my-voice --teacher models/big-tts`. Add `--ears` to
also use the EARS recordings. A teacher can be any model Turing Tester can benchmark, or an API
(`--teacher apis/my-provider.yaml`) if its terms allow training on its output. The teacher must be
able to clone voices.

**Your own voice.** Put the recordings in a folder:

```
voices/my-voice/
  clip_001.wav
  clip_002.wav
  ...
  metadata.csv        # optional: file,text   (missing transcripts are made with Whisper)
```

Use clean, single-speaker recordings: aim for at least 30 minutes, in clips of 3-15 seconds (at
least 50 clips). Turing Tester holds back 10% of the clips (at least 20) as a test set, so you also get a
benchmark on your own voice, and 5% (at least 10) as a dev set for checking training. With a
teacher, the teacher speaks in your voice too.

**Training sentences** (used when a teacher speaks new text):

- **Default:** 2,396 hand-written sentences shipped with Turing Tester
  (`turing_tester/data/sentence_pool.jsonl`), grouped by the speech weaknesses they exercise.
- `--texts texts/my_sentences.txt`: your own sentences, one per line.
- **Optional LLM:** set `TURING_TESTER_LLM_BASE_URL`, `TURING_TESTER_LLM_MODEL` and, if needed,
  `TURING_TESTER_LLM_API_KEY` (any OpenAI-compatible server, including local ones like Ollama or LM
  Studio; see `.env.example`). Turing Tester then has SpeechJudge-GRM diagnose the model, the LLM tags
  the judge's critiques into weaknesses, and it writes 4,000 sentences aimed at them, as in our
  experiments.
- `--review-texts` (optional): once the diagnosis and the sentences are ready, Turing Tester shows them
  and waits. You can edit `runs/sft/<name>/texts.txt` before the teacher starts.

**What a run does** (`--mode quick`, the default): one training run with the settings from our
experiments, picked by the data you give it:

| Data | Settings (MOSS-TTS-Nano) | From our experiments |
|---|---|---|
| EARS only | lr 2e-5, 2 epochs | pilot arm CU |
| a teacher (with or without your voice) | lr 2e-5, 6 epochs | pilot arm B, our best SFT model (`RESULTS.md`) |
| EARS + a teacher or your voice | lr 2e-5, 3 epochs, 35% of the hours from the non-EARS data | the main-round mix |
| your voice only | lr 2e-5, 6 epochs | not run; uses arm B's settings |

The teacher's clips are kept only if they are complete, at a normal speaking rate, have WER ≤ 5%,
match the voice (similarity ≥ 0.88), and are rated above the student's own attempt by
SpeechJudge-BTRM. In our run 43% passed.

Every SFT run scores the model before and after training on a held-out dev set. These are 4 EARS
speakers who are not in the benchmark, or your dev clips. The checkpoint is kept as
`runs/sft/<name>/final/` only if WER, speaker similarity and UTMOS stay within guard limits. It is
then benchmarked against the starting model, and `runs/sft/<name>/report.md` shows both with
paired 95% intervals.

`--mode full` instead reproduces our complete procedure. It first runs a pilot that compares six
data sources, then training rounds with a fresh diagnosis each round (`sft/PLAN.md`). It needs
Linux or WSL2, a MOSS-TTS-Nano student, the Orpheus teacher, the optional LLM, and several GPU-days
(ours took ~76 GPU-hours).

Other model families use their own trainers, and none of them were part of our experiments:

- Orpheus and `turing-tester.yaml` models train with LoRA (rank 16, lr 1e-4).
- Qwen3-TTS uses Qwen's own fine-tuning script, which makes a single-speaker model. It therefore
  works only with `--voice` alone.
- `synthesize.py` models use the folder's `train.py`.

---

## 3. DPO: learn from its own best and worst samples

```
> /dpo runs/sft/my-run/final
```

DPO does not learn from recordings, only from the model's own samples. To speak, the model
clones a voice from a short reference clip. By default those are the EARS speakers' voices, which
improves the model across many voices.

`--voice voices/my-voice` (optional) makes the model clone **your** voice during DPO, so the
improvement is aimed at the voice you will deploy. Your dev clips then pick the best round and your
test clips give the before/after score. Use it only if you want the model to speak in your voice, for example after
`/sft --voice voices/my-voice`.

DPO works exactly as in our experiments. Each round:

1. The model speaks 1,000 new sentences 4 times each.
2. SpeechJudge-BTRM scores the samples.
3. The best and the worst valid sample form a pair. Samples with too many wrong words, the wrong
   voice or broken timing are excluded.
4. The model is trained to prefer the best over the worst, staying close to where it started
   (β 0.1, AdamW lr 1e-6, 16 pairs per step: our trial 1, `dpo/archive/`).

It runs up to 3 rounds (`--rounds`) and stops early when a round stops helping on the dev set or
breaks the guard limits. The sentences come from `--texts`, the optional LLM, or the bundled pool.
The pool is smaller than 3 × 1,000, so its sentences repeat across rounds with other voices. The
result is `runs/dpo/<name>/final/` with a before/after benchmark, written only if a round improved
the model. In our runs, DPO did not give a clear gain on top of SFT, and the two judges disagreed
(see `RESULTS.md`). Treat it as an experiment.

---

## What can be trained

| | Benchmark | SFT | DPO |
|---|---|---|---|
| MOSS-TTS-Nano | ✅ | ✅ as in our experiments | ✅ as in our experiments |
| Orpheus | ✅ | ✅ LoRA | ✅ LoRA |
| other LLM-style TTS, with `turing-tester.yaml` | ✅ | ✅ LoRA | ✅ LoRA |
| Qwen3-TTS (Base) | ✅ | your voice only | ❌ |
| any other model, with `synthesize.py` | ✅ | with your own `train.py` | ❌ |
| a TTS API | ✅ | ❌ (can be a teacher) | ❌ |

DPO needs the model to report how likely each audio token was. Only LLM-style TTS models do that,
and Qwen3-TTS's two-stage decoder has no DPO adapter yet. APIs cannot be trained because their
weights are not available. Only the MOSS-TTS-Nano rows were run in our experiments. The other
trainers follow the same method with settings that are not tuned.

## How long it takes

On one NVIDIA L40S (48 GB), a full 494-sentence benchmark of MOSS-TTS-Nano with both judges
took about 4.7 hours. A 24 GB card is slower: the GRM judge (about 22 GB) does not fully fit, so
part of it runs from system memory at full precision. The generators and judges print a running
time estimate. Use `--limit 10` for a quick check first. Every command can be stopped and re-run;
it continues where it left off.

## Licences and privacy

- SpeechJudge (both judges) and the EARS dataset are **CC BY-NC 4.0: non-commercial use only**.
  LibriTTS-R is CC BY 4.0. Your own models and APIs keep their own terms.
- API keys are read from environment variables only and never written to files or logs.
- `models/`, `voices/`, `.env` and `runs/` audio are never committed to git.
- With the optional LLM, Turing Tester sends it the judge's written critiques and requests for
  sentences, never audio.

## More

- [`RESULTS.md`](RESULTS.md): what we measured so far, with the SFT write-ups in
  [`docs/sft_experiment/`](docs/sft_experiment/). Your own runs go to `runs/`, which is never committed.
- [`docs/adding_a_model.md`](docs/adding_a_model.md): `turing-tester.yaml`, `synthesize.py` and `train.py`.
- [`TuringTester_SpeechJudge_Project_Plan.md`](TuringTester_SpeechJudge_Project_Plan.md): the original design and the reasoning behind the metrics.
- [`docs/internal_pipeline.md`](docs/internal_pipeline.md): the experiment scripts (team reference; the AWS worker scripts are in `tools/aws/`).
- Tests: `python -m unittest discover -s tests` (CPU only; needs numpy, soundfile, librosa, pyyaml; the interface tests also need prompt_toolkit and rich).
