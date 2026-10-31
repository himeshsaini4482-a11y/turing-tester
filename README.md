# Turing Tester

Measure how close a text-to-speech (TTS) model sounds to real human speech, then make it
closer with SFT and DPO, all on your own machine.

> **Status: new.** Each command wraps the code that produced [`RESULTS.md`](RESULTS.md), which
> ran on an NVIDIA L40S. The commands themselves have so far been tested only without a GPU
> (`tests/`). The first full GPU runs are still to come, so please report anything that breaks.
> The scripts the experiments used are documented in [`docs/internal_pipeline.md`](docs/internal_pipeline.md).

```
your TTS model ──► turing-tester benchmark ──► score vs real human recordings
      │
      ├──► turing-tester sft ──► learns from human recordings, your voice, or a bigger model
      │
      └──► turing-tester dpo ──► learns from its own best and worst samples
                               └──► every run ends with a before/after benchmark
```

## What you need

| | |
|---|---|
| **GPU** | NVIDIA with **24 GB+** of memory (Windows or Linux), or an Apple Silicon Mac with **32 GB+** of memory (best effort, much slower) |
| **Disk** | about 100 GB free for benchmarking, about 150 GB for SFT with a teacher or your voice, about 250 GB for SFT on the EARS recordings. Keep the repo outside OneDrive/Dropbox-style synced folders |
| **Software** | `git` and an up-to-date NVIDIA driver. Everything else (Python, libraries, models, datasets) is installed automatically the first time a command needs it |
| **Internet** | for the first run of each command (downloads: ~45 GB of judge weights; a teacher or DPO run also fetches one reference clip per EARS speaker, which means downloading each speaker's ~600 MB archive once); after that, runs work offline except API benchmarks |

Check your machine before anything else:

```bash
git clone https://github.com/himeshsaini4482-a11y/voicegym.git
cd voicegym
./turing-tester doctor            # Linux / macOS / Git Bash
.\turing-tester doctor            # Windows (PowerShell or cmd)
```

`doctor` reports the GPU, free memory, free disk and what is already installed, and tells you
which commands your machine can run. The first time, the launcher downloads
[uv](https://docs.astral.sh/uv/) and Python 3.12 into `.turing-tester/`. Nothing is installed outside
the repo except the usual Hugging Face download cache. Each stage gets its own environment, built on
first use. To build the environments ahead of time, run `./turing-tester setup tts judge`.

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

```bash
./turing-tester benchmark --model models/my-tts
```

Turing Tester detects the model type from the folder:

| Model in the folder | What happens |
|---|---|
| Qwen3-TTS, MOSS-TTS-Nano, Orpheus | recognised automatically (a Hugging Face id such as `OpenMOSS-Team/MOSS-TTS-Nano-100M` also works) |
| another LLM-style TTS (a language-model backbone that outputs audio-codec tokens) | add a short `turing-tester.yaml` in the folder that names its codec and prompt format ([guide](docs/adding_a_model.md)) |
| anything else (VITS, Kokoro, F5-TTS, …) | add a `synthesize.py` with two functions, `load(device)` and `synthesize(text, ref_audio, ref_text)` ([guide](docs/adding_a_model.md)) |

A `requirements.txt` in the model folder gets its own environment.

### A TTS API

Describe the API in one small file in `apis/` and keep the key in an environment variable:

```bash
export ELEVENLABS_API_KEY=...            # Windows: $env:ELEVENLABS_API_KEY="..."
./turing-tester benchmark --api apis/elevenlabs.yaml
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
| Speaker similarity | does it sound like the target speaker (0–1) | 1.0 |
| UTMOS | predicted naturalness, 1–5 | ~4.0 |

These are automatic judges, not human listeners. A score above 100 means the judge preferred
the TTS clip, not that people would.

---

## 2. SFT: train on good examples

```bash
./turing-tester sft --model models/my-tts                              # learn from EARS human recordings
./turing-tester sft --model models/my-tts --voice voices/my-voice      # learn your own voice
./turing-tester sft --model models/my-tts --teacher models/big-tts     # learn from a bigger model
```

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

Use clean, single-speaker recordings: aim for at least 30 minutes, in clips of 3–15 seconds (at
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

```bash
./turing-tester dpo --model runs/sft/my-run/final
```

DPO does not learn from recordings, only from the model's own samples. To speak, the model
clones a voice from a short reference clip. By default those are the EARS speakers' voices, which
improves the model across many voices.

`--voice voices/my-voice` (optional) makes the model clone **your** voice during DPO, so the
improvement is aimed at the voice you will deploy. Your dev clips then pick the best round and your
test clips give the before/after score. Use it only if you want the model to speak in your voice, for example after
`turing-tester sft --voice voices/my-voice`.

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

- [`RESULTS.md`](RESULTS.md): what we measured so far.
- [`docs/adding_a_model.md`](docs/adding_a_model.md): `turing-tester.yaml`, `synthesize.py` and `train.py`.
- [`TuringTester_SpeechJudge_Project_Plan.md`](TuringTester_SpeechJudge_Project_Plan.md): the original design and the reasoning behind the metrics.
- [`docs/internal_pipeline.md`](docs/internal_pipeline.md): the experiment scripts (team reference; the AWS worker scripts are in `tools/aws/`).
- Tests: `python -m unittest discover -s tests` (CPU only; needs numpy, soundfile, librosa, pyyaml).
