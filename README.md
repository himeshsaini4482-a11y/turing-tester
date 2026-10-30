# VoiceGym

Measure how close a text-to-speech (TTS) model sounds to real human speech, then make it
closer with SFT and DPO, all on your own machine.

> **Status: design draft.** This README describes the `voicegym` command being built now.
> The commands below do not work yet. Until they do, the experiment scripts are documented in
> [`docs/internal_pipeline.md`](docs/internal_pipeline.md), and what we have measured so far is in
> [`RESULTS.md`](RESULTS.md).

```
your TTS model ──► voicegym benchmark ──► score vs real human recordings
      │
      ├──► voicegym sft ──► learns from human recordings, your voice, or a bigger model
      │
      └──► voicegym dpo ──► learns from its own best and worst samples
                               └──► every run ends with a before/after benchmark
```

## What you need

| | |
|---|---|
| **GPU** | NVIDIA with **24 GB+** of memory (Windows or Linux), or an Apple Silicon Mac with **32 GB+** of memory (best effort, much slower) |
| **Disk** | about 100 GB free for benchmarking, about 250 GB for SFT on the EARS dataset |
| **Software** | `git` and an up-to-date NVIDIA driver. Everything else (Python, libraries, models, datasets) is installed automatically the first time a command needs it |
| **Internet** | for the first run of each command (downloads); after that, runs work offline except API benchmarks |

Check your machine before anything else:

```bash
git clone https://github.com/himeshsaini4482-a11y/voicegym.git
cd voicegym
./voicegym doctor            # Linux / macOS
.\voicegym doctor            # Windows (PowerShell or cmd)
```

`doctor` reports the GPU, free memory, free disk and what is already installed, and tells you
which commands your machine can run.

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

Every benchmark speaks the same locked set of sentences in the voices of real speakers, then
compares each clip with the real person's recording of that sentence.

### Local model weights

Copy the model folder (the one with `config.json` and the weights) into `models/` and run:

```bash
./voicegym benchmark --model models/my-tts
```

VoiceGym detects the model type from the folder:

| Model in the folder | What happens |
|---|---|
| Qwen3-TTS, MOSS-TTS-Nano, Orpheus | recognised automatically |
| another LLM-style TTS (a language-model backbone that outputs audio-codec tokens) | add a short `voicegym.yaml` in the folder that names its codec and prompt format ([guide](docs/adding_a_model.md)) |
| anything else (VITS, Kokoro, F5-TTS, …) | add a `synthesize.py` with two functions, `load()` and `synthesize(text, ref_audio, ref_text)` ([guide](docs/adding_a_model.md)) |

### A TTS API

Describe the API in one small file in `apis/` and keep the key in an environment variable:

```bash
export ELEVENLABS_API_KEY=...            # Windows: $env:ELEVENLABS_API_KEY="..."
./voicegym benchmark --api apis/elevenlabs.yaml
```

Ready-made configs: `apis/openai-compatible.yaml` (any server with `/v1/audio/speech`) and
`apis/elevenlabs.yaml`. For any other provider, copy `apis/template.yaml`:

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
voice_cloning: per_request                   # per_request | create_voice | none
requests_per_minute: 60
```

If an API cannot clone voices (`voice_cloning: none`), its clips are compared with a different
speaker's recording. That makes the comparison less fair, so the report marks speaker similarity
as not available and adds a warning.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--set` | `ears` | sentence set: `ears` (494 clips, 26 speakers), `libritts-r` (500 clips, 10 speakers), or `voices/<name>` (your own recordings) |
| `--judges` | `grm,btrm` | which SpeechJudge judges to run |
| `--limit N` | all | first N sentences only, for a quick test |
| `--name` | from the model | run folder name under `runs/` |

### What you get

`runs/<name>/report.md`, plus the raw numbers as JSON:

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
./voicegym sft --model models/my-tts                              # learn from EARS human recordings
./voicegym sft --model models/my-tts --voice voices/my-voice      # learn your own voice
./voicegym sft --model models/my-tts --teacher models/big-tts     # learn from a bigger model
```

Sources can be combined, for example `--voice voices/my-voice --teacher models/big-tts`. A
teacher can also be an API (`--teacher apis/my-provider.yaml`) if its terms allow training on
its output.

**Your own voice.** Put the recordings in a folder:

```
voices/my-voice/
  clip_001.wav
  clip_002.wav
  ...
  metadata.csv        # optional: file,text   (missing transcripts are made with Whisper)
```

Use clean, single-speaker recordings: aim for at least 30 minutes, in clips of 3–15 seconds.
VoiceGym holds back 10% of the clips (at least 20) as a test set, so you also get a benchmark on
your own voice.

**Training sentences** (used when a teacher speaks new text):

- **Default:** a sentence pool shipped with VoiceGym.
- `--texts texts/my_sentences.txt`: your own sentences, one per line.
- **Optional LLM:** set `VOICEGYM_LLM_BASE_URL`, `VOICEGYM_LLM_API_KEY` and
  `VOICEGYM_LLM_MODEL` (any OpenAI-compatible server, including local ones like Ollama or
  LM Studio). VoiceGym then writes sentences aimed at the weaknesses the judge found.
- `--review-texts` (optional): after the judge diagnoses the model's weaknesses, VoiceGym shows
  them and waits so you can add your own sentences before training.

**Modes:**

- `--mode quick` (default): one training run with the settings from our experiments.
- `--mode full`: the complete procedure from our experiments (pilot comparison of data sources,
  then training rounds). Slower, but it chooses the recipe for you.

Every SFT run benchmarks the model before and after training and keeps a checkpoint only if WER,
speaker similarity and UTMOS stay within guard limits. The result is
`runs/sft/<name>/final/` and `runs/sft/<name>/report.md`.

---

## 3. DPO: learn from its own best and worst samples

```bash
./voicegym dpo --model runs/sft/my-run/final
```

DPO does not learn from recordings. It learns only from the model's own samples, so it needs
no training data. To speak, the model clones a voice from a short reference clip; by default
those are the EARS speakers' voices, which improves the model across many voices.

`--voice voices/my-voice` (optional) makes the model clone **your** voice during DPO, so the
improvement is aimed at the voice you will deploy. Your held-out clips are then also used for the
before/after score. Use it only if you want the model to speak in your voice, for example after
`voicegym sft --voice voices/my-voice`.

DPO works exactly as in our experiments. Each round:

1. The model speaks each new sentence 4 times.
2. SpeechJudge-BTRM scores the samples.
3. The best and the worst valid sample form a pair. Samples with too many wrong words, the wrong
   voice or broken timing are excluded.
4. The model is trained to prefer the best over the worst, staying close to where it started.

It runs up to 3 rounds and stops early when a round stops helping. The result is
`runs/dpo/<name>/final/`, with a before/after benchmark. In our runs, DPO did not give a clear gain
on top of SFT (see `RESULTS.md`), so treat it as an experiment.

---

## What can be trained

| | Benchmark | SFT | DPO |
|---|---|---|---|
| Qwen3-TTS, MOSS-TTS-Nano, Orpheus | ✅ | ✅ | ✅ |
| other LLM-style TTS, with `voicegym.yaml` | ✅ | ✅ | ✅ |
| any other model, with `synthesize.py` | ✅ | with your own training script | ❌ |
| a TTS API | ✅ | ❌ (can be a teacher) | ❌ |

DPO needs the model to report how likely each audio token was, which only LLM-style TTS models
do. APIs cannot be trained because their weights are not available.

## How long it takes

On one NVIDIA L40S (48 GB), a full 494-sentence benchmark of MOSS-TTS-Nano with both judges
took about 4.7 hours. A 24 GB card is slower: the GRM judge (about 22 GB) does not fully fit, so
part of it runs from system memory. VoiceGym prints its estimate after the first few sentences.
Use `--limit 10` for a quick check first. Every command can be stopped and re-run; it continues
where it left off.

## Licences and privacy

- SpeechJudge (both judges) and the EARS dataset are **CC BY-NC 4.0: non-commercial use only**.
  LibriTTS-R is CC BY 4.0. Your own models and APIs keep their own terms.
- API keys are read from environment variables only and never written to files or logs.
- `models/`, `voices/` and `runs/` audio are never committed to git.

## More

- [`RESULTS.md`](RESULTS.md): what we measured so far.
- [`docs/adding_a_model.md`](docs/adding_a_model.md): `voicegym.yaml` and `synthesize.py` (written with the CLI).
- [`VoiceGym_SpeechJudge_Project_Plan.md`](VoiceGym_SpeechJudge_Project_Plan.md): the original design and the reasoning behind the metrics.
- [`docs/internal_pipeline.md`](docs/internal_pipeline.md): the experiment scripts (team reference).
