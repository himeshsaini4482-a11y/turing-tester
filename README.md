# VoiceGym: human-closeness benchmark

Measures how close a TTS model sounds to real human speech. Each benchmark sentence has a
real LibriTTS-R recording; the TTS speaks the same sentence in that speaker's cloned voice;
SpeechJudge-GRM scores both clips.

```
per-sentence closeness = TTS score / human score × 100
overall closeness      = mean over sentences (also reported per speaker)
```

Each sentence is judged in both A/B orders and the scores are averaged before the ratio.
Full design and caveats: `VoiceGym_SpeechJudge_Project_Plan.md` §8.6.

## Benchmark set

- LibriTTS-R `test-clean` (CC BY 4.0), 10 speakers (5 M / 5 F) × 50 sentences = 500
- Sentences 3–15 s, ≥ 5 words; selection is deterministic (`--seed 1234`)
- Voice prompt per sentence: a different 4–10 s clip from the same speaker **and chapter**
- TTS model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (voice clone). The paper's Qwen2.5-0.5B-TTS
  has no official release.

## VM requirements

- Linux + NVIDIA GPU. SpeechJudge-GRM is a ~22 GB bf16 checkpoint: 48 GB+ VRAM recommended
  (A6000 / L40S / A100 / H100). Stages run one at a time, so the TTS and judge never share the GPU.
- ~60 GB free disk (judge 22 GB, Whisper 3 GB, Qwen3-TTS, LibriTTS-R ~3 GB, venvs)
- Python 3.10–3.12, internet access to Hugging Face, OpenSLR and GitHub

## Run

```bash
git clone <your-repo> voicegym && cd voicegym
bash scripts/setup_vm.sh                      # once; INSTALL_FLASH_ATTN=1 to also build flash-attn

RUN_NAME=smoke LIMIT=5 bash scripts/run_benchmark.sh   # 5-sentence end-to-end test first
bash scripts/run_sanity_check.sh              # human vs itself → should be ~100%
bash scripts/run_benchmark.sh                 # full 500-sentence benchmark → runs/base/
```

After the first run, **commit `benchmark/manifests/libritts_r_test_clean_10x50.csv`**. That file
is the locked benchmark set; every later run reuses it.

To benchmark a trained checkpoint on the same sentences:

```bash
RUN_NAME=trained TTS_MODEL=/path/to/checkpoint bash scripts/run_benchmark.sh
```

Options (env vars): `VOTES=3` (judge calls per A/B order), `LIMIT=N`, `SKIP_METRICS=1`.
Every stage is resumable: if the VM drops, re-run the same command.

## Direct Qwen 3.5 9B client

VoiceGym includes a minimal client that sends any text prompt directly to the
standalone Qwen 3.5 9B vLLM endpoint and prints the answer. It does **not** use
UAI `router.py`, retrieval, databases, chat history, or any other UAI application
data. Later code can invoke this client when it needs general text such as
candidate sentences for TTS or SFT.

```bash
cp .env.example .env
# Set VOICEGYM_QWEN_API_KEY in .env; obtain it from the endpoint administrator.

python scripts/qwen9b_client.py "Write five varied sentences for a TTS model to speak."

# stdin and interactive input also work:
echo "Explain why clear pauses matter in speech." | python scripts/qwen9b_client.py
python scripts/qwen9b_client.py
```

Use `--system` to set a system instruction, `--temperature` and `--max-tokens`
to override generation settings, or `--raw` to print the complete
OpenAI-compatible response JSON. The script uses only Python's standard library.

Endpoint configuration lives in `.env`:

| Variable | Purpose |
|---|---|
| `VOICEGYM_QWEN_BASE_URL` | OpenAI-compatible base URL; currently `https://api.uaitesting.com/general-llm/v1` |
| `VOICEGYM_QWEN_API_KEY` | Bearer token (secret; never commit it) |
| `VOICEGYM_QWEN_MODEL` | Served model name, `qwen/qwen3.5-9b` |
| `VOICEGYM_QWEN_TIMEOUT_SECONDS` | HTTP timeout for a generation request |
| `VOICEGYM_QWEN_TEMPERATURE` | Sentence diversity |
| `VOICEGYM_QWEN_MAX_TOKENS` | Maximum response tokens |

The gateway exposes only `GET /models` and `POST /chat/completions` under this
base URL. Missing/invalid tokens return HTTP 401, and other vLLM paths are not
published. Rotate a leaked token immediately.

Long runs should survive SSH disconnects, so start them in `tmux` or with `nohup`:

```bash
mkdir -p runs && nohup bash scripts/run_benchmark.sh > runs/base.log 2>&1 &
tail -f runs/base.log
```

## Outputs (`runs/<RUN_NAME>/`)

| File | Contents |
|---|---|
| `summary.json` | **Human-closeness**: overall, per speaker, per sentence (TTS/human scores) |
| `judgments.jsonl` | Every SpeechJudge call: order, scores, full written analysis |
| `metrics_summary.json` | WER (TTS and human), speaker similarity, UTMOS (TTS and human) |
| `metrics.jsonl` | Per-sentence metrics and Whisper transcripts |
| `generation_config.json` | TTS model, seed, settings |
| `tts/*.wav` | Generated audio |

Re-compute the summary without the GPU: `.venv-judge/bin/python benchmark/judge.py --out-dir runs/base --summarize-only`

## Layout

```
benchmark/prepare_data.py   stage 1: download LibriTTS-R, select the 500 sentences
benchmark/generate_tts.py   stage 2: Qwen3-TTS voice-clone generation
benchmark/judge.py          stage 3: SpeechJudge scoring + closeness formula
benchmark/metrics.py        stage 4: WER, speaker similarity, UTMOS
benchmark/data/             LibriSpeech test-clean speaker genders
scripts/qwen9b_client.py    send a prompt directly to the standalone Qwen 9B endpoint
scripts/                    setup_vm.sh, run_benchmark.sh, run_sanity_check.sh
```

Two venvs because the packages conflict: `qwen-tts` pins `transformers==4.57.3`,
SpeechJudge needs `transformers==4.52.3`. SpeechJudge code is pinned to commit `1fbd1c8`.

## Licences

SpeechJudge-GRM is CC BY-NC 4.0 (non-commercial). LibriTTS-R is CC BY 4.0. Qwen3-TTS is Apache-2.0.
