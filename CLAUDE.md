# Turing Tester repository guidance

## Project purpose

Turing Tester benchmarks TTS human-closeness using LibriTTS-R, Qwen3-TTS, and
SpeechJudge-GRM. Preserve the locked benchmark split and keep automated judge
results distinct from human-listener claims.

## SFT pipeline (MOSS-TTS-Nano)

Current work: SFT of `OpenMOSS-Team/MOSS-TTS-Nano-100M` to raise its SpeechJudge-BTRM score on
the EARS benchmark. The spec is `sft/PLAN.md`. If you are running on the VM (UAI server) to
execute it, follow `sft/RUNBOOK.md` step by step. Never run `scripts/gpu_down.sh` mid-project,
and never tune on the 494 benchmark sentences.

## Standalone Qwen 3.5 9B endpoint

`scripts/qwen9b_client.py` is a minimal general-purpose client: it accepts prompt
text and prints Qwen's response. Other Turing Tester stages may invoke it when they
need text, including future TTS/SFT sentence generation. Its production base URL
is:

```text
https://api.uaitesting.com/general-llm/v1
```

Supported routes are `GET /models` and `POST /chat/completions`. This is a direct
nginx-to-vLLM path. It does not call the UAI chat service, `router.py`, retrieval,
databases, chat history, JWT auth, or other UAI application components.

Configuration is read from `.env` (see `.env.example`):

- `VOICEGYM_QWEN_BASE_URL`
- `VOICEGYM_QWEN_API_KEY`
- `VOICEGYM_QWEN_MODEL`
- `VOICEGYM_QWEN_TIMEOUT_SECONDS`
- `VOICEGYM_QWEN_TEMPERATURE`
- `VOICEGYM_QWEN_MAX_TOKENS`

Never commit `.env`, bearer tokens, generated credentials, user audio, or private
SpeechJudge analyses. Keep the token in the `Authorization: Bearer ...` header,
not query parameters or logs. Do not send UAI data or personal/confidential
content to this general-purpose endpoint.

## Development checks

The Qwen client intentionally uses only the Python standard library so it works
in either project virtual environment. Before committing changes to it, run:

```bash
python -m compileall -q benchmark scripts
```

For a live smoke test, configure a local ignored `.env` and run
`python scripts/qwen9b_client.py "Reply with exactly: QWEN_OK"`.
