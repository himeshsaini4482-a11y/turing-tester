#!/usr/bin/env python3
"""Optional LLM for VoiceGym: any OpenAI-compatible chat endpoint (a hosted API, or a local server
such as Ollama, LM Studio or vLLM). Standard library only.

VoiceGym uses it, when configured, to tag the judge's critiques into weaknesses and to write
training sentences aimed at them (sft/tag_weaknesses.py, sft/gen_texts.py). Without it, the
bundled sentence pool is used. Configuration, from the environment or a local .env (never committed):

  VOICEGYM_LLM_BASE_URL         e.g. http://localhost:11434/v1 (Ollama) or https://api.openai.com/v1
  VOICEGYM_LLM_MODEL            the served model name
  VOICEGYM_LLM_API_KEY          bearer token (optional for local servers)
  VOICEGYM_LLM_TEMPERATURE      default 0.7
  VOICEGYM_LLM_MAX_TOKENS       default 1024
  VOICEGYM_LLM_TIMEOUT_SECONDS  default 300
  VOICEGYM_LLM_NO_THINKING=1    send chat_template_kwargs.enable_thinking=false (vLLM reasoning models)

The key is sent only in the Authorization header and never logged. Only the text VoiceGym builds
is sent (judge critiques and sentence requests), never audio.

  python scripts/llm_client.py "Reply with exactly: LLM_OK"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """KEY=VALUE lines from .env, without overriding variables already set."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def settings() -> dict:
    """The endpoint settings; the team's older VOICEGYM_QWEN_* names are accepted as a fallback."""
    load_dotenv()

    def get(name, default=""):
        return (os.getenv(f"VOICEGYM_LLM_{name}") or os.getenv(f"VOICEGYM_QWEN_{name}") or default).strip()

    legacy = not os.getenv("VOICEGYM_LLM_BASE_URL") and bool(os.getenv("VOICEGYM_QWEN_BASE_URL"))
    return {
        "base_url": get("BASE_URL"),
        "api_key": get("API_KEY"),
        "model": get("MODEL", "qwen/qwen3.5-9b" if legacy else ""),
        "temperature": float(get("TEMPERATURE", "0.7")),
        "max_tokens": int(get("MAX_TOKENS", "1024")),
        "timeout": float(get("TIMEOUT_SECONDS", "300")),
        "no_thinking": legacy or os.getenv("VOICEGYM_LLM_NO_THINKING", "") == "1",
    }


def configured() -> bool:
    s = settings()
    return bool(s["base_url"] and s["model"])


def chat(prompt: str, system: str = "You are a helpful assistant.", temperature: float | None = None,
         max_tokens: int | None = None) -> dict:
    """One chat completion; returns the OpenAI-compatible response JSON."""
    s = settings()
    if not s["base_url"] or not s["model"]:
        raise RuntimeError("set VOICEGYM_LLM_BASE_URL and VOICEGYM_LLM_MODEL (and VOICEGYM_LLM_API_KEY if the "
                           "server needs one) to use an LLM")
    body = {"model": s["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": s["temperature"] if temperature is None else temperature,
            "max_tokens": s["max_tokens"] if max_tokens is None else max_tokens}
    if s["no_thinking"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json", "User-Agent": "VoiceGym-LLM-Client/1.0"}
    if s["api_key"]:
        headers["Authorization"] = f"Bearer {s['api_key']}"
    req = urllib.request.Request(f"{s['base_url'].rstrip('/')}/chat/completions",
                                 data=json.dumps(body).encode("utf-8"), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=s["timeout"]) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach the LLM endpoint: {exc.reason}") from exc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("prompt", nargs="*")
    p.add_argument("--system", default="You are a helpful assistant.")
    args = p.parse_args()
    prompt = " ".join(args.prompt).strip() or sys.stdin.read().strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    print(chat(prompt, system=args.system)["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
