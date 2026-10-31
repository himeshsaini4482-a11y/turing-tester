#!/usr/bin/env python3
"""Send a prompt directly to Turing Tester's standalone Qwen 3.5 9B endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*", help="prompt text; reads stdin or asks interactively when omitted")
    parser.add_argument("--system", default="You are a helpful general-purpose assistant.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--raw", action="store_true", help="print the complete OpenAI-compatible JSON response")
    return parser.parse_args()


def get_prompt(parts: list[str]) -> str:
    if parts:
        return " ".join(parts).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return input("Prompt: ").strip()


def chat(
    prompt: str,
    system: str = "You are a helpful general-purpose assistant.",
    temperature: float | None = None,
    max_tokens: int | None = None,
    env_file: Path = REPO_ROOT / ".env",
) -> dict:
    """Send one prompt and return the OpenAI-compatible response JSON.

    Other Turing Tester stages import this (e.g. sft/gen_texts.py); main() is the CLI around it.
    """
    load_dotenv(env_file)
    base_url = os.getenv("VOICEGYM_QWEN_BASE_URL", "").strip()
    api_key = os.getenv("VOICEGYM_QWEN_API_KEY", "").strip()
    model = os.getenv("VOICEGYM_QWEN_MODEL", "qwen/qwen3.5-9b").strip()
    if not base_url or not api_key:
        raise RuntimeError(
            "set VOICEGYM_QWEN_BASE_URL and VOICEGYM_QWEN_API_KEY in .env "
            "(copy .env.example first)"
        )

    if temperature is None:
        temperature = float(os.getenv("VOICEGYM_QWEN_TEMPERATURE", "0.7"))
    if max_tokens is None:
        max_tokens = int(os.getenv("VOICEGYM_QWEN_MAX_TOKENS", "1024"))
    timeout = float(os.getenv("VOICEGYM_QWEN_TIMEOUT_SECONDS", "300"))

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TuringTester-Qwen9B-Client/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Qwen endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach Qwen endpoint: {exc.reason}") from exc


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    prompt = get_prompt(args.prompt)
    if not prompt:
        raise ValueError("prompt must not be empty")

    result = chat(prompt, system=args.system, temperature=args.temperature,
                  max_tokens=args.max_tokens, env_file=args.env_file)

    if args.raw:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    try:
        print(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Qwen response is missing choices[0].message.content") from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
