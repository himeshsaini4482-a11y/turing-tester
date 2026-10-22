"""Stage 2 (Fish Audio Drama 3): cloned-voice TTS through a running SGLang-Omni server.

Drama 3 (fishaudio/drama-3, 35B MoE, ~3B active) is too big to load in-process next to
the other models, so scripts/run_drama3.sh serves it with `sgl-omni serve` and this
script sends one /v1/audio/speech request per sentence. The voice prompt goes inline as
base64 WAV with its transcript, exactly like generate_tts.py's voice clone.

Writes the same outputs as generate_tts.py: <run-dir>/tts/<sentence_id>.wav,
<run-dir>/manifest.csv (input rows + tts_wav) and <run-dir>/generation_config.json.
Standard library only (runs in any venv). Resumable: existing output wavs are skipped.

  python benchmark/generate_drama3.py --manifest benchmark/manifests/ears_regular_26x19.csv \
      --run-dir runs/drama3_smoke --limit 5
"""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request

from common import read_manifest, write_manifest


def synthesize(url, text, prompt_wav, prompt_text, timeout):
    with open(prompt_wav, "rb") as f:
        prompt = base64.b64encode(f.read()).decode("ascii")
    body = {
        "input": text,
        "response_format": "wav",
        # Drama 3 rejects a seed ("seed not supported"), so its sampling is not reproducible
        "references": [{"data": prompt, "media_type": "audio/wav", "text": prompt_text}],
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Drama 3 server returned {e.code}: {e.read().decode(errors='replace')}") from e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="fishaudio/drama-3-int4", help="recorded in generation_config.json only")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (smoke test)")
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    tts_dir = os.path.join(args.run_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    for r in rows:
        r["tts_wav"] = os.path.join(tts_dir, f"{r['sentence_id']}.wav")

    todo = [r for r in rows if not os.path.exists(r["tts_wav"])]
    print(f"{len(rows) - len(todo)} already generated, {len(todo)} to generate")

    start = time.time()
    for i, r in enumerate(todo, 1):
        audio = synthesize(f"{args.server}/v1/audio/speech", r["text"], r["prompt_wav"], r["prompt_text"],
                           args.timeout)
        with open(r["tts_wav"] + ".part", "wb") as f:
            f.write(audio)
        os.replace(r["tts_wav"] + ".part", r["tts_wav"])
        elapsed = time.time() - start
        print(f"[{i}/{len(todo)}] {r['sentence_id']}  {elapsed / 60:.1f} min elapsed, "
              f"~{elapsed / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "server": "sglang-omni /v1/audio/speech",
            "source_manifest": args.manifest,
            "n_sentences": len(rows),
        }, f, indent=2)


if __name__ == "__main__":
    main()
