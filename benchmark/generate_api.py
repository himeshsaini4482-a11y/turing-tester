"""Stage 2 (TTS API): one request per benchmark sentence to the API described by an apis/*.yaml.

Voice cloning, as the config says:
  per_request   the row's prompt clip is sent with every request ({ref_audio_base64}, {ref_text})
  create_voice  one cloned voice per prompt clip, created once (ids cached in <run-dir>/api_voices.json)
                and deleted at the end when the config has delete_voice
  none          the API's own voice; the report marks speaker similarity as not comparable

Audio is converted to mono wav. Writes the same outputs as generate_tts.py: <run-dir>/tts/<sentence_id>.wav,
<run-dir>/manifest.csv and <run-dir>/generation_config.json. Resumable: existing wavs are skipped.
Keys are read from environment variables by the config and never logged.

Runs in the TTS environment (soundfile for mp3/flac/ogg decoding).
"""
import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turing_tester import apiconfig  # noqa: E402

from common import read_manifest, write_manifest  # noqa: E402


def to_wav(data, fmt, sample_rate, path):
    import numpy as np
    import soundfile as sf
    if fmt == "pcm_s16le":
        wav, sr = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype("float32") / 32768.0, sample_rate
    else:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if len(wav) == 0:
        raise apiconfig.ApiError("empty audio")
    sf.write(path + ".part.wav", wav, sr)
    os.replace(path + ".part.wav", path)
    return len(wav) / sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", required=True, help="apis/<provider>.yaml")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234, help="unused (APIs sample on their side)")
    args = p.parse_args()

    cfg = apiconfig.load(args.config)
    missing = apiconfig.missing_env(cfg)
    if missing:
        sys.exit(f"set these environment variables first: {', '.join(missing)}")
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    tts_dir = os.path.join(args.run_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    for r in rows:
        r["tts_wav"] = os.path.join(tts_dir, f"{r['sentence_id']}.wav")
    todo = [r for r in rows if not os.path.exists(r["tts_wav"])]
    print(f"{len(rows) - len(todo)} already generated, {len(todo)} to generate ({cfg['name']}, "
          f"voice cloning: {cfg['voice_cloning']})", flush=True)

    client = apiconfig.Client(cfg)
    voices_path = os.path.join(args.run_dir, "api_voices.json")
    voices = json.load(open(voices_path, encoding="utf-8")) if os.path.exists(voices_path) else {}
    failed = []
    start = time.time()
    for i, r in enumerate(todo, 1):
        voice_id = None
        if cfg["voice_cloning"] == "create_voice":
            key = r.get("prompt_id") or os.path.basename(r["prompt_wav"])
            if key not in voices:
                voices[key] = client.create_voice(r["prompt_wav"], key)
                with open(voices_path, "w", encoding="utf-8") as f:
                    json.dump(voices, f, indent=2)
            voice_id = voices[key]
        try:
            data, fmt = client.synthesize(r["text"], r.get("prompt_wav"), r.get("prompt_text"), voice_id)
            secs = to_wav(data, fmt, cfg["response"].get("sample_rate"), r["tts_wav"])
        except apiconfig.ApiError as e:
            print(f"[warn] {r['sentence_id']}: {e}", flush=True)
            failed.append({"sentence_id": r["sentence_id"], "error": str(e)[:300]})
            continue
        elapsed = time.time() - start
        print(f"[{i}/{len(todo)}] {r['sentence_id']}  {secs:.1f} s audio, {elapsed / 60:.1f} min elapsed, "
              f"~{elapsed / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    if cfg.get("delete_voice") and voices and not failed:
        for vid in voices.values():
            client.delete_voice(vid)
        os.remove(voices_path)
    if failed:
        with open(os.path.join(args.run_dir, "api_failed.json"), "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        sys.exit(f"{len(failed)} of {len(todo)} requests failed (see {args.run_dir}/api_failed.json); "
                 "re-run the same command to retry them")
    write_manifest(os.path.join(args.run_dir, "manifest.csv"), rows)
    with open(os.path.join(args.run_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": f"api:{cfg['name']}", "url_host": cfg["url"].split("/")[2] if "//" in cfg["url"] else "",
                   "voice_cloning": cfg["voice_cloning"], "response_format": cfg["response"].get("format", "wav"),
                   "source_manifest": args.manifest, "n_sentences": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
