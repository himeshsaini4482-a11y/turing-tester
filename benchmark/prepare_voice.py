"""Prepare your own voice recordings (voices/<name>/) for benchmarking and training.

Input: audio files (wav, flac, mp3, ogg, m4a via soundfile/librosa) in voices/<name>/, and optionally
metadata.csv with columns file,text (missing transcripts are made with Whisper large-v3).

Clips shorter than 1.5 s or longer than 20 s are left out (the training code's limits). The rest are
converted to mono 16-bit wav and split, deterministically:
  test   max(20, 10%) clips: the held-out set for `--set voices/<name>` and the before/after scores
  dev    max(10, 5%) clips: what SFT and DPO use to accept a checkpoint or pick a round (never the test clips)
  train  the rest
The voice prompt (the reference clip a model clones from) is the training clip closest to 8 s
(within 4-12 s); test sentences are always cloned from it, never from themselves.

Writes data/voices/<name>/:
  clips/<id>.wav, clips.jsonl (every clip with its split), test.csv and dev.csv (benchmark manifest columns),
  train.jsonl (training records: id, audio, text, ref_audio, speaker, duration), summary.json

Runs in the TTS environment. Resumable: transcripts are cached in transcripts.json.

  python benchmark/prepare_voice.py --voice voices/my-voice --out data/voices/my-voice
"""
import argparse
import csv
import json
import os
import random
import sys

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
MIN_S, MAX_S = 1.5, 20.0
PROMPT_TARGET, PROMPT_MIN, PROMPT_MAX = 8.0, 4.0, 12.0
MIN_TEST, TEST_SHARE, MIN_DEV, DEV_SHARE, MIN_TRAIN = 20, 0.10, 10, 0.05, 20
SEED = 1234


def read_metadata(voice_dir):
    path = os.path.join(voice_dir, "metadata.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "file" in sample.splitlines()[0].lower() if sample else False
        reader = csv.reader(f, delimiter="|" if "|" in sample.splitlines()[0] else ",")
        for i, row in enumerate(reader):
            if not row or (i == 0 and has_header):
                continue
            if len(row) < 2:
                continue
            name = os.path.basename(row[0].strip())
            out[name] = out[os.path.splitext(name)[0]] = ",".join(row[1:]).strip()
    return out


def stable_id(name, used):
    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.splitext(name)[0]) or "clip"
    i, cand = 1, base
    while cand in used:
        i += 1
        cand = f"{base}_{i}"
    used.add(cand)
    return cand


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voice", required=True, help="voices/<name>")
    p.add_argument("--out", required=True, help="data/voices/<name>")
    p.add_argument("--asr-model", default="openai/whisper-large-v3")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import librosa
    import numpy as np
    import soundfile as sf

    name = os.path.basename(os.path.normpath(args.voice))
    files = sorted(f for f in os.listdir(args.voice) if f.lower().endswith(AUDIO_EXT))
    if not files:
        sys.exit(f"no audio files in {args.voice} (wav, flac, mp3, ogg, m4a)")
    meta = read_metadata(args.voice)
    clip_dir = os.path.join(args.out, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    clips, skipped, used = [], {"too_short": 0, "too_long": 0, "unreadable": 0}, set()
    for f in files:
        src = os.path.join(args.voice, f)
        cid = stable_id(f, used)
        dst = os.path.join(clip_dir, cid + ".wav")
        try:
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                wav, sr = librosa.load(src, sr=None, mono=True)
                peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
                if peak > 0.99:
                    wav = wav * (0.99 / peak)
                sf.write(dst, wav, sr, subtype="PCM_16")
            dur = sf.info(dst).duration
        except Exception as e:  # noqa: BLE001  (one bad file must not stop the rest)
            print(f"[warn] {f}: cannot read ({type(e).__name__}: {e}); skipped", flush=True)
            skipped["unreadable"] += 1
            continue
        if dur < MIN_S:
            skipped["too_short"] += 1
            continue
        if dur > MAX_S:
            skipped["too_long"] += 1
            print(f"[warn] {f}: {dur:.1f} s is longer than {MAX_S:.0f} s; split it into sentences to use it", flush=True)
            continue
        clips.append({"id": cid, "file": f, "wav": dst.replace("\\", "/"), "duration": round(dur, 3),
                      "text": meta.get(f) or meta.get(os.path.splitext(f)[0]) or ""})

    cache_path = os.path.join(args.out, "transcripts.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    need = [c for c in clips if not c["text"] and c["file"] not in cache]
    if need:
        import torch
        from transformers import pipeline
        device = args.device if torch.cuda.is_available() or args.device == "mps" else "cpu"
        print(f"Transcribing {len(need)} clips with {args.asr_model} (no transcript in metadata.csv)", flush=True)
        asr = pipeline("automatic-speech-recognition", model=args.asr_model, device=device,
                       torch_dtype=torch.float16 if "cuda" in device else torch.float32)
        for i in range(0, len(need), 16):
            batch = need[i:i + 16]
            outs = asr([{"raw": librosa.load(c["wav"], sr=16000)[0], "sampling_rate": 16000} for c in batch],
                       batch_size=16, generate_kwargs={"language": "english", "task": "transcribe"})
            for c, o in zip(batch, outs):
                cache[c["file"]] = o["text"].strip()
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=1, ensure_ascii=False)
            print(f"  {min(i + 16, len(need))}/{len(need)}", flush=True)
    for c in clips:
        c["text"] = c["text"] or cache.get(c["file"], "")
        c["transcript"] = "metadata.csv" if c["file"] in meta or os.path.splitext(c["file"])[0] in meta else "whisper"
    empty = [c for c in clips if not c["text"]]
    clips = [c for c in clips if c["text"]]
    if empty:
        print(f"[warn] {len(empty)} clips have no words (silence?) and are left out", flush=True)

    n = len(clips)
    n_test = max(MIN_TEST, round(TEST_SHARE * n))
    n_dev = max(MIN_DEV, round(DEV_SHARE * n))
    if n - n_test - n_dev < MIN_TRAIN:
        sys.exit(f"{args.voice}: {n} usable clips; at least {MIN_TEST + MIN_DEV + MIN_TRAIN} are needed "
                 f"({MIN_TEST} held out for testing, {MIN_DEV} for checking training). "
                 "Aim for 30+ minutes in 3-15 s clips.")
    order = sorted(clips, key=lambda c: c["id"])
    random.Random(SEED).shuffle(order)
    prompt_pool = [c for c in order if PROMPT_MIN <= c["duration"] <= PROMPT_MAX] or order
    prompt = min(prompt_pool, key=lambda c: (abs(c["duration"] - PROMPT_TARGET), c["id"]))
    rest = [c for c in order if c is not prompt]
    test, dev, train = rest[:n_test], rest[n_test:n_test + n_dev], rest[n_test + n_dev:] + [prompt]
    for split, group in (("test", test), ("dev", dev), ("train", train)):
        for c in group:
            c["split"] = split

    with open(os.path.join(args.out, "clips.jsonl"), "w", encoding="utf-8") as f:
        for c in sorted(clips, key=lambda c: c["id"]):
            f.write(json.dumps({**c, "prompt": c is prompt}, ensure_ascii=False) + "\n")
    cols = ["speaker_id", "gender", "chapter_id", "sentence_id", "text", "duration", "human_wav",
            "prompt_id", "prompt_wav", "prompt_text"]
    for split, group in (("test", test), ("dev", dev)):
        with open(os.path.join(args.out, f"{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for c in sorted(group, key=lambda c: c["id"]):
                w.writerow({"speaker_id": name, "gender": "", "chapter_id": "", "sentence_id": f"{name}_{c['id']}",
                            "text": c["text"], "duration": f"{c['duration']:.2f}", "human_wav": c["wav"],
                            "prompt_id": f"{name}_{prompt['id']}", "prompt_wav": prompt["wav"],
                            "prompt_text": prompt["text"]})
    # training records: the prompt clip as reference for 70%, another 4-10 s training clip otherwise
    refs = [c for c in train if 4.0 <= c["duration"] <= 10.0 and c is not prompt]
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for c in sorted(train, key=lambda c: c["id"]):
            rng = random.Random(f"{SEED}:{c['id']}")
            others = [r for r in refs if r is not c]
            ref = prompt if (c is not prompt and (rng.random() < 0.7 or not others)) else (
                rng.choice(others) if others else prompt)
            f.write(json.dumps({"id": f"voice_{name}_{c['id']}", "audio": c["wav"], "text": c["text"],
                                "ref_audio": ref["wav"], "speaker": name, "duration": c["duration"],
                                "category": "voice", "source": "voice"}, ensure_ascii=False) + "\n")
    minutes = sum(c["duration"] for c in clips) / 60
    summary = {"voice": name, "n_clips": n, "minutes": round(minutes, 1), "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
               "train_minutes": round(sum(c["duration"] for c in train) / 60, 1),
               "prompt": {"id": prompt["id"], "duration": prompt["duration"], "text": prompt["text"]},
               "skipped": skipped, "no_words": len(empty),
               "transcripts": {"metadata.csv": sum(c["transcript"] == "metadata.csv" for c in clips),
                               "whisper": sum(c["transcript"] == "whisper" for c in clips)}}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"{name}: {n} clips, {minutes:.1f} min ({len(train)} train, {len(dev)} dev, {len(test)} test); prompt {prompt['id']} "
          f"({prompt['duration']:.1f} s); skipped {skipped}", flush=True)
    if minutes < 30:
        print(f"[warn] only {minutes:.0f} minutes of audio; 30+ minutes is recommended for training", flush=True)


if __name__ == "__main__":
    main()
