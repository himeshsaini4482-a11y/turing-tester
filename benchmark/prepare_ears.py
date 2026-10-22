"""Stage 1 (EARS): download EARS speakers and select the locked benchmark set.

EARS (facebookresearch/ears_dataset, CC BY-NC 4.0) is studio-recorded, anechoic 48 kHz
speech. Every speaker reads the same scripted texts; only the read portions have
transcripts. We use the 20 neutral "regular"-style readings per speaker (7-18 s each):

  - speakers: native American English, half male / half female, sampled with --seed
  - voice prompt: the speaker's rainbow_01_regular (7.7 s, same text for everyone)
  - test sentences: the other 19 regular readings

Emotional, whisper, fast/slow/loud/pitch styles are left out: a TTS without style control
would lose on style, not naturalness.

Writes the same manifest columns as prepare_data.py. If --out already exists it is reused
as the locked set and only missing audio is re-downloaded (a fresh GPU worker has none).

Runs in the TTS environment (.venv-tts).
"""
import argparse
import json
import os
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

from common import read_manifest, write_manifest

REPO = "https://github.com/facebookresearch/ears_dataset"
RAW = "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main"
PROMPT = "rainbow_01_regular"


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def ensure_speakers(wanted, ears_dir):
    """wanted: {speaker: [wav names]}. Downloads in parallel (~9 MB/s per GitHub connection)."""
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda kv: ensure_speaker(kv[0], ears_dir, kv[1]), wanted.items()))


def ensure_speaker(speaker, ears_dir, names):
    """Download the speaker's zip and keep only the wavs in `names`."""
    spk_dir = os.path.join(ears_dir, speaker)
    if all(os.path.exists(os.path.join(spk_dir, n + ".wav")) for n in names):
        return
    archive = os.path.join(ears_dir, speaker + ".zip")
    print(f"Downloading EARS {speaker} (~600 MB) ...", flush=True)
    fetch(f"{REPO}/releases/download/dataset/{speaker}.zip", archive)
    with zipfile.ZipFile(archive) as z:
        for n in names:
            z.extract(f"{speaker}/{n}.wav", ears_dir)
    os.remove(archive)


def select(args, transcripts, stats):
    readings = sorted(k for k in transcripts if k.endswith("_regular") and k != PROMPT)
    rng = random.Random(args.seed)
    chosen = []
    for gender, k in (("male", (args.speakers + 1) // 2), ("female", args.speakers // 2)):
        pool = sorted(s for s, v in stats.items()
                      if v["gender"] == gender and v["native language"] == "american english")
        if len(pool) < k:
            sys.exit(f"Only {len(pool)} native American English {gender} speakers; need {k}")
        chosen += rng.sample(pool, k)
    return sorted(chosen), readings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--speakers", type=int, default=26)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=0,
                   help="with a locked manifest: only fetch audio for its first N sentences (smoke test)")
    args = p.parse_args()

    ears_dir = os.path.join(args.data_root, "EARS")
    os.makedirs(ears_dir, exist_ok=True)
    transcripts = json.load(open(fetch(f"{RAW}/transcripts.json", os.path.join(ears_dir, "transcripts.json"))))

    if os.path.exists(args.out):
        rows = read_manifest(args.out)
        speakers = sorted({r["speaker_id"] for r in rows})
        print(f"Using locked manifest {args.out} ({len(rows)} sentences, {len(speakers)} speakers)")
        if args.limit:
            rows = rows[: args.limit]
            speakers = sorted({r["speaker_id"] for r in rows})
        ensure_speakers({s: [PROMPT] + [r["sentence_id"][len(s) + 1:] for r in rows if r["speaker_id"] == s]
                         for s in speakers}, ears_dir)
        return

    stats = json.load(open(fetch(f"{RAW}/speaker_statistics.json", os.path.join(ears_dir, "speaker_statistics.json"))))
    speakers, readings = select(args, transcripts, stats)
    ensure_speakers({s: [PROMPT] + readings for s in speakers}, ears_dir)
    rows = []
    for speaker in speakers:
        prompt_wav = f"{ears_dir}/{speaker}/{PROMPT}.wav"
        for name in readings:
            wav = f"{ears_dir}/{speaker}/{name}.wav"
            rows.append({
                "speaker_id": speaker,
                "gender": "M" if stats[speaker]["gender"] == "male" else "F",
                "chapter_id": "",
                "sentence_id": f"{speaker}_{name}",
                "text": transcripts[name],
                "duration": f"{sf.info(wav).duration:.2f}",
                "human_wav": wav,
                "prompt_id": f"{speaker}_{PROMPT}",
                "prompt_wav": prompt_wav,
                "prompt_text": transcripts[PROMPT],
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_manifest(args.out, rows)
    total_min = sum(float(r["duration"]) for r in rows) / 60
    print(f"Wrote {len(rows)} sentences from {len(speakers)} speakers ({total_min:.1f} min of human audio) to {args.out}")
    for speaker in speakers:
        print(f"  {speaker} ({stats[speaker]['gender']})")


if __name__ == "__main__":
    main()
