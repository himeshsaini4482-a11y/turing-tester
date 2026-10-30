"""Stage 1: download LibriTTS(-R) test-clean and select the locked benchmark set.

Selection (deterministic for a given --seed):
  - half male / half female speakers (genders from benchmark/data/test_clean_speakers.tsv)
  - per chapter, one voice-prompt clip (closest to --prompt-target-dur seconds); a test
    sentence is always cloned from a prompt of the same speaker and chapter
  - test sentences are --min-dur..--max-dur seconds, >= --min-words words, never a prompt clip

Writes a manifest CSV with one row per test sentence:
  speaker_id, gender, chapter_id, sentence_id, text, duration, human_wav,
  prompt_id, prompt_wav, prompt_text

Runs in the TTS environment (.venv-tts).
"""
import argparse
import csv
import math
import os
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict

import soundfile as sf

from common import write_manifest

CORPORA = {
    "libritts-r": ("https://www.openslr.org/resources/141/test_clean.tar.gz", "LibriTTS_R"),
    "libritts": ("https://www.openslr.org/resources/60/test-clean.tar.gz", "LibriTTS"),
}
SPEAKERS_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_clean_speakers.tsv")


def download_and_extract(url, data_root, corpus_dir):
    split_dir = os.path.join(data_root, corpus_dir, "test-clean")
    if os.path.isdir(split_dir):
        return split_dir

    os.makedirs(data_root, exist_ok=True)
    archive = os.path.join(data_root, os.path.basename(url))
    if not os.path.exists(archive):
        print(f"Downloading {url} ...")

        def progress(blocks, block_size, total):
            done = blocks * block_size
            if total > 0 and blocks % 2000 == 0:
                print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)

        urllib.request.urlretrieve(url, archive + ".part", progress)
        os.replace(archive + ".part", archive)

    print(f"Extracting {archive} ...")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(data_root, filter="data")
        else:
            tar.extractall(data_root)
    return split_dir


def scan(split_dir):
    clips = defaultdict(list)
    for root, _, files in os.walk(split_dir):
        for name in files:
            if not name.endswith(".wav"):
                continue
            uid = name[:-4]
            txt = os.path.join(root, uid + ".normalized.txt")
            if not os.path.exists(txt):
                continue
            with open(txt, encoding="utf-8") as f:
                text = f.read().strip()
            path = os.path.join(root, name)
            speaker, chapter = uid.split("_")[:2]
            clips[speaker].append({
                "id": uid,
                "chapter": chapter,
                "path": path,
                "text": text,
                "duration": sf.info(path).duration,
            })
    return clips


def speaker_pool(clips, args):
    """Returns ({chapter: prompt clip}, [eligible test clips])."""
    by_chapter = defaultdict(list)
    for c in clips:
        by_chapter[c["chapter"]].append(c)

    prompts, pool, seen_texts = {}, [], set()
    for chapter, chapter_clips in sorted(by_chapter.items()):
        candidates = [c for c in chapter_clips if args.prompt_min_dur <= c["duration"] <= args.prompt_max_dur]
        if not candidates:
            continue
        prompt = min(candidates, key=lambda c: (abs(c["duration"] - args.prompt_target_dur), c["id"]))
        prompts[chapter] = prompt
        seen_texts.add(prompt["text"])
        for c in sorted(chapter_clips, key=lambda c: c["id"]):
            if c is prompt or c["text"] in seen_texts:
                continue
            if args.min_dur <= c["duration"] <= args.max_dur and len(c["text"].split()) >= args.min_words:
                pool.append(c)
                seen_texts.add(c["text"])
    return prompts, pool


def read_genders():
    with open(SPEAKERS_TSV, newline="", encoding="utf-8") as f:
        return {row["speaker_id"]: row["gender"] for row in csv.DictReader(f, delimiter="\t")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=sorted(CORPORA), default="libritts-r")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--speakers", type=int, default=10)
    p.add_argument("--sentences-per-speaker", type=int, default=50)
    p.add_argument("--min-dur", type=float, default=3.0)
    p.add_argument("--max-dur", type=float, default=15.0)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--prompt-min-dur", type=float, default=4.0)
    p.add_argument("--prompt-max-dur", type=float, default=10.0)
    p.add_argument("--prompt-target-dur", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    url, corpus_dir = CORPORA[args.corpus]
    split_dir = download_and_extract(url, args.data_root, corpus_dir)
    if os.path.exists(args.out):
        # the locked set: only make sure its audio is on this machine (a fresh clone has none)
        print(f"Using locked manifest {args.out}; audio in {split_dir}")
        return
    clips = scan(split_dir)
    genders = read_genders()
    rng = random.Random(args.seed)
    n = args.sentences_per_speaker

    eligible = {"M": [], "F": []}
    pools = {}
    for speaker in sorted(clips, key=int):
        prompts, pool = speaker_pool(clips[speaker], args)
        pools[speaker] = (prompts, pool)
        if len(pool) >= n and genders.get(speaker) in eligible:
            eligible[genders[speaker]].append(speaker)

    wanted = {"M": math.ceil(args.speakers / 2), "F": args.speakers // 2}
    chosen = []
    for gender, k in wanted.items():
        if len(eligible[gender]) < k:
            sys.exit(f"Only {len(eligible[gender])} {gender} speakers have >= {n} eligible sentences; need {k}. "
                     f"Lower --sentences-per-speaker or relax the duration filters.")
        chosen += rng.sample(eligible[gender], k)

    rows = []
    for speaker in sorted(chosen, key=int):
        prompts, pool = pools[speaker]
        for c in sorted(rng.sample(pool, n), key=lambda c: c["id"]):
            prompt = prompts[c["chapter"]]
            rows.append({
                "speaker_id": speaker,
                "gender": genders[speaker],
                "chapter_id": c["chapter"],
                "sentence_id": c["id"],
                "text": c["text"],
                "duration": f"{c['duration']:.2f}",
                "human_wav": os.path.relpath(c["path"]).replace(os.sep, "/"),
                "prompt_id": prompt["id"],
                "prompt_wav": os.path.relpath(prompt["path"]).replace(os.sep, "/"),
                "prompt_text": prompt["text"],
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_manifest(args.out, rows)

    total_min = sum(float(r["duration"]) for r in rows) / 60
    print(f"Wrote {len(rows)} sentences from {len(chosen)} speakers ({total_min:.1f} min of human audio) to {args.out}")
    for speaker in sorted(chosen, key=int):
        print(f"  speaker {speaker} ({genders[speaker]}): {n} sentences, eligible pool {len(pools[speaker][1])}")


if __name__ == "__main__":
    main()
