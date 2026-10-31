"""SFT stage S2a (EARS): download all 107 EARS speakers and write the file inventory.

Downloads every speaker's zip (~600 MB, `{EARS_REPO}/releases/download/dataset/pXXX.zip`) to the
scratch NVMe, extracts every wav into `$SCR/ears/<spk>/`, deletes the zip and writes
`$SCR/ears/<spk>/.complete` (resumable per speaker). Then classifies every wav and flags it per
sft/INTERFACES.md §2.1:

  - bench_file:      (speaker, reading name) is one of the 494 benchmark rows
  - is_bench_text:   the transcript is one of the 19 benchmark texts
  - f1_styled_bench: benchmark speaker x benchmark text x style != regular (1,404 files)
  - dev_speaker / dev_leak_probe: the 4 held-out dev speakers (and their styled benchmark texts)
  - excluded:        "benchmark_file" | "nonverbal" | "other" | null (dev speakers and F1 files
                     are flagged, never excluded here)

Outputs:
  $SCR/ears/<spk>/<name>.wav                    raw EARS audio (48 kHz)
  $SCR/ears/{transcripts,speaker_statistics}.json
  runs/sft/data/ears_inventory.jsonl            one line per wav (all downloaded speakers)
  runs/sft/data/inventory_summary.json          counts and hours per class, flag, exclusion

Env: TTS (.venv-tts; needs soundfile).

  python sft/ears_inventory.py [--speakers p001,p002] [--workers 8]
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

DATA = sftlib.DATA
INVENTORY = DATA / "ears_inventory.jsonl"
SUMMARY = DATA / "inventory_summary.json"
DEV_SPEAKERS = Path(__file__).resolve().parent / "data" / "dev_speakers.json"
FLAGS = ("is_bench_text", "bench_file", "f1_styled_bench", "dev_speaker", "dev_leak_probe")


# ---- download -------------------------------------------------------------------------------

def fetch(url, path, retries=5, timeout=60):
    """Download url to path via path.part (never leaves a half file at path). Retries with backoff."""
    path = Path(path)
    if path.exists():
        return path
    part = path.with_name(path.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            os.replace(part, path)
            return path
        except Exception as e:  # network errors, timeouts, HTTP 5xx
            if part.exists():
                part.unlink()
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"  retry {attempt}/{retries} for {url} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def ensure_speaker(speaker, ears_dir, retries=4):
    """Download, extract every wav (flat, by basename), delete the zip, write .complete."""
    spk_dir = Path(ears_dir) / speaker
    marker = spk_dir / ".complete"
    if marker.exists():
        return speaker, False
    spk_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(ears_dir) / f"{speaker}.zip"
    url = f"{sftlib.EARS_REPO}/releases/download/dataset/{speaker}.zip"
    for attempt in range(1, retries + 1):
        try:
            fetch(url, archive)
            n = 0
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if info.is_dir() or not base.lower().endswith(".wav") or base.startswith("."):
                        continue
                    part = spk_dir / (base + ".part")
                    with z.open(info) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                    os.replace(part, spk_dir / base)
                    n += 1
            if n == 0:
                raise RuntimeError(f"{archive} contains no wav files")
            archive.unlink()
            marker.write_text(f"{n}\n")
            return speaker, True
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as e:
            if archive.exists():
                archive.unlink()     # corrupt / truncated download: fetch again
            if attempt == retries:
                raise
            print(f"  {speaker}: attempt {attempt} failed ({e}); retrying", flush=True)
            time.sleep(10 * attempt)


# ---- flags (pure, tested without audio) ------------------------------------------------------

def benchmark_sets(manifest_rows):
    """-> (bench_pairs {(spk, name)}, bench_speakers, bench_texts, reading names in manifest order)."""
    pairs, texts, names = set(), set(), []
    for r in manifest_rows:
        spk = r["speaker_id"]
        prefix = spk + "_"
        if not r["sentence_id"].startswith(prefix):
            raise ValueError(f"unexpected sentence_id {r['sentence_id']} for {spk}")
        name = r["sentence_id"][len(prefix):]
        pairs.add((spk, name))
        texts.add(text_key(r["text"]))
        if name not in names:
            names.append(name)
    return pairs, {s for s, _ in pairs}, texts, names


def text_key(text):
    """Comparison key for 'same text' (robust to whitespace/punctuation/case differences)."""
    return " ".join(sftlib.normalize_words(text))


def inventory_row(speaker, name, path, duration, transcripts, stats, bench, dev_speakers):
    """One §2.1 record. bench = benchmark_sets(...) output."""
    bench_pairs, bench_speakers, bench_texts, _ = bench
    cls, style, emotion = sftlib.classify_ears(name)
    transcript = transcripts.get(name) if cls.startswith("read-") else None
    is_bench_text = transcript is not None and text_key(transcript) in bench_texts
    bench_file = (speaker, name) in bench_pairs
    styled = style is not None and style != "regular"
    dev = speaker in dev_speakers
    if bench_file:
        excluded = "benchmark_file"
    elif cls == "nonverbal":
        excluded = "nonverbal"
    elif cls == "other":
        excluded = "other"
    else:
        excluded = None
    return {
        "speaker": speaker,
        "gender": (stats.get(speaker) or {}).get("gender"),
        "name": name,
        "path": str(path),
        "class": cls,
        "style": style,
        "emotion": emotion,
        "transcript": transcript,
        "duration": round(float(duration), 3),
        "is_bench_text": is_bench_text,
        "bench_file": bench_file,
        "f1_styled_bench": speaker in bench_speakers and is_bench_text and styled,
        "dev_speaker": dev,
        "dev_leak_probe": dev and is_bench_text and styled,
        "excluded": excluded,
    }


def summarize(rows):
    def add(d, key, row):
        e = d.setdefault(str(key), {"files": 0, "hours": 0.0})
        e["files"] += 1
        e["hours"] += row["duration"] / 3600

    by_class, by_flag, by_excl, trainable = {}, {}, {}, {}
    for r in rows:
        add(by_class, r["class"], r)
        add(by_excl, r["excluded"], r)
        for f in FLAGS:
            if r[f]:
                add(by_flag, f, r)
        if r["excluded"] is None and not r["dev_speaker"]:
            add(trainable, "with_f1", r)
            if not r["f1_styled_bench"]:
                add(trainable, "without_f1", r)
    for d in (by_class, by_flag, by_excl, trainable):
        for e in d.values():
            e["hours"] = round(e["hours"], 3)
    return {
        "n_files": len(rows),
        "n_speakers": len({r["speaker"] for r in rows}),
        "hours": round(sum(r["duration"] for r in rows) / 3600, 3),
        "by_class": dict(sorted(by_class.items())),
        "by_flag": by_flag,
        "by_exclusion": dict(sorted(by_excl.items())),
        "trainable_non_dev": trainable,
        "expected": {"bench_file": 494, "f1_styled_bench": 1404, "dev_leak_probe": 216},
    }


def check_counts(summary, n_speakers_expected=107):
    """Warnings (strings) when the full-dataset counts differ from the contract."""
    out = []
    if summary["n_speakers"] < n_speakers_expected:
        out.append(f"only {summary['n_speakers']}/{n_speakers_expected} speakers in the inventory")
    for flag, want in summary["expected"].items():
        got = summary["by_flag"].get(flag, {}).get("files", 0)
        if got != want:
            out.append(f"{flag}: {got} files, expected {want}")
    return out


# ---- main ---------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speakers", default="", help="comma list (default: all 107)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    import soundfile as sf

    t0 = time.time()
    ears_dir = sftlib.scratch_dir() / "ears"
    ears_dir.mkdir(parents=True, exist_ok=True)
    transcripts = json.loads(fetch(f"{sftlib.EARS_RAW}/transcripts.json",
                                   ears_dir / "transcripts.json").read_text(encoding="utf-8"))
    stats = json.loads(fetch(f"{sftlib.EARS_RAW}/speaker_statistics.json",
                             ears_dir / "speaker_statistics.json").read_text(encoding="utf-8"))
    dev_speakers = set(sftlib.read_json(DEV_SPEAKERS)["speakers"])
    bench = benchmark_sets(sftlib.read_csv(sftlib.BENCH_MANIFEST))

    wanted = [s.strip() for s in args.speakers.split(",") if s.strip()] or sorted(stats)
    todo = [s for s in wanted if not (ears_dir / s / ".complete").exists()]
    print(f"EARS: {len(wanted) - len(todo)}/{len(wanted)} speakers already present, "
          f"{len(todo)} to download (~600 MB each) with {args.workers} workers", flush=True)
    start, failed = time.time(), []
    if todo:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(ensure_speaker, s, ears_dir): s for s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                spk = futures[fut]
                try:
                    fut.result()
                    status = "done"
                except Exception as e:      # keep the other downloads going; re-run resumes
                    failed.append(spk)
                    status = f"FAILED ({type(e).__name__}: {e})"
                el = time.time() - start
                print(f"[{i}/{len(todo)}] {spk} {status}, {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(todo) - i) / 60:.1f} min left", flush=True)

    # inventory over every complete speaker (so --speakers runs accumulate)
    rows = []
    speakers = sorted(d.name for d in ears_dir.iterdir() if (d / ".complete").exists())
    for spk in speakers:
        for wav in sorted((ears_dir / spk).glob("*.wav")):
            rows.append(inventory_row(spk, wav.stem, wav, sf.info(str(wav)).duration,
                                      transcripts, stats, bench, dev_speakers))
    sftlib.write_jsonl(INVENTORY, rows)
    summary = summarize(rows)
    warnings = check_counts(summary, len(stats))
    summary["warnings"] = warnings
    sftlib.write_json(SUMMARY, summary)

    print(f"Wrote {len(rows)} files from {len(speakers)} speakers ({summary['hours']:.1f} h) to {INVENTORY}")
    for k, v in summary["by_exclusion"].items():
        print(f"  excluded={k}: {v['files']} files, {v['hours']:.1f} h")
    for k, v in summary["by_flag"].items():
        print(f"  {k}: {v['files']} files, {v['hours']:.1f} h")
    for k, v in summary["trainable_non_dev"].items():
        print(f"  trainable (non-dev) {k}: {v['files']} files, {v['hours']:.1f} h")
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    sftlib.log_timing("ears_inventory", time.time() - t0, files=len(rows), speakers=len(speakers))
    if failed:
        sys.exit(f"{len(failed)} speakers failed to download: {','.join(sorted(failed))} (re-run to resume)")
    # With every speaker present the exclusion flags must hit their exact counts: a miss means
    # benchmark material (or the F1 / dev-probe sets) would silently leak into training data.
    if summary["n_speakers"] >= len(stats) and any(not w.startswith("only ") for w in warnings):
        sys.exit("inventory flag counts differ from the contract (see WARNING lines); refusing to continue")


if __name__ == "__main__":
    main()
