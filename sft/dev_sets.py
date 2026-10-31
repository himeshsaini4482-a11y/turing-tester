"""SFT dev sets (EARS): dev-mirror / dev-fresh manifests and the multi-seed eval manifests.

  mirror         the 4 held-out dev speakers (sft/data/dev_speakers.json) x the 19 benchmark
                 reading names (*_regular), prompt = their rainbow_01_regular -> 76 rows, set=mirror
  fresh          10 freeform chunks per dev speaker from runs/sft/data/ears_raw.jsonl
                 (class freeform, 4-12 s, random.Random(4321)) -> 40 rows, set=fresh
  eval-manifest  mirror (+ fresh) rows repeated --seeds times with sentence_id = <item_id>__s<k>
                 (generate_moss_nano.py seeds from crc32(sentence_id))

Columns are the benchmark manifest's (speaker_id, gender, chapter_id, sentence_id, text, duration,
human_wav, prompt_id, prompt_wav, prompt_text) plus set and item_id (sft/INTERFACES.md §2.4).
Every human / prompt wav is copied into runs/sft/dev/wav/ (durable) and the CSVs point at those
copies with repo-relative paths, like the benchmark manifest.

Outputs: runs/sft/dev/{dev_mirror.csv, dev_fresh.csv, wav/}, and the --out of eval-manifest.

Env: TTS (standard library only).

  python sft/dev_sets.py mirror
  python sft/dev_sets.py fresh
  python sft/dev_sets.py eval-manifest --sets mirror,fresh --seeds 3 --out runs/sft/dev/eval_full.csv
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

DEV = sftlib.DEV
WAV_DIR = DEV / "wav"
INVENTORY = sftlib.DATA / "ears_inventory.jsonl"
EARS_RAW = sftlib.DATA / "ears_raw.jsonl"
DEV_SPEAKERS = Path(__file__).resolve().parent / "data" / "dev_speakers.json"
SET_FILES = {"mirror": DEV / "dev_mirror.csv", "fresh": DEV / "dev_fresh.csv"}
FRESH_PER_SPEAKER = 10
FRESH_MIN_PER_SPEAKER = 5    # fewer eligible chunks than FRESH_PER_SPEAKER: use them all, if at least this many
FRESH_MIN, FRESH_MAX = 4.0, 12.0
COLUMNS = ["speaker_id", "gender", "chapter_id", "sentence_id", "text", "duration", "human_wav",
           "prompt_id", "prompt_wav", "prompt_text", "set", "item_id"]


def gender_code(g):
    return {"male": "M", "female": "F"}.get(str(g).lower(), str(g))


def rel(path):
    """Repo-relative posix path (runs/ is copied back to the UAI server; absolute paths break)."""
    try:
        return Path(path).resolve().relative_to(sftlib.REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def copy_in(src, name):
    """Copy src to runs/sft/dev/wav/<name>.wav once; return the repo-relative copy path."""
    dst = WAV_DIR / f"{name}.wav"
    if not (dst.exists() and dst.stat().st_size == Path(src).stat().st_size):
        WAV_DIR.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, part)
        os.replace(part, dst)
    return rel(dst)


def bench_reading_names(manifest_rows):
    """The 19 benchmark reading names (sentence_id minus '<spk>_'), checked equal for all speakers."""
    per_spk = {}
    for r in manifest_rows:
        prefix = r["speaker_id"] + "_"
        assert r["sentence_id"].startswith(prefix), r["sentence_id"]
        per_spk.setdefault(r["speaker_id"], []).append(r["sentence_id"][len(prefix):])
    sets = {tuple(sorted(v)) for v in per_spk.values()}
    if len(sets) != 1:
        raise SystemExit(f"benchmark speakers read different names: {len(sets)} variants")
    return next(iter(per_spk.values()))      # manifest order


def dev_speakers():
    return sftlib.read_json(DEV_SPEAKERS)["speakers"]   # {spk: "female"|"male"}


def mirror_rows(speakers, names, inventory, copy=copy_in):
    """speakers {spk: gender}; inventory {(spk, name): row}. -> manifest rows."""
    rows = []
    for spk in sorted(speakers):
        prompt = inventory.get((spk, sftlib.PROMPT_READING))
        if prompt is None:
            raise SystemExit(f"{spk}/{sftlib.PROMPT_READING} missing from the inventory")
        prompt_wav = copy(prompt["path"], f"{spk}_{sftlib.PROMPT_READING}")
        for name in names:
            inv = inventory.get((spk, name))
            if inv is None:
                raise SystemExit(f"{spk}/{name} missing from the inventory")
            sid = f"{spk}_{name}"
            rows.append({
                "speaker_id": spk, "gender": gender_code(speakers[spk]), "chapter_id": "",
                "sentence_id": sid, "text": inv["transcript"], "duration": f"{inv['duration']:.2f}",
                "human_wav": copy(inv["path"], sid),
                "prompt_id": f"{spk}_{sftlib.PROMPT_READING}", "prompt_wav": prompt_wav,
                "prompt_text": prompt["transcript"], "set": "mirror", "item_id": sid,
            })
    return rows


def fresh_rows(speakers, chunks, inventory, copy=copy_in, seed=sftlib.DEV_SEED, k=FRESH_PER_SPEAKER):
    """chunks: ears_raw.jsonl records. One RNG, speakers in sorted order, candidates sorted by id."""
    rng = random.Random(seed)
    rows = []
    for spk in sorted(speakers):
        cands = sorted((c for c in chunks if c["speaker"] == spk and c["class"] == "freeform"
                        and FRESH_MIN <= c["duration"] <= FRESH_MAX), key=lambda c: c["id"])
        if len(cands) < min(k, FRESH_MIN_PER_SPEAKER):
            raise SystemExit(f"{spk}: only {len(cands)} freeform chunks of {FRESH_MIN:g}-{FRESH_MAX:g} s")
        if len(cands) < k:
            print(f"WARNING: {spk}: only {len(cands)} freeform chunks of {FRESH_MIN:g}-{FRESH_MAX:g} s "
                  f"(want {k}); using all of them", flush=True)
        prompt = inventory.get((spk, sftlib.PROMPT_READING))
        if prompt is None:
            raise SystemExit(f"{spk}/{sftlib.PROMPT_READING} missing from the inventory")
        prompt_wav = copy(prompt["path"], f"{spk}_{sftlib.PROMPT_READING}")
        for c in sorted(rng.sample(cands, min(k, len(cands))), key=lambda c: c["id"]):
            rows.append({
                "speaker_id": spk, "gender": gender_code(speakers[spk]), "chapter_id": "",
                "sentence_id": c["id"], "text": c["text"], "duration": f"{c['duration']:.2f}",
                "human_wav": copy(c["audio"], c["id"]),
                "prompt_id": f"{spk}_{sftlib.PROMPT_READING}", "prompt_wav": prompt_wav,
                "prompt_text": prompt["transcript"], "set": "fresh", "item_id": c["id"],
            })
    return rows


def eval_rows(rows, seeds):
    out = []
    for r in rows:
        item = r.get("item_id") or r["sentence_id"]
        for k in range(seeds):
            out.append({**r, "item_id": item, "sentence_id": f"{item}__s{k}"})
    return out


def write(path, rows):
    sftlib.write_csv(path, [{c: r.get(c, "") for c in COLUMNS} for r in rows])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mirror")
    sub.add_parser("fresh")
    e = sub.add_parser("eval-manifest")
    e.add_argument("--sets", default="mirror")
    e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--out", required=True)
    args = p.parse_args()

    if args.cmd == "eval-manifest":
        rows = []
        for s in [s.strip() for s in args.sets.split(",") if s.strip()]:
            if s not in SET_FILES:
                raise SystemExit(f"unknown set {s}; choose from {sorted(SET_FILES)}")
            if not SET_FILES[s].exists():
                raise SystemExit(f"{SET_FILES[s]} missing; run `dev_sets.py {s}` first")
            rows += sftlib.read_csv(SET_FILES[s])
        out = eval_rows(rows, args.seeds)
        write(args.out, out)
        print(f"Wrote {len(out)} rows ({len(rows)} items x {args.seeds} seeds) to {args.out}")
        return

    inventory = {(r["speaker"], r["name"]): r for r in sftlib.read_jsonl(INVENTORY)}
    if not inventory:
        raise SystemExit(f"{INVENTORY} is empty; run sft/ears_inventory.py first")
    speakers = dev_speakers()
    if args.cmd == "mirror":
        names = bench_reading_names(sftlib.read_csv(sftlib.BENCH_MANIFEST))
        rows = mirror_rows(speakers, names, inventory)
    else:
        chunks = sftlib.read_jsonl(EARS_RAW)
        if not chunks:
            raise SystemExit(f"{EARS_RAW} is empty; run sft/build_ears.py first")
        rows = fresh_rows(speakers, chunks, inventory)
    write(SET_FILES[args.cmd], rows)
    print(f"Wrote {len(rows)} rows to {SET_FILES[args.cmd]} (wavs copied to {WAV_DIR})")


if __name__ == "__main__":
    main()
