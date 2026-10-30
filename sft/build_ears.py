"""SFT stage S2c (EARS): sentence units -> MOSS training chunks (audio clips + raw jsonl).

Reads runs/sft/data/units.jsonl (sft/align.py) and ears_inventory.jsonl. For every file with
status "ok" (not excluded), consecutive good units are merged greedily like MOSS's
_split_text_into_best_sentences: a unit is added while the running MOSS token count of the
normalised text stays <= --max-tokens (sft/INTERFACES.md §2.3). A chunk starts at a unit with
left_ok, ends at a unit with right_ok (the run is backed off to the last right_ok unit), never
contains a bad unit or a non-silent gap / extra word between units, and lasts 1.5-20 s.

Audio is cut from the raw 48 kHz EARS wav at [cut_start(first), cut_end(last)], edge silence
trimmed to <= 0.15 s (same file-relative energy threshold as align.py), written as mono 16-bit
wav. ref_audio: the speaker's rainbow_01_regular in 70% of records, otherwise (and always for
chunks of rainbow_01_regular itself) another chunk of the same speaker (read-regular or freeform,
4-10 s, different source file); deterministic per id. Never writes `instruction` or `language`.

Outputs:
  $SCR/ears_chunks/<spk>/<id>.wav       (+ $SCR/ears_chunks/cuts.json: id -> cut span, so a
                                         re-run with another --max-tokens re-cuts changed ids)
  runs/sft/data/ears_raw.jsonl          one line per chunk (rebuilt on every run)
  runs/sft/data/build_summary.json      hours per class / flag, drop reasons

Env: MOSS ($MOSS_PY: transformers 4.57.1, sentencepiece, soundfile, numpy).

  python sft/build_ears.py --max-tokens 50 --moss-repo $MOSS_REPO --model-dir $MOSS_MODEL_DIR
"""
import argparse
import hashlib
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402
from align import (FRAME, GAP_BAD, GAP_SHRINK, SPEECH_FRAC_BAD, frame_db, read_jsonl_tolerant,  # noqa: E402
                   speech_fraction, speech_threshold)

DATA = sftlib.DATA
INVENTORY = DATA / "ears_inventory.jsonl"
UNITS = DATA / "units.jsonl"
RAW = DATA / "ears_raw.jsonl"
SUMMARY = DATA / "build_summary.json"

MIN_DUR, MAX_DUR = 1.5, 20.0
EDGE_SIL = 0.15
REF_SHARE = 0.70
REF_CLASSES = ("read-regular", "freeform")
REF_MIN, REF_MAX = 4.0, 10.0
FLAG_KEYS = ("is_bench_text", "f1_styled_bench", "dev_speaker", "dev_leak_probe")


# ---- chunk composition (pure) -----------------------------------------------------------------

def join_texts(texts):
    return " ".join(t.strip() for t in texts if t.strip())


def compose_chunks(units, max_tokens, count, norm, merge_ok=None, max_dur=MAX_DUR, min_dur=MIN_DUR):
    """Greedy MOSS-style merge of one file's units.

    count(text) -> MOSS tokens; norm(raw text) -> MOSS-normalised text.
    merge_ok[j]: units j and j+1 may share a chunk (no extra word / speech in between).
    Returns (chunks, drops): chunks = [(i, k, normalised text, n_tokens)], drops = [(reason, unit index)].
    """
    n = len(units)
    merge_ok = merge_ok if merge_ok is not None else [True] * max(n - 1, 0)
    tok = [count(norm(u["text"])) for u in units]
    chunks, drops = [], []
    i = 0
    while i < n:
        u = units[i]
        if u["bad"]:
            drops.append(("bad_unit", i)); i += 1; continue
        if not u["left_ok"]:
            drops.append(("no_left_boundary", i)); i += 1; continue
        if tok[i] > max_tokens:
            drops.append(("too_many_tokens", i)); i += 1; continue
        if u["cut_end"] - u["cut_start"] > max_dur:
            drops.append(("too_long", i)); i += 1; continue
        # grow greedily: MOSS adds a sentence while count(current chunk) + count(sentence) <= max
        j, cur = i, tok[i]
        while (j + 1 < n and merge_ok[j] and not units[j + 1]["bad"]
               and cur + tok[j + 1] <= max_tokens
               and units[j + 1]["cut_end"] - u["cut_start"] <= max_dur):
            j += 1
            cur = count(join_texts(norm(units[m]["text"]) for m in range(i, j + 1)))
        # back off to the last unit that can end a chunk, and to a text that fits after normalising
        k = j
        while k >= i:
            if units[k]["right_ok"] and units[k]["cut_end"] - u["cut_start"] <= max_dur:
                text = norm(join_texts(units[m]["text"] for m in range(i, k + 1)))
                n_tok = count(text)
                if n_tok <= max_tokens:
                    break
            k -= 1
        if k < i:
            reason = "too_long" if units[i]["right_ok"] else "no_right_boundary"
            drops.append((reason, i)); i += 1; continue
        if units[k]["cut_end"] - u["cut_start"] < min_dur:
            drops.extend(("too_short", m) for m in range(i, k + 1))
        else:
            chunks.append((i, k, text, n_tok))
        i = k + 1
    return chunks, drops


def merge_flags(units, db, thr, frame=FRAME):
    """merge_ok[j] for adjacent units: same cut point (no ASR word between them) and their gap
    holds no untranscribed speech (same rule as the in-unit `bad` check)."""
    out = []
    for a, b in zip(units, units[1:]):
        ok = abs(a["cut_end"] - b["cut_start"]) < 1e-6
        if ok and b["start"] - a["end"] > GAP_BAD:
            ok = speech_fraction(db, thr, a["end"] + GAP_SHRINK, b["start"] - GAP_SHRINK, frame) <= SPEECH_FRAC_BAD
        out.append(ok)
    return out


def trim_bounds(db, thr, a, b, frame=FRAME, keep=EDGE_SIL):
    """Shrink [a, b] so at most `keep` s of sub-threshold audio remains at each edge."""
    i0, i1 = int(-(-a // frame)), min(len(db), int(b // frame))
    speech = [i for i in range(max(i0, 0), i1) if db[i] > thr]
    if not speech:
        return a, b
    s = speech[0] * frame
    e = (speech[-1] + 1) * frame
    return max(a, s - keep), min(b, e + keep)


def rng_for(chunk_id):
    return random.Random(int(hashlib.sha256(chunk_id.encode()).hexdigest()[:16], 16))


def assign_refs(records, prompt_paths):
    """Set rec["ref_audio"] in place. prompt_paths: speaker -> raw rainbow_01_regular wav.
    Returns the number of records without any valid reference (they are removed by the caller)."""
    by_spk = {}
    for r in records:
        if r["class"] in REF_CLASSES and REF_MIN <= r["duration"] <= REF_MAX:
            by_spk.setdefault(r["speaker"], []).append(r)
    for v in by_spk.values():
        v.sort(key=lambda r: r["id"])
    missing = 0
    for r in records:
        rng = rng_for(r["id"])
        use_prompt = rng.random() < REF_SHARE
        prompt = prompt_paths.get(r["speaker"])
        if use_prompt and prompt and r["src"] != sftlib.PROMPT_READING:
            r["ref_audio"] = prompt
            continue
        cands = [c for c in by_spk.get(r["speaker"], []) if c["src"] != r["src"]]
        if cands:
            r["ref_audio"] = rng.choice(cands)["audio"]
        elif prompt and r["src"] != sftlib.PROMPT_READING:
            r["ref_audio"] = prompt
        else:
            r["ref_audio"] = None
            missing += 1
    return missing


# ---- main -------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--moss-repo", required=True, help="checkout of github.com/OpenMOSS/MOSS-TTS-Nano")
    p.add_argument("--model-dir", required=True, help="MOSS-TTS-Nano-100M (for its tokenizer)")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.moss_repo))
    import numpy as np
    import soundfile as sf
    from text_normalization_pipeline import prepare_tts_request_texts
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    @lru_cache(maxsize=None)
    def norm(text):
        return str(prepare_tts_request_texts(text=text, enable_wetext=False, enable_normalize_tts_text=True,
                                             text_normalizer_manager=None)["text"]).strip()

    @lru_cache(maxsize=None)
    def count(text):
        return len(tok.encode(text, add_special_tokens=False))

    inventory = {(r["speaker"], r["name"]): r for r in sftlib.read_jsonl(INVENTORY)}
    units = {}
    for rec in read_jsonl_tolerant(UNITS):         # last record per file wins
        units[(rec["speaker"], rec["name"])] = rec
    prompt_paths = {r["speaker"]: r["path"] for r in inventory.values() if r["name"] == sftlib.PROMPT_READING}

    chunk_root = sftlib.scratch_dir() / "ears_chunks"
    cuts_path = chunk_root / "cuts.json"
    cuts = sftlib.read_json(cuts_path, default={})

    drops = {}       # reason -> {"n": count, "seconds": s}

    def drop(reason, n=1, seconds=0.0):
        d = drops.setdefault(reason, {"n": 0, "seconds": 0.0})
        d["n"] += n
        d["seconds"] += seconds

    todo = []
    for key, inv in sorted(inventory.items()):
        if inv["excluded"] is not None:
            continue
        rec = units.get(key)
        if rec is None:
            drop("file_not_aligned", seconds=inv["duration"])
        elif rec["status"] != "ok":
            drop(f"file_{rec['status']}", seconds=inv["duration"])
        elif not rec["units"]:
            drop("file_no_units", seconds=inv["duration"])
        else:
            todo.append((inv, rec))
    print(f"{len(todo)} aligned files to chunk (max {args.max_tokens} tokens)", flush=True)

    records, n_cut, start = [], 0, time.time()
    for fi, (inv, rec) in enumerate(todo, 1):
        us = rec["units"]
        wav, sr = sf.read(inv["path"], dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        dur = len(wav) / sr
        db = frame_db(wav, sr)
        thr = speech_threshold(db)
        chunks, unit_drops = compose_chunks(us, args.max_tokens, count, norm, merge_flags(us, db, thr))
        for reason, ui in unit_drops:
            drop(reason, seconds=us[ui]["end"] - us[ui]["start"])
        spk_dir = chunk_root / inv["speaker"]
        for ci, (i, k, text, n_tok) in enumerate(chunks):
            a, b = trim_bounds(db, thr, max(0.0, us[i]["cut_start"]), min(dur, us[k]["cut_end"]))
            s0, s1 = int(round(a * sr)), int(round(b * sr))
            if (s1 - s0) / sr < MIN_DUR:
                drop("too_short_after_trim", k - i + 1, (s1 - s0) / sr)
                continue
            cid = f"{inv['speaker']}__{inv['name']}__{ci:03d}"
            out = spk_dir / f"{cid}.wav"
            span = [s0, s1, sr]
            if not (out.exists() and cuts.get(cid) == span):
                spk_dir.mkdir(parents=True, exist_ok=True)
                part = out.with_name(out.name + ".part")
                sf.write(str(part), np.clip(wav[s0:s1], -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
                os.replace(part, out)
                cuts[cid] = span
                n_cut += 1
            rec_out = {
                "id": cid, "audio": str(out), "text": text, "ref_audio": None,
                "speaker": inv["speaker"], "gender": inv["gender"], "src": inv["name"],
                "class": inv["class"], "style": inv["style"], "emotion": inv["emotion"],
                "duration": round((s1 - s0) / sr, 3), "n_text_tokens": n_tok,
            }
            rec_out.update({f: inv[f] for f in FLAG_KEYS})
            records.append(rec_out)
        if fi % 200 == 0 or fi == len(todo):
            sftlib.write_json(cuts_path, cuts)
            el = time.time() - start
            print(f"[{fi}/{len(todo)}] {len(records)} chunks, {n_cut} wavs cut, {el / 60:.1f} min elapsed, "
                  f"~{el / fi * (len(todo) - fi) / 60:.1f} min left", flush=True)

    missing = assign_refs(records, prompt_paths)
    if missing:
        for r in records:
            if r["ref_audio"] is None:
                drop("no_ref_audio", seconds=r["duration"])
        records = [r for r in records if r["ref_audio"] is not None]
    for r in records:
        assert "instruction" not in r and "language" not in r
    sftlib.write_jsonl(RAW, records)
    sftlib.write_json(cuts_path, cuts)

    def agg(key_fn):
        out = {}
        for r in records:
            for k in key_fn(r):
                e = out.setdefault(str(k), {"chunks": 0, "hours": 0.0})
                e["chunks"] += 1
                e["hours"] += r["duration"] / 3600
        for e in out.values():
            e["hours"] = round(e["hours"], 3)
        return dict(sorted(out.items()))

    summary = {
        "max_tokens": args.max_tokens,
        "n_chunks": len(records),
        "hours": round(sum(r["duration"] for r in records) / 3600, 3),
        "by_class": agg(lambda r: [r["class"]]),
        "by_flag": agg(lambda r: [f for f in FLAG_KEYS if r[f]]),
        "trainable_non_dev": agg(lambda r: [] if r["dev_speaker"] else
                                 (["with_f1"] + ([] if r["f1_styled_bench"] else ["without_f1"]))),
        "ref_audio_prompt_share": round(sum(r["ref_audio"].endswith(f"{sftlib.PROMPT_READING}.wav")
                                            for r in records) / max(len(records), 1), 3),
        "mean_duration": round(sum(r["duration"] for r in records) / max(len(records), 1), 2),
        "mean_text_tokens": round(sum(r["n_text_tokens"] for r in records) / max(len(records), 1), 1),
        "drops": {k: {"n": v["n"], "hours": round(v["seconds"] / 3600, 3)} for k, v in sorted(drops.items())},
    }
    sftlib.write_json(SUMMARY, summary)
    print(f"Wrote {len(records)} chunks ({summary['hours']:.1f} h) to {RAW}")
    for k, v in summary["by_class"].items():
        print(f"  {k}: {v['chunks']} chunks, {v['hours']:.1f} h")
    for k, v in summary["drops"].items():
        print(f"  dropped {k}: {v['n']} ({v['hours']:.2f} h)")
    sftlib.log_timing("build_ears", time.time() - t0, chunks=len(records), hours=summary["hours"],
                      max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
