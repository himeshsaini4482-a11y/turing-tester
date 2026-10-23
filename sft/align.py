"""SFT stage S2b (EARS): Whisper word alignment -> sentence units with safe cut points.

For every non-excluded file of runs/sft/data/ears_inventory.jsonl (dev speakers and F1 files
included; they are filtered downstream), run openai/whisper-large-v3 (transformers ASR pipeline,
fp16, word timestamps, 30 s chunks, batched, English) with a disfluent prompt so fillers are
transcribed (falls back to no prompt if the prompted run errors or returns broken timestamps).
Then build one unit per sentence (sft/INTERFACES.md §2.2):

  - read classes: the script is aligned word by word to the ASR words (difflib), align_wer =
    WER(script, ASR) (status "misread" if > 0.10). Unit text is the script's own sentence; times
    come from its first/last matched words ("align_fail" if a sentence has no matched word).
  - freeform / freeform-emotion / interjection: Whisper's words, split after . ! ?
  - energy (20 ms RMS, threshold = file p95 - 35 dB):
      bad        a word gap > 0.35 s inside the unit with > 30% speech frames (dropped filler),
                 or (read) ASR words inside the sentence that the script does not have
      cut_*      lowest-energy point between the unit's edge word and the neighbouring ASR word
                 (or within 0.15 s beyond the edge word at the file edge)
      left/right_ok  that gap >= 0.2 s and the cut point is low-energy (file edge: ok)

Output: runs/sft/data/units.jsonl, one line per wav, appended (resumable: done files skipped).

Env: TTS (.venv-tts: torch, transformers, librosa, jiwer).

  python sft/align.py [--batch-size 16] [--limit N]
"""
import argparse
import difflib
import gc
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

DATA = sftlib.RUNS / "data"
INVENTORY = DATA / "ears_inventory.jsonl"
UNITS = DATA / "units.jsonl"

SR = 16000
FRAME = 0.02                 # energy frame (s)
REL_DB = 35.0                # speech threshold = p95(frame dB) - REL_DB
GAP_BAD = 0.35               # an inner word gap longer than this is checked for speech
GAP_SHRINK = 0.10            # ... after shrinking it this much on each side
SPEECH_FRAC_BAD = 0.30       # ... and it is "bad" if more than this share of frames is speech
GAP_OK = 0.20                # min gap at a chunk boundary
EDGE_PAD = 0.15              # search window beyond the edge word at the file edge
MISREAD_WER = 0.10
READ_CLASSES = ("read-regular", "read-styled", "read-emotion")
PROMPT_TEXT = "Um, so, I was like, uh, thinking about it, you know, and, hmm, yeah."

_CLOSERS = "\"'”’)]}»"
_SENT_END = re.compile(r"[.!?][" + re.escape(_CLOSERS) + r"]*(?=\s|$)")


# ---- text -----------------------------------------------------------------------------------

def split_sentences(text):
    """Script -> sentences (original text), split after . ! ? (+ closing quotes) and whitespace."""
    out, pos = [], 0
    for m in _SENT_END.finditer(text):
        s = text[pos:m.end()].strip()
        if s:
            out.append(s)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append(tail)
    return out


def ends_sentence(word_text):
    return word_text.strip().rstrip(_CLOSERS)[-1:] in (".", "!", "?")


def wer(ref_words, hyp_words):
    """WER via jiwer when available (worker), else the same Levenshtein word distance."""
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    try:
        import jiwer
        return float(jiwer.wer(" ".join(ref_words), " ".join(hyp_words)))
    except ImportError:
        prev = list(range(len(hyp_words) + 1))
        for i, r in enumerate(ref_words, 1):
            cur = [i] + [0] * len(hyp_words)
            for j, h in enumerate(hyp_words, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
            prev = cur
        return prev[-1] / len(ref_words)


# ---- ASR output -> word list ------------------------------------------------------------------

def asr_tokens(chunks, duration):
    """Whisper word chunks -> (chunks, tokens). Fixes None / out-of-range / non-monotonic times.

    chunks: [{"text", "start", "end", "brk"}] (brk: the chunk ends a sentence)
    tokens: [{"w": normalised word, "start", "end", "c": chunk index}] (a chunk can give 0..n tokens)
    """
    clean = []
    for ch in chunks:
        ts = ch.get("timestamp") or (None, None)
        clean.append({"text": ch.get("text") or "", "start": ts[0], "end": ts[1]})
    # fill missing times from neighbours; the final None end is the audio end
    for i, ch in enumerate(clean):
        if ch["start"] is None:
            ch["start"] = clean[i - 1]["end"] if i > 0 and clean[i - 1]["end"] is not None else 0.0
    for i, ch in enumerate(clean):
        if ch["end"] is None:
            ch["end"] = clean[i + 1]["start"] if i + 1 < len(clean) else duration
    last_s = last_e = 0.0                      # starts and ends made non-decreasing
    for ch in clean:
        ch["start"] = min(max(float(ch["start"]), last_s, 0.0), duration)
        ch["end"] = min(max(float(ch["end"]), ch["start"], last_e), duration)
        last_s, last_e = ch["start"], ch["end"]
    # merge punctuation-only chunks into the previous chunk (their "." still ends the sentence)
    out = []
    for ch in clean:
        if not sftlib.normalize_words(ch["text"]) and out:
            out[-1]["text"] += ch["text"].strip()
            out[-1]["end"] = max(out[-1]["end"], ch["end"])
            continue
        out.append(ch)
    tokens = []
    for ci, ch in enumerate(out):
        ch["brk"] = ends_sentence(ch["text"])
        for w in sftlib.normalize_words(ch["text"]):
            tokens.append({"w": w, "start": ch["start"], "end": ch["end"], "c": ci})
    return out, tokens


def strip_prompt_chunks(chunks, prompt=PROMPT_TEXT):
    """Remove every chunk run whose words spell out the whole prompt (prompt leaking into output)."""
    pw = sftlib.normalize_words(prompt)
    flat = [(ci, w) for ci, ch in enumerate(chunks) for w in sftlib.normalize_words(ch.get("text") or "")]
    drop, i = set(), 0
    while i + len(pw) <= len(flat):
        if [w for _, w in flat[i:i + len(pw)]] == pw:
            drop.update(ci for ci, _ in flat[i:i + len(pw)])
            i += len(pw)
        else:
            i += 1
    return [ch for ci, ch in enumerate(chunks) if ci not in drop], bool(drop)


def timestamps_sane(chunks, duration):
    """Raw pipeline chunks: starts non-decreasing (0.5 s slack), inside the audio."""
    last = -1.0
    for ch in chunks:
        s, e = (ch.get("timestamp") or (None, None))
        if s is None:
            return False
        if s < last - 0.5 or s > duration + 0.5 or (e is not None and e < s - 0.01):
            return False
        last = s
    return True


def timing_shift(chunks_a, chunks_b, duration):
    """Start-time differences (s) of the words two transcripts of one file share (difflib match).
    Used to check that the prompt does not shift Whisper's word timestamps."""
    _, ta = asr_tokens(chunks_a, duration)
    _, tb = asr_tokens(chunks_b, duration)
    sm = difflib.SequenceMatcher(None, [t["w"] for t in ta], [t["w"] for t in tb], autojunk=False)
    return [abs(ta[b.a + k]["start"] - tb[b.b + k]["start"])
            for b in sm.get_matching_blocks() for k in range(b.size)]


def median(xs):
    s = sorted(xs)
    return (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2 if s else 0.0


def read_jsonl_tolerant(path):
    """sftlib.read_jsonl, but a truncated last line (worker killed mid-append) is skipped."""
    import json
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    print(f"WARNING: skipping a corrupt line in {path}", flush=True)
    return rows


def ensure_trailing_newline(path):
    """So the next append never glues onto a half-written line."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


# ---- sentence spans ---------------------------------------------------------------------------

def read_spans(transcript, tokens):
    """Align script to ASR tokens. -> (spans or None, align_wer, n_untimed).

    span: {"text", "i0", "i1" (token index range, inclusive), "edge_l", "edge_r" (the sentence's
           first / last script word itself is matched), "inserted" (ASR words inside the span
           that the script does not have)}
    """
    sentences = split_sentences(transcript)
    sent_words = [sftlib.normalize_words(s) for s in sentences]
    script = [w for ws in sent_words for w in ws]
    hyp = [t["w"] for t in tokens]
    a_wer = wer(script, hyp)
    sm = difflib.SequenceMatcher(None, script, hyp, autojunk=False)
    match = {}                              # script index -> token index
    extra = set()                           # token indices the script does not have
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            # equal-length substitutions ("colour"/"color") still time the word one-to-one
            for k in range(a1 - a0):
                match[a0 + k] = b0 + k
        elif tag in ("insert", "replace") and b1 - b0 > a1 - a0:
            extra.update(range(b0, b1))
    spans, n_untimed, base = [], 0, 0
    for text, ws in zip(sentences, sent_words):
        idx = [match[i] for i in range(base, base + len(ws)) if i in match]
        if not idx:
            n_untimed += 1
            spans.append(None)
        else:
            i0, i1 = min(idx), max(idx)
            spans.append({
                "text": text, "i0": i0, "i1": i1,
                "edge_l": base in match, "edge_r": (base + len(ws) - 1) in match,
                "inserted": any(j in extra for j in range(i0, i1 + 1)),
            })
        base += len(ws)
    return spans, a_wer, n_untimed


def freeform_spans(chunks, tokens):
    """Sentences from Whisper's own words: break after a chunk ending in . ! ? (quotes ignored)."""
    if not tokens:
        return []
    first_tok, last_tok = {}, {}
    for ti, t in enumerate(tokens):
        first_tok.setdefault(t["c"], ti)
        last_tok[t["c"]] = ti
    spans, cur = [], []
    for ci, ch in enumerate(chunks):
        if ci not in first_tok:
            continue
        cur.append(ci)
        if ch["brk"]:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    out = []
    for cs in spans:
        text = re.sub(r"\s+", " ", "".join(chunks[c]["text"] for c in cs)).strip()
        out.append({"text": text, "i0": first_tok[cs[0]], "i1": last_tok[cs[-1]],
                    "edge_l": True, "edge_r": True, "inserted": False})
    return out


# ---- energy -----------------------------------------------------------------------------------

def frame_db(audio, sr=SR, frame=FRAME):
    """RMS dB of non-overlapping frames (numpy array in, list out)."""
    import numpy as np
    n = int(round(sr * frame))
    k = len(audio) // n
    if k == 0:
        return []
    x = np.asarray(audio[: k * n], dtype=np.float64).reshape(k, n)
    rms = np.sqrt((x * x).mean(axis=1))
    return (20 * np.log10(rms + 1e-10)).tolist()


def speech_threshold(db, rel=REL_DB):
    if not db:
        return 0.0
    s = sorted(db)
    return s[min(len(s) - 1, int(0.95 * (len(s) - 1)))] - rel


def _frames_in(a, b, n_frames, frame=FRAME):
    """Indices of frames lying fully inside [a, b]."""
    i0 = max(0, int(-(-a // frame)))          # ceil
    i1 = min(n_frames, int(b // frame))       # exclusive
    return range(i0, i1)


def speech_fraction(db, thr, a, b, frame=FRAME):
    idx = _frames_in(a, b, len(db), frame)
    if len(idx) == 0:
        return 0.0
    return sum(db[i] > thr for i in idx) / len(idx)


def lowest_point(db, a, b, frame=FRAME):
    """(time, dB) of the quietest frame inside [a, b]; midpoint if no frame fits."""
    idx = _frames_in(a, b, len(db), frame)
    if len(idx) == 0:
        mid = (a + b) / 2
        i = min(max(int(mid // frame), 0), len(db) - 1) if db else 0
        return mid, (db[i] if db else 0.0)
    i = min(idx, key=lambda k: db[k])
    return (i + 0.5) * frame, db[i]


def build_units(spans, tokens, db, thr, duration, frame=FRAME):
    """spans (all timed, in order) + tokens + frame dB -> §2.2 units."""
    units = []
    for sp in spans:
        words = tokens[sp["i0"]: sp["i1"] + 1]
        start, end = words[0]["start"], words[-1]["end"]
        bad = sp["inserted"]
        for w1, w2 in zip(words, words[1:]):
            if w2["start"] - w1["end"] > GAP_BAD and speech_fraction(
                    db, thr, w1["end"] + GAP_SHRINK, w2["start"] - GAP_SHRINK, frame) > SPEECH_FRAC_BAD:
                bad = True
        # left: gap back to the previous ASR word (any word: unit or untimed insertion)
        if sp["i0"] > 0:
            prev_end = tokens[sp["i0"] - 1]["end"]
            cut_s, db_s = lowest_point(db, prev_end, start, frame)
            left_ok = start - prev_end >= GAP_OK and db_s <= thr
        else:
            cut_s, _ = lowest_point(db, max(0.0, start - EDGE_PAD), start, frame)
            left_ok = True
        if sp["i1"] + 1 < len(tokens):
            next_start = tokens[sp["i1"] + 1]["start"]
            cut_e, db_e = lowest_point(db, end, next_start, frame)
            right_ok = next_start - end >= GAP_OK and db_e <= thr
        else:
            cut_e, _ = lowest_point(db, end, min(duration, end + EDGE_PAD), frame)
            right_ok = True
        units.append({
            "text": sp["text"], "start": round(start, 3), "end": round(end, 3),
            # the same gap gives the same cut point on both sides; build_ears relies on that
            # (equal cut_end / next cut_start = no ASR word between the two units)
            "cut_start": round(cut_s, 3), "cut_end": round(cut_e, 3),
            "left_ok": bool(left_ok and sp["edge_l"]), "right_ok": bool(right_ok and sp["edge_r"]),
            "bad": bool(bad),
        })
    return units


def process_file(row, chunks, audio_db, duration):
    """Inventory row + raw pipeline chunks (prompt already stripped) -> §2.2 record (no I/O)."""
    rec = {"speaker": row["speaker"], "name": row["name"], "path": row["path"],
           "status": "ok", "align_wer": None, "units": []}
    chunks, tokens = asr_tokens(chunks, duration)
    if not tokens:
        rec["status"] = "asr_fail"
        return rec
    thr = speech_threshold(audio_db)
    if row["class"] in READ_CLASSES and row.get("transcript"):
        spans, a_wer, n_untimed = read_spans(row["transcript"], tokens)
        rec["align_wer"] = round(a_wer, 4)
        if a_wer > MISREAD_WER:
            rec["status"] = "misread"
        elif n_untimed:
            rec["status"] = "align_fail"
            return rec
        spans = [s for s in spans if s is not None]
    else:
        spans = freeform_spans(chunks, tokens)
    rec["units"] = build_units(spans, tokens, audio_db, thr, duration)
    return rec


# ---- main -------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="process at most N new files")
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--attn", default="eager",
                   help="attention implementation; eager is the safe choice for word timestamps "
                        "(cross-attention weights)")
    p.add_argument("--files-per-call", type=int, default=0,
                   help="files per pipeline call (default: batch size)")
    p.add_argument("--no-prompt", action="store_true", help="skip the disfluent prompt")
    args = p.parse_args()

    import librosa
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoProcessor, pipeline

    t0 = time.time()
    inventory = [r for r in sftlib.read_jsonl(INVENTORY) if r["excluded"] is None]
    ensure_trailing_newline(UNITS)
    # asr_fail files are retried on every run (a config / backend error must not stick);
    # build_ears uses the last record per file
    done = {(r["speaker"], r["name"]) for r in read_jsonl_tolerant(UNITS) if r.get("status") != "asr_fail"}
    todo = [r for r in inventory if (r["speaker"], r["name"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    total_s = sum(r["duration"] for r in todo)
    print(f"{len(done)} files already aligned, {len(todo)} to do ({total_s / 3600:.1f} h of audio)",
          flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    asr = pipeline("automatic-speech-recognition", model=args.model, torch_dtype=dtype, device=device,
                   model_kwargs={"attn_implementation": args.attn})
    processor = AutoProcessor.from_pretrained(args.model)
    use_prompt = not args.no_prompt
    prompt_ids = None
    if use_prompt:
        try:
            prompt_ids = processor.get_prompt_ids(PROMPT_TEXT, return_tensors="pt").to(device)
        except Exception as e:
            print(f"prompt_ids unavailable ({e}); running without the disfluent prompt", flush=True)
            use_prompt = False

    bs = [args.batch_size]

    def run(audios, with_prompt):
        """CUDA OOM -> free the memory and halve the batch size (kept for later calls). Word
        timestamps keep every encoder layer's eager attention, ~2.7 GB per 30 s chunk."""
        kw = {"language": "english", "task": "transcribe"}
        if with_prompt:
            kw["prompt_ids"] = prompt_ids
        while True:
            inputs = [{"raw": a, "sampling_rate": SR} for a in audios]   # the pipeline pops "raw"
            try:
                return asr(inputs, batch_size=bs[0], return_timestamps="word", chunk_length_s=30,
                           generate_kwargs=kw)
            except torch.cuda.OutOfMemoryError as e:
                msg = str(e).split(".")[0]
            # outside the except block: the traceback's reference cycles hold the CUDA tensors
            gc.collect()
            torch.cuda.empty_cache()
            if bs[0] == 1:
                raise RuntimeError(f"CUDA OOM at batch size 1 ({msg})")
            bs[0] //= 2
            print(f"WARNING: CUDA OOM; Whisper batch size -> {bs[0]}", flush=True)

    def transcribe(audios):
        """-> list of (chunks or None, used_prompt). Prompt errors -> no prompt from then on."""
        nonlocal use_prompt
        outs, used = None, False
        if use_prompt:
            try:
                outs, used = run(audios, True), True
            except Exception as e:
                print(f"WARNING: prompted Whisper failed ({type(e).__name__}: {e}); "
                      f"continuing without the disfluent prompt", flush=True)
                use_prompt = False
        if outs is None:
            try:
                outs = run(audios, False)
            except Exception as e:
                print(f"WARNING: batch failed ({type(e).__name__}: {e}); retrying file by file", flush=True)
                outs = []
                for a in audios:
                    try:
                        outs.append(run([a], False)[0])
                    except Exception as e2:
                        print(f"  file failed: {type(e2).__name__}: {e2}", flush=True)
                        outs.append(None)
        return [(o, used) for o in outs]

    per_call = args.files_per_call or args.batch_size
    pool = ThreadPoolExecutor(4)
    start, done_s, n_prompt_fallback = time.time(), 0.0, 0
    counts, prompt_checked, group_status = {}, False, []
    for g in range(0, len(todo), per_call):
        group = todo[g: g + per_call]
        audios = list(pool.map(lambda r: librosa.load(r["path"], sr=SR)[0], group))
        results = transcribe(audios)
        if use_prompt and not prompt_checked and any(used for _, used in results):
            # once: the prompt must not shift word timestamps (cut points depend on them)
            prompt_checked = True
            plain = transcribe_batch_plain(run, audios)
            diffs = []
            for a, (out, _), pl in zip(audios, results, plain):
                if out is not None and pl is not None:
                    diffs += timing_shift(strip_prompt_chunks(list(out.get("chunks") or []))[0],
                                          list(pl.get("chunks") or []), len(a) / SR)
            shift = median(diffs)
            print(f"Prompt timing check: {len(diffs)} shared words, median start shift {shift:.3f} s", flush=True)
            if not diffs or shift > 0.10:
                print("WARNING: the disfluent prompt shifts word timestamps; continuing without it", flush=True)
                use_prompt = False
                results = [(pl, False) for pl in plain]
        for r, a, (out, used) in zip(group, audios, results):
            dur = len(a) / SR
            chunks = None if out is None else list(out.get("chunks") or [])
            if used and chunks is not None:
                chunks, _ = strip_prompt_chunks(chunks)
                if not timestamps_sane(chunks, dur) or (dur > 1.0 and not chunks):
                    n_prompt_fallback += 1
                    retry = transcribe_plain(run, a)
                    chunks = None if retry is None else strip_prompt_chunks(list(retry.get("chunks") or []))[0]
            if chunks is None:
                rec = {"speaker": r["speaker"], "name": r["name"], "path": r["path"],
                       "status": "asr_fail", "align_wer": None, "units": []}
            else:
                rec = process_file(r, chunks, frame_db(a), dur)
            sftlib.append_jsonl(UNITS, rec)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            group_status.append(rec["status"])
            done_s += r["duration"]
        if g == 0 and all(s == "asr_fail" for s in group_status):
            sys.exit("Every file of the first group failed ASR; fix the Whisper setup "
                     "(e.g. --attn eager / --no-prompt) and re-run (asr_fail files are retried).")
        group_status.clear()
        el = time.time() - start
        n = min(g + per_call, len(todo))
        print(f"[{n}/{len(todo)}] {done_s / 3600:.2f}/{total_s / 3600:.2f} h audio, "
              f"{el / 60:.1f} min elapsed, ~{el / max(done_s, 1e-6) * (total_s - done_s) / 60:.1f} min left, "
              f"status {counts}, prompt {'on' if use_prompt else 'off'} "
              f"(per-file fallbacks {n_prompt_fallback})", flush=True)
    pool.shutdown()
    sftlib.log_timing("align", time.time() - t0, files=len(todo), audio_hours=round(total_s / 3600, 2),
                      prompt=use_prompt, prompt_fallbacks=n_prompt_fallback, **counts)


def transcribe_batch_plain(run, audios):
    try:
        return run(audios, False)
    except Exception as e:
        print(f"  no-prompt batch failed: {type(e).__name__}: {e}", flush=True)
        return [transcribe_plain(run, a) for a in audios]


def transcribe_plain(run, audio):
    try:
        return run([audio], False)[0]
    except Exception as e:
        print(f"  no-prompt retry failed: {type(e).__name__}: {e}", flush=True)
        return None


if __name__ == "__main__":
    main()
