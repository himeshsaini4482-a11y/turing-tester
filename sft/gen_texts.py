"""Round stage R-3: the Qwen 3.5 9B writes new sentences that target the confirmed weaknesses.

Reads a weakness report (sft/tag_weaknesses.py, INTERFACES §2.5) and appends texts to
runs/sft/<stage>/texts.jsonl (§2.7): {"id": "<stage>_t00042", "text", "category", "stage"}.

How many: --n N, or budget mode --budget-gpu-hours H --throughput <previous filter_summary.json>:
  N = ceil(1.2 x H x 3600 x n_generated / gen_gpu_s)   (render capacity of the time box + 20%)
falling back to --n when the throughput file is missing or empty.
Mix: 75% targeted, split across the confirmed categories by share (largest remainder), 25%
"general". Nothing confirmed -> 100% general.

Each 9B call asks for 25 sentences (JSON array of {"text", "category"}) for one category, with
its taxonomy description, sftlib.WEAKNESS_TEXT_HINT and up to 3 judge evidence quotes. A text is
kept only if it is 8-40 words, 1-3 sentences, plain prose without digits, quotes or markdown,
and not a duplicate: normalised exact match or word-5-gram Jaccard >= 0.6 against every earlier
runs/sft/*/texts.jsonl and the 19 benchmark texts (any benchmark sentence contained verbatim is
also rejected, so benchmark text never leaks into training).

Env: STD (any venv). Resumable: appends until N texts exist in --out.

  python sft/gen_texts.py --report runs/sft/pilot/diag/weakness_report.json \
      --out runs/sft/pilot/texts.jsonl --stage pilot --n 3000
"""
import argparse
import math
import os
import random
import re
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402
import tag_weaknesses  # noqa: E402  (extract_json, llm)

PER_CALL = 25
TARGETED_SHARE = 0.75
RENDER_MARGIN = 1.2   # budget mode: texts = 1.2 x what the teacher can render in the time box
MIN_WORDS, MAX_WORDS = 8, 40
JACCARD_MAX = 0.6
NGRAM = 5
MAX_BARREN_CALLS = 8      # consecutive calls with no accepted text before a category is given up
MAX_TRANSPORT_ERRORS = 5  # consecutive endpoint failures before the stage aborts (re-run resumes)

GENERAL = "general"
GENERAL_DESC = "general retention set: varied everyday English, read naturally"

TOPICS = [
    "cooking at home", "a train journey", "gardening", "a job interview", "childhood memories",
    "the weather", "a football match", "visiting a museum", "a broken washing machine", "moving house",
    "a birthday party", "learning a language", "a hospital visit", "space exploration", "a lost wallet",
    "a family dinner", "online shopping", "a mountain hike", "a small bakery", "a school exam",
    "a city council meeting", "a camping trip", "a new puppy", "a power cut", "a wedding",
    "an old bookshop", "a traffic jam", "the ocean", "a science experiment", "a music lesson",
    "a neighbour's complaint", "a late flight", "a farmers' market", "a phone call with a friend",
    "a history lecture", "a rainy weekend", "a car repair", "a theatre rehearsal", "a fishing village",
    "a startup office", "a library", "a snowstorm", "a hotel check-in", "a doctor's advice",
    "a recipe gone wrong", "a chess tournament", "a bicycle ride", "a garden party", "a new job",
    "a ferry crossing", "an art class", "a surprise visit", "a lost cat", "a volcano", "a coffee shop",
    "a court case", "a radio interview", "an apartment search", "a jazz concert", "a desert road trip",
]

SYSTEM = ("You write natural English sentences for training a text-to-speech voice. You answer "
          "with one JSON array and nothing else.")

_ALLOWED = re.compile(r"[A-Za-z .,!?;:'()\-—–]+")
_SENT_END = re.compile(r"[.!?]+(?=\s|\)|$)")


# ---- validation and dedupe ----------------------------------------------------------------

def clean_text(text):
    text = " ".join(str(text).split())
    return text.replace("’", "'").replace("‘", "'")


def invalid_reason(text):
    """None if the text is a usable training sentence, else the reason it is not."""
    if not text:
        return "empty"
    if re.search(r"\d", text):
        return "digits"
    n = len(text.split())
    if not MIN_WORDS <= n <= MAX_WORDS:
        return "length"
    if not _ALLOWED.fullmatch(text):
        return "chars"            # quotes, markdown, symbols, non-ASCII letters
    if not text[0].isalpha():
        return "start"            # bullets, list markers
    if not re.search(r"[.!?]\)?$", text):
        return "no_final_punct"
    if not 1 <= len(_SENT_END.findall(text)) <= 3:
        return "sentences"
    return None


def norm(text):
    return " ".join(sftlib.normalize_words(text))


def ngrams(words, n=NGRAM):
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class Deduper:
    """Exact (normalised) + word-5-gram Jaccard dedupe, plus a verbatim benchmark-sentence guard."""

    def __init__(self):
        self.exact = set()
        self.grams = []           # gram sets of indexed texts
        self.index = {}           # gram -> indices into self.grams
        self.bench_sentences = []  # normalised benchmark sentences (>= 5 words), padded with spaces

    def add(self, text):
        words = sftlib.normalize_words(text)
        self.exact.add(" ".join(words))
        g = ngrams(words)
        k = len(self.grams)
        self.grams.append(g)
        for gram in g:
            self.index.setdefault(gram, []).append(k)

    def add_bench(self, text):
        self.add(text)
        for sent in re.split(r"(?<=[.!?])\s+", text):
            words = sftlib.normalize_words(sent)
            if len(words) >= NGRAM:
                self.add(sent)
                self.bench_sentences.append(" " + " ".join(words) + " ")

    def is_dup(self, text):
        words = sftlib.normalize_words(text)
        joined = " ".join(words)
        if joined in self.exact:
            return True
        padded = " " + joined + " "
        if any(s in padded for s in self.bench_sentences):
            return True
        g = ngrams(words)
        candidates = {k for gram in g for k in self.index.get(gram, ())}
        for k in candidates:
            other = self.grams[k]
            if len(g & other) / len(g | other) >= JACCARD_MAX:
                return True
        return False


def bench_texts():
    if not sftlib.BENCH_MANIFEST.exists():
        print(f"[warn] benchmark manifest {sftlib.BENCH_MANIFEST} missing: no benchmark dedupe", flush=True)
        return []
    return sorted({r["text"] for r in sftlib.read_csv(sftlib.BENCH_MANIFEST)})


def build_deduper(out_path):
    d = Deduper()
    for t in bench_texts():
        d.add_bench(t)
    paths = {p.resolve() for p in sftlib.RUNS.glob("*/texts.jsonl")} | {Path(out_path).resolve()}
    n = 0
    for path in sorted(paths):
        for r in sftlib.read_jsonl(path):
            d.add(r["text"])
            n += 1
    print(f"dedupe index: {n} earlier texts from {len(paths)} files + benchmark texts", flush=True)
    return d


# ---- how many texts, of which categories ---------------------------------------------------

def target_n(n, budget_hours, throughput_path):
    """N from --n or the budget formula (falls back to --n if the throughput file is unusable)."""
    if budget_hours is not None:
        s = sftlib.read_json(throughput_path, default=None) if throughput_path else None
        n_gen, gpu_s = (s or {}).get("n_generated"), (s or {}).get("gen_gpu_s")
        if n_gen and gpu_s:
            # the teacher must *render* texts for the whole time box, accepted or not, so size N by
            # render capacity (texts per GPU-second), with RENDER_MARGIN spare; unused texts are harmless
            render = budget_hours * 3600 * n_gen / gpu_s
            n_budget = math.ceil(RENDER_MARGIN * render)
            print(f"budget mode: {budget_hours} GPU-h x {n_gen / gpu_s:.4f} texts rendered/GPU s "
                  f"= ~{render:.0f} renderable x {RENDER_MARGIN} -> N = {n_budget} "
                  f"(yield last time {s.get('yield', 0):.0%}; from {throughput_path})", flush=True)
            return n_budget
        print(f"[warn] throughput file {throughput_path} missing or empty; using --n", flush=True)
    if n is None:
        raise SystemExit("need --n (or a usable --budget-gpu-hours/--throughput)")
    return n


def largest_remainder(total, weights):
    """Split an integer total proportionally to weights; parts sum to total exactly."""
    s = sum(weights.values())
    if total <= 0 or not weights or s <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    parts = {k: int(math.floor(v)) for k, v in raw.items()}
    left = total - sum(parts.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - parts[k]), k))[:left]:
        parts[k] += 1
    return parts


def quotas(n, report):
    confirmed = [c for c in report.get("confirmed", []) if c in sftlib.WEAKNESS_TAXONOMY]
    if not confirmed:
        return {GENERAL: n}
    targeted = round(TARGETED_SHARE * n)
    q = largest_remainder(targeted, {c: report["categories"][c]["share"] for c in confirmed})
    q[GENERAL] = n - targeted
    return q


# ---- 9B prompt ----------------------------------------------------------------------------

def build_prompt(category, report, rng):
    if category == GENERAL:
        desc, hint, evidence = GENERAL_DESC, sftlib.WEAKNESS_TEXT_HINT["other"], []
    else:
        desc = sftlib.WEAKNESS_TAXONOMY[category]
        hint = sftlib.WEAKNESS_TEXT_HINT[category]
        evidence = report["categories"][category].get("evidence", [])[:3]
    lines = [f"Write {PER_CALL} new English texts for training a text-to-speech voice.", ""]
    if category == GENERAL:
        lines.append(f"Purpose: {desc}.")
    else:
        lines.append(f"The current voice has this weakness: {category.replace('_', ' ')} ({desc}).")
        lines.append(f"Write texts that exercise it: {hint}.")
        if evidence:
            lines.append("What a listener said about the current voice:")
            lines += [f"- {e}" for e in evidence]
    if category == GENERAL:
        lines.append(f"Kind of text: {hint}.")
    topics = rng.sample(TOPICS, 6)
    lines += [
        "",
        "Every text must be:",
        "- natural, idiomatic English that someone would really say or read aloud;",
        f"- one to three complete sentences, {MIN_WORDS} to {MAX_WORDS} words in total, ending with . ! or ?;",
        # numbers are avoided entirely: Whisper writes spoken numbers as digits, which would fail the
        # teacher's WER filter for reasons that have nothing to do with the audio
        "- containing no numbers, years, times, dates or amounts at all (neither digits nor number words);",
        "- plain prose: no lists, bullet points, markdown, emojis, hashtags, abbreviations like etc.,",
        "  no double quotation marks and no quoted dialogue with speech tags such as 'she said';",
        "- different from the others: vary the topic, the sentence structure, the opening words and the length.",
        "Do not write about rainbows, prisms or light.",
        f"Draw topics from these (and others): {', '.join(topics)}.",
        "",
        f'Answer with only a JSON array of {PER_CALL} objects: [{{"text": "...", "category": "{category}"}}, ...]',
    ]
    return "\n".join(lines)


def ask(category, report, seed):
    """One 9B call -> list of candidate texts (raises RuntimeError on transport errors)."""
    rng = random.Random(seed)
    reply = tag_weaknesses.llm(build_prompt(category, report, rng), SYSTEM, temperature=0.9, max_tokens=3000)
    items = tag_weaknesses.extract_json(reply, list) or []
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("text"), str):
            out.append(it["text"])
        elif isinstance(it, str):
            out.append(it)
    return out


# ---- main loop ----------------------------------------------------------------------------

def next_index(rows, stage):
    idx = -1
    for r in rows:
        m = re.fullmatch(re.escape(stage) + r"_t(\d+)", r.get("id", ""))
        if m:
            idx = max(idx, int(m.group(1)))
    return max(idx + 1, len(rows))


def generate(report, n, out, stage, workers=4):
    rows = sftlib.read_jsonl(out)
    q = quotas(n, report)
    have = {}
    for r in rows:
        have[r["category"]] = have.get(r["category"], 0) + 1
    remaining = {c: max(0, q[c] - have.get(c, 0)) for c in q}
    total_left = max(0, n - len(rows))
    # never exceed N overall (quotas can disagree with an earlier, differently-mixed run)
    print(f"quotas {q}; {len(rows)} texts exist, {total_left} to write", flush=True)
    if total_left == 0:
        return 0
    if sum(remaining.values()) < total_left:
        remaining[GENERAL] = remaining.get(GENERAL, 0) + total_left - sum(remaining.values())

    dedupe = build_deduper(out)
    idx = next_index(rows, stage)
    rejected = {}
    barren = {c: 0 for c in remaining}
    given_up = set()
    transport_errors = 0
    call_no = len(rows) // PER_CALL * 7 + 1   # varies the topic sampling across resumes
    written, start = 0, time.time()

    def pick(in_flight):
        open_cats = [c for c, v in remaining.items() if v > 0 and c not in given_up]
        # at most ~one call in flight per 20 texts still needed in a category
        cands = [c for c in open_cats if in_flight.get(c, 0) < max(1, math.ceil(remaining[c] / 20))]
        return max(cands, key=lambda c: remaining[c] / max(1, q.get(c, 1))) if cands else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures, in_flight = {}, {}
        while True:
            while len(futures) < workers and written < total_left:
                cat = pick(in_flight)
                if cat is None:
                    break
                seed = zlib.crc32(f"{stage}:{cat}:{call_no}".encode())
                call_no += 1
                futures[pool.submit(ask, cat, report, seed)] = cat
                in_flight[cat] = in_flight.get(cat, 0) + 1
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                cat = futures.pop(fut)
                in_flight[cat] -= 1
                try:
                    cands = fut.result()
                    transport_errors = 0
                except RuntimeError as exc:
                    transport_errors += 1
                    print(f"[error] 9B call failed ({transport_errors}): {exc}", file=sys.stderr, flush=True)
                    if transport_errors >= MAX_TRANSPORT_ERRORS:
                        for f in futures:
                            f.cancel()
                        raise SystemExit(f"9B unreachable; {written} texts written, re-run to resume")
                    time.sleep(5)
                    continue
                accepted = 0
                for raw in cands:
                    if remaining[cat] <= 0 or written >= total_left:
                        break
                    text = clean_text(raw)
                    reason = invalid_reason(text) or ("duplicate" if dedupe.is_dup(text) else None)
                    if reason:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        continue
                    dedupe.add(text)
                    sftlib.append_jsonl(out, {"id": f"{stage}_t{idx:05d}", "text": text,
                                              "category": cat, "stage": stage})
                    idx += 1
                    accepted += 1
                    written += 1
                    remaining[cat] -= 1
                barren[cat] = 0 if accepted else barren[cat] + 1
                if barren[cat] >= MAX_BARREN_CALLS and cat not in given_up:
                    given_up.add(cat)
                    print(f"[warn] {cat}: {MAX_BARREN_CALLS} calls in a row gave no new text; "
                          f"moving its {remaining[cat]} remaining to {GENERAL}", flush=True)
                    if cat != GENERAL:
                        remaining[GENERAL] = remaining.get(GENERAL, 0) + remaining[cat]
                        barren.setdefault(GENERAL, 0)
                    remaining[cat] = 0
                elapsed = time.time() - start
                eta = elapsed / written * (total_left - written) if written else float("nan")
                print(f"[{written}/{total_left}] {cat}: +{accepted}/{len(cands)}  "
                      f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left", flush=True)
    print(f"wrote {written} texts; rejected {rejected}", flush=True)
    if written < total_left:
        print(f"[warn] only {len(rows) + written}/{n} texts (all categories gave up)", flush=True)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--report", required=True, help="weakness_report.json (tag_weaknesses.py)")
    p.add_argument("--out", required=True, help="runs/sft/<stage>/texts.jsonl")
    p.add_argument("--stage", required=True)
    p.add_argument("--n", type=int, help="number of texts (also the fallback in budget mode)")
    p.add_argument("--budget-gpu-hours", type=float)
    p.add_argument("--throughput", help="a previous teacher/filter_summary.json")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.n is None and args.budget_gpu_hours is None:
        p.error("give --n or --budget-gpu-hours with --throughput")

    report = sftlib.read_json(args.report)
    if report is None:
        sys.exit(f"no weakness report at {args.report}")
    n = target_n(args.n, args.budget_gpu_hours, args.throughput)
    start = time.time()
    written = generate(report, n, args.out, args.stage, args.workers)
    rows = sftlib.read_jsonl(args.out)
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"{len(rows)} texts in {args.out}: {by_cat}", flush=True)
    if written:
        sftlib.log_timing(f"{args.stage}_gen_texts", time.time() - start, n_texts=written)


if __name__ == "__main__":
    main()
