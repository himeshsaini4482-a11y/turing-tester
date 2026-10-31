"""Round stage R-2: tag the GRM judge's critiques of the student into the fixed weakness taxonomy.

Input is a diagnosis dir written by scripts/sft/diagnose.sh: SpeechJudge-GRM compared the student
(TTS) clip with the human clip of each seed-0 dev-mirror item in both A/B orders
(<diag-dir>/judgments.jsonl). For every judgment where the judge preferred the human
(human_score > tts_score), the Qwen 3.5 9B reads the judge's analysis and labels the TTS clip's
weaknesses with keys of sftlib.WEAKNESS_TAXONOMY. Only the analysis text and the taxonomy are sent
to the 9B (no sentence text, ids or audio).

The 9B must answer strict JSON {"weaknesses": [...], "evidence": "<short quote>"}. The first JSON
object in the reply is used and unknown keys are dropped; an answer with no valid key is retried
twice, then tagged ["other"]. Transport errors (endpoint down) are NOT cached: the stage exits 1
and a re-run resumes.

Aggregation (INTERFACES §2.5): per category count (tagged human-preferred judgments), share =
count / n_human_preferred, n_items (distinct dev items), evidence quotes; confirmed = share >= 0.15
and n_items >= 8, ordered by share. checkpoint_metrics come from the eval dir (btrm.jsonl ->
sftlib.btrm_pct, metrics.jsonl means of tts_wer / speaker_sim) over the diagnosed rows.

Writes <diag-dir>/tags.jsonl (cache, one row per tagged judgment), weakness_report.json and
weakness_report.md. Env: STD (any venv). Resumable: judgments already in tags.jsonl are skipped.

  python sft/tag_weaknesses.py --diag-dir runs/sft/pilot/diag --eval-dir runs/sft/eval/base_chunked --stage pilot
"""
import argparse
import http.client
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

SHARE_MIN = 0.15
ITEMS_MIN = 8
N_EVIDENCE = 5          # quotes kept per category in the report
ATTEMPTS = 3            # first try + 2 retries

# Tests replace this with a mock; by default scripts/llm_client.chat is imported on first use.
CHAT = None


def llm(prompt, system, temperature, max_tokens):
    """One 9B call -> reply text. Raises RuntimeError on transport errors."""
    global CHAT
    if CHAT is None:
        sftlib.add_scripts_to_path()
        from llm_client import chat
        CHAT = chat
    try:
        resp = CHAT(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # timeouts, resets, IncompleteRead, bad JSON bodies: the client only wraps HTTP/URL errors
        raise RuntimeError(f"9B call failed: {type(exc).__name__}: {exc}") from exc
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"9B response without choices[0].message.content: {str(resp)[:200]}") from exc


# ---- robust JSON extraction from LLM replies ----------------------------------------------

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


def _clean(text):
    return _THINK.sub("", text or "")


def extract_json(text, kind=dict):
    """The first JSON value of type `kind` (dict or list) in a messy reply, or None.

    Handles code fences, prose around the JSON, <think> blocks and trailing garbage. For lists, a
    reply cut off mid-array (max_tokens) is salvaged as the list of complete {...} objects in it.
    """
    text = _clean(text)
    dec = json.JSONDecoder()
    opener = "{" if kind is dict else "["
    for m in re.finditer(re.escape(opener), text):
        try:
            value, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(value, kind):
            return value
    if kind is list:
        start = text.find("[")
        objs = []
        pos = start + 1 if start >= 0 else 0
        while True:
            i = text.find("{", pos)
            if i < 0:
                break
            try:
                value, end = dec.raw_decode(text, i)
            except ValueError:
                pos = i + 1
                continue
            if isinstance(value, dict):
                objs.append(value)
            pos = end
        return objs or None
    return None


def normalize_key(key):
    return re.sub(r"[^a-z]+", "_", str(key).lower()).strip("_")


def parse_tags(reply):
    """9B reply -> (weaknesses, evidence) or None if it has no valid taxonomy key."""
    obj = extract_json(reply, dict)
    if not obj:
        return None
    raw = obj.get("weaknesses")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    keys = []
    for k in raw:
        k = normalize_key(k)
        if k in sftlib.WEAKNESS_TAXONOMY and k not in keys:
            keys.append(k)
    if not keys:
        return None
    evidence = obj.get("evidence", "")
    if isinstance(evidence, list):
        evidence = " ".join(str(e) for e in evidence)
    evidence = " ".join(str(evidence).split())[:240]
    return keys, evidence


# ---- tagging ------------------------------------------------------------------------------

SYSTEM = ("You are a careful annotator of speech-quality critiques. You answer with one strict JSON "
          "object and nothing else.")


def tts_output(order):
    """judge.py: tts_first -> the TTS clip is Output A; human_first -> it is Output B."""
    return "A" if order == "tts_first" else "B"


def build_prompt(analysis, order):
    tts = tts_output(order)
    other = "B" if tts == "A" else "A"
    taxonomy = "\n".join(f"- {k}: {v}" for k, v in sftlib.WEAKNESS_TAXONOMY.items())
    return (
        "A speech-quality judge compared two recordings of the same sentence, Output A and Output B, "
        "and wrote the analysis below.\n"
        f"Output {tts} is the synthetic (TTS) clip. Output {other} is a human recording.\n\n"
        f"Label the weaknesses of the TTS clip (Output {tts}) only, using these categories:\n"
        f"{taxonomy}\n\n"
        "Rules:\n"
        f"- Only list problems the analysis attributes to Output {tts}. Ignore remarks about Output {other} "
        "and positive remarks.\n"
        "- Use only the category keys above; use \"other\" only if no other key fits.\n"
        "- \"evidence\" is a short quote (at most 20 words) from the analysis supporting the main weakness.\n"
        'Answer with exactly one JSON object: {"weaknesses": ["<key>", ...], "evidence": "<short quote>"}\n\n'
        f"Judge's analysis:\n\"\"\"\n{analysis}\n\"\"\""
    )


def tag_one(j):
    """-> tags row, or raises RuntimeError when every attempt failed on transport."""
    prompt = build_prompt(j.get("analysis") or "", j["order"])
    transport_errors, last_reply = 0, ""
    for attempt in range(ATTEMPTS):
        try:
            reply = llm(prompt, SYSTEM, temperature=0.2 if attempt == 0 else 0.5, max_tokens=300)
        except RuntimeError as exc:
            transport_errors += 1
            last_reply = str(exc)
            time.sleep(2 * (attempt + 1))
            continue
        last_reply = reply
        parsed = parse_tags(reply)
        if parsed:
            return _row(j, parsed[0], parsed[1], "ok", attempt + 1)
    if transport_errors == ATTEMPTS:
        raise RuntimeError(f"9B unreachable for {j['sentence_id']}/{j['order']}: {last_reply[:200]}")
    return _row(j, ["other"], "", "fallback", ATTEMPTS)


def _row(j, weaknesses, evidence, status, attempts):
    return {"sentence_id": j["sentence_id"], "speaker_id": j.get("speaker_id"), "order": j["order"],
            "vote": j.get("vote", 0), "tts_output": tts_output(j["order"]),
            "tts_score": j["tts_score"], "human_score": j["human_score"],
            "weaknesses": weaknesses, "evidence": evidence, "status": status, "attempts": attempts}


def key_of(r):
    # INTERFACES keys tags by sentence_id + order; the vote is included so --votes > 1 also works
    return (r["sentence_id"], r["order"], int(r.get("vote", 0)))


def run_tagging(judgments, tags_path, workers):
    done = {key_of(r) for r in sftlib.read_jsonl(tags_path)}
    todo = [j for j in judgments if j["human_score"] > j["tts_score"] and key_of(j) not in done]
    print(f"{len(done)} judgments already tagged, {len(todo)} to tag", flush=True)
    if not todo:
        return 0
    lock = threading.Lock()
    start, n_done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(tag_one, j): j for j in todo}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except RuntimeError as exc:
                failed += 1
                print(f"[error] {exc}", file=sys.stderr, flush=True)
                continue
            with lock:
                sftlib.append_jsonl(tags_path, row)
                n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                elapsed = time.time() - start
                eta = elapsed / n_done * (len(todo) - n_done - failed)
                print(f"[{n_done}/{len(todo)}] {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                      flush=True)
    return failed


# ---- aggregation --------------------------------------------------------------------------

def aggregate(judgments, tags):
    """INTERFACES §2.5 categories / confirmed from the judgments and their tags."""
    preferred = [j for j in judgments if j["human_score"] > j["tts_score"]]
    tag_of = {key_of(t): t for t in tags}
    n_pref = len(preferred)
    cats = {}
    for cat in sftlib.WEAKNESS_TAXONOMY:
        rows = [tag_of[key_of(j)] for j in preferred
                if key_of(j) in tag_of and cat in tag_of[key_of(j)]["weaknesses"]]
        evidence = []
        for t in rows:
            e = t.get("evidence") or ""
            if e and e not in evidence and len(evidence) < N_EVIDENCE:
                evidence.append(e)
        count = len(rows)
        share = count / n_pref if n_pref else 0.0
        n_items = len({sftlib.item_id(t["sentence_id"]) for t in rows})
        cats[cat] = {"count": count, "share": round(share, 4), "n_items": n_items,
                     "confirmed": share >= SHARE_MIN and n_items >= ITEMS_MIN, "evidence": evidence}
    confirmed = sorted((c for c, v in cats.items() if v["confirmed"]),
                       key=lambda c: (-cats[c]["share"], -cats[c]["count"], c))
    return n_pref, cats, confirmed


def checkpoint_metrics(eval_dir, sentence_ids):
    """BTRM % and mean WER / speaker sim of the eval run over the diagnosed rows."""
    out = {"btrm_pct": None, "wer": None, "sim": None, "n_rows": 0}
    if not eval_dir:
        return out
    btrm = [r for r in sftlib.read_jsonl(os.path.join(eval_dir, "btrm.jsonl")) if r["sentence_id"] in sentence_ids]
    if btrm:
        out["btrm_pct"] = round(sftlib.btrm_pct(btrm), 2)
        out["n_rows"] = len(btrm)
    metrics = [r for r in sftlib.read_jsonl(os.path.join(eval_dir, "metrics.jsonl")) if r["sentence_id"] in sentence_ids]
    for key, name in (("tts_wer", "wer"), ("speaker_sim", "sim")):
        vals = [r[key] for r in metrics if isinstance(r.get(key), (int, float))]
        if vals:
            out[name] = round(mean(vals), 4)
    return out


def write_markdown(path, report):
    lines = [f"# Weakness report: {report['stage']} (eval `{report['eval']}`)", "",
             f"- GRM judgments: {report['n_judgments']}, human preferred: {report['n_human_preferred']}",
             f"- Tagged: {report['n_tagged']} ({report['n_fallback']} fell back to `other`)",
             f"- Confirmed (share >= {SHARE_MIN:.0%} and >= {ITEMS_MIN} items): "
             + (", ".join(f"`{c}`" for c in report["confirmed"]) or "none (texts will be 100% general)")]
    m = report["checkpoint_metrics"]
    fmt = lambda v, f: "n/a" if v is None else format(v, f)  # noqa: E731
    lines += [f"- Checkpoint on the diagnosed rows: BTRM {fmt(m['btrm_pct'], '.1f')}%, "
              f"WER {fmt(m['wer'], '.4f')}, speaker sim {fmt(m['sim'], '.4f')}", "",
              "| category | count | share | items | confirmed |", "|---|---:|---:|---:|:---:|"]
    order = sorted(report["categories"], key=lambda c: -report["categories"][c]["share"])
    for c in order:
        v = report["categories"][c]
        lines.append(f"| {c} | {v['count']} | {v['share']:.1%} | {v['n_items']} | {'yes' if v['confirmed'] else ''} |")
    lines += ["", "## Evidence (judge quotes about the TTS clip)", ""]
    for c in order:
        v = report["categories"][c]
        if v["evidence"]:
            lines.append(f"**{c}**")
            lines += [f"- \"{e}\"" for e in v["evidence"]]
            lines.append("")
    tmp = str(path) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--diag-dir", required=True, help="dir with judgments.jsonl (diagnose.sh)")
    p.add_argument("--eval-dir", default="", help="runs/sft/eval/<name> with btrm.jsonl, metrics.jsonl")
    p.add_argument("--stage", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    judgments = sftlib.read_jsonl(os.path.join(args.diag_dir, "judgments.jsonl"))
    if not judgments:
        sys.exit(f"no judgments in {args.diag_dir}/judgments.jsonl (run diagnose.sh first)")
    tags_path = os.path.join(args.diag_dir, "tags.jsonl")
    failed = run_tagging(judgments, tags_path, args.workers)
    if failed:
        sys.exit(f"{failed} judgments could not be tagged (9B unreachable); re-run to resume")

    tags = sftlib.read_jsonl(tags_path)
    n_pref, cats, confirmed = aggregate(judgments, tags)
    manifest = os.path.join(args.diag_dir, "manifest.csv")
    sids = ({r["sentence_id"] for r in sftlib.read_csv(manifest)} if os.path.exists(manifest)
            else {j["sentence_id"] for j in judgments})
    eval_name = os.path.basename(os.path.normpath(args.eval_dir)) if args.eval_dir else None
    used = {key_of(j) for j in judgments if j["human_score"] > j["tts_score"]}
    used_tags = [t for t in tags if key_of(t) in used]
    report = {
        "stage": args.stage, "eval": eval_name,
        "n_judgments": len(judgments), "n_human_preferred": n_pref,
        "n_tagged": len(used_tags), "n_fallback": sum(t["status"] == "fallback" for t in used_tags),
        "categories": cats, "confirmed": confirmed,
        "checkpoint_metrics": checkpoint_metrics(args.eval_dir, sids),
        "thresholds": {"share": SHARE_MIN, "n_items": ITEMS_MIN},
    }
    sftlib.write_json(os.path.join(args.diag_dir, "weakness_report.json"), report)
    write_markdown(os.path.join(args.diag_dir, "weakness_report.md"), report)
    print(f"{n_pref}/{len(judgments)} judgments preferred the human; confirmed: {confirmed or 'none'}", flush=True)
    for c in sorted(cats, key=lambda c: -cats[c]["share"]):
        if cats[c]["count"]:
            print(f"  {c}: {cats[c]['count']} ({cats[c]['share']:.0%}, {cats[c]['n_items']} items)"
                  f"{'  CONFIRMED' if cats[c]['confirmed'] else ''}", flush=True)


if __name__ == "__main__":
    main()
