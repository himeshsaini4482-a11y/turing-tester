"""report.md for a benchmark run, and before/after comparisons for SFT and DPO. Standard library only.

Scores (README "What you get"):
  GRM closeness  per sentence TTS score / human score x 100 (SpeechJudge-GRM, both A/B orders), averaged
  BTRM %         mean TTS reward / mean human reward x 100 (SpeechJudge-BTRM); undefined when the human
                 mean is <= 0 (rewards are unbounded logits)
  BTRM parity    200 x mean sigmoid(r_tts - r_human): 100 = as natural as the recording under Bradley-Terry
95% intervals resample sentences (10,000 bootstrap draws); comparisons are paired by sentence.
"""
import json
import math
import random
from pathlib import Path
from statistics import mean

N_BOOT = 10000
DISCLAIMER = ("These are automatic judges (SpeechJudge-GRM, SpeechJudge-BTRM, Whisper, WavLM, UTMOS), not human "
              "listeners. A score above 100 means the judge preferred the TTS clip, not that people would.")


def _jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _manifest_ids(run_dir):
    import csv
    p = Path(run_dir) / "manifest.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return [r["sentence_id"] for r in csv.DictReader(f)]


def load_run(run_dir):
    """Per-sentence scores of one run dir: {sid: {speaker, grm_tts, grm_human, r_tts, r_human, wer, sim, utmos, ...}}."""
    run_dir = Path(run_dir)
    ids = _manifest_ids(run_dir)
    rows = {}

    def row(sid, spk=None):
        r = rows.setdefault(sid, {"speaker": spk})
        if spk and not r.get("speaker"):
            r["speaker"] = spk
        return r

    grm = {}
    for j in _jsonl(run_dir / "judgments.jsonl"):
        g = grm.setdefault(j["sentence_id"], {"speaker": j["speaker_id"], "tts": [], "human": []})
        g["tts"].append(j["tts_score"])
        g["human"].append(j["human_score"])
    for sid, g in grm.items():
        r = row(sid, g["speaker"])
        r["grm_tts"], r["grm_human"] = mean(g["tts"]), mean(g["human"])
    for b in _jsonl(run_dir / "btrm.jsonl"):
        r = row(b["sentence_id"], b.get("speaker_id"))
        r["r_tts"], r["r_human"] = b["tts_reward"], b["human_reward"]
    for m in _jsonl(run_dir / "metrics.jsonl"):
        r = row(m["sentence_id"], m.get("speaker_id"))
        for k in ("tts_wer", "human_wer", "speaker_sim", "tts_utmos", "human_utmos"):
            if isinstance(m.get(k), (int, float)) and not (isinstance(m[k], float) and math.isnan(m[k])):
                r[k] = m[k]
    if ids is not None:
        keep = set(ids)
        rows = {k: v for k, v in rows.items() if k in keep}
    return rows


# ---- statistics ---------------------------------------------------------------------------

def grm_closeness(rows):
    vals = [r["grm_tts"] / r["grm_human"] * 100 for r in rows if "grm_tts" in r and r["grm_human"]]
    return mean(vals) if vals else None


def btrm_pct(rows):
    rs = [r for r in rows if "r_tts" in r]
    if not rs:
        return None
    h = mean(r["r_human"] for r in rs)
    return 100 * mean(r["r_tts"] for r in rs) / h if h > 0 else None


def btrm_parity(rows):
    rs = [r for r in rows if "r_tts" in r]
    if not rs:
        return None
    return 200 * mean(1 / (1 + math.exp(r["r_human"] - r["r_tts"])) for r in rs)


def metric(rows, key):
    vals = [r[key] for r in rows if key in r]
    return mean(vals) if vals else None


def bootstrap(stat, rows, n=N_BOOT, seed=0):
    """(value, lo, hi) of stat over rows, resampling rows; None when stat is undefined."""
    v = stat(rows)
    if v is None or not rows:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        s = stat(rng.choices(rows, k=len(rows)))
        if s is not None:
            draws.append(s)
    if len(draws) < n * 0.9:
        return (v, None, None)
    draws.sort()
    return (v, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1])


def paired(stat, a_rows, b_rows, n=N_BOOT, seed=0):
    """stat(a) - stat(b) over common sentences, with a paired 95% interval."""
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        return None
    a = [a_rows[k] for k in common]
    b = [b_rows[k] for k in common]
    va, vb = stat(a), stat(b)
    if va is None or vb is None:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        x, y = stat([a[i] for i in idx]), stat([b[i] for i in idx])
        if x is not None and y is not None:
            draws.append(x - y)
    draws.sort()
    if len(draws) < n * 0.9:
        return (va - vb, None, None, len(common))
    return (va - vb, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1], len(common))


# ---- formatting ---------------------------------------------------------------------------

def f1(v, signed=False):
    if v is None:
        return "—"
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def ci(t, signed=False):
    if not t:
        return "—"
    if t[1] is None:
        return f1(t[0], signed)
    return f"{f1(t[0], signed)} [{f1(t[1], signed)}, {f1(t[2], signed)}]"


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def summary(rows_by_id):
    rows = list(rows_by_id.values())
    return {
        "n_sentences": len(rows),
        "grm_closeness": bootstrap(grm_closeness, [r for r in rows if "grm_tts" in r]),
        "btrm_pct": bootstrap(btrm_pct, [r for r in rows if "r_tts" in r]),
        "btrm_parity": btrm_parity(rows),
        "btrm_wins": sum(r["r_tts"] > r["r_human"] for r in rows if "r_tts" in r),
        "btrm_n": sum("r_tts" in r for r in rows),
        "human_btrm_mean": metric(rows, "r_human"),
        "wer": metric(rows, "tts_wer"), "human_wer": metric(rows, "human_wer"),
        "sim": metric(rows, "speaker_sim"),
        "utmos": metric(rows, "tts_utmos"), "human_utmos": metric(rows, "human_utmos"),
    }


def benchmark_report(run_dir, meta):
    """Write <run_dir>/report.md and report.json; returns the summary dict."""
    run_dir = Path(run_dir)
    rows = load_run(run_dir)
    s = summary(rows)
    no_clone = meta.get("voice_cloning") == "none"
    lines = [f"# VoiceGym benchmark: {meta.get('name', run_dir.name)}", "",
             f"- Model: {meta.get('model_label', '?')} (`{meta.get('model', '?')}`)",
             f"- Sentence set: {meta.get('set_description', meta.get('set', '?'))} ({s['n_sentences']} sentences"
             + (f", first {meta['limit']} only" if meta.get("limit") else "") + ")",
             f"- Judges: {', '.join(meta.get('judges', []))}", ""]
    if no_clone:
        lines += ["> **Warning:** this API cannot clone voices, so each clip is compared with a recording of a "
                  "different speaker. The comparison is less fair, and speaker similarity is not available.", ""]
    if meta.get("limit"):
        lines += [f"> Only {meta['limit']} sentences: a quick check, too small to trust.", ""]
    hb = s["human_btrm_mean"]
    btrm_note = ""
    if s["btrm_n"] and s["btrm_pct"] is None:
        btrm_note = (f" (undefined: the mean human reward is {hb:.2f} <= 0 on this set; use BTRM parity)"
                     if hb is not None else "")
    lines += ["| Score | This model | Real human |", "|---|---|---|",
              f"| **GRM closeness** | {ci(s['grm_closeness'])} | 100 |",
              f"| **BTRM %** | {ci(s['btrm_pct'])}{btrm_note} | 100 |",
              f"| BTRM parity | {f1(s['btrm_parity'])} ({s['btrm_wins']}/{s['btrm_n']} sentences preferred over the human) | 100 |",
              f"| WER | {pct(s['wer'])} | {pct(s['human_wer'])} |",
              f"| Speaker similarity | {'not available' if no_clone else f3(s['sim'])} | 1.0 |",
              f"| UTMOS | {f2(s['utmos'])} | {f2(s['human_utmos'])} |", "",
              "GRM closeness: SpeechJudge-GRM scores the TTS and the human clip of each sentence in both A/B "
              "orders; TTS ÷ human × 100, averaged over sentences. BTRM %: mean SpeechJudge-BTRM reward of "
              "the TTS clips ÷ that of the human clips × 100. Brackets: 95% intervals over sentences.", ""]
    speakers = sorted({r.get("speaker") or "?" for r in rows.values()})
    if len(speakers) > 1:
        lines += ["## Per speaker", "", "| Speaker | Sentences | GRM closeness | BTRM % | WER | Similarity |",
                  "|---|---|---|---|---|---|"]
        for spk in speakers:
            rs = [r for r in rows.values() if (r.get("speaker") or "?") == spk]
            lines.append(f"| {spk} | {len(rs)} | {f1(grm_closeness(rs))} | {f1(btrm_pct(rs))} | "
                         f"{pct(metric(rs, 'tts_wer'))} | {'—' if no_clone else f3(metric(rs, 'speaker_sim'))} |")
        lines.append("")
    lines += [DISCLAIMER, "", "Raw numbers: `report.json`; per sentence: `judgments.jsonl` (GRM, with the judge's "
              "written analysis), `btrm.jsonl`, `metrics.jsonl`; audio: `tts/`.", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    out = {"meta": meta, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}}
    (run_dir / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return s


def comparison_lines(base_dir, new_dir, base_label, new_label):
    """Markdown table: base vs new on the same sentences, with paired differences."""
    a, b = load_run(new_dir), load_run(base_dir)
    sa, sb = summary(a), summary(b)
    d_grm = paired(grm_closeness, {k: v for k, v in a.items() if "grm_tts" in v},
                   {k: v for k, v in b.items() if "grm_tts" in v})
    d_btrm = paired(btrm_pct, {k: v for k, v in a.items() if "r_tts" in v}, {k: v for k, v in b.items() if "r_tts" in v})

    def diff(d):
        if not d:
            return "—"
        if d[1] is None:
            return f1(d[0], True)
        sig = " (clear gain)" if d[1] > 0 else " (clear loss)" if d[2] < 0 else " (not significant)"
        return f"{f1(d[0], True)} [{f1(d[1], True)}, {f1(d[2], True)}]{sig}"

    return [f"| Score | {base_label} | {new_label} | Change (paired) |", "|---|---|---|---|",
            f"| GRM closeness | {ci(sb['grm_closeness'])} | {ci(sa['grm_closeness'])} | {diff(d_grm)} |",
            f"| BTRM % | {ci(sb['btrm_pct'])} | {ci(sa['btrm_pct'])} | {diff(d_btrm)} |",
            f"| BTRM parity | {f1(sb['btrm_parity'])} | {f1(sa['btrm_parity'])} | |",
            f"| WER | {pct(sb['wer'])} | {pct(sa['wer'])} | |",
            f"| Speaker similarity | {f3(sb['sim'])} | {f3(sa['sim'])} | |",
            f"| UTMOS | {f2(sb['utmos'])} | {f2(sa['utmos'])} | |", "",
            "Change: new − base over the same sentences, 95% paired interval. \"Clear\" = the interval "
            "excludes 0.", ""]
