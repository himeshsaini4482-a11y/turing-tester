"""Compare dev-set evals: BTRM % with paired-bootstrap CIs, and the guard rails vs base.

An eval dir (runs/sft/eval/<name>/, written by scripts/sft/eval_ckpt.sh) holds the benchmark
formats: manifest.csv (with the dev `set` column, and `item_id`), btrm.jsonl, metrics.jsonl and
generation_config.json. Only rows present in the manifest, btrm.jsonl AND metrics.jsonl are used.
Each dev item is generated with several seeds (`<item_id>__s<k>`); all CIs resample items, so
the seeds of one item move together (sftlib.bootstrap_pct / bootstrap_diff).

  summarize --eval DIR [--base DIR] [--out PATH] -> INTERFACES §2.6 eval_summary.json
                                                    (default out: DIR/eval_summary.json; without
                                                    --base: no vs_base, guards {"pass": true,
                                                    "note": "base"})
  diff --a DIR --b DIR --set mirror [--items ID,ID]  -> prints bootstrap_diff JSON (a - b)

Guards (INTERFACES §2.6) compare mirror + fresh rows combined against the base eval's same rows
(the sentence_ids both evals have): wer <= base + 0.01, sim >= base - 0.02, utmos >= base - 0.10.
A guard whose metric is missing on either side is null (unchecked) and does not fail `pass`.

Env: SFT_BOOTSTRAP_N overrides the number of bootstrap resamples (default 10000; tests only).
Standard library only. Importable: load_eval(dir), summarize(...), diff(...), set_rows(...).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

GUARD_SETS = ("mirror", "fresh")
WER_TOL, SIM_TOL, UTMOS_TOL = 0.01, 0.02, 0.10
EPS = 1e-9
METRIC_KEYS = {"wer": "tts_wer", "sim": "speaker_sim", "utmos": "tts_utmos"}


def n_boot():
    return int(os.environ.get("SFT_BOOTSTRAP_N", "10000"))


def _num(v):
    """A finite float, else None (metrics can be missing, None or NaN)."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _dedupe(rows):
    """btrm.jsonl is appended across incremental evals: keep the last row per sentence_id."""
    out = {}
    for r in rows:
        if "sentence_id" in r:
            out[r["sentence_id"]] = r
    return out


def load_eval(path):
    """Eval dir -> {"name", "dir", "model", "rows": [...], "notes": [...]} or None if absent.

    Each row: sentence_id, item, set, speaker_id, text, tts_reward, human_reward, tts_wer,
    speaker_sim, tts_utmos (metrics may be None). Rows = manifest ∩ btrm ∩ metrics. If the
    manifest has no `set` column every row gets set "all"; without a manifest, btrm order is used.
    """
    path = Path(path)
    btrm_path, metrics_path = path / "btrm.jsonl", path / "metrics.jsonl"
    if not path.is_dir() or not btrm_path.exists():
        return None
    notes = []
    btrm = _dedupe(sftlib.read_jsonl(btrm_path))
    metrics = _dedupe(sftlib.read_jsonl(metrics_path)) if metrics_path.exists() else None
    if metrics is None:
        notes.append("metrics.jsonl missing: guards unchecked, btrm rows used alone")

    manifest = {}
    if (path / "manifest.csv").exists():
        for r in sftlib.read_csv(path / "manifest.csv"):
            manifest[r["sentence_id"]] = r
    else:
        notes.append("manifest.csv missing: set = 'all'")

    order = list(manifest) if manifest else list(btrm)
    rows = []
    for sid in order:
        b = btrm.get(sid)
        if b is None or _num(b.get("tts_reward")) is None or _num(b.get("human_reward")) is None:
            continue
        if metrics is not None and sid not in metrics:
            continue
        m = metrics.get(sid, {}) if metrics is not None else {}
        man = manifest.get(sid, {})
        rows.append({
            "sentence_id": sid,
            "item": man.get("item_id") or sftlib.item_id(sid),
            "set": man.get("set") or "all",
            "speaker_id": man.get("speaker_id") or b.get("speaker_id"),
            "text": man.get("text"),
            "tts_reward": float(b["tts_reward"]),
            "human_reward": float(b["human_reward"]),
            "p_tts_better": _num(b.get("p_tts_better")),
            "tts_wer": _num(m.get("tts_wer")),
            "speaker_sim": _num(m.get("speaker_sim")),
            "tts_utmos": _num(m.get("tts_utmos")),
        })
    gen = sftlib.read_json(path / "generation_config.json", {}) or {}
    return {"name": path.name, "dir": str(path), "model": gen.get("model"), "rows": rows,
            "notes": notes}


def set_rows(ev, set_name=None, items=None):
    """Rows of one set ('all' or None = every row), optionally restricted to some item ids."""
    rows = ev["rows"] if ev else []
    if set_name not in (None, "all", "*"):
        rows = [r for r in rows if r["set"] == set_name]
    if items is not None:
        items = set(items)
        rows = [r for r in rows if r["item"] in items]
    return rows


def mirror_rows(ev):
    """Dev-mirror rows; an eval without set information counts as all-mirror."""
    if not ev:
        return []
    rows = set_rows(ev, "mirror")
    if not rows and all(r["set"] == "all" for r in ev["rows"]):
        rows = ev["rows"]
    return rows


def _boot_rows(rows):
    """sftlib's bootstrap groups by item_id(sentence_id); hand it rows keyed by our item."""
    return [{"sentence_id": r["item"] + f"__s{i}", "tts_reward": r["tts_reward"],
             "human_reward": r["human_reward"]} for i, r in enumerate(rows)]


def pct(rows):
    """BTRM % with a 95% CI (items resampled), or None if there are no rows."""
    if not rows:
        return None
    return sftlib.bootstrap_pct(_boot_rows(rows), n=n_boot())


def paired_diff(rows_a, rows_b):
    """BTRM %(a) - BTRM %(b) on common items with a paired CI, or None if nothing in common."""
    if not rows_a or not rows_b:
        return None
    try:
        return sftlib.bootstrap_diff(_boot_rows(rows_a), _boot_rows(rows_b), n=n_boot())
    except ValueError:
        return None


def metric_means(rows):
    out = {}
    for short, key in METRIC_KEYS.items():
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[short] = mean(vals) if vals else None
    return out


def guards(ev, base):
    """Guard rails on mirror + fresh rows vs the base eval's same sentence_ids."""
    a = {r["sentence_id"]: r for r in ev["rows"] if r["set"] in GUARD_SETS}
    b = {r["sentence_id"]: r for r in base["rows"] if r["set"] in GUARD_SETS}
    if not a or not b:   # e.g. a run without dev sets: fall back to every row
        a = {r["sentence_id"]: r for r in ev["rows"]}
        b = {r["sentence_id"]: r for r in base["rows"]}
    common = sorted(set(a) & set(b))
    ma = metric_means([a[s] for s in common])
    mb = metric_means([b[s] for s in common])

    def check(key, ok):
        if ma[key] is None or mb[key] is None:
            return None
        return bool(ok(ma[key], mb[key]))

    g = {
        "wer_ok": check("wer", lambda x, y: x <= y + WER_TOL + EPS),
        "sim_ok": check("sim", lambda x, y: x >= y - SIM_TOL - EPS),
        "utmos_ok": check("utmos", lambda x, y: x >= y - UTMOS_TOL - EPS),
    }
    g["pass"] = all(v is not False for v in g.values()) and bool(common)
    g["unchecked"] = [k for k, v in g.items() if v is None]
    g["n_rows"] = len(common)
    g["values"] = {k: ma[k] for k in METRIC_KEYS}
    g["base_values"] = {k: mb[k] for k in METRIC_KEYS}
    return g


def sets_of(ev):
    return sorted({r["set"] for r in ev["rows"]}, key=lambda s: (s != "mirror", s != "fresh", s))


def summarize(eval_dir, base_dir=None, out=None):
    """INTERFACES §2.6 summary of one eval (vs base if given). Writes `out` when given."""
    ev = load_eval(eval_dir)
    if ev is None or not ev["rows"]:
        raise SystemExit(f"no scored rows in {eval_dir} (btrm.jsonl + metrics.jsonl needed)")
    base = None
    if base_dir and Path(base_dir).resolve() != Path(eval_dir).resolve():
        base = load_eval(base_dir)
        if base is None or not base["rows"]:
            print(f"WARNING: base eval {base_dir} has no scored rows; vs_base/guards skipped",
                  flush=True)
            base = None

    summary = summary_of(ev, base)
    if base is None and base_dir and Path(base_dir).resolve() != Path(eval_dir).resolve():
        summary["guards"] = {"pass": True, "note": f"base eval {base_dir} missing: unchecked"}
    if out:
        sftlib.write_json(out, summary)
    return summary


def summary_of(ev, base=None):
    """§2.6 summary of an already-loaded eval (vs an already-loaded base, or None)."""
    summary = {"name": ev["name"], "model": ev["model"], "sets": {}}
    for s in sets_of(ev):
        rows = set_rows(ev, s)
        try:
            p = pct(rows)
        except ValueError as e:
            # BTRM % = mean TTS reward / mean human reward needs a positive human mean. EARS
            # freeform (dev-fresh) human clips average below 0 under BTRM, so the % is undefined
            # there: record it as such with the raw reward gap. Every decision reads dev-mirror.
            gap = [r["tts_reward"] - r["human_reward"] for r in rows]
            summary["sets"][s] = {"btrm_pct": None, "lo": None, "hi": None, "undefined": str(e),
                                  "mean_reward_gap": mean(gap) if gap else None,
                                  "n_items": len({r["item"] for r in rows}), "n_rows": len(rows),
                                  **metric_means(rows)}
            continue
        summary["sets"][s] = {"btrm_pct": p["pct"], "lo": p["lo"], "hi": p["hi"],
                              "n_items": p["n_items"], "n_rows": p["n_rows"], **metric_means(rows)}
    if base is not None:
        vs = {"base": base["name"]}
        for s in summary["sets"]:
            d = paired_diff(set_rows(ev, s), set_rows(base, s))
            if d is not None:
                vs[s] = d
        summary["vs_base"] = vs
        summary["guards"] = guards(ev, base)
    else:   # a base eval summarised on its own (core contract): no vs_base, trivial guards
        summary["guards"] = {"pass": True, "note": "base"}
    if ev["notes"]:
        summary["notes"] = ev["notes"]
    return summary


def diff(a_dir, b_dir, set_name="mirror", items=None):
    """bootstrap_diff(a - b) on one set (optionally some items). Raises if nothing is common."""
    a, b = load_eval(a_dir), load_eval(b_dir)
    for name, ev in ((a_dir, a), (b_dir, b)):
        if ev is None:
            raise SystemExit(f"no eval at {name}")
    d = paired_diff(set_rows(a, set_name, items), set_rows(b, set_name, items))
    if d is None:
        raise SystemExit(f"no common scored items between {a_dir} and {b_dir} (set {set_name})")
    return d


def fmt_ci(d, signed=False):
    """'41.2 [39.0, 43.5]' (or '+5.1 [+2.9, +7.2]') for a pct/diff dict; '—' when missing."""
    if not d:
        return "—"
    v = d.get("diff", d.get("pct", d.get("btrm_pct")))
    if v is None:
        return "—"
    f = "{:+.1f}" if signed else "{:.1f}"
    return f"{f.format(v)} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--eval", required=True)
    s.add_argument("--base", default="")
    s.add_argument("--out", default="")
    d = sub.add_parser("diff")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--set", default="mirror")
    d.add_argument("--items", default="")
    args = p.parse_args()

    if args.cmd == "summarize":
        out = args.out or os.path.join(args.eval, "eval_summary.json")
        summary = summarize(args.eval, args.base or None, out)
        for s_name, v in summary["sets"].items():
            vs = (summary.get("vs_base") or {}).get(s_name)
            print(f"{summary['name']} {s_name}: BTRM {fmt_ci(v)}  vs base {fmt_ci(vs, True)}  "
                  f"(n_items {v['n_items']}, rows {v['n_rows']})", flush=True)
        g = summary["guards"]
        print(f"guards: wer_ok {g.get('wer_ok')}  sim_ok {g.get('sim_ok')}  "
              f"utmos_ok {g.get('utmos_ok')}  pass {g['pass']}  -> {out}", flush=True)
    else:
        items = [x for x in args.items.split(",") if x] or None
        print(json.dumps(diff(args.a, args.b, args.set, items)), flush=True)


if __name__ == "__main__":
    main()
