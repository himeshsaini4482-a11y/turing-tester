"""Final SFT report: runs/sft/REPORT.md (PLAN §11).

Reads whatever exists and notes what is missing (every section renders with partial inputs):
  - 494-set run dirs runs/{moss_base_chunked, moss_base_whole, sft_final_base, sft_final}
    (btrm.jsonl, btrm_summary.json, summary.json = GRM, metrics_summary.json)
  - runs/sft/pilot/recipe.json (+ pilot_report.md), runs/sft/decoding.json
  - runs/sft/main_r*/round_summary.json (core round_select.py) and runs/sft/main/best.json
  - runs/sft/*/diag/weakness_report.json, runs/sft/*/teacher/filter_summary.json
  - runs/sft/data/*summary.json, runs/sft/timings.json, runs/sft/DECISIONS.md

Headline: BTRM % with a 95% CI (sentences resampled), paired diff vs base (runs/sft_final_base if
present, else runs/moss_base_<decoding mode>), BTRM parity / ratio / wins, GRM closeness, WER,
speaker sim, UTMOS; per-speaker BTRM %. Then pilot, rounds, weakness shares, data, teacher
yields, worker time and cost ($2.24/h), and the Disclosure section, which is always printed.

  python sft/report.py [--runs-dir runs/sft] [--out runs/sft/REPORT.md]

Env: SFT_BOOTSTRAP_N (tests only). Standard library only.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402
import compare  # noqa: E402

USD_PER_HOUR = 2.24
BENCH_RUNS = ("moss_base_chunked", "moss_base_whole", "sft_final_base", "sft_final", "sft_final_runner_up")
RUN_LABEL = {"moss_base_chunked": "base, chunked (50)", "moss_base_whole": "base, whole (100)",
             "sft_final_base": "base, chosen decoding", "sft_final": "**SFT final**",
             "sft_final_runner_up": "SFT runner-up"}


def f1(v, signed=False):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def f3(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"


def ci(d, signed=False):
    return compare.fmt_ci(d, signed)


def stage_key(name):
    m = re.fullmatch(r"main_r(\d+)", name)
    return (0, 0) if name == "pilot" else (1, int(m.group(1))) if m else (2, name)


# ---- 494-set runs -------------------------------------------------------------------------

def load_bench(run_dir):
    """A 494 run dir -> dict with btrm rows (item = sentence_id) and its summaries, or None."""
    btrm = {r["sentence_id"]: r for r in sftlib.read_jsonl(run_dir / "btrm.jsonl")
            if "sentence_id" in r}
    if not btrm and not (run_dir / "btrm_summary.json").exists():
        return None
    rows = [{"sentence_id": sid, "item": sid, "speaker_id": r.get("speaker_id"),
             "tts_reward": float(r["tts_reward"]), "human_reward": float(r["human_reward"])}
            for sid, r in btrm.items()
            if r.get("tts_reward") is not None and r.get("human_reward") is not None]
    return {"name": run_dir.name, "rows": rows,
            "btrm": sftlib.read_json(run_dir / "btrm_summary.json", {}) or {},
            "grm": sftlib.read_json(run_dir / "summary.json"),
            "metrics": (sftlib.read_json(run_dir / "metrics_summary.json", {}) or {}).get("overall", {})}


def headline(lines, notes, mode):
    # the voicegym CLI (--mode full) keeps them under runs/sft/<name>/ via SFT_BENCH_ROOT + SFT_BENCH_PREFIX
    root = Path(os.environ["SFT_BENCH_ROOT"]) if os.environ.get("SFT_BENCH_ROOT") else sftlib.RUNS.parent
    prefix = os.environ.get("SFT_BENCH_PREFIX", "")
    runs = {}
    for name in BENCH_RUNS:
        r = load_bench(root / (prefix + name))
        if r is None:
            notes.append(f"494 run `runs/{name}` missing: skipped")
        else:
            runs[name] = r
    ref = "sft_final_base" if "sft_final_base" in runs else f"moss_base_{mode}"
    lines += ["## Headline: the 494-sentence benchmark", ""]
    if not runs:
        lines += ["No 494-set run found.", ""]
        return
    if ref not in runs:
        notes.append(f"reference base `runs/{ref}` missing: no paired diff vs base")
    lines += [f"Paired differences are vs `runs/{ref}` (base under the chosen decoding).", "",
              "| run | n | BTRM % [95% CI] | Δ vs base [paired CI] | BTRM parity | ratio | "
              "TTS wins | GRM closeness | WER | spk sim | UTMOS |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        p = compare.pct(r["rows"])
        d = None
        if name != ref and ref in runs:
            d = compare.paired_diff(r["rows"], runs[ref]["rows"])
        b, m = r["btrm"], r["metrics"]
        if r["grm"] and r["grm"].get("overall_closeness") is not None:
            grm = f1(r["grm"]["overall_closeness"])
        elif name.startswith("moss_base"):
            grm = "n/a (GRM skipped)"
        else:
            grm = "missing"
        wins = b.get("tts_wins")
        lines.append(
            f"| {RUN_LABEL.get(name, name)} | {len(r['rows'])} | {ci(p)} | "
            f"{ci(d, True) if d else ('ref' if name == ref else '—')} | "
            f"{f1(b.get('human_parity_score'))} | {f1(b.get('ratio_score'))} | "
            f"{wins if wins is not None else '—'}/{b.get('n_sentences', '—')} | {grm} | "
            f"{f3(m.get('tts_wer'))} | {f3(m.get('speaker_sim'))} | {f3(m.get('tts_utmos'))} |")
    lines += ["", "BTRM % = mean TTS reward / mean human reward × 100 over the same sentences "
              "(human = 100). Parity = 200 × mean sigmoid(r_tts − r_human): a different scale, "
              "never divided by the human reward. CIs: 95%, "
              f"{compare.n_boot()} bootstrap draws over sentences.", ""]
    if "sft_final" in runs and ref in runs:
        verdict(lines, runs["sft_final"], runs[ref])

    # per-speaker BTRM %
    speakers = sorted({r["speaker_id"] for run in runs.values() for r in run["rows"]
                       if r["speaker_id"]})
    if speakers:
        names = list(runs)
        lines += ["### Per-speaker BTRM %", "",
                  "| speaker | " + " | ".join(RUN_LABEL.get(n, n) for n in names) + " |",
                  "|---" * (len(names) + 1) + "|"]
        for spk in speakers:
            cells = []
            for n in names:
                rows = [r for r in runs[n]["rows"] if r["speaker_id"] == spk]
                cells.append(f1(sftlib.btrm_pct(rows)) if rows else "—")
            lines.append(f"| {spk} | " + " | ".join(cells) + " |")
        lines.append("")


def verdict(lines, final, base):
    """PLAN §1 success criteria."""
    d = compare.paired_diff(final["rows"], base["rows"])
    fm, bm = final["metrics"], base["metrics"]
    checks = []
    if d:
        checks.append(("BTRM % higher, CI excludes 0", d["lo"] > 0))
    for key, label, ok in (("tts_wer", "WER ≤ base + 0.01", lambda a, b: a <= b + 0.01 + 1e-9),
                           ("speaker_sim", "speaker sim ≥ base − 0.02", lambda a, b: a >= b - 0.02 - 1e-9),
                           ("tts_utmos", "UTMOS ≥ base − 0.10", lambda a, b: a >= b - 0.10 - 1e-9)):
        if isinstance(fm.get(key), (int, float)) and isinstance(bm.get(key), (int, float)):
            checks.append((label, ok(fm[key], bm[key])))
        else:
            checks.append((label + " (metric missing)", None))
    success = bool(checks) and all(c[1] for c in checks)
    lines += [f"**Success criteria (PLAN §1): {'MET' if success else 'NOT MET'}.** " +
              "; ".join(f"{label}: {'yes' if v else ('no' if v is False else 'unknown')}"
                        for label, v in checks) + ".", ""]


# ---- pilot, rounds, weaknesses, data, teacher, time ---------------------------------------

def pilot_section(lines, notes):
    recipe = sftlib.read_json(sftlib.RUNS / "pilot" / "recipe.json")
    lines += ["## Pilot (test round)", ""]
    if not recipe:
        notes.append("pilot/recipe.json missing: pilot section empty")
        lines += ["No pilot results (`runs/sft/pilot/recipe.json` missing).", ""]
        return recipe
    ev = recipe.get("evidence", {})
    best = recipe.get("pilot_best") or {}
    lines += [f"Status **{recipe.get('status')}**; decoding {recipe.get('decoding')}; recipe "
              f"**{recipe.get('recipe')}**; freeform tag {recipe.get('freeform_tag')}; F1 include "
              f"{recipe.get('f1_include')}; EARS in mix {recipe.get('ears_in_mix')}; teacher "
              f"{recipe.get('teacher_gpu_hours_per_round')} GPU-h/round, oversample "
              f"{recipe.get('teacher_oversample')}; pilot best "
              f"{best.get('arm', 'none')} ({f1(best.get('btrm_pct'))}).", "",
              f"Dev sets (unseen speakers), base = `{(ev.get('base') or {}).get('eval', '?')}`.", "",
              "| arm | mirror BTRM % [CI] | Δ vs base | fresh BTRM % [CI] | Δ vs base | guards |",
              "|---|---|---|---|---|---|"]
    base_sets = (ev.get("base") or {}).get("sets") or {}
    if base_sets:
        lines.append(f"| base | {ci(base_sets.get('mirror'))} | — | {ci(base_sets.get('fresh'))} "
                     "| — | — |")
    for arm, a in (ev.get("arms") or {}).items():
        sets, vs = a.get("sets") or {}, a.get("vs_base") or {}
        g = a.get("guards") or {}
        lines.append(f"| {arm}{' (not selectable)' if arm == 'F' else ''} | "
                     f"{ci(sets.get('mirror'))} | {ci(vs.get('mirror'), True)} | "
                     f"{ci(sets.get('fresh'))} | {ci(vs.get('fresh'), True)} | "
                     f"{'pass' if g.get('pass') else 'FAIL'} |")
    for arm in ev.get("missing_arms") or []:
        lines.append(f"| {arm} | missing | — | — | — | — |")
    d1, d2 = (ev.get("delta1") or {}), (ev.get("delta2") or {})
    f1e = ev.get("f1") or {}
    lines += ["", f"- Δ1 teacher vs EARS hour for hour, diff(B − A): {ci(d1.get('mirror'), True)}",
              f"- Δ2 teacher on top of EARS, diff(D − CU): {ci(d2.get('mirror'), True)}",
              f"- freeform tag, diff(CT − CU): {ci((ev.get('tag') or {}).get('mirror'), True)}",
              f"- F1, diff(F − CU): {ci(f1e.get('mirror'), True)}; on the "
              f"{f1e.get('n_probe_items', '—')} leak-probe items {ci(f1e.get('probe_items'), True)}, "
              f"on the others {ci(f1e.get('other_items'), True)} (memorisation estimate)", "",
              "Full table and reasoning: `runs/sft/pilot/pilot_report.md`.", ""]
    return recipe


def rounds_section(lines, notes):
    paths = sorted(sftlib.RUNS.glob("main_r*/round_summary.json"), key=lambda p: stage_key(p.parent.name))
    lines += ["## Main rounds", ""]
    if not paths:
        notes.append("no main_r*/round_summary.json: rounds section empty")
        lines += ["No main-round results.", ""]
    else:
        lines += ["| round | best eval | mirror BTRM % [CI] | fresh BTRM % [CI] | vs previous best "
                  "[paired CI] | confirmed weaknesses | teacher yield | early stop |",
                  "|---|---|---|---|---|---|---|---|"]
        for p in paths:
            s = sftlib.read_json(p, {}) or {}
            es = sftlib.read_json(sftlib.RUNS / "eval" / str(s.get("best_eval")) / "eval_summary.json",
                                  {}) or {}
            sets = es.get("sets") or {}
            mirror = sets.get("mirror") or ({"btrm_pct": s.get("btrm_pct"), "lo": None}
                                            if s.get("btrm_pct") is not None else None)
            mtxt = ci(mirror) if mirror and mirror.get("lo") is not None else f1(s.get("btrm_pct"))
            vp = s.get("vs_prev")
            if isinstance(vp, dict) and vp.get("diff") is not None:
                vtxt = f"{ci(vp, True)} vs {vp.get('prev_eval', '?')}"
            else:
                vtxt = f1(vp, True) if isinstance(vp, (int, float)) else "—"
            ty = s.get("teacher_yield")
            if ty is None and isinstance(s.get("teacher"), dict):
                ty = s["teacher"].get("yield")
            lines.append(f"| {s.get('round', p.parent.name)} | {s.get('best_eval', '—')} | {mtxt} | "
                         f"{ci(sets.get('fresh'))} | {vtxt} | "
                         f"{', '.join(s.get('confirmed') or []) or '—'} | {f3(ty)} | "
                         f"{s.get('early_stop', '—')} |")
        lines.append("")
        lines += ["Source re-weighting check (alt arm: the smaller source's share of the mix raised, "
                  "same size and learning rate; `runs/sft/main/mix_state.json`):", ""]
        for p in paths:
            s = sftlib.read_json(p, {}) or {}
            rw = s.get("reweight") or {}
            if not rw:
                text = "—"
            elif not rw.get("tested"):
                text = f"not tested ({rw.get('reason', '?')})"
            else:
                d = rw.get("diff")
                text = (f"{rw.get('kind')}: teacher share {f3(rw.get('main_teacher_share'))} → "
                        f"{f3(rw.get('alt_teacher_share'))}; {rw.get('why', '')}"
                        + (f", alt − main {ci(d, True)}" if isinstance(d, dict) else "")
                        + (" — **adopted** for later rounds" if rw.get("adopted") else " — not adopted"))
            lines.append(f"- {s.get('stage', p.parent.name)}: {text}")
        lines.append("")
    best = sftlib.read_json(sftlib.RUNS / "main" / "best.json")
    if best:
        lines += [f"Overall best checkpoint: `{best.get('ckpt')}` (eval {best.get('eval')}, stage "
                  f"{best.get('stage')}, dev-mirror BTRM {f1(best.get('btrm_pct'))}).", ""]


def weakness_section(lines):
    reports = {}
    for p in sftlib.RUNS.glob("*/diag/weakness_report.json"):
        reports[p.parent.parent.name] = sftlib.read_json(p, {}) or {}
    lines += ["## Weakness shares across stages (GRM diagnosis, dev-mirror seed 0)", ""]
    if not reports:
        lines += ["No weakness reports.", ""]
        return
    stages = sorted(reports, key=stage_key)
    lines += ["| category | " + " | ".join(f"{s} ({reports[s].get('eval', '?')})" for s in stages) + " |",
              "|---" * (len(stages) + 1) + "|"]
    for cat in sftlib.WEAKNESS_TAXONOMY:
        cells, any_val = [], False
        for s in stages:
            c = (reports[s].get("categories") or {}).get(cat)
            if c and c.get("share") is not None:
                any_val = any_val or c.get("count", 0) > 0
                mark = " ✓" if cat in (reports[s].get("confirmed") or []) else ""
                cells.append(f"{c['share']:.2f}{mark}")
            else:
                cells.append("—")
        if any_val:
            lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines += ["| n human-preferred / judgments | " + " | ".join(
        f"{reports[s].get('n_human_preferred', '—')} / {reports[s].get('n_judgments', '—')}"
        for s in stages) + " |", "", "Share = count / judgments where GRM preferred the human "
              "clip; ✓ = confirmed (share ≥ 0.15 and ≥ 8 items).", ""]


def flatten(obj, prefix=""):
    """Numeric / short-string leaves of a JSON object as (dotted key, value)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (int, float, bool)) or (isinstance(obj, str) and len(obj) <= 60):
        out.append((prefix, obj))
    return out


def data_section(lines, max_rows=80):
    paths = sorted((sftlib.DATA).glob("*summary.json"))
    mixes = sorted(sftlib.RUNS.glob("*/*/mix_summary.json"), key=lambda p: stage_key(p.parent.parent.name))
    lines += ["## Data", ""]
    if not paths and not mixes:
        lines += ["No data summaries.", ""]
        return
    for p in paths:
        leaves = flatten(sftlib.read_json(p, {}))
        lines += [f"### `{p.name}`", "", "| key | value |", "|---|---|"]
        for k, v in leaves[:max_rows]:
            lines.append(f"| {k} | {f3(v) if isinstance(v, float) else v} |")
        if len(leaves) > max_rows:
            lines.append(f"| … | {len(leaves) - max_rows} more keys in the file |")
        lines.append("")
    if mixes:
        lines += ["### Training mixes", "", "| stage / arm | records | hours |", "|---|---|---|"]
        for p in mixes:
            m = sftlib.read_json(p, {}) or {}
            rec, hrs = m.get("records"), m.get("hours")
            rec = rec.get("total", "—") if isinstance(rec, dict) else rec
            hrs = hrs.get("total", "—") if isinstance(hrs, dict) else hrs
            lines.append(f"| {p.parent.parent.name} / {p.parent.name} | "
                         f"{rec if rec is not None else '—'} | "
                         f"{f1(hrs) if isinstance(hrs, (int, float)) else (hrs or '—')} |")
        lines.append("")


def teacher_section(lines):
    paths = sorted(sftlib.RUNS.glob("*/teacher/filter_summary.json"),
                   key=lambda p: stage_key(p.parent.parent.name))
    lines += ["## Teacher (Orpheus 3B) yields", ""]
    if not paths:
        lines += ["No teacher rounds.", ""]
        return
    lines += ["| stage | generated | accepted | yield | rejected (eos / rate / wer / sim / btrm) | "
              "accepted h | GPU-h | audio s per GPU-s |", "|---|---|---|---|---|---|---|---|"]
    for p in paths:
        t = sftlib.read_json(p, {}) or {}
        rej = t.get("rejected") or {}
        rtxt = " / ".join(str(rej.get(k, "—")) for k in ("no_eos", "rate", "wer", "sim", "btrm"))
        acc_h = (t.get("accepted_audio_s") or 0) / 3600
        gpu_h = (t.get("gen_gpu_s") or 0) / 3600
        lines.append(f"| {p.parent.parent.name} | {t.get('n_generated', '—')} | "
                     f"{t.get('n_accepted', '—')} | {f3(t.get('yield'))} | {rtxt} | {f1(acc_h)} | "
                     f"{f1(gpu_h)} | {f3(t.get('accepted_audio_per_gpu_s'))} |")
    lines.append("")


def parse_at(s):
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return None


def time_section(lines):
    stages = (sftlib.read_json(sftlib.RUNS / "timings.json", {}) or {}).get("stages") or []
    lines += ["## Worker time and cost", ""]
    if not stages:
        lines += ["No `runs/sft/timings.json`.", ""]
        return
    total_s = sum(s.get("seconds") or 0 for s in stages)
    ends = [(parse_at(s.get("at")), s.get("seconds") or 0) for s in stages]
    ends = [(t, sec) for t, sec in ends if t is not None]
    span_h = (max(t for t, _ in ends) - min(t - sec for t, sec in ends)) / 3600 if ends else None
    by_prefix = {}
    for s in stages:
        key = re.split(r"_\d", str(s.get("stage", "?")), maxsplit=1)[0]
        by_prefix[key] = by_prefix.get(key, 0) + (s.get("seconds") or 0)
    lines += [f"- Wall span of logged stages: **{f1(span_h)} h** → **${f1((span_h or 0) * USD_PER_HOUR)}** "
              f"at ${USD_PER_HOUR}/h (g6e.xlarge). Idle worker time outside logged stages is not "
              "included.",
              f"- Sum of logged stage times: {f1(total_s / 3600)} h (may double-count nested "
              "stages that log their own timings).", "",
              "| stage group | hours |", "|---|---|"]
    for k, v in sorted(by_prefix.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {f1(v / 3600)} |")
    lines.append("")


def disclosure(lines, recipe):
    if recipe and "f1_include" in recipe:
        if recipe["f1_include"]:
            f1_text = ("**F1 = included**: the 26 benchmark speakers' *styled* readings (fast, slow, "
                       "loud, whisper, high/low pitch) of 9 of their own benchmark texts (1,404 "
                       "files) are in training. These are the same voice saying the same test "
                       "sentence in another style, so the benchmark score on those 9 texts "
                       "includes a memorisation effect (estimated on the dev-leak probe in the "
                       "pilot section).")
        else:
            f1_text = ("F1 = excluded: the benchmark speakers' styled readings of their own "
                       "benchmark texts (1,404 files) are not in training.")
        f1_text += f" (F1_POLICY {((recipe.get('evidence') or {}).get('f1') or {}).get('policy', '?')})"
    else:
        f1_text = "F1 setting: unknown (`runs/sft/pilot/recipe.json` missing)."
    lines += [
        "## Disclosure", "",
        "- **Seen texts:** other speakers' readings of the 19 benchmark texts are in the training "
        "data (decision D5).",
        "- **Seen speakers:** the 26 benchmark speakers' non-benchmark audio (freeform, emotional, "
        "styled, their `rainbow_01_regular` prompt clip) is in the training data (D6). The 494 "
        "benchmark recordings themselves are not.",
        f"- {f1_text}",
        "- The benchmark result is therefore an **in-domain** result (seen texts, seen speakers), "
        "not held-out generalisation. Dev-fresh (unseen speakers, unseen text) is the one clean "
        "generalisation number.",
        "- **SpeechJudge-BTRM was used in the loop:** to filter teacher data (r_teacher > r_student) "
        "and to choose checkpoints and pilot decisions on the dev sets (never on the 494). "
        "The headline metric is the same judge, so part of any gain may be judge-specific.",
        "- **Licences:** SpeechJudge (models and data) and EARS are CC BY-NC 4.0 (non-commercial).",
        "- **BTRM is not human preference.** It is an automatic reward model; SpeechJudge reports it "
        "near chance on real human vs TTS pairs. No human listening test is reported here.", ""]


def build(out_path):
    notes = []
    dec = sftlib.read_json(sftlib.RUNS / "decoding.json") or {}
    mode = dec.get("mode", "chunked")
    lines = ["# VoiceGym SFT: MOSS-TTS-Nano-100M report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Decoding: **{mode}** "
             f"(voice_clone_max_text_tokens {dec.get('voice_clone_max_text_tokens', '?')})"
             + ("" if dec else " — `decoding.json` missing, assumed chunked") + ".", ""]
    headline(lines, notes, mode)
    recipe = pilot_section(lines, notes)
    rounds_section(lines, notes)
    weakness_section(lines)
    data_section(lines)
    teacher_section(lines)
    time_section(lines)
    disclosure(lines, recipe)
    if (sftlib.RUNS / "DECISIONS.md").exists():
        lines += ["Every decision with its numbers: `runs/sft/DECISIONS.md`.", ""]
    if notes:
        lines += ["## Missing inputs", ""] + [f"- {n}" for n in notes] + [""]
    text = "\n".join(lines)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out_path)
    return text, notes


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", default="", help="override runs/sft (494 runs are read from its parent)")
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.runs_dir:
        sftlib.RUNS = Path(args.runs_dir)
    out = args.out or sftlib.RUNS / "REPORT.md"
    text, notes = build(out)
    print(text, flush=True)
    print(f"-> {out} ({len(notes)} missing inputs noted)", flush=True)


if __name__ == "__main__":
    main()
