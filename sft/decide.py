"""Turn pilot measurements into decisions (INTERFACES §2.8, §4).

  decoding --chunked DIR --whole DIR
      Q1. Base MOSS on dev-mirror under both decodings. `whole` iff its mirror BTRM % is higher
      AND its mean WER <= chunked's + 0.005; else `chunked`. -> runs/sft/decoding.json

  pilot
      Reads runs/sft/eval/pilot_{A,B,CU,CT,D,F}, the base eval runs/sft/eval/base_<mode> (mode
      from decoding.json), runs/sft/pilot/teacher/filter_summary.json, the arms' mix_summary.json
      and runs/sft/data/ears_inventory.jsonl, then applies §4 exactly (all comparisons on
      dev-mirror; dev-fresh is logged alongside):
        Q2 freeform tag   T iff diff(CT - CU) CI lo > 0, else U
        Q3 Δ1 = diff(B - A)   Q4 Δ2 = diff(D - CU)
        recipe            Δ1 lo > 0 -> T (EARS kept iff pct(D) >= pct(B));
                          else Δ1 CI contains 0 or Δ2 lo > 0 -> M; else H
        Q5 F1             F1_POLICY auto / include / exclude. auto includes iff the memorisation
                          effect = diff(F - CU) on the dev-leak-probe items minus diff(F - CU) on the
                          other mirror items has CI lo > 0 (F also heard the dev voices, which raises
                          every dev item and would not carry over to the benchmark)
        Δ1 is ignored when the pilot accepted < 1 h of teacher audio (MIN_TEACHER_AUDIO_S)
        go / stop         stop iff no non-F arm beats base (diff CI lo > 0) on dev-mirror
        best checkpoint   max mirror BTRM % among non-F arms whose guards pass
      A decision whose inputs are missing falls back to its safe default and says so.
      -> runs/sft/pilot/recipe.json, runs/sft/pilot/pilot_report.md, runs/sft/DECISIONS.md.
      Exit 3 on stop (recipe.json and the report are written first). Exit 2 on a bad F1_POLICY.

Env: F1_POLICY (auto | include | exclude), SFT_BOOTSTRAP_N (tests only).
--runs-dir redirects runs/sft (tests). Standard library only.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402
import compare  # noqa: E402

ARMS = ("A", "B", "CU", "CT", "D", "F")
NON_F = ("A", "B", "CU", "CT", "D")
TEACHER_GPU_HOURS = {"H": 4, "M": 10, "T": 20}
PILOT_LR = 2e-5
MIN_TEACHER_AUDIO_S = 3600      # Δ1 (teacher vs EARS per hour) needs >= 1 h of accepted teacher audio
MAIN_ROUNDS = 2
D_OVERSAMPLE = 3.0          # arm D's teacher oversampling (pilot.sh)
M_TEACHER_SHARE = 0.35      # PLAN §7.2 recipe M: teacher ~30-40% of samples
M_OVERSAMPLE_RANGE = (1.0, 5.0)
WER_WHOLE_TOL = 0.005
EXPECTED_PROBE_ITEMS = 36   # 9 styled benchmark texts x 4 dev speakers
EPS = 1e-9


def rel(path):
    """Repo-relative posix path when possible (recipe paths are read from the repo root)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(sftlib.REPO).as_posix()
    except ValueError:
        return path.as_posix()


def f1(v, signed=False):
    if v is None:
        return "—"
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def ci(d, signed=True):
    return compare.fmt_ci(d, signed)


def mirror_pct(summary):
    return ((summary or {}).get("sets", {}).get("mirror") or {}).get("btrm_pct")


# ---- Q1: decoding -------------------------------------------------------------------------

def decide_decoding(chunked_dir, whole_dir):
    ch, wh = compare.load_eval(chunked_dir), compare.load_eval(whole_dir)
    rc, rw = compare.mirror_rows(ch), compare.mirror_rows(wh)
    notes = []
    evidence = {"chunked_eval": rel(chunked_dir), "whole_eval": rel(whole_dir)}
    whole = False
    if not rc or not rw:
        notes.append(f"missing scored rows (chunked {len(rc)}, whole {len(rw)}): "
                     "safe default chunked")
    else:
        common = {r["sentence_id"] for r in rc} & {r["sentence_id"] for r in rw}
        rc = [r for r in rc if r["sentence_id"] in common]
        rw = [r for r in rw if r["sentence_id"] in common]
        pc, pw = compare.pct(rc), compare.pct(rw)
        mc, mw = compare.metric_means(rc), compare.metric_means(rw)
        d = compare.paired_diff(rw, rc)
        evidence.update({"chunked": {**pc, **mc}, "whole": {**pw, **mw},
                         "whole_minus_chunked": d, "n_rows_compared": len(common)})
        better = pw["pct"] > pc["pct"]
        if mc["wer"] is None or mw["wer"] is None:
            notes.append("WER missing: cannot check the WER condition, safe default chunked")
            wer_ok = False
        else:
            wer_ok = mw["wer"] <= mc["wer"] + WER_WHOLE_TOL + EPS
        whole = better and wer_ok
        evidence["rule"] = (f"whole iff BTRM% whole > chunked ({f1(pw['pct'])} vs "
                            f"{f1(pc['pct'])}: {better}) and WER whole <= chunked + "
                            f"{WER_WHOLE_TOL} ({f3(mw['wer'])} vs {f3(mc['wer'])}: {wer_ok})")
    evidence["notes"] = notes
    mode = "whole" if whole else "chunked"
    return {"mode": mode, "voice_clone_max_text_tokens": 100 if whole else 50,
            "train_max_text_tokens": 75 if whole else 50, "evidence": evidence}


def cmd_decoding(args):
    out = decide_decoding(args.chunked, args.whole)
    sftlib.write_json(sftlib.RUNS / "decoding.json", out)
    ev = out["evidence"]
    lines = [f"**Decoding: {out['mode']}** (voice_clone_max_text_tokens "
             f"{out['voice_clone_max_text_tokens']}, train_max_text_tokens "
             f"{out['train_max_text_tokens']}).", ""]
    if "chunked" in ev:
        lines += [
            "| decoding | BTRM % [95% CI] | WER | sim | UTMOS |", "|---|---|---|---|---|",
            f"| chunked (50) | {ci(ev['chunked'], False)} | {f3(ev['chunked']['wer'])} | "
            f"{f3(ev['chunked']['sim'])} | {f3(ev['chunked']['utmos'])} |",
            f"| whole (100) | {ci(ev['whole'], False)} | {f3(ev['whole']['wer'])} | "
            f"{f3(ev['whole']['sim'])} | {f3(ev['whole']['utmos'])} |", "",
            f"Paired diff whole − chunked: {ci(ev['whole_minus_chunked'])} "
            f"({ev['n_rows_compared']} rows, base MOSS on dev-mirror).", "",
            f"Rule (INTERFACES §4 Q1): {ev['rule']}."]
    for n in ev["notes"]:
        lines.append(f"- NOTE: {n}")
    sftlib.log_decision("Q1 decoding", "\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"-> {sftlib.RUNS / 'decoding.json'}", flush=True)
    return 0


# ---- pilot helpers ------------------------------------------------------------------------

def regular_name(name, style):
    """'rainbow_02_fast' -> 'rainbow_02_regular' (the dev-mirror reading of the same text)."""
    if not style:
        _, style, _ = sftlib.classify_ears(name)
    if style and name.endswith("_" + style):
        return name[: -len(style)] + "regular"
    return re.sub(r"_[a-z]+$", "_regular", name)


def probe_items(ev, notes):
    """Dev-mirror item ids whose text the dev-leak-probe records (styled readings) also cover."""
    inv_path = sftlib.RUNS / "data" / "ears_inventory.jsonl"
    probes = [r for r in sftlib.read_jsonl(inv_path) if r.get("dev_leak_probe")]
    rows = compare.mirror_rows(ev)
    items = {r["item"] for r in rows}
    if not probes:
        notes.append(f"no dev_leak_probe records in {rel(inv_path)}: memorisation split skipped")
        return set()
    by_id = {f"{r['speaker']}_{regular_name(r['name'], r.get('style'))}" for r in probes}
    hits = by_id & items
    how = "item_id = <spk>_<regular name>"
    if not hits:   # dev_sets.py may name items differently: match (speaker, text) instead
        texts = {(r["speaker"], " ".join(sftlib.normalize_words(r.get("transcript") or "")))
                 for r in probes if r.get("transcript")}
        hits = {r["item"] for r in rows
                if (r["speaker_id"], " ".join(sftlib.normalize_words(r.get("text") or "")))
                in texts}
        how = "(speaker, normalised text) match (item ids did not match <spk>_<name>)"
    if len(hits) != EXPECTED_PROBE_ITEMS:
        notes.append(f"memorisation split: {len(hits)} probe items found via {how} "
                     f"(expected {EXPECTED_PROBE_ITEMS} = 9 styled texts x 4 dev speakers)")
    return hits


def by_source_count(ms, key):
    """Records of one source ("ears" / "teacher", copies included) in a make_mix mix_summary.json."""
    if not isinstance(ms, dict):
        return None
    for field in ("records_by_source", "by_source"):
        v = ms.get(field)
        if isinstance(v, dict) and key in v:
            x = v[key]
            if isinstance(x, dict):
                x = x.get("records", x.get("n", x.get("count")))
            if isinstance(x, (int, float)):
                return float(x)
    return None


def total_records(ms):
    if not isinstance(ms, dict):
        return None
    x = ms.get("records", ms.get("n_records"))
    if isinstance(x, dict):
        x = x.get("total", x.get("n"))
    return float(x) if isinstance(x, (int, float)) else None


def m_oversample(teacher, notes):
    """Recipe M: oversample teacher so it is ~35% of samples after the first main round."""
    d_mix = sftlib.read_json(sftlib.RUNS / "pilot" / "D" / "mix_summary.json")
    cu_mix = sftlib.read_json(sftlib.RUNS / "pilot" / "CU" / "mix_summary.json")
    n_teacher = by_source_count(d_mix, "teacher")
    if n_teacher is not None:
        n_teacher /= D_OVERSAMPLE
    elif teacher and teacher.get("n_accepted"):
        n_teacher = float(teacher["n_accepted"])
    n_ears = by_source_count(d_mix, "ears") or total_records(cu_mix)
    if not n_teacher or not n_ears:
        notes.append(f"teacher_oversample: record counts unavailable, using arm D's "
                     f"{D_OVERSAMPLE}")
        return D_OVERSAMPLE, None
    pilot_h = (teacher or {}).get("gen_gpu_s")
    pilot_h = pilot_h / 3600 if pilot_h else 5.0
    projected = n_teacher * (1 + TEACHER_GPU_HOURS["M"] / pilot_h)   # pilot + main round 1
    raw = M_TEACHER_SHARE / (1 - M_TEACHER_SHARE) * n_ears / projected
    lo, hi = M_OVERSAMPLE_RANGE
    value = round(min(max(raw, lo), hi), 2)
    calc = {"n_ears_records": n_ears, "n_teacher_pilot": n_teacher, "pilot_teacher_gpu_h": pilot_h,
            "projected_teacher_after_round1": projected, "target_share": M_TEACHER_SHARE,
            "raw": raw, "clamped_to": [lo, hi]}
    return value, calc


# ---- pilot --------------------------------------------------------------------------------

def run_pilot():
    runs = sftlib.RUNS
    policy = (os.environ.get("F1_POLICY") or "auto").strip().lower()
    if policy not in ("auto", "include", "exclude"):
        print(f"ERROR: F1_POLICY must be auto | include | exclude, got {policy!r}", flush=True)
        return 2
    notes = []
    dec = sftlib.read_json(runs / "decoding.json")
    if dec and dec.get("mode") in ("chunked", "whole"):
        mode = dec["mode"]
    else:
        mode = "chunked"
        notes.append("decoding.json missing: assuming chunked")
    base_name = f"base_{mode}"
    base = compare.load_eval(runs / "eval" / base_name)
    if base is None or not compare.mirror_rows(base):
        base = None
        notes.append(f"base eval {base_name} missing: no arm can be compared with base")
    base_summary = compare.summary_of(base) if base else None

    evs, summaries, missing = {}, {}, []
    for arm in ARMS:
        ev = compare.load_eval(runs / "eval" / f"pilot_{arm}")
        if ev is None or not compare.mirror_rows(ev):
            missing.append(arm)
            continue
        evs[arm] = ev
        summaries[arm] = compare.summary_of(ev, base)
        print(f"pilot_{arm}: mirror BTRM {f1(mirror_pct(summaries[arm]))}", flush=True)
    if missing:
        notes.append(f"arms without dev-mirror results: {', '.join(missing)}")
    teacher = sftlib.read_json(runs / "pilot" / "teacher" / "filter_summary.json")
    if teacher is None:
        notes.append("pilot/teacher/filter_summary.json missing")

    def d(a, b, set_name="mirror", items=None):
        if a not in evs or b not in evs:
            return None
        if set_name == "mirror":
            ra, rb = compare.mirror_rows(evs[a]), compare.mirror_rows(evs[b])
            if items is not None:
                ra = [r for r in ra if r["item"] in items]
                rb = [r for r in rb if r["item"] in items]
        else:
            ra, rb = (compare.set_rows(evs[x], set_name, items) for x in (a, b))
        return compare.paired_diff(ra, rb)

    decisions = {}   # title -> markdown text for DECISIONS.md

    # Q2 freeform tag
    tag_d, tag_fresh = d("CT", "CU"), d("CT", "CU", "fresh")
    if tag_d is None:
        tag = "U"
        tag_why = "CT or CU missing: safe default U (freeform instruction unset)"
    else:
        tag = "T" if tag_d["lo"] > 0 else "U"
        tag_why = (f"diff(CT − CU) on dev-mirror = {ci(tag_d)}; CI lo "
                   f"{'> 0 → T' if tag == 'T' else '≤ 0 → U (simpler)'}")
    decisions["Q2 freeform tag"] = (
        f"**freeform_tag = {tag}.** {tag_why}. dev-fresh diff(CT − CU) = {ci(tag_fresh)} "
        "(secondary). Rule: INTERFACES §4 Q2 (T only if diff CI lo > 0; this replaces PLAN §8's "
        "'CT lo above CU mean').")

    # Q3 / Q4 / recipe. Zero teacher yield (voice filter 0.88 vs Orpheus ~0.85) is possible:
    # pilot.sh then skips A, B and D, and the recipe falls back to H.
    teacher_arms_missing = all(a not in evs for a in ("A", "B", "D"))
    if teacher is not None:
        no_teacher_data = not teacher.get("n_accepted")
    else:   # no filter summary: judge from the files pilot.sh checks
        no_teacher_data = teacher_arms_missing and not any(
            (runs / "pilot" / "teacher" / "codes").glob("*.jsonl")) and not sftlib.read_jsonl(
            runs / "pilot" / "teacher" / "accepted_raw.jsonl")
    if no_teacher_data:
        notes.append("teacher gave no usable data: arms A, B, D absent, recipe H")
    delta1, delta2 = d("B", "A"), d("D", "CU")
    # A per-hour comparison on a few minutes of audio is noise; its CI trivially contains 0 and
    # would pick M (10 teacher GPU-h/round) on no evidence. Require MIN_TEACHER_AUDIO_S first.
    accepted_s = float((teacher or {}).get("accepted_audio_s") or 0)
    if delta1 is not None and accepted_s < MIN_TEACHER_AUDIO_S:
        notes.append(f"Δ1 ignored: only {accepted_s / 60:.0f} min of accepted teacher audio "
                     f"(< {MIN_TEACHER_AUDIO_S / 60:.0f} min) is too little for a per-hour comparison")
        delta1 = None
    delta1_fresh, delta2_fresh = d("B", "A", "fresh"), d("D", "CU", "fresh")
    pct_b, pct_d = mirror_pct(summaries.get("B")), mirror_pct(summaries.get("D"))
    ears_in_mix = True
    if delta1 is not None and delta1["lo"] > 0:
        recipe = "T"
        if pct_b is not None and pct_d is not None:
            ears_in_mix = pct_d >= pct_b
            ears_why = f"pct(D) {f1(pct_d)} {'≥' if ears_in_mix else '<'} pct(B) {f1(pct_b)}"
        else:
            ears_why = "D or B missing: safe default, EARS kept"
        why = f"Δ1 CI lo > 0 → T; EARS in mix: {ears_in_mix} ({ears_why})"
    elif delta1 is not None and delta1["lo"] <= 0 <= delta1["hi"]:
        recipe, why = "M", "Δ1 CI contains 0 → M"
    elif delta2 is not None and delta2["lo"] > 0:
        recipe, why = "M", "Δ2 CI lo > 0 → M"
    else:
        recipe = "H"
        if no_teacher_data:
            why = ("teacher gave no usable data (pilot n_accepted "
                   f"{(teacher or {}).get('n_accepted', '?')}; arms A, B, D not trained) → H. "
                   "The teacher is still tried in the main rounds at the H budget, because the "
                   "student (and so the r_teacher > r_student filter) changes")
        elif delta1 is None and delta2 is None:
            why = "Δ1 and Δ2 both unavailable: safe default H (smallest teacher budget)"
        else:
            why = ("Δ1 CI entirely < 0" if delta1 is not None else "Δ1 unavailable") + \
                  (" and Δ2 CI lo ≤ 0" if delta2 is not None else ", Δ2 unavailable") + " → H"
    if delta1 is None and recipe != "H":
        why += " (Δ1 unavailable)"
    oversample, over_calc = 1.0, None
    if recipe == "M":
        oversample, over_calc = m_oversample(teacher, notes)
    elif recipe == "T" and ears_in_mix:
        # EARS is kept because pct(D) >= pct(B), and D had teacher x3: ship the mix that was tested
        oversample = D_OVERSAMPLE
        over_calc = {"why": "recipe T with EARS kept: arm D's tested teacher oversampling"}
    decisions["Recipe"] = (
        f"**recipe = {recipe}** ({why}); teacher GPU-h per main round "
        f"{TEACHER_GPU_HOURS[recipe]}, teacher_oversample {oversample}, ears_in_mix {ears_in_mix}.\n\n"
        f"- Δ1 = diff(B − A), teacher vs EARS hour for hour, dev-mirror: {ci(delta1)} "
        f"(dev-fresh {ci(delta1_fresh)})\n"
        f"- Δ2 = diff(D − CU), teacher on top of full EARS, dev-mirror: {ci(delta2)} "
        f"(dev-fresh {ci(delta2_fresh)})\n"
        f"- Rule: INTERFACES §4 recipe row."
        + (f"\n- teacher_oversample for M: {json.dumps(over_calc)}" if over_calc else ""))

    # Q5 F1
    items = probe_items(evs.get("F") or evs.get("CU") or base, notes)
    f1_all, f1_fresh = d("F", "CU"), d("F", "CU", "fresh")
    mirror_items = {r["item"] for r in compare.mirror_rows(evs.get("F") or evs.get("CU"))}
    f1_probe = d("F", "CU", items=items) if items else None
    f1_other = d("F", "CU", items=mirror_items - items) if items else None
    if policy == "include":
        f1_include, f1_why = True, "F1_POLICY=include (user override)"
    elif policy == "exclude":
        f1_include, f1_why = False, "F1_POLICY=exclude (user override)"
    elif f1_probe is None or f1_other is None:
        f1_include, f1_why = False, "F1_POLICY=auto but F, CU or the probe items are missing: safe default exclude"
    else:
        # What F1 changes on the benchmark is the memorisation part only: arm F also heard the dev
        # speakers' voices (their styled readings), a gain on *every* dev item that does not carry
        # over, because benchmark-speaker audio is already in every mix. So decide on probe items
        # minus the other items (independent item sets; CI half-widths combined in quadrature).
        mem = f1_probe["diff"] - f1_other["diff"]
        half = ((f1_probe["hi"] - f1_probe["lo"]) ** 2 + (f1_other["hi"] - f1_other["lo"]) ** 2) ** 0.5 / 2
        f1_mem = {"diff": mem, "lo": mem - half, "hi": mem + half}
        f1_include = f1_mem["lo"] > 0
        f1_why = (f"F1_POLICY=auto: memorisation effect (probe − other items) {ci(f1_mem)}, CI lo "
                  f"{'> 0 → include' if f1_include else '≤ 0 → exclude'}. diff(F − CU) on all dev-mirror "
                  f"items {ci(f1_all)} is not used: it includes the dev-voice exposure F had")
    decisions["Q5 F1 (styled readings of own test texts)"] = (
        f"**f1_include = {f1_include}.** {f1_why}.\n\n"
        f"- Memorisation estimate, diff(F − CU) on the {len(items)} dev-mirror items whose text "
        f"the dev-leak-probe records cover: {ci(f1_probe)}\n"
        f"- … on the other {len(mirror_items - items) if items else '—'} mirror items: "
        f"{ci(f1_other)}\n"
        f"- dev-fresh diff(F − CU): {ci(f1_fresh)}\n"
        "- F is excluded from best-checkpoint selection (its dev-mirror score is contaminated by "
        "design).")

    # go / stop and best checkpoint
    beats, table = [], []
    for arm in NON_F:
        s = summaries.get(arm)
        if not s:
            continue
        vs = (s.get("vs_base") or {}).get("mirror")
        if vs is not None and vs["lo"] > 0:
            beats.append(arm)
        table.append(f"{arm} {ci(vs)}")
    if base is None:
        status, go_why = "stop", "base eval missing: cannot show any arm beats base"
    elif not any(a in summaries for a in NON_F):
        status, go_why = "stop", "no non-F arm has results"
    elif beats:
        status, go_why = "go", f"arms beating base (diff CI lo > 0): {', '.join(beats)}"
    else:
        status, go_why = "stop", "no non-F arm beats base with CI lo > 0"
    eligible = [a for a in NON_F if a in summaries and summaries[a]["guards"].get("pass")]
    best = max(eligible, key=lambda a: mirror_pct(summaries[a])) if eligible else None
    pilot_best = None
    if best:
        pilot_best = {"arm": best, "eval": f"pilot_{best}",
                      "ckpt": rel(runs / "ckpt" / "pilot" / best / "checkpoint-last"),
                      "btrm_pct": mirror_pct(summaries[best])}
    failed = [a for a in NON_F if a in summaries and a not in eligible]
    best_text = (f"best = **{best}** (mirror BTRM {f1(mirror_pct(summaries[best]))}) among "
                 f"guard-passing non-F arms {', '.join(eligible)}" if best else
                 "WARNING: no non-F arm passes the guard rails: pilot_best = null")
    if failed:
        best_text += f"; guard failures: {', '.join(failed)}"
    if status == "go" and not best:
        notes.append("go, but no guard-passing arm: the main workflow must not start from a "
                     "pilot checkpoint")
    decisions["Go / stop and best pilot checkpoint"] = (
        f"**status = {status}** ({go_why}). Diffs vs {base_name} on dev-mirror: "
        f"{'; '.join(table) or 'none'}.\n\n{best_text}.")

    arms_ev = {}
    for arm, s in summaries.items():
        arms_ev[arm] = {"sets": s["sets"], "vs_base": s.get("vs_base"), "guards": s["guards"]}
    recipe_json = {
        "status": status, "decoding": mode, "freeform_tag": tag, "f1_include": f1_include,
        "recipe": recipe, "ears_in_mix": ears_in_mix,
        "teacher_gpu_hours_per_round": TEACHER_GPU_HOURS[recipe],
        "teacher_oversample": oversample, "lr": PILOT_LR, "main_rounds": MAIN_ROUNDS,
        "pilot_best": pilot_best,
        "evidence": {
            "delta1": {"mirror": delta1, "fresh": delta1_fresh},
            "delta2": {"mirror": delta2, "fresh": delta2_fresh, "pct_B": pct_b, "pct_D": pct_d},
            "tag": {"mirror": tag_d, "fresh": tag_fresh, "why": tag_why},
            "f1": {"policy": policy, "mirror": f1_all, "fresh": f1_fresh,
                   "probe_items": f1_probe, "other_items": f1_other, "n_probe_items": len(items),
                   "why": f1_why},
            "go": {"beats_base": beats, "why": go_why},
            "teacher_oversample": over_calc, "no_teacher_data": no_teacher_data,
            "base": {"eval": base_name, "sets": base_summary["sets"] if base_summary else None},
            "arms": arms_ev, "missing_arms": missing, "teacher": teacher, "notes": notes,
            "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    sftlib.write_json(runs / "pilot" / "recipe.json", recipe_json)
    report = pilot_report(recipe_json, decisions, summaries, base_summary, base_name)
    (runs / "pilot").mkdir(parents=True, exist_ok=True)
    tmp = runs / "pilot" / "pilot_report.md.part"
    tmp.write_text(report, encoding="utf-8")
    os.replace(tmp, runs / "pilot" / "pilot_report.md")
    if no_teacher_data:
        sftlib.log_decision(
            "Pilot: teacher gave no usable data",
            f"The pilot teacher round accepted {(teacher or {}).get('n_accepted', 0)} of "
            f"{(teacher or {}).get('n_generated', '?')} clips (rejections: "
            f"{json.dumps((teacher or {}).get('rejected'))}), so arms A, B and D were not trained "
            "and Q3/Q4 cannot be measured. Recipe H with 4 teacher GPU-h per main round: the "
            "teacher is still tried, since the student changes each round.")
    for title, text in decisions.items():
        sftlib.log_decision(f"Pilot: {title}", text)
    if notes:
        sftlib.log_decision("Pilot: notes", "\n".join(f"- {n}" for n in notes))
    print(report, flush=True)
    print(f"-> {runs / 'pilot' / 'recipe.json'}", flush=True)
    return 3 if status == "stop" else 0


def pilot_report(recipe, decisions, summaries, base_summary, base_name):
    ev = recipe["evidence"]
    lines = [
        "# Pilot report", "",
        f"Status: **{recipe['status'].upper()}**. Decoding: {recipe['decoding']}. Recipe "
        f"**{recipe['recipe']}**, freeform tag **{recipe['freeform_tag']}**, F1 include "
        f"**{recipe['f1_include']}**, EARS in mix {recipe['ears_in_mix']}, teacher "
        f"{recipe['teacher_gpu_hours_per_round']} GPU-h per round, oversample "
        f"{recipe['teacher_oversample']}.", "",
        f"BTRM % = mean TTS reward / mean human reward × 100 on the same rows (human = 100). "
        f"CIs: 95%, items resampled ({compare.n_boot()} draws), paired for differences. "
        f"Base = `{base_name}`.", "",
        "## Arms vs base", "",
        "| arm | set | BTRM % [CI] | Δ vs base [paired CI] | WER | sim | UTMOS | guards |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if base_summary:
        for s, v in base_summary["sets"].items():
            lines.append(f"| base | {s} | {ci(v, False)} | — | {f3(v['wer'])} | {f3(v['sim'])} | "
                         f"{f3(v['utmos'])} | — |")
    for arm in ARMS:
        s = summaries.get(arm)
        if not s:
            lines.append(f"| {arm} | — | missing | — | — | — | — | — |")
            continue
        g = s["guards"]
        if "wer_ok" in g:
            gtxt = ("pass" if g["pass"] else "FAIL") + "".join(
                f" {k[:-3]}✗" for k in ("wer_ok", "sim_ok", "utmos_ok") if g.get(k) is False) + \
                "".join(f" {k[:-3]}?" for k in g.get("unchecked", []) if k != "pass")
        else:
            gtxt = g.get("note", "—")
        if arm == "F":
            gtxt += " (not selectable)"
        for set_name, v in s["sets"].items():
            vs = (s.get("vs_base") or {}).get(set_name)
            lines.append(f"| {arm} | {set_name} | {ci(v, False)} | {ci(vs)} | {f3(v['wer'])} | "
                         f"{f3(v['sim'])} | {f3(v['utmos'])} | {gtxt} |")
    lines += ["", "Guards (mirror + fresh rows vs base): WER ≤ base + 0.01, sim ≥ base − 0.02, "
              "UTMOS ≥ base − 0.10. `?` = metric missing (unchecked).", ""]
    best = recipe["pilot_best"]
    lines += ["## Pilot best checkpoint", "",
              (f"`{best['ckpt']}` (arm {best['arm']}, mirror BTRM {f1(best['btrm_pct'])})"
               if best else "none (no non-F arm passed the guards)"), ""]
    lines += ["## Decisions", ""]
    for title, text in decisions.items():
        lines += [f"### {title}", "", text, ""]
    t = ev.get("teacher")
    if t:
        rej = t.get("rejected") or {}
        lines += ["## Pilot teacher round", "",
                  f"Generated {t.get('n_generated')}, accepted {t.get('n_accepted')} "
                  f"(yield {f3(t.get('yield'))}); rejected: "
                  + ", ".join(f"{k} {v}" for k, v in rej.items())
                  + f". Accepted audio {f1((t.get('accepted_audio_s') or 0) / 3600)} h from "
                  f"{f1((t.get('gen_gpu_s') or 0) / 3600)} GPU-h.", ""]
    if ev["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in ev["notes"]] + [""]
    return "\n".join(lines)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs-dir", default="", help="override runs/sft (tests)")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    dp = sub.add_parser("decoding", parents=[common])
    dp.add_argument("--chunked", required=True)
    dp.add_argument("--whole", required=True)
    sub.add_parser("pilot", parents=[common])
    args = p.parse_args()
    if args.runs_dir:
        sftlib.RUNS = Path(args.runs_dir)
    if args.cmd == "decoding":
        sys.exit(cmd_decoding(args))
    rc = run_pilot()
    if rc == 3:
        print("STOP: no pilot arm beats base on dev-mirror (see pilot_report.md)", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
