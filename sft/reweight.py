"""Per-round source re-weighting check for the main rounds (INTERFACES.md §4, "Main rounds").

The pilot fixes a starting mix (recipe H / M / T). Each main round also trains one extra arm,
"alt", whose mix gives the *smaller* source (teacher or EARS) a larger share of the hours, at the
same record count (so the same optimizer steps) and learning rate as the main arm, so only the
share differs. Both arms are
candidates for the round's best checkpoint. If alt's best checkpoint beats the main arm's best
(dev-mirror BTRM %, both passing the guard rails), its teacher share is adopted for the next
rounds' main mix. The teacher GPU-hour budget is never changed here: a clear win (paired CI
above 0) only writes a suggestion to DECISIONS.md.

The adopted state is a teacher *share* of hours, not a repeat factor: the teacher pool grows every
round, so a fixed factor would drift. make_mix.py turns the share into weights each round.

State: runs/sft/main/mix_state.json. decide() records each round there before round_select.py
writes round_summary.json, so a crash between the two never loses a decision. Each record keeps
`state_before`, so a round can be redone by hand.

  python sft/reweight.py config --stage main_r1 --variant main    # mix config JSON on stdout
  python sft/reweight.py config --stage main_r1 --variant alt     # empty stdout = no alt arm
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

TESTED_MAX_OVERSAMPLE = 3.0   # arm D's teacher x3: the highest repeat factor the pilot tested
EARS_IN_SHARE = 0.25          # recipe T without EARS: alt brings EARS in at this share
MIN_SHARE_MOVE = 0.02         # a smaller achievable move is not worth an arm


def state_path():
    return sftlib.RUNS / "main" / "mix_state.json"


def recipe_path():
    return sftlib.RUNS / "pilot" / "recipe.json"


def init_state(recipe):
    oversample = float(recipe.get("teacher_oversample", 1.0))
    return {"teacher_share": None, "ears_in_mix": bool(recipe.get("ears_in_mix", True)),
            "ears_weight": 1.0, "teacher_oversample": oversample,
            "max_oversample": max(TESTED_MAX_OVERSAMPLE, oversample), "decided": {}}


def load_state():
    state = sftlib.read_json(state_path())
    if state is None:
        state = init_state(sftlib.read_json(recipe_path(), {}))
        sftlib.write_json(state_path(), state)
    return state


def teacher_codes():
    return sorted(str(d) for d in sftlib.RUNS.glob("*/teacher/codes") if any(d.glob("*.jsonl")))


def main_config(state, recipe, report):
    """The round's main mix: recipe settings, the round's weakness weights, the adopted share."""
    return {
        "ears": {"use": state["ears_in_mix"], "freeform_tag": recipe["freeform_tag"],
                 "f1_include": bool(recipe["f1_include"]), "leak_probe": False, "match_teacher": None,
                 "category_weights": {c: 2.0 for c in report.get("confirmed", [])},
                 "weight": state["ears_weight"]},
        "teacher": {"codes": teacher_codes(), "oversample": state["teacher_oversample"],
                    "max_oversample": state["max_oversample"]},
        "teacher_share": state["teacher_share"],
        "seed": 0,
    }


def reachable_teacher_share(summary, cap):
    """Highest teacher hour share at the main mix's record count with the teacher repeated <= cap."""
    cfg, recs, hrs = summary["config"], summary["records_by_source"], summary["hours_by_source"]
    over = float(cfg["teacher"]["oversample"]) or 1.0
    weight = float(cfg["ears"].get("weight", 1.0)) or 1.0
    tn, th = recs["teacher"] / over, hrs["teacher"] / over
    en, eh = recs.get("ears", 0) / weight, hrs.get("ears", 0) / weight
    if cap * tn >= summary["records"] or not en:
        return 1.0
    ears_weight = (summary["records"] - cap * tn) / en
    return th * cap / (th * cap + eh * ears_weight)


def alt_share(main_summary, state):
    """-> (kind, target teacher share, why-skipped). kind is teacher_up | ears_up | None."""
    s = main_summary.get("teacher_share")
    hours = main_summary.get("hours_by_source") or {}
    if not s or not hours.get("teacher"):
        return None, None, "the main mix has no teacher data, so there is nothing to re-weight"
    if not main_summary["config"]["ears"]["use"]:
        return "ears_up", 1 - EARS_IN_SHARE, None
    if s <= 0.5:
        target = min(2 * s, s + 0.25)
        # the teacher may not repeat more than max_oversample times at the same record count
        if min(target, reachable_teacher_share(main_summary, state["max_oversample"])) - s < MIN_SHARE_MOVE:
            return None, None, (f"teacher share {s:.3f} cannot grow: the teacher pool is already at "
                                f"the x{state['max_oversample']:g} repeat cap")
        return "teacher_up", round(target, 4), None
    minor = 1 - s
    return "ears_up", round(1 - min(2 * minor, minor + 0.25), 4), None


def alt_config(stage, state, recipe, report):
    """The round's alt mix config, or None (reason in runs/sft/<stage>/alt_skip.json)."""
    summary = sftlib.read_json(sftlib.RUNS / stage / "mix" / "mix_summary.json")
    if summary is None:
        raise SystemExit(f"{stage}: build the main mix before the alt mix")
    kind, target, why = alt_share(summary, state)
    if kind is None:
        sftlib.write_json(sftlib.RUNS / stage / "alt_skip.json", {"reason": why})
        return None
    config = main_config(state, recipe, report)
    config["ears"]["use"] = True
    config["teacher_share"] = target
    config["total_records"] = summary["records"]
    config["reweight"] = {"kind": kind, "main_teacher_share": summary["teacher_share"], "target": target}
    return config


def decide(stage, round_no, candidates, rows):
    """Compare the round's alt arm with its main arm(s); adopt alt's share if it wins.

    candidates: round_select.py's list ({eval, btrm_pct, guards}); rows(eval) -> dev-mirror rows.
    Idempotent per stage. Returns the record stored in the state and in round_summary.json.
    """
    state = load_state()
    if stage in state["decided"]:
        return state["decided"][stage]
    before = {k: state[k] for k in ("teacher_share", "ears_in_mix", "ears_weight")}
    alt = [c for c in candidates if "_alt_" in c["eval"]]
    main = [c for c in candidates if "_alt_" not in c["eval"]]
    if not alt:
        skip = sftlib.read_json(sftlib.RUNS / stage / "alt_skip.json", {})
        result = {"tested": False, "reason": skip.get("reason", "REWEIGHT=0 or no alt arm trained"),
                  "adopted": False}
    else:
        alt_mix = sftlib.read_json(sftlib.RUNS / stage / "alt" / "mix_summary.json", {})
        main_mix = sftlib.read_json(sftlib.RUNS / stage / "mix" / "mix_summary.json", {})
        kind = (sftlib.read_json(sftlib.RUNS / stage / "alt_config.json", {}).get("reweight") or {}).get("kind")
        result = {"tested": True, "kind": kind, "main_teacher_share": main_mix.get("teacher_share"),
                  "alt_teacher_share": alt_mix.get("teacher_share"), "adopted": False, "diff": None,
                  "records": {"main": main_mix.get("records"), "alt": alt_mix.get("records")}}
        best_alt = max((c for c in alt if c["guards"]), key=lambda c: c["btrm_pct"], default=None)
        best_main = max((c for c in main if c["guards"]), key=lambda c: c["btrm_pct"], default=None)
        if best_alt is None:
            result["why"] = "no alt checkpoint passed the guard rails"
        elif best_main is None:
            result.update(adopted=True, why=f"only alt passed the guard rails ({best_alt['eval']})")
        else:
            diff = sftlib.bootstrap_diff(rows(best_alt["eval"]), rows(best_main["eval"]))
            result.update(diff=diff, adopted=diff["diff"] > 0,
                          why=f"{best_alt['eval']} {best_alt['btrm_pct']:.1f}% vs {best_main['eval']} "
                              f"{best_main['btrm_pct']:.1f}%")
        if result["adopted"]:
            state["teacher_share"] = result["alt_teacher_share"]
            state["ears_in_mix"] = True
            if not main_mix.get("config", {}).get("ears", {}).get("use", True):
                state["ears_weight"] = 0.0   # size the mix by the teacher alone, as alt was
    result["state_before"] = before
    state["decided"][stage] = result
    sftlib.write_json(state_path(), state)
    log(stage, round_no, result, state)
    return result


def log(stage, round_no, result, state):
    if not result["tested"]:
        sftlib.log_decision(f"{stage}: source re-weighting", f"Not tested: {result['reason']}.")
        return
    diff = result.get("diff")
    lines = [f"Alt arm ({result['kind']}): teacher share of the mix hours {result['main_teacher_share']} "
             f"(main) → {result['alt_teacher_share']} (alt), same learning rate; records main "
             f"{result['records']['main']} vs alt {result['records']['alt']}.",
             f"{result.get('why', '')}" + (f": diff {diff['diff']:+.2f} pts (95% CI {diff['lo']:+.2f}.."
                                            f"{diff['hi']:+.2f})" if diff else "") + ".",
             ("Adopted: later rounds' main mix uses teacher share "
              f"{state['teacher_share']}." if result["adopted"] else "Not adopted: the main mix stays.")]
    if diff and diff["lo"] > 0:
        more = result["kind"] == "teacher_up"
        lines.append(f"The CI is above 0, so {'more' if more else 'less'} teacher data is clearly better. "
                     f"Suggestion (not applied; the 4-day budget is tight): "
                     f"{'raise' if more else 'lower'} TEACHER_GPU_HOURS for the next main run.")
    sftlib.log_decision(f"{stage}: source re-weighting (round {round_no})", "\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("--stage", required=True)
    c.add_argument("--variant", choices=("main", "alt"), required=True)
    args = p.parse_args()

    recipe = sftlib.read_json(recipe_path())
    if recipe is None:
        raise SystemExit(f"{recipe_path()} is missing; run scripts/sft/pilot.sh first")
    state = load_state()
    report = sftlib.read_json(sftlib.RUNS / args.stage / "diag" / "weakness_report.json", {})
    if args.variant == "main":
        print(json.dumps(main_config(state, recipe, report)))
        return
    config = alt_config(args.stage, state, recipe, report)
    if config is None:
        skip = sftlib.read_json(sftlib.RUNS / args.stage / "alt_skip.json", {})
        print(f"{args.stage}: no alt arm ({skip.get('reason')})", file=sys.stderr, flush=True)
        return
    print(json.dumps(config))


if __name__ == "__main__":
    main()
