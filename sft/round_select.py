"""Pick a main round's best checkpoint and decide whether to continue.

Among the round's evals (runs/sft/eval/<name>/eval_summary.json from compare.py), main and alt
arms alike, the best is the highest dev-mirror BTRM % whose guard rails pass. It is compared
with the previous best (paired bootstrap on dev-mirror). The overall best moves only if the new
one has a higher mean; the round is an early stop if it does not. reweight.decide() then settles
the alt arm's source re-weighting (adopt its teacher share for later rounds or not).

Writes runs/sft/<stage>/round_summary.json and (if improved) runs/sft/main/best.json, appends
to runs/sft/DECISIONS.md, and prints "continue" or "stop" as the last line.

  python sft/round_select.py --stage main_r1 --round 1 --evals main_r1_lr2e-5_e1 main_r1_lr2e-5_e2 \
      --ckpts runs/sft/ckpt/main_r1/lr2e-5/checkpoint-epoch-1 runs/sft/ckpt/main_r1/lr2e-5/checkpoint-epoch-2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reweight  # noqa: E402
import sftlib  # noqa: E402

BEST = sftlib.RUNS / "main" / "best.json"


def mirror_rows(eval_name):
    """BTRM rows of the dev-mirror set of one eval dir."""
    path = sftlib.RUNS / "eval" / eval_name
    sets = {r["sentence_id"]: r.get("set", "mirror") for r in sftlib.read_csv(path / "manifest.csv")}
    return [r for r in sftlib.read_jsonl(path / "btrm.jsonl") if sets.get(r["sentence_id"]) == "mirror"]


def current_best():
    best = sftlib.read_json(BEST)
    if best:
        return best
    recipe = sftlib.read_json(sftlib.RUNS / "pilot" / "recipe.json", {})
    pilot = recipe.get("pilot_best")
    if pilot:
        best = {"eval": pilot["eval"], "ckpt": pilot["ckpt"], "btrm_pct": pilot["btrm_pct"], "stage": "pilot"}
    else:
        # The pilot said go, but every arm that beat base failed a guard rail: the main rounds
        # start from the base model instead.
        mode = sftlib.read_json(sftlib.RUNS / "decoding.json", {}).get("mode", "chunked")
        base_eval = f"base_{mode}"
        summary = sftlib.read_json(sftlib.RUNS / "eval" / base_eval / "eval_summary.json")
        if not summary or not os.environ.get("MOSS_MODEL_DIR"):
            raise SystemExit(f"no pilot_best in recipe.json and no usable base eval {base_eval}")
        best = {"eval": base_eval, "ckpt": os.environ["MOSS_MODEL_DIR"],
                "btrm_pct": summary["sets"]["mirror"]["btrm_pct"], "stage": "base"}
        sftlib.log_decision("main: starting point", f"The pilot has no eligible best arm (all failed guard "
                            f"rails), so the main rounds start from the base model ({base_eval}, "
                            f"{best['btrm_pct']:.1f}% dev-mirror BTRM).")
        print("main: no eligible pilot arm; starting from the base model", flush=True)
    sftlib.write_json(BEST, best)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--evals", nargs="+", required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint dir per eval, same order")
    args = p.parse_args()
    if len(args.evals) != len(args.ckpts):
        p.error("--evals and --ckpts must have the same length")

    prev = current_best()
    candidates = []
    for name, ckpt in zip(args.evals, args.ckpts):
        summary = sftlib.read_json(sftlib.RUNS / "eval" / name / "eval_summary.json")
        if not summary:
            print(f"[warn] no eval_summary for {name}", file=sys.stderr)
            continue
        candidates.append({"eval": name, "ckpt": ckpt, "btrm_pct": summary["sets"]["mirror"]["btrm_pct"],
                           "guards": summary.get("guards", {}).get("pass", False)})
    eligible = [c for c in candidates if c["guards"]]
    report = sftlib.read_json(sftlib.RUNS / args.stage / "diag" / "weakness_report.json", {})
    teacher = sftlib.read_json(sftlib.RUNS / args.stage / "teacher" / "filter_summary.json", {})
    summary = {"round": args.round, "stage": args.stage, "candidates": candidates,
               "confirmed": report.get("confirmed", []), "teacher_yield": teacher.get("yield"),
               "prev_best": prev}

    if not eligible:
        summary.update(best_eval=None, best_ckpt=None, btrm_pct=None, vs_prev=None, early_stop=True)
        verdict = "stop"
        why = "no checkpoint of this round passed the guard rails"
    else:
        best = max(eligible, key=lambda c: c["btrm_pct"])
        diff = sftlib.bootstrap_diff(mirror_rows(best["eval"]), mirror_rows(prev["eval"]))
        improved = diff["diff"] > 0
        summary.update(best_eval=best["eval"], best_ckpt=best["ckpt"], btrm_pct=best["btrm_pct"],
                       vs_prev={"prev_eval": prev["eval"], **diff}, early_stop=not improved)
        if improved:
            sftlib.write_json(BEST, {"eval": best["eval"], "ckpt": best["ckpt"], "btrm_pct": best["btrm_pct"],
                                     "stage": args.stage})
        verdict = "continue" if improved else "stop"
        why = (f"best {best['eval']} dev-mirror BTRM {best['btrm_pct']:.1f}% vs previous best {prev['eval']} "
               f"{prev['btrm_pct']:.1f}%: diff {diff['diff']:+.2f} pts (95% CI {diff['lo']:+.2f}..{diff['hi']:+.2f}). "
               + ("New overall best." if improved else "Not better: early stop, the previous best stays."))

    # before round_summary.json: main.sh skips a round that has one, so the decision must be durable first
    summary["reweight"] = reweight.decide(args.stage, args.round, candidates, mirror_rows)
    sftlib.write_json(sftlib.RUNS / args.stage / "round_summary.json", summary)
    table = "\n".join(f"- {c['eval']}: {c['btrm_pct']:.1f}% guards {'pass' if c['guards'] else 'FAIL'}"
                      for c in candidates)
    sftlib.log_decision(f"{args.stage}: checkpoint choice", f"{why}\n\n{table}")
    print(why, flush=True)
    print(verdict)


if __name__ == "__main__":
    main()
