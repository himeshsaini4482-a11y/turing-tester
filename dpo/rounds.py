"""DPO round bookkeeping: pick the best policy, decide whether to go on, rank sweeps, write the report.

  select --round R   After round R's dev eval (runs/dpo/eval/rRR/eval_summary.json, compared with
                     r00 = pi_ref). A round counts only if the SFT guard rails pass. The best round is
                     the highest dev-mirror BTRM %. Writes runs/dpo/rRR/round_summary.json and
                     runs/dpo/best.json; exit code 10 = stop (guard rails failed, `patience` rounds
                     without a new best, or no rounds left), 0 = go on.
  sweep              Ranks runs/dpo/sweep/<tag>/ (train.py --tag, eval.sh sweep_<tag>) by the same
                     rule -> runs/dpo/sweep/SWEEP_REPORT.md. The choice stays the user's.
  report             runs/dpo/REPORT.md: per-round training, KL and eval table, best round, final
                     494 benchmark vs pi_ref and the base model when present.

Env: STD.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpolib  # noqa: E402

EXIT_STOP = 10


def eval_of(name):
    return dpolib.read_json(dpolib.RUNS / "eval" / name / "eval_summary.json")


def mirror(ev):
    return (ev or {}).get("sets", {}).get("mirror", {})


def num(v, spec):
    return "—" if v is None else format(v, spec)


def fmt_ci(d, key):
    if not d or d.get(key) is None:
        return "—"
    return f"{d[key]:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]" if key == "diff" else \
        f"{d[key]:.1f} [{d['lo']:.1f}, {d['hi']:.1f}]"


def cmd_select(args):
    h = dpolib.load_hparams()
    r = args.round
    done = dpolib.read_json(dpolib.round_dir(r) / "round_summary.json")
    if done is not None:             # idempotent: a rerun repeats the decision instead of counting twice
        print(f"round {r} already selected: {done['stop'] or 'continue'}", flush=True)
        sys.exit(EXIT_STOP if done["stop"] else 0)
    ev = eval_of(f"r{r:02d}")
    if ev is None:
        sys.exit(f"no eval summary for round {r}")
    best = dpolib.read_json(dpolib.RUNS / "best.json", {"round": 0, "btrm_pct": mirror(eval_of("r00")).get("btrm_pct"),
                                                         "policy": h["models"][h["model"]]["policy_init"],
                                                         "rounds_without_gain": 0})
    pct = mirror(ev).get("btrm_pct")
    guards = dict(ev.get("guards", {}))
    if guards.get("unchecked"):      # compare.py counts a missing metric as passing; here it is a failure
        guards["pass"] = False
        guards["note"] = f"unchecked guard rails count as failed: {guards['unchecked']}"
    train = dpolib.read_json(dpolib.round_dir(r) / "train_summary.json", {})
    gain = guards.get("pass") and pct is not None and (best["btrm_pct"] is None or pct > best["btrm_pct"])
    if gain:
        best = {"round": r, "btrm_pct": pct, "policy": train.get("policy_out"), "rounds_without_gain": 0}
    else:
        best["rounds_without_gain"] = best.get("rounds_without_gain", 0) + 1
    dpolib.write_json(dpolib.RUNS / "best.json", best)
    why = None
    if not guards.get("pass"):
        why = f"guard rails failed in round {r}: {guards}"
    elif best["rounds_without_gain"] >= h["patience"]:
        why = f"{best['rounds_without_gain']} round(s) without a new best"
    elif r >= h["rounds"]:
        why = "all rounds done"
    dpolib.write_json(dpolib.round_dir(r) / "round_summary.json", {
        "round": r, "btrm_pct": pct, "vs_pi_ref": ev.get("vs_base", {}).get("mirror"), "guards": guards,
        "metrics": {k: mirror(ev).get(k) for k in ("wer", "sim", "utmos")},
        "new_best": bool(gain), "best": best, "stop": why})
    dpolib.append_jsonl(dpolib.RUNS / "rounds.jsonl", {"round": r, "btrm_pct": pct, "guards_pass": guards.get("pass"),
                                                        **{k: mirror(ev).get(k) for k in ("wer", "sim", "utmos")}})
    print(f"round {r}: dev BTRM % {pct}, vs pi_ref {fmt_ci(ev.get('vs_base', {}).get('mirror'), 'diff')}, "
          f"guards {'pass' if guards.get('pass') else 'FAIL'}; best = round {best['round']}"
          + (f"; stopping: {why}" if why else ""), flush=True)
    if why:
        dpolib.log_decision(f"stop after round {r}", f"{why}. Best: round {best['round']} ({best['policy']}).")
        sys.exit(EXIT_STOP)


def cmd_sweep(args):
    rows = []
    for d in sorted((dpolib.RUNS / "sweep").glob("*/")):
        tr = dpolib.read_json(d / "train_summary.json")
        ev = eval_of(f"sweep_{d.name}")
        if tr is None:
            continue
        o, b, k = tr["hparams"]["optimizer"], tr["hparams"]["beta"], tr["hparams"]["kl"]
        m = mirror(ev)
        g = (ev or {}).get("guards", {})
        rows.append({"tag": d.name, "optimizer": o["name"], "lr": o["lr"], "beta": f"{b['mode']} {b['beta0']}",
                     "pct": m.get("btrm_pct"), "vs": (ev or {}).get("vs_base", {}).get("mirror"),
                     "guards": (g.get("pass") and not g.get("unchecked")) if ev else None, "acc": tr.get("mean_accuracy"),
                     "wer": m.get("wer"), "sim": m.get("sim"), "utmos": m.get("utmos"), "steps": tr.get("steps"),
                     "seconds": tr.get("seconds"),
                     "d_hat": tr.get("d_hat_end"), "beta_end": tr.get("beta_end"), "violation": tr.get("violation")})
    rows.sort(key=lambda x: (not x["guards"], -(x["pct"] or -1e9)))
    lines = ["# Optimizer / beta sweep (round-1 pairs, dev mirror)", "",
             "Ranked by: SFT guard rails pass, then dev BTRM %. The pick is yours (dpo/hparams.json).", "",
             "| tag | optimizer | lr | beta | dev BTRM % | vs pi_ref | WER | sim | UTMOS | guards | pair acc | D_hat end | "
             "beta end | KL action | steps | train s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        guards = "pass" if x["guards"] else "FAIL" if x["guards"] is False else "—"
        lines.append(f"| {x['tag']} | {x['optimizer']} | {x['lr']} | {x['beta']} | {num(x['pct'], '.1f')} | "
                     f"{fmt_ci(x['vs'], 'diff')} | {num(x['wer'], '.3f')} | {num(x['sim'], '.3f')} | "
                     f"{num(x['utmos'], '.2f')} | {guards} | {num(x['acc'], '.2f')} | {num(x['d_hat'], '.3g')} | "
                     f"{num(x['beta_end'], '.4g')} | {x['violation'] or '—'} | {x['steps']} | {num(x['seconds'], '.0f')} |")
    out = dpolib.RUNS / "sweep" / "SWEEP_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_report(args):
    h = dpolib.load_hparams()
    name = h["model"]
    lines = [f"# DPO report: {name}", "", f"pi_ref: `{h['models'][name]['policy_init']}`", "",
             "| round | pairs | pair rate | sigma_r^2 | steps | beta end | D_hat end | pair acc | KL action | "
             "dev BTRM % | vs pi_ref | WER | sim | UTMOS | guards |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    r0 = mirror(eval_of("r00"))
    if r0:
        lines.append(f"| 0 (pi_ref) | | | | | | | | | {fmt_ci(r0, 'btrm_pct')} | | {num(r0.get('wer'), '.3f')} | "
                     f"{num(r0.get('sim'), '.3f')} | {num(r0.get('utmos'), '.2f')} | |")
    r = 1
    while (dpolib.round_dir(r)).exists():
        ps = dpolib.read_json(dpolib.round_dir(r) / "pairs_summary.json", {})
        tr = dpolib.read_json(dpolib.round_dir(r) / "train_summary.json", {})
        ev = eval_of(f"r{r:02d}") or {}
        s2 = ps.get("paper", {}).get("sigma_r2")
        f = num
        lines.append(f"| {r} | {ps.get('n_pairs', '—')} | {f(ps.get('pair_rate'), '.2f')} | {f(s2, '.3f')} | "
                     f"{tr.get('steps', '—')} | {f(tr.get('beta_end'), '.4g')} | {f(tr.get('d_hat_end'), '.3g')} | "
                     f"{f(tr.get('mean_accuracy'), '.2f')} | {tr.get('violation') or '—'} | "
                     f"{fmt_ci(mirror(ev), 'btrm_pct')} | {fmt_ci(ev.get('vs_base', {}).get('mirror'), 'diff')} | "
                     f"{f(mirror(ev).get('wer'), '.3f')} | {f(mirror(ev).get('sim'), '.3f')} | "
                     f"{f(mirror(ev).get('utmos'), '.2f')} | "
                     f"{'pass' if ev.get('guards', {}).get('pass') and not ev.get('guards', {}).get('unchecked') else '—' if not ev else 'FAIL'} |")
        r += 1
    best = dpolib.read_json(dpolib.RUNS / "best.json")
    if best:
        lines += ["", f"Best: round {best['round']} (`{best['policy']}`), dev BTRM % {best['btrm_pct']}"]
    fin = dpolib.read_json(dpolib.REPO / "runs" / f"dpo_final_{name}" / "btrm_summary.json")
    if fin:
        ref = dpolib.read_json(dpolib.REPO / "runs" / f"dpo_final_{name}_ref" / "btrm_summary.json") or \
            dpolib.read_json(dpolib.REPO / "runs" / "sft_final" / "btrm_summary.json")   # final.sh REF_DONE

        def pct(s):
            return 100 * s["mean_tts_reward"] / s["mean_human_reward"]
        lines += ["", "## Final 494 benchmark (BTRM % = mean TTS reward / mean human reward x 100)", "",
                  f"- DPO best: {pct(fin):.1f}" + (f"; pi_ref: {pct(ref):.1f}" if ref else "")]
    lines += ["", "Automated-judge results (SpeechJudge-BTRM, GRM): not human-listener claims."]
    (dpolib.RUNS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--round", type=int, required=True)
    sub.add_parser("sweep")
    sub.add_parser("report")
    args = p.parse_args()
    {"select": cmd_select, "sweep": cmd_sweep, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
