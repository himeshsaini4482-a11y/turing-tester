"""DPO round stage 3: candidates + rewards -> preference pairs (Alg. 1 lines 4-6).

A candidate is invalid (line 5) if it has no audio_end (hit max_new), no audio, WER > reward.wer_max, speaker similarity to its voice prompt < reward.sim_min, or a
duration outside reward.sec_per_word x words, or no BTRM score (score_btrm.py scores only the
candidates that pass every other gate, in prompts with >= 2 of them). Per prompt, among the valid candidates:
y_w = argmax BTRM reward, y_l = argmin (pair_rule best_vs_worst); fewer than 2 valid -> skipped;
r_w - r_l < reward.tie_delta -> a tie, skipped (line 6).

Also the paper's diagnostics, reported and never applied:
  sigma_r^2 = mean over prompts of the within-prompt reward variance (unbiased, valid candidates,
  prompts with >= 2). Round 1 samples from pi_theta_1 = pi_ref, so there it is Thm 3.2's
  sigma_bar_r^2 (with the rollout's decoding; see README section 5.2); later rounds track drift.
  beta* = sqrt(sigma_r^2 / (2 eps)) for a grid of eps (and the user's kl.epsilon / kl.lambda if
  set), and E[KL] = sigma_r^2 / (2 beta^2) at beta.beta0 if set (Eqs. 9, 14). BTRM rewards are
  Bradley-Terry logits, the scale Eq. 1 assumes.

Writes runs/dpo/r<rr>/pairs.jsonl ({"pair_id", "prompt_id", "chosen", "rejected", "r_w", "r_l",
"category"}), pairs_summary.json and beta_diagnostic.md.
Env: STD.

  python dpo/pairs.py --round 1
"""
import argparse
import os
import sys
from statistics import mean, variance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpolib  # noqa: E402

EPS_GRID = [0.5, 1, 2, 5, 10, 20, 50]


def invalid_reason(c, reward, metrics, rw):
    if c["n_frames"] == 0 or not c["decoded"]:
        return "empty_or_undecodable"
    if not c["ended"]:
        return "no_audio_end"
    if metrics is None or metrics.get("tts_wer") is None or metrics.get("speaker_sim") is None:
        return "no_metrics"
    if metrics["tts_wer"] > rw["wer_max"]:
        return "wer"
    if metrics["speaker_sim"] < rw["sim_min"]:
        return "sim"
    words = max(1, len(c["text"].split()))
    lo, hi = rw["sec_per_word"]
    if not lo <= c["seconds"] / words <= hi:
        return "duration"
    if reward is None:      # score_btrm.py skips candidates that cannot be in a pair
        return "unscored"
    return None


def scorable(cands, metrics, rw):
    """cids worth a BTRM call: valid on every gate but the reward, in prompts with >= 2 such."""
    ok = {}
    for c in cands:
        if invalid_reason(c, 0.0, metrics.get(c["cid"]), rw) is None:
            ok.setdefault(c["prompt_id"], []).append(c["cid"])
    return {cid for ids in ok.values() if len(ids) >= 2 for cid in ids}


def side_stats(pairs, cands, metrics):
    by_cid = {c["cid"]: c for c in cands}
    out = {}
    for key, get in (("wer", lambda c: metrics[c["cid"]].get("tts_wer")),
                     ("sim", lambda c: metrics[c["cid"]].get("speaker_sim")),
                     ("utmos", lambda c: metrics[c["cid"]].get("tts_utmos")),
                     ("sec_per_word", lambda c: c["seconds"] / max(1, len(c["text"].split())))):
        w = [get(by_cid[q["chosen"]]) for q in pairs]
        l_ = [get(by_cid[q["rejected"]]) for q in pairs]
        ok = [(a, b) for a, b in zip(w, l_) if a is not None and b is not None]
        if ok:
            out[key] = {"chosen": mean(a for a, _ in ok), "rejected": mean(b for _, b in ok),
                        "diff": mean(a - b for a, b in ok)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--round", type=int, required=True)
    args = p.parse_args()

    h = dpolib.load_hparams()
    rw = dpolib.reward_cfg(h)
    if rw["pair_rule"] != "best_vs_worst":
        sys.exit(f"pair_rule {rw['pair_rule']!r} not implemented")
    rdir = dpolib.round_dir(args.round)
    cands = dpolib.read_jsonl(rdir / "candidates.jsonl")
    rewards = {r["sentence_id"]: r["tts_reward"] for r in dpolib.read_jsonl(rdir / "btrm.jsonl")}
    metrics = {r["sentence_id"]: r for r in dpolib.read_jsonl(rdir / "metrics.jsonl")}
    if not cands:
        sys.exit(f"no candidates in {rdir}")

    reasons, by_prompt = {}, {}
    for c in cands:
        why = invalid_reason(c, rewards.get(c["cid"]), metrics.get(c["cid"]), rw)
        reasons[why or "valid"] = reasons.get(why or "valid", 0) + 1
        by_prompt.setdefault(c["prompt_id"], []).append((c, None if why else rewards[c["cid"]]))

    pairs, ties, too_few, variances, gaps = [], 0, 0, [], []
    for pid, group in sorted(by_prompt.items()):
        valid = [(c, r) for c, r in group if r is not None]
        if len(valid) >= 2:
            variances.append(variance([r for _, r in valid]))
        else:
            too_few += 1
            continue
        w = max(valid, key=lambda cr: cr[1])
        l_ = min(valid, key=lambda cr: cr[1])
        gaps.append(w[1] - l_[1])
        if w[1] - l_[1] < rw["tie_delta"]:
            ties += 1
            continue
        pairs.append({"pair_id": f"r{args.round:02d}_{pid}", "prompt_id": pid, "chosen": w[0]["cid"],
                      "rejected": l_[0]["cid"], "r_w": w[1], "r_l": l_[1], "category": w[0]["category"],
                      "shard_w": w[0]["shard"], "shard_l": l_[0]["shard"],
                      "ended_w": w[0]["ended"], "ended_l": l_[0]["ended"]})
    dpolib.write_jsonl(rdir / "pairs.jsonl", pairs)

    sigma2 = mean(variances) if variances else None
    kl, beta = h.get("kl", {}), h.get("beta", {})
    diag = {
        "sigma_r2": sigma2,
        "sigma_r2_is_pi_ref": args.round == 1,
        "n_prompts_in_sigma": len(variances),
        "beta_star_by_epsilon": {str(e): dpolib.beta_star(sigma2, epsilon=e).get("from_epsilon") for e in EPS_GRID},
        "beta_star_user": dpolib.beta_star(sigma2, epsilon=kl.get("epsilon"), lam=kl.get("lambda")),
        "expected_kl_at_beta0": dpolib.expected_kl(sigma2, beta.get("beta0")),
        "note": "Thm 3.2 second-order estimates in nats per sequence; reported only, never applied.",
    }
    n_prompts = len(by_prompt)
    summary = {
        "round": args.round, "n_candidates": len(cands), "n_prompts": n_prompts, "n_pairs": len(pairs),
        "pair_rate": len(pairs) / max(1, n_prompts), "n_ties": ties, "n_too_few_valid": too_few,
        "candidates_by_status": reasons,
        "mean_reward_valid": mean(r for g in by_prompt.values() for _, r in g if r is not None)
        if reasons.get("valid") else None,
        "mean_gap": mean(gaps) if gaps else None,
        "mean_margin_reward": mean(q["r_w"] - q["r_l"] for q in pairs) if pairs else None,
        "mean_reward_scored": mean(rewards.values()) if rewards else None,
        # chosen vs rejected on the non-BTRM metrics: BTRM preferring longer/slower or less intelligible
        # samples (reward hacking) shows up here before it shows up on the dev set
        "chosen_vs_rejected": side_stats(pairs, cands, metrics),
        "paper": diag,
    }
    dpolib.write_json(rdir / "pairs_summary.json", summary)

    lines = [f"# Round {args.round}: reward variance and the paper's beta*", "",
             f"sigma_r^2 = {sigma2:.4f} (mean within-prompt BTRM variance over {len(variances)} prompts"
             f"{'; round 1 = under pi_ref' if args.round == 1 else ''})" if sigma2 is not None else
             "sigma_r^2: not enough valid candidates", "",
             "| epsilon (target E[KL], nats/sequence) | beta* = sqrt(sigma^2 / 2 eps) |", "|---|---|"]
    lines += [f"| {e} | {v:.4f} |" for e, v in diag["beta_star_by_epsilon"].items() if v is not None]
    if diag["beta_star_user"]:
        lines += ["", f"With your kl settings: {diag['beta_star_user']}"]
    if diag["expected_kl_at_beta0"] is not None:
        lines += ["", f"E[KL] at beta0 = {beta['beta0']}: {diag['expected_kl_at_beta0']:.3f} nats (Eq. 14)"]
    lines += ["", f"pairs {len(pairs)}/{n_prompts} prompts, ties {ties}, too few valid {too_few}; "
                  f"candidates {reasons}", "",
              "These are second-order estimates (Eqs. 11-14). The beta settings are yours: dpo/hparams.json."]
    (rdir / "beta_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
