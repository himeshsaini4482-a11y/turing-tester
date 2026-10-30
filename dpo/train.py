"""DPO round stage 4: update the policy on this round's fresh pairs (Alg. 1 lines 8-12).

Loss = Eq. 2 exactly: -log sigmoid(beta (u_w - u_l)), u = log pi_theta(y|x) - log pi_ref(y|x),
with pi_ref = hparams models.<model>.policy_init, frozen for the whole run (its log-probs are computed once per
round, before training, with the same code and precision as the policy's). Per optimizer step:
  line 9   k_hat = 1/2 (u_w^2 + u_l^2)   (per sequence, or with u divided by frames: kl.k_hat_per)
  line 10  D_hat <- (1 - alpha) D_hat + alpha mean(k_hat)
  line 11  primal step with beta_t (the configured optimizer, clipping, schedule)
  line 12  beta.mode = dual_ascent only: beta <- clip(beta + eta_t (D_hat - eps), beta_min, beta_max),
           eta_t = eta_beta, or eta_beta / sqrt(t) with beta.eta_schedule = "inv_sqrt" (section 4)
beta_t, D_hat and the step count carry across rounds (runs/dpo/state.json), as t runs on in Alg. 1.
On D_hat > kl.epsilon, kl.on_violation decides: none (log only), stop_round (keep the steps so
far, end the round), rollback (discard this round's update; beta/D_hat keep what they learned).

Every value of beta, kl and optimizer is read from dpo/hparams.json and none has a default here.
The script exits with code 3, listing the missing fields, until the user has set them.

A plain PyTorch loop rather than trl's DPOTrainer: MOSS-TTS-Nano has no logits/labels forward,
3-D [T, 17] rows and 17 heads, so DPOTrainer's tokenisation, collator, loss and reference paths
would all be overridden anyway, and Alg. 1 line 12 (adapting beta) has no hook there.

Writes runs/dpo/ckpt/r<rr>/ (eval_ckpt.sh-loadable), runs/dpo/r<rr>/{ref_logps.json,
train_log.jsonl, train_summary.json}, runs/dpo/{state.json, current.json} and TensorBoard
events to runs/dpo/tb/run/ (sweeps: --tag TAG writes everything to runs/dpo/sweep/TAG/ and
runs/dpo/tb/sweep_TAG/ instead, reusing the round's pairs and pi_ref log-probs).
Model-specific work (loading, log-probs, pi_ref, saving) goes through the adapter (dpo/adapters/).
Env: the model's venv (hparams models.<model>.venv). Re-running a finished round is a no-op; an interrupted round restarts from its start.

  python dpo/train.py --round 1
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpolib  # noqa: E402

EXIT_NEEDS_USER = 3
EXIT_NONFINITE = 4
PAPER_TAGS = ("k_hat", "d_hat", "margin", "margin_abs_max", "u_w", "u_l", "beta", "beta_next", "logratio_var")


def build_optimizer(params, o):
    import torch
    if o["name"] == "adamw":
        return torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                 weight_decay=o["weight_decay"])
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                weight_decay=o["weight_decay"])
    if o["name"] == "rmsprop":
        return torch.optim.RMSprop(params, lr=o["lr"], alpha=o["alpha"], eps=o["eps"],
                                   weight_decay=o["weight_decay"])
    return torch.optim.SGD(params, lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])


def build_scheduler(opt, o, total_steps):
    import torch
    warm, kind = o["warmup_steps"], o["scheduler"]

    def factor(step):
        if warm and step < warm:
            return (step + 1) / warm
        if kind == "constant":
            return 1.0
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 1.0 - prog if kind == "linear" else 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def tb_writer(log_dir, purge_step=None):
    """purge_step drops events an interrupted attempt of this round already wrote at those steps."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(log_dir), purge_step=purge_step, flush_secs=30)
    except ImportError:
        print("[warn] tensorboard is not installed in this venv (scripts/dpo/setup.sh installs it); "
              "metrics go to train_log.jsonl only", flush=True)
        return None


def load_examples(rdir, pairs):
    import torch
    shards, out = {}, []
    for q in pairs:
        for s in (q["shard_w"], q["shard_l"]):
            if s not in shards:
                shards[s] = torch.load(rdir / "codes" / s)
        sw, sl = shards[q["shard_w"]], shards[q["shard_l"]]
        out.append({"pair_id": q["pair_id"], "prompt": sw["prompts"][q["prompt_id"]],
                    "chosen": sw["codes"][q["chosen"]], "chosen_ended": q["ended_w"],
                    "rejected": sl["codes"][q["rejected"]], "rejected_ended": q["ended_l"]})
    return out


def batches(n, size, rng):
    order = list(range(n))
    rng.shuffle(order)
    return [order[i: i + size] for i in range(0, n, size)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-pairs", type=int, default=0, help="only the first N pairs (smoke test)")
    p.add_argument("--ref-batch", type=int, default=16)
    p.add_argument("--tag", default="", help="sweep run: write to runs/dpo/sweep/TAG/ and leave the run's "
                                             "current.json/state.json untouched")
    p.add_argument("--overlay", default="", help="JSON file whose beta/kl/optimizer keys override hparams "
                                                 "(one sweep setting, see scripts/dpo/sweep.sh)")
    args = p.parse_args()

    h = dpolib.load_hparams()
    if args.overlay:
        for sec, vals in dpolib.read_json(args.overlay).items():
            if sec in ("beta", "kl", "optimizer"):
                h[sec] = {**h[sec], **vals}
    missing = dpolib.check_user_review(h)
    if missing:
        print("dpo/hparams.json: these USER_REVIEW values are not set yet:\n  " + "\n  ".join(missing) +
              "\nSet them (README section 6), then re-run. Rollout, scores and pairs of this round are kept.",
              flush=True)
        sys.exit(EXIT_NEEDS_USER)

    r = args.round
    rdir = dpolib.round_dir(r)                       # inputs: pairs, code shards, pi_ref log-probs
    odir = dpolib.RUNS / "sweep" / args.tag if args.tag else rdir          # outputs
    state_file = odir / "state.json" if args.tag else dpolib.RUNS / "state.json"
    ckpt_out = odir / "ckpt" if args.tag else dpolib.ckpt_dir(r)
    if (odir / "train_summary.json").exists():
        print(f"already trained ({odir / 'train_summary.json'})", flush=True)
        return

    import torch
    import torch.nn.functional as F
    import adapters

    bcfg, kcfg, o = h["beta"], h["kl"], h["optimizer"]
    name, cfg, _ = dpolib.model_cfg(h)
    current = {"policy": cfg["policy_init"], "round": 0}
    if not args.tag:                   # a sweep setting always trains from pi_ref (scripts/dpo/sweep.sh)
        current = dpolib.read_json(dpolib.RUNS / "current.json", current)
    # an interrupted round restarts from the state it started with
    before = odir / "state_before.json"
    if not before.exists():
        start_state = dpolib.read_json(state_file, {
            "beta": bcfg["beta0"], "d_hat": kcfg["epsilon"] if kcfg.get("epsilon") is not None else 0.0,
            "global_step": 0})
        start_state["policy"] = current["policy"]
        dpolib.write_json(before, start_state)
    state = dpolib.read_json(before)
    current = {"policy": state.get("policy", current["policy"]), "round": r - 1}
    beta, d_hat, gstep = float(state["beta"]), float(state["d_hat"]), int(state["global_step"])
    if bcfg["mode"] == "fixed":
        beta = float(bcfg["beta0"])    # a changed beta0 applies from the next round on

    pairs = dpolib.read_jsonl(rdir / "pairs.jsonl")
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        sys.exit(f"no pairs in {rdir}: nothing to train on")
    device = torch.device(args.device)
    examples = load_examples(rdir, pairs)
    # policy = pi_theta_r (current.json); the adapter also holds pi_ref = models.<model>.policy_init
    A = adapters.get(name)(dpolib.resolve(cfg["policy_init"]), dpolib.resolve(current["policy"]), device, cfg,
                           for_training=True)

    # ---- pi_ref log-probs (once per round; frozen model, same code path and precision) ----------
    ref_path = rdir / "ref_logps.json"
    ref = dpolib.read_json(ref_path, {})
    todo = [i for i, ex in enumerate(examples) if ex["pair_id"] not in ref]
    if todo:
        A.eval()
        t0 = time.time()
        for i in range(0, len(todo), args.ref_batch):
            chunk = [examples[j] for j in todo[i: i + args.ref_batch]]
            lw, ll = A.reference_logps(A.collate(chunk)).float().chunk(2)
            for ex, a, c in zip(chunk, lw.tolist(), ll.tolist()):
                ref[ex["pair_id"]] = [a, c]
            if (i // args.ref_batch) % 20 == 0:
                print(f"  pi_ref log-probs {min(i + args.ref_batch, len(todo))}/{len(todo)}", flush=True)
        dpolib.write_json(ref_path, ref)
        print(f"pi_ref log-probs for {len(todo)} pairs in {time.time() - t0:.0f}s", flush=True)
        dpolib.log_timing(f"r{r:02d}_ref_logps{'_' + args.tag if args.tag else ''}", time.time() - t0, pairs=len(todo))
    A.free_reference()

    # ---- policy, optimizer -----------------------------------------------------------------------
    A.train()
    params = A.trainable_parameters()
    opt = build_optimizer(params, o)
    opt_path = odir / "optimizer.pt" if args.tag else dpolib.RUNS / "state" / "optimizer.pt"
    if o["state_across_rounds"] == "carry" and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location=device))
        for g in opt.param_groups:          # this round's lr/betas from hparams, moments carried
            g["lr"] = o["lr"]
            g["initial_lr"] = o["lr"]
    rng = random.Random(h["rollout"]["seed"] + r)
    micro = [mb for _ in range(o["epochs_per_round"]) for mb in batches(len(examples), o["batch_pairs"], rng)]
    total_steps = math.ceil(len(micro) / o["grad_accum"])
    sched = build_scheduler(opt, o, total_steps)
    tb = tb_writer(dpolib.RUNS / "tb" / (f"sweep_{args.tag}" if args.tag else "run"), purge_step=gstep + 1)
    log_path = odir / "train_log.jsonl"
    if log_path.exists():           # an interrupted attempt: keep its log as evidence, start a fresh one
        n_old = len(list(odir.glob("train_log.attempt*.jsonl")))
        log_path.rename(odir / f"train_log.attempt{n_old + 1}.jsonl")
    print(f"round {r}: {len(examples)} pairs, {len(micro)} micro-batches, {total_steps} optimizer steps; "
          f"beta {beta:.4g} ({bcfg['mode']}), D_hat {d_hat:.4g}, policy {current['policy']}", flush=True)

    def per_frame(u, frames):
        return u / frames.clamp_min(1) if kcfg["k_hat_per"] == "frame" else u

    acc = {"loss": 0.0, "margin": [], "u_w": [], "u_l": [], "k_hat": [], "correct": [], "lw": [], "ll": [], "lw_f": [], "ll_f": [],
           "ent": []}
    violated, steps_done, t0 = False, 0, time.time()
    first_entropy = last_entropy = None
    t_step = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    for m, idx in enumerate(micro):
        chunk = [examples[i] for i in idx]
        b = A.collate(chunk)
        lp, ent = A.sequence_logps(b, return_entropy=True)
        lw, ll = lp.float().chunk(2)
        rr_ = torch.tensor([ref[ex["pair_id"]] for ex in chunk], device=device)
        u_w, u_l = lw - rr_[:, 0], ll - rr_[:, 1]                        # Alg. 1 line 8
        margin = u_w - u_l                                              # line 9
        loss = -F.logsigmoid(beta * margin).mean()                      # Eq. 2
        (loss / o["grad_accum"]).backward()
        with torch.no_grad():
            fw, fl = b["n_frames"].to(device).chunk(2)
            k_hat = 0.5 * (per_frame(u_w, fw) ** 2 + per_frame(u_l, fl) ** 2)   # line 9
            acc["loss"] += loss.item() * len(chunk)
            for key, v in (("margin", margin), ("u_w", u_w), ("u_l", u_l), ("k_hat", k_hat),
                           ("correct", (margin > 0).float()), ("lw", lw), ("ll", ll),
                           ("lw_f", lw / fw.clamp_min(1)), ("ll_f", ll / fl.clamp_min(1))):
                acc[key] += v.tolist()
            acc["ent"].append(ent.tolist())
        if (m + 1) % o["grad_accum"] and m + 1 < len(micro):
            continue
        if o["max_grad_norm"] > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, o["max_grad_norm"]))
        else:
            gnorm = float(torch.norm(torch.stack([q.grad.norm() for q in params if q.grad is not None])))
        if not (math.isfinite(gnorm) and math.isfinite(acc["loss"])):
            dpolib.log_decision(f"round {r}{' ' + args.tag if args.tag else ''}: non-finite training values",
                                f"loss sum {acc['loss']}, grad norm {gnorm} at step {gstep + 1}: stopped before the "
                                f"optimizer step; nothing was saved. Check train_log.jsonl and the pairs.")
            if tb:
                tb.close()
            sys.exit(EXIT_NONFINITE)
        lr_used = sched.get_last_lr()[0]
        opt.step()                                                      # line 11 (with beta_t)
        sched.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
        steps_done += 1
        n = len(acc["k_hat"])
        k_mean = sum(acc["k_hat"]) / n
        d_hat = (1 - kcfg["ema_alpha"]) * d_hat + kcfg["ema_alpha"] * k_mean   # line 10
        beta_used = beta
        if bcfg["mode"] == "dual_ascent":                               # line 12
            # section 4: the O(sqrt T) regret bound uses eta_t = O(1/sqrt t); t = global step, across rounds
            eta = bcfg["eta_beta"] / math.sqrt(gstep) if bcfg.get("eta_schedule") == "inv_sqrt" else bcfg["eta_beta"]
            beta = min(bcfg["beta_max"], max(bcfg["beta_min"], beta + eta * (d_hat - kcfg["epsilon"])))
        ent_mean = [sum(col) / len(col) for col in zip(*acc["ent"])]
        first_entropy = first_entropy or ent_mean
        last_entropy = ent_mean
        mean_ = lambda k: sum(acc[k]) / n   # noqa: E731
        u_all = acc["u_w"] + acc["u_l"]
        u_mean = sum(u_all) / len(u_all)
        row = {"step": gstep, "round": r, "pairs": n, "t": round(time.time(), 1),
               "step_seconds": round(time.time() - t_step, 3),
               "max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
               "loss": acc["loss"] / n, "beta": beta_used, "beta_next": beta,
               "margin": mean_("margin"), "margin_abs_max": max(abs(x) for x in acc["margin"]),
               "u_w": mean_("u_w"), "u_l": mean_("u_l"), "k_hat": k_mean, "d_hat": d_hat,
               "logratio_var": sum((u - u_mean) ** 2 for u in u_all) / max(1, len(u_all) - 1),
               "accuracy": mean_("correct"), "logp_chosen": mean_("lw"), "logp_rejected": mean_("ll"),
               "logp_rejected_min": min(acc["ll"]), "logp_chosen_per_frame": mean_("lw_f"),
               "logp_rejected_per_frame": mean_("ll_f"),
               "grad_norm": gnorm, "lr": lr_used, "entropy_mean": sum(ent_mean) / len(ent_mean),
               "entropy_first": ent_mean[0],   # MOSS: codebook 0; Orpheus: the only entry
               "entropy": ent_mean}
        dpolib.append_jsonl(log_path, row)
        if tb:
            for k, v in row.items():
                if k in ("step", "round", "pairs", "t", "entropy") or v is None:
                    continue
                tb.add_scalar(f"{'paper' if k in PAPER_TAGS else 'train'}/{k}", v, gstep)
            for i, e in enumerate(ent_mean):
                tb.add_scalar(f"entropy/{i:02d}", e, gstep)
            if kcfg.get("epsilon") is not None:     # reference lines for D_hat <= eps, log-ratio var <= 2 eps
                tb.add_scalar("paper/epsilon", kcfg["epsilon"], gstep)
                tb.add_scalar("paper/two_epsilon", 2 * kcfg["epsilon"], gstep)
        t_step = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if steps_done == 1 or steps_done % 10 == 0 or steps_done == total_steps:
            el = time.time() - t0
            print(f"[step {steps_done}/{total_steps}] loss {row['loss']:.4f} margin {row['margin']:+.3f} "
                  f"acc {row['accuracy']:.2f} k_hat {k_mean:.3g} D_hat {d_hat:.3g} beta {beta_used:.4g} "
                  f"|g| {gnorm:.3g}  {el / 60:.1f} min, ~{el / steps_done * (total_steps - steps_done) / 60:.1f} min left",
                  flush=True)
        acc = {k: ([] if k != "loss" else 0.0) for k in acc}
        if kcfg["on_violation"] != "none" and d_hat > kcfg["epsilon"]:
            violated = True
            print(f"D_hat {d_hat:.4g} > epsilon {kcfg['epsilon']}: {kcfg['on_violation']}", flush=True)
            break

    action = kcfg["on_violation"] if violated else None
    if action == "stop_round":
        dpolib.log_decision(f"round {r}{' ' + args.tag if args.tag else ''}: stopped early",
                            f"D_hat {d_hat:.4g} > epsilon {kcfg['epsilon']} after {steps_done}/{total_steps} steps; "
                            f"the steps so far are kept.")
    if action == "rollback":
        new_policy = current["policy"]
        dpolib.log_decision(f"round {r}: rolled back",
                            f"D_hat {d_hat:.4g} > epsilon {kcfg['epsilon']} after {steps_done} steps; the round's "
                            f"update was discarded, the policy stays {new_policy}. beta {beta:.4g} and D_hat are kept.")
    else:
        A.save_checkpoint(ckpt_out)
        new_policy = str(ckpt_out.relative_to(dpolib.REPO)) if ckpt_out.is_relative_to(dpolib.REPO) else str(ckpt_out)
        if o["state_across_rounds"] == "carry":
            opt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(opt.state_dict(), opt_path)
    if tb:
        tb.add_scalar("round/steps", steps_done, r)
        tb.add_scalar("round/violation", 1.0 if violated else 0.0, r)
        tb.flush()
        tb.close()

    log = dpolib.read_jsonl(log_path)
    summary = {
        "round": r, "policy_in": current["policy"], "policy_out": new_policy, "n_pairs": len(examples),
        "steps": steps_done, "planned_steps": total_steps, "violation": action,
        "beta_start": float(state["beta"]) if bcfg["mode"] == "dual_ascent" else float(bcfg["beta0"]),
        "beta_end": beta, "d_hat_end": d_hat, "global_step": gstep,
        "final": log[-1] if log else None,
        "mean_accuracy": sum(x["accuracy"] for x in log) / len(log) if log else None,
        "entropy_first": first_entropy, "entropy_last": last_entropy,
        "model": name, "hparams": {"beta": bcfg, "kl": kcfg, "optimizer": o, "precision": h["precision"]},
        "seconds": round(time.time() - t0, 1),
    }
    dpolib.write_json(state_file, {"beta": beta, "d_hat": d_hat, "global_step": gstep})
    if not args.tag:
        dpolib.write_json(dpolib.RUNS / "current.json", {"policy": new_policy, "round": r})
    summary["tag"] = args.tag or None
    dpolib.write_json(odir / "train_summary.json", summary)
    dpolib.log_timing(f"r{r:02d}_train", time.time() - t0, steps=steps_done, pairs=len(examples))
    print(f"round {r} trained: {steps_done} steps, policy -> {new_policy}", flush=True)


if __name__ == "__main__":
    main()
