"""DPO pre-flight gates for the hparams "model" adapter: run on the GPU before round 1 trains.
Any failure stops the run (D6 is information only).

  D1 log-prob = sampler.  adapter.sequence_logps (batched, teacher-forced: what train.py
     differentiates) equals adapter.incremental_logp (step by step through the KV cache, the way
     the sampler runs) on real samples of pi_ref, for both deployment and on-policy decoding.
     If this fails, the DPO loss scores sequences under another distribution than the policy's.
  D2 padding invariance.  A sequence's log-prob alone equals its log-prob in a padded batch.
  D3 no dropout.  Every dropout probability is 0, so train-mode log-probs equal eval-mode ones
     and u = log pi_theta - log pi_ref is exactly 0 at the start of round 1.
  D4 gradient direction.  A small step along +grad(margin) raises the margin of a pair (no beta,
     no optimizer settings: those are the user's).
  D5 checkpoint round trip.  save_checkpoint -> reload for training gives the same log-probs.
  D6 rollout throughput.  One batch at the configured batch size: audio s/s, peak VRAM.
  D7 fast sampler identical (MOSS).  The vectorised repetition penalty (nano.fast_repetition_penalty)
     and the pinned original give the same codes for the same seeded batch.

Uses round 1's first prompts (runs/dpo/prompts.jsonl). Writes runs/dpo/gates.json.
Env: the model's venv.

  python dpo/gates.py --repo-dir $MOSS_REPO
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpolib  # noqa: E402

TOL_ABS = 0.05      # nats per sequence (sequences have ~1-4k token terms)
TOL_REL = 1e-4


def close(a, b):
    return abs(a - b) <= max(TOL_ABS, TOL_REL * abs(b))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=3, help="prompts used for D1/D2")
    args = p.parse_args()

    import torch
    import adapters
    from adapters.base import Voice

    h = dpolib.load_hparams()
    name, cfg, ro = dpolib.model_cfg(h)
    Adapter = adapters.get(name)
    init = dpolib.resolve(cfg["policy_init"])
    prompts = dpolib.read_jsonl(dpolib.RUNS / "prompts.jsonl")[: args.n]
    if not prompts:
        sys.exit("runs/dpo/prompts.jsonl is empty: run dpo/prompts.py first")
    res = {}

    # samples of pi_ref (rollout-mode load, as rollout.py does)
    gen = Adapter(init, init, args.device, cfg, for_training=False, repo_dir=args.repo_dir)
    built = []
    for q in prompts:
        text, _ = gen.prepare_text(q["text"])
        if text is not None:
            built.append(gen.build_prompt(text, Voice(q["speaker"], str(dpolib.REPO / q["voice_wav"]),
                                                      q.get("voice_transcript", ""))))
    samples = []
    for i, pr in enumerate(built):
        for mode, kw in (("deployment", gen.deployment_sampling), ("on_policy", gen.on_policy_sampling)):
            s = gen.generate([pr], kw, ro["max_new"], seed=1000 + i)[0]
            if s.n_frames:
                samples.append({"prompt": pr, "completion": s.completion, "ended": s.ended, "decoding": mode})

    # D6 on the same loaded model
    B = ro["batch_size"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    batch = gen.generate([built[0]] * B, gen.deployment_sampling if ro["sampling"] == "deployment"
                         else gen.on_policy_sampling, ro["max_new"], seed=7)
    torch.cuda.synchronize()
    dt = time.time() - t0
    t1 = time.time()
    gen.decode(batch[:16])
    dec = time.time() - t1
    audio = sum(s.n_frames for s in batch) / gen.frame_rate
    res["D6_rollout_throughput"] = {"pass": True, "info_only": True, "batch_size": B, "generate_s": dt,
                                    "audio_s": audio, "audio_s_per_s": audio / dt, "decode_16_s": dec,
                                    "n_ended": sum(s.ended for s in batch),
                                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
    # D7: speed-ups inside the sampler must leave the samples bit-identical (adapters that have one)
    if hasattr(gen, "sampler_equivalence"):
        eq_prompts = [pr for pr in built for _ in range(ro["k"])][:16]
        res["D7_fast_sampler_identical"] = gen.sampler_equivalence(eq_prompts, gen.deployment_sampling,
                                                                   min(ro["max_new"], 120), seed=11)
    del gen
    torch.cuda.empty_cache()

    # training-mode adapter: D1-D5
    A = Adapter(init, init, args.device, cfg, for_training=True)
    A.eval()

    def logps(items, adapter=A, amp=True):
        exs = [{"prompt": s["prompt"], "chosen": s["completion"], "chosen_ended": s["ended"],
                "rejected": s["completion"], "rejected_ended": s["ended"]} for s in items]
        saved = getattr(adapter, "amp", None)    # the adapters' torch.autocast kwargs
        if not amp and saved is not None:       # same code path, autocast off (the loaded dtype, like incremental)
            adapter.amp = {**saved, "enabled": False}
        try:
            with torch.no_grad():
                return adapter.sequence_logps(adapter.collate(exs))[: len(items)].float().tolist()
        finally:
            if saved is not None:
                adapter.amp = saved

    fast = logps(samples)
    # D1 compares in one precision: bf16 autocast alone moves a ~300-frame x 17-term sum by ~1 nat
    # (CPU test with a tiny random model: 0.4-1.6 nats at <= 60 frames), far above TOL_ABS
    fast_ld = logps(samples, amp=False)
    slow = [A.incremental_logp(s["prompt"], s["completion"], s["ended"]) for s in samples]
    res["D1_logp_matches_sampler"] = {
        "pass": bool(samples) and all(close(a, b) for a, b in zip(fast_ld, slow)),
        "rows": [{"decoding": s["decoding"], "frames": len(s["completion"]), "ended": s["ended"], "batched": a,
                  "incremental": b, "abs_diff": abs(a - b), "batched_bf16_autocast": c, "bf16_abs_diff": abs(c - b)}
                 for s, a, b, c in zip(samples, fast_ld, slow, fast)],
        "note": "batched and incremental both in the loaded dtype (autocast off): diffs are precision, "
                "a wrong shift or support shows up as tens of nats. bf16_abs_diff (training precision) is info only.",
    }
    # D2/D3 compare in the loaded dtype like D1: under bf16 autocast a different batch shape alone moves a
    # sequence's sum by ~1-2 nats (first GPU run: 2.09), which a real padding or dropout bug dwarfs.
    # Training never mixes shapes: policy and pi_ref always see the same batch (checked below, in bf16).
    alone = [logps([s], amp=False)[0] for s in samples]
    alone_bf16 = [logps([s])[0] for s in samples]
    res["D2_padding_invariance"] = {"pass": bool(samples) and all(close(a, b) for a, b in zip(alone, fast_ld)),
                                    "max_abs_diff": max((abs(a - b) for a, b in zip(alone, fast_ld)), default=None),
                                    "bf16_max_abs_diff_info": max((abs(a - b) for a, b in zip(alone_bf16, fast)),
                                                                  default=None)}
    drops = A.dropout_probs()
    nonzero = {k: v for k, v in drops.items() if v}
    eval_lp = logps(samples[:2], amp=False)
    A.train()
    train_lp = logps(samples[:2], amp=False)
    A.eval()
    res["D3_no_dropout"] = {"pass": not nonzero and all(close(a, b) for a, b in zip(train_lp, eval_lp)),
                            "nonzero": nonzero, "n_checked": len(drops)}
    # u at init in training precision: policy and pi_ref on the same batch, as train.py computes them
    pol = logps(samples[:2])
    with torch.no_grad():
        exs = [{"prompt": s["prompt"], "chosen": s["completion"], "chosen_ended": s["ended"],
                "rejected": s["completion"], "rejected_ended": s["ended"]} for s in samples[:2]]
        ref = A.reference_logps(A.collate(exs))[:2].float().tolist()
    res["D3_no_dropout"]["u_at_init"] = [a - b for a, b in zip(pol, ref)]
    res["D3_no_dropout"]["pass"] &= all(abs(a - b) <= TOL_ABS for a, b in zip(pol, ref))

    # D4: chosen = sample 0, rejected = another sample of the same prompt
    pair = [s for s in samples if s["prompt"] is samples[0]["prompt"]][:2] if samples else []
    if len(pair) == 2:
        A.train()
        b = A.collate([{"prompt": pair[0]["prompt"], "chosen": pair[0]["completion"], "chosen_ended": pair[0]["ended"],
                        "rejected": pair[1]["completion"], "rejected_ended": pair[1]["ended"]}])
        params = A.trainable_parameters()

        def margin():
            lw, ll = A.sequence_logps(b).float()
            return lw - ll
        m0 = margin()
        m0.backward()
        with torch.no_grad():
            gn = torch.sqrt(sum((q.grad.float() ** 2).sum() for q in params if q.grad is not None))
            for q in params:
                if q.grad is not None:
                    q.add_(q.grad, alpha=1e-3 / float(gn))     # unit-direction step of length 1e-3
                    q.grad = None
            m1 = margin()
        res["D4_gradient_direction"] = {"pass": float(m1) > float(m0), "margin_before": float(m0),
                                        "margin_after": float(m1), "grad_norm": float(gn)}
        A.eval()
    else:
        res["D4_gradient_direction"] = {"pass": False, "note": "need 2 non-empty samples of one prompt"}

    # D5: the (slightly stepped) policy saved and reloaded for training gives the same log-probs
    before = logps(samples[:2])
    with tempfile.TemporaryDirectory(dir=dpolib.scratch_dir()) as tmp:
        out = os.path.join(tmp, "ckpt")
        A.save_checkpoint(out)
        del A
        torch.cuda.empty_cache()
        again = Adapter(init, out, args.device, cfg, for_training=True)
        again.eval()
        rt = logps(samples[:2], again)
        del again
    res["D5_checkpoint_roundtrip"] = {"pass": all(close(a, b) for a, b in zip(rt, before)),
                                      "before": before, "after_reload": rt}

    ok = all(v["pass"] for v in res.values())
    dpolib.write_json(dpolib.RUNS / "gates.json", {"pass": ok, "model": name, "policy_init": init, "gates": res})
    for k, v in sorted(res.items()):
        print(f"{'PASS' if v['pass'] else 'FAIL'} {k}", flush=True)
    d6 = res["D6_rollout_throughput"]
    print(f"D6: {d6['audio_s_per_s']:.1f} audio s per s at batch {B}, peak {d6['peak_vram_gib']:.1f} GiB", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
