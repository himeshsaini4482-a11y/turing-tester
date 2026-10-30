"""DPO round stage 1: the current policy speaks each of this round's prompts K times (Alg. 1 line 3).

Model-agnostic: everything model-specific goes through the hparams "model" adapter
(dpo/adapters/). Prompts are round r's slice of runs/dpo/prompts.jsonl. The adapter normalises
the text as the benchmark generator would and may drop it (e.g. over MOSS's 50-token single-chunk
limit: one generate call must be one completion). Each prompt is built once and replicated K
times; K x prompts are sampled in batches of rollout.batch_size, sorted by text and prompt length
so a batch stops together (length bucketing), and each batch's max_new is capped at the longest
sample the duration gate can still accept (reward.sec_per_word upper bound x its longest text).

Writes, in runs/dpo/r<rr>/:
  candidates.jsonl  {"cid": "<prompt_id>_k1", "prompt_id", "k", "speaker", "text", "text_id", "category",
                     "n_text_tokens", "n_frames", "seconds", "ended", "decoded", "wav", "shard", "voice_wav"}
  codes/b<nnnn>.pt  {"prompts": {prompt_id: prompt tensor}, "codes": {cid: completion tensor}}
  manifest.csv      for the scorers: sentence_id = cid, tts_wav, human_wav = prompt_wav = voice clip
  rollout_summary.json, rollout_skipped.json, rollout_log.jsonl (one row per batch: generate/decode
                    seconds, audio s/s, ended fraction, length spread, peak VRAM)
and mono wavs to <scratch>/r<rr>/wav/ (re-decoded from the shards if the NVMe was wiped).

Env: the model's venv (hparams models.<model>.venv). Resumable per prompt.

  python dpo/rollout.py --round 1 --policy runs/sft/best/final --repo-dir $MOSS_REPO
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpolib  # noqa: E402
from prompts import round_prompts  # noqa: E402


def write_wav(path, wav, sr):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.wav")
    sf.write(str(tmp), wav, sr)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--policy", required=True, help="checkpoint of pi_theta_r (runs/dpo/current.json)")
    p.add_argument("--repo-dir", default=None, help="MOSS-TTS-Nano checkout (its text normaliser); MOSS only")
    p.add_argument("--decode-batch", type=int, default=1,
                   help="codec decode batch; 1 = what the benchmark generator does (batched decodes may "
                        "differ from batch-1 ones, as batched encodes did in SFT gate G0a)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    args = p.parse_args()

    import torch
    import adapters
    from adapters.base import Sample, Voice

    h = dpolib.load_hparams()
    name, cfg, ro = dpolib.model_cfg(h)
    K, B, max_new = ro["k"], ro["batch_size"], ro["max_new"]
    if B % K:
        sys.exit(f"rollout batch_size {B} must be a multiple of k {K} (a prompt's K samples share a batch)")

    rdir = dpolib.round_dir(args.round)
    wav_dir = dpolib.scratch_dir() / rdir.name / "wav"
    (rdir / "codes").mkdir(parents=True, exist_ok=True)
    cand_path = rdir / "candidates.jsonl"
    prompts = round_prompts(dpolib.read_jsonl(dpolib.RUNS / "prompts.jsonl"), args.round, ro["prompts_per_round"])
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        sys.exit(f"no prompts for round {args.round} in runs/dpo/prompts.jsonl")

    done = dpolib.read_jsonl(cand_path)
    have = {}
    for c in done:
        have[c["prompt_id"]] = have.get(c["prompt_id"], 0) + 1
    partial = {pid for pid, n in have.items() if n != K}
    if partial:     # a crash mid-write: drop those prompts' rows, they are rolled out again
        print(f"dropping {len(partial)} partially written prompts from {cand_path}", flush=True)
        done = [c for c in done if c["prompt_id"] not in partial]
        dpolib.write_jsonl(cand_path, done)
        have = {pid: n for pid, n in have.items() if pid not in partial}
    skipped = dpolib.read_json(rdir / "rollout_skipped.json", {})
    todo = [q for q in prompts if have.get(q["prompt_id"], 0) < K and q["prompt_id"] not in skipped]
    print(f"round {args.round} ({name}): {len(prompts)} prompts, {len(prompts) - len(todo)} done or skipped, "
          f"{len(todo)} to roll out (K={K}, batch {B}, sampling={ro['sampling']})", flush=True)

    A = None

    def adapter():
        nonlocal A
        if A is None:
            A = adapters.get(name)(dpolib.resolve(cfg["policy_init"]), dpolib.resolve(args.policy), args.device, cfg,
                                   for_training=False, repo_dir=args.repo_dir)
        return A

    # wavs lost with the NVMe: re-decode them from the durable shards
    lost = [c for c in done if c["decoded"] and not os.path.exists(c["wav"])]
    if lost:
        print(f"re-decoding {len(lost)} candidate wavs missing from {wav_dir}", flush=True)
        shards = {}
        for i in range(0, len(lost), args.decode_batch):
            chunk = lost[i: i + args.decode_batch]
            for c in chunk:
                if c["shard"] not in shards:
                    shards[c["shard"]] = torch.load(rdir / "codes" / c["shard"])["codes"]
            wavs = adapter().decode([Sample(shards[c["shard"]][c["cid"]], c["ended"], c["n_frames"]) for c in chunk])
            for c, w in zip(chunk, wavs):
                write_wav(Path(c["wav"]), w, A.decoded_sample_rate)

    if todo:
        A = adapter()
        sampling = A.deployment_sampling if ro["sampling"] == "deployment" else A.on_policy_sampling
        items = []
        for q in todo:
            text, n_tok = A.prepare_text(q["text"])
            if text is None:
                skipped[q["prompt_id"]] = f"{n_tok} text tokens over the model's single-chunk limit"
                continue
            voice = Voice(q["speaker"], str(dpolib.REPO / q["voice_wav"]), q.get("voice_transcript", ""))
            items.append((q, n_tok, A.build_prompt(text, voice)))
        dpolib.write_json(rdir / "rollout_skipped.json", skipped)
        items.sort(key=lambda it: (it[1], len(it[2])))
        per_batch = B // K
        hi = dpolib.reward_cfg(h)["sec_per_word"][1]    # longer samples fail the duration gate anyway
        n_shards = len(list((rdir / "codes").glob("b*.pt")))
        start, gen_frames = time.time(), 0
        for bi in range(0, len(items), per_batch):
            batch = items[bi: bi + per_batch]
            seed = ro["seed"] + zlib.crc32(f"{args.round}:{batch[0][0]['prompt_id']}".encode()) % 100_000
            cap = A.max_new_for_seconds(hi * max(len(q["text"].split()) for q, _, _ in batch))
            batch_max = min(max_new, cap) if cap else max_new
            torch.cuda.reset_peak_memory_stats()
            t_gen = time.time()
            samples = A.generate([it[2] for it in batch for _ in range(K)], sampling, batch_max, seed)
            t_dec = time.time()
            wavs = []
            for i in range(0, len(samples), args.decode_batch):
                wavs += A.decode(samples[i: i + args.decode_batch])
            t_end = time.time()
            shard = f"b{n_shards:04d}.pt"
            n_shards += 1
            cands = []
            for j, (s, w) in enumerate(zip(samples, wavs)):
                q, n_tok, _ = batch[j // K]
                k = j % K + 1
                cid = f"{q['prompt_id']}_k{k}"
                cands.append({"cid": cid, "prompt_id": q["prompt_id"], "k": k, "speaker": q["speaker"], "text": q["text"],
                              "text_id": q["text_id"], "category": q["category"], "n_text_tokens": n_tok,
                              "n_frames": s.n_frames, "seconds": s.n_frames / A.frame_rate, "ended": s.ended,
                              "max_new": batch_max,
                              "decoded": w is not None, "wav": str(wav_dir / f"{cid}.wav"), "shard": shard,
                              "voice_wav": q["voice_wav"]})
            torch.save({"prompts": {q["prompt_id"]: pr for q, _, pr in batch},
                        "codes": {c["cid"]: s.completion for c, s in zip(cands, samples)}}, rdir / "codes" / shard)
            for c, w in zip(cands, wavs):
                if w is not None:
                    write_wav(Path(c["wav"]), w, A.decoded_sample_rate)
            # after the wavs, one write per batch: a listed candidate always has its audio, and a crash
            # leaves at most one partial prompt (dropped on resume, above)
            with open(cand_path, "a", encoding="utf-8") as f:
                f.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cands))
            frames = sum(s.n_frames for s in samples)
            gen_frames += frames
            el = time.time() - start
            n_done = min(bi + per_batch, len(items))
            dpolib.append_jsonl(rdir / "rollout_log.jsonl", {
                "t": round(time.time(), 1), "shard": shard, "rows": len(samples), "max_new": batch_max,
                "gen_s": round(t_dec - t_gen, 2), "dec_s": round(t_end - t_dec, 2),
                "audio_s": round(frames / A.frame_rate, 1), "audio_s_per_s": round(frames / A.frame_rate / (t_end - t_gen), 2),
                "ended_frac": sum(x.ended for x in samples) / len(samples),
                "max_over_mean_frames": max(x.n_frames for x in samples) / max(1e-9, frames / len(samples)),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            print(f"[{n_done}/{len(items)} prompts] batch: generate {t_dec - t_gen:.1f}s, decode {t_end - t_dec:.1f}s, "
                  f"{frames / A.frame_rate / (t_end - t_gen):.1f} audio s/s | {el / 60:.1f} min elapsed, "
                  f"~{el / n_done * (len(items) - n_done) / 60:.1f} min left, "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if items:
            dpolib.log_timing(f"r{args.round:02d}_rollout_gen", time.time() - start, prompts=len(items),
                              candidates=len(items) * K, audio_seconds=round(gen_frames / A.frame_rate, 1))

    cands = [c for c in dpolib.read_jsonl(cand_path) if c["prompt_id"] in {q["prompt_id"] for q in prompts}]
    dpolib.write_csv(rdir / "manifest.csv", [
        {"sentence_id": c["cid"], "speaker_id": c["speaker"], "text": c["text"], "tts_wav": c["wav"],
         "human_wav": str(dpolib.REPO / c["voice_wav"]), "prompt_wav": str(dpolib.REPO / c["voice_wav"])}
        for c in cands if c["decoded"]])
    dpolib.write_json(rdir / "rollout_summary.json", {
        "round": args.round, "model": name, "policy": args.policy, "sampling": ro["sampling"], "k": K,
        "max_new": max_new, "n_prompts": len({c["prompt_id"] for c in cands}), "n_candidates": len(cands),
        "n_skipped_prompts": len(skipped), "n_not_ended": sum(not c["ended"] for c in cands),
        "n_undecodable": sum(not c["decoded"] for c in cands),
        "mean_seconds": sum(c["seconds"] for c in cands) / max(1, len(cands)),
    })
    print(f"{len(cands)} candidates; manifest {rdir / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
