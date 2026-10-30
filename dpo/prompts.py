"""DPO prompt pool: texts written by the Qwen 3.5 9B, each paired with one training voice.

  texts  The 9B writes hparams texts.n sentences through sft/gen_texts.py, unchanged: same
         prompt, same validation (8-40 words, 1-3 sentences, no digits), same dedupe against every
         runs/sft/*/texts.jsonl and the 19 benchmark texts, same verbatim benchmark-sentence guard.
         --report picks the weakness mix: a weakness_report.json, "auto" (the newest SFT main-round
         report, i.e. what the SFT's last diagnosis on dev speakers found still weak) or "general".
         The 494 benchmark judgments are never used to steer texts.
  build  Voices = every EARS speaker's rainbow_01_regular (the benchmark's voice-prompt
         convention), minus the 4 dev speakers (held out for the per-round eval). Each text gets
         one voice (balanced, seeded), the pool is shuffled once and round r uses slice r, so every
         round sees fresh prompts (online: no prompt is reused across rounds).

Writes runs/dpo/texts.jsonl (sft/INTERFACES.md section 2.7 format), runs/dpo/voices/<spk>.wav
(durable copies, ~1.5 MB each) and runs/dpo/prompts.jsonl:
  {"prompt_id": "dpo_p00042", "text_id": "dpo_t00042", "text", "category", "speaker", "voice_wav",
   "voice_transcript"}   (the transcript conditions Orpheus; MOSS voice_clone ignores it)

Env: texts = STD (calls the 9B over HTTPS, see scripts/qwen9b_client.py); build = TTS (soundfile,
only for the EARS re-download fallback when the NVMe was wiped). Both resumable.

  python dpo/prompts.py texts --n 6000 --report auto
  python dpo/prompts.py build
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpolib  # noqa: E402

STAGE = "dpo"
PROMPT_READING = "rainbow_01_regular"


def newest_sft_report():
    reports = sorted(dpolib.SFT_RUNS.glob("main_r*/diag/weakness_report.json"),
                     key=lambda p: int(p.parent.parent.name.split("_r")[-1]))
    return reports[-1] if reports else None


def cmd_texts(args):
    import gen_texts   # sft/, on sys.path via dpolib
    if args.report == "general":
        report, source = {}, "general"
    else:
        path = newest_sft_report() if args.report == "auto" else Path(args.report)
        report = dpolib.read_json(path, None) if path else None
        if report is None:
            print(f"no weakness report ({args.report}); writing general texts only", flush=True)
            report, source = {}, "general"
        else:
            source = str(path)
    out = dpolib.RUNS / "texts.jsonl"
    print(f"9B texts -> {out} (weakness mix from {source})", flush=True)
    gen_texts.generate(report, args.n, out, STAGE, args.workers)
    rows = dpolib.read_jsonl(out)
    dpolib.write_json(dpolib.RUNS / "texts_summary.json", {
        "n": len(rows), "report": source,
        "by_category": {c: sum(r["category"] == c for r in rows) for c in sorted({r["category"] for r in rows})},
    })
    print(f"{len(rows)} texts in {out}", flush=True)


def voices(include_bench):
    """{speaker: durable voice wav}, copying each prompt clip to runs/dpo/voices/ once."""
    inv = dpolib.read_jsonl(dpolib.sftlib.DATA / "ears_inventory.jsonl")
    if not inv:
        sys.exit("runs/sft/data/ears_inventory.jsonl missing: the SFT pipeline makes it (pilot step 03)")
    dev = set(dpolib.read_json(dpolib.DEV_SPEAKERS)["speakers"])
    bench = {r["speaker_id"] for r in dpolib.read_csv(dpolib.sftlib.BENCH_MANIFEST)}
    rows = {r["speaker"]: r for r in inv if r["name"] == PROMPT_READING}
    chosen = sorted(s for s in rows if s not in dev and (include_bench or s not in bench))
    vdir = dpolib.RUNS / "voices"
    vdir.mkdir(parents=True, exist_ok=True)
    missing = {s: [PROMPT_READING] for s in chosen
               if not (vdir / f"{s}.wav").exists() and not os.path.exists(rows[s]["path"])}
    if missing:   # NVMe wiped since the inventory was made: re-download those speakers' prompt clip
        dpolib.sftlib.add_benchmark_to_path()
        from prepare_ears import ensure_speakers
        ears_dir = Path(rows[chosen[0]]["path"]).parents[1]
        ears_dir.mkdir(parents=True, exist_ok=True)
        ensure_speakers(missing, str(ears_dir))
    out = {}
    for s in chosen:
        dst = vdir / f"{s}.wav"
        if not dst.exists():
            shutil.copy2(rows[s]["path"], dst)
        out[s] = {"wav": str(dst.relative_to(dpolib.REPO)), "transcript": rows[s]["transcript"]}
    return out, len(set(chosen) & bench)


def cmd_build(args):
    h = dpolib.load_hparams()
    texts = dpolib.read_jsonl(dpolib.RUNS / "texts.jsonl")
    if not texts:
        sys.exit("runs/dpo/texts.jsonl is empty: run `prompts.py texts` first")
    out = dpolib.RUNS / "prompts.jsonl"
    if out.exists() and not args.force:
        print(f"{out} exists ({len(dpolib.read_jsonl(out))} prompts); --force rebuilds it", flush=True)
        return
    if args.voices:   # voicegym CLI: an explicit voice list ({speaker, path, transcript} per line)
        bench = {r["speaker_id"] for r in dpolib.read_csv(dpolib.sftlib.BENCH_MANIFEST)}
        vmap = {v["speaker"]: {"wav": v["path"], "transcript": v["transcript"]} for v in dpolib.read_jsonl(args.voices)}
        n_bench = len(set(vmap) & bench)
    else:
        vmap, n_bench = voices(h.get("voices", {}).get("include_benchmark_speakers", True))
    speakers = sorted(vmap)
    rng = random.Random(h["rollout"]["seed"])
    texts = sorted(texts, key=lambda r: r["id"])
    rng.shuffle(texts)
    # balanced voices: cycle through a shuffled speaker list, reshuffled every pass
    order = []
    while len(order) < len(texts):
        block = speakers[:]
        rng.shuffle(block)
        order += block
    rows = [{"prompt_id": f"{STAGE}_p{i:05d}", "text_id": t["id"], "text": t["text"],
             "category": t["category"], "speaker": spk, "voice_wav": vmap[spk]["wav"],
             "voice_transcript": vmap[spk]["transcript"]}
            for i, (t, spk) in enumerate(zip(texts, order))]
    dpolib.write_jsonl(out, rows)
    need = h["rounds"] * h["rollout"]["prompts_per_round"]
    print(f"{len(rows)} prompts over {len(speakers)} voices ({n_bench} of them benchmark speakers, "
          f"dev speakers excluded) -> {out}", flush=True)
    if len(rows) < need:
        print(f"[warn] {len(rows)} prompts < rounds x prompts_per_round = {need}: later rounds get fewer",
              flush=True)


def round_prompts(rows, r, per_round):
    """Round r's (1-based) slice of the shuffled pool."""
    return rows[(r - 1) * per_round: r * per_round]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("texts")
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--report", default="auto", help='weakness_report.json | "auto" | "general"')
    t.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    b.add_argument("--voices", default="", help="voices jsonl instead of the SFT EARS inventory (voicegym CLI)")
    args = p.parse_args()
    {"texts": cmd_texts, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
