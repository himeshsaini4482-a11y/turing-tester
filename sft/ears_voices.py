"""EARS pieces the turing-tester CLI needs without the full EARS inventory (sft/ears_inventory.py):

  voices      every non-dev EARS speaker's rainbow_01_regular (the benchmark's voice-prompt
              convention), the voices a teacher speaks in and DPO rolls out in, as in our runs
              -> <out>/ears_voices.jsonl  {"speaker", "path", "transcript", "gender"}
  dev-mirror  the 4 held-out dev speakers (sft/data/dev_speakers.json) x the 19 benchmark reading
              names, cloned from their rainbow_01_regular, each repeated with 3 seeds: the dev set
              every SFT checkpoint and DPO round was judged on (sft/dev_sets.py mirror +
              eval-manifest --seeds 3) -> <out>/eval_mirror.csv

Audio goes to data/EARS/<spk>/ like the benchmark's (benchmark/prepare_ears.py downloads each
speaker's zip once and keeps only the files asked for). Env: TTS (soundfile).

  python sft/ears_voices.py voices --out data/sft
  python sft/ears_voices.py dev-mirror --out data/sft
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
import sftlib  # noqa: E402
from prepare_ears import PROMPT, RAW, ensure_speakers, fetch  # noqa: E402

BENCH = HERE.parent / "benchmark" / "manifests" / "ears_regular_26x19.csv"


def meta(ears_dir):
    os.makedirs(ears_dir, exist_ok=True)
    transcripts = json.load(open(fetch(f"{RAW}/transcripts.json", os.path.join(ears_dir, "transcripts.json"))))
    stats = json.load(open(fetch(f"{RAW}/speaker_statistics.json", os.path.join(ears_dir, "speaker_statistics.json"))))
    return transcripts, stats


def dev_speakers():
    return sftlib.read_json(HERE / "data" / "dev_speakers.json")["speakers"]


def cmd_voices(args):
    transcripts, stats = meta(args.ears_dir)
    dev = set(dev_speakers())
    speakers = sorted(s for s in stats if s not in dev)
    ensure_speakers({s: [PROMPT] for s in speakers}, args.ears_dir)
    rows = [{"speaker": s, "path": Path(args.ears_dir, s, PROMPT + ".wav").as_posix(),
             "transcript": transcripts[PROMPT], "gender": stats[s].get("gender")} for s in speakers]
    sftlib.write_jsonl(Path(args.out) / "ears_voices.jsonl", rows)
    print(f"{len(rows)} EARS voices (dev speakers excluded) -> {Path(args.out) / 'ears_voices.jsonl'}", flush=True)


def cmd_dev_mirror(args):
    import soundfile as sf
    import dev_sets
    transcripts, _ = meta(args.ears_dir)
    names = dev_sets.bench_reading_names(sftlib.read_csv(BENCH))
    speakers = dev_speakers()
    ensure_speakers({s: [PROMPT] + names for s in speakers}, args.ears_dir)
    inventory = {}
    for s in speakers:
        for n in [PROMPT] + names:
            path = Path(args.ears_dir, s, n + ".wav")
            inventory[(s, n)] = {"path": path.as_posix(), "transcript": transcripts[n],
                                 "duration": sf.info(str(path)).duration}
    rows = dev_sets.mirror_rows(speakers, names, inventory, copy=lambda src, name: Path(src).as_posix())
    out = Path(args.out) / "eval_mirror.csv"
    dev_sets.write(out, dev_sets.eval_rows(rows, args.seeds))
    print(f"dev mirror: {len(rows)} items x {args.seeds} seeds -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("voices", "dev-mirror"))
    p.add_argument("--out", required=True)
    p.add_argument("--ears-dir", default="data/EARS")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    {"voices": cmd_voices, "dev-mirror": cmd_dev_mirror}[args.cmd](args)


if __name__ == "__main__":
    main()
