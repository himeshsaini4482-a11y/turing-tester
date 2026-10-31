"""turing-tester benchmark: generate the set's sentences with the model, then judge them against the humans.

  1. sentence set + human audio      (sets.py)
  2. TTS clips                       the model's generator in its environment (models.py)
  3. SpeechJudge-BTRM                benchmark/btrm_score.py     (judge environment)
  4. SpeechJudge-GRM                 benchmark/judge.py          (judge environment)
  5. WER / speaker sim / UTMOS       benchmark/metrics.py        (tts environment)
  6. report.md                       report.py
Generation and both judges skip work already on disk, so a stopped run continues where it left off
(the metrics stage is cheap and is recomputed).
"""
import time
from pathlib import Path

from turing_tester import envs, hardware, models, paths, report, sets
from turing_tester.util import Fail, dir_fingerprint, file_key, py, read_json, run, say, write_json

JUDGES = ("grm", "btrm")


def parse_judges(arg):
    js = [j.strip().lower() for j in str(arg).split(",") if j.strip()]
    bad = [j for j in js if j not in JUDGES]
    if bad or not js:
        raise Fail(f"--judges takes a comma list of {', '.join(JUDGES)} (got {arg!r})")
    return js


def model_identity(model):
    if model.kind == "api":
        return {"api": model.api.get("name"), "url": model.api.get("url"), "cloning": model.api.get("voice_cloning")}
    return {"kind": model.kind, "source": model.source,
            "weights": dir_fingerprint(model.source) if model.local else model.source}


def run_benchmark(model, bench_set, run_dir, judges=JUDGES, limit=0, votes=1, name=None, metrics=True):
    """Benchmark `model` on a prepared BenchSet into run_dir; returns the report summary."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ident = {"model": model_identity(model), "set": bench_set.name, "limit": int(limit or 0),
             "sentences": file_key(bench_set.manifest)}
    meta_path = run_dir / "turing-tester.json"
    old = read_json(meta_path)
    if old and old.get("identity") != ident:
        raise Fail(f"{paths.rel(run_dir)} already holds results for another model, set or --limit:\n"
                   f"  {old.get('identity')}\n  choose another --name, or delete that folder to start over")
    meta = {"identity": ident, "name": name or run_dir.name, "model": model.source if model.kind != "api" else
            paths.rel(model.source), "model_label": model.label, "set": bench_set.name,
            "set_description": bench_set.description, "set_licence": bench_set.licence,
            "judges": list(judges), "limit": int(limit or 0),
            "voice_cloning": model.api.get("voice_cloning") if model.kind == "api" else "per_request",
            "started": (old or {}).get("started") or time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(meta_path, meta)
    log = run_dir / "turing-tester.log"
    device = hardware.device_string()

    say(f"generating with {model.label} ({paths.rel(model.source) if model.local or model.kind == 'api' else model.source})")
    model.ensure_env()
    script, args = model.generate_cmd(bench_set.manifest, run_dir, limit=limit, device=device)
    env_extra = moss_env() if model.kind == "moss_tts_nano" else None
    run([paths.env_python(model.env()), paths.REPO / script, *args], log=log, env=env_extra)
    manifest = run_dir / "manifest.csv"
    if not manifest.exists():
        raise Fail(f"the generator did not write {paths.rel(manifest)}")

    judge_env = {"TURING_TESTER_DEVICE": hardware.judge_device()}
    if "btrm" in judges:
        say("SpeechJudge-BTRM (downloads ~22 GB of weights the first time)")
        envs.ensure("judge")
        py("judge", "benchmark/btrm_score.py", "--manifest", manifest, "--out-dir", run_dir, "--limit", limit,
           log=log, env=judge_env)
    if "grm" in judges:
        say("SpeechJudge-GRM (downloads ~22 GB of weights the first time; about 30 s per sentence)")
        envs.ensure("judge")
        py("judge", "benchmark/judge.py", "--manifest", manifest, "--out-dir", run_dir, "--votes", votes,
           "--limit", limit, log=log, env=judge_env)
    if metrics:
        say("WER (Whisper), speaker similarity (WavLM), UTMOS")
        envs.ensure("tts")
        py("tts", "benchmark/metrics.py", "--manifest", manifest, "--out-dir", run_dir, "--limit", limit, log=log)

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(meta_path, meta)
    s = report.benchmark_report(run_dir, meta)
    return s


def moss_env():
    """MOSS-TTS-Nano needs its pinned code checkout; nothing else for generation."""
    return {"MOSS_REPO": envs.checkout_dir("moss")}


def main(args):
    if bool(args.model) == bool(args.api):
        raise Fail("give either --model <folder> or --api <apis/provider.yaml>")
    model = models.detect(args.model) if args.model else models.detect_api(args.api)
    if model.kind == "api":
        from turing_tester import apiconfig
        missing = apiconfig.missing_env(model.api)
        if missing:
            raise Fail(f"set {', '.join(missing)} in your environment first (keys are never read from files)")
    judges = parse_judges(args.judges)
    hw = hardware.require_gpu(args.allow_cpu)
    say(f"hardware: {hw}")
    bench_set = sets.prepare(args.set, args.limit)
    from turing_tester.util import run_name
    name = run_name(args.name, model.default_name + ("" if bench_set.name == "ears" else f"_{bench_set.name.replace(':', '_')}")
                    + (f"_limit{args.limit}" if args.limit else ""))
    if name in ("sft", "dpo"):      # runs/sft and runs/dpo hold training runs
        name += "_benchmark"
    run_dir = paths.RUNS / name
    s = run_benchmark(model, bench_set, run_dir, judges=judges, limit=args.limit, votes=args.votes)
    print()
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    say(f"report: {paths.rel(run_dir / 'report.md')}")
    return 0 if s["n_sentences"] else 1
