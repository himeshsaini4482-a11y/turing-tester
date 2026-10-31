"""turing-tester dpo: online DPO exactly as in our experiments (dpo/README.md), for one model.

Each round the model speaks 1,000 fresh prompts 4 times, SpeechJudge-BTRM scores the samples, the
best and the worst valid sample form a pair (invalid: WER > 0.2, speaker similarity < 0.85, outside
0.15-1.0 s per word, no stop), and the model is trained to prefer the best over the worst against a
frozen copy of where it started. Settings: dpo/archive/hparams_trial_01_beta0.1_adamw.json (beta 0.1
fixed, AdamW lr 1e-6, 16 pairs per step, one pass per round); up to 3 rounds, stopping after a round
that does not raise the dev BTRM % or fails the guard rails. Then the best round and the starting
model are benchmarked (EARS 494, or your voice's test clips).

Prompts: texts from --texts, from the LLM when TURING_TESTER_LLM_* is set (as in our runs), or from the
bundled pool; voices = every non-dev EARS speaker's reading (as in our runs), or yours with --voice.
Resumable at every step (runs/dpo/<name>/.done/).
"""
import json
import random
import shutil
from pathlib import Path

from turing_tester import bench, envs, hardware, models, paths, report, sets
from turing_tester.sft import POOL, SFT_DATA, llm_configured, read_csv, read_jsonl, user_texts, write_csv, write_jsonl
from turing_tester.util import Fail, Steps, py, read_json, run_name, say, write_json

TRIAL = paths.REPO / "dpo" / "archive" / "hparams_trial_01_beta0.1_adamw.json"
ADAPTER = {"moss_tts_nano": ("moss_nano", "moss"), "orpheus": ("orpheus", "lm"), "token_lm": ("tokenlm", "lm")}
RESERVED = {"eval", "ckpt", "sweep", "tb", "logs", "state", "voices"}
EXIT_PAUSED, EXIT_STOP = 3, 10


def abs_policy(p):
    """dpo/train.py records policies relative to the repo; the CLI may run from anywhere."""
    q = Path(p)
    return str(q if q.is_absolute() else paths.REPO / q)


class Run:
    def __init__(self, args):
        self.args = args
        self.model = models.detect(args.model)
        if not self.model.can_dpo:
            raise Fail(f"DPO needs a model that reports token log-probabilities: MOSS-TTS-Nano, Orpheus or a "
                       f"turing-tester.yaml model ({self.model.label} cannot; see README 'What can be trained')")
        self.adapter, self.env_name = ADAPTER[self.model.kind]
        if self.model.kind == "token_lm":
            self.env_name = self.model.env()      # "lm", or the folder's own environment with its requirements.txt
        self.voice = Path(args.voice).resolve() if args.voice else None
        src = Path(self.model.source)
        base = (src.parent.name if src.name in ("final", "checkpoint-last") else src.name)
        self.name = run_name(args.name, base + ("_" + self.voice.name if self.voice else ""))
        if self.name in RESERVED or self.name.startswith("r0"):
            raise Fail(f"--name {self.name} is reserved inside runs/dpo/")
        self.dir = paths.RUNS / "dpo" / self.name
        self.steps = Steps(self.dir)
        self.device = hardware.device_string()
        self.log = self.dir / "turing-tester.log"
        self.policy_init = str(src.resolve()) if self.model.local else self.model.source
        if self.model.kind == "moss_tts_nano" and not self.model.local:
            # the MOSS adapter and checkpoint code need a local folder (they copy its code files)
            self.policy_init = str(envs.ensure_moss_weights()[0])

    def env(self):
        e = {"DPO_RUNS": self.dir, "DPO_HPARAMS": self.dir / "hparams.json", "DPO_SCRATCH": self.dir / "scratch",
             "SFT_DATA": SFT_DATA / "ears", "SFT_DEV": SFT_DATA / "dev",
             "TURING_TESTER_DEVICE": hardware.judge_device()}
        if self.model.kind == "moss_tts_nano":
            e["MOSS_REPO"] = envs.checkout_dir("moss")
        return e

    def dpy(self, script, *args, check=True):
        return py(self.env_name, script, *args, log=self.log, env=self.env(), check=check)

    def repo_arg(self):
        return ["--repo-dir", envs.checkout_dir("moss")] if self.model.kind == "moss_tts_nano" else []

    # ---- setup -------------------------------------------------------------------------------
    def identity(self):
        return {"model": bench.model_identity(self.model), "voice": str(self.voice) if self.voice else None,
                "texts": str(Path(self.args.texts).resolve()) if self.args.texts else None, "rounds": self.args.rounds}

    def write_hparams(self, dev_manifest):
        h = json.loads(TRIAL.read_text(encoding="utf-8"))
        defaults = json.loads((paths.REPO / "dpo" / "hparams.json").read_text(encoding="utf-8"))["models"]
        cfg = dict(defaults.get(self.adapter) or defaults["orpheus"])
        if self.adapter == "tokenlm":
            cfg = {k: v for k, v in defaults["orpheus"].items() if k not in ("reward", "_reward", "_policy_init")}
            cfg["venv"], cfg["eval_generator"] = "LM", "benchmark/generate_tokenlm.py"
        cfg["policy_init"] = self.policy_init
        cfg["_policy_init"] = "pi_ref = the model given to turing-tester dpo, frozen for the whole run."
        h["models"] = {self.adapter: cfg}
        h["model"] = self.adapter
        h["rounds"] = self.args.rounds
        h["eval"] = {**h.get("eval", {}), "manifest": str(dev_manifest), "final_skip_grm": False}
        h["_turing_tester"] = ("Written by turing-tester dpo from dpo/archive/hparams_trial_01_beta0.1_adamw.json (our trial 1): "
                          "only the model, pi_ref, rounds and the dev manifest differ.")
        write_json(self.dir / "hparams.json", h)
        return h

    def dev_set(self):
        if self.voice:
            vdir = sets.prepare_voice(self.voice)
            out = self.dir / "dev" / "eval.csv"
            src = read_json(vdir / ".source.json", {}).get("key")
            if not out.exists() or read_json(self.dir / "dev" / "source.json", {}).get("key") != src:
                rows = [{**r, "set": "mirror", "item_id": r["sentence_id"], "sentence_id": f"{r['sentence_id']}__s{k}"}
                        for r in read_csv(vdir / "dev.csv") for k in range(3)]
                write_csv(out, rows)
                write_json(self.dir / "dev" / "source.json", {"key": src})
            return out
        out = SFT_DATA / "dev" / "eval_mirror.csv"
        if not out.exists():
            envs.ensure("tts")
            py("tts", "sft/ears_voices.py", "dev-mirror", "--out", SFT_DATA / "dev", log=self.log)
        return out

    def voices_file(self):
        out = self.dir / "voices.jsonl"
        if self.voice:
            vdir = paths.DATA / "voices" / self.voice.name
            clip = next(c for c in read_jsonl(vdir / "clips.jsonl") if c.get("prompt"))
            write_jsonl(out, [{"speaker": self.voice.name, "path": clip["wav"], "transcript": clip["text"]}])
        else:
            src = SFT_DATA / "ears_voices.jsonl"
            if not src.exists():
                envs.ensure("tts")
                py("tts", "sft/ears_voices.py", "voices", "--out", SFT_DATA, log=self.log)
            write_jsonl(out, read_jsonl(src))
        return out

    def texts(self, h):
        out = self.dir / "texts.jsonl"
        need = h["rounds"] * h["rollout"]["prompts_per_round"]
        if self.args.texts:
            lines = user_texts(self.args.texts)
            src = [{"text": t, "category": "yours"} for t in lines]
            what = f"{len(src)} sentences from {self.args.texts}"
        elif llm_configured():
            self.dpy("dpo/prompts.py", "texts", "--n", h["texts"]["n"], "--report", "general")
            say(f"{len(read_jsonl(out))} sentences written by the LLM")
            return
        else:
            src = read_jsonl(POOL)
            what = f"{len(src)} sentences from the bundled pool"
        rng = random.Random(h["rollout"]["seed"])
        rows, passes = [], 0
        while len(rows) < need and src:
            block = src[:]
            rng.shuffle(block)
            rows += [{"id": f"dpo_t{len(rows) + i:05d}", "text": r["text"], "category": r["category"], "stage": "dpo"}
                     for i, r in enumerate(block[: need - len(rows)])]
            passes += 1
        write_jsonl(out, rows)
        note = f"; {passes} passes, so texts repeat across rounds (with other voices)" if passes > 1 else ""
        say(f"{what} -> {len(rows)} prompts{note}")
        write_json(self.dir / "texts_summary.json", {"n": len(rows), "source": what, "passes": passes})

    # ---- evaluation ----------------------------------------------------------------------------
    def eval_policy(self, name, policy, base=None):
        d = self.dir / "eval" / name
        bench.run_benchmark(models.detect(policy), sets.BenchSet("dev", Path(self.dev_manifest), "dev set", ""), d,
                            judges=("btrm",))
        args = ["summarize", "--eval", d] + (["--base", self.dir / "eval" / base] if base else [])
        py("tts", "sft/compare.py", *args, log=self.log)

    def current_policy(self):
        cur = read_json(self.dir / "current.json")
        return abs_policy(cur["policy"]) if cur else self.policy_init

    # ---- the run -------------------------------------------------------------------------------
    def go(self):
        meta = read_json(self.dir / "turing-tester.json")
        if meta and meta.get("identity") != self.identity():
            raise Fail(f"runs/dpo/{self.name} was started with other settings; use another --name or delete it")
        write_json(self.dir / "turing-tester.json", {"identity": self.identity(), "name": self.name})
        say(f"DPO run runs/dpo/{self.name}: {self.model.label} (`{self.policy_init}`), up to {self.args.rounds} rounds")
        self.model.ensure_env()
        envs.ensure(self.env_name)
        envs.ensure("tts")
        envs.ensure("judge")
        self.dev_manifest = self.dev_set()
        h = self.write_hparams(self.dev_manifest)
        voices = self.voices_file()
        self.steps("texts", lambda: self.texts(h))
        self.steps("prompts", lambda: self.dpy("dpo/prompts.py", "build", "--voices", voices))
        self.steps("gates", lambda: self.dpy("dpo/gates.py", *self.repo_arg(), "--device", self.device))
        self.steps("eval_r00", lambda: self.eval_policy("r00", self.policy_init))
        stopped = None
        for r in range(1, h["rounds"] + 1):
            R = f"r{r:02d}"
            rd = self.dir / R
            if (rd / "round_summary.json").exists():
                s = read_json(rd / "round_summary.json")
                if s.get("stop"):
                    stopped = s["stop"]
                    break
                continue
            policy = self.current_policy()
            self.steps(f"{R}_rollout", lambda: self.dpy("dpo/rollout.py", "--round", r, "--policy", policy,
                                                        *self.repo_arg(), "--device", self.device))
            self.steps(f"{R}_metrics", lambda: py("tts", "benchmark/metrics.py", "--manifest", rd / "manifest.csv",
                                                  "--out-dir", rd, "--tts-only", log=self.log))
            self.steps(f"{R}_btrm", lambda: py("judge", "dpo/score_btrm.py", "--manifest", rd / "manifest.csv",
                                               "--out-dir", rd, "--round", r, log=self.log, env=self.env()))
            self.steps(f"{R}_pairs", lambda: py("tts", "dpo/pairs.py", "--round", r, log=self.log, env=self.env()))
            rc = self.dpy("dpo/train.py", "--round", r, "--device", self.device, check=False)
            if rc == EXIT_PAUSED:
                raise Fail(f"training paused: {paths.rel(self.dir / 'hparams.json')} is missing settings")
            if rc:
                raise Fail(f"dpo/train.py failed (exit {rc}); log: {paths.rel(self.log)}")
            self.steps(f"{R}_eval", lambda: self.eval_policy(R, self.current_policy(), "r00"))
            rc = py("tts", "dpo/rounds.py", "select", "--round", r, log=self.log, env=self.env(), check=False)
            if rc == EXIT_STOP:
                stopped = (read_json(rd / "round_summary.json") or {}).get("stop")
                break
            if rc:
                raise Fail(f"dpo/rounds.py select failed (exit {rc})")
        best = read_json(self.dir / "best.json") or {"round": 0, "policy": self.policy_init}
        final = self.dir / "final"
        improved = best["round"] > 0 and best.get("policy")
        if improved:
            src = Path(abs_policy(best["policy"]))
            from turing_tester.util import dir_fingerprint
            if not (final / ".source").exists() or (final / ".source").read_text() != dir_fingerprint(src):
                shutil.rmtree(final, ignore_errors=True)
                shutil.copytree(src, final)
                (final / ".source").write_text(dir_fingerprint(src))
        py("tts", "dpo/rounds.py", "report", log=self.log, env=self.env())
        if improved and not self.args.skip_final:
            bset = sets.prepare(str(self.voice) if self.voice else "ears")
            self.steps("final_ref", lambda: bench.run_benchmark(self.model, bset, self.dir / "bench_start"))
            self.steps("final_best", lambda: bench.run_benchmark(models.detect(str(final)), bset, self.dir / "bench_final"))
        self.write_report(best, stopped, improved)
        print()
        print((self.dir / "report.md").read_text(encoding="utf-8"))
        say(f"report: {paths.rel(self.dir / 'report.md')}")
        return 0

    def write_report(self, best, stopped, improved):
        lines = [f"# Turing Tester DPO: {self.name}", "", f"- Model: {self.model.label} (`{self.policy_init}`)",
                 f"- Settings: our trial 1 (beta 0.1 fixed, AdamW lr 1e-6, K = 4, 1,000 prompts per round), "
                 f"up to {self.args.rounds} rounds", f"- Voices: {self.voice.name if self.voice else 'EARS speakers (dev speakers held out)'}",
                 f"- Stopped: {stopped or 'all rounds done'}", ""]
        if self.model.kind != "moss_tts_nano":
            lines += ["> Our DPO experiments ran on MOSS-TTS-Nano only; this adapter passed its pre-flight gates "
                      "(gates.json) but its settings are not tuned.", ""]
        dpo_report = self.dir / "REPORT.md"
        if dpo_report.exists():
            lines += ["## Rounds (dev set)", "", *dpo_report.read_text(encoding="utf-8").splitlines()[2:], ""]
        if improved:
            lines += [f"Best: round {best['round']} -> `{paths.rel(self.dir / 'final')}`", ""]
            if (self.dir / "bench_final" / "report.json").exists():
                lines += ["## Benchmark before and after", "",
                          *report.comparison_lines(self.dir / "bench_start", self.dir / "bench_final", "start", "after DPO")]
        else:
            lines += ["No round beat the starting model on the dev set within the guard rails, so there is no final "
                      "model: keep using the one you started from.", ""]
        lines += ["In our runs DPO did not give a clear gain on top of SFT, and the two judges disagreed (RESULTS.md); "
                  "BTRM is the training reward, so check GRM closeness before trusting a BTRM gain.", "",
                  report.DISCLAIMER, ""]
        (self.dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(args):
    say(f"hardware: {hardware.require_gpu(args.allow_cpu)}")
    return Run(args).go()
