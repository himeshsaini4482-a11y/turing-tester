"""turing-tester sft: fine-tune a TTS model on good examples, then benchmark it before and after.

Sources (combinable): EARS human recordings (default), your own voice (--voice), a teacher model or
API (--teacher). Quick mode = one training run with the settings of our experiments
(runs/sft/pilot/recipe.json, scripts/sft/train.sh), for MOSS-TTS-Nano:

  recipe   when                         data                                         lr     epochs
  CU       EARS only                    EARS recordings (pilot arm CU)                2e-5   2
  B        teacher (± your voice)       accepted teacher clips (pilot arm B, our      2e-5   6
                                        best SFT model)
  M        EARS + anything else         EARS + the rest at a 35% share of the hours   2e-5   3
                                        (the main-round mix)
  V        your voice only              your clips (not run in our experiments;       2e-5   6
                                        B's settings: small, clean, targeted data)
Every recipe: global batch 32, max length 640, cosine schedule with 3% warm-up, fp32 master weights
with bf16 autocast, MOSS's 50-token chunking. Other model families: sft/train_lm.py (LoRA, Orpheus
and turing-tester.yaml models), sft/train_qwen3.py (Qwen's own script, your voice only),
sft/train_custom.py (your train.py); none of these were part of our experiments.

Flow, resumable at every step (runs/sft/<name>/.done/):
  dev set -> base dev eval -> [LLM: GRM diagnosis + weakness tags] -> training sentences
  -> [--review-texts pause] -> teacher round -> data -> train -> dev eval + guard rails
  -> final/ (only if WER, speaker similarity and UTMOS stay within the guard limits)
  -> before/after benchmark (EARS 494, or your voice's test clips) -> report.md
--mode full runs the complete experiment procedure instead (scripts/sft/pilot.sh + main.sh).
"""
import csv
import json
import os
import random
import shutil
import sys
import time
import wave
from pathlib import Path

from turing_tester import bench, envs, hardware, models, paths, report, sets
from turing_tester.util import (Fail, Steps, confirm_pause, dir_fingerprint, file_key, py, read_json, run, run_name,
                               say, write_json)

SFT_DATA = paths.DATA / "sft"
EARS_DATA = SFT_DATA / "ears"            # EARS training data shared by all runs (inventory, units, chunks, codes)
SCRATCH = SFT_DATA / "scratch"           # raw EARS audio, cut chunks
POOL = paths.REPO / "turing_tester" / "data" / "sentence_pool.jsonl"
RESERVED = {"pilot", "main", "eval", "ckpt", "data", "dev", "best", "gates", "pilot_report", "REPORT"}

RECIPES = {
    "CU": {"lr": 2e-5, "epochs": 2, "about": "EARS recordings only (pilot arm CU)"},
    "B": {"lr": 2e-5, "epochs": 6, "about": "teacher clips only (pilot arm B, our best SFT model)"},
    "M": {"lr": 2e-5, "epochs": 3, "about": "EARS + other sources at a 35% share of the hours (main-round mix)"},
    "V": {"lr": 2e-5, "epochs": 6, "about": "your recordings (B's settings; not run in our experiments)"},
}
LORA_LR = 1e-4
TEACHER_SHARE = 0.35
MAX_OVERSAMPLE = 5.0


def pick_recipe(sources):
    if "ears" in sources and len(sources) > 1:
        return "M"
    if sources == {"ears"}:
        return "CU"
    if "teacher" in sources:
        return "B"
    return "V"


def wav_seconds(path):
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, OSError, EOFError):
        return 0.0


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(p.suffix + ".part"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p.with_suffix(p.suffix + ".part"), p)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def user_texts(path):
    """Sentences from a --texts file, minus any that copy a benchmark sentence (same guard as gen_texts.py)."""
    sys.path.insert(0, str(paths.REPO / "sft"))
    import gen_texts
    lines = [ln.strip() for ln in open(path, encoding="utf-8-sig") if ln.strip()]
    if not lines:
        raise Fail(f"{path} has no sentences")
    guard = gen_texts.Deduper()
    for t in {r["text"] for r in read_csv(paths.REPO / sets.EARS_MANIFEST)} |             {r["text"] for r in read_csv(paths.REPO / sets.LIBRITTS_MANIFEST)}:
        guard.add_bench(t)
    keep = [t for t in lines if not guard.is_dup(t)]
    if len(keep) < len(lines):
        say(f"[warn] left out {len(lines) - len(keep)} of your sentences: they repeat benchmark sentences, which "
            "must never be trained on")
    if not keep:
        raise Fail(f"no usable sentences in {path}")
    return keep


def llm_configured():
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import llm_client
    return llm_client.configured()


class Run:
    """One `turing-tester sft` run: runs/sft/<name>/."""

    def __init__(self, args):
        self.args = args
        self.student = models.detect(args.model)
        if not self.student.can_sft:
            raise Fail(f"{self.student.label} models cannot be trained by Turing Tester "
                       "(see README 'What can be trained'; a synthesize.py model needs a train.py)")
        self.teacher = models.detect_any(args.teacher) if args.teacher else None
        if self.teacher and self.teacher.kind == "api" and self.teacher.api.get("voice_cloning") == "none":
            raise Fail("a teacher must clone voices (its clips are filtered for speaker similarity); this API config "
                       "has voice_cloning: none")
        if self.teacher and self.teacher.kind == "api":
            from turing_tester import apiconfig
            missing = apiconfig.missing_env(self.teacher.api)
            if missing:
                raise Fail(f"the teacher API needs {', '.join(missing)} in your environment")
        self.voice = Path(args.voice).resolve() if args.voice else None
        self.sources = set()
        if self.voice:
            self.sources.add("voice")
        if self.teacher:
            self.sources.add("teacher")
        if args.ears or not self.sources:
            self.sources.add("ears")
        if self.student.kind == "qwen3_tts":
            if self.sources != {"voice"}:
                raise Fail("Qwen3-TTS can only be fine-tuned on one voice (Qwen's own fine-tuning script makes a "
                           "single-speaker model): use --voice voices/<name> alone")
            if self.student.custom_voice_speaker:
                raise Fail("this Qwen3-TTS folder is already a fine-tuned single-speaker model; start from a Base model")
        self.recipe = pick_recipe(self.sources)
        base = self.student.default_name + ("_" + self.voice.name if self.voice else "")
        self.name = run_name(args.name, base)
        if self.name in RESERVED:
            raise Fail(f"--name {self.name} is reserved (runs/sft/{self.name} belongs to the experiment pipeline)")
        self.dir = paths.RUNS / "sft" / self.name
        self.steps = Steps(self.dir)
        self.device = hardware.device_string()
        self.log = self.dir / "turing-tester.log"

    # ---- bookkeeping -------------------------------------------------------------------------
    def identity(self):
        return {"student": bench.model_identity(self.student), "teacher": bench.model_identity(self.teacher)
                if self.teacher else None, "voice": str(self.voice) if self.voice else None,
                "sources": sorted(self.sources), "recipe": self.recipe,
                "texts": str(Path(self.args.texts).resolve()) if self.args.texts else None}

    def check_identity(self):
        meta = read_json(self.dir / "turing-tester.json")
        ident = self.identity()
        if meta and meta.get("identity") != ident:
            raise Fail(f"runs/sft/{self.name} was started with other settings:\n  {meta.get('identity')}\n"
                       "  use another --name, or delete that folder to start over")
        write_json(self.dir / "turing-tester.json", {"identity": ident, "name": self.name, "started":
                   (meta or {}).get("started") or time.strftime("%Y-%m-%d %H:%M:%S")})

    def stage_env(self):
        env = {"SFT_RUNS": self.dir, "SFT_DATA": EARS_DATA, "SFT_DEV": SFT_DATA / "dev", "SFT_SCRATCH": SCRATCH}
        if self.student.kind == "moss_tts_nano" or "ears" in self.sources:
            env.update(envs.moss_env_vars())
        return env

    # ---- dev set and evaluations -----------------------------------------------------------------
    def dev_set(self):
        """EARS dev mirror (4 held-out speakers x 19 sentences x 3 seeds), or your voice's test clips x 3 seeds."""
        if self.voice:
            vdir = sets.prepare_voice(self.voice)
            out = self.dir / "dev" / "eval.csv"
            src = read_json(vdir / ".source.json", {}).get("key")
            if not out.exists() or read_json(self.dir / "dev" / "source.json", {}).get("key") != src:
                rows = []
                for r in read_csv(vdir / "dev.csv"):
                    for k in range(3):
                        rows.append({**r, "set": "voice", "item_id": r["sentence_id"],
                                     "sentence_id": f"{r['sentence_id']}__s{k}"})
                write_csv(out, rows)
                write_json(self.dir / "dev" / "source.json", {"key": src})
            return sets.BenchSet("dev:voice", out, f"held-out clips of {self.voice.name} (x3 seeds)", "")
        out = SFT_DATA / "dev" / "eval_mirror.csv"
        if not out.exists():
            envs.ensure("tts")
            py("tts", "sft/ears_voices.py", "dev-mirror", "--out", SFT_DATA / "dev", log=self.log)
        return sets.BenchSet("dev:mirror", out, "EARS dev speakers p015 p016 p046 p061 (x3 seeds)", "CC BY-NC 4.0")

    def dev_eval(self, model, label, base_label=None):
        d = self.dir / "eval" / label
        bench.run_benchmark(model, self.devset, d, judges=("btrm",))
        args = ["summarize", "--eval", d]
        if base_label:
            args += ["--base", self.dir / "eval" / base_label]
        py("tts", "sft/compare.py", *args, log=self.log)
        return read_json(d / "eval_summary.json")

    # ---- diagnosis and texts -----------------------------------------------------------------------
    def diagnose(self):
        """GRM on the base model's seed-0 dev clips, then the LLM tags the judge's critiques."""
        diag = self.dir / "diag"
        rows = [r for r in read_csv(self.dir / "eval" / "base" / "manifest.csv") if r["sentence_id"].endswith("__s0")]
        write_csv(diag / "manifest.csv", rows)
        envs.ensure("judge")
        py("judge", "benchmark/judge.py", "--manifest", diag / "manifest.csv", "--out-dir", diag, "--votes", 1,
           log=self.log, env={"TURING_TESTER_DEVICE": hardware.judge_device()})
        py("tts", "sft/tag_weaknesses.py", "--diag-dir", diag, "--eval-dir", self.dir / "eval" / "base",
           "--stage", "diag", log=self.log, env=self.stage_env())

    def make_texts(self):
        out = self.dir / "texts.jsonl"
        n = self.args.n_texts or 0
        if self.args.texts:
            lines = user_texts(self.args.texts)
            long_ = [t for t in lines if len(t.split()) > 60]
            if long_:
                say(f"[warn] {len(long_)} lines have more than 60 words; long sentences often fail the teacher's checks")
            rows = [{"id": f"teacher_t{i:05d}", "text": t, "category": "yours", "stage": "teacher"}
                    for i, t in enumerate(lines)]
            write_jsonl(out, rows)
            return f"{len(rows)} sentences from {self.args.texts}"
        if llm_configured():
            report_path = self.dir / "diag" / "weakness_report.json"
            if not report_path.exists():
                write_json(self.dir / "diag" / "general_report.json", {})
                report_path = self.dir / "diag" / "general_report.json"
            py("tts", "sft/gen_texts.py", "--report", report_path, "--out", out, "--stage", "teacher",
               "--n", n or 4000, log=self.log, env=self.stage_env())
            return f"{len(read_jsonl(out))} sentences written by the LLM (aimed at the diagnosed weaknesses)"
        pool = read_jsonl(POOL)
        if n:
            pool = random.Random(0).sample(pool, min(n, len(pool)))
        rows = [{"id": f"teacher_t{i:05d}", "text": r["text"], "category": r["category"], "stage": "teacher"}
                for i, r in enumerate(pool)]
        write_jsonl(out, rows)
        return f"{len(rows)} sentences from the bundled pool (set TURING_TESTER_LLM_* for diagnosis-targeted ones)"

    def review_texts(self):
        rows = read_jsonl(self.dir / "texts.jsonl")
        txt = self.dir / "texts.txt"
        txt.write_text("\n".join(r["text"] for r in rows) + "\n", encoding="utf-8")
        msg = []
        wr = self.dir / "diag" / "weakness_report.md"
        if wr.exists():
            msg += ["What the judge found weak in the base model:", "", wr.read_text(encoding="utf-8"), ""]
        msg += [f"The {len(rows)} training sentences are in {paths.rel(txt)} (one per line).",
                "Edit, delete or add lines there now; they are what the teacher will speak."]
        confirm_pause("\n".join(msg))
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        old = {r["text"]: r for r in rows}
        new = [{"id": f"teacher_t{i:05d}", "text": t, "category": old.get(t, {}).get("category", "yours"),
                "stage": "teacher"} for i, t in enumerate(lines)]
        write_jsonl(self.dir / "texts.jsonl", new)
        say(f"{len(new)} training sentences after your review")

    # ---- teacher round ------------------------------------------------------------------------------
    def teacher_voices(self):
        out = self.dir / "teacher" / "voices.jsonl"
        if self.voice:
            s = read_json(paths.DATA / "voices" / self.voice.name / "summary.json")
            clip = next(c for c in read_jsonl(paths.DATA / "voices" / self.voice.name / "clips.jsonl") if c.get("prompt"))
            write_jsonl(out, [{"speaker": self.voice.name, "path": clip["wav"], "transcript": s["prompt"]["text"]}])
        else:
            src = SFT_DATA / "ears_voices.jsonl"
            if not src.exists():
                envs.ensure("tts")
                py("tts", "sft/ears_voices.py", "voices", "--out", SFT_DATA, log=self.log)
            write_jsonl(out, read_jsonl(src))
        return out

    def teacher_generate(self):
        T = self.dir / "teacher"
        voices = sorted(read_jsonl(self.teacher_voices()), key=lambda v: v["speaker"])
        texts = read_jsonl(self.dir / "texts.jsonl")
        t = self.teacher
        if t.kind == "orpheus":    # our batched, as-experimented teacher
            envs.ensure("tts")
            py("tts", "sft/teacher.py", "--texts", self.dir / "texts.jsonl", "--out-dir", T, "--gpu-hours", 10000,
               "--voices", T / "voices.jsonl", "--model", t.source, "--device", self.device,
               log=self.log, env=self.stage_env())
            return
        rows = []
        for i, x in enumerate(texts):
            v = voices[i % len(voices)]
            rows.append({"speaker_id": v["speaker"], "sentence_id": x["id"], "text": x["text"], "human_wav": "",
                         "prompt_id": v["speaker"], "prompt_wav": v["path"], "prompt_text": v["transcript"]})
        write_csv(T / "gen_in.csv", rows)
        t.ensure_env()
        script, args = t.generate_cmd(T / "gen_in.csv", T / "gen", device=self.device)
        t0 = time.time()
        run([paths.env_python(t.env()), paths.REPO / script, *args], log=self.log,
            env=bench.moss_env() if t.kind == "moss_tts_nano" else None)
        per = (time.time() - t0) / max(1, len(rows))
        cat = {x["id"]: x["category"] for x in texts}
        gen = []
        for r in read_csv(T / "gen" / "manifest.csv"):
            if not os.path.exists(r["tts_wav"]):
                continue
            gen.append({"id": r["sentence_id"], "text": r["text"], "category": cat.get(r["sentence_id"], "general"),
                        "speaker": r["speaker_id"], "ref_wav": r["prompt_wav"], "ref_text": r["prompt_text"],
                        "wav": str(Path(r["tts_wav"]).resolve()), "duration": round(wav_seconds(r["tts_wav"]), 3),
                        "hit_eos": True, "gen_seconds": round(per, 3), "stage": "teacher"})
        write_jsonl(T / "generated.jsonl", gen)

    def teacher_round(self):
        T = self.dir / "teacher"
        env = self.stage_env()

        def score_manifest(src, dst, tts, human):
            pick = {"teacher": lambda g: g["wav"], "ref": lambda g: g["ref_wav"], "none": lambda g: ""}
            rows = [{"speaker_id": g["speaker"], "sentence_id": g["id"], "text": g["text"],
                     "tts_wav": pick[tts](g), "human_wav": pick[human](g), "prompt_wav": g["ref_wav"],
                     "prompt_text": g["ref_text"]} for g in read_jsonl(src)]
            write_csv(dst, rows)
            return len(rows)

        self.steps("teacher_generate", self.teacher_generate)

        def prescore():
            if not score_manifest(T / "generated.jsonl", T / "pre" / "manifest_in.csv", "teacher", "ref"):
                raise Fail("the teacher produced no clips")
            py("tts", "benchmark/metrics.py", "--manifest", T / "pre" / "manifest_in.csv", "--out-dir", T / "pre",
               "--tts-only", "--skip-utmos", log=self.log)
        self.steps("teacher_prescore", prescore)
        self.steps("teacher_prefilter", lambda: py("tts", "sft/filter_teacher.py", "--teacher-dir", T, "--prefilter",
                                                   log=self.log, env=env))

        def student():
            if not read_jsonl(T / "candidates.jsonl"):
                say("[warn] no teacher clip passed the cheap checks (see teacher/prefilter_summary.json)")
                return
            score_manifest(T / "candidates.jsonl", T / "student" / "manifest_in.csv", "none", "teacher")
            self.student.ensure_env()
            script, args = self.student.generate_cmd(T / "student" / "manifest_in.csv", T / "student", device=self.device)
            run([paths.env_python(self.student.env()), paths.REPO / script, *args], log=self.log,
                env=bench.moss_env() if self.student.kind == "moss_tts_nano" else None)
        self.steps("teacher_student", student)

        def btrm():
            if (T / "student" / "manifest.csv").exists():
                envs.ensure("judge")
                py("judge", "benchmark/btrm_score.py", "--manifest", T / "student" / "manifest.csv",
                   "--out-dir", T / "student", log=self.log,
                   env={"TURING_TESTER_DEVICE": hardware.judge_device()})
        self.steps("teacher_btrm", btrm)
        self.steps("teacher_filter", lambda: py("tts", "sft/filter_teacher.py", "--teacher-dir", T, log=self.log,
                                                env=env))
        acc = read_jsonl(T / "accepted_raw.jsonl")
        s = read_json(T / "filter_summary.json", {})
        say(f"teacher: {len(acc)} of {s.get('n_generated', '?')} clips accepted "
            f"({s.get('accepted_audio_s', 0) / 60:.0f} min); rejected {s.get('rejected')}")
        return acc

    # ---- data ------------------------------------------------------------------------------------
    def ears_data(self):
        """EARS recordings as training chunks (shared by all runs): ~100 GB download, hours of Whisper."""
        env = self.stage_env()
        st = Steps(EARS_DATA)
        envs.ensure("tts")
        st("ears_inventory", lambda: py("tts", "sft/ears_inventory.py", log=self.log, env=env))
        st("ears_align", lambda: py("tts", "sft/align.py", log=self.log, env=env))
        st("ears_build", lambda: run([env["MOSS_PY"], paths.REPO / "sft" / "build_ears.py", "--max-tokens", 50,
                                      "--moss-repo", env["MOSS_REPO"], "--model-dir", env["MOSS_MODEL_DIR"]],
                                     log=self.log, env=env))
        return EARS_DATA / "ears_raw.jsonl"

    def moss_encode(self, raw, out_dir, shard_size=2000):
        """MOSS codes for a raw records file, in resumable shards (scripts/sft/encode.sh)."""
        import hashlib
        env = self.stage_env()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        src = hashlib.md5(Path(raw).read_bytes()).hexdigest()[:16]
        if (out_dir / ".source").exists() and (out_dir / ".source").read_text().strip() != src:
            for f in out_dir.glob("codes.rank*.jsonl"):
                f.unlink()
        (out_dir / ".source").write_text(src)
        n = len(read_jsonl(raw))
        shards = max(1, (n + shard_size - 1) // shard_size)
        for k in range(shards):
            dst = out_dir / f"codes.rank{k:05d}-of-{shards:05d}.jsonl"
            if dst.exists() and dst.stat().st_size:
                continue
            tmp = out_dir / ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            run([env["MOSS_PY"], Path(env["MOSS_REPO"]) / "finetuning" / "prepare_data.py", "--codec-path",
                 env["MOSS_CODEC_DIR"], "--device", "cuda" if self.device.startswith("cuda") else self.device,
                 "--input-jsonl", raw, "--output-jsonl", tmp / "codes.jsonl", "--num-shards", shards,
                 "--shard-rank", k, "--save-shard-suffix", "--batch-size", 8], log=self.log, env=env)
            made = list(tmp.glob("codes.rank*.jsonl"))
            if not made:
                raise Fail(f"MOSS prepare_data wrote no codes for {raw}")
            os.replace(made[0], dst)
        shutil.rmtree(out_dir / ".tmp", ignore_errors=True)
        return out_dir

    def records(self):
        """{source: raw records jsonl} for this run's sources."""
        out = {}
        if "voice" in self.sources:
            out["voice"] = paths.DATA / "voices" / self.voice.name / "train.jsonl"
        if "teacher" in self.sources:
            out["teacher"] = self.dir / "teacher" / "accepted_raw.jsonl"
        if "ears" in self.sources:
            out["ears"] = EARS_DATA / "ears_raw.jsonl"
        return {k: v for k, v in out.items() if read_jsonl(v)}

    # ---- training ----------------------------------------------------------------------------------
    def train_moss(self, recs):
        env = self.stage_env()
        r = RECIPES[self.recipe]
        others = []
        for src, raw in recs.items():
            if src == "ears":
                self.steps("encode_ears", lambda: self.moss_encode(raw, EARS_DATA / "ears_codes"), key=file_key(raw))
            else:
                d = self.dir / "data" / f"{src}_codes"
                self.steps(f"encode_{src}", lambda raw=raw, d=d: self.moss_encode(raw, d), key=file_key(raw))
                others.append(str(d))
        weights = {c: 2.0 for c in (read_json(self.dir / "diag" / "weakness_report.json", {}) or {}).get("confirmed", [])}
        ears_cfg = ({"use": True, "freeform_tag": "U", "f1_include": False, "leak_probe": False, "match_teacher": None,
                     "category_weights": weights if self.recipe == "M" else {}} if "ears" in recs else {"use": False})
        cfg = {"ears": ears_cfg, "teacher": {"codes": others, "oversample": 1.0, "max_oversample": MAX_OVERSAMPLE},
               "seed": 0}
        if self.recipe == "M" and others:
            cfg["teacher_share"] = TEACHER_SHARE
        write_json(self.dir / "train" / "mix_config.json", cfg)
        mix_env = {**env, "SFT_SCRATCH": self.dir / "scratch"}
        ckpt = self.dir / "ckpt"

        def mix_and_train():
            run([env["MOSS_PY"], paths.REPO / "sft" / "make_mix.py", "--stage", "train", "--arm", "mix", "--config",
                 "@" + str(self.dir / "train" / "mix_config.json"), "--ears-codes", EARS_DATA / "ears_codes"],
                log=self.log, env=mix_env)
            mix = self.dir / "scratch" / "mixes" / "train_mix.jsonl"
            shutil.rmtree(ckpt, ignore_errors=True)
            ckpt.mkdir(parents=True)
            run([env["MOSS_ACCEL"], "launch", "--num_processes", 1, "--num_machines", 1, "--mixed_precision", "bf16",
                 "--dynamo_backend", "no", Path(env["MOSS_REPO"]) / "finetuning" / "sft.py",
                 "--model-path", env["MOSS_MODEL_DIR"] if self.student.source == envs.MOSS_MODEL else self.student.source,
                 "--codec-path", env["MOSS_CODEC_DIR"], "--train-jsonl", mix, "--output-dir", ckpt,
                 "--per-device-batch-size", 16, "--gradient-accumulation-steps", 2, "--learning-rate", r["lr"],
                 "--warmup-ratio", 0.03, "--lr-scheduler-type", "cosine", "--num-epochs", r["epochs"],
                 "--save-every-epochs", 100, "--mixed-precision", "bf16", "--fp32-master-weights", "--max-length", 640,
                 "--channelwise-loss-weight", "1,32", "--seed", 42, "--logging-steps", 20],
                log=self.dir / "train" / "train.log", env=env)
            for d in ckpt.glob("checkpoint-epoch-*"):     # only checkpoint-last is kept (pilot arms)
                shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(self.dir / "scratch", ignore_errors=True)
        self.steps("train", mix_and_train, key=json.dumps([cfg, {k: file_key(v) for k, v in recs.items()}],
                                                         sort_keys=True))
        return ckpt / "checkpoint-last"

    def ref_texts(self):
        m = {}
        for v in read_jsonl(self.dir / "teacher" / "voices.jsonl"):
            m[v["path"]] = v["transcript"]
        if self.voice:
            for c in read_jsonl(paths.DATA / "voices" / self.voice.name / "clips.jsonl"):
                m[c["wav"]] = c["text"]
        if "ears" in self.sources:
            for c in read_jsonl(EARS_DATA / "ears_raw.jsonl"):
                m[c["audio"]] = c["text"]
            for v in read_jsonl(EARS_DATA / "ears_inventory.jsonl"):
                if v.get("name") == "rainbow_01_regular":
                    m[v["path"]] = v["transcript"]
        write_json(self.dir / "train" / "ref_texts.json", m)
        return self.dir / "train" / "ref_texts.json"

    def train_other(self, recs):
        r = RECIPES[self.recipe]
        merged = self.dir / "train" / "records.jsonl"
        write_jsonl(merged, [x for raw in recs.values() for x in read_jsonl(raw)])
        ckpt = self.dir / "ckpt"
        k = self.student.kind
        if k in ("orpheus", "token_lm"):
            env_name = self.student.env() if k == "token_lm" else "lm"
            fn = lambda: py(env_name, "sft/train_lm.py", "--model", self.student.source, "--records", merged,
                            "--ref-texts", self.ref_texts(), "--out", ckpt, "--epochs", r["epochs"], "--lr", LORA_LR,
                            "--device", self.device, log=self.dir / "train" / "train.log")
            self.student.ensure_env() if k == "token_lm" else envs.ensure("lm")
        elif k == "qwen3_tts":
            s = read_json(paths.DATA / "voices" / self.voice.name / "summary.json")
            prompt = next(c for c in read_jsonl(paths.DATA / "voices" / self.voice.name / "clips.jsonl") if c.get("prompt"))
            envs.ensure("qwen3-train")
            fn = lambda: py("qwen3-train", "sft/train_qwen3.py", "--model", self.student.source, "--records", merged,
                            "--ref-audio", prompt["wav"], "--speaker", s["voice"], "--out", ckpt,
                            "--device", self.device, log=self.dir / "train" / "train.log")
        else:
            self.student.ensure_env()
            fn = lambda: run([paths.env_python(self.student.env()), paths.REPO / "sft" / "train_custom.py", "--model",
                              self.student.source, "--records", merged, "--out", ckpt, "--device", self.device],
                             log=self.dir / "train" / "train.log")
        self.steps("train", fn, key=json.dumps({k: file_key(v) for k, v in recs.items()}, sort_keys=True))
        return ckpt / "checkpoint-last"

    # ---- the whole run -----------------------------------------------------------------------------
    def go(self):
        self.check_identity()
        say(f"SFT run runs/sft/{self.name}: {self.student.label} student, sources {', '.join(sorted(self.sources))}, "
            f"recipe {self.recipe}" + (f" ({RECIPES[self.recipe]['about']})" if self.student.kind == "moss_tts_nano" else ""))
        self.devset = self.dev_set()
        dev_key = file_key(self.devset.manifest)
        if not self.steps.done("eval_base", dev_key):          # the dev set changed (e.g. new recordings)
            shutil.rmtree(self.dir / "eval", ignore_errors=True)
            self.steps.reset("eval_")
        self.steps("eval_base", lambda: self.dev_eval(self.student, "base"), key=dev_key)

        if "teacher" in self.sources:
            if llm_configured() and not self.args.texts:
                self.steps("diagnose", self.diagnose)
            self.steps("texts", lambda: say(self.make_texts()))
            if self.args.review_texts:
                self.steps("review_texts", self.review_texts)
            self.teacher_round()
        if "ears" in self.sources:
            self.steps("ears_data", self.ears_data)
        recs = self.records()
        if not recs:
            raise Fail("no training data: the teacher's clips were all rejected and there is no other source "
                       f"(see {paths.rel(self.dir / 'teacher' / 'filter_summary.json')})")
        say("training data: " + ", ".join(f"{k} {len(read_jsonl(v))} clips" for k, v in recs.items()))
        ckpt = self.train_moss(recs) if self.student.kind == "moss_tts_nano" else self.train_other(recs)
        trained = models.detect(str(ckpt))
        key = dir_fingerprint(ckpt) + dev_key
        if not self.steps.done("eval_trained", key):     # a retrained checkpoint invalidates its old scores
            shutil.rmtree(self.dir / "eval" / "trained", ignore_errors=True)
        self.steps("eval_trained", lambda: self.dev_eval(trained, "trained", "base"), key=key)
        ev = read_json(self.dir / "eval" / "trained" / "eval_summary.json", {})
        guards = ev.get("guards", {})
        passed = bool(guards.get("pass")) and not guards.get("unchecked")
        final = self.dir / "final"
        if passed:
            if not (final / ".source").exists() or (final / ".source").read_text() != dir_fingerprint(ckpt):
                shutil.rmtree(final, ignore_errors=True)
                shutil.copytree(ckpt, final)
                (final / ".source").write_text(dir_fingerprint(ckpt))
            say(f"the trained model passed the guard rails: {paths.rel(final)}")
        else:
            shutil.rmtree(final, ignore_errors=True)
            say(f"the trained model FAILED the guard rails ({guards}); it is kept in {paths.rel(ckpt)} but not "
                "promoted to final/")
        if passed and not self.args.skip_final:
            bset = sets.prepare(str(self.voice) if self.voice else "ears")
            self.steps("final_base", lambda: bench.run_benchmark(self.student, bset, self.dir / "bench_base"))
            fkey = dir_fingerprint(final)
            if not self.steps.done("final_trained", fkey):
                shutil.rmtree(self.dir / "bench_final", ignore_errors=True)
            self.steps("final_trained", lambda: bench.run_benchmark(models.detect(str(final)), bset,
                                                                    self.dir / "bench_final"), key=fkey)
        self.write_report(passed, ev)
        print()
        print((self.dir / "report.md").read_text(encoding="utf-8"))
        say(f"report: {paths.rel(self.dir / 'report.md')}")
        return 0

    def write_report(self, passed, ev):
        r = RECIPES[self.recipe]
        moss = self.student.kind == "moss_tts_nano"
        lines = [f"# Turing Tester SFT: {self.name}", "",
                 f"- Student: {self.student.label} (`{self.student.source}`)",
                 f"- Data: {', '.join(sorted(self.sources))}" + (f"; teacher {self.teacher.label} (`{self.teacher.source}`)"
                                                                if self.teacher else ""),
                 f"- Recipe: {self.recipe}, {r['about'] if moss else 'LoRA / model-specific trainer'}; "
                 f"lr {r['lr'] if moss else LORA_LR}, {r['epochs']} epochs" if self.student.kind != "qwen3_tts"
                 else "- Trainer: Qwen's sft_12hz.py defaults (lr 2e-5, 3 epochs, batch 2 x 4)", ""]
        if not moss:
            lines += ["> This model family was not part of our experiments: the training settings are not tuned.", ""]
        teacher = read_json(self.dir / "teacher" / "filter_summary.json")
        if teacher:
            lines += [f"Teacher clips: {teacher['n_accepted']} of {teacher['n_generated']} accepted "
                      f"({teacher['accepted_audio_s'] / 60:.0f} min); rejected {teacher['rejected']}.", ""]
        g = ev.get("guards", {})
        lines += ["## Dev set (used to accept or reject the checkpoint)", "",
                  *report.comparison_lines(self.dir / "eval" / "base", self.dir / "eval" / "trained", "base", "trained")]
        lines += [f"Guard rails (WER ≤ base + 0.01, similarity ≥ base − 0.02, UTMOS ≥ base − 0.10): "
                  f"{'pass' if passed else 'FAIL'} ({g.get('values')} vs base {g.get('base_values')}).", ""]
        if (self.dir / "bench_final" / "report.json").exists():
            lines += ["## Benchmark before and after", "",
                      *report.comparison_lines(self.dir / "bench_base", self.dir / "bench_final", "before", "after")]
        elif passed:
            lines += ["Benchmark before/after: skipped (--skip-final).", ""]
        else:
            lines += ["No benchmark: the checkpoint did not pass the guard rails, so there is no final model.", ""]
        lines += [report.DISCLAIMER, ""]
        (self.dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def full_mode(args):
    """The complete experiment procedure (scripts/sft/pilot.sh then main.sh), into runs/sft/<name>/."""
    if paths.WINDOWS:
        raise Fail("--mode full runs the experiment's bash pipeline: use Linux or WSL2")
    model = models.detect(args.model)
    if model.kind != "moss_tts_nano" or args.voice or (args.teacher and models.detect_any(args.teacher).kind != "orpheus"):
        raise Fail("--mode full reproduces our experiments exactly: a MOSS-TTS-Nano student, EARS, and the Orpheus "
                   "teacher (--teacher unsloth/orpheus-3b-0.1-pretrained or a folder with it); use --mode quick otherwise")
    if not llm_configured():
        raise Fail("--mode full needs an LLM for the diagnosis and the teacher's sentences: set TURING_TESTER_LLM_BASE_URL "
                   "and TURING_TESTER_LLM_MODEL (see .env.example)")
    name = run_name(args.name, model.default_name + "_full")
    rdir = paths.RUNS / "sft" / name
    for e in ("tts", "judge", "moss"):
        envs.ensure(e)
    env = {**envs.moss_env_vars(), "SFT_RUNS": rdir, "SFT_BENCH_ROOT": paths.RUNS, "SFT_BENCH_PREFIX": f"sft/{name}/bench_",
           "TTS_PY": paths.env_python("tts"), "JUDGE_PY": paths.env_python("judge"),
           "MOSS_DIR": paths.HOME / "moss", "PATH": f"{paths.HOME / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    rc = run(["bash", "scripts/sft/pilot.sh"], log=rdir / "pilot.log", env=env, check=False)
    if rc == 3:
        say(f"the pilot found no data source that beats the base model; see {paths.rel(rdir / 'pilot' / 'pilot_report.md')}")
        return 3
    if rc:
        raise Fail(f"scripts/sft/pilot.sh failed ({rc}); log: {paths.rel(rdir / 'pilot.log')}")
    run(["bash", "scripts/sft/main.sh"], log=rdir / "main.log", env=env)
    say(f"report: {paths.rel(rdir / 'REPORT.md')}; deliverable checkpoint: {paths.rel(rdir / 'best' / 'final')}")
    return 0


def main(args):
    say(f"hardware: {hardware.require_gpu(args.allow_cpu)}")
    if args.mode == "full":
        return full_mode(args)
    return Run(args).go()
