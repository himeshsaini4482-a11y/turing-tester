"""Per-stage Python environments, built on demand with uv.

The stages need conflicting pins (qwen-tts wants transformers 4.57.3, SpeechJudge 4.52.3,
MOSS-TTS-Nano torch 2.7.0 + transformers 4.57.1), so each gets its own environment under
.turing-tester/envs/. An environment is (re)built only when its spec changes; ensure() is cheap
otherwise. torch comes from the PyTorch index that matches the installed NVIDIA driver
(uv --torch-backend auto), or is the CPU / Apple MPS build without an NVIDIA GPU.

Pinned code checkouts live in third_party/ (SpeechJudge, MOSS-TTS-Nano, Qwen3-TTS).
"""
import subprocess

from turing_tester import paths
from turing_tester.util import Fail, fingerprint, read_json, run, say, write_json

PYTHON = "3.12"
TORCH = "torch==2.14.1"   # what the experiments' tts/judge environments ran (runs/*/generation_config.json)

ENVS = {
    # Qwen3-TTS and Orpheus generation, data preparation, Whisper / WavLM / UTMOS metrics, API calls
    "tts": {"requirements": ["requirements-tts.txt"], "packages": [TORCH, "pyyaml==6.0.2", "accelerate"],
            "checkouts": [], "about": "TTS generation (Qwen3-TTS, Orpheus, APIs, your own models) and metrics"},
    # SpeechJudge-GRM and SpeechJudge-BTRM
    "judge": {"requirements": ["requirements-judge.txt"], "packages": [TORCH], "checkouts": ["speechjudge"],
              "about": "SpeechJudge-GRM and SpeechJudge-BTRM judges"},
    # MOSS-TTS-Nano generation, SFT (its patched finetuning code) and DPO
    "moss": {"requirements": [], "checkouts": ["moss"],
             "packages": ["torch==2.7.0", "torchaudio==2.7.0", "transformers==4.57.1", "numpy>=1.24",
                          "sentencepiece>=0.1.99", "soundfile", "accelerate>=1.0.0", "tqdm>=4.66.0",
                          "tensorboard>=2.17", "huggingface_hub<1.0"],
             "about": "MOSS-TTS-Nano generation, SFT and DPO"},
    # LoRA SFT and DPO of LLM-style TTS (Orpheus, turing-tester.yaml models)
    "lm": {"requirements": ["requirements-dpo-orpheus.txt"], "packages": ["pyyaml==6.0.2"], "checkouts": [],
           "about": "SFT and DPO of LLM-style TTS models (Orpheus, turing-tester.yaml)"},
    # Qwen3-TTS official single-speaker fine-tuning
    "qwen3-train": {"requirements": ["requirements-tts.txt"], "packages": [TORCH, "accelerate", "safetensors", "tensorboard"],
                    "checkouts": ["qwen3tts"], "about": "Qwen3-TTS fine-tuning on your own voice"},
}

CHECKOUTS = {
    "speechjudge": {"url": "https://github.com/AmphionTeam/SpeechJudge.git",
                    "commit": "1fbd1c883b03f3f7437506b421f7cf57bad78071", "dir": "SpeechJudge"},
    "moss": {"url": "https://github.com/OpenMOSS/MOSS-TTS-Nano.git",
             "commit": "8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb", "dir": "MOSS-TTS-Nano"},
    "qwen3tts": {"url": "https://github.com/QwenLM/Qwen3-TTS.git",
                 "commit": "022e286b98fbec7e1e916cb940cdf532cd9f488e", "dir": "Qwen3-TTS"},
}

# Hugging Face weights that the SFT/DPO code reads from local folders
MOSS_MODEL = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
MOSS_CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"


def checkout_dir(name):
    return paths.THIRD_PARTY / CHECKOUTS[name]["dir"]


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


# checkouts whose code Turing Tester patches (idempotent scripts), and the environment that runs the patch
PATCHES = {"moss": ("sft/patch_moss.py", "moss"), "qwen3tts": ("sft/patch_qwen3_ft.py", "qwen3-train")}


def ensure_checkout(name):
    """The pinned commit of a code checkout, patched when Turing Tester patches it. Returns its folder."""
    spec = CHECKOUTS[name]
    d = checkout_dir(name)
    changed = False
    if not (d / ".git").exists():
        say(f"downloading {spec['url']} (pinned {spec['commit'][:8]})")
        d.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "-q", spec["url"], str(d))
        changed = True
    if _git("rev-parse", "HEAD", cwd=d) != spec["commit"]:
        _git("checkout", "-q", "--", ".", cwd=d)      # drop our earlier patch so the switch is clean
        try:
            _git("checkout", "-q", spec["commit"], cwd=d)
        except subprocess.CalledProcessError:
            _git("fetch", "-q", "origin", cwd=d)
            _git("checkout", "-q", spec["commit"], cwd=d)
        changed = True
    if name in PATCHES:
        script, env = PATCHES[name]
        if changed or not (d / ".turing-tester-patched").exists():
            if paths.env_python(env).exists():   # otherwise ensure(env) patches right after building it
                run([paths.env_python(env), paths.REPO / script, "--repo-dir", d])
                (d / ".turing-tester-patched").write_text(spec["commit"])
    return d


def _spec_key(name):
    spec = ENVS[name]
    reqs = [(r, (paths.REPO / r).read_text(encoding="utf-8")) for r in spec["requirements"]]
    return fingerprint({"python": PYTHON, "reqs": reqs, "packages": spec["packages"]})


def installed(name):
    marker = read_json(paths.ENVS / name / ".turing-tester.json", {})
    return marker.get("spec") == _spec_key(name) and paths.env_python(name).exists()


def ensure(name):
    """Build (or update) environment `name` if needed and return its Python interpreter."""
    if name not in ENVS:
        raise Fail(f"unknown environment {name!r} ({', '.join(ENVS)})")
    spec = ENVS[name]
    for c in spec["checkouts"]:
        ensure_checkout(c)
    py = paths.env_python(name)
    if installed(name):
        return py
    d = paths.ENVS / name
    say(f"setting up the '{name}' environment ({spec['about']}); this happens once and can take a while")
    if not py.exists():
        run([paths.uv(), "venv", "--quiet", "--python", PYTHON, "--python-preference", "only-managed", d])
    # CUDA wheels matching the NVIDIA driver; anywhere else the plain (CPU / Apple MPS) build. uv's own
    # "auto" would pick e.g. Intel XPU wheels for an integrated Intel GPU, which Turing Tester cannot use.
    from turing_tester import hardware
    backend = "auto" if hardware.nvidia_gpus() else "cpu"
    cmd = [paths.uv(), "pip", "install", "--python", py, "--torch-backend", backend]
    for r in spec["requirements"]:
        cmd += ["-r", paths.REPO / r]
    cmd += spec["packages"]
    run(cmd, log=d / "install.log")
    for c in spec["checkouts"]:
        if c in PATCHES and PATCHES[c][1] == name:
            run([py, paths.REPO / PATCHES[c][0], "--repo-dir", checkout_dir(c)], log=d / "install.log")
            (checkout_dir(c) / ".turing-tester-patched").write_text(CHECKOUTS[c]["commit"])
    write_json(d / ".turing-tester.json", {"spec": _spec_key(name), "torch_backend": backend})
    return py


def ensure_moss_weights():
    """Base MOSS-TTS-Nano model and codec in local folders (the finetuning code reads paths)."""
    py = ensure("moss")
    model_dir = paths.PRETRAINED / "MOSS-TTS-Nano-100M"
    codec_dir = paths.PRETRAINED / "MOSS-Audio-Tokenizer-Nano"
    if not (model_dir / ".complete").exists() or not (codec_dir / ".complete").exists():
        code = ("from huggingface_hub import snapshot_download\n"
                "import sys, pathlib\n"
                "for repo, path in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):\n"
                "    snapshot_download(repo, local_dir=path)\n"
                "    pathlib.Path(path, '.complete').write_text('ok')\n")
        run([py, "-c", code, MOSS_MODEL, model_dir, MOSS_CODEC, codec_dir])
    return model_dir, codec_dir


def moss_env_vars():
    """Variables the SFT/DPO stages read to find MOSS (see scripts/sft/env.sh)."""
    model_dir, codec_dir = ensure_moss_weights()
    return {"MOSS_REPO": checkout_dir("moss"), "MOSS_PY": paths.env_python("moss"),
            "MOSS_ACCEL": paths.env_script("moss", "accelerate"),
            "MOSS_MODEL_DIR": model_dir, "MOSS_CODEC_DIR": codec_dir}


def status():
    return {name: installed(name) for name in ENVS}
