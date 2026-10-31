"""Where Turing Tester keeps things. Everything is inside the repo and gitignored."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = REPO / ".turing-tester"            # uv, Python, environments, caches
ENVS = HOME / "envs"
THIRD_PARTY = REPO / "third_party"   # pinned code checkouts (SpeechJudge, MOSS-TTS-Nano, Qwen3-TTS)
PRETRAINED = REPO / "pretrained"     # judge weights (SpeechJudge-GRM, Qwen2.5-Omni-7B + BTRM)
DATA = REPO / "data"                 # downloaded datasets and prepared voices
RUNS = REPO / "runs"
MODELS = REPO / "models"
VOICES = REPO / "voices"
APIS = REPO / "apis"

WINDOWS = sys.platform == "win32"


def env_python(name):
    """Interpreter of a Turing Tester environment (it may not exist yet)."""
    d = ENVS / name
    return d / "Scripts" / "python.exe" if WINDOWS else d / "bin" / "python"


def env_script(name, script):
    """A console script installed in an environment (e.g. accelerate)."""
    d = ENVS / name
    return d / "Scripts" / f"{script}.exe" if WINDOWS else d / "bin" / script


def uv():
    """The pinned uv the launcher installed (TURING_TESTER_UV), else one on PATH."""
    if os.environ.get("TURING_TESTER_UV"):
        return os.environ["TURING_TESTER_UV"]
    own = HOME / "bin" / ("uv.exe" if WINDOWS else "uv")
    return str(own) if own.exists() else "uv"


def rel(path):
    """Repo-relative string when inside the repo (for messages and manifests), else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)
