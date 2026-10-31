"""The / commands of the interactive interface: the list, closest-match ranking, and turning a typed
line into the arguments of the regular commands (cli.py). Standard library only, so it is testable
without the interface's packages; shell.py draws the screen.
"""
import shlex
from dataclasses import dataclass
from pathlib import Path

from turing_tester import paths


@dataclass(frozen=True)
class Command:
    name: str
    about: str
    usage: str = ""
    job: bool = False            # runs a long job (live status line, Ctrl+C stops it)


COMMANDS = [
    Command("doctor", "check this machine: GPU, memory, disk, what can run"),
    Command("models", "list models in models/ and APIs in apis/, and what each can do"),
    Command("voices", "list your recordings in voices/"),
    Command("benchmark", "score a model or API against real human recordings",
            "/benchmark <models/folder | HF id | apis/x.yaml> [--set ears] [--limit N] ...", job=True),
    Command("sft", "fine-tune a model on good examples",
            "/sft <models/folder> [--voice voices/x] [--teacher models/y] [--ears] ...", job=True),
    Command("dpo", "improve a model on its own best and worst samples",
            "/dpo <runs/sft/x/final> [--voice voices/x] [--rounds 3] ...", job=True),
    Command("runs", "list past runs with their headline scores"),
    Command("report", "show a run's results", "/report <run>"),
    Command("setup", "build environments now instead of on first use", "/setup [tts judge moss lm qwen3-train]",
            job=True),
    Command("help", "list the commands, or show one command's options", "/help [command]"),
    Command("clear", "clear the screen"),
    Command("exit", "leave Turing Tester (also Ctrl+D)"),
]
BY_NAME = {c.name: c for c in COMMANDS}
ALIASES = {"quit": "exit", "q": "exit", "?": "help", "h": "help", "bench": "benchmark", "cls": "clear"}

# options and values offered after a command name (the options themselves come from cli.py's parser)
CHOICES = {
    "--set": ["ears", "libritts-r", "voices/"],
    "--judges": ["grm,btrm", "grm", "btrm"],
    "--mode": ["quick", "full"],
    "--limit": ["10", "50"],
    "--rounds": ["1", "2", "3"],
}
SETUP_ENVS = ["tts", "judge", "moss", "lm", "qwen3-train"]
PATH_ROOTS = ["models/", "apis/", "voices/", "runs/", "texts/"]


# ---- closest-match ranking --------------------------------------------------------------------

def _subsequence(q, name):
    it = iter(name)
    return all(ch in it for ch in q)


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_tier(query, name):
    """0 starts with, 1 contains, 2 same letters in order, 3 a typo or two away; None = no match."""
    q = query.lower()
    if not q or name.startswith(q):
        return 0
    if q in name:
        return 1
    if _subsequence(q, name):
        return 2
    if len(q) >= 3:
        d = min(_edit_distance(q, name), _edit_distance(q, name[:len(q)]))
        if d <= (1 if len(q) <= 4 else 2):
            return 3
    return None


def rank(query, names):
    """Names matching `query`, closest first; with no query, in their given order."""
    query = query.lstrip("/").lower()
    if not query:
        return list(names)
    scored = [(t, n) for n in names if (t := match_tier(query, n)) is not None]
    return [n for _, n in sorted(scored)]


def rank_commands(query):
    return [BY_NAME[n] for n in rank(query, [c.name for c in COMMANDS])]


def resolve(word):
    """Command for a typed name: exact, an alias, or a prefix only one command has. None otherwise."""
    w = word.lstrip("/").lower()
    if w in BY_NAME:
        return BY_NAME[w]
    if w in ALIASES:
        return BY_NAME[ALIASES[w]]
    starts = [c for c in COMMANDS if c.name.startswith(w)] if w else []
    return starts[0] if len(starts) == 1 else None


def closest(word):
    """The best guess for an unknown command name, or None."""
    ranked = rank_commands(word)
    return ranked[0] if ranked else None


# ---- typed line -> regular command arguments -----------------------------------------------------

def split(line):
    """Words of a typed line. Backslashes become slashes so Windows paths survive shlex."""
    try:
        return shlex.split(line.replace("\\", "/"))
    except ValueError as e:              # an unclosed quote
        raise ValueError(f"could not read that line: {e}")


def to_argv(cmd, args):
    """Arguments typed after /benchmark, /sft or /dpo -> argv for cli.main. The first word may be
    the model folder, Hugging Face id or API config without --model / --api."""
    args = list(args)
    if cmd.name in ("benchmark", "sft", "dpo") and args and not args[0].startswith("-"):
        first = args.pop(0)
        is_api = first.lower().endswith((".yaml", ".yml"))
        if is_api and cmd.name != "benchmark":
            raise ValueError(f"/{cmd.name} trains a model, so it needs a model folder, not an API config")
        args = (["--api", first] if is_api else ["--model", first]) + args
    return [cmd.name, *args]


# ---- what is on disk ----------------------------------------------------------------------------

def model_folders():
    """Folders in models/ plus finished training runs (runs/sft/*/final, runs/dpo/*/final)."""
    found = []
    if paths.MODELS.is_dir():
        found += sorted(p for p in paths.MODELS.iterdir() if p.is_dir() and not p.name.startswith("."))
    for kind in ("sft", "dpo"):
        root = paths.RUNS / kind
        if root.is_dir():
            found += sorted(p / "final" for p in root.iterdir() if (p / "final").is_dir())
    return found


def api_configs():
    if not paths.APIS.is_dir():
        return []
    return sorted(p for p in paths.APIS.glob("*.y*ml") if p.stem != "template")


def voice_folders():
    if not paths.VOICES.is_dir():
        return []
    return sorted(p for p in paths.VOICES.iterdir() if p.is_dir() and not p.name.startswith("."))


def path_candidates(word):
    """Repo paths completing `word` (folders end in /). With nothing typed: the usual folders."""
    word = word.replace("\\", "/")
    if not word:
        return [r for r in PATH_ROOTS if (paths.REPO / r).is_dir()]
    base, slash, stem = word.rpartition("/")
    if slash and not base:                       # "/..." on Linux / macOS
        base, folder = "", Path("/")
    else:
        folder = Path(base) if Path(base).is_absolute() else paths.REPO / base
    if not folder.is_dir():
        return []
    prefix = word[:len(word) - len(stem)]
    out = []
    try:
        entries = sorted(folder.iterdir(), key=lambda x: x.name.lower())
    except OSError:
        return []
    for p in entries:
        if p.name.startswith(".") or not p.name.lower().startswith(stem.lower()):
            continue
        out.append(prefix + p.name + ("/" if p.is_dir() else ""))
    return out
