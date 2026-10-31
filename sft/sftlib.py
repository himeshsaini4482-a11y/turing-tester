"""Helpers shared by the SFT pipeline stages. Standard library only, so it imports in every venv.

Paths: durable outputs (everything that cost GPU time) live under runs/sft/ on the worker's EBS
disk and are copied back to the UAI server by tools/aws/gpu_run.sh. Bulky re-creatable files (raw
EARS audio, cut training clips, training mixes) live under the scratch dir on the local NVMe.

Named sftlib, not common, so it never shadows benchmark/common.py on sys.path.
"""
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
# One SFT run's outputs. The experiment pipeline (scripts/sft/) uses runs/sft itself; the turing-tester CLI
# points SFT_RUNS at runs/sft/<name>, and SFT_DATA / SFT_DEV at prepared data shared by all runs.
RUNS = Path(os.environ["SFT_RUNS"]) if os.environ.get("SFT_RUNS") else REPO / "runs" / "sft"
DATA = Path(os.environ["SFT_DATA"]) if os.environ.get("SFT_DATA") else RUNS / "data"
DEV = Path(os.environ["SFT_DEV"]) if os.environ.get("SFT_DEV") else RUNS / "dev"
BENCH_MANIFEST = REPO / "benchmark" / "manifests" / "ears_regular_26x19.csv"


def scratch_dir():
    """Local NVMe on the AWS worker (fast, wiped on stop), else a dir inside the repo."""
    if os.environ.get("SFT_SCRATCH"):
        path = Path(os.environ["SFT_SCRATCH"])
    elif Path("/opt/dlami/nvme").is_dir():
        path = Path("/opt/dlami/nvme/sft")
    else:
        path = REPO / "data" / "sft_scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- EARS conventions ---------------------------------------------------------------------

EARS_REPO = "https://github.com/facebookresearch/ears_dataset"
EARS_RAW = "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main"
PROMPT_READING = "rainbow_01_regular"   # the benchmark's voice prompt
DEV_SEED = 4321                         # picks the 4 held-out dev speakers
DEV_PER_GENDER = 2

STYLE_INSTRUCTION = {
    "regular": None,
    "fast": "Speak fast.",
    "slow": "Speak slowly.",
    "loud": "Speak loudly.",
    "whisper": "Whisper.",
    "highpitch": "Speak in a high pitch.",
    "lowpitch": "Speak in a low pitch.",
}
FREEFORM_TAG = "Speak spontaneously."   # pilot arm C-T; arm C-U leaves freeform untagged
EXCLUDED_PREFIXES = ("nonverbal_", "vegetative_", "melodic_")


def emotion_instruction(emotion):
    return None if emotion in (None, "", "neutral") else f"Speak with {emotion}."


def classify_ears(name):
    """EARS file stem -> (class, style, emotion)."""
    if name.startswith(EXCLUDED_PREFIXES):
        return "nonverbal", None, None
    if name.startswith("emo_"):
        emotion, kind = name[4:].rsplit("_", 1)
        return ("read-emotion" if kind == "sentences" else "freeform-emotion"), None, emotion
    if name.startswith("freeform_speech"):
        return "freeform", None, None
    if name.startswith("interjection_"):
        return "interjection", None, None
    m = re.fullmatch(r"(rainbow|sentences)_\d+_(\w+)", name)
    if m:
        style = m.group(2)
        return ("read-regular" if style == "regular" else "read-styled"), style, None
    return "other", None, None


def normalize_words(text):
    """Lower-case word list for WER and alignment (same rule as benchmark/metrics.py)."""
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", text).split()


# ---- weakness taxonomy (GRM diagnosis -> 9B tags -> targeted texts / EARS re-weighting) ---

WEAKNESS_TAXONOMY = {
    "flat_prosody": "monotone or flat intonation, little pitch movement",
    "unnatural_pauses": "pauses that are missing, too long, or in the wrong place",
    "pacing_too_fast": "speaking rate too fast or rushed",
    "pacing_too_slow": "speaking rate too slow or dragging",
    "sentence_final_intonation": "wrong rise/fall at the end of sentences or questions",
    "stress_emphasis": "wrong or missing word stress / emphasis",
    "mispronunciation": "mispronounced, slurred or skipped words",
    "robotic_or_metallic": "robotic, synthetic, buzzy or metallic timbre",
    "audio_artifacts_noise": "clicks, glitches, noise or other artifacts",
    "muffled_bandwidth": "muffled, dull or low-bandwidth sound",
    "breathiness_voice_quality": "unnatural breathiness, creakiness or voice quality",
    "emotion_mismatch": "emotion or expressiveness does not fit the text",
    "speaker_mismatch": "voice does not sound like the reference speaker",
    "truncation_or_omission": "cut off, truncated or missing parts of the text",
    "other": "anything else",
}

# What kind of text exercises each weakness: shown to the 9B when it writes targeted texts.
WEAKNESS_TEXT_HINT = {
    "flat_prosody": "expressive sentences with contrast, surprise or enthusiasm that need lively pitch",
    "unnatural_pauses": "long sentences with commas, clauses, parentheticals and lists",
    "pacing_too_fast": "long multi-clause sentences and dense information that need measured pacing",
    "pacing_too_slow": "short conversational sentences and quick exchanges that need natural speed",
    "sentence_final_intonation": "yes/no questions, wh-questions, tag questions and statements ending in a clear fall",
    "stress_emphasis": "sentences whose meaning depends on contrastive stress on a particular word",
    "mispronunciation": "uncommon words, proper names, place names and tricky consonant clusters (no numbers)",
    "robotic_or_metallic": "natural everyday conversational sentences of varied length",
    "audio_artifacts_noise": "natural everyday sentences of varied length",
    "muffled_bandwidth": "sentences rich in fricatives and sibilants (s, sh, f, th)",
    "breathiness_voice_quality": "longer passages with natural breathing points",
    "emotion_mismatch": "sentences that clearly carry an emotion (joy, sadness, worry, excitement)",
    "speaker_mismatch": "neutral read sentences of varied length",
    "truncation_or_omission": "long sentences of two or three clauses that must be spoken completely",
    "other": "varied natural English sentences",
}


def matches_category(text, cls, style, category):
    """Does an EARS training record exercise this weakness? Used to up-weight it in the mix."""
    words = text.split()
    if category == "sentence_final_intonation":
        return "?" in text
    if category == "unnatural_pauses":
        return text.count(",") >= 2 or len(words) >= 25
    if category in ("pacing_too_fast", "pacing_too_slow"):
        return style in ("fast", "slow") or len(words) >= 25
    if category in ("flat_prosody", "emotion_mismatch", "stress_emphasis"):
        return cls in ("read-emotion", "freeform-emotion") or "!" in text
    if category == "mispronunciation":
        return any(len(w.strip(".,!?;:\"'")) >= 10 for w in words) or any(
            w[:1].isupper() for w in words[1:])
    if category == "truncation_or_omission":
        return len(words) >= 25
    return False


# ---- files --------------------------------------------------------------------------------

def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Atomic write: a crash never leaves a half-written file that a resumed stage would trust."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_json(path, default=None):
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def log_timing(stage, seconds, **extra):
    """Append one stage's wall time to runs/sft/timings.json (the runbook re-plans from it)."""
    path = RUNS / "timings.json"
    data = read_json(path, default={"stages": []})
    data["stages"].append({"stage": stage, "seconds": round(seconds, 1),
                           "at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})
    write_json(path, data)


def log_decision(title, text):
    """Append a decision with its numbers to runs/sft/DECISIONS.md."""
    path = RUNS / "DECISIONS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} — {title}\n\n{text.strip()}\n")


# ---- BTRM % and its uncertainty -----------------------------------------------------------

SEED_SUFFIX = re.compile(r"__s\d+$")


def item_id(sentence_id):
    """Eval manifests repeat each dev item once per seed as <id>__s<k>; the item is <id>."""
    return SEED_SUFFIX.sub("", sentence_id)


def btrm_pct(rows):
    """The user's score: mean TTS reward / mean human reward x 100 over the same rows."""
    human = mean(r["human_reward"] for r in rows)
    if human <= 0:
        # BTRM rewards are unbounded; a ratio over a non-positive human mean flips every comparison
        raise ValueError(f"mean human BTRM reward is {human:.3f} <= 0; BTRM % is undefined on this set")
    return 100 * mean(r["tts_reward"] for r in rows) / human


def _groups(rows):
    groups = {}
    for r in rows:
        groups.setdefault(item_id(r["sentence_id"]), []).append(r)
    return groups


def bootstrap_pct(rows, n=10000, seed=0):
    """BTRM % with a 95% CI, resampling items (all seeds of an item move together)."""
    groups = list(_groups(rows).values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [r for g in rng.choices(groups, k=len(groups)) for r in g]
        stats.append(btrm_pct(sample))
    stats.sort()
    return {"pct": btrm_pct(rows), "lo": stats[int(0.025 * n)], "hi": stats[int(0.975 * n) - 1],
            "n_items": len(groups), "n_rows": len(rows)}


def bootstrap_diff(rows_a, rows_b, n=10000, seed=0):
    """BTRM %(a) - BTRM %(b) on the items both cover, paired by item, with a 95% CI."""
    ga, gb = _groups(rows_a), _groups(rows_b)
    common = sorted(set(ga) & set(gb))
    if not common:
        raise ValueError("no common items to compare")
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        pick = rng.choices(common, k=len(common))
        a = [r for k in pick for r in ga[k]]
        b = [r for k in pick for r in gb[k]]
        stats.append(btrm_pct(a) - btrm_pct(b))
    stats.sort()
    a_all = [r for k in common for r in ga[k]]
    b_all = [r for k in common for r in gb[k]]
    return {"diff": btrm_pct(a_all) - btrm_pct(b_all), "lo": stats[int(0.025 * n)],
            "hi": stats[int(0.975 * n) - 1], "n_items": len(common)}


# ---- import helpers -----------------------------------------------------------------------

def add_benchmark_to_path():
    """Lets sft stages reuse benchmark/*.py (load_btrm, judge, Orpheus encode/decode)."""
    path = str(REPO / "benchmark")
    if path not in sys.path:
        sys.path.insert(0, path)


def add_scripts_to_path():
    path = str(REPO / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)
