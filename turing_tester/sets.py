"""Benchmark sentence sets (--set): locked manifests plus their audio, or your own voice's test clips."""
from dataclasses import dataclass
from pathlib import Path

from turing_tester import envs, hardware, paths
from turing_tester.util import Fail, dir_fingerprint, py, read_json, write_json

EARS_MANIFEST = "benchmark/manifests/ears_regular_26x19.csv"
LIBRITTS_MANIFEST = "benchmark/manifests/libritts_r_test_clean_10x50.csv"


@dataclass
class BenchSet:
    name: str            # "ears" | "libritts-r" | "voice:<name>"
    manifest: Path
    description: str
    licence: str


def prepare_voice(voice_dir):
    """voices/<name> -> data/voices/<name>/ (clips, test.csv, train.jsonl). Re-done when the folder changes."""
    voice_dir = Path(voice_dir).resolve()
    if not voice_dir.is_dir():
        raise Fail(f"voice folder not found: {voice_dir}")
    out = paths.DATA / "voices" / voice_dir.name
    key = dir_fingerprint_audio(voice_dir)
    state = read_json(out / ".source.json", {})
    if state.get("key") != key or not (out / "test.csv").exists():
        envs.ensure("tts")
        py("tts", "benchmark/prepare_voice.py", "--voice", voice_dir, "--out", out,
           "--device", hardware.device_string(), log=out / "prepare.log")
        write_json(out / ".source.json", {"key": key, "voice": str(voice_dir)})
    return out


def dir_fingerprint_audio(d):
    from turing_tester.util import fingerprint
    items = []
    for f in sorted(Path(d).iterdir()):
        if f.is_file():
            st = f.stat()
            items.append((f.name, st.st_size, int(st.st_mtime)))
    return fingerprint(items)


def prepare(set_arg, limit=0):
    """Make sure the set's manifest and human audio exist; -> BenchSet."""
    s = str(set_arg).replace("\\", "/").rstrip("/")
    if s == "ears":
        envs.ensure("tts")
        py("tts", "benchmark/prepare_ears.py", "--speakers", 26, "--out", EARS_MANIFEST, "--limit", limit,
           "--data-root", "data")
        return BenchSet("ears", paths.REPO / EARS_MANIFEST,
                        "EARS: 26 speakers x 19 read sentences, studio recordings", "CC BY-NC 4.0")
    if s in ("libritts-r", "librittsr", "libritts"):
        envs.ensure("tts")
        py("tts", "benchmark/prepare_data.py", "--corpus", "libritts-r", "--speakers", 10,
           "--sentences-per-speaker", 50, "--out", LIBRITTS_MANIFEST, "--data-root", "data")
        return BenchSet("libritts-r", paths.REPO / LIBRITTS_MANIFEST,
                        "LibriTTS-R test-clean: 10 speakers x 50 sentences, audiobook recordings", "CC BY 4.0")
    p = Path(set_arg)
    if p.is_dir():
        out = prepare_voice(p)
        summary = read_json(out / "summary.json", {})
        return BenchSet(f"voice:{p.resolve().name}", out / "test.csv",
                        f"your voice {p.resolve().name}: {summary.get('n_test', '?')} held-out clips",
                        "your recordings")
    raise Fail(f"unknown --set {set_arg!r}: use ears, libritts-r, or a voices/<name> folder")


__all__ = ["BenchSet", "prepare", "prepare_voice", "dir_fingerprint"]
