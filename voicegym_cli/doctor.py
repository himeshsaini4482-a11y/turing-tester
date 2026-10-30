"""voicegym doctor: what this machine has and which commands it can run."""
import platform
import shutil
import sys

from voicegym_cli import envs, hardware, paths

DISK_BENCHMARK_GB = 100
DISK_SFT_EARS_GB = 250
DISK_SFT_SMALL_GB = 150


def mark(ok):
    return "ok " if ok else "-- "


def main(args):
    print("VoiceGym doctor\n")
    print(f"  system      {platform.system()} {platform.release()} ({platform.machine()}), Python {sys.version.split()[0]} (uv-managed)")
    git = shutil.which("git")
    print(f"  git         {'found' if git else 'NOT FOUND (needed to download pinned code)'}")
    sox = shutil.which("sox")
    print(f"  sox         {'found' if sox else 'not found (qwen-tts warns without it; our experiment machines had it installed)'}")
    ff = shutil.which("ffmpeg")
    print(f"  ffmpeg      {'found' if ff else 'not found (only needed if an API returns mp3/ogg audio)'}")

    gpus = hardware.nvidia_gpus()
    if gpus:
        for g in gpus:
            print(f"  GPU         {g['name']}: {g['total_gb']:.1f} GB ({g['free_gb']:.1f} GB free), driver {g['driver']}")
    elif hardware.apple_silicon():
        print("  GPU         Apple Silicon (Metal)")
    else:
        print("  GPU         none found (nvidia-smi did not answer)")
    ram = hardware.ram_gb()
    disk = hardware.disk_free_gb()
    print(f"  memory      {ram:.1f} GB")
    print(f"  disk free   {disk:.0f} GB (on the drive holding {paths.REPO})")
    if any(s in str(paths.REPO).lower() for s in ("onedrive", "dropbox", "icloud", "google drive")):
        print("  WARNING     this repo is inside a synced folder: environments, judge weights and datasets (100+ GB)\n"
              "              would be synced too, and sync locks can break runs. Clone VoiceGym outside it.")

    print("\n  environments (built automatically on first use, or now with ./voicegym setup):")
    for name, ok in envs.status().items():
        print(f"    {mark(ok)}{name:<12} {envs.ENVS[name]['about']}")
    judges = {"SpeechJudge-GRM": paths.PRETRAINED / "SpeechJudge-GRM",
              "SpeechJudge-BTRM": paths.PRETRAINED / "SpeechJudge-BTRM",
              "Qwen2.5-Omni-7B (BTRM base)": paths.PRETRAINED / "Qwen2.5-Omni-7B"}
    print("\n  judge weights (downloaded on first use, ~22 GB + ~22 GB):")
    for label, d in judges.items():
        have = d.is_dir() and any(d.glob("*.safetensors"))
        print(f"    {mark(have)}{label}")

    ok, why = hardware.can_run_models()
    print("\n  what you can run:")
    if not ok:
        print(f"    -- benchmark / sft / dpo: {why}")
        print("    -- benchmark --api is blocked too: the judges always run on this machine's GPU")
        return 1
    print(f"    {mark(disk >= DISK_BENCHMARK_GB)}benchmark   ({why}; needs ~{DISK_BENCHMARK_GB} GB free disk)")
    print(f"    {mark(disk >= DISK_SFT_SMALL_GB)}sft         (teacher or your voice: ~{DISK_SFT_SMALL_GB} GB; "
          f"EARS recordings: ~{DISK_SFT_EARS_GB} GB)")
    print(f"    {mark(disk >= DISK_BENCHMARK_GB)}dpo         (~{DISK_BENCHMARK_GB} GB)")
    if not git:
        print("\n  Install git first: https://git-scm.com/downloads")
    kind, gpu = hardware.accelerator()
    if kind == "cuda" and gpu["total_gb"] < 40:
        print(f"\n  Note: with {gpu['total_gb']:.0f} GB, SpeechJudge-GRM (~22 GB of weights) partly runs from system "
              "memory. It stays at full precision, so scores are comparable, but it is slower.")
    if kind == "mps":
        print("\n  Note: Apple Silicon support is best effort; expect runs to be several times slower than on NVIDIA.")
    return 0
