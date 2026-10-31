"""What this machine has: GPU, memory, disk. Standard library only (no torch needed)."""
import ctypes
import platform
import re
import shutil
import subprocess
import sys

from turing_tester import paths

MIN_NVIDIA_GB = 24
MIN_APPLE_GB = 32


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_gpus():
    """[{"name", "total_gb", "free_gb", "driver"}] from nvidia-smi ([] when there is none)."""
    out = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                gpus.append({"name": parts[0], "total_gb": int(parts[1]) / 1024,
                             "free_gb": int(parts[2]) / 1024, "driver": parts[3]})
            except ValueError:
                pass
    return gpus


def ram_gb():
    try:
        if sys.platform == "win32":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MemStatus()
            m.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
        if sys.platform == "darwin":
            return int(_cmd(["sysctl", "-n", "hw.memsize"]).strip() or 0) / 2**30
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        return kb / 2**20
    except Exception:  # noqa: BLE001  (reporting only)
        return 0.0


def apple_silicon():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def disk_free_gb(path=None):
    return shutil.disk_usage(str(path or paths.REPO)).free / 2**30


def accelerator():
    """("cuda", gpu dict) | ("mps", None) | (None, None), using the largest NVIDIA GPU."""
    gpus = nvidia_gpus()
    if gpus:
        return "cuda", max(gpus, key=lambda g: g["total_gb"])
    if apple_silicon():
        return "mps", None
    return None, None


def device_string():
    """cuda:0 is the largest NVIDIA GPU: util.stage_env makes it the only visible one."""
    kind, _ = accelerator()
    return {"cuda": "cuda:0", "mps": "mps"}.get(kind, "cpu")


def judge_device():
    """TURING_TESTER_DEVICE for the judge scripts."""
    return {"cuda:0": "cuda"}.get(device_string(), device_string())


def visible_gpu():
    """Index (PCI bus order) of the largest NVIDIA GPU when there are several, else None."""
    out = _cmd(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"])
    rows = []
    for line in out.strip().splitlines():
        try:
            i, mem = (int(x.strip()) for x in line.split(","))
            rows.append((mem, -i, i))
        except ValueError:
            pass
    return max(rows)[2] if len(rows) > 1 else None


def can_run_models():
    """(ok, reason): the minimum for every GPU command."""
    kind, gpu = accelerator()
    if kind == "cuda":
        if gpu["total_gb"] + 0.5 < MIN_NVIDIA_GB:
            return False, (f"{gpu['name']} has {gpu['total_gb']:.0f} GB; Turing Tester needs an NVIDIA GPU with "
                           f"{MIN_NVIDIA_GB} GB or more (the judges are 7B models run at full precision)")
        return True, f"{gpu['name']}, {gpu['total_gb']:.0f} GB"
    if kind == "mps":
        ram = ram_gb()
        if ram + 0.5 < MIN_APPLE_GB:
            return False, f"Apple Silicon with {ram:.0f} GB memory; Turing Tester needs {MIN_APPLE_GB} GB or more"
        return True, f"Apple Silicon, {ram:.0f} GB unified memory (best effort, slow)"
    return False, ("no NVIDIA GPU found (nvidia-smi did not answer) and this is not an Apple Silicon Mac; "
                   "install the NVIDIA driver, or use a machine with a supported GPU")


def require_gpu(allow_cpu=False):
    ok, why = can_run_models()
    if ok or allow_cpu:
        return why
    from turing_tester.util import Fail
    raise Fail(why + "\n  (run ./turing-tester doctor for details)")
