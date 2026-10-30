"""Subprocess, logging and resumable-step helpers shared by the commands."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from voicegym_cli import paths


class Fail(SystemExit):
    """A clean, user-facing error: printed without a traceback."""

    def __init__(self, msg):
        super().__init__(f"voicegym: {msg}")


def say(msg):
    print(f"voicegym: {msg}", flush=True)


def stage_env(extra=None):
    """Environment for stage subprocesses: unbuffered UTF-8 output, repo on sys.path, and no
    system CUDA on LD_LIBRARY_PATH (pip torch bundles its own CUDA; a system one shadows it and
    breaks cuDNN)."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "VOICEGYM_REPO": str(paths.REPO)})
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        from voicegym_cli import hardware
        gpu = _visible_gpu(hardware)
        if gpu is not None:      # several GPUs: use the largest one (cuda:0 inside the stage)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


_GPU = []


def _visible_gpu(hardware):
    if not _GPU:
        _GPU.append(hardware.visible_gpu())
    return _GPU[0]


def run(cmd, log=None, env=None, cwd=None, check=True):
    """Run cmd, streaming its output to the terminal and (appended) to `log`. Returns the exit code."""
    cmd = [str(c) for c in cmd]
    cwd = str(cwd or paths.REPO)
    fh = None
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a", encoding="utf-8")
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
    proc = None
    try:
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=stage_env(env), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise Fail(f"cannot run {cmd[0]}: not found (an environment may be incomplete; "
                       "delete .voicegym/envs/<name> to rebuild it)")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if fh:
                fh.write(line)
                fh.flush()
        rc = proc.wait()
    except KeyboardInterrupt:
        if proc:
            proc.terminate()
        raise
    finally:
        if fh:
            fh.close()
    if check and rc != 0:
        where = f" (log: {paths.rel(log)})" if log else ""
        raise Fail(f"step failed with exit code {rc}: {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd[:4]))} ...{where}")
    return rc


def py(env_name, script, *args, log=None, env=None, check=True):
    """Run a repo script (path relative to the repo) in a VoiceGym environment."""
    return run([paths.env_python(env_name), paths.REPO / script, *args], log=log, env=env, check=check)


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_key(path):
    """Content hash of a file ("" if missing): resume keys that notice changed inputs."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def dir_fingerprint(path):
    """Cheap identity of a model folder: names, sizes and mtimes of its weight/config files."""
    p = Path(path)
    if not p.is_dir():
        return str(path)
    items = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix in (".bin", ".safetensors", ".json", ".pt", ".py", ".yaml", ".model"):
            st = f.stat()
            items.append((f.relative_to(p).as_posix(), st.st_size, int(st.st_mtime)))
    return fingerprint(items)


class Steps:
    """Resumable steps: each finished step leaves <run_dir>/.done/<name> holding a fingerprint of
    its inputs. A re-run skips finished steps whose inputs did not change."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir) / ".done"

    def done(self, name, key=""):
        m = self.dir / name
        return m.exists() and m.read_text(encoding="utf-8").split("\n")[0] == str(key)

    def __call__(self, name, fn, key=""):
        if self.done(name, key):
            print(f"== [skip] {name} (done)", flush=True)
            return
        print(f"== [{time.strftime('%H:%M:%S')}] {name}", flush=True)
        t0 = time.time()
        fn()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_text(f"{key}\n{time.time() - t0:.0f}s\n", encoding="utf-8")
        print(f"== [done] {name} in {time.time() - t0:.0f}s", flush=True)

    def reset(self, prefix):
        if self.dir.is_dir():
            for m in self.dir.iterdir():
                if m.name.startswith(prefix):
                    m.unlink()


def run_name(name, default):
    """A safe run-folder name."""
    import re
    name = name or default
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "run"
    return name


def confirm_pause(message):
    """--review-texts: wait for the user (Enter continues, Ctrl+C stops)."""
    print("\n" + message, flush=True)
    try:
        input("Press Enter to continue (Ctrl+C to stop; re-run the same command to resume) ")
    except EOFError:
        raise Fail("--review-texts needs an interactive terminal")
