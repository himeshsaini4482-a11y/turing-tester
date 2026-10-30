"""Helpers shared by the DPO stages. Standard library only, so it imports in every venv.

Paths follow the SFT conventions (sft/INTERFACES.md section 0): durable outputs (everything that
cost GPU time, plus the small reports) live under runs/dpo/ on the worker's EBS disk; bulky
re-creatable files (candidate wavs) live under the scratch dir on the local NVMe.

The file I/O, bootstrap and timing helpers are sftlib's, re-exported so the DPO stages read and
write files exactly like the SFT stages do.
"""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sft"))
import sftlib  # noqa: E402

from sftlib import (  # noqa: E402,F401  (re-exported)
    append_jsonl, bootstrap_diff, bootstrap_pct, normalize_words, read_csv, read_json, read_jsonl,
    write_csv, write_json, write_jsonl,
)

REPO = sftlib.REPO
RUNS = Path(os.environ["DPO_RUNS"]) if os.environ.get("DPO_RUNS") else REPO / "runs" / "dpo"   # tests override it
# the voicegym CLI gives each run its own copy (runs/dpo/<name>/hparams.json)
HPARAMS = Path(os.environ["DPO_HPARAMS"]) if os.environ.get("DPO_HPARAMS") else Path(__file__).resolve().parent / "hparams.json"
SFT_RUNS = sftlib.RUNS
DEV_SPEAKERS = REPO / "sft" / "data" / "dev_speakers.json"


def scratch_dir():
    """Local NVMe on the AWS worker (fast, wiped on stop), else a dir inside the repo."""
    if os.environ.get("DPO_SCRATCH"):
        path = Path(os.environ["DPO_SCRATCH"])
    elif Path("/opt/dlami/nvme").is_dir():
        path = Path("/opt/dlami/nvme/dpo")
    else:
        path = REPO / "data" / "dpo_scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_dir(r):
    return RUNS / f"r{int(r):02d}"


def ckpt_dir(r):
    """Policy after round r (round 0 = pi_ref, never written: it is hparams policy_init)."""
    return RUNS / "ckpt" / f"r{int(r):02d}"


def log_timing(stage, seconds, **extra):
    path = RUNS / "timings.json"
    t = read_json(path, {"stages": []})
    t["stages"].append({"stage": stage, "seconds": round(seconds, 1),
                        "at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})
    write_json(path, t)


def log_decision(title, text):
    """Append a dated entry to runs/dpo/DECISIONS.md (what the pipeline decided and why)."""
    path = RUNS / "DECISIONS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} {title}\n\n{text.strip()}\n")


# ---- hparams --------------------------------------------------------------------------------

def load_hparams(path=None):
    h = read_json(path or HPARAMS)
    if h is None:
        raise SystemExit(f"no hparams at {path or HPARAMS}")
    return h


def _need(h, section, key, missing, check=None):
    v = h.get(section, {}).get(key)
    if v is None:
        missing.append(f"{section}.{key}")
    elif check is not None and not check(v):
        missing.append(f"{section}.{key} (invalid: {v!r})")
    return v


def check_user_review(h):
    """List of USER_REVIEW fields that still need a (valid) value. Empty list = ready to train.

    Only the fields the chosen modes actually use are required, e.g. beta_min/beta_max/eta_beta
    only for beta.mode = dual_ascent, betas/eps only for adam(w).
    """
    m = []
    pos = lambda v: isinstance(v, (int, float)) and v > 0          # noqa: E731
    nonneg = lambda v: isinstance(v, (int, float)) and v >= 0      # noqa: E731
    posint = lambda v: isinstance(v, int) and v > 0                # noqa: E731
    mode = _need(h, "beta", "mode", m, lambda v: v in ("fixed", "dual_ascent"))
    _need(h, "beta", "beta0", m, pos)
    if mode == "dual_ascent":
        lo = _need(h, "beta", "beta_min", m, pos)
        hi = _need(h, "beta", "beta_max", m, pos)
        _need(h, "beta", "eta_beta", m, pos)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
            m.append("beta.beta_min > beta.beta_max")

    _need(h, "kl", "ema_alpha", m, lambda v: isinstance(v, (int, float)) and 0 < v < 1)
    _need(h, "kl", "k_hat_per", m, lambda v: v in ("sequence", "frame"))
    act = _need(h, "kl", "on_violation", m, lambda v: v in ("none", "stop_round", "rollback"))
    if mode == "dual_ascent" or act in ("stop_round", "rollback"):
        _need(h, "kl", "epsilon", m, pos)

    name = _need(h, "optimizer", "name", m, lambda v: v in ("adamw", "adam", "rmsprop", "sgd"))
    _need(h, "optimizer", "lr", m, pos)
    if name in ("adamw", "adam"):
        _need(h, "optimizer", "betas", m,
              lambda v: isinstance(v, list) and len(v) == 2 and all(0 <= b < 1 for b in v))
        _need(h, "optimizer", "eps", m, pos)
    if name == "rmsprop":
        _need(h, "optimizer", "alpha", m, lambda v: isinstance(v, (int, float)) and 0 < v < 1)
        _need(h, "optimizer", "eps", m, pos)
    if name == "sgd":
        _need(h, "optimizer", "momentum", m, nonneg)
    _need(h, "optimizer", "weight_decay", m, nonneg)
    _need(h, "optimizer", "max_grad_norm", m, nonneg)
    _need(h, "optimizer", "scheduler", m, lambda v: v in ("constant", "linear", "cosine"))
    _need(h, "optimizer", "warmup_steps", m, lambda v: isinstance(v, int) and v >= 0)
    _need(h, "optimizer", "batch_pairs", m, posint)
    _need(h, "optimizer", "grad_accum", m, posint)
    _need(h, "optimizer", "epochs_per_round", m, posint)
    _need(h, "optimizer", "state_across_rounds", m, lambda v: v in ("reset", "carry"))
    return m


# ---- paper diagnostics (Thm 3.2): reported only, never applied --------------------------------

def beta_star(sigma2, epsilon=None, lam=None):
    """beta* from Eq. 9: sqrt(sigma_r^2 / (2 eps)) and/or (lambda sigma_r^2 / 2)^(1/3)."""
    out = {}
    if sigma2 and sigma2 > 0:
        if epsilon:
            out["from_epsilon"] = math.sqrt(sigma2 / (2 * epsilon))
        if lam:
            out["from_lambda"] = (lam * sigma2 / 2) ** (1 / 3)
    return out


def expected_kl(sigma2, beta):
    """Eq. 14: E[KL(pi*_beta || pi_ref)] = sigma_r^2 / (2 beta^2)."""
    return sigma2 / (2 * beta * beta) if sigma2 and beta else None


def model_cfg(h):
    """(model name, models.<name> settings, rollout settings with the model's overrides merged)."""
    name = h["model"]
    cfg = h["models"][name]
    ro = {k: v for k, v in h["rollout"].items() if not k.startswith("_")}
    ro.update(cfg.get("rollout", {}))
    return name, cfg, ro


def reward_cfg(h):
    """hparams reward with the model's overrides (models.<model>.reward) merged in."""
    rw = {k: v for k, v in h["reward"].items() if not k.startswith("_")}
    rw.update(h["models"][h["model"]].get("reward", {}))
    return rw


def resolve(path_or_id):
    """A repo-relative path -> absolute; an HF model id (no such local path) is returned as is."""
    p = REPO / path_or_id
    return str(p) if p.exists() or str(path_or_id).startswith("runs/") else str(path_or_id)
