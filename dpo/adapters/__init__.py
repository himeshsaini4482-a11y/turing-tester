"""Adapter registry: hparams.json "model" -> its Adapter class (imported lazily: each needs its own venv)."""


def get(name):
    if name == "moss_nano":
        from adapters.moss_nano import MossNanoAdapter
        return MossNanoAdapter
    if name == "orpheus":
        from adapters.orpheus import OrpheusAdapter
        return OrpheusAdapter
    raise SystemExit(f"unknown model {name!r}: add an adapter in dpo/adapters/ (contract: adapters/base.py)")
