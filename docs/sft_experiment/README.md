# SFT experiment write-ups

The detailed records behind section 5 of [`RESULTS.md`](../../RESULTS.md), written by the
experiment scripts (`sft/report.py`, `sft/decide.py`, `sft/sftlib.py`):

- [`REPORT.md`](REPORT.md): the full SFT report.
- [`DECISIONS.md`](DECISIONS.md): every decision taken during the run, with its numbers.
- [`pilot_report.md`](pilot_report.md): the pilot that compared six data sources.

Paths such as `runs/sft/...` in them are where the scripts wrote on the GPU worker; those files
(audio, checkpoints, per-sentence scores) are not in the repo.
