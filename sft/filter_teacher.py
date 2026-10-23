"""Round stage R-4b: keep only teacher clips that are clean, on-voice and better than the student.

Two passes, so the expensive scoring (student generation + BTRM) only runs on clips that already
pass the cheap rules:

  --prefilter   generated.jsonl + pre/metrics.jsonl -> candidates.jsonl (clips passing no_eos,
                rate, wer, sim) and prefilter_summary.json
  (final)       + student/btrm.jsonl -> accepted_raw.jsonl (MOSS raw records + scores) and
                filter_summary.json

Joins, by id (= sentence_id in the scoring runs):
  <teacher-dir>/generated.jsonl          teacher.py: text, voice, wav, duration, hit_eos, gen_seconds
  <teacher-dir>/pre/metrics.jsonl        metrics.py --tts-only --skip-utmos on tts_wav = teacher wav,
                                         human_wav = ref wav: tts_wer = teacher WER, speaker_sim =
                                         teacher vs reference
  <teacher-dir>/student/btrm.jsonl       btrm_score.py on the candidates: human_reward = teacher,
                                         tts_reward = student

A clip is accepted only if all hold (INTERFACES §2.7); rejections are counted under the first rule
that fails, in this order (a missing score fails its rule):
  no_eos  hit_eos
  rate    8 <= len(text) / duration <= 22 characters per second (raw text, spaces included)
  wer     teacher_wer <= 0.05
  sim     sim >= 0.88 (env TEACHER_SIM_MIN overrides; the override is recorded in the summary)
  btrm    r_teacher > r_student

Env: STD (any venv). Cheap and deterministic, so each pass simply recomputes its files.

  python sft/filter_teacher.py --teacher-dir runs/sft/pilot/teacher --prefilter
  python sft/filter_teacher.py --teacher-dir runs/sft/pilot/teacher
"""
import argparse
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

RULES = ("no_eos", "rate", "wer", "sim", "btrm")
CHEAP_RULES = RULES[:-1]
RATE_MIN, RATE_MAX = 8.0, 22.0
WER_MAX = 0.05
SIM_MIN_DEFAULT = 0.88


def sim_min():
    """0.88 (PLAN §4.5), or an explicit TEACHER_SIM_MIN override set by the user."""
    raw = os.environ.get("TEACHER_SIM_MIN", "").strip()
    return float(raw) if raw else SIM_MIN_DEFAULT


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def join(generated, btrm, pre):
    """generated rows + their scores (None when missing)."""
    b = {r["sentence_id"]: r for r in btrm}
    p = {r["sentence_id"]: r for r in pre}
    rows = []
    for g in generated:
        i = g["id"]
        rows.append({**g,
                     "r_teacher": _num(b.get(i, {}).get("human_reward")),
                     "r_student": _num(b.get(i, {}).get("tts_reward")),
                     "teacher_wer": _num(p.get(i, {}).get("tts_wer")),
                     "sim": _num(p.get(i, {}).get("speaker_sim"))})
    return rows


def first_failure(r, smin=None):
    """Name of the first rule the joined row fails, or None if it is accepted."""
    smin = sim_min() if smin is None else smin
    if not r.get("hit_eos"):
        return "no_eos"
    dur = _num(r.get("duration")) or 0
    if dur <= 0 or not RATE_MIN <= len(r["text"]) / dur <= RATE_MAX:
        return "rate"
    if r["teacher_wer"] is None or r["teacher_wer"] > WER_MAX:
        return "wer"
    if r["sim"] is None or r["sim"] < smin:
        return "sim"
    if r["r_teacher"] is None or r["r_student"] is None or not r["r_teacher"] > r["r_student"]:
        return "btrm"
    return None


def passes_cheap(r, smin=None):
    return first_failure(r, smin) in (None, "btrm")


def stage_of(r):
    # ids are f"{stage}_t{index:05d}" (gen_texts.py)
    return r.get("stage") or r["id"].rsplit("_t", 1)[0]


def to_record(r):
    return {"id": r["id"], "audio": r["wav"], "text": r["text"], "ref_audio": r["ref_wav"],
            "speaker": r["speaker"], "category": r["category"], "source": "teacher", "stage": stage_of(r),
            "duration": r["duration"], "r_teacher": r["r_teacher"], "r_student": r["r_student"],
            "teacher_wer": r["teacher_wer"], "sim": r["sim"]}


def thresholds():
    smin = sim_min()
    out = {"rate": [RATE_MIN, RATE_MAX], "wer": WER_MAX, "sim": smin}
    if smin != SIM_MIN_DEFAULT:
        out["sim_override"] = f"TEACHER_SIM_MIN={smin} (default {SIM_MIN_DEFAULT})"
    return out


def summarize(rows, failed=()):
    """failed: teacher.py's failed.jsonl rows (decode errors); their GPU time still counts."""
    smin = sim_min()
    rejected = {k: 0 for k in RULES}
    accepted = []
    for r in rows:
        why = first_failure(r, smin)
        if why:
            rejected[why] += 1
        else:
            accepted.append(r)
    n = len(rows)
    acc_s = sum(r["duration"] for r in accepted)
    gen_s = sum(_num(r.get("gen_seconds")) or 0 for r in list(rows) + list(failed))
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"generated": 0, "accepted": 0})
        c["generated"] += 1
    for r in accepted:
        by_cat[r["category"]]["accepted"] += 1

    def m(key, subset):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(mean(vals), 4) if vals else None

    candidates = [r for r in rows if passes_cheap(r, smin)]
    summary = {
        "n_generated": n,
        "n_candidates": len(candidates),
        "n_accepted": len(accepted),
        "yield": round(len(accepted) / n, 4) if n else 0.0,
        "rejected": rejected,
        "accepted_audio_s": round(acc_s, 1),
        "gen_gpu_s": round(gen_s, 1),
        "accepted_audio_per_gpu_s": round(acc_s / gen_s, 5) if gen_s else 0.0,
        "mean_accepted_duration": round(acc_s / len(accepted), 3) if accepted else 0.0,
        # extras for the report / budget planning
        "generated_audio_s": round(sum(r["duration"] for r in rows), 1),
        "eos_rate": round(sum(bool(r.get("hit_eos")) for r in rows) / n, 4) if n else 0.0,
        "n_decode_failed": len(failed),
        "by_category": by_cat,
        "means_all": {k: m(k, rows) for k in ("teacher_wer", "sim", "r_teacher", "r_student")},
        "means_accepted": {k: m(k, accepted) for k in ("teacher_wer", "sim", "r_teacher", "r_student")},
        "missing_scores": {"teacher_wer": sum(r["teacher_wer"] is None for r in rows),
                           "sim": sum(r["sim"] is None for r in rows),
                           # BTRM is only run on the candidates, so only they can miss it
                           "r_teacher": sum(r["r_teacher"] is None for r in candidates)},
        "thresholds": thresholds(),
    }
    return accepted, summary


def prefilter(rows, failed=()):
    """Cheap rules only -> (candidate rows, summary)."""
    smin = sim_min()
    rejected = {k: 0 for k in CHEAP_RULES}
    cands = []
    for r in rows:
        why = first_failure(r, smin)
        if why in (None, "btrm"):
            cands.append(r)
        else:
            rejected[why] += 1
    n = len(rows)
    summary = {"n_generated": n, "n_candidates": len(cands),
               "candidate_rate": round(len(cands) / n, 4) if n else 0.0, "rejected": rejected,
               "candidate_audio_s": round(sum(r["duration"] for r in cands), 1),
               "n_decode_failed": len(failed),
               "missing_scores": {"teacher_wer": sum(r["teacher_wer"] is None for r in rows),
                                  "sim": sum(r["sim"] is None for r in rows)},
               "thresholds": thresholds()}
    return cands, summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--teacher-dir", required=True, help="runs/sft/<stage>/teacher")
    p.add_argument("--prefilter", action="store_true",
                   help="cheap rules only (no BTRM needed): write candidates.jsonl + prefilter_summary.json")
    args = p.parse_args()
    d = args.teacher_dir

    generated = sftlib.read_jsonl(os.path.join(d, "generated.jsonl"))
    if not generated:
        sys.exit(f"no generated clips in {d}/generated.jsonl")
    pre = sftlib.read_jsonl(os.path.join(d, "pre", "metrics.jsonl"))
    failed = sftlib.read_jsonl(os.path.join(d, "failed.jsonl"))
    th = thresholds()
    if "sim_override" in th:
        print(f"[note] voice-match threshold overridden: {th['sim_override']}", flush=True)

    if args.prefilter:
        rows = join(generated, [], pre)
        cands, summary = prefilter(rows, failed)
        for k, v in summary["missing_scores"].items():
            if v:
                print(f"[warn] {v} clips have no {k} (counted as rejected)", flush=True)
        # candidates.jsonl keeps the generated rows (teacher_round.sh builds the student manifest from it)
        keep = {r["id"] for r in cands}
        sftlib.write_jsonl(os.path.join(d, "candidates.jsonl"), [g for g in generated if g["id"] in keep])
        sftlib.write_json(os.path.join(d, "prefilter_summary.json"), summary)
        print(f"prefilter: {summary['n_candidates']}/{summary['n_generated']} candidates "
              f"({summary['candidate_rate']:.0%}) go to student generation + BTRM; rejected "
              f"{summary['rejected']}", flush=True)
        return

    rows = join(generated, sftlib.read_jsonl(os.path.join(d, "student", "btrm.jsonl")), pre)
    accepted, summary = summarize(rows, failed)
    for k, v in summary["missing_scores"].items():
        if v:
            print(f"[warn] {v} clips have no {k} (counted as rejected)", flush=True)
    sftlib.write_jsonl(os.path.join(d, "accepted_raw.jsonl"), [to_record(r) for r in accepted])
    sftlib.write_json(os.path.join(d, "filter_summary.json"), summary)
    print(f"accepted {summary['n_accepted']}/{summary['n_generated']} ({summary['yield']:.0%}; "
          f"{summary['n_candidates']} candidates after the cheap rules), "
          f"{summary['accepted_audio_s'] / 60:.1f} min audio; rejected {summary['rejected']}; "
          f"{summary['accepted_audio_per_gpu_s']:.3f} accepted audio s per GPU s", flush=True)
    if summary["n_generated"] and summary["yield"] < 0.2:
        print(f"[warn] yield below 20% (sim rejects {summary['rejected']['sim']}): report this, "
              f"do not lower thresholds silently (PLAN §4.5)", flush=True)


if __name__ == "__main__":
    main()
