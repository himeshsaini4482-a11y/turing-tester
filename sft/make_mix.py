"""Build one arm's SFT training file from EARS codes and teacher codes (INTERFACES.md §2.9).

Streams records (the code lists are large), sets each record's MOSS `instruction` field for the
arm, repeats up-weighted records, and drops any record whose packed sequence would exceed
--max-length (MOSS's dataset would silently truncate it and cut off the end-of-audio token).
sft.py shuffles, so the output is not shuffled here.

Writes $SCR/mixes/<stage>_<arm>.jsonl (path printed as the last stdout line) and
runs/sft/<stage>/<arm>/mix_summary.json.

Runs in the MOSS venv (tokenizer, patched finetuning/dataset.py and the text normaliser).

  $MOSS_PY sft/make_mix.py --stage pilot --arm CU --config '{"ears": {"use": true, "freeform_tag": "U"}}'
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

DEFAULT_CONFIG = {
    "ears": {"use": False, "freeform_tag": "U", "f1_include": False, "leak_probe": False,
             "match_teacher": None, "category_weights": {}, "weight": 1.0},
    "teacher": {"codes": [], "oversample": 1.0, "max_oversample": 3.0},
    # share mode (main-round re-weighting, INTERFACES §2.9): teacher_share sets the teacher's share
    # of the mix *hours*; total_records (default: the mix's record count at the weights above)
    # fixes the record count, i.e. the optimizer steps
    "teacher_share": None,
    "total_records": None,
    "seed": 0,
}


def load_config(spec):
    raw = json.loads(open(spec[1:], encoding="utf-8").read() if spec.startswith("@") else spec)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    for key in ("ears", "teacher"):
        config[key].update(raw.get(key) or {})
    for key in ("teacher_share", "total_records", "seed"):
        if raw.get(key) is not None:
            config[key] = raw[key]
    return config


def share_weights(base, share, total_n, ears_weight, oversample, max_oversample):
    """EARS weight and teacher oversample giving the teacher `share` of the hours of a mix of
    `total_n` records.

    base = {"ears_h", "ears_n", "teacher_h", "teacher_n"}: hours and record counts at EARS weight 1
    (category weights applied) and teacher oversample 1. The record count is what is held fixed,
    since batches are a fixed number of records: equal hours would give the teacher-heavier mix
    (shorter clips) more optimizer steps. total_n defaults to the record count at
    (ears_weight, oversample) (ears_weight 0 sizes the mix by the teacher alone). If the teacher
    would need more than max_oversample repeats, it gets max_oversample, EARS fills the rest of
    the records and the achieved share falls short of the target.
    -> (ears_weight, oversample, info)
    """
    eh, en, th, tn = base["ears_h"], base["ears_n"], base["teacher_h"], base["teacher_n"]
    if total_n is None:
        total_n = en * ears_weight + tn * oversample
    ratio = share * eh / ((1 - share) * th)          # oversample / ears weight for that hour share
    weight = total_n / (en + ratio * tn)
    over = ratio * weight
    capped = over > max_oversample
    if capped:   # then max_oversample * tn < total_n, so EARS keeps a positive weight
        over = max_oversample
        weight = max(0.0, (total_n - over * tn) / en)
    hours = eh * weight + th * over
    return weight, over, {"target_share": share, "achieved_share": round(th * over / hours, 4),
                          "total_records": round(total_n), "hours": round(hours, 3),
                          "ears_weight": round(weight, 4), "teacher_oversample": round(over, 4),
                          "capped_at_max_oversample": capped,
                          "base": {k: round(v, 3) for k, v in base.items()}}


def code_files(spec):
    """A codes dir (all *.jsonl in it) or a single file."""
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "*.jsonl")))
    return [spec] if os.path.exists(spec) else []


def iter_records(files):
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def instruction_for(record, freeform_tag):
    """MOSS `instruction` prompt field for an EARS record; None (= "None") is the benchmark condition."""
    cls = record.get("class")
    if cls == "read-styled":
        return sftlib.STYLE_INSTRUCTION.get(record.get("style"))
    if cls in ("read-emotion", "freeform-emotion"):
        return sftlib.emotion_instruction(record.get("emotion"))
    if cls == "freeform" and freeform_tag == "T":
        return sftlib.FREEFORM_TAG
    return None


def ears_keep(record, ears_cfg):
    if record.get("dev_speaker"):
        return bool(ears_cfg["leak_probe"] and record.get("dev_leak_probe"))
    if record.get("f1_styled_bench") and not ears_cfg["f1_include"]:
        return False
    return True


def repeats(weight, rng):
    """Integer copies for a fractional weight: 2.5 -> 2 or 3 with equal probability."""
    whole = int(weight)
    return whole + (1 if rng.random() < weight - whole else 0)


class PackedLength:
    """Exact packed length (rows) of a record in the patched MOSS dataset, without building tensors.

    total = prefix + [audio_start] + ref frames + [audio_end] + fields(instruction) + text
            + assistant prefix (incl. its [audio_start]) + target frames + [audio_end]
    The pieces are encoded separately in the patched dataset, so the sum is exact. check()
    verifies it against the real dataset on the first records.
    """

    def __init__(self, moss_repo, model_dir):
        from transformers import AutoConfig, AutoTokenizer

        sys.path.insert(0, moss_repo)
        from finetuning import dataset as ds_module

        self.ds_module = ds_module
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.dataset = ds_module.MossTTSNanoSFTDataset([], tokenizer=self.tok, model_config=self.config,
                                                       max_length=1 << 20)
        enc = lambda s: ds_module.encode_text(self.tok, s)  # noqa: E731
        self.enc = enc
        self.prefix = 1 + len(enc(ds_module.USER_ROLE_PREFIX)) + len(enc(ds_module.USER_TEMPLATE_REFERENCE_PREFIX))
        self.assistant = (len(enc(ds_module.USER_TEMPLATE_SUFFIX)) + 1 + len(enc(ds_module.ASSISTANT_TURN_PREFIX))
                          + 1 + len(enc(ds_module.ASSISTANT_ROLE_PREFIX)) + 1)
        self._fields = {}

    def fields(self, record):
        key = tuple(record.get(f) for f, _ in self.ds_module.OPTIONAL_MESSAGE_FIELDS)
        if key not in self._fields:
            suffix = self.dataset._build_suffix_text({**record, "text": ""})
            self._fields[key] = len(self.enc(suffix))
        return self._fields[key]

    def __call__(self, record):
        return (self.prefix + 1 + len(record["ref_audio_codes"]) + 1 + self.fields(record)
                + len(self.enc(str(record["text"]))) + self.assistant + len(record["audio_codes"]) + 1)

    def check(self, record):
        example = self.dataset._build_example(record, index=0)
        real = int(example["seq_len"])
        if real != self(record):
            raise SystemExit(f"packed-length formula {self(record)} != dataset {real} for {record.get('id')}; "
                             "the MOSS dataset code changed? fix PackedLength")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--config", required=True, help="JSON string, or @path to a JSON file")
    p.add_argument("--ears-codes", default=str(sftlib.RUNS / "data" / "ears_codes"))
    p.add_argument("--max-length", type=int, default=640)
    args = p.parse_args()
    start = time.time()

    config = load_config(args.config)
    ears_cfg, teacher_cfg = config["ears"], config["teacher"]
    rng = random.Random(config["seed"])
    moss_repo, model_dir = os.environ["MOSS_REPO"], os.environ["MOSS_MODEL_DIR"]
    sys.path.insert(0, moss_repo)
    from text_normalization_pipeline import prepare_tts_request_texts

    def normalize(text):
        return str(prepare_tts_request_texts(text=text, enable_wetext=False, enable_normalize_tts_text=True,
                                             text_normalizer_manager=None)["text"])

    packed = PackedLength(moss_repo, model_dir)
    out_path = sftlib.scratch_dir() / "mixes" / f"{args.stage}_{args.arm}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "unique": 0, "seconds": 0.0, "dropped_overlength": 0,
             "by_source": Counter(), "by_class": Counter(), "seconds_by_source": defaultdict(float),
             "by_instruction": Counter(), "weighted_up": 0}
    checked = 0

    # Arm A: an EARS subset matched to the teacher's accepted speakers and total duration
    selected = None
    if ears_cfg["use"] and ears_cfg.get("match_teacher"):
        teacher_rows = sftlib.read_jsonl(ears_cfg["match_teacher"])
        speakers = {r["speaker"] for r in teacher_rows}
        target = sum(float(r["duration"]) for r in teacher_rows)
        pool = [(r["id"], float(r["duration"])) for r in iter_records(code_files(args.ears_codes))
                if ears_keep(r, ears_cfg) and r["speaker"] in speakers]
        rng.shuffle(pool)
        selected, total = set(), 0.0
        for rid, dur in pool:
            if total >= target:
                break
            selected.add(rid)
            total += dur
        stats["match_teacher"] = {"target_s": round(target, 1), "selected_s": round(total, 1),
                                  "speakers": len(speakers)}
        print(f"match_teacher: {len(selected)} EARS records, {total / 3600:.2f} h "
              f"(target {target / 3600:.2f} h, {len(speakers)} speakers)", flush=True)

    weights = {k: float(v) for k, v in (ears_cfg.get("category_weights") or {}).items()}

    def category_weight(record):
        return max([w for c, w in weights.items()
                    if sftlib.matches_category(record["text"], record.get("class"), record.get("style"), c)],
                   default=1.0)

    # Share mode: one extra pass over the codes for each source's base hours and record count
    share = None
    if config["teacher_share"] is not None:
        base = {"ears_h": 0.0, "ears_n": 0.0, "teacher_h": 0.0, "teacher_n": 0.0}
        if ears_cfg["use"]:
            for r in iter_records(code_files(args.ears_codes)):
                if ears_keep(r, ears_cfg) and (selected is None or r["id"] in selected):
                    w = category_weight(r)
                    base["ears_h"] += w * float(r.get("duration", 0)) / 3600
                    base["ears_n"] += w
        for spec in teacher_cfg.get("codes") or []:
            for r in iter_records(code_files(spec)):
                base["teacher_h"] += float(r.get("duration", 0)) / 3600
                base["teacher_n"] += 1
        if base["ears_n"] > 0 and base["teacher_n"] > 0:
            ears_cfg["weight"], teacher_cfg["oversample"], share = share_weights(
                base, float(config["teacher_share"]), config["total_records"], float(ears_cfg["weight"]),
                float(teacher_cfg["oversample"]), float(teacher_cfg["max_oversample"]))
            print(f"share mode: {json.dumps(share)}", flush=True)
        else:
            share = {"target_share": config["teacher_share"], "ignored": "one source is empty",
                     "base": {k: round(v, 3) for k, v in base.items()}}
            if float(ears_cfg["weight"]) <= 0:   # weight 0 was only the size basis
                ears_cfg["weight"] = 1.0
            print(f"[warn] share mode ignored: EARS {base['ears_n']:.0f} records, teacher "
                  f"{base['teacher_n']:.0f} records", flush=True)
    ears_weight = float(ears_cfg["weight"])
    tmp_path = out_path.with_suffix(".jsonl.part")
    with open(tmp_path, "w", encoding="utf-8") as out:

        def emit(record, copies, source):
            nonlocal checked
            if checked < 20:
                packed.check(record)
                checked += 1
            if packed(record) > args.max_length:
                stats["dropped_overlength"] += 1
                return
            line = json.dumps(record, ensure_ascii=False) + "\n"
            for _ in range(copies):
                out.write(line)
            stats["written"] += copies
            stats["unique"] += 1
            stats["seconds"] += copies * float(record.get("duration", 0))
            stats["seconds_by_source"][source] += copies * float(record.get("duration", 0))
            stats["by_source"][source] += copies
            stats["by_class"][record.get("class", source)] += copies
            stats["by_instruction"][record.get("instruction") or "None"] += copies

        if ears_cfg["use"]:
            for record in iter_records(code_files(args.ears_codes)):
                if not ears_keep(record, ears_cfg) or (selected is not None and record["id"] not in selected):
                    continue
                instruction = instruction_for(record, ears_cfg["freeform_tag"])
                record.pop("language", None)
                if instruction:
                    record["instruction"] = instruction
                else:
                    record.pop("instruction", None)
                weight = category_weight(record)
                if weight > 1.0:
                    stats["weighted_up"] += 1
                weight *= ears_weight
                emit(record, repeats(weight, random.Random(zlib.crc32(record["id"].encode()) + config["seed"])),
                     "ears")

        for spec in teacher_cfg.get("codes") or []:
            files = code_files(spec)
            if not files:
                print(f"[warn] no teacher codes at {spec}", file=sys.stderr, flush=True)
            for record in iter_records(files):
                record["text"] = normalize(record["text"])
                record.pop("instruction", None)
                record.pop("language", None)
                record["class"] = "teacher"
                copies = repeats(float(teacher_cfg.get("oversample", 1.0)),
                                 random.Random(zlib.crc32(record["id"].encode()) + config["seed"]))
                emit(record, copies, "teacher")
    os.replace(tmp_path, out_path)

    if stats["written"] == 0:
        raise SystemExit(f"mix {args.stage}/{args.arm} is empty; check the config and the codes dirs")
    summary = {
        "stage": args.stage, "arm": args.arm, "config": config, "mix": str(out_path),
        "records": stats["written"], "unique_records": stats["unique"], "hours": round(stats["seconds"] / 3600, 2),
        "hours_by_source": {k: round(v / 3600, 2) for k, v in stats["seconds_by_source"].items()},
        "records_by_source": dict(stats["by_source"]), "records_by_class": dict(stats["by_class"]),
        "records_by_instruction": dict(stats["by_instruction"]), "weighted_up_records": stats["weighted_up"],
        "dropped_overlength": stats["dropped_overlength"], "max_length": args.max_length,
        "match_teacher": stats.get("match_teacher"), "share": share,
        "teacher_share": round(stats["seconds_by_source"]["teacher"] / stats["seconds"], 4) if stats["seconds"] else None,
    }
    sftlib.write_json(sftlib.RUNS / args.stage / args.arm / "mix_summary.json", summary)
    sftlib.log_timing(f"make_mix_{args.stage}_{args.arm}", time.time() - start, records=stats["written"])
    print(f"mix {args.stage}/{args.arm}: {stats['written']} records ({stats['unique']} unique), "
          f"{summary['hours']} h, dropped {stats['dropped_overlength']} over-length", flush=True)
    print(out_path)


if __name__ == "__main__":
    main()
