"""Stage 3: human-closeness score with SpeechJudge-GRM.

SpeechJudge compares each TTS clip with the LibriTTS human recording of the same sentence
(the TTS voice is cloned from that speaker).

Per sentence:  closeness = TTS score / human score * 100
Reported per speaker and overall as the mean of per-sentence closeness.

Each sentence is judged in both A/B orders (and optionally several votes per order);
the TTS and human scores are averaged over those calls before taking the ratio.

Writes <out-dir>/judgments.jsonl (every call, with the judge's analysis) and
<out-dir>/summary.json.

Runs in the judge environment (.venv-judge). Resumable: re-running skips calls already
in judgments.jsonl.

  python benchmark/judge.py --manifest runs/base/manifest.csv --out-dir runs/base
  python benchmark/judge.py --out-dir runs/base --summarize-only
  python benchmark/judge.py --manifest <data manifest> --out-dir runs/sanity --self-check
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from statistics import mean

from common import read_manifest, resolve_attn

SPEECHJUDGE_REPO = "RMSnow/SpeechJudge-GRM"
ORDERS = ("tts_first", "human_first")


def load_judge(speechjudge_dir, model_path, attn, thinker_only):
    # SpeechJudge's infer/ modules import each other as top-level modules
    sys.path.insert(0, os.path.join(speechjudge_dir, "infer"))
    import torch
    from huggingface_hub import snapshot_download
    from main_grm import compare_wavs
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    # Same weights and dtype as SpeechJudge's load_model(), but with a configurable attention backend
    snapshot_download(SPEECHJUDGE_REPO, local_dir=model_path)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model_cls = Qwen2_5OmniThinkerForConditionalGeneration if thinker_only else Qwen2_5OmniForConditionalGeneration
    model = model_cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=resolve_attn(attn)
    )
    model.eval()

    def judge(text, wav_a, wav_b, retries=3):
        for _ in range(retries):
            rating, analysis = compare_wavs(processor, model, text, wav_a, wav_b, is_omni=not thinker_only)
            if rating is not None:
                # extract_rating returns strings; compare as numbers ("10" < "9" as text)
                return float(rating["output_a"]), float(rating["output_b"]), analysis
        return None

    return judge


def run_judging(manifest, judge, results_path, votes):
    done = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["sentence_id"], r["order"], r["vote"]))

    total = len(manifest) * len(ORDERS) * votes
    remaining = total - len(done)
    print(f"{len(done)} judge calls already done, {remaining} to run")
    start, calls = time.time(), 0

    with open(results_path, "a", encoding="utf-8") as out:
        for row in manifest:
            for order in ORDERS:
                if order == "tts_first":
                    wav_a, wav_b = row["tts_wav"], row["human_wav"]
                else:
                    wav_a, wav_b = row["human_wav"], row["tts_wav"]
                for vote in range(votes):
                    key = (row["sentence_id"], order, vote)
                    if key in done:
                        continue
                    res = judge(row["text"], wav_a, wav_b)
                    calls += 1
                    if res is None:
                        print(f"[warn] unparseable judge output, skipped: {key}", file=sys.stderr)
                        continue
                    score_a, score_b, analysis = res
                    tts, human = (score_a, score_b) if order == "tts_first" else (score_b, score_a)
                    out.write(json.dumps({
                        "speaker_id": row["speaker_id"],
                        "sentence_id": row["sentence_id"],
                        "order": order,
                        "vote": vote,
                        "tts_score": tts,
                        "human_score": human,
                        "analysis": analysis,
                    }) + "\n")
                    out.flush()
                    if calls % 10 == 0:
                        elapsed = time.time() - start
                        eta = elapsed / calls * (remaining - calls)
                        print(f"[{calls}/{remaining}] {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                              flush=True)


def summarize(results_path):
    scores = defaultdict(lambda: {"tts": [], "human": []})
    speaker_of = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            scores[r["sentence_id"]]["tts"].append(r["tts_score"])
            scores[r["sentence_id"]]["human"].append(r["human_score"])
            speaker_of[r["sentence_id"]] = r["speaker_id"]

    per_sentence = {}
    for sid, s in scores.items():
        tts, human = mean(s["tts"]), mean(s["human"])
        if human == 0:
            print(f"[warn] human score 0, sentence excluded: {sid}", file=sys.stderr)
            continue
        per_sentence[sid] = {
            "closeness": tts / human * 100,
            "tts_score": tts,
            "human_score": human,
            "n_calls": len(s["tts"]),
        }

    by_speaker = defaultdict(list)
    for sid, s in per_sentence.items():
        by_speaker[speaker_of[sid]].append(s["closeness"])

    return {
        "overall_closeness": mean(s["closeness"] for s in per_sentence.values()),
        "n_sentences": len(per_sentence),
        "per_speaker": {
            spk: {"closeness": mean(v), "n_sentences": len(v)}
            for spk, v in sorted(by_speaker.items(), key=lambda kv: int(kv[0]))
        },
        "per_sentence": per_sentence,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--speechjudge-dir", default="third_party/SpeechJudge")
    p.add_argument("--model-path", default="pretrained/SpeechJudge-GRM")
    p.add_argument("--votes", type=int, default=1, help="judge calls per A/B order")
    p.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    p.add_argument("--thinker-only", action="store_true",
                   help="load only the Omni thinker (less GPU memory; untested with this checkpoint)")
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (smoke test)")
    p.add_argument("--self-check", action="store_true",
                   help="judge each human clip against itself; closeness should come out near 100")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "judgments.jsonl")

    if not args.summarize_only:
        if not args.manifest:
            p.error("--manifest is required unless --summarize-only")
        manifest = read_manifest(args.manifest)
        if args.limit:
            manifest = manifest[: args.limit]
        if args.self_check:
            for row in manifest:
                row["tts_wav"] = row["human_wav"]
        judge = load_judge(args.speechjudge_dir, args.model_path, args.attn, args.thinker_only)
        run_judging(manifest, judge, results_path, args.votes)

    summary = summarize(results_path)
    summary["settings"] = {"votes": args.votes, "self_check": args.self_check, "judge_model": SPEECHJUDGE_REPO}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Overall human-closeness: {summary['overall_closeness']:.1f}% ({summary['n_sentences']} sentences)")
    for spk, s in summary["per_speaker"].items():
        print(f"  speaker {spk}: {s['closeness']:.1f}% ({s['n_sentences']} sentences)")


if __name__ == "__main__":
    main()
