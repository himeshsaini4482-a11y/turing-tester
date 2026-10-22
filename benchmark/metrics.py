"""Stage 4: supporting metrics, independent of SpeechJudge.

  - WER: Whisper transcript vs target text, for TTS and human clips (human = floor)
  - Speaker similarity: WavLM x-vector cosine similarity, TTS clip vs human clip
  - UTMOS: predicted naturalness MOS (1-5), for TTS and human clips

Writes <out-dir>/metrics.jsonl (per sentence) and <out-dir>/metrics_summary.json.

Runs in the TTS environment (.venv-tts).
"""
import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean

from common import read_manifest, speaker_key

SR = 16000


def normalize(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="run manifest with tts_wav (from generate_tts.py)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--asr-model", default="openai/whisper-large-v3")
    p.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-wer", action="store_true")
    p.add_argument("--skip-sim", action="store_true")
    p.add_argument("--skip-utmos", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import jiwer
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {r["sentence_id"]: {"speaker_id": r["speaker_id"]} for r in rows}

    audio = {}

    def load(path):
        if path not in audio:
            audio[path] = librosa.load(path, sr=SR)[0]
        return audio[path]

    if not args.skip_wer:
        asr = pipeline(
            "automatic-speech-recognition",
            model=args.asr_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
        )
        for kind in ("human", "tts"):
            inputs = [{"raw": load(r[f"{kind}_wav"]), "sampling_rate": SR} for r in rows]
            outputs = asr(inputs, batch_size=args.batch_size,
                          generate_kwargs={"language": "english", "task": "transcribe"})
            for r, out in zip(rows, outputs):
                res = results[r["sentence_id"]]
                res[f"{kind}_transcript"] = out["text"].strip()
                res[f"{kind}_wer"] = jiwer.wer(normalize(r["text"]), normalize(out["text"]))
            print(f"WER done for {kind} clips", flush=True)
        del asr
        torch.cuda.empty_cache()

    if not args.skip_sim:
        extractor = AutoFeatureExtractor.from_pretrained(args.sv_model)
        sv = WavLMForXVector.from_pretrained(args.sv_model).to(device).eval()

        def embed(wav):
            inputs = extractor(wav, sampling_rate=SR, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = sv(**inputs).embeddings
            return torch.nn.functional.normalize(emb, dim=-1)[0]

        for r in rows:
            sim = embed(load(r["tts_wav"])) @ embed(load(r["human_wav"]))
            results[r["sentence_id"]]["speaker_sim"] = float(sim)
        print("Speaker similarity done", flush=True)
        del sv
        torch.cuda.empty_cache()

    if not args.skip_utmos:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        for r in rows:
            for kind in ("human", "tts"):
                wav = torch.from_numpy(load(r[f"{kind}_wav"])).unsqueeze(0).to(device)
                with torch.no_grad():
                    results[r["sentence_id"]][f"{kind}_utmos"] = float(utmos(wav, SR))
        print("UTMOS done", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for sid, res in results.items():
            f.write(json.dumps({"sentence_id": sid, **res}) + "\n")

    numeric = sorted({k for res in results.values() for k, v in res.items() if isinstance(v, float)})
    by_speaker = defaultdict(list)
    for res in results.values():
        by_speaker[res["speaker_id"]].append(res)
    summary = {
        "overall": {k: mean(res[k] for res in results.values()) for k in numeric},
        "per_speaker": {
            spk: {k: mean(res[k] for res in group) for k in numeric}
            for spk, group in sorted(by_speaker.items(), key=lambda kv: speaker_key(kv[0]))
        },
        "n_sentences": len(results),
        "models": {"asr": args.asr_model, "speaker_verification": args.sv_model, "utmos": "utmos22_strong"},
    }
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for k, v in summary["overall"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
