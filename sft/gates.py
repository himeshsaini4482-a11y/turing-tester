"""Gates that must pass before any GPU hours are spent on training (PLAN.md §6 S0).

  prompt     G0a (MOSS venv): for real/synthetic reference clips and several texts, the training
             sequence built by the (patched) finetuning/dataset.py must equal, token for token,
             the prompt that inference builds (model.build_inference_input_ids), and the
             reference codes from finetuning/prepare_data.py must equal the codes inference
             computes from the same wav (batch 1 and batch 8).
  roundtrip  G0b (MOSS venv): prepare_data -> 1 sft.py step at lr 1e-12 -> the saved checkpoint
             (a) holds the same weights as the base model after the bf16 cast, (b) loads and
             generates through benchmark/generate_moss_nano.py, (c) produces the same audio as
             the base model with the same seeds (checked only if base generation is itself
             deterministic).
  qwen       G0c (any venv): the Qwen 9B endpoint answers. Non-fatal for the pipeline.

Each writes runs/sft/gates/<gate>.json and exits non-zero on failure.

  $MOSS_PY sft/gates.py prompt
  $MOSS_PY sft/gates.py roundtrip
  $TTS_PY  sft/gates.py qwen
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sftlib  # noqa: E402

GATES = sftlib.RUNS / "gates"
TEXTS = [
    "This is a short test.",
    "Do you really think it will rain tomorrow, or are you just guessing?",
]


def env_path(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set; run through scripts/sft/*.sh (source env.sh)")
    return value


def work_dir():
    path = sftlib.scratch_dir() / "gates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reference_wavs():
    """Real benchmark prompts if this machine has them, else synthetic speech-like clips."""
    real = []
    if sftlib.BENCH_MANIFEST.exists():
        for row in sftlib.read_csv(sftlib.BENCH_MANIFEST):
            path = sftlib.REPO / row["prompt_wav"]
            if path.exists() and str(path) not in real:
                real.append(str(path))
            if len(real) == 3:
                return real
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        sr, dur = 48000, 5.0 + i
        t = np.arange(int(sr * dur)) / sr
        f0 = 110 + 40 * i + 20 * np.sin(2 * np.pi * 0.7 * t)
        voice = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / sr) / k for k in range(1, 8))
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.1 * t)) ** 2
        wav = 0.08 * voice * envelope + 0.003 * rng.standard_normal(t.size)
        path = work_dir() / f"synthetic_ref_{i}.wav"
        sf.write(path, wav.astype("float32"), sr, subtype="PCM_16")
        paths.append(str(path))
    return paths


def bench_texts(n):
    rows = sftlib.read_csv(sftlib.BENCH_MANIFEST)
    return list(dict.fromkeys(r["text"] for r in rows))[:n]


def finish(name, ok, details):
    sftlib.write_json(GATES / f"{name}.json", {"gate": name, "pass": ok, "details": details,
                                               "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"{name}: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def gate_prompt(_args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    moss_repo, model_dir, codec_dir = env_path("MOSS_REPO"), env_path("MOSS_MODEL_DIR"), env_path("MOSS_CODEC_DIR")
    sys.path.insert(0, moss_repo)
    from finetuning.dataset import MossTTSNanoSFTDataset
    from finetuning.prepare_data import encode_audio_paths, load_codec
    from text_normalization_pipeline import prepare_tts_request_texts

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device=device, dtype=torch.bfloat16).eval()
    tok_inf = model._load_text_tokenizer(text_tokenizer=None, text_tokenizer_path=None)
    audio_tok = model._load_audio_tokenizer(
        audio_tokenizer=None, audio_tokenizer_type="moss-audio-tokenizer-nano",
        audio_tokenizer_pretrained_name_or_path=codec_dir, device=device,
    )
    tok_trn = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)   # as sft.py loads it
    nq = model._resolve_inference_nq(None)
    sample_rate = model._resolve_audio_tokenizer_sample_rate(audio_tok)
    channels = model._resolve_audio_tokenizer_channels(audio_tok)

    wavs = reference_wavs()
    codec = load_codec(codec_dir, device=str(device))
    codes_b1 = encode_audio_paths(codec, wavs, batch_size=1, n_vq=None)
    codes_b8 = encode_audio_paths(codec, wavs, batch_size=8, n_vq=None)

    details = {"references": wavs, "codes": [], "prompts": []}
    ok = True
    for wav in wavs:
        waveform, sr = model._load_reference_audio(wav, sample_rate, channels)
        encoded = model._call_audio_encode(audio_tokenizer=audio_tok, waveform=waveform.to(device), sample_rate=sr)
        codes_inf = model._mask_unused_audio_channels(model._normalize_audio_codes(encoded), nq=nq).cpu()
        same_b1 = torch.equal(codes_inf, torch.tensor(codes_b1[wav]))
        same_b8 = torch.equal(torch.tensor(codes_b1[wav]), torch.tensor(codes_b8[wav]))
        details["codes"].append({"wav": wav, "frames": int(codes_inf.shape[0]),
                                 "train_equals_inference": same_b1, "batch8_equals_batch1": same_b8})
        ok &= same_b1 and same_b8

        for text in TEXTS + bench_texts(5):
            norm = str(prepare_tts_request_texts(text=text, enable_wetext=False, enable_normalize_tts_text=True,
                                                 text_normalizer_manager=None)["text"])
            ids_inf, _ = model.build_inference_input_ids(
                text=norm, text_tokenizer=tok_inf, mode="voice_clone",
                prompt_audio_codes=codes_inf.to(device), device=device,
            )
            ids_inf = ids_inf[0].cpu()
            record = {"text": norm, "audio_codes": [[0] * nq] * 4, "ref_audio_codes": codes_inf.tolist()}
            example = MossTTSNanoSFTDataset([record], tokenizer=tok_trn, model_config=model.config,
                                            max_length=100000)[0]
            ids_trn = example["full_input_ids"][: int(example["prompt_length"])]
            same = ids_trn.shape == ids_inf.shape and torch.equal(ids_trn, ids_inf)
            entry = {"text": norm[:60], "train_rows": int(ids_trn.shape[0]), "inference_rows": int(ids_inf.shape[0]),
                     "identical": same}
            if not same:
                n = min(ids_trn.shape[0], ids_inf.shape[0])
                rows_differ = (ids_trn[:n] != ids_inf[:n]).any(dim=1).nonzero()
                first = int(rows_differ[0]) if len(rows_differ) else n
                entry["first_mismatch_row"] = first
                entry["train_text_ids"] = ids_trn[max(0, first - 3): first + 3, 0].tolist()
                entry["inference_text_ids"] = ids_inf[max(0, first - 3): first + 3, 0].tolist()
                print(f"  MISMATCH at row {first}: train {entry['train_text_ids']} vs inference "
                      f"{entry['inference_text_ids']}", flush=True)
            details["prompts"].append(entry)
            ok &= same
        print(f"  {wav}: codes ok={same_b1 and same_b8}", flush=True)
    finish("G0a_prompt", ok, details)


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def gate_roundtrip(_args):
    import numpy as np
    import soundfile as sf
    import torch

    moss_repo, model_dir, codec_dir = env_path("MOSS_REPO"), env_path("MOSS_MODEL_DIR"), env_path("MOSS_CODEC_DIR")
    accelerate, py = env_path("MOSS_ACCEL"), sys.executable
    work = work_dir() / "roundtrip"
    work.mkdir(exist_ok=True)
    wavs = reference_wavs()
    texts = bench_texts(8)

    raw = [{"id": f"rt{i}", "audio": wavs[i % len(wavs)], "text": texts[i],
            "ref_audio": wavs[(i + 1) % len(wavs)]} for i in range(8)]
    sftlib.write_jsonl(work / "raw.jsonl", raw)
    run([py, f"{moss_repo}/finetuning/prepare_data.py", "--codec-path", codec_dir,
         "--input-jsonl", work / "raw.jsonl", "--output-jsonl", work / "codes.jsonl"])
    ckpt_root = work / "ckpt"
    run([accelerate, "launch", "--num_processes", "1", "--num_machines", "1", "--mixed_precision", "bf16",
         "--dynamo_backend", "no", f"{moss_repo}/finetuning/sft.py",
         "--model-path", model_dir, "--codec-path", codec_dir, "--train-jsonl", work / "codes.jsonl",
         "--output-dir", ckpt_root, "--per-device-batch-size", "8", "--gradient-accumulation-steps", "1",
         "--learning-rate", "1e-12", "--weight-decay", "0", "--warmup-ratio", "0", "--num-epochs", "1",
         "--max-train-steps", "1", "--mixed-precision", "bf16", "--fp32-master-weights",
         "--max-length", "640", "--save-every-epochs", "100", "--logging-steps", "1"])
    ckpt = ckpt_root / "checkpoint-last"
    details = {"checkpoint": str(ckpt)}

    # (a) weights: identical after the bf16 cast the generator applies
    base_sd = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    ckpt_sd = torch.load(ckpt / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, extra = sorted(set(base_sd) - set(ckpt_sd)), sorted(set(ckpt_sd) - set(base_sd))
    max_diff = max((float((base_sd[k].to(torch.bfloat16).float() - ckpt_sd[k].to(torch.bfloat16).float())
                          .abs().max()) for k in base_sd if k in ckpt_sd and base_sd[k].is_floating_point()),
                   default=0.0)
    details["weights"] = {"missing_keys": missing[:10], "extra_keys": extra[:10], "max_abs_diff_bf16": max_diff,
                          "ckpt_dtype": str(next(iter(ckpt_sd.values())).dtype)}
    weights_ok = not missing and not extra and max_diff == 0.0

    # (b)/(c) generation through the benchmark generator
    rows = [{"speaker_id": "rt", "sentence_id": f"rt_{i}", "text": texts[i], "prompt_wav": wavs[0],
             "human_wav": wavs[0], "prompt_text": ""} for i in range(3)]
    sftlib.write_csv(work / "gen_manifest.csv", rows)
    outs = {}
    for name, model in (("base", model_dir), ("base_again", model_dir), ("ckpt", ckpt)):
        run_dir = work / f"gen_{name}"
        if run_dir.exists():
            subprocess.run(["rm", "-rf", str(run_dir)], check=True)
        run([py, sftlib.REPO / "benchmark" / "generate_moss_nano.py", "--manifest", work / "gen_manifest.csv",
             "--run-dir", run_dir, "--repo-dir", moss_repo, "--model", model])
        outs[name] = [sf.read(run_dir / "tts" / f"rt_{i}.wav")[0] for i in range(3)]

    def max_abs(a, b):
        return max(float(np.abs(x - y).max()) if x.shape == y.shape else float("inf") for x, y in zip(a, b))

    audible = all(len(w) > 0.5 * 48000 and float(np.abs(w).max()) > 1e-3 for w in outs["ckpt"])
    deterministic = max_abs(outs["base"], outs["base_again"]) < 1e-3
    same_as_base = max_abs(outs["base"], outs["ckpt"]) < 1e-3
    details["generation"] = {"ckpt_audible": audible, "base_deterministic": deterministic,
                             "ckpt_equals_base": same_as_base if deterministic else "not checked (base nondeterministic)"}
    ok = weights_ok and audible and (same_as_base or not deterministic)
    finish("G0b_roundtrip", ok, details)


def gate_qwen(_args):
    sftlib.add_scripts_to_path()
    from qwen9b_client import chat

    try:
        start = time.time()
        reply = chat("Reply with exactly: QWEN_OK", temperature=0.0, max_tokens=16)
        content = reply["choices"][0]["message"]["content"]
        finish("G0c_qwen", "QWEN_OK" in content, {"reply": content[:80], "seconds": round(time.time() - start, 1)})
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        finish("G0c_qwen", False, {"error": str(exc)[:300]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gate", choices=["prompt", "roundtrip", "qwen"])
    args = p.parse_args()
    {"prompt": gate_prompt, "roundtrip": gate_roundtrip, "qwen": gate_qwen}[args.gate](args)


if __name__ == "__main__":
    main()
