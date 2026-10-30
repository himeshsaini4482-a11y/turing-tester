"""LoRA SFT for LLM-style TTS (Orpheus, or any voicegym.yaml token-lm model).

Not part of our experiments (those trained MOSS-TTS-Nano with its own finetuning code); it applies the
same SFT idea to models that are one causal LM over text + audio-codec tokens:

  input  = the model's voice-clone prompt (reference clip + its transcript + the text), as at inference
  target = the clip's codec tokens + stop token; the loss is on the target tokens only

LoRA (peft) on the attention and MLP projections, r 16, alpha 32, dropout 0: the same shape as the
Orpheus DPO adapter (dpo/hparams.json). The base weights stay frozen in bf16; LoRA weights train in
fp32 under bf16 autocast. Each saved checkpoint is the merged full model (what the benchmark loads)
plus sft_lora/ (the LoRA weights alone, for reference; already merged) and the base folder's voicegym.yaml
(and codec file).

Records: JSON lines {"audio", "text", "ref_audio"} (voicegym CLI: teacher clips, your voice, EARS chunks);
--ref-texts maps each ref_audio path to its transcript. Codec tokens are computed once and cached
in <out>/codes.jsonl (resumable).

Env: lm (requirements-dpo-orpheus.txt).

  python sft/train_lm.py --model models/orpheus --records train.jsonl --ref-texts ref_texts.json \
      --out runs/sft/x/ckpt --epochs 6 --lr 1e-4
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

LOGIT_CHUNK = 1024


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="base model folder (or HF id for Orpheus)")
    p.add_argument("--records", required=True)
    p.add_argument("--ref-texts", required=True, help="JSON {ref_audio path: transcript}")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=16, help="batch x accum = 32 sequences per step, as the MOSS SFT")
    p.add_argument("--max-len", type=int, default=4096, help="longer sequences are dropped")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

    from tokenlm import TokenLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint-last" / "config.json").exists():
        print(f"{out}/checkpoint-last exists, skipping training", flush=True)
        return
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    lm = TokenLM(args.model, args.device)
    ref_texts = json.load(open(args.ref_texts, encoding="utf-8"))

    # ---- codec tokens, cached ------------------------------------------------------------------
    cache_path = out / "codes.jsonl"
    cache = {r["id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    records = read_jsonl(args.records)
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            if r["id"] in cache:
                continue
            ref_text = ref_texts.get(r["ref_audio"])
            if ref_text is None and lm.clones():
                print(f"[warn] {r['id']}: no transcript for its reference clip; skipped", flush=True)
                continue
            prompt = lm.prompt(r["text"], r["ref_audio"], ref_text)
            target = lm.completion(r["audio"])
            row = {"id": r["id"], "prompt": prompt, "target": target}
            cache[r["id"]] = row
            f.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                print(f"  encoded {i}/{len(records)} ({time.time() - t0:.0f}s)", flush=True)
    data = [cache[r["id"]] for r in records if r["id"] in cache
            and len(cache[r["id"]]["prompt"]) + len(cache[r["id"]]["target"]) <= args.max_len]
    dropped = len(records) - len(data)
    print(f"{len(data)} training sequences ({dropped} dropped: too long or no reference transcript)", flush=True)
    if not data:
        sys.exit("no training data")
    lm._codec = None          # free the codec before the LM loads
    torch.cuda.empty_cache()

    # ---- model ---------------------------------------------------------------------------------
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for n, prm in model.named_parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data) / (args.batch * args.accum))
    total = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    inner = model.get_base_model()
    backbone, head = inner.model, inner.lm_head
    amp = dict(device_type=torch.device(args.device).type, dtype=torch.bfloat16)

    def batch_loss(batch):
        seqs = [torch.tensor(b["prompt"] + b["target"]) for b in batch]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), lm.pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        tmask = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, (s, b) in enumerate(zip(seqs, batch)):
            ids[i, :len(s)] = s
            attn[i, :len(s)] = 1
            tmask[i, len(b["prompt"]):len(s)] = True
        ids, attn, tmask = ids.to(args.device), attn.to(args.device), tmask.to(args.device)
        with torch.autocast(**amp):
            hidden = backbone(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
        sel = tmask[:, 1:]
        h = hidden[:, :-1][sel]
        tgt = ids[:, 1:][sel]

        def chunk_nll(hc, tc):
            with torch.autocast(**amp):
                logits = head(hc)
            return -logits.float().log_softmax(-1).gather(-1, tc[:, None]).squeeze(-1)
        nll = torch.cat([checkpoint(chunk_nll, h[i:i + LOGIT_CHUNK], tgt[i:i + LOGIT_CHUNK], use_reentrant=False)
                         for i in range(0, h.shape[0], LOGIT_CHUNK)])
        return nll.sum(), nll.numel()

    def save(name):
        d = out / name
        tmp = out / (name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(str(tmp / "sft_lora"))   # not adapter/: DPO would load it on top of the merged weights
        import copy
        merged = copy.deepcopy(model).merge_and_unload()
        merged = merged.to(torch.bfloat16)
        merged.config.use_cache = True
        merged.save_pretrained(str(tmp), safe_serialization=True)
        lm.tok.save_pretrained(str(tmp))
        src = Path(args.model)
        for extra in ("voicegym.yaml", lm.spec.get("codec", "")):
            if extra and (src / extra).is_file():
                shutil.copy2(src / extra, tmp / extra)
        del merged
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(order)
        # length-sorted within windows of 64, so batches have similar lengths
        windows = [sorted(order[i:i + 64], key=lambda k: len(data[k]["prompt"]) + len(data[k]["target"]))
                   for i in range(0, len(order), 64)]
        flat = [k for w in windows for k in w]
        batches = [[data[k] for k in flat[i:i + args.batch]] for i in range(0, len(flat), args.batch)]
        for g in range(0, len(batches), args.accum):
            group = batches[g:g + args.accum]
            n_tok = sum(len(b["target"]) for bb in group for b in bb)
            total_loss = 0.0
            for bb in group:
                loss_sum, _ = batch_loss(bb)
                (loss_sum / n_tok).backward()
                total_loss += float(loss_sum)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                row = {"epoch": epoch, "step": step, "loss": total_loss / n_tok, "grad_norm": float(gn),
                       "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"epoch {epoch} step {step}/{total} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                      f"({row['seconds'] / 60:.1f} min)", flush=True)
    save("checkpoint-last")     # like the MOSS pilot arms: only the last epoch is kept and evaluated
    print(f"done: {out / 'checkpoint-last'}", flush=True)


if __name__ == "__main__":
    main()
