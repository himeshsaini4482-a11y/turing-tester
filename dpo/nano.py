"""MOSS-TTS-Nano pieces shared by rollout.py, train.py and gates.py (torch; DPO venv).

- load_policy / load_codec / text helpers: the same loading and text normalisation as
  benchmark/generate_moss_nano.py, so DPO samples come from the decoder the benchmark uses.
- sequence_logps: log pi(y|x) of a completion, teacher-forced. It mirrors finetuning/sft.py
  compute_supervised_loss (global pass -> 17-position local transformer -> 1 text head + 16 audio
  heads) with three differences: the local pass runs on completion rows only, the stop/continue
  channel is a 2-way softmax over {slot, end} (the only two tokens the sampler can emit), and the
  result is a per-sequence sum instead of a mean cross-entropy. gates.py checks it against an
  independent frame-by-frame recomputation that follows the generation loop.
- collate: prompt rows + completion rows -> [2B, L, 17], right-padded like finetuning/dataset.py.
- save_checkpoint: a dir that from_pretrained(trust_remote_code=True), scripts/sft/eval_ckpt.sh
  and scripts/run_moss_nano.sh load like an SFT checkpoint.

Uses private methods of the pinned MOSS-TTS-Nano model code (commit 44502f8 of the HF repo, the
one every SFT checkpoint copies); re-check them if the model code is ever updated.
"""
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# benchmark/generate_moss_nano.py's SAMPLING (= upstream infer.py defaults)
DEPLOYMENT_SAMPLING = {
    "text_temperature": 1.0, "text_top_p": 1.0, "text_top_k": 50,
    "audio_temperature": 0.8, "audio_top_p": 0.95, "audio_top_k": 25, "audio_repetition_penalty": 1.2,
}
# exact samples from pi_theta at temperature 1 (what sequence_logps scores)
ON_POLICY_SAMPLING = {
    "text_temperature": 1.0, "text_top_p": 1.0, "text_top_k": 0,
    "audio_temperature": 1.0, "audio_top_p": 1.0, "audio_top_k": 0, "audio_repetition_penalty": 1.0,
}
CODEC_TYPE = "moss-audio-tokenizer-nano"


def sampling_kwargs(name):
    return {"deployment": DEPLOYMENT_SAMPLING, "on_policy": ON_POLICY_SAMPLING}[name]


def fast_repetition_penalty(self, logits, previous_token_ids, repetition_penalty):
    """Drop-in for MossTTSNanoForCausalLM._apply_repetition_penalty without its Python loop over
    rows (per row a mask index, torch.unique and a scatter, all host syncs: 64 rows x 16 codebooks
    per frame at batch 64). Same elementwise ops on the same values, so samples are bit-identical;
    gates.py D7 checks that against the original on the GPU before any rollout uses it."""
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if repetition_penalty == 1.0 or previous_token_ids is None:
        return logits
    ids = torch.as_tensor(previous_token_ids, device=logits.device, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    elif ids.ndim > 2:
        ids = ids.reshape(ids.shape[0], -1)
    vocab = logits.shape[-1]
    ids = torch.where((ids >= 0) & (ids < vocab), ids, torch.full_like(ids, vocab))   # pad -> dummy column
    seen = torch.zeros((ids.shape[0], vocab + 1), dtype=torch.bool, device=logits.device)
    seen = seen.scatter_(1, ids, True)[:, :vocab]
    penalised = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
    return torch.where(seen, penalised, logits)


def use_fast_sampler(model, on=True):
    """Install (or undo) fast_repetition_penalty on the model's class; returns the previous one."""
    cls = type(model)
    if not hasattr(cls, "_original_repetition_penalty"):
        cls._original_repetition_penalty = cls._apply_repetition_penalty
    prev = cls._apply_repetition_penalty
    cls._apply_repetition_penalty = fast_repetition_penalty if on else cls._original_repetition_penalty
    return prev


def _restore_hf_modules_cache():
    """MOSS's remote code points HF_MODULES_CACHE at <its module dir>/.cache/huggingface/modules when
    it loads its tokenizer (_patch_hf_dynamic_module_cache_dir). That dir is not on sys.path, so the
    next trust_remote_code load in the same process (pi_ref + policy, gate D5's reload) fails with
    "No module named 'transformers_modules.<dir>'". Point it back at the importable default."""
    import transformers.dynamic_module_utils as dmu
    from transformers.utils import HF_MODULES_CACHE   # set at import, before MOSS can patch anything
    default = os.environ.get("VG_HF_MODULES_CACHE") or HF_MODULES_CACHE
    os.environ["VG_HF_MODULES_CACHE"] = default
    os.environ["HF_MODULES_CACHE"] = default
    dmu.HF_MODULES_CACHE = default


def load_policy(path, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    _restore_hf_modules_cache()
    model = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True, torch_dtype=dtype)
    model.to(device=device, dtype=dtype)
    model._set_attention_implementation("sdpa")
    use_fast_sampler(model, os.environ.get("DPO_ORIGINAL_SAMPLER") != "1")
    return model


def load_codec(model, device):
    codec_path = os.environ.get("MOSS_CODEC_DIR") or "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
    return model._load_audio_tokenizer(audio_tokenizer=None, audio_tokenizer_type=CODEC_TYPE,
                                       audio_tokenizer_pretrained_name_or_path=codec_path, device=device)


def text_normalizer(repo_dir):
    """MOSS inference text normaliser (sft/INTERFACES.md section 0)."""
    sys.path.insert(0, os.path.abspath(repo_dir))
    from text_normalization_pipeline import prepare_tts_request_texts

    def norm(text):
        return str(prepare_tts_request_texts(text=text, enable_wetext=False, enable_normalize_tts_text=True,
                                             text_normalizer_manager=None)["text"])
    return norm


@torch.no_grad()
def encode_voice(model, codec, wav_path, nq, device):
    """Reference-clip codes [T, 16], exactly as model.inference() encodes the prompt (batch 1:
    the SFT gate G0a found batched codec encodes differ from batch-1 ones)."""
    sr = model._resolve_audio_tokenizer_sample_rate(codec)
    ch = model._resolve_audio_tokenizer_channels(codec)
    wav, sr = model._load_reference_audio(wav_path, sr, ch)
    enc = model._call_audio_encode(audio_tokenizer=codec, waveform=wav.to(device), sample_rate=sr)
    return model._mask_unused_audio_channels(model._normalize_audio_codes(enc), nq=nq).cpu()


# ---- sequences --------------------------------------------------------------------------------

def completion_rows(cfg, codes, ended):
    """Generated frames [F, 16] (+ the audio_end row when the sample stopped) as [F(+1), 17] rows."""
    codes = torch.as_tensor(codes, dtype=torch.long)
    rows = torch.full((codes.shape[0] + int(ended), cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long)
    rows[: codes.shape[0], 0] = cfg.audio_assistant_slot_token_id
    rows[: codes.shape[0], 1:] = codes
    if ended:
        rows[-1, 0] = cfg.audio_end_token_id   # audio columns stay pad: no audio targets on the stop row
    return rows


def collate(cfg, examples):
    """examples: [{"prompt_rows", "chosen_codes", "chosen_ended", "rejected_codes", "rejected_ended"}].

    Returns input_ids [2B, L, 17] (first B = chosen, last B = rejected), attention_mask [2B, L],
    completion_mask [2B, L] and n_frames [2B] (generated frames per completion, for per-frame k_hat).
    """
    seqs, comp, frames = [], [], []
    for side in ("chosen", "rejected"):
        for ex in examples:
            prompt = torch.as_tensor(ex["prompt_rows"], dtype=torch.long)
            c = completion_rows(cfg, ex[f"{side}_codes"], bool(ex[f"{side}_ended"]))
            seqs.append(torch.cat([prompt, c], 0))
            comp.append(torch.cat([torch.zeros(len(prompt)), torch.ones(len(c))]))
            frames.append(len(ex[f"{side}_codes"]))
    L = max(s.shape[0] for s in seqs)
    input_ids = torch.full((len(seqs), L, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long)
    input_ids[:, :, 0] = cfg.pad_token_id
    attention_mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    completion_mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (s, c) in enumerate(zip(seqs, comp)):    # right padding, as finetuning/dataset.py
        input_ids[i, : len(s)] = s
        attention_mask[i, : len(s)] = True
        completion_mask[i, : len(s)] = c.bool()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "completion_mask": completion_mask,
            "n_frames": torch.tensor(frames, dtype=torch.float32)}


def sequence_logps(model, input_ids, attention_mask, completion_mask, return_entropy=False):
    """log pi(y|x) per sequence = sum over completion rows of log p(slot|end) + sum_c log p(a_c).

    Row t's global hidden state predicts row t+1 (the shift of finetuning/dataset.py). Codebooks
    whose target is audio_pad (the stop row, channels >= nq) carry no term, as in the SFT loss.
    """
    cfg = model.config
    out = model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False,
                return_dict=True)
    hidden = out.global_hidden_states                                     # [N, L-1, H]
    targets = input_ids[:, 1:]
    sel = completion_mask[:, 1:] & attention_mask[:, 1:]                  # predicted rows = completion rows
    owner = sel.nonzero(as_tuple=True)[0]
    h = hidden[sel]                                                       # [M, H]
    tgt = targets[sel]                                                    # [M, 17]
    text_t, audio_t = tgt[:, 0], tgt[:, 1:]
    audio_valid = audio_t.ne(cfg.audio_pad_token_id)

    local_dtype = model.local_transformer.ln_f.weight.dtype
    local_in = [h.to(local_dtype), model.transformer.wte(text_t).to(local_dtype)]
    for c in range(cfg.n_vq - 1):                                         # teacher-forced codebooks 0..14
        ids = audio_t[:, c].masked_fill(~audio_valid[:, c], 0)
        local_in.append((model.audio_embeddings[c](ids) * audio_valid[:, c, None]).to(local_dtype))
    local_in = torch.stack(local_in, dim=1)                               # [M, 17, H]
    local_h = model.local_transformer(
        input_ids=None, attention_mask=torch.ones(local_in.shape[:2], dtype=torch.bool, device=local_in.device),
        position_ids=None, inputs_embeds=local_in, use_cache=False, return_dict=True,
        cu_seqlens=None, num_sequences=None,
    ).last_hidden_state                                                   # [M, 17, H]

    cand = torch.tensor([cfg.audio_assistant_slot_token_id, cfg.audio_end_token_id], device=h.device)
    text_logits = F.linear(local_h[:, 0], model.text_lm_head.weight.index_select(0, cand)).float()
    row_logp = text_logits.log_softmax(-1).gather(-1, text_t.eq(cfg.audio_end_token_id).long()[:, None]).squeeze(-1)

    entropy = torch.zeros(cfg.n_vq, device=h.device)
    for c in range(cfg.n_vq):
        logp = model.audio_lm_heads[c](local_h[:, c + 1]).float().log_softmax(-1)
        ids = audio_t[:, c].masked_fill(~audio_valid[:, c], 0)
        row_logp = row_logp + logp.gather(-1, ids[:, None]).squeeze(-1) * audio_valid[:, c]
        if return_entropy:
            with torch.no_grad():
                ent = -(logp.exp() * logp).sum(-1)
                entropy[c] = (ent * audio_valid[:, c]).sum() / audio_valid[:, c].sum().clamp_min(1)

    seq = torch.zeros(input_ids.shape[0], device=h.device, dtype=torch.float32).index_add(0, owner, row_logp)
    return (seq, entropy) if return_entropy else seq


@torch.no_grad()
def incremental_logp(model, prompt_rows, codes, ended):
    """Independent check of sequence_logps (gates.py D1): frame by frame through the global KV cache
    and the growing local prefix, like _iter_generation_events does when it samples."""
    cfg = model.config
    dev = next(model.parameters()).device
    local_dtype = model.local_transformer.ln_f.weight.dtype
    cand = [cfg.audio_assistant_slot_token_id, cfg.audio_end_token_id]
    cur = torch.as_tensor(prompt_rows, dtype=torch.long, device=dev)[None]
    mask = torch.ones(cur.shape[:2], dtype=torch.bool, device=dev)
    codes = torch.as_tensor(codes, dtype=torch.long, device=dev)
    past, total = None, 0.0
    for f in range(codes.shape[0] + int(ended)):
        g = model.transformer(input_ids=None, past_key_values=past, attention_mask=mask, position_ids=None,
                              inputs_embeds=model._build_inputs_embeds(cur), use_cache=True, return_dict=True,
                              cu_seqlens=None, num_sequences=None)
        local = g.last_hidden_state[:, -1, :].to(local_dtype)[:, None]
        text_lp = model.text_lm_head(model._decode_local_last_hidden_state(local))[:, cand].float().log_softmax(-1)
        if f == codes.shape[0]:            # the stop row
            total += float(text_lp[0, 1])
            break
        total += float(text_lp[0, 0])
        x = model.transformer.wte(torch.tensor([cfg.audio_assistant_slot_token_id], device=dev)).to(local_dtype)
        for c in range(cfg.n_vq):
            a = codes[f, c]
            if a == cfg.audio_pad_token_id:   # channels >= nq
                break
            local = torch.cat([local, x[:, None]], dim=1)
            lp = model.audio_lm_heads[c](model._decode_local_last_hidden_state(local)).float().log_softmax(-1)
            total += float(lp[0, a])
            x = model.audio_embeddings[c](a[None]).to(local_dtype)
        cur = completion_rows(cfg, codes[f:f + 1].cpu(), False).to(dev)[None]
        mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.bool, device=dev)], dim=1)
        past = g.past_key_values
    return total


# ---- checkpoints ------------------------------------------------------------------------------

def save_checkpoint(model, init_dir, out_dir):
    """Like finetuning/sft.py save_checkpoint: every non-weight file (config with the codec path,
    tokenizer, remote code) is copied from the init checkpoint, then the weights are written.
    Written to a temp dir and renamed, so a crash never leaves a half checkpoint behind."""
    out_dir, init_dir = Path(out_dir), Path(init_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in init_dir.iterdir():
        if f.is_file() and f.name not in ("pytorch_model.bin", "model.safetensors", "SOURCE"):
            shutil.copy2(f, tmp / f.name)
    # the codec load_codec attached (model._cached_audio_tokenizer) is not part of a MOSS checkpoint
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if not k.startswith("_cached_audio_tokenizer.")}, tmp / "pytorch_model.bin")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
