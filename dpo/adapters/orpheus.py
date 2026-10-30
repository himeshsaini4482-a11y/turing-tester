"""DPO adapter for Orpheus-TTS 3B pretrained (unsloth/orpheus-3b-0.1-pretrained), in-context cloning.

Orpheus is Llama-3.2-3B with SNAC 24 kHz codes as extra tokens, 7 per frame (~11.7 frames/s), so it
is an ordinary causal LM: log pi(y|x) is the sum of next-token log-probs over the completion.
Prompting, codec encode/decode and the special tokens are benchmark/generate_orpheus.py's, imported
unchanged:

  prompt     = [SOH] ref text [EOT EOH SOA SOS] ref codes [EOS EOA] [SOH] text [EOT EOH SOA SOS]
  completion = sampled tokens up to and including EOS (EOS is the stop decision; ended = EOS seen)

Training: LoRA (peft) on the attention and MLP projections; pi_ref = the same model with the
adapters disabled, which is exactly init_dir (base weights never change). Full fine-tuning of 3B
with AdamW (~48 GB of weights + moments) does not fit one 48 GB L40S. Log-probs are computed from
the backbone's hidden states on completion positions only: full-sequence logits over the 156k
vocabulary would be ~8 GB per batch.

Checkpoint = merged full model (what generate_orpheus.py --model loads) + adapter/ (what the next
round trains on). Env: .venv-dpo-orpheus (requirements-dpo-orpheus.txt).
"""
import copy
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from adapters.base import Adapter, Sample

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from generate_orpheus import AUDIO, EOA, EOH, EOS, EOT, SOA, SOH, SOS, SR, decode, encode  # noqa: E402

LOGIT_CHUNK = 1024   # completion positions per lm_head chunk (156k-way log-softmax)


class OrpheusAdapter(Adapter):
    name = "orpheus"
    frame_rate = SR / 2048            # SNAC 24 kHz coarsest level: one 7-token frame per 2048 samples
    decoded_sample_rate = SR
    # benchmark/generate_orpheus.py defaults (top_k is not passed there, so HF's default applies too)
    deployment_sampling = {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1}
    on_policy_sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}

    def __init__(self, init_dir, policy_dir, device, cfg, for_training=False, repo_dir=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device(device)
        self.cfg = cfg
        self.init_dir = Path(init_dir) if os.path.isdir(init_dir) else init_dir   # HF id allowed for pi_ref
        self.tok = AutoTokenizer.from_pretrained(str(init_dir))
        self.amp = dict(device_type=self.device.type, dtype=torch.bfloat16)
        self.snac = None
        self._voice_ids = {}
        self.peft = for_training
        if not for_training:      # rollouts: the merged checkpoint (or the base) in bf16
            self.model = AutoModelForCausalLM.from_pretrained(str(policy_dir), dtype=torch.bfloat16,
                                                              attn_implementation="sdpa").to(self.device).eval()
            return
        from peft import LoraConfig, PeftModel, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(str(init_dir), dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to(self.device)
        base.config.use_cache = False
        adapter = Path(policy_dir) / "adapter"
        # a round's adapter applies to pi_ref's weights; policy_dir == init_dir means round 1 (fresh LoRA)
        if adapter.is_dir() and Path(str(policy_dir)).resolve() != Path(str(init_dir)).resolve():
            self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        else:                     # round 1: pi_theta_1 = pi_ref, fresh zero-initialised LoRA
            lc = cfg["lora"]
            self.model = get_peft_model(base, LoraConfig(
                r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias="none",
                target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
        if cfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

    # ---- rollout ---------------------------------------------------------------------------
    def _turn(self, text):
        return [SOH] + self.tok(text).input_ids + [EOT, EOH, SOA, SOS]

    def prepare_text(self, text):
        return text, len(self.tok(text).input_ids)   # generate_orpheus.py feeds raw text, no chunking

    def build_prompt(self, text, voice):
        if voice.speaker not in self._voice_ids:
            self._voice_ids[voice.speaker] = self._turn(voice.transcript) + encode(self._codec(), voice.wav,
                                                                                    self.device) + [EOS, EOA]
        return torch.tensor(self._voice_ids[voice.speaker] + self._turn(text), dtype=torch.long)

    def _codec(self):
        if self.snac is None:
            from snac import SNAC
            self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
        return self.snac

    @staticmethod
    def _frames(tokens):
        return sum(t >= AUDIO for t in tokens) // 7

    def generate(self, prompts, sampling, max_new, seed):
        L = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), L), EOS, dtype=torch.long)
        attn = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):            # left padding; HF derives positions from the mask
            ids[i, L - len(p):] = p
            attn[i, L - len(p):] = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = self.model.generate(ids.to(self.device), attention_mask=attn.to(self.device), max_new_tokens=max_new,
                                      do_sample=True, eos_token_id=EOS, pad_token_id=EOS, **sampling)
        res = []
        for row in out[:, L:].tolist():
            ended = EOS in row
            comp = row[: row.index(EOS) + 1] if ended else row
            res.append(Sample(completion=torch.tensor(comp, dtype=torch.long), ended=ended,
                              n_frames=self._frames(comp)))
        return res

    def max_new_for_seconds(self, seconds):
        return 7 * int(math.ceil(seconds * self.frame_rate)) + 16   # 7 tokens per frame + stop/control slack

    def decode(self, samples):
        out = []
        for s in samples:
            try:
                out.append(decode(self._codec(), s.completion.tolist(), self.device) if s.n_frames else None)
            except ValueError:       # not a valid SNAC frame sequence
                out.append(None)
        return out

    # ---- training --------------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def collate(self, examples):
        seqs, starts, frames = [], [], []
        for side in ("chosen", "rejected"):
            for e in examples:
                comp = torch.as_tensor(e[side], dtype=torch.long)
                seqs.append(torch.cat([torch.as_tensor(e["prompt"], dtype=torch.long), comp]))
                starts.append(len(e["prompt"]))
                frames.append(self._frames(comp.tolist()))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), EOS, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        comp_mask = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, (s, st) in enumerate(zip(seqs, starts)):     # right padding
            ids[i, : len(s)] = s
            attn[i, : len(s)] = 1
            comp_mask[i, st: len(s)] = True
        return {"input_ids": ids.to(self.device), "attention_mask": attn.to(self.device),
                "completion_mask": comp_mask.to(self.device),
                "n_frames": torch.tensor(frames, dtype=torch.float32)}

    def _backbone_and_head(self):
        m = self.model.get_base_model() if self.peft else self.model
        return m.model, m.lm_head

    def _logps(self, batch, return_entropy):
        backbone, head = self._backbone_and_head()
        with torch.autocast(**self.amp):
            hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              use_cache=False).last_hidden_state                       # [N, L, H]
        sel = batch["completion_mask"][:, 1:]                   # position t-1 predicts token t
        owner = sel.nonzero(as_tuple=True)[0]
        h = hidden[:, :-1][sel]                                  # [M, H]
        tgt = batch["input_ids"][:, 1:][sel]
        def chunk_logp(hc, tc):
            with torch.autocast(**self.amp):
                logits = head(hc)
            return logits.float().log_softmax(-1).gather(-1, tc[:, None]).squeeze(-1)

        lp = []
        for i in range(0, h.shape[0], LOGIT_CHUNK):
            hc, tc = h[i: i + LOGIT_CHUNK], tgt[i: i + LOGIT_CHUNK]
            # with grad: recompute the chunk's 156k-way log-softmax in backward instead of keeping it
            # (~0.6 MB per completion token in fp32, ~10 GB at 8 pairs); same forward values
            lp.append(checkpoint(chunk_logp, hc, tc, use_reentrant=False) if torch.is_grad_enabled()
                      else chunk_logp(hc, tc))
        seq = torch.zeros(batch["input_ids"].shape[0], device=h.device).index_add(0, owner, torch.cat(lp))
        if return_entropy:          # a sample estimate: the first chunk's positions, no grad
            with torch.no_grad(), torch.autocast(**self.amp):
                logp = head(h[:LOGIT_CHUNK].detach()).float().log_softmax(-1)
            return seq, (-(logp.exp() * logp).sum(-1)).mean()[None]
        return seq

    def sequence_logps(self, batch, return_entropy=False):
        return self._logps(batch, return_entropy)

    @torch.no_grad()
    def reference_logps(self, batch):
        with self.model.disable_adapter():           # base weights = init_dir = pi_ref
            return self._logps(batch, False)

    @torch.no_grad()
    def incremental_logp(self, prompt, completion, ended):
        m = self.model
        ids = torch.as_tensor(prompt, dtype=torch.long, device=self.device)[None]
        out = m(input_ids=ids, use_cache=True)
        past, logits, total = out.past_key_values, out.logits[:, -1], 0.0
        for t in torch.as_tensor(completion, dtype=torch.long).tolist():
            total += float(logits.float().log_softmax(-1)[0, t])
            out = m(input_ids=torch.tensor([[t]], device=self.device), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
        return total

    def save_checkpoint(self, out_dir):
        out_dir = Path(out_dir)
        tmp = out_dir.with_name(out_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.model.save_pretrained(str(tmp / "adapter"))
        merged = copy.deepcopy(self.model).merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(tmp), safe_serialization=True)
        self.tok.save_pretrained(str(tmp))
        del merged
        torch.cuda.empty_cache()
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp.rename(out_dir)

    def free_reference(self):
        pass                           # pi_ref is the adapter-disabled policy, nothing to free

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def dropout_probs(self):
        out = {n: float(mod.p) for n, mod in self.model.named_modules() if isinstance(mod, torch.nn.Dropout)}
        out["config.attention_dropout"] = float(getattr(self.model.config, "attention_dropout", 0.0) or 0.0)
        return out
