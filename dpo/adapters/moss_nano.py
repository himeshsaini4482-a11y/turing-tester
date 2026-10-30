"""DPO adapter for OpenMOSS-Team/MOSS-TTS-Nano-100M (and every SFT checkpoint of it).

The model internals (log-probs, the generation-loop recomputation, collate, checkpoint saving) are
in dpo/nano.py; this file maps them onto the adapter contract (dpo/adapters/base.py).

prompt     = build_inference_input_ids(voice_clone) rows [T_p, 17] (reference codes in the user slot)
completion = generated frames [F, 16] (the stop row is implied by ended, see nano.completion_rows)
Full fine-tuning with fp32 master weights and bf16 autocast (SFT decision D13); pi_ref is a second,
frozen copy of init_dir. Env: the MOSS venv ($MOSS_PY) + tensorboard.
"""
import math
from pathlib import Path

import torch

import nano
from adapters.base import Adapter, Sample


class MossNanoAdapter(Adapter):
    name = "moss_nano"
    frame_rate = 12.5
    deployment_sampling = nano.DEPLOYMENT_SAMPLING
    on_policy_sampling = nano.ON_POLICY_SAMPLING

    def __init__(self, init_dir, policy_dir, device, cfg, for_training=False, repo_dir=None):
        self.device = torch.device(device)
        self.cfg = cfg
        self.init_dir = Path(init_dir)
        self.nq = int(cfg.get("nq", 16))
        self.max_text_tokens = int(cfg.get("max_text_tokens", 50))
        self.amp = dict(device_type=self.device.type, dtype=torch.bfloat16)
        dtype = torch.float32 if for_training else torch.bfloat16
        self.model = nano.load_policy(policy_dir, self.device, dtype=dtype)
        self.model.eval()
        self.ref = None
        self.codec = self.tok = self.norm = None
        self._voice_codes = {}
        if for_training:
            self.ref = nano.load_policy(self.init_dir, self.device, dtype=torch.float32).eval().requires_grad_(False)
        else:
            self.codec = nano.load_codec(self.model, self.device)
            self.decoded_sample_rate = self.model._resolve_audio_tokenizer_sample_rate(self.codec)
        self.tok = self.model._load_text_tokenizer(text_tokenizer=None, text_tokenizer_path=None)
        if repo_dir:
            self.norm = nano.text_normalizer(repo_dir)

    # ---- rollout ---------------------------------------------------------------------------
    def prepare_text(self, text):
        t = self.norm(text) if self.norm else text
        n = self.model._count_text_tokens(self.tok, t)
        return (t if n <= self.max_text_tokens else None), n

    def build_prompt(self, text, voice):
        if voice.speaker not in self._voice_codes:
            if self.codec is None:
                self.codec = nano.load_codec(self.model, self.device)
            self._voice_codes[voice.speaker] = nano.encode_voice(self.model, self.codec, voice.wav, self.nq, self.device)
        ids, _ = self.model.build_inference_input_ids(text=text, text_tokenizer=self.tok, mode="voice_clone",
                                                      prompt_audio_codes=self._voice_codes[voice.speaker],
                                                      device=self.device)
        return ids[0].cpu()

    def generate(self, prompts, sampling, max_new, seed):
        m = self.model
        ids, attn = m._left_pad_inference_batch(
            [p[None].to(self.device) for p in prompts],
            [torch.ones((1, p.shape[0]), dtype=torch.bool, device=self.device) for p in prompts], device=self.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = m.generate(input_ids=ids, attention_mask=attn, nq=self.nq, max_new_frames=max_new,
                             do_sample=True, use_kv_cache=True, **sampling)
        res = []
        for j in range(len(prompts)):
            c = m._trim_generated_audio_token_ids(out.audio_token_ids[j], effective_nq=self.nq).cpu()
            res.append(Sample(completion=c, ended=bool(c.shape[0] < max_new), n_frames=int(c.shape[0])))
        return res

    def max_new_for_seconds(self, seconds):
        return int(math.ceil(seconds * self.frame_rate)) + 1     # frames (+1: room to emit the stop row)

    def sampler_equivalence(self, prompts, sampling, max_new, seed):
        """gates.py D7: the same seeded batch with the original and with the fast repetition penalty
        (nano.fast_repetition_penalty) must give identical codes."""
        out = {}
        for name, fast in (("original", False), ("fast", True)):
            nano.use_fast_sampler(self.model, fast)
            out[name] = self.generate(prompts, sampling, max_new, seed)
        same = all(a.completion.shape == b.completion.shape and bool((a.completion == b.completion).all())
                   for a, b in zip(out["original"], out["fast"]))
        return {"pass": same, "n": len(prompts), "frames": [s.n_frames for s in out["fast"]]}

    def decode(self, samples):
        out = [None] * len(samples)
        idx = [i for i, s in enumerate(samples) if s.n_frames > 0]
        if not idx:
            return out
        if self.codec is None:
            self.codec = nano.load_codec(self.model, self.device)
        m = self.model
        sr = m._resolve_audio_tokenizer_sample_rate(self.codec)
        wavs, self.decoded_sample_rate = m._decode_audio_token_id_batch_to_waveforms(
            audio_tokenizer=self.codec, audio_token_id_batches=[samples[i].completion.to(self.device) for i in idx],
            target_sample_rate=sr, effective_nq=self.nq, resolved_device=self.device)
        for i, w in zip(idx, wavs):
            w = w.detach().float().cpu()
            out[i] = (w.mean(dim=0) if w.ndim == 2 else w).numpy()   # stereo -> mono like the benchmark
        return out

    # ---- training --------------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def collate(self, examples):
        b = nano.collate(self.model.config, [
            {"prompt_rows": e["prompt"], "chosen_codes": e["chosen"], "chosen_ended": e["chosen_ended"],
             "rejected_codes": e["rejected"], "rejected_ended": e["rejected_ended"]} for e in examples])
        return {k: v.to(self.device) for k, v in b.items()}

    def _logps(self, model, batch, return_entropy):
        with torch.autocast(**self.amp):
            return nano.sequence_logps(model, batch["input_ids"], batch["attention_mask"], batch["completion_mask"],
                                       return_entropy=return_entropy)

    def sequence_logps(self, batch, return_entropy=False):
        return self._logps(self.model, batch, return_entropy)

    @torch.no_grad()
    def reference_logps(self, batch):
        return self._logps(self.ref, batch, False)

    def incremental_logp(self, prompt, completion, ended):
        return nano.incremental_logp(self.model, prompt, completion, ended)

    def save_checkpoint(self, out_dir):
        nano.save_checkpoint(self.model, self.init_dir, out_dir)

    def free_reference(self):
        self.ref = None
        torch.cuda.empty_cache()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def dropout_probs(self):
        out = {}
        for n, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Dropout):
                out[n] = float(mod.p)
            elif hasattr(mod, "attn_dropout") and isinstance(mod.attn_dropout, float):
                out[n + ".attn_dropout"] = mod.attn_dropout
        return out
