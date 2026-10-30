"""DPO adapter for any LLM-style TTS described by a voicegym.yaml (benchmark/tokenlm.py).

The Orpheus adapter with the model's own prompt layout, stop/pad tokens and codec instead of
Orpheus' constants: LoRA training, pi_ref = the adapters disabled, log-probs from the backbone's
hidden states on completion positions only (adapters/orpheus.py has the details). Checkpoints are
the merged model + adapter/ + the folder's voicegym.yaml (and codec file), so the next round and
benchmark/generate_tokenlm.py load them like the original folder.

Not run in our experiments (only the MOSS-TTS-Nano adapter was); gates.py D1-D7 check it on the GPU.
Env: lm (requirements-dpo-orpheus.txt).
"""
import copy
import math
import shutil
import sys
from pathlib import Path

import torch

from adapters.base import Sample
from adapters.orpheus import OrpheusAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from tokenlm import TokenLM  # noqa: E402


class TokenLMAdapter(OrpheusAdapter):
    name = "tokenlm"

    def __init__(self, init_dir, policy_dir, device, cfg, for_training=False, repo_dir=None):
        self.lm = TokenLM(str(init_dir), torch.device(device))
        self.deployment_sampling = self.lm.sampling()
        self.frame_rate = self.lm.frame_rate
        self.decoded_sample_rate = self.lm.codec.sample_rate
        super().__init__(init_dir, policy_dir, device, cfg, for_training=for_training, repo_dir=repo_dir)
        self.tok = self.lm.tok

    # ---- rollout ---------------------------------------------------------------------------
    def prepare_text(self, text):
        return text, len(self.lm.text_ids(text))

    def build_prompt(self, text, voice):
        return torch.tensor(self.lm.prompt(text, voice.wav if self.lm.clones() else None, voice.transcript),
                            dtype=torch.long)

    def _frames(self, tokens):
        return self.lm.n_frames([t for t in tokens if t != self.lm.stop])

    def generate(self, prompts, sampling, max_new, seed):
        stop, pad = self.lm.stop, self.lm.pad
        L = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), L), pad, dtype=torch.long)
        attn = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = p
            attn[i, L - len(p):] = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = self.model.generate(ids.to(self.device), attention_mask=attn.to(self.device), max_new_tokens=max_new,
                                      do_sample=True, eos_token_id=stop, pad_token_id=pad, **sampling)
        res = []
        for row in out[:, L:].tolist():
            ended = stop in row
            comp = row[: row.index(stop) + 1] if ended else row
            res.append(Sample(completion=torch.tensor(comp, dtype=torch.long), ended=ended, n_frames=self._frames(comp)))
        return res

    def max_new_for_seconds(self, seconds):
        return self.lm.codec.tokens_per_frame * int(math.ceil(seconds * self.frame_rate)) + 16

    def decode(self, samples):
        out = []
        for s in samples:
            toks = [t for t in s.completion.tolist() if t != self.lm.stop]
            try:
                out.append(self.lm.decode(toks) if s.n_frames else None)
            except ValueError:
                out.append(None)
        return out

    # ---- training --------------------------------------------------------------------------
    def collate(self, examples):
        pad = self.lm.pad
        seqs, starts, frames = [], [], []
        for side in ("chosen", "rejected"):
            for e in examples:
                comp = torch.as_tensor(e[side], dtype=torch.long)
                seqs.append(torch.cat([torch.as_tensor(e["prompt"], dtype=torch.long), comp]))
                starts.append(len(e["prompt"]))
                frames.append(self._frames(comp.tolist()))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        comp_mask = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, (s, st) in enumerate(zip(seqs, starts)):
            ids[i, : len(s)] = s
            attn[i, : len(s)] = 1
            comp_mask[i, st: len(s)] = True
        return {"input_ids": ids.to(self.device), "attention_mask": attn.to(self.device),
                "completion_mask": comp_mask.to(self.device),
                "n_frames": torch.tensor(frames, dtype=torch.float32)}

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
        src = Path(self.lm.folder)
        for extra in ("voicegym.yaml", str(self.lm.spec.get("codec", ""))):
            if extra and (src / extra).is_file():
                shutil.copy2(src / extra, tmp / extra)
        del merged
        torch.cuda.empty_cache()
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp.rename(out_dir)
