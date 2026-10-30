"""LLM-style TTS runtime: one causal LM over text tokens + audio-codec tokens, described by the
model folder's voicegym.yaml (docs/adding_a_model.md). Used by generate_tokenlm.py (benchmark),
sft/train_lm.py (LoRA SFT) and dpo/adapters/tokenlm.py (DPO).

  prompt     = the yaml `prompt` list, with `text` / `ref_text` tokenized by the folder's tokenizer
               and `ref_audio` replaced by the codec tokens of the voice clip
  completion = audio tokens, ended by `stop_token`

Codecs: `snac_24khz` built in (Orpheus' layout: 7 tokens per frame, slot k at offset + k * 4096,
as in generate_orpheus.py), or a .py file in the model folder with
  sample_rate, frame_rate, tokens_per_frame,
  load(device) -> state, encode(state, wav_path) -> [token ids], decode(state, [token ids]) -> mono float numpy
(decode raises ValueError for an invalid token sequence).

ORPHEUS is the voicegym.yaml equivalent of generate_orpheus.py, so Orpheus folders can use this
runtime too (LoRA SFT); the benchmark and DPO keep the original Orpheus code.
"""
import importlib.util
import os
from pathlib import Path

ORPHEUS = {
    "type": "token-lm",
    "codec": "snac_24khz",
    "audio_token_offset": 128266,
    "stop_token": 128258,
    "pad_token": 128258,
    "prompt": [[128259], "ref_text", [128009, 128260, 128261, 128257], "ref_audio", [128258, 128262],
               [128259], "text", [128009, 128260, 128261, 128257]],
    "sampling": {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1},
    "max_new_tokens": 2400,
}


def load_spec(folder):
    import yaml
    path = Path(folder) / "voicegym.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
        if spec.get("family") == "orpheus" or str(spec.get("type", "")).replace("_", "-") != "token-lm":
            return dict(ORPHEUS)
        return spec
    return dict(ORPHEUS)   # an Orpheus folder without voicegym.yaml


class Snac24:
    """Orpheus' SNAC 24 kHz layout, with a configurable token offset."""
    sample_rate = 24000
    frame_rate = 24000 / 2048
    tokens_per_frame = 7

    def __init__(self, offset, device):
        from snac import SNAC
        self.offset = offset
        self.device = device
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    def encode(self, wav_path):
        import librosa
        import torch
        wav, _ = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        with torch.inference_mode():
            c0, c1, c2 = (c[0].tolist() for c in self.snac.encode(torch.from_numpy(wav)[None, None].to(self.device)))
        ids = []
        for i in range(len(c0)):
            frame = [c0[i], c1[2 * i], c2[4 * i], c2[4 * i + 1], c1[2 * i + 1], c2[4 * i + 2], c2[4 * i + 3]]
            ids += [self.offset + k * 4096 + code for k, code in enumerate(frame)]
        return ids

    def decode(self, ids):
        import torch
        codes = [t - self.offset for t in ids if t >= self.offset]
        codes = codes[: len(codes) // 7 * 7]
        c0, c1, c2 = [], [], []
        for i in range(0, len(codes), 7):
            f = [c - k * 4096 for k, c in enumerate(codes[i: i + 7])]
            c0.append(f[0])
            c1 += [f[1], f[4]]
            c2 += [f[2], f[3], f[5], f[6]]
        if not c0 or not all(0 <= c < 4096 for c in c0 + c1 + c2):
            raise ValueError("generated tokens are not a valid SNAC frame sequence")
        with torch.inference_mode():
            audio = self.snac.decode([torch.tensor(c, device=self.device)[None] for c in (c0, c1, c2)])
        return audio[0, 0].float().cpu().numpy()

    def is_audio(self, t):
        return t >= self.offset


class FileCodec:
    """A codec.py in the model folder."""

    def __init__(self, path, device):
        spec = importlib.util.spec_from_file_location(f"vg_codec_{abs(hash(str(path)))}", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        for attr in ("sample_rate", "frame_rate", "tokens_per_frame", "load", "encode", "decode"):
            if not hasattr(self.mod, attr):
                raise SystemExit(f"{path}: the codec module must define `{attr}` (docs/adding_a_model.md)")
        self.sample_rate = self.mod.sample_rate
        self.frame_rate = self.mod.frame_rate
        self.tokens_per_frame = self.mod.tokens_per_frame
        self.state = self.mod.load(device)
        self.audio_ids = getattr(self.mod, "is_audio_token", None)

    def encode(self, wav_path):
        return list(self.mod.encode(self.state, wav_path))

    def decode(self, ids):
        return self.mod.decode(self.state, list(ids))

    def is_audio(self, t):
        return self.audio_ids(t) if self.audio_ids else True


class TokenLM:
    """Spec + tokenizer + codec (the LM itself is loaded by the caller: plain, LoRA, or merged)."""

    def __init__(self, folder, device, spec=None):
        from transformers import AutoTokenizer
        self.folder = str(folder)
        self.device = device
        self.spec = spec or load_spec(folder)
        self.tok = AutoTokenizer.from_pretrained(self.folder)
        self.stop = int(self.spec["stop_token"])
        self.pad = int(self.spec.get("pad_token", self.stop))
        self._codec = None
        self._voice = {}

    @property
    def codec(self):
        if self._codec is None:
            name = self.spec["codec"]
            if name == "snac_24khz":
                self._codec = Snac24(int(self.spec["audio_token_offset"]), self.device)
            else:
                self._codec = FileCodec(os.path.join(self.folder, name), self.device)
        return self._codec

    @property
    def frame_rate(self):
        return self.codec.frame_rate

    def clones(self):
        return "ref_audio" in self.spec["prompt"]

    def text_ids(self, text):
        return self.tok(text).input_ids   # with BOS when the tokenizer adds one (as generate_orpheus.py)

    def voice_ids(self, ref_wav):
        if ref_wav not in self._voice:
            self._voice[ref_wav] = self.codec.encode(ref_wav)
        return self._voice[ref_wav]

    def prompt(self, text, ref_wav=None, ref_text=None):
        ids = []
        for item in self.spec["prompt"]:
            if isinstance(item, list):
                ids += [int(t) for t in item]
            elif item == "text":
                ids += self.text_ids(text)
            elif item == "ref_text":
                ids += self.text_ids(ref_text or "")
            elif item == "ref_audio":
                if not ref_wav:
                    raise ValueError("this model clones a voice: a reference clip is required")
                ids += self.voice_ids(ref_wav)
        return ids

    def completion(self, target_wav):
        """Training target: the clip's codec tokens + stop."""
        return self.codec.encode(target_wav) + [self.stop]

    def cut(self, tokens):
        """Generated tokens up to (not including) the stop token, and whether it was emitted."""
        if self.stop in tokens:
            return tokens[: tokens.index(self.stop)], True
        return tokens, False

    def n_frames(self, tokens):
        return sum(1 for t in tokens if self.codec.is_audio(t)) // self.codec.tokens_per_frame

    def decode(self, tokens):
        return self.codec.decode(tokens)

    def sampling(self):
        s = {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1}
        s.update(self.spec.get("sampling") or {})
        return s

    def max_new_tokens(self):
        return int(self.spec.get("max_new_tokens", 2400))
