"""What kind of TTS a --model folder (or Hugging Face id) or --api config is, and how to run it.

Detection, first match wins:
  1. synthesize.py in the folder          -> "custom"   (your load() / synthesize() functions)
  2. turing-tester.yaml with type: token-lm     -> "token_lm" (LLM backbone + audio-codec tokens)
     (turing-tester.yaml may also force a built-in family with `family: moss_tts_nano | qwen3_tts | orpheus`)
  3. config.json model_type:
       qwen3_tts                           -> "qwen3_tts"  (Base = voice cloning; custom_voice = one fine-tuned speaker)
       moss_tts_nano                       -> "moss_tts_nano"
       llama with Orpheus' 156,9xx vocab   -> "orpheus"
A --api file is always "api".
"""
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from turing_tester import envs, paths
from turing_tester.util import Fail, fingerprint

FAMILIES = {
    "qwen3_tts": "Qwen3-TTS",
    "moss_tts_nano": "MOSS-TTS-Nano",
    "orpheus": "Orpheus-TTS",
    "token_lm": "LLM-style TTS (turing-tester.yaml)",
    "custom": "your synthesize.py",
    "api": "TTS API",
}


@dataclass
class Model:
    kind: str
    source: str                  # local folder (absolute) or Hugging Face id, or the API yaml path
    config: dict = field(default_factory=dict)    # config.json
    vg: dict = field(default_factory=dict)        # turing-tester.yaml
    api: dict = field(default_factory=dict)       # API config

    @property
    def label(self):
        return FAMILIES[self.kind]

    @property
    def local(self):
        return Path(self.source).is_dir()

    @property
    def default_name(self):
        if self.kind == "api":
            return self.api.get("name") or Path(self.source).stem
        p = Path(self.source.rstrip("/\\"))
        if p.name in ("final", "checkpoint-last", "best") or p.name.startswith(("checkpoint-", "r0")):
            # runs/sft/<name>/final -> <name>_final; runs/dpo/<name>/ckpt/r02 -> <name>_r02
            parent = p.parent.parent if p.parent.name == "ckpt" else p.parent
            return f"{parent.name}_{p.name}"
        return p.name

    # what each kind supports (README "What can be trained")
    @property
    def can_sft(self):
        return self.kind in ("moss_tts_nano", "orpheus", "token_lm", "qwen3_tts") or (
            self.kind == "custom" and self.local and (Path(self.source) / "train.py").exists())

    @property
    def can_dpo(self):
        return self.kind in ("moss_tts_nano", "orpheus", "token_lm")

    @property
    def custom_voice_speaker(self):
        """A fine-tuned Qwen3-TTS (custom_voice) speaks one built-in speaker instead of cloning."""
        if self.kind != "qwen3_tts" or self.config.get("tts_model_type", "base") == "base":
            return None
        spk = (self.config.get("talker_config") or {}).get("spk_id") or {}
        return next(iter(spk), None)

    def env(self):
        """Name of the environment that runs this model's generator."""
        if self.kind == "moss_tts_nano":
            return "moss"
        if self.kind in ("custom", "token_lm") and self.local and (Path(self.source) / "requirements.txt").exists():
            return custom_env(Path(self.source), self.kind)
        if self.kind == "token_lm":
            return "lm"
        return "tts"

    def generate_cmd(self, manifest, run_dir, limit=0, seed=1234, device="cuda:0"):
        """(script, args) of the benchmark generator for this model."""
        common = ["--manifest", manifest, "--run-dir", run_dir, "--limit", limit, "--seed", seed]
        if self.kind == "qwen3_tts":
            extra = ["--speaker", self.custom_voice_speaker] if self.custom_voice_speaker else []
            return "benchmark/generate_tts.py", common + ["--model", self.source, "--device", device] + extra
        if self.kind == "moss_tts_nano":
            # 50 = MOSS's default chunk limit, chosen in our pilot (runs/sft/decoding.json)
            return "benchmark/generate_moss_nano.py", common + [
                "--model", self.source, "--repo-dir", envs.checkout_dir("moss"),
                "--voice-clone-max-text-tokens", 50, "--device", device]
        if self.kind == "orpheus":
            return "benchmark/generate_orpheus.py", common + ["--model", self.source, "--device", device]
        if self.kind == "token_lm":
            return "benchmark/generate_tokenlm.py", common + ["--model", self.source, "--device", device]
        if self.kind == "custom":
            return "benchmark/generate_custom.py", common + ["--model", self.source, "--device", device]
        if self.kind == "api":
            return "benchmark/generate_api.py", common + ["--config", self.source]
        raise Fail(f"no generator for {self.kind}")

    def ensure_env(self):
        name = self.env()
        if name.startswith("model-"):
            return ensure_custom_env(Path(self.source), self.kind)
        if name == "moss":
            envs.ensure_checkout("moss")
        return envs.ensure(name)


def custom_env(folder, kind="custom"):
    req = (folder / "requirements.txt").read_text(encoding="utf-8")
    return f"model-{fingerprint([folder.name, kind, req])[:10]}"


def ensure_custom_env(folder, kind="custom"):
    """Environment for a model folder that ships a requirements.txt: a synthesize.py model gets torch +
    soundfile + numpy + its requirements; a turing-tester.yaml model the lm environment's packages + its requirements
    (generation, LoRA SFT and DPO all run there)."""
    name = custom_env(folder, kind)
    base = {"requirements": ["requirements-dpo-orpheus.txt"], "packages": ["pyyaml==6.0.2"]} if kind == "token_lm"         else {"requirements": [], "packages": [envs.TORCH, "soundfile", "numpy", "pyyaml==6.0.2"]}
    envs.ENVS.setdefault(name, {"requirements": base["requirements"], "checkouts": [], "about": f"your model {folder.name}",
                                "packages": base["packages"] + ["-r", str(folder / "requirements.txt")]})
    return envs.ensure(name)


def load_yaml(path):
    import yaml
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise Fail(f"{paths.rel(path)} is not valid YAML: {e}")
    if not isinstance(data, dict):
        raise Fail(f"{paths.rel(path)} must be a YAML mapping (key: value lines)")
    return data


def _hf_config(repo_id):
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise Fail(f"{repo_id!r} is not a folder, and its config.json could not be read from Hugging Face ({e})")


def _family_from_config(cfg):
    mt = cfg.get("model_type")
    if mt == "qwen3_tts":
        return "qwen3_tts"
    if mt == "moss_tts_nano":
        return "moss_tts_nano"
    if mt == "llama" and 156_900 <= int(cfg.get("vocab_size", 0)) <= 157_100:
        return "orpheus"
    return None


def detect(model_arg):
    """--model PATH|HF_ID -> Model."""
    p = Path(model_arg).expanduser()
    if p.exists():
        if not p.is_dir():
            raise Fail(f"--model must be a folder (the one with config.json and the weights), got {model_arg}")
        p = p.resolve()
        vg = load_yaml(p / "turing-tester.yaml") if (p / "turing-tester.yaml").exists() else {}
        cfg = {}
        if (p / "config.json").exists():
            with open(p / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
        if (p / "synthesize.py").exists():
            return Model("custom", str(p), cfg, vg)
        if vg.get("family"):
            fam = str(vg["family"])
            if fam not in ("moss_tts_nano", "qwen3_tts", "orpheus"):
                raise Fail(f"{paths.rel(p / 'turing-tester.yaml')}: family must be moss_tts_nano, qwen3_tts or orpheus")
            return Model(fam, str(p), cfg, vg)
        if str(vg.get("type", "")).replace("_", "-") == "token-lm":
            from turing_tester import tokenlm_config
            tokenlm_config.validate(vg, p)
            return Model("token_lm", str(p), cfg, vg)
        fam = _family_from_config(cfg)
        if fam:
            return Model(fam, str(p), cfg, vg)
        what = f"model_type {cfg.get('model_type')!r}" if cfg else "no config.json"
        raise Fail(f"could not tell what kind of TTS {paths.rel(p)} is ({what}).\n"
                   "  Add a turing-tester.yaml (LLM-style models) or a synthesize.py (anything else) to the folder;\n"
                   "  see docs/adding_a_model.md")
    local_roots = ("models", "runs", "voices", "apis", "data", "pretrained", "~")
    if (model_arg.count("/") == 1 and "\\" not in model_arg and not model_arg.startswith((".", "/"))
            and model_arg.split("/")[0] not in local_roots):
        cfg = _hf_config(model_arg)
        fam = _family_from_config(cfg)
        if not fam:
            raise Fail(f"{model_arg} on Hugging Face is not a model Turing Tester recognises (model_type "
                       f"{cfg.get('model_type')!r}); download it into models/ and add a turing-tester.yaml or synthesize.py")
        return Model(fam, model_arg, cfg, {})
    raise Fail(f"model folder not found: {model_arg}")


def detect_api(api_arg):
    p = Path(api_arg).expanduser()
    if not p.is_file():
        raise Fail(f"API config not found: {api_arg} (copy apis/template.yaml)")
    from turing_tester import apiconfig
    cfg = apiconfig.load(p)
    return Model("api", str(p.resolve()), api=cfg)


def detect_any(arg):
    """--teacher: a model folder / HF id, or an API yaml."""
    if str(arg).endswith((".yaml", ".yml")) and Path(arg).is_file():
        return detect_api(arg)
    return detect(arg)
