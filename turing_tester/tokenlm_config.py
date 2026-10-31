"""Validation of turing-tester.yaml for LLM-style TTS (type: token-lm). Standard library only.

The runtime side (tokenizer, codec, prompt building) is benchmark/tokenlm.py, which runs in the
model's environment. docs/adding_a_model.md documents the format.
"""
from pathlib import Path

from turing_tester.util import Fail

BUILTIN_CODECS = ("snac_24khz",)
SEGMENTS = ("text", "ref_text", "ref_audio")


def validate(vg, folder):
    where = f"{Path(folder).name}/turing-tester.yaml"

    def bad(msg):
        raise Fail(f"{where}: {msg} (see docs/adding_a_model.md)")

    codec = vg.get("codec")
    if not codec:
        bad("`codec` is missing: a built-in codec (snac_24khz) or the name of a .py file in the model folder")
    if codec not in BUILTIN_CODECS and not (Path(folder) / str(codec)).is_file():
        bad(f"codec {codec!r} is neither built in ({', '.join(BUILTIN_CODECS)}) nor a file in the folder")
    if codec == "snac_24khz" and not isinstance(vg.get("audio_token_offset"), int):
        bad("snac_24khz needs `audio_token_offset`: the token id of SNAC code 0 in the first slot")
    if not isinstance(vg.get("stop_token"), int):
        bad("`stop_token` (the token id that ends the audio) must be an integer")
    prompt = vg.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        bad("`prompt` must be a list of token-id lists and the words text / ref_text / ref_audio")
    for item in prompt:
        if isinstance(item, list):
            if not all(isinstance(t, int) for t in item):
                bad(f"prompt item {item!r}: token lists may only hold integers")
        elif item not in SEGMENTS:
            bad(f"prompt item {item!r}: use a token-id list or one of {', '.join(SEGMENTS)}")
    if "text" not in prompt:
        bad("`prompt` must contain `text` (where the sentence to speak goes)")
    if ("ref_audio" in prompt) != ("ref_text" in prompt) and "ref_audio" not in prompt:
        bad("`ref_text` without `ref_audio` makes no sense")
    sampling = vg.get("sampling", {})
    if not isinstance(sampling, dict):
        bad("`sampling` must be a mapping, e.g. {temperature: 0.6, top_p: 0.95}")
    return True


def clones_voice(vg):
    return "ref_audio" in (vg.get("prompt") or [])
