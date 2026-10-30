"""VoiceGym command-line tool: benchmark, SFT and DPO for TTS models on your own machine.

Run it through the launcher at the repo root (./voicegym or .\\voicegym), which provides Python
3.12 and PyYAML. The CLI itself is standard library + PyYAML; every GPU stage runs in its own
uv-managed environment (voicegym_cli/envs.py) as a subprocess.
"""
