"""Helpers shared by the benchmark stages. Standard library only, so it works in both venvs."""
import csv
import importlib.util


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_attn(attn):
    """'auto' -> flash_attention_2 when flash-attn is installed, otherwise PyTorch SDPA."""
    if attn != "auto":
        return attn
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


def speaker_key(speaker_id):
    """Sort key for speaker ids: numeric LibriTTS ids ("121") and EARS ids ("p001")."""
    return (0, int(speaker_id), "") if speaker_id.isdigit() else (1, 0, speaker_id)
