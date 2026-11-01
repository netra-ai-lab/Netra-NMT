"""Load a trained Netra NMT model in PyTorch from any of its three forms.

    a release folder   model.safetensors (fp16) + config.json + spm_32k.model
                       (what export_checkpoint.py writes, and pytorch/ on HF)
    a training file    checkpoint.pt with "model", "cfg" and "vocab_size"
    "published"        pytorch/ of Netra-Lab/Netra-NMT, downloaded once

Returns ``(model, config, spm_path)``: the model in fp32 eval mode, the config
as a dict (the keys of config.json) and the SentencePiece file next to it, or
None when the form carries none.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .hub import PUBLISHED_REPO, PUBLISHED_REVISION, TRAINING_DIR
from .model import NetraNMT

ARCH_KEYS = ("vocab_size", "d_model", "enc_layers", "dec_layers", "n_heads", "ffn_dim", "max_len", "pad_id")
DEFAULT_SPM = TRAINING_DIR / "tokenizer" / "spm_32k_kmseg.model"


def _clean_state(state: dict) -> dict:
    """Strip torch.compile()'s '_orig_mod.' prefix if present."""
    return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}


def build(config: dict, state: dict) -> NetraNMT:
    model = NetraNMT(**{k: config[k] for k in ARCH_KEYS if k in config}, dropout=0.0)
    # lm_head.weight is tied to embed.weight and not stored in releases.
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint doesn't match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval()


def load(spec: str | Path):
    if str(spec) == "published":
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(PUBLISHED_REPO, revision=PUBLISHED_REVISION, allow_patterns=["pytorch/*"]))
        spec = root / "pytorch"
    path = Path(spec)
    if path.is_dir():
        from safetensors.torch import load_file
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        spm = path / "spm_32k.model"
        return build(config, load_file(str(path / "model.safetensors"))), config, spm if spm.exists() else None
    if path.is_file():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = {**ckpt["cfg"], "vocab_size": ckpt["vocab_size"]}
        return build(config, _clean_state(ckpt["model"])), config, None
    raise FileNotFoundError(f"No checkpoint at {path}")
