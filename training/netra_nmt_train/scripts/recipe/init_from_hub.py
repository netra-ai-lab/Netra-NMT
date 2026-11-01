"""
init_from_hub.py — seed a training checkpoint from the released Hub model.

The Hub repo `Darayut/netra-nmt-small` ships an *exported* model:
    model.safetensors   # fp16 weights, lm_head.weight dropped (re-tied on load)
    config.json         # architecture + tokenizer config

The trainer (09_train_student.py) initialises a fresh fine-tuning phase from
`BASE_CKPT = checkpoints/best/checkpoint.pt` via `load_weights_only()`, which
expects a torch `.pt` containing a "model" state_dict. This script performs the
exact inverse of `scripts/export_checkpoint.py`:

    safetensors (fp16, no lm_head)  ->  checkpoints/best/checkpoint.pt (fp32)

`lm_head.weight` is intentionally absent — NetraNMT re-ties it to
`decoder.embed.weight` at construction, so a strict=False load is clean.

Usage
-----
    HF_TOKEN=... python scripts/training/init_from_hub.py
    python scripts/training/init_from_hub.py --repo Darayut/netra-nmt-small \
        --out checkpoints/best/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]   # training/
DEFAULT_REPO = "Darayut/netra-nmt-small"
DEFAULT_OUT = ROOT / "checkpoints" / "best" / "checkpoint.pt"

# Trainer CFG architecture defaults (must match 09_train_student.py CFG).
_ARCH_DEFAULTS = dict(
    d_model=512, enc_layers=6, dec_layers=6, n_heads=8,
    ffn_dim=2048, max_len=256, dropout=0.1,
    pad_id=0, bos_id=2, eos_id=3,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"Hugging Face model repo id (default: {DEFAULT_REPO}).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output checkpoint path (default: {DEFAULT_OUT}).")
    args = p.parse_args(argv)

    print(f"Downloading weights from {args.repo} …")
    weights_path = Path(hf_hub_download(repo_id=args.repo, filename="model.safetensors"))
    config_path = Path(hf_hub_download(repo_id=args.repo, filename="config.json"))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    print(f"  config.json vocab_size={vocab_size}  d_model={config.get('d_model')}")

    # safetensors -> fp32 state_dict
    fp16_state = load_file(str(weights_path))
    state_dict = {k: v.float().contiguous() for k, v in fp16_state.items()}
    # lm_head is tied on load; make sure it isn't carried (it shouldn't be).
    state_dict.pop("lm_head.weight", None)
    print(f"  loaded {len(state_dict)} tensors, cast to fp32")

    # Build a cfg consistent with the trainer, sourced from the exported config
    # where available, falling back to the known architecture defaults.
    cfg = dict(_ARCH_DEFAULTS)
    for k in cfg:
        if k in config:
            cfg[k] = config[k]

    ckpt = {
        "model": state_dict,
        "optimizer": None,        # fresh optimizer in the fine-tuning phase
        "scaler": None,           # fresh AMP scaler
        "cfg": cfg,
        "vocab_size": vocab_size,
        "epoch": -1,              # so trainer's BASE_CKPT branch starts at epoch 0
        "global_step": 0,
        "val_loss": math.inf,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Seed checkpoint written → {args.out}  ({size_mb:.1f} MB)")
    print("Run 09_train_student.py with no --resume-from to initialise from this seed.")


if __name__ == "__main__":
    main()
