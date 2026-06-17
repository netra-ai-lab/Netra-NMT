"""
export_checkpoint.py — turn a training checkpoint into release artifacts.

Reads a fat training checkpoint (model + optimizer + AMP scaler state, ~1.1 GB)
and writes a slim, distributable set:

    export/
      model.safetensors   # fp16 model weights only (~180 MB)
      config.json         # architecture + tokenizer config
      spm_32k.model       # SentencePiece tokenizer

Optionally uploads the export directory to the Hugging Face Hub.

Usage
-----
    # Default: export checkpoints/epoch_04 -> export/
    python scripts/export_checkpoint.py

    # Pick a different checkpoint / output dir:
    python scripts/export_checkpoint.py \
        --checkpoint checkpoints/best/checkpoint.pt --out export

    # Export and push to the Hub (requires `huggingface-cli login`):
    python scripts/export_checkpoint.py --push <user>/netra-nmt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "epoch_04" / "checkpoint.pt"
DEFAULT_SPM = ROOT / "tokenizer" / "spm_32k.model"
DEFAULT_OUT = ROOT / "export"

# Config keys copied from the training cfg into the slim config.json.
_CFG_KEYS = ["d_model", "enc_layers", "dec_layers", "n_heads", "ffn_dim", "max_len",
             "pad_id", "bos_id", "eos_id"]

LANG_MARKERS = {"en": "<2en>", "km": "<2km>"}


def export(checkpoint: Path, out_dir: Path, spm_path: Path) -> Path:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    vocab_size = ckpt["vocab_size"]
    state_dict = ckpt["model"]

    # Strip the torch.compile() "_orig_mod." prefix if present.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # lm_head.weight is tied to decoder.embed.weight (shared storage). safetensors
    # rejects shared tensors, and it is re-tied on load, so drop it here.
    state_dict.pop("lm_head.weight", None)

    # Cast to fp16 for a ~2x smaller download.
    fp16_state = {k: v.half().contiguous() for k, v in state_dict.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "model.safetensors"
    save_file(fp16_state, str(weights_path))

    config = {k: cfg[k] for k in _CFG_KEYS if k in cfg}
    config["vocab_size"] = vocab_size
    config["lang_markers"] = LANG_MARKERS
    config["dtype"] = "float16"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if spm_path.exists():
        shutil.copy(spm_path, out_dir / "spm_32k.model")
    else:
        print(f"  WARNING: tokenizer not found at {spm_path}; skipped.")

    n_params = sum(v.numel() for v in fp16_state.values())
    size_mb = weights_path.stat().st_size / 1e6
    print(f"  Epoch        : {ckpt.get('epoch', '?')}")
    print(f"  Global step  : {ckpt.get('global_step', '?')}")
    print(f"  Val loss     : {ckpt.get('val_loss', float('nan')):.4f}")
    print(f"  Vocab size   : {vocab_size:,}")
    print(f"  Params (fp16): {n_params:,}")
    print(f"  Weights      : {weights_path}  ({size_mb:.1f} MB)")
    print(f"  Export dir   : {out_dir}")
    return out_dir


def _model_card(repo_id: str) -> str:
    return f"""---
language:
- en
- km
license: mit
library_name: netra-nmt
pipeline_tag: translation
tags:
- translation
- khmer
- english
---

# netra-nmt

A compact, from-scratch encoder-decoder model for **English ↔ Khmer** translation.

## Usage

```bash
pip install netra-nmt
```

```python
from netra_nmt import NetraTranslator
t = NetraTranslator(repo_id="{repo_id}")
print(t.translate("Hello, how are you?", direction="en2km"))
print(t.translate("សួស្តី", direction="km2en"))
```

Files: `model.safetensors` (fp16 weights), `config.json`, `spm_32k.model` (SentencePiece tokenizer).
"""


def push(out_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    # Write a model card alongside the artifacts.
    (out_dir / "README.md").write_text(_model_card(repo_id), encoding="utf-8")

    print(f"Uploading {out_dir} → https://huggingface.co/{repo_id} …")
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    print(f"Done: https://huggingface.co/{repo_id}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                   help=f"Path to checkpoint.pt (default: {DEFAULT_CKPT}).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output export directory (default: {DEFAULT_OUT}).")
    p.add_argument("--spm", type=Path, default=DEFAULT_SPM,
                   help=f"SentencePiece model to bundle (default: {DEFAULT_SPM}).")
    p.add_argument("--push", type=str, default=None, metavar="REPO_ID",
                   help="Upload the export dir to this Hugging Face repo id.")
    args = p.parse_args(argv)

    out_dir = export(args.checkpoint, args.out, args.spm)

    if args.push:
        push(out_dir, args.push)


if __name__ == "__main__":
    main()
