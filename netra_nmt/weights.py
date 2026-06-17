"""
netra_nmt.weights — resolve model weights / config / tokenizer.

Weights and config are hosted on the Hugging Face Hub and downloaded (and
cached) on first use. The SentencePiece tokenizer ships inside the wheel, so
it is always available locally without a network round-trip.

Power users can point at a local export directory (the layout produced by
``scripts/export_checkpoint.py``) via ``local_dir`` to skip the Hub entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import (
    CONFIG_FILENAME,
    HF_REPO_ID,
    SPM_FILENAME,
    WEIGHTS_FILENAME,
)

# Tokenizer bundled with the package.
_BUNDLED_SPM = Path(__file__).resolve().parent / "assets" / SPM_FILENAME


def _hf_download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def resolve(
    repo_id: str | None = None,
    local_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """
    Resolve (weights_path, config_path, spm_path).

    Args:
        repo_id:   Hugging Face repo id. Defaults to ``NETRA_NMT_REPO_ID`` env
                   var, then :data:`config.HF_REPO_ID`.
        local_dir: directory containing ``model.safetensors`` and
                   ``config.json`` (and optionally ``spm_32k.model``). When
                   given, the Hub is not contacted.

    Returns:
        Absolute paths to (weights, config, tokenizer).
    """
    if local_dir is not None:
        d = Path(local_dir)
        weights = d / WEIGHTS_FILENAME
        config = d / CONFIG_FILENAME
        if not weights.exists() or not config.exists():
            raise FileNotFoundError(
                f"Expected {WEIGHTS_FILENAME} and {CONFIG_FILENAME} in {d}"
            )
        spm = d / SPM_FILENAME
        if not spm.exists():
            spm = _BUNDLED_SPM
        return weights, config, spm

    repo = repo_id or os.environ.get("NETRA_NMT_REPO_ID") or HF_REPO_ID
    weights = _hf_download(repo, WEIGHTS_FILENAME)
    config = _hf_download(repo, CONFIG_FILENAME)

    # Prefer the tokenizer shipped in the wheel; fall back to the Hub.
    spm = _BUNDLED_SPM if _BUNDLED_SPM.exists() else _hf_download(repo, SPM_FILENAME)
    return weights, config, spm
