"""
netra_nmt.diffusion_weights — resolve diffusion model weights / config / tokenizer.

Mirrors :mod:`netra_nmt.weights`, for the experimental non-autoregressive
diffusion model (:mod:`netra_nmt.diffusion_model`) — same size as the
autoregressive model, trading some translation quality for much faster CPU
decoding (see the README's "Diffusion model" section). Weights are published
at :data:`config.HF_DIFFUSION_REPO_ID` and downloaded (and cached) on first
use, same as the AR model. Power users can point at a local export directory
(the layout produced by ``scripts/export_diffusion_checkpoint.py``) via
``local_dir`` to skip the Hub entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import CONFIG_FILENAME, HF_DIFFUSION_REPO_ID, SPM_FILENAME, WEIGHTS_FILENAME

# Tokenizer bundled with the package (shared with the AR model — same vocab).
_BUNDLED_SPM = Path(__file__).resolve().parent / "assets" / SPM_FILENAME


def _hf_download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def resolve_diffusion(
    repo_id: str | None = None,
    local_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """
    Resolve (weights_path, config_path, spm_path) for the diffusion model.

    Args:
        repo_id:   Hugging Face repo id. Defaults to
                   ``NETRA_NMT_DIFFUSION_REPO_ID`` env var, then
                   :data:`config.HF_DIFFUSION_REPO_ID`.
        local_dir: directory containing ``model.safetensors`` + ``config.json``,
                   as produced by ``scripts/export_diffusion_checkpoint.py``.
                   When given, the Hub is not contacted.

    Raises:
        FileNotFoundError: local_dir is given but missing the expected files.
    """
    if local_dir is not None:
        d = Path(local_dir)
        weights = d / WEIGHTS_FILENAME
        config = d / CONFIG_FILENAME
        if not weights.exists() or not config.exists():
            raise FileNotFoundError(
                f"Expected {WEIGHTS_FILENAME} and {CONFIG_FILENAME} in {d}. "
                f"Run scripts/export_diffusion_checkpoint.py to produce them."
            )
        spm = d / SPM_FILENAME
        if not spm.exists():
            spm = _BUNDLED_SPM
        return weights, config, spm

    repo = repo_id or os.environ.get("NETRA_NMT_DIFFUSION_REPO_ID") or HF_DIFFUSION_REPO_ID
    weights = _hf_download(repo, WEIGHTS_FILENAME)
    config = _hf_download(repo, CONFIG_FILENAME)
    spm = _BUNDLED_SPM if _BUNDLED_SPM.exists() else _hf_download(repo, SPM_FILENAME)
    return weights, config, spm
