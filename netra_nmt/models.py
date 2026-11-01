"""Where the model comes from, and one loaded copy per process.

The model is the folder ``model/`` of the Hugging Face repo ``MODEL_REPO``,
pinned at ``MODEL_REVISION`` (a commit, so a model upload never changes what
an installed version runs). ``NETRA_NMT_BUNDLES`` points at a local folder with
the same layout (e.g. a fine-tuned model from ``training/export.sh``); without
a ``model/`` inside it, the model comes from Hugging Face. ``NETRA_NMT_DEVICE``
picks the device ("cpu", "cuda", ...; unset: GPU if usable).
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_REPO = "hf://Netra-Lab/Netra-NMT"
# The commit of MODEL_REPO this version runs. Move it only together with a
# package release, after uploading the new folder (docs/RELEASING.md).
MODEL_REVISION = "main"   # TODO(release): the commit hash, set after the upload

# Model key -> folder in MODEL_REPO.
FOLDERS = {"model": "model"}

_loaded: dict = {}


def source(key: str = "model") -> str:
    """``key``'s model folder: local under NETRA_NMT_BUNDLES when present, else on Hugging Face."""
    if key not in FOLDERS:
        raise ValueError(f"Unknown model '{key}'. Available: {list(FOLDERS)}")
    root = os.environ.get("NETRA_NMT_BUNDLES")
    if root and (Path(root) / FOLDERS[key]).is_dir():
        return str(Path(root) / FOLDERS[key])
    return f"{MODEL_REPO}/{FOLDERS[key]}@{MODEL_REVISION}"


def load(source_: str | Path | None = None, device: str | None = None):
    """The model at ``source_`` (default: :func:`source`) on ``device``, loaded once per process."""
    from .inference import NMTModel

    src = str(source_ or source())
    device = device or os.environ.get("NETRA_NMT_DEVICE") or "auto"
    key = (src, device)
    if key not in _loaded:
        _loaded[key] = NMTModel(src, device=device)
    return _loaded[key]
