# From the Netra Lab kit (templates/repo), after Netra-OCR's
# training/netra_ocr_train/utils.py. Adapt: PUBLISHED_CHECKPOINTS and the revision.
"""Paths, the project-local Hugging Face cache and the published checkpoints."""

import os
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = TRAINING_DIR / "runs"
HF_CACHE_DIR = TRAINING_DIR / ".hf_cache"

PUBLISHED_REPO = "Netra-Lab/Netra-NMT"
PUBLISHED_REVISION = "7c650316bdd92a7039383e7108cf066cb0b9d108"   # holds model/ and pytorch/
PUBLISHED_CHECKPOINTS = {"published": "pytorch/"}   # a release folder (see checkpoint.py)


def use_hf_cache(cache_dir: str | Path = HF_CACHE_DIR):
    """Keep Hugging Face downloads (datasets, checkpoints) under ``cache_dir``.

    Sets the hub and datasets caches rather than HF_HOME, so the user's saved
    login (``hf auth login``) still applies; with HF_HOME moved, downloads go
    anonymous and hit rate limits. Call before importing ``datasets`` /
    ``huggingface_hub``.
    """
    cache_dir = Path(cache_dir).resolve()
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))

