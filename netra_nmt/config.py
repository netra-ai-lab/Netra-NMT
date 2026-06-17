"""
netra_nmt.config — model configuration and shared constants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Hugging Face repo that hosts the released weights / tokenizer.
# Override at call time via NetraTranslator(repo_id=...) or the
# NETRA_NMT_REPO_ID environment variable.
# --------------------------------------------------------------------------
HF_REPO_ID = "Darayut/netra-nmt-small"

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
SPM_FILENAME = "spm_32k.model"

# Direction → language marker prepended to the *source* text.
# Training built sources as "<2km> {en}" (EN→KM) and "<2en> {km}" (KM→EN),
# so the marker names the *target* language.
LANG_MARKERS = {"en": "<2en>", "km": "<2km>"}

# Supported translation directions ("src2tgt").
DIRECTIONS = {
    "en2km": ("en", "km"),
    "km2en": ("km", "en"),
}
DEFAULT_DIRECTION = "en2km"


@dataclass
class ModelConfig:
    """Architecture + tokenizer configuration, mirrors the exported config.json."""

    vocab_size: int = 32000
    d_model: int = 512
    enc_layers: int = 6
    dec_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048
    max_len: int = 256

    # special token ids (must match the SentencePiece model)
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3

    lang_markers: dict = field(default_factory=lambda: dict(LANG_MARKERS))
    dtype: str = "float16"

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
