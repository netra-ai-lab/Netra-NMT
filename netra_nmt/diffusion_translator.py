"""
netra_nmt.diffusion_translator — high-level API for the diffusion model.

Mirrors :class:`netra_nmt.translator.NetraTranslator`, but the decoder is
non-autoregressive Mask-Predict (:func:`netra_nmt.diffusion_decoding.diffusion_decode`)
instead of autoregressive. Same public shape (``repo_id`` / ``local_dir`` /
``device``, ``.translate()`` / ``.translate_batch()``) so it's a drop-in
alternative model, not a special case.

This is an **experimental, opt-in** model: same parameter count as the AR
model, but currently lower translation quality in exchange for fixed-cost,
parallel CPU decoding (see the README's "Diffusion model" section for the
measured trade-off). No diffusion weights are published on the Hub by
default — load from a local export via ``local_dir`` (see
``scripts/export_diffusion_checkpoint.py``), or pass ``repo_id`` if you've
published your own.

Example
-------
    from netra_nmt import DiffusionTranslator

    t = DiffusionTranslator(local_dir="export_diffusion")
    t.translate("I love Cambodia", direction="en2km")

Case handling (the reason case-normalization was added at training time):
  * en2km  — English *source* is normalized with ``normalize_en_source`` (lower)
             so "I love Cambodia" and "i love cambodia" become one input.
  * km2en  — Khmer source is untouched; generated English keeps its case.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
import torch
from safetensors.torch import load_file

from .config import DEFAULT_DIRECTION, DIRECTIONS, LANG_MARKERS
from .diffusion_decoding import diffusion_decode
from .diffusion_model import DiffusionNMT
from .diffusion_weights import resolve_diffusion
from .textnorm import normalize_en_source

DEFAULT_DECODE_STEPS = 10


class DiffusionTranslator:
    """
    Loads the diffusion model + tokenizer once and translates text in either
    direction with a fixed, small number of parallel denoising passes.

    Args:
        repo_id:   Hugging Face repo to pull weights from. Only meaningful if
                   diffusion weights have been published there (also read
                   from ``NETRA_NMT_DIFFUSION_REPO_ID``).
        local_dir: local export directory (see
                   ``scripts/export_diffusion_checkpoint.py``) to load from
                   instead of the Hub. This is the supported path until
                   diffusion weights are published somewhere.
        device:    "cuda" | "cpu" | torch.device. Auto-detected if None.
        steps:     default number of Mask-Predict denoising passes (K).
                   Higher = better quality, still a fixed, parallel cost.
                   Overridable per call via ``translate(..., steps=...)``.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        local_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        steps: int | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights_path, config_path, spm_path = resolve_diffusion(
            repo_id=repo_id, local_dir=local_dir
        )
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.steps = steps or self.cfg.get("decode_steps", DEFAULT_DECODE_STEPS)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_path))

        vocab_size = self.cfg["vocab_size"]
        model = DiffusionNMT(
            vocab_size=vocab_size,
            d_model=self.cfg["d_model"],
            enc_layers=self.cfg["enc_layers"],
            dec_layers=self.cfg["dec_layers"],
            n_heads=self.cfg["n_heads"],
            ffn_dim=self.cfg["ffn_dim"],
            max_len=self.cfg["max_len"],
            dropout=0.0,
        )
        state_dict = load_file(str(weights_path))
        # lm_head.weight is tied to decoder.embed.weight and intentionally not
        # stored, so strict=False is expected here (same convention as the AR
        # NetraTranslator).
        model.load_state_dict(state_dict, strict=False)
        self.model = model.to(self.device).eval()

        self.pad_id = self.cfg.get("pad_id", 0)
        self.eos_id = self.cfg.get("eos_id", 3)
        self.max_len = self.cfg["max_len"]

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _encode(self, text: str, direction: str):
        _src_lang, tgt_lang = DIRECTIONS[direction]
        if direction == "en2km":
            text = normalize_en_source(text)     # lowercase English source only
        else:
            text = text.strip()
        marker = LANG_MARKERS[tgt_lang]
        ids = self.sp.encode(f"{marker} {text}", out_type=int)[: self.max_len]
        if not ids:
            raise ValueError("Input text encoded to zero tokens — is it empty?")
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        src_mask = torch.zeros_like(src, dtype=torch.bool)
        return src, src_mask

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    @torch.no_grad()
    def translate(
        self,
        text: str,
        direction: str = DEFAULT_DIRECTION,
        steps: int | None = None,
    ) -> str:
        """
        Translate a single string.

        Args:
            direction: "en2km" (English→Khmer) or "km2en" (Khmer→English).
            steps:     override the default number of denoising passes (K)
                       for this call only.
        """
        if direction not in DIRECTIONS:
            raise ValueError(
                f"Unknown direction {direction!r}. Choose one of {sorted(DIRECTIONS)}."
            )
        src, src_mask = self._encode(text, direction)
        ids = diffusion_decode(
            self.model, src, src_key_padding_mask=src_mask,
            steps=steps or self.steps, pad_id=self.pad_id, eos_id=self.eos_id,
        )[0]
        return self.sp.decode(ids)

    def translate_batch(
        self,
        texts: list[str],
        direction: str = DEFAULT_DIRECTION,
        steps: int | None = None,
    ) -> list[str]:
        """Translate a list of strings (sequentially). Returns a list of outputs."""
        return [self.translate(t, direction=direction, steps=steps) for t in texts]


# ---------------------------------------------------------------------------
# Module-level convenience wrapper with a lazily-built default translator.
# ---------------------------------------------------------------------------

_DEFAULT: DiffusionTranslator | None = None


def translate(text: str, direction: str = DEFAULT_DIRECTION, **kwargs) -> str:
    """
    One-shot translation using a cached default :class:`DiffusionTranslator`.

    Requires ``local_dir`` (or ``repo_id``) to be resolvable via
    ``NETRA_NMT_DIFFUSION_REPO_ID`` on first call, since there is no built-in
    default diffusion repo. Prefer constructing :class:`DiffusionTranslator`
    directly to control that explicitly.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DiffusionTranslator()
    return _DEFAULT.translate(text, direction=direction, **kwargs)
