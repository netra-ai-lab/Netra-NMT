"""
netra_nmt.translator — the high-level API.

:class:`NetraTranslator` loads the model once (ONNX Runtime, no PyTorch) and
translates in either direction. The source text is prepared exactly the way
the training data was built:

    en2km   English NFC, whitespace collapsed, lowercased -> "<2km> ..."
    km2en   Khmer word-segmented with khmercut          -> "<2en> ..."

and Khmer output is de-segmented (the inserted word spaces removed).
"""

from __future__ import annotations

from pathlib import Path

from . import models
from .config import DEFAULT_DIRECTION, DIRECTIONS, MODES
from .khseg import desegment_km, segment_km
from .textnorm import normalize_en_source


class NetraTranslator:
    """
    Loads the model + tokenizer once and translates text in either direction.

    Args:
        local_dir: a model folder (``netra-model.json`` + graphs + tokenizer),
                   e.g. from ``training/export.sh``. Default: the pinned
                   ``Netra-Lab/Netra-NMT`` release (or ``NETRA_NMT_BUNDLES``).
        device:    "auto" (GPU if usable, else CPU), "cpu", "cuda", "cuda:1",
                   "tensorrt", "openvino". Default: ``NETRA_NMT_DEVICE`` or auto.
    """

    def __init__(self, local_dir: str | Path | None = None, device: str | None = None):
        self.model = models.load(local_dir, device)
        self.device = self.model.device

    def translate(
        self,
        text: str,
        direction: str = DEFAULT_DIRECTION,
        mode: str = "greedy",
        beam_size: int = 5,
        length_penalty: float = 0.6,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128,
        seed: int | None = None,
    ) -> str:
        """
        Translate a single string.

        Args:
            direction: "en2km" (English→Khmer) or "km2en" (Khmer→English).
            mode:      "greedy" | "beam" | "sample".
            seed:      random seed for ``mode="sample"``.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction {direction!r}. Choose one of {sorted(DIRECTIONS)}.")
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}. Choose greedy | beam | sample.")
        _src_lang, tgt_lang = DIRECTIONS[direction]

        text = normalize_en_source(text) if direction == "en2km" else segment_km(text)
        ids = self.model.encode_text(text, tgt_lang)

        if mode == "greedy":
            tokens = self.model.greedy(ids, max_new_tokens)
        elif mode == "beam":
            tokens = self.model.beam(ids, max_new_tokens, beam_size=beam_size, length_penalty=length_penalty)
        else:
            tokens = self.model.sample(ids, max_new_tokens, temperature=temperature, top_p=top_p, seed=seed)

        out = self.model.decode_ids(tokens)
        # Khmer targets were trained word-segmented; restore natural (spaceless) Khmer.
        return desegment_km(out) if tgt_lang == "km" else out

    def translate_batch(self, texts: list[str], direction: str = DEFAULT_DIRECTION, **kwargs) -> list[str]:
        """Translate a list of strings, one after another. Returns a list of outputs."""
        return [self.translate(t, direction=direction, **kwargs) for t in texts]


_DEFAULT: NetraTranslator | None = None


def translate(text: str | list[str], direction: str = DEFAULT_DIRECTION, **kwargs) -> str | list[str]:
    """
    Translate one string (or a list) with a cached default :class:`NetraTranslator`.

        >>> from netra_nmt import translate
        >>> translate("Hello, how are you?")                 # English -> Khmer
        >>> translate("សួស្តី", direction="km2en")
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = NetraTranslator()
    if isinstance(text, list):
        return _DEFAULT.translate_batch(text, direction=direction, **kwargs)
    return _DEFAULT.translate(text, direction=direction, **kwargs)
