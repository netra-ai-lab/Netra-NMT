"""
netra_nmt.khseg — Khmer word segmentation for tokenizer/data/inference.

Khmer is written without spaces between words. Segmenting it with ``khmercut``
before SentencePiece gives the tokenizer real word boundaries, which the same
pipeline must reproduce at inference time. Khmer *output* is de-segmented
(inter-word spaces removed) before it is shown to the user.

The one rule, used identically at tokenizer-train, data-build and inference time:

    segment_km(text)    ->  khmercut words joined by single spaces
    desegment_km(text)  ->  spaces sitting between two Khmer characters removed

``khmercut`` is imported lazily so the package still loads without the training
extras; if it is unavailable, ``segment_km`` returns the text unchanged.
"""

from __future__ import annotations

import re

# Khmer script blocks: main block (U+1780–U+17FF) + symbols (U+19E0–U+19FF).
_KHMER = r"ក-៿᧠-᧿"
# A run of spaces flanked on both sides by Khmer characters = an inserted
# segmentation boundary (natural Khmer has no such spaces). Everything else
# (spaces around Latin, digits, punctuation) is preserved.
_INTER_KHMER_SPACE = re.compile(f"(?<=[{_KHMER}]) +(?=[{_KHMER}])")

_tokenize = None
_tokenize_loaded = False


def _get_tokenizer():
    global _tokenize, _tokenize_loaded
    if not _tokenize_loaded:
        _tokenize_loaded = True
        try:
            from khmercut import tokenize
            _tokenize = tokenize
        except Exception:
            _tokenize = None
    return _tokenize


def segment_km(text: str) -> str:
    """Word-segment Khmer and join with single spaces. No-op if khmercut is absent."""
    tok = _get_tokenizer()
    if tok is None:
        return text
    words = [w for w in tok(text) if w.strip()]
    return " ".join(words) if words else text


def desegment_km(text: str) -> str:
    """Undo segmentation: drop spaces that sit between two Khmer characters."""
    return _INTER_KHMER_SPACE.sub("", text)
