"""
netra_nmt.textnorm — deterministic source-side text normalization.

One rule, used identically at training-data build time and at inference:

    normalize_en_source(text)  ->  NFC + collapsed whitespace + lowercase

It is applied ONLY to English when English is the *source* of the en2km
direction (i.e. the text following the ``<2km>`` marker). It is never applied
to English when English is the *target* (km2en output), so generated English
keeps its natural casing. Khmer has no case and is never touched.

Baking this into training means "I love Cambodia" and "i love cambodia" become
the exact same model input, which is the real fix for the case-sensitivity bug
(the old inference-only heuristic in translator.py could not fix a model that
was trained on mixed-case English).
"""

from __future__ import annotations

import unicodedata


def normalize_en_source(text: str) -> str:
    """Normalize English text used as an en2km source.

    NFC-normalize, collapse all runs of whitespace to single spaces, strip, and
    lowercase. Deterministic and idempotent.
    """
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.lower()
