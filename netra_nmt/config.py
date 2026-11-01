"""Shared constants: translation directions and language markers.

The model's own settings (sizes, special ids, max length) live in each model
folder's ``netra-model.json``; see :mod:`netra_nmt.models`.
"""

from __future__ import annotations

# Direction -> language marker prepended to the *source* text. Training built
# sources as "<2km> {en}" (EN->KM) and "<2en> {km}" (KM->EN), so the marker
# names the *target* language.
LANG_MARKERS = {"en": "<2en>", "km": "<2km>"}

# Supported translation directions ("src2tgt").
DIRECTIONS = {
    "en2km": ("en", "km"),
    "km2en": ("km", "en"),
}
DEFAULT_DIRECTION = "en2km"
MODES = ("greedy", "beam", "sample")
