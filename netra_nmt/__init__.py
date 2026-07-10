"""
netra-nmt — a compact, from-scratch English↔Khmer neural machine translation model.

Quick start
-----------
    from netra_nmt import NetraTranslator, translate

    # one-shot helper (caches a default translator):
    translate("Hello, how are you?", direction="en2km")

    # reuse a translator across calls:
    t = NetraTranslator(device="cpu")
    t.translate("សួស្តី", direction="km2en")
"""

from .config import DIRECTIONS, ModelConfig
from .model import NetraNMT
from .translator import NetraTranslator, translate

__version__ = "0.1.0"

__all__ = [
    "NetraTranslator",
    "translate",
    "NetraNMT",
    "ModelConfig",
    "DIRECTIONS",
    "__version__",
]
