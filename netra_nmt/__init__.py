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

Experimental: non-autoregressive diffusion decoder
----------------------------------------------------
Same parameter count as the default model, trading some quality for fixed-cost
parallel CPU decoding. Opt-in; requires a local export
(see scripts/export_diffusion_checkpoint.py) or a repo_id.

    from netra_nmt import DiffusionTranslator

    t = DiffusionTranslator(local_dir="export_diffusion")
    t.translate("I love Cambodia", direction="en2km")
"""

from .config import DIRECTIONS, ModelConfig
from .diffusion_model import DiffusionNMT
from .diffusion_translator import DiffusionTranslator
from .model import NetraNMT
from .translator import NetraTranslator, translate

__version__ = "0.1.0"

__all__ = [
    "NetraTranslator",
    "translate",
    "NetraNMT",
    "DiffusionTranslator",
    "DiffusionNMT",
    "ModelConfig",
    "DIRECTIONS",
    "__version__",
]
