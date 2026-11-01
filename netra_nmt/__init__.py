"""
netra-nmt — a compact English↔Khmer translation model (77M parameters), run
on ONNX Runtime. No PyTorch needed.

    from netra_nmt import translate
    translate("Hello, how are you?")                   # English -> Khmer
    translate("សួស្តី", direction="km2en")              # Khmer -> English

    from netra_nmt import NetraTranslator
    t = NetraTranslator(device="cpu")                  # load once, reuse
    t.translate("Hello", mode="beam", beam_size=5)

Package map
-----------
    translator.py       NetraTranslator, translate(): text in, text out
    models.py           MODEL_REPO / MODEL_REVISION (pinned), NETRA_NMT_BUNDLES, NETRA_NMT_DEVICE
    inference/          ONNX Runtime: session.py (devices), hub.py (model folders),
                        nmt.py (encoder + KV-cache decoder; greedy, beam, sampling)
    textnorm.py         English source normalisation (as in training)
    khseg.py            Khmer word segmentation (khmercut) and de-segmentation
    cli.py              the ``netra-nmt`` command (translate, REPL, ``serve``)
    server/             FastAPI app + web UI (extra: ``netra-nmt[server]``)

Training, evaluation and export live in ``training/`` (package ``netra_nmt_train``),
never imported from here.
"""

from importlib.metadata import PackageNotFoundError, version

from .config import DEFAULT_DIRECTION, DIRECTIONS
from .translator import NetraTranslator, translate

try:
    __version__ = version("netra-nmt")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.0.0+local"

__all__ = ["NetraTranslator", "translate", "DIRECTIONS", "DEFAULT_DIRECTION", "__version__"]
