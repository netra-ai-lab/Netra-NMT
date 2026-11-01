"""Netra NMT on ONNX Runtime (no PyTorch).

The model is a folder of two ONNX graphs (encoder, decoder step with a KV
cache), the SentencePiece tokenizer and ``netra-model.json``, published in the
Hugging Face repo ``Netra-Lab/Netra-NMT``. This package does everything around
the graphs: picking the device, downloading only the files that device needs,
and the decoding loops.

    from netra_nmt.inference import NMTModel
    model = NMTModel("hf://Netra-Lab/Netra-NMT/model", device="auto")
"""

from .nmt import NMTModel
from .session import available_devices, resolve_device

__all__ = ["NMTModel", "available_devices", "resolve_device"]
