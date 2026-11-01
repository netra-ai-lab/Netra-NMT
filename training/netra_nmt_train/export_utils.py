# From the Netra Lab kit (templates/repo), after Netra-OCR's
# training/netra_ocr_train/{model_folder.py,scripts/export_recognizer.py}.
# Generic: use write_config as is; call parity_check from the export script.
# See playbooks/INFERENCE.md §3, §5.
"""Model folders: write netra-model.json, and check an exported folder against PyTorch."""

import hashlib
import json
from pathlib import Path

from tqdm import tqdm

CONFIG_NAME = "netra-model.json"
FORMAT, FORMAT_VERSION = "netra-bundle", 1


def write_config(root: str | Path, config: dict) -> Path:
    """Write ``config`` into ``root``, filling in the format header and the sha256 of every file."""
    root = Path(root)
    out = {"format": FORMAT, "format_version": FORMAT_VERSION,
           **{k: v for k, v in config.items() if k not in ("format", "format_version", "files")}}
    out["files"] = {str(p.relative_to(root)): _sha256(p)
                    for p in sorted(root.rglob("*")) if p.is_file() and p.name != CONFIG_NAME}
    path = root / CONFIG_NAME
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parity_check(onnx_predict, torch_predict, inputs: list, label: str, device: str = "cpu",
                 batch_size: int = 16) -> float:
    """The share of inputs on which the exported folder (run through the
    package's own inference code, ``onnx_predict``) gives exactly the PyTorch
    checkpoint's output (``torch_predict``). Both take a list, return a list.

    The export script exits non-zero below 0.99.
    """
    expected, got = [], []
    for i in tqdm(range(0, len(inputs), batch_size), desc=f"checking on {device}", unit="batch",
                  dynamic_ncols=True, leave=False, disable=None):
        batch = inputs[i:i + batch_size]
        expected += torch_predict(batch)
        got += onnx_predict(batch)
    same = sum(a == b for a, b in zip(got, expected))
    print(f"identical to PyTorch on {same}/{len(inputs)} of {label}")
    for i, (want, have) in [(i, p) for i, p in enumerate(zip(expected, got)) if p[0] != p[1]][:5]:
        print(f"   #{i}\n     torch: {want}\n     onnx:  {have}")
    return same / max(1, len(inputs))


MIN_STORED_FP16 = 256   # smaller weights (norms) stay float32: no size gain


def store_fp16(path: str | Path) -> str:
    """Rewrite ``path`` so its float weights are stored as fp16 behind a Cast.

    ONNX Runtime folds the casts at load time, so the maths stays float32 and
    only the file halves. From Netra-OCR's make_variant.py store-fp16.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path = Path(path)
    before = path.stat().st_size
    model = onnx.load(str(path))
    graph = model.graph
    initializers, casts = [], []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= MIN_STORED_FP16:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            initializers.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
        else:
            initializers.append(init)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.save(model, str(path))
    return f"{before / 1e6:.1f} -> {path.stat().st_size / 1e6:.1f} MB"
