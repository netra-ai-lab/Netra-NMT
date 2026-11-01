from pathlib import Path

import pytest

from netra_nmt import models
from netra_nmt.inference.hub import graph_files, model_file, pick_variant

CONFIG = {
    "graphs": {"encoder": "encoder.onnx", "decoder_step": "decoder_step.onnx"},
    "variants": {"fp16": {"graphs": {"encoder": "encoder.fp16.onnx"}, "devices": ["cuda"]}},
}


def test_variant_per_device():
    assert pick_variant(CONFIG, "cpu") is None
    assert pick_variant(CONFIG, "cuda") == "fp16"
    assert graph_files(CONFIG, "fp16")["encoder"] == "encoder.fp16.onnx"
    assert graph_files(CONFIG, None)["encoder"] == "encoder.onnx"


@pytest.mark.parametrize("name", ["../x.onnx", "/etc/passwd", "a/../../b"])
def test_model_file_refuses_outside_names(name):
    with pytest.raises(ValueError):
        model_file(Path("/tmp/model"), name)


def test_source_is_pinned_hf_by_default(monkeypatch):
    monkeypatch.delenv("NETRA_NMT_BUNDLES", raising=False)
    assert models.source() == f"{models.MODEL_REPO}/model@{models.MODEL_REVISION}"


def test_bundles_override(monkeypatch, tmp_path):
    (tmp_path / "model").mkdir()
    monkeypatch.setenv("NETRA_NMT_BUNDLES", str(tmp_path))
    assert models.source() == str(tmp_path / "model")
    monkeypatch.setenv("NETRA_NMT_BUNDLES", str(tmp_path / "missing"))
    assert models.source().startswith("hf://")
