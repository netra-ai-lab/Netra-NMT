"""The real model against tests/golden/translations.json (run with --slow).

The golden file was written by the package after the export parity check
(training/export.sh --check alt: identical to PyTorch on the ALT test set).
Regenerate it only for a deliberate model change:
    python tests/make_golden.py
"""

import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden" / "translations.json"


@pytest.fixture(scope="module")
def translator():
    from netra_nmt import NetraTranslator
    return NetraTranslator(device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("case", json.loads(GOLDEN.read_text(encoding="utf-8")), ids=lambda c: f"{c['direction']}-{c['mode']}-{c['id']}")
def test_golden(translator, case):
    out = translator.translate(case["input"], direction=case["direction"], mode=case["mode"])
    assert out == case["output"]
