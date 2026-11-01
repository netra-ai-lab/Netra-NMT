from netra_nmt.khseg import desegment_km
from netra_nmt.textnorm import normalize_en_source


def test_normalize_en_source_lowercases_and_collapses():
    assert normalize_en_source("  Hello,\tWORLD \n again ") == "hello, world again"


def test_normalize_en_source_is_idempotent():
    once = normalize_en_source("I Love  Cambodia")
    assert normalize_en_source(once) == once


def test_desegment_removes_only_spaces_between_khmer():
    assert desegment_km("ខ្ញុំ ទៅ សាលា") == "ខ្ញុំទៅសាលា"
    assert desegment_km("ខ្ញុំ ទៅ ABC ១២") == "ខ្ញុំទៅ ABC ១២"
    assert desegment_km("hello world") == "hello world"


def test_khmercut_is_installed():
    # A plain install must segment Khmer the way the training data was built.
    from netra_nmt.khseg import segment_km
    assert segment_km("ខ្ញុំទៅសាលារៀន").count(" ") >= 1
