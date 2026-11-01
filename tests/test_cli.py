import pytest

from netra_nmt import cli


def test_parse_defaults():
    args = cli.parse_args(["--text", "hi"])
    assert (args.text, args.direction, args.mode, args.beam_size) == ("hi", "en2km", "greedy", 5)


def test_help_names_the_command(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])
    assert "netra-nmt" in capsys.readouterr().out


def test_bad_direction_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["--text", "hi", "--direction", "fr2km"])


def test_package_has_no_torch():
    import sys

    import netra_nmt  # noqa: F401
    import netra_nmt.inference  # noqa: F401
    assert "torch" not in sys.modules
