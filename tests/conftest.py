import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="also run the tests that load the real model")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="loads the real model: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the real model (download on first run)")
