"""The web app and REST API (extra: ``netra-nmt[server]``). See :mod:`netra_nmt.server.app`."""


def main(argv=None):
    from importlib.util import find_spec
    if find_spec("fastapi") is None or find_spec("uvicorn") is None:
        raise SystemExit('The web app needs the server extra: pip install "netra-nmt[server,cpu]"')
    from .app import main as _main
    return _main(argv)
