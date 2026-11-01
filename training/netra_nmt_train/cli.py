# From the Netra Lab kit (templates/repo), copied from Netra-OCR's
# training/netra_ocr_train/cli.py. Adapt: COMMANDS -> your script modules.
"""``netra-nmt-train``: one command for check / train / evaluate / export.

    netra-nmt-train check my_data/
    netra-nmt-train train --init published --data my_data/
    netra-nmt-train evaluate published training/runs/<run>/best.pth
    netra-nmt-train export --model training/runs/<run>/best.pth --out bundles/model

The bash scripts in training/ wrap these with sensible defaults.
"""

import sys

COMMANDS = {
    "check": "netra_nmt_train.scripts.check_data",
    "train": "netra_nmt_train.scripts.train",
    "evaluate": "netra_nmt_train.scripts.evaluate",
    "export": "netra_nmt_train.scripts.export",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip() + f"\n\ncommands: {', '.join(COMMANDS)} (add --help to any)")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    from importlib import import_module
    sys.argv = [f"netra-nmt-train {argv[0]}"] + argv[1:]
    import_module(COMMANDS[argv[0]]).main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
