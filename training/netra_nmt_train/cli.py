"""``netra-nmt-train``: one command for check / prepare / evaluate / export.

    netra-nmt-train check my_pairs.tsv
    netra-nmt-train prepare my_pairs.tsv --out training/runs/<run>/data
    netra-nmt-train evaluate published training/runs/<run>/bundles/model
    netra-nmt-train export --model training/runs/<run>/checkpoints/best/checkpoint.pt --out bundles/model

Training itself is netra_nmt_train/scripts/recipe/09_train_student.py (see
training/train.sh and finetune.sh).

The bash scripts in training/ wrap these with sensible defaults.
"""

import sys

COMMANDS = {
    "check": "netra_nmt_train.scripts.check_data",
    "prepare": "netra_nmt_train.scripts.prepare",
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
