# From the Netra Lab kit (templates/repo).
"""Netra NMT's training package: never imported by netra_nmt itself.

    data_specs.py    every dataset form + presets -> rows of {"input", "target"}
    progress.py      bars, step lines, epoch summaries, metrics.csv, early stopping
    hub.py           paths, the project-local HF cache, the published checkpoint
    export_utils.py  netra-model.json + the parity check against PyTorch
    cli.py           netra-nmt-train check|train|evaluate|export
    scripts/         the four commands (write them for the task; Netra-OCR's
                     training/netra_ocr_train/scripts/{check_data,train,evaluate,
                     export_recognizer}.py are the reference)
"""
