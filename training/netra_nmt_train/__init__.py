"""Netra NMT's training package: never imported by netra_nmt itself.

    model.py         the PyTorch model (RoPE, RMSNorm, SwiGLU, 12 enc / 2 dec, tied embeddings)
    decoding.py      PyTorch greedy / beam / sampling (the reference for netra_nmt.inference)
    checkpoint.py    load a release folder, a checkpoint.pt or "published"
    data.py          parallel data specs -> (en, km) pairs (ALT preset, TSV/CSV/JSONL, HF ids)
    onnx_graphs.py   the encoder and decoder-step graphs the package runs
    export_utils.py  netra-model.json, fp16 storage
    hub.py           paths, the project-local HF cache, the published repo
    cli.py           netra-nmt-train check | prepare | evaluate | export
    scripts/         those commands, and recipe/: the full data -> tokenizer ->
                     training -> distillation pipeline that built the model
"""
