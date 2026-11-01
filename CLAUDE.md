# Netra-NMT — project notes

English↔Khmer translation, 77M params (12 enc / 2 dec, RoPE, RMSNorm, SwiGLU, tied embeddings).
Package `netra-nmt` (inference only, ONNX Runtime); training package `netra-nmt-train` in `training/`.
Layout follows the Netra Lab kit (Netra-OCR is the reference).

## Decisions

- **2026-11-01, user:** full ONNX Runtime port, no torch in the package (was torch 2 GB+ install).
- **2026-11-01, user:** one command `netra-nmt` (verbs `serve`, `download`); old `netra-translate` /
  `netra-web` removed because `netra-translate` collides with the Netra-Translate app. Hence 1.0.0.
- **2026-11-01, user:** PyPI via Trusted Publishing only; HF via the user's own `hf auth login`.
  Never use tokens pasted in chat.
- **2026-11-01, user:** model, dataset and Space move to the `Netra-Lab` HF org
  (`Netra-Lab/Netra-NMT`, `Netra-Lab/bilingual-en-km`, Space `Netra-Lab/Netra-NMT`).
  The old `Darayut/*` repos stay (Netra-Translate and the website pin them).
- **2026-11-01, user:** keep the existing README graphs (`assets/*_benchmark.png`, `model_architecture.png`).
- `khmercut` is a core dependency: `segment_km` silently no-ops without it, and km2en then
  sees unsegmented Khmer (worse output). 0.1.0 and 0.2.0 segment all 1,018 ALT km sentences identically.
- The Space stays Gradio (the website playground calls its `/predict`).
- **2026-11-01, user:** training sources owned by Darayut are re-uploaded to Netra-Lab under MIT;
  third-party sources are only linked to their original HF repos. Only Darayut/khmer-english-pairs-raw
  qualified; its `final.parquet` (a merge including third-party rows) was not copied.

## Model folder (`model/` on HF)

`encoder.onnx` (tokens, mask → cross_k/v for both decoder layers) + `decoder_step.onnx`
(step_ids, past_k/v, cross_k/v, cross_mask → logits, present_k/v), opset 18, dynamo exporter,
weights stored fp16 (released weights were fp16 already → lossless). RoPE position = cache length.
Exporter: `training/netra_nmt_train/scripts/export.py` (wrappers in `onnx_graphs.py`).

## Measured (details: docs/RESULTS.md)

- **Parity, 2026-11-01:** `export.py --check alt`, CPU, torch vs ONNX generated tokens:
  1018/1018 identical in en2km/km2en × greedy/beam-5 (training/runs/export_parity.log).
- **Old torch package (main@83dd5b9) vs 1.0.0:** 40/40 identical texts on tests/golden.
- **Beam early stop** (exact bound score / lp(1+max_new_tokens)): identical to full-length beam on
  1018/1018 both directions; 704 → 193 s (en2km), 648 → 241 s (km2en) on ALT.
- **Speed, i7-13700KF, 20 sentences, median ms/sent:** torch 0.1.0 greedy 62.6 / beam 701.1;
  ONNX 1.0.0 greedy 39.1 / beam 63.5. GPU (onnxruntime-gpu) not measured.
- **Size:** wheel 0.05 MB; `[cpu]` install 199 MB; model download 181 MB (HF cache).
- **HF:** model Netra-Lab/Netra-NMT@7c65031 (pinned); dataset Netra-Lab/bilingual-en-km@f0d82d5;
  Netra-Lab/khmer-english-pairs-raw@67954ce (MIT, data/ only).

## Pitfalls

- `.gitignore` has a bare `data` rule: any folder named `data/` is ignored. That is why the recipe
  lives in `scripts/recipe/`, not `scripts/data/`.
- Removing the old hatchling editable install left an empty `site-packages/netra_nmt/` that
  shadowed the new editable install as a namespace package (`cannot import name ... unknown location`).
  `rmdir` it.
- `09_train_student.py` uses `torch.compile`: state-dict keys get `_orig_mod.`; `--init-weights` handles it.
- Tiny datasets: steps/epoch = rows / (batch 32 × accum 2), so `--train-steps 20` on 96 rows = 20 epochs.
- Python stdout to a file is block-buffered: `until grep ... log` waits for the whole run. Use `python -u`.
- `pip install onnx onnxruntime` into myenv upgraded numpy/protobuf past unbabel-comet's pins.

## Pending

- PyPI: user adds the Trusted Publisher (owner netra-ai-lab, repo Netra-NMT, workflow pypi.yml,
  env pypi), then tag v1.0.0. Then create the Space (needs netra-nmt 1.0.0 on PyPI).
- Netra-Translate can move from the git pin to `netra-nmt>=1.0` (its models.py imports
  `netra_nmt.config.HF_REPO_ID`, which no longer exists: use `netra-nmt download`).
