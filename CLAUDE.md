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

## Model folder (`model/` on HF)

`encoder.onnx` (tokens, mask → cross_k/v for both decoder layers) + `decoder_step.onnx`
(step_ids, past_k/v, cross_k/v, cross_mask → logits, present_k/v), opset 18, dynamo exporter,
weights stored fp16 (released weights were fp16 already → lossless). RoPE position = cache length.
Exporter: `training/netra_nmt_train/scripts/export.py` (wrappers in `onnx_graphs.py`).

## Measured

- **Parity, 2026-11-01:** `export.py --check alt` (export_rope = live HF weights, sha256 e202b349…),
  CPU, identical generated tokens torch vs ONNX: see docs/RESULTS.md (50-pair quick check: 100% in
  all four of en2km/km2en × greedy/beam-5).
- **Old torch package (main@83dd5b9) vs new package:** 40/40 identical texts on tests/golden.
- **Wheel:** 0.05 MB (was ~1 MB with the bundled tokenizer). Model download 189 MB.
- Speed: docs/RESULTS.md (ONNX vs PyTorch, same machine i7-13700KF).

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

- Upload to HF (user logs in), pin `MODEL_REVISION` (`netra_nmt/models.py`) and
  `PUBLISHED_REVISION` (`training/netra_nmt_train/hub.py`) to the returned commit.
- PyPI: user adds the Trusted Publisher (owner netra-ai-lab, repo Netra-NMT, workflow pypi.yml,
  env pypi), then tag v1.0.0. Then create the Space (needs netra-nmt 1.0.0 on PyPI).
- Licences of Darayut/khmer-english-pairs-raw, lyfeyvutha/nllb-en-km-316K,
  KrorngAI/ParaCrawl-English-Khmer-v2 (cards state none): TODO in README §8 and THIRD_PARTY_LICENSES.md.
- Netra-Translate can move from the git pin to `netra-nmt>=1.0` (its models.py imports
  `netra_nmt.config.HF_REPO_ID`, which no longer exists: use `netra-nmt download`).
