# Releasing Netra NMT

A release is a version tag. It publishes:
- the package `netra-nmt` to PyPI (`.github/workflows/pypi.yml`);
- the app image `ghcr.io/netra-ai-lab/netra-nmt` for `linux/amd64` and `linux/arm64` (`.github/workflows/docker.yml`).

The model lives in the Hugging Face repo [Netra-Lab/Netra-NMT](https://huggingface.co/Netra-Lab/Netra-NMT)
(`model/` for the package, `pytorch/` for fine-tuning), the training corpus in
[Netra-Lab/bilingual-en-km](https://huggingface.co/datasets/Netra-Lab/bilingual-en-km), and the demo in the
Space [Netra-Lab/Netra-NMT](https://huggingface.co/spaces/Netra-Lab/Netra-NMT) (source: `training/release/space/`).

## One-time setup

1. **PyPI Trusted Publishing.** On pypi.org → `netra-nmt` → Manage → Publishing, add a GitHub publisher:
   - owner `netra-ai-lab`;
   - repository `Netra-NMT`;
   - workflow `pypi.yml`;
   - environment `pypi`.

   Before the first release, use a *pending* publisher. Then create the environment `pypi` in the GitHub repo (Settings → Environments).
2. **Make the image public** after the first release: GitHub → netra-ai-lab → Packages → `netra-nmt` → Package settings → Change visibility → Public.
3. **Docker Hub** (optional): the variable `DOCKERHUB_IMAGE` and the secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`.

## Each release

0. **If the models changed:**
   - export and check: `bash training/export.sh MODEL training/bundles`;
   - upload: `python training/release/upload_model.py --model-folder training/bundles/model --pytorch RELEASE_FOLDER`
     (log in first with `hf auth login`);
   - set `MODEL_REVISION` in `netra_nmt/models.py` and `PUBLISHED_REVISION` in
     `training/netra_nmt_train/hub.py` to the printed commit.
1. **Check:** `pytest tests --slow` and `python scripts/benchmark.py --device cpu` (compare with the last release).
2. **Bump** `version` in `pyproject.toml`, then commit and push.
3. **Tag:**

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

4. **Watch the workflows.**
   - *docker*: builds both architectures, each running `docker/smoke_test.py`, then publishes `X.Y.Z`, `X.Y` and `latest`.
   - *pypi*: fails if the tag doesn't match the version.
5. **Check from a clean machine:**
   - `pip install "netra-nmt[cpu]"` and the README examples;
   - `docker run --rm -p 8000:8000 ghcr.io/netra-ai-lab/netra-nmt:X.Y.Z`.

If the pypi workflow fails with `400 File already exists`, that version is already on PyPI. Release the next version instead of rerunning.

6. **Space:** after the PyPI release, update `netra-nmt[cpu]==X.Y.Z` in `training/release/space/requirements.txt`
   and upload the folder to the Space `Netra-Lab/Netra-NMT`.

## Facts for the product page

Measured on 2026-11-01 with a local build of `docker/Dockerfile` (linux/amd64, Intel i7-13700KF).

| | |
| :--- | :--- |
| Download | 286 MB compressed, 819 MB on disk |
| Memory | 591 MiB idle after start, 630 MiB after a few translations |
| Start | ready (`/health` ok) 3 s after `docker run` |
| Speed | ~42 ms per short English sentence through the REST API (greedy) |
| GPU | Not needed (CPU-only image) |
| Network | None after the pull: every model is inside the image |
