<!-- From the Netra Lab kit (templates/repo), after Netra-OCR docs/RELEASING.md.
     Adapt: drop the Docker parts if the repo has no image; fill "Facts" with measured numbers only. -->
# Releasing Netra NMT

A release is a version tag. It publishes:
- the package `netra-nmt` to PyPI (`.github/workflows/pypi.yml`);
- the app image `ghcr.io/netra-ai-lab/netra-nmt` for `linux/amd64` and `linux/arm64` (`.github/workflows/docker.yml`).

The models live in the Hugging Face repo [Netra-Lab/Netra-NMT](https://huggingface.co/Netra-Lab/Netra-NMT).

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
   - upload the model folders to `Netra-Lab/Netra-NMT` (export commands in `training/README.md`);
   - set `MODEL_REVISION` in `netra_nmt/models.py` to that commit.
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

## Facts for the product page

<!-- Measured, with the machine named: download size (compressed / on disk), memory (idle / peak), speed, GPU needed, network after the pull. -->

| | |
| :--- | :--- |
| Download | |
| Memory | |
| Speed | |
| GPU | Not needed (CPU-only image) |
| Network | None after the pull: every model is inside the image |
