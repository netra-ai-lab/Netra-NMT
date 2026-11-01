"""Upload a release to the Hugging Face model repo Netra-Lab/Netra-NMT.

    python training/release/upload_model.py --model-folder training/bundles/model \
        --pytorch training/export_rope [--repo Netra-Lab/Netra-NMT] [--dry-run]

Uploads, in one commit:
    model/     the ONNX model folder the package runs (netra-model.json, graphs, tokenizer)
    pytorch/   the release weights for fine-tuning (model.safetensors, config.json, spm_32k.model)
    README.md  training/release/cards/model-card.md
    assets/    the card's figures (from the repo's assets/)

Log in first in your own terminal (`hf auth login`, a token that can write to
Netra-Lab). Prints the commit hash: put it in netra_nmt/models.py
MODEL_REVISION and training/netra_nmt_train/hub.py PUBLISHED_REVISION.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CARD = Path(__file__).parent / "cards" / "model-card.md"
CARD_ASSETS = ["header.png", "speed_benchmark.png", "metrics_benchmark.png", "model_architecture.png"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-folder", type=Path, required=True)
    p.add_argument("--pytorch", type=Path, required=True)
    p.add_argument("--repo", default="Netra-Lab/Netra-NMT")
    p.add_argument("--message", default="Netra NMT release: ONNX model folder + PyTorch weights")
    p.add_argument("--dry-run", action="store_true", help="assemble and list the files, upload nothing")
    args = p.parse_args()

    if not (args.model_folder / "netra-model.json").exists():
        raise SystemExit(f"{args.model_folder} is not a model folder (no netra-model.json)")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copytree(args.model_folder, stage / "model")
        (stage / "pytorch").mkdir()
        for name in ("model.safetensors", "config.json", "spm_32k.model"):
            shutil.copy2(args.pytorch / name, stage / "pytorch" / name)
        shutil.copy2(CARD, stage / "README.md")
        (stage / "assets").mkdir()
        for name in CARD_ASSETS:
            shutil.copy2(REPO_ROOT / "assets" / name, stage / "assets" / name)
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(stage)}  {f.stat().st_size / 1e6:.1f} MB")
        if args.dry_run:
            return
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
        info = api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model",
                                 commit_message=args.message)
        print(f"\nUploaded: {info.commit_url}\ncommit: {info.oid}")


if __name__ == "__main__":
    main()
