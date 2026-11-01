"""
08b_build_combined_dataset.py — build the continued-training dataset.

Combines three sources into the trainer's {"source", "target"} JSONL format
with <2km>/<2en> direction tokens (same convention as
08_build_bilingual_dataset.py):

  1. Darayut/bilingual-en-km          — already in {source, target} form, with
                                         direction tokens; train + validation.
  2. rinabuoy/Khm-Eng-Sent-Pairs      — raw EN/KM pairs; expanded both ways.
  3. rinabuoy/khmer-english-parallel  — raw EN/KM pairs; expanded both ways.

(2) and (3) are GATED on the Hub — your HF account must have accepted their
terms, and HF_TOKEN must be set, or load_dataset will 403.

Output:
  data/processed/combined_train.jsonl  = bilingual[train] + rinabuoy(2) + rinabuoy(3)
  data/processed/combined_valid.jsonl  = bilingual[validation]   (new data is train-only)

New rinabuoy examples are exact-deduped against themselves and against the
bilingual set (hash on "source\ttarget").

Usage
-----
    HF_TOKEN=... python scripts/training/08b_build_combined_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]   # training/
OUT_DIR = ROOT / "data" / "processed"

BILINGUAL_REPO = "Darayut/bilingual-en-km"
RINABUOY_REPOS = [
    "rinabuoy/Khm-Eng-Sent-Pairs",
    "rinabuoy/khmer-english-parallel",
]

EN_CANDIDATES = ["en", "english", "eng", "English", "EN", "source_en", "en_text"]
KM_CANDIDATES = ["km", "kh", "khmer", "khm", "Khmer", "KM", "target_km", "km_text"]


def _hash(source: str, target: str) -> int:
    return hash(source + "\t" + target)


def _detect_keys(cols: list[str]) -> tuple[str | None, str | None]:
    en = next((k for k in EN_CANDIDATES if k in cols), None)
    km = next((k for k in KM_CANDIDATES if k in cols), None)
    return en, km


def _iter_pairs(ds):
    """Yield (en, km) from a dataset, auto-detecting columns.

    Handles flat columns (en/km variants) and a nested 'translation' dict.
    """
    cols = ds.column_names
    en_key, km_key = _detect_keys(cols)

    if en_key and km_key:
        for row in ds:
            yield row.get(en_key), row.get(km_key)
        return

    if "translation" in cols:
        sample = ds[0]["translation"]
        tcols = list(sample.keys()) if isinstance(sample, dict) else []
        en_key, km_key = _detect_keys(tcols)
        if en_key and km_key:
            for row in ds:
                t = row["translation"]
                yield t.get(en_key), t.get(km_key)
            return

    raise RuntimeError(
        f"Could not find EN/KM columns.\n"
        f"  EN candidates: {EN_CANDIDATES}\n"
        f"  KM candidates: {KM_CANDIDATES}\n"
        f"  Columns found: {cols}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / "combined_train.jsonl"
    valid_path = args.out_dir / "combined_valid.jsonl"

    seen: set[int] = set()
    n_train = 0
    n_valid = 0

    # ── 1. bilingual-en-km (already source/target with direction tokens) ──
    print(f"Loading {BILINGUAL_REPO} …")
    bilingual = load_dataset(BILINGUAL_REPO)
    print(f"  splits: {dict((k, len(v)) for k, v in bilingual.items())}")

    train_split = "train"
    valid_split = "validation" if "validation" in bilingual else (
        "valid" if "valid" in bilingual else None
    )

    with train_path.open("w", encoding="utf-8") as ftrain:
        for row in bilingual[train_split]:
            src = str(row["source"]).strip()
            tgt = str(row["target"]).strip()
            if not src or not tgt:
                continue
            seen.add(_hash(src, tgt))
            ftrain.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
            n_train += 1
            if n_train % 1_000_000 == 0:
                print(f"  bilingual train written: {n_train:,}")
        print(f"  bilingual train total: {n_train:,}")

        # ── 2 & 3. rinabuoy sets → expand both directions, dedup, append ──
        for repo in RINABUOY_REPOS:
            print(f"Loading {repo} (gated) …")
            ds = load_dataset(repo)
            # Concatenate all splits — we want every pair for training.
            added = 0
            dropped = 0
            for split_name, split in ds.items():
                print(f"  {repo}[{split_name}]: {len(split):,} rows, cols={split.column_names}")
                for en, km in _iter_pairs(split):
                    if not en or not km:
                        dropped += 1
                        continue
                    en, km = str(en).strip(), str(km).strip()
                    if not en or not km:
                        dropped += 1
                        continue
                    for source, target in (
                        (f"<2km> {en}", km),   # English → Khmer
                        (f"<2en> {km}", en),   # Khmer → English
                    ):
                        h = _hash(source, target)
                        if h in seen:
                            dropped += 1
                            continue
                        seen.add(h)
                        ftrain.write(json.dumps({"source": source, "target": target},
                                                ensure_ascii=False) + "\n")
                        n_train += 1
                        added += 1
            print(f"  {repo}: added {added:,} examples ({dropped:,} dropped null/dup)")

    # ── validation = bilingual validation only ──
    with valid_path.open("w", encoding="utf-8") as fvalid:
        if valid_split is not None:
            for row in bilingual[valid_split]:
                src = str(row["source"]).strip()
                tgt = str(row["target"]).strip()
                if not src or not tgt:
                    continue
                fvalid.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
                n_valid += 1
        else:
            print("  [warn] no validation split found in bilingual set; valid file is empty")

    print("\n── Done ──")
    print(f"  train: {n_train:,}  →  {train_path}")
    print(f"  valid: {n_valid:,}  →  {valid_path}")


if __name__ == "__main__":
    main()
