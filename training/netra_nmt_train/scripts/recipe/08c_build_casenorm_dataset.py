"""
08c_build_casenorm_dataset.py — combined dataset, case-normalized + Khmer-segmented.

Same three sources and {"source","target"} format as
08b_build_combined_dataset.py, with TWO changes:
  1. English is lowercased (via netra_nmt.textnorm.normalize_en_source) ONLY when
     it is the en2km source (the text after ``<2km>``). English kept as the km2en
     *target* retains its natural case.
  2. Khmer is word-segmented with khmercut (via netra_nmt.khseg.segment_km)
     everywhere it appears — as the en2km *target* and as the km2en *source* — so
     it matches the segmented tokenizer. This must pair with the *_kmseg SPM model
     and desegment_km at inference.

  1. Darayut/bilingual-en-km          — already {source, target} with markers;
                                         normalize the remainder of ``<2km> …``.
  2. rinabuoy/Khm-Eng-Sent-Pairs      — raw EN/KM; expand both ways.
  3. rinabuoy/khmer-english-parallel  — raw EN/KM; expand both ways.

(2) and (3) are GATED — HF_TOKEN must be set and terms accepted.

Output:
  data/processed/combined_casenorm_kmseg_train.jsonl
  data/processed/combined_casenorm_kmseg_valid.jsonl

Usage
-----
    HF_TOKEN=... python scripts/training/08c_build_casenorm_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from netra_nmt.textnorm import normalize_en_source
from netra_nmt.khseg import segment_km

ROOT = Path(__file__).resolve().parents[3]   # training/
OUT_DIR = ROOT / "data" / "processed"

BILINGUAL_REPO = "Darayut/bilingual-en-km"
RINABUOY_REPOS = [
    "rinabuoy/Khm-Eng-Sent-Pairs",
    "rinabuoy/khmer-english-parallel",
]

EN_CANDIDATES = ["en", "english", "eng", "English", "EN", "source_en", "en_text"]
KM_CANDIDATES = ["km", "kh", "khmer", "khm", "Khmer", "KM", "target_km", "km_text"]

EN2KM = "<2km>"
KM2EN = "<2en>"


def _hash(source: str, target: str) -> int:
    return hash(source + "\t" + target)


def _detect_keys(cols: list[str]) -> tuple[str | None, str | None]:
    en = next((k for k in EN_CANDIDATES if k in cols), None)
    km = next((k for k in KM_CANDIDATES if k in cols), None)
    return en, km


def _iter_pairs(ds):
    """Yield (en, km) from a dataset, auto-detecting flat or nested columns."""
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
        f"Could not find EN/KM columns.\n  EN {EN_CANDIDATES}\n  KM {KM_CANDIDATES}\n  cols={cols}"
    )


def _process_bilingual(source: str, target: str) -> tuple[str, str]:
    """
    Normalize a pre-built {source, target} bilingual row:
      * en2km (``<2km> {en}`` → km): lowercase the English source, segment the
        Khmer target.
      * km2en (``<2en> {km}`` → en): segment the Khmer source, keep English case.
    """
    src = source.strip()
    tgt = target.strip()
    if src.startswith(EN2KM):
        rest = src[len(EN2KM):].strip()
        return f"{EN2KM} {normalize_en_source(rest)}", segment_km(tgt)
    if src.startswith(KM2EN):
        rest = src[len(KM2EN):].strip()
        return f"{KM2EN} {segment_km(rest)}", tgt
    return src, tgt  # unknown marker — leave untouched


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / "combined_casenorm_kmseg_train.jsonl"
    valid_path = args.out_dir / "combined_casenorm_kmseg_valid.jsonl"

    seen: set[int] = set()
    n_train = 0
    n_valid = 0

    print(f"Loading {BILINGUAL_REPO} …")
    bilingual = load_dataset(BILINGUAL_REPO)
    print(f"  splits: {dict((k, len(v)) for k, v in bilingual.items())}")

    valid_split = "validation" if "validation" in bilingual else (
        "valid" if "valid" in bilingual else None
    )

    with train_path.open("w", encoding="utf-8") as ftrain:
        # ── 1. bilingual-en-km: normalize en2km source, keep targets ──
        for row in bilingual["train"]:
            src, tgt = _process_bilingual(str(row["source"]), str(row["target"]))
            if not src or not tgt:
                continue
            seen.add(_hash(src, tgt))
            ftrain.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
            n_train += 1
            if n_train % 1_000_000 == 0:
                print(f"  bilingual train written: {n_train:,}")
        print(f"  bilingual train total: {n_train:,}")

        # ── 2 & 3. rinabuoy: en2km source lowercased, km2en target keeps case ──
        for repo in RINABUOY_REPOS:
            print(f"Loading {repo} (gated) …")
            ds = load_dataset(repo)
            added = dropped = 0
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
                    km_seg = segment_km(km)
                    for source, target in (
                        (f"{EN2KM} {normalize_en_source(en)}", km_seg),  # en2km: src lowercased, Khmer tgt segmented
                        (f"{KM2EN} {km_seg}", en),                       # km2en: Khmer src segmented, English tgt keeps case
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

    # ── validation = bilingual validation only (same normalization) ──
    with valid_path.open("w", encoding="utf-8") as fvalid:
        if valid_split is not None:
            for row in bilingual[valid_split]:
                src, tgt = _process_bilingual(str(row["source"]), str(row["target"]))
                if not src or not tgt:
                    continue
                fvalid.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
                n_valid += 1
        else:
            print("  [warn] no validation split found; valid file is empty")

    print("\n── Done ──")
    print(f"  train: {n_train:,}  →  {train_path}")
    print(f"  valid: {n_valid:,}  →  {valid_path}")


if __name__ == "__main__":
    main()
