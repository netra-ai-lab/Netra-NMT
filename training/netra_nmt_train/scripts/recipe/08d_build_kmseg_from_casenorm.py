"""
08d_build_kmseg_from_casenorm.py — Khmer-segment the existing casenorm dataset.

The practical entrypoint when data/processed/final.parquet is unavailable: it
reads the already-built, case-normalized dataset

    data/processed/combined_casenorm_{train,valid}.jsonl

and word-segments the Khmer side (via netra_nmt.khseg.segment_km) wherever it
appears — the en2km *target* and the km2en *source* — producing

    data/processed/combined_casenorm_kmseg_{train,valid}.jsonl   (for 09_train_student)
    tokenizer/corpus.txt                                          (for 07_train_tokenizer)

English case-normalization is already baked into the input, so it is left as-is
(segment_km only touches Khmer; markers and English are untouched).

Pipeline:  08d  →  07_train_tokenizer (spm_32k_kmseg)  →  09_train_student

Usage
-----
    python scripts/training/08d_build_kmseg_from_casenorm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # training/
sys.path.insert(0, str(ROOT))
from netra_nmt.khseg import segment_km

DATA = ROOT / "data" / "processed"
TOKENIZER_DIR = ROOT / "tokenizer"

EN2KM = "<2km>"
KM2EN = "<2en>"


def _segment_row(source: str, target: str) -> tuple[str, str]:
    """Segment the Khmer side of a {source, target} row based on its marker."""
    src = source.strip()
    tgt = target.strip()
    if src.startswith(EN2KM):                       # en2km: Khmer target
        return src, segment_km(tgt)
    if src.startswith(KM2EN):                        # km2en: Khmer source
        rest = src[len(KM2EN):].strip()
        return f"{KM2EN} {segment_km(rest)}", tgt
    return src, tgt


def _transform(in_path: Path, out_path: Path, corpus_fh=None) -> int:
    n = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src, tgt = _segment_row(str(row["source"]), str(row["target"]))
            if not src or not tgt:
                continue
            fout.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
            if corpus_fh is not None:
                corpus_fh.write(src + "\n")
                corpus_fh.write(tgt + "\n")
            n += 1
            if n % 500_000 == 0:
                print(f"  {in_path.name}: {n:,} rows")
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--corpus", type=Path, default=TOKENIZER_DIR / "corpus.txt",
                   help="Tokenizer corpus to (re)write from the train split.")
    args = p.parse_args(argv)
    args.corpus.parent.mkdir(parents=True, exist_ok=True)

    pairs = [
        (args.data_dir / "combined_casenorm_train.jsonl",
         args.data_dir / "combined_casenorm_kmseg_train.jsonl"),
        (args.data_dir / "combined_casenorm_valid.jsonl",
         args.data_dir / "combined_casenorm_kmseg_valid.jsonl"),
    ]
    for src, _ in pairs:
        if not src.exists():
            sys.exit(f"Input not found: {src}")

    # Train split also feeds the tokenizer corpus; valid does not.
    with args.corpus.open("w", encoding="utf-8") as corpus_fh:
        n_train = _transform(pairs[0][0], pairs[0][1], corpus_fh=corpus_fh)
    n_valid = _transform(pairs[1][0], pairs[1][1], corpus_fh=None)

    print("\n── Done ──")
    print(f"  train: {n_train:,}  →  {pairs[0][1]}")
    print(f"  valid: {n_valid:,}  →  {pairs[1][1]}")
    print(f"  corpus: {args.corpus}  (source+target lines from train)")
    print("\nNext:  python scripts/training/07_train_tokenizer.py"
          "   then   torchrun scripts/training/09_train_student.py")


if __name__ == "__main__":
    main()
