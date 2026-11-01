"""Turn en/km pairs into the trainer's JSONL, prepared exactly like the
published model's training data (recipe/08c + 08d):

    {"source": "<2km> <english, NFC + lowercased>", "target": "<khmer, word-segmented>"}
    {"source": "<2en> <khmer, word-segmented>",      "target": "<english, as written>"}

    netra-nmt-train prepare my_pairs.tsv --out training/runs/NAME/data
    netra-nmt-train prepare my_pairs.tsv --val-data my_dev.tsv --out ...

Writes OUT/train.jsonl and OUT/valid.jsonl. Without --val-data, 5% of the
pairs (at most 2,000) are held out, chosen with a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from netra_nmt.config import LANG_MARKERS
from netra_nmt.khseg import segment_km
from netra_nmt.textnorm import normalize_en_source

from ..data import load_pairs


def rows(pairs):
    for en, km in pairs:
        km_seg = segment_km(km)
        yield {"source": f"{LANG_MARKERS['km']} {normalize_en_source(en)}", "target": km_seg}
        yield {"source": f"{LANG_MARKERS['en']} {km_seg}", "target": en}


def write(path: Path, pairs) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows(pairs):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="+", help="data specs (see netra_nmt_train/data.py)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-data", nargs="*", help="held-out pairs (default: 5%% of DATA)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    pairs = [p for spec in args.data for p in load_pairs(spec)]
    if args.val_data:
        valid = [p for spec in args.val_data for p in load_pairs(spec)]
    else:
        random.Random(args.seed).shuffle(pairs)
        n_val = max(1, min(2000, len(pairs) // 20))
        valid, pairs = pairs[:n_val], pairs[n_val:]
    args.out.mkdir(parents=True, exist_ok=True)
    n_train = write(args.out / "train.jsonl", pairs)
    n_valid = write(args.out / "valid.jsonl", valid)
    print(f"{len(pairs):,} train pairs -> {n_train:,} rows, {len(valid):,} validation pairs -> {n_valid:,} rows "
          f"(both directions) in {args.out}")


if __name__ == "__main__":
    main()
