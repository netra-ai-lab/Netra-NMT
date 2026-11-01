"""Parallel English-Khmer data as ``(en, km)`` pairs, from any of:

    alt                              preset: mutiyama/alt test split (1,018 usable pairs)
    alt:validation                   a preset with another split
    my_pairs.tsv                     en<TAB>km per line (no header)
    my_pairs.csv / .jsonl            columns/keys "en" and "km"
    user/name[:split]                a Hugging Face dataset with en/km columns
                                     (or a "translation" dict column)

Empty pairs are dropped. ``limit`` keeps the first N pairs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PRESETS = {"alt": ("mutiyama/alt", "test")}
EN_KEYS = ("en", "english", "eng", "source")
KM_KEYS = ("km", "khm", "kh", "khmer", "target")

FORMATS = __doc__


class DataError(ValueError):
    pass


def _pick(row: dict, keys: tuple[str, ...]):
    if "translation" in row and isinstance(row["translation"], dict):
        row = row["translation"]
    for k in keys:
        if row.get(k):
            return str(row[k]).strip()
    return None


def _clean(rows) -> list[tuple[str, str]]:
    pairs = []
    for row in rows:
        en, km = _pick(row, EN_KEYS), _pick(row, KM_KEYS)
        if en and km:
            pairs.append((en, km))
    return pairs


def load_pairs(spec: str, limit: int | None = None) -> list[tuple[str, str]]:
    name, _, split = str(spec).partition(":")
    path = Path(spec)
    if path.is_file():
        suffix = path.suffix.lower()
        with open(path, encoding="utf-8") as f:
            if suffix == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            elif suffix == ".csv":
                rows = list(csv.DictReader(f))
            else:
                rows = [dict(zip(("en", "km"), line.rstrip("\n").split("\t"))) for line in f if "\t" in line]
        pairs = _clean(rows)
    elif name in PRESETS or "/" in name:
        repo, default_split = PRESETS.get(name, (name, "test"))
        from datasets import load_dataset
        pairs = _clean(load_dataset(repo, split=split or default_split))
    else:
        raise DataError(f"Can't read '{spec}'.\n{FORMATS}")
    if not pairs:
        raise DataError(f"No en/km pairs in '{spec}'.\n{FORMATS}")
    return pairs[:limit] if limit else pairs
