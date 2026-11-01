"""Check parallel data before training on it.

    netra-nmt-train check my_pairs.tsv [more specs ...]

For each spec: the number of usable pairs, duplicates, length statistics in
tokens (the model cuts sources at 256), pairs whose sides look swapped, and a
few examples. Exits 1 when a spec can't be read or has no usable pair.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

from ..data import DataError, load_pairs

KHMER = re.compile(r"[ក-៿]")


def report(spec: str, sp=None) -> bool:
    try:
        pairs = load_pairs(spec)
    except DataError as e:
        print(f"{spec}: {e}")
        return False
    print(f"\n{spec}: {len(pairs):,} pairs")
    dupes = len(pairs) - len(set(pairs))
    swapped = sum(1 for en, km in pairs if KHMER.search(en) and not KHMER.search(km))
    no_khmer = sum(1 for _, km in pairs if not KHMER.search(km))
    for side, texts in (("en", [p[0] for p in pairs]), ("km", [p[1] for p in pairs])):
        lengths = [len(sp.encode(t)) for t in texts] if sp else [len(t) for t in texts]
        unit = "tokens" if sp else "chars"
        print(f"  {side}: median {statistics.median(lengths):.0f} {unit}, max {max(lengths)}"
              + (f", {sum(n > 256 for n in lengths)} over 256 (cut)" if sp else ""))
    print(f"  duplicates: {dupes:,}   km side without Khmer: {no_khmer:,}   looks swapped: {swapped:,}")
    for en, km in pairs[:3]:
        print(f"    en: {en[:100]}\n    km: {km[:100]}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="+")
    args = parser.parse_args(argv)
    sp = None
    try:
        import sentencepiece as spm

        from ..checkpoint import DEFAULT_SPM
        if DEFAULT_SPM.exists():
            sp = spm.SentencePieceProcessor(model_file=str(DEFAULT_SPM))
    except ImportError:
        pass
    ok = all([report(spec, sp) for spec in args.data])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
