from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(
    ROOT / "data" / "processed" / "dedup.parquet"
)

print("Before:", len(df))


# light checks only (NOT aggressive filtering)

def is_valid(row):
    en = str(row["en"])
    km = str(row["km"])

    # remove empty
    if len(en.strip()) < 2 or len(km.strip()) < 2:
        return False

    # remove extreme garbage
    if re.fullmatch(r"[\W_]+", en):
        return False

    if re.fullmatch(r"[\W_]+", km):
        return False

    # remove obvious mismatches
    if len(en) > 5 and len(km) < 3:
        return False

    return True


df = df[df.apply(is_valid, axis=1)]

print("After:", len(df))

df.to_parquet(
    ROOT / "data" / "processed" / "final.parquet",
    index=False
)