import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(
    ROOT / "data/processed/teacher_outputs.parquet"
)

def quality_score(row):
    pred = row["teacher_km"]

    # remove garbage outputs
    if len(pred) < 3:
        return False

    if re.fullmatch(r"[\W_]+", pred):
        return False

    # avoid repetition loops
    if len(set(pred)) < 3:
        return False

    return True

df = df[df.apply(quality_score, axis=1)]

df.to_parquet(
    ROOT / "data/processed/teacher_filtered.parquet",
    index=False
)

print("Filtered size:", len(df))