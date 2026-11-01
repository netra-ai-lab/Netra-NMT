from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]   # training/

merged_file = ROOT / "data" / "processed" / "merged.parquet"

df = pd.read_parquet(merged_file)

before = len(df)

df = df.drop_duplicates(
    subset=["en", "km"]
)

after = len(df)

print(f"Before: {before:,}")
print(f"After : {after:,}")
print(f"Removed: {before-after:,}")

df.to_parquet(
    ROOT / "data" / "processed" / "dedup.parquet",
    index=False
)