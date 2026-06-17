import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

sources = [
    (ROOT / "data" / "processed" / "final.parquet",                     "existing"),
    (ROOT / "data" / "raw" / "paracrawl_v2" / "data_clean.parquet",     "paracrawl_v2"),
    (ROOT / "data" / "raw" / "seyha_en_kh_all" / "data_clean.parquet",  "seyha_en_kh_all"),
]

frames = []
for path, label in sources:
    df = pd.read_parquet(path)
    # Normalise column names to [en, km]
    if "kh" in df.columns and "km" not in df.columns:
        df = df.rename(columns={"kh": "km"})
    df = df[["en", "km"]].copy()
    df["source"] = label
    print(f"{label}: {len(df):,} rows")
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
print(f"\nTotal merged: {len(merged):,}")

merged.to_parquet(PROCESSED / "merged.parquet", index=False)
print(f"Saved to {PROCESSED / 'merged.parquet'}")
