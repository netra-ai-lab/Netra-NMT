from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(
    ROOT / "data" / "processed" / "final.parquet"
)

print("\n===== FINAL DATASET =====")
print("Total pairs:", len(df))

print("\nAvg EN words:", df["en"].str.split().str.len().mean())
print("Avg KM chars:", df["km"].str.len().mean())

print("\nMin/Max EN length:",
      df["en"].str.split().str.len().min(),
      df["en"].str.split().str.len().max())

print("\nSources distribution:")
print(df["source"].value_counts())