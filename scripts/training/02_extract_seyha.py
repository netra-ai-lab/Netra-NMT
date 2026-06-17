from datasets import load_dataset
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "raw" / "seyha_en_kh_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading SeyhaLite/Translate-English-Khmer-All ...")
ds = load_dataset("SeyhaLite/Translate-English-Khmer-All", split="train")

df = ds.to_pandas()
print(f"Loaded {len(df):,} rows")
print("Columns:", df.columns.tolist())

df = df.rename(columns={"eng": "en", "kh": "km"})
df["source"] = "seyha_en_kh_all"
df = df[["en", "km", "source"]]

output_file = OUTPUT_DIR / "data.parquet"
df.to_parquet(output_file, index=False)
print(f"Saved {len(df):,} rows to {output_file}")
