from datasets import load_dataset
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "raw" / "paracrawl_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading KrorngAI/ParaCrawl-English-Khmer-v2 ...")
ds = load_dataset("KrorngAI/ParaCrawl-English-Khmer-v2", split="train")

df = ds.to_pandas()
print(f"Loaded {len(df):,} rows")
print("Columns:", df.columns.tolist())

df = df.rename(columns={"english": "en", "khmer": "km"})
df = df.drop(columns=["id"], errors="ignore")
df["source"] = "paracrawl_v2"
df = df[["en", "km", "source"]]

output_file = OUTPUT_DIR / "data.parquet"
df.to_parquet(output_file, index=False)
print(f"Saved {len(df):,} rows to {output_file}")
