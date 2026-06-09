from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUTPUT_DIR = "data/raw/nllb_en_km_316k"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

ds = load_dataset(
    "lyfeyvutha/nllb-en-km-316K",
    split="train"
)

# Extract nested records
records = ds[0]["train"]["data"]

print(f"Found {len(records):,} records")

# Convert to dataframe
df = pd.DataFrame(records)

# Rename columns
df = df.rename(
    columns={
        "eng_Latn": "en",
        "khm_Khmr": "km"
    }
)

print("\nColumns:")
print(df.columns.tolist())

print("\nSample:")
print(df.iloc[0].to_dict())

print("\nTotal rows:")
print(f"{len(df):,}")

# Save
output_file = f"{OUTPUT_DIR}/data.parquet"

df.to_parquet(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")