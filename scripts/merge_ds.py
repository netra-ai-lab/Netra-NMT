import pandas as pd

from pathlib import Path

Path("../data/processed").mkdir(
    parents=True,
    exist_ok=True
)

df1 = pd.read_parquet(
    "data/raw/nllb_en_km_316k/data.parquet"
)

df2 = pd.read_parquet(
    "data/raw/khmer_english_pairs_raw/data.parquet"
)

df2 = df2.rename(
    columns={"kh": "km"}
)

df1["source"] = "nllb_316k"
df2["source"] = "khmer_raw_200k"

merged = pd.concat(
    [df1, df2],
    ignore_index=True
)

print("Dataset A:", len(df1))
print("Dataset B:", len(df2))
print("Merged:", len(merged))

merged.to_parquet(
    "../data/processed/merged.parquet",
    index=False
)