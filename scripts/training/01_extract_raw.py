from datasets import load_dataset

# print("Downloading Dataset A...")
# ds_a = load_dataset(
#     "lyfeyvutha/nllb-en-km-316K",
#     split="train"
# )

print("Downloading Dataset B...")
ds_b = load_dataset(
    "Darayut/khmer-english-pairs-raw",
    split="train"
)

# ds_a.to_parquet(
#     "data/raw/nllb_en_km_316k/data.parquet"
# )

ds_b.to_parquet(
    "data/raw/khmer_english_pairs_raw/data.parquet"
)

print("Done.")
