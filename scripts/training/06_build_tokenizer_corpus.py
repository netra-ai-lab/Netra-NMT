from pathlib import Path
import pandas as pd

print("\nSCRIPT STARTED")

ROOT = Path(__file__).resolve().parent.parent

df_path = ROOT / "data" / "processed" / "final.parquet"
print("Loading:", df_path)

df = pd.read_parquet(df_path)

print("Loaded rows:", len(df))

out_dir = ROOT / "tokenizer"
out_dir.mkdir(exist_ok=True)

out_file = out_dir / "corpus.txt"

print("Writing corpus...")

count = 0

with open(out_file, "w", encoding="utf-8") as f:
    for row in df.itertuples(index=False):

        en = str(row.en).strip()
        km = str(row.km).strip()

        if not en or not km:
            continue

        # EN → KM direction
        f.write("<2km> " + en + "\n")
        f.write(km + "\n")

        # KM → EN direction
        f.write("<2en> " + km + "\n")
        f.write(en + "\n")

        count += 4

print("Lines written:", count)
print("Saved to:", out_file)