# scripts/07_build_bilingual_dataset.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import json

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(
    ROOT / "data" / "processed" / "final.parquet"
)

print(f"Original pairs: {len(df):,}")

examples = []

for _, row in df.iterrows():

    en = str(row["en"]).strip()
    km = str(row["km"]).strip()

    if not en or not km:
        continue

    examples.append({
        "source": f"<2km> {en}",
        "target": km
    })

    examples.append({
        "source": f"<2en> {km}",
        "target": en
    })

print(f"Expanded examples: {len(examples):,}")

train_data, valid_data = train_test_split(
    examples,
    test_size=0.01,
    random_state=42,
    shuffle=True
)

print(f"Train: {len(train_data):,}")
print(f"Valid: {len(valid_data):,}")

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(exist_ok=True)

with open(out_dir / "bilingual_train.jsonl", "w", encoding="utf-8") as f:
    for x in train_data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open(out_dir / "bilingual_valid.jsonl", "w", encoding="utf-8") as f:
    for x in valid_data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("Done")