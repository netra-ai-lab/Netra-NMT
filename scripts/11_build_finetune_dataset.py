"""
Build a fine-tuning JSONL dataset from:
  - mutiyama/alt  (train split)
  - rinabuoy/khmer-english-parallel

Both are high-quality, human-curated parallel corpora.
Output format matches bilingual_train.jsonl (bidirectional, direction-prefixed).
"""

import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)


# ── helpers ──────────────────────────────────────────────────────────────────

def to_pairs(en_list, km_list, source_label):
    pairs = []
    for en, km in zip(en_list, km_list):
        en = str(en).strip()
        km = str(km).strip()
        if en and km:
            pairs.append({"en": en, "km": km, "source": source_label})
    return pairs


def detect_key(available, candidates):
    return next((k for k in candidates if k in available), None)


# ── load mutiyama/alt (train split) ──────────────────────────────────────────

print("Loading mutiyama/alt [train] …")
alt_ds = load_dataset("mutiyama/alt", split="train", trust_remote_code=True)

sample   = alt_ds[0]["translation"]
keys     = set(sample.keys())
en_key   = detect_key(keys, ["en", "english", "eng"])
kh_key   = detect_key(keys, ["khm", "km", "kh", "khmer"])

if not en_key or not kh_key:
    raise RuntimeError(f"Could not find EN/KH columns in alt. Available: {sorted(keys)}")

print(f"  EN key: '{en_key}'  |  KH key: '{kh_key}'")

alt_en, alt_km = [], []
for row in alt_ds:
    en = row["translation"].get(en_key, "")
    km = row["translation"].get(kh_key, "")
    if en and km and en.strip() and km.strip():
        alt_en.append(en.strip())
        alt_km.append(km.strip())

print(f"  {len(alt_en):,} pairs loaded from alt")
alt_pairs = to_pairs(alt_en, alt_km, "alt")


# ── load rinabuoy/khmer-english-parallel ─────────────────────────────────────

print("\nLoading rinabuoy/khmer-english-parallel …")
try:
    rina_ds = load_dataset("rinabuoy/khmer-english-parallel", split="train", trust_remote_code=True)
except Exception:
    rina_ds = load_dataset("rinabuoy/khmer-english-parallel", trust_remote_code=True)
    rina_ds = rina_ds[list(rina_ds.keys())[0]]

cols   = rina_ds.column_names
print(f"  Columns: {cols}")
en_key = detect_key(cols, ["en", "english", "eng", "English"])
kh_key = detect_key(cols, ["km", "kh", "khmer", "khm", "Khmer"])

if not en_key or not kh_key:
    raise RuntimeError(f"Could not find EN/KH columns in rinabuoy. Available: {cols}")

print(f"  EN key: '{en_key}'  |  KH key: '{kh_key}'")

rina_en = [str(r[en_key]).strip() for r in rina_ds]
rina_km = [str(r[kh_key]).strip() for r in rina_ds]
rina_pairs = to_pairs(rina_en, rina_km, "rinabuoy")
print(f"  {len(rina_pairs):,} pairs loaded from rinabuoy")


# ── merge & deduplicate ───────────────────────────────────────────────────────

all_pairs = alt_pairs + rina_pairs
print(f"\nTotal before dedup: {len(all_pairs):,}")

seen = set()
deduped = []
for p in all_pairs:
    key = (p["en"].lower(), p["km"])
    if key not in seen:
        seen.add(key)
        deduped.append(p)

print(f"Total after dedup : {len(deduped):,}  (removed {len(all_pairs) - len(deduped):,})")


# ── expand bidirectionally ────────────────────────────────────────────────────

examples = []
for p in deduped:
    examples.append({"source": f"<2km> {p['en']}", "target": p["km"]})
    examples.append({"source": f"<2en> {p['km']}", "target": p["en"]})

print(f"Bidirectional examples: {len(examples):,}")

random.shuffle(examples)
split = int(len(examples) * 0.95)
train_data = examples[:split]
valid_data = examples[split:]

print(f"Train: {len(train_data):,}  |  Valid: {len(valid_data):,}")


# ── save ──────────────────────────────────────────────────────────────────────

train_path = OUT_DIR / "finetune_train.jsonl"
valid_path = OUT_DIR / "finetune_valid.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for x in train_data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open(valid_path, "w", encoding="utf-8") as f:
    for x in valid_data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print(f"\nSaved → {train_path}")
print(f"Saved → {valid_path}")
print("Done.")
