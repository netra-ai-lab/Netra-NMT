import pandas as pd
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-600M"

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(ROOT / "data/processed/final.parquet")

print(f"Dataset size: {len(df):,}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

forced_bos_token_id = tokenizer.convert_tokens_to_ids("khm_Khmr")

BATCH_SIZE = 32   # 👉 adjust: 16–64 depending on VRAM


def translate_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=4
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


results = []

start = time.time()

print("\n🚀 Starting FAST batched generation...\n")

for i in tqdm(range(0, len(df), BATCH_SIZE)):

    batch = df.iloc[i:i+BATCH_SIZE]

    src_texts = batch["en"].tolist()

    preds = translate_batch(src_texts)

    for en, km, pred in zip(batch["en"], batch["km"], preds):
        results.append({
            "en": en,
            "km": km,
            "teacher_km": pred
        })

    # progress logging
    if i % (BATCH_SIZE * 50) == 0:
        elapsed = time.time() - start
        speed = len(results) / elapsed

        eta = (len(df) - len(results)) / speed / 60

        print(
            f"\n✔ {len(results):,}/{len(df):,} "
            f"| Speed: {speed:.2f} samples/sec "
            f"| ETA: {eta:.1f} min"
        )

# save
pd.DataFrame(results).to_parquet(
    ROOT / "data/processed/teacher_outputs.parquet",
    index=False
)

print("Done ✔")