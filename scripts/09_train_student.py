from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR

from model_mini_nllb import MiniNLLB

# ============================================================
# SPEED + STABILITY FLAGS
# ============================================================

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data/processed/bilingual_train.jsonl"
VALID_FILE = ROOT / "data/processed/bilingual_valid.jsonl"
TOKENIZER_PATH = ROOT / "tokenizer/spm_32k.model"
CKPT_DIR = ROOT / "checkpoints"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

MAX_LEN = 128

BATCH_SIZE = 32
EPOCHS = 3

LR = 1e-4   # ✅ FIX: lower LR (critical)

PAD_ID = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))

vocab_size = sp.vocab_size()
print("Vocab size:", vocab_size)

# ============================================================
# DATASET
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "valid": str(VALID_FILE)
    }
)

def encode(text):
    return sp.encode(text, out_type=int)

def preprocess(example):
    return {
        "src": encode(example["source"])[:MAX_LEN],
        "tgt": encode(example["target"])[:MAX_LEN]
    }

dataset = dataset.map(preprocess)
dataset.set_format(type="torch")

# ============================================================
# COLLATE (SAFE PAD + MASK READY)
# ============================================================

def collate(batch):

    src_batch, dec_in_batch, labels_batch = [], [], []

    for b in batch:

        src = b["src"].tolist()
        tgt = b["tgt"].tolist()

        if len(src) == 0 or len(tgt) == 0:
            continue

        src_batch.append(src)

        dec_in = [2] + tgt[:-1]
        dec_in_batch.append(dec_in)
        labels_batch.append(tgt)

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [PAD_ID] * (max_len - len(s)) for s in seqs],
            dtype=torch.long
        )

    return {
        "input_ids": pad(src_batch),
        "decoder_input_ids": pad(dec_in_batch),
        "labels": pad(labels_batch),
    }

# ============================================================
# DATALOADER (OPTIMIZED)
# ============================================================

train_loader = DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

valid_loader = DataLoader(
    dataset["valid"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

# ============================================================
# MODEL
# ============================================================

model = MiniNLLB(
    vocab_size=vocab_size,
    d_model=512,
    enc_layers=6,
    dec_layers=6,
    n_heads=8,
    ffn_dim=2048
).to(DEVICE)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.98),
    weight_decay=0.01
)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.cuda.amp.GradScaler()

# ============================================================
# TRAIN STEP
# ============================================================

def train_one_epoch(epoch):

    model.train()

    total_loss = 0
    step = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:

        src = batch["input_ids"].to(DEVICE, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():

            logits = model(src, dec_in)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

        # ========================================================
        # NAN CHECK (IMPORTANT FIX)
        # ========================================================
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ NaN/Inf loss detected, skipping step")
            continue

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        step += 1

        pbar.set_postfix(loss=round(loss.item(), 4))

    return total_loss / max(step, 1)

# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0
    step = 0

    for batch in valid_loader:

        src = batch["input_ids"].to(DEVICE, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():

            logits = model(src, dec_in)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

        if torch.isnan(loss):
            continue

        total_loss += loss.item()
        step += 1

    return total_loss / max(step, 1)

# ============================================================
# TRAIN LOOP
# ============================================================

best_val = float("inf")

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch+1}")
    print("=" * 60)

    train_loss = train_one_epoch(epoch)
    val_loss = evaluate()

    scheduler.step()  # ✅ FIX: LR schedule

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    ckpt_path = CKPT_DIR / f"epoch_{epoch}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_path / "model.pt")

    print(f"Saved checkpoint: {ckpt_path}")

    # best model
    if val_loss < best_val:

        best_val = val_loss

        torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")

        print("🔥 New best model saved!")