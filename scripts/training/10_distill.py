"""
distill.py  —  Sequence-level knowledge distillation for Mini NLLB
===================================================================
Teacher: facebook/nllb-200-distilled-600M  (decoded sentences, no logits)
Student: MiniNLLB  (our small model)

Dataset columns expected in the parquet files:
  en          — original English sentence
  km          — original human Khmer reference
  teacher_km  — NLLB-600M greedy translation of en → km
  teacher_en  — NLLB-600M greedy translation of km → en

Training directions (both run in the same epoch):
  EN → KM :  src=en,  teacher target=teacher_km,  human target=km
  KM → EN :  src=km,  teacher target=teacher_en,  human target=en

Loss:
  L = α * CE(student, teacher_target)   ← sequence distillation
    + (1-α) * CE(student, human_target)  ← ground truth signal

Usage
-----
  # Fresh distillation run (loads pretrained checkpoint automatically):
  python distill.py --base-checkpoint ../checkpoints/best/checkpoint.pt

  # Resume a distillation run:
  python distill.py --extra-epochs 5

  # Dual GPU:
  torchrun --nproc_per_node=2 distill.py --base-checkpoint ../checkpoints/best/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import sentencepiece as spm
from tqdm import tqdm

from netra_nmt.model import NetraNMT as MiniNLLB


# ============================================================
# DISTRIBUTED HELPERS  (identical to train.py)
# ============================================================

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank  = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world
    return 0, 1

def cleanup_ddp(world: int):
    if world > 1:
        dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0


# ============================================================
# PATHS
# ============================================================

ROOT           = Path(__file__).resolve().parent.parent
TEACHER_KM     = ROOT / "data/processed/teacher_km_outputs.parquet"
TEACHER_EN     = ROOT / "data/processed/teacher_en_outputs.parquet"
TOKENIZER_PATH = ROOT / "tokenizer/spm_32k.model"
CKPT_DIR       = ROOT / "checkpoints_distill"
LOG_FILE       = ROOT / "distill_log.jsonl"
BASE_CKPT      = ROOT / "checkpoints/best/checkpoint.pt"  # pretrain weights to start from


# ============================================================
# CONFIG
# ============================================================

CFG = dict(
    # model — must match the pretrained checkpoint exactly
    d_model    = 512,
    enc_layers = 6,
    dec_layers = 6,
    n_heads    = 8,
    ffn_dim    = 2048,
    max_len    = 128,
    dropout    = 0.1,

    # distillation
    alpha            = 0.3,
    label_smoothing  = 0.1,

    # training
    max_len_tokens   = 128,
    batch_size       = 32,
    grad_accum_steps = 2,
    epochs           = 15,
    warmup_steps     = 500,    # short warmup for warm restart (not cold start)
    peak_lr          = 2e-5,   # moderate restart peak — not as high as first run
    min_lr           = 1e-7,
    weight_decay     = 0.01,
    grad_clip        = 1.0,

    # special tokens
    pad_id = 0,
    bos_id = 2,
    eos_id = 3,

    # misc
    seed               = 42,
    num_workers        = 4,
    compile_model      = True,
    log_every          = 50,
    sample_every_epoch = True,
    n_samples          = 3,
    valid_split        = 0.02,

    # ── warm restart ──────────────────────────────────────────
    # Set to True to reset the LR schedule from step 0 when resuming.
    # Use this when the LR has decayed too low and training has stalled.
    # After one session with this True, set back to False so subsequent
    # --extra-epochs runs don't reset the schedule again.
    reset_schedule     = False,
)


# ============================================================
# LR SCHEDULE  (identical to train.py)
# ============================================================

def get_lr(step, warmup, total, peak, min_lr):
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + (peak - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================================================
# DISTILLATION LOSS
# ============================================================

class DistillationLoss(nn.Module):
    """
    Sequence-level distillation loss:

      L = α  * CE(logits, teacher_labels)   — match teacher output sentences
        + (1-α) * CE(logits, human_labels)   — still learn from gold references

    Both terms use label smoothing.
    When teacher_labels is None (missing teacher output), falls back to
    human labels only — so the dataset does not have to be 100% complete.
    """

    def __init__(self, vocab_size: int, pad_id: int,
                 smoothing: float = 0.1, alpha: float = 0.7):
        super().__init__()
        self.pad_id    = pad_id
        self.smoothing = smoothing
        self.alpha     = alpha
        self.vocab     = vocab_size

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Label-smoothed CE, ignoring pad positions."""
        # logits : (N, V)   targets : (N,)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (self.vocab - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth[:, self.pad_id] = 0.0
            mask = targets == self.pad_id
            smooth[mask] = 0.0
        loss    = -(smooth * log_probs).sum(dim=-1)
        non_pad = (~mask).sum()
        return loss.sum() / non_pad.clamp(min=1)

    def forward(
        self,
        logits:         torch.Tensor,           # (N, V)  — flattened student logits
        teacher_labels: torch.Tensor,           # (N,)    — flattened teacher token ids
        human_labels:   torch.Tensor,           # (N,)    — flattened human token ids
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_loss = self._ce(logits, teacher_labels)
        human_loss   = self._ce(logits, human_labels)
        total        = self.alpha * teacher_loss + (1 - self.alpha) * human_loss
        return total, teacher_loss, human_loss


# ============================================================
# DATASET
# ============================================================

class BilingualDistillDataset(Dataset):
    """
    Loads the two teacher parquet files, merges them, and exposes
    both translation directions as individual examples.

    Each example is a dict:
        src_text      — source string
        tgt_text      — human reference target string
        teacher_text  — teacher translated target string
        direction     — "en2km" or "km2en"
    """

    def __init__(self, km_parquet: Path, en_parquet: Path,
                 split: str = "train", valid_frac: float = 0.02, seed: int = 42):

        df_km = pd.read_parquet(km_parquet)[["en", "km", "teacher_km"]].dropna()
        df_en = pd.read_parquet(en_parquet)[["en", "km", "teacher_en"]].dropna()

        # EN → KM examples
        en2km = pd.DataFrame({
            "src":     df_km["en"],
            "tgt":     df_km["km"],
            "teacher": df_km["teacher_km"],
        })

        # KM → EN examples
        km2en = pd.DataFrame({
            "src":     df_en["km"],
            "tgt":     df_en["en"],
            "teacher": df_en["teacher_en"],
        })

        combined = pd.concat([en2km, km2en], ignore_index=True)
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_valid = max(1, int(len(combined) * valid_frac))
        if split == "valid":
            self.data = combined.iloc[:n_valid].reset_index(drop=True)
        else:
            self.data = combined.iloc[n_valid:].reset_index(drop=True)

        print(f"  [{split}] {len(self.data):,} examples  "
              f"(en2km: ~{len(en2km):,}  km2en: ~{len(km2en):,})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "src":     row["src"],
            "tgt":     row["tgt"],
            "teacher": row["teacher"],
        }


# ============================================================
# COLLATE
# ============================================================

def make_distill_collate(sp: spm.SentencePieceProcessor,
                         pad_id: int, bos_id: int, max_len: int):
    """
    Tokenises on the fly and returns:
      input_ids              (B, S)  — encoded source
      src_key_padding_mask   (B, S)
      decoder_input_ids      (B, T)  — [BOS] + teacher[:-1]  (teacher forcing)
      tgt_key_padding_mask   (B, T)
      teacher_labels         (B, T)  — teacher token ids (prediction target, primary)
      human_labels           (B, T)  — human reference token ids (prediction target, secondary)

    The decoder is driven by teacher_labels (teacher forcing on the teacher sequence)
    because we want the student to learn to reproduce the teacher's output.
    Human labels are used only in the loss, not for decoder input.
    """

    def encode(text: str) -> list[int]:
        return sp.encode(text.strip(), out_type=int)[:max_len]

    def pad_batch(seqs: list[list[int]]):
        max_l  = max(len(s) for s in seqs)
        padded = torch.tensor(
            [s + [pad_id] * (max_l - len(s)) for s in seqs],
            dtype=torch.long,
        )
        return padded, (padded == pad_id)

    def collate(batch):
        src_seqs, dec_in_seqs, teacher_seqs, human_seqs = [], [], [], []

        for b in batch:
            src     = encode(b["src"])
            teacher = encode(b["teacher"])
            human   = encode(b["tgt"])

            if not src or not teacher or not human:
                continue

            src_seqs.append(src)
            # Decoder is driven by teacher sequence (teacher forcing)
            dec_in_seqs.append([bos_id] + teacher[:-1][:max_len - 1])
            teacher_seqs.append(teacher[:max_len])
            human_seqs.append(human[:max_len])

        if not src_seqs:
            return None

        src_ids,     src_mask  = pad_batch(src_seqs)
        dec_in_ids,  tgt_mask  = pad_batch(dec_in_seqs)
        teacher_ids, _         = pad_batch(teacher_seqs)
        human_ids,   _         = pad_batch(human_seqs)

        return {
            "input_ids":             src_ids,
            "src_key_padding_mask":  src_mask,
            "decoder_input_ids":     dec_in_ids,
            "tgt_key_padding_mask":  tgt_mask,
            "teacher_labels":        teacher_ids,
            "human_labels":          human_ids,
        }

    return collate


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def save_checkpoint(path: Path, model_raw, optimizer, scaler,
                    epoch, global_step, val_loss, cfg, vocab_size):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "global_step": global_step,
        "val_loss":    val_loss,
        "model":       model_raw.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict(),
        "cfg":         cfg,
        "vocab_size":  vocab_size,
    }, path / "checkpoint.pt")


def load_checkpoint(path: Path, model_raw, optimizer, scaler, device):
    ckpt = torch.load(path / "checkpoint.pt", map_location=device)
    sd   = ckpt["model"]

    # Normalise checkpoint keys to plain first
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # Re-add prefix if the target model (compiled) expects it
    model_keys = set(dict(model_raw.named_parameters()).keys())
    if any(k.startswith("_orig_mod.") for k in model_keys):
        sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

    model_raw.load_state_dict(sd)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["val_loss"]


def load_weights_only(path: Path, model_raw, device):
    """
    Load just model weights from a pretrain checkpoint — no optimizer state.
    Handles both compiled (_orig_mod. prefix) and uncompiled checkpoints,
    and loads into both compiled and uncompiled target models correctly.
    """
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model", ckpt)

    # Normalise checkpoint keys to plain (strip _orig_mod. if present)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # Check if the target model (after compile) expects _orig_mod. prefix
    model_keys = set(dict(model_raw.named_parameters()).keys())
    if any(k.startswith("_orig_mod.") for k in model_keys):
        sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

    missing, unexpected = model_raw.load_state_dict(sd, strict=False)
    # Filter out expected cross-ties (lm_head shares decoder embed weight)
    missing   = [k for k in missing   if "lm_head" not in k]
    unexpected = [k for k in unexpected if "lm_head" not in k]
    if missing:
        print(f"  [warn] Missing keys:    {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("  Weights loaded cleanly ✓")


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    candidates = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.startswith("epoch_"):
            try:
                n = int(p.name.split("_")[1])
                if (p / "checkpoint.pt").exists():
                    candidates.append((n, p))
            except (IndexError, ValueError):
                pass
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


# ============================================================
# TRANSLATION SAMPLES  (for sanity check after each epoch)
# ============================================================

@torch.no_grad()
def print_translation_samples(model_raw, sp, valid_loader, device,
                               bos_id, eos_id, pad_id, n=3):
    model_raw.eval()
    collected = 0
    print("\n" + "─" * 64)
    print("Distillation samples (greedy)  —  SRC / TEACHER / STUDENT")
    print("─" * 64)

    for batch in valid_loader:
        if batch is None:
            continue
        B = batch["input_ids"].size(0)

        for i in range(min(B, n - collected)):
            src_ids      = batch["input_ids"][i:i+1].to(device)
            s_mask       = batch["src_key_padding_mask"][i:i+1].to(device)
            teacher_ids  = batch["teacher_labels"][i].tolist()
            human_ids    = batch["human_labels"][i].tolist()

            def clean(ids):
                return sp.decode([t for t in ids if t not in (0, bos_id, eos_id)])

            src_text     = clean(batch["input_ids"][i].tolist())
            teacher_text = clean(teacher_ids)
            human_text   = clean(human_ids)

            gen = model_raw.generate(
                src_ids, bos_token_id=bos_id, eos_token_id=eos_id,
                max_new_tokens=60, src_key_padding_mask=s_mask,
            )
            student_text = clean(gen[0].tolist())

            print(f"  SRC     : {src_text}")
            print(f"  TEACHER : {teacher_text}")
            print(f"  HUMAN   : {human_text}")
            print(f"  STUDENT : {student_text}")
            print()
            collected += 1
            if collected >= n:
                break
        if collected >= n:
            break

    print("─" * 64 + "\n")


# ============================================================
# TRAIN / EVAL STEPS
# ============================================================

def train_one_epoch(epoch, model, loader, loss_fn, optimizer, scaler,
                    device, amp_dtype, cfg, global_step, total_steps, rank):
    model.train()
    total_loss = total_teacher = total_human = 0.0
    n_steps = 0
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc=f"Distill epoch {epoch+1}", disable=not is_main(rank))
    optimizer.zero_grad(set_to_none=True)

    for local_step, batch in enumerate(pbar):
        if batch is None:
            continue

        src      = batch["input_ids"].to(device, non_blocking=True)
        dec_in   = batch["decoder_input_ids"].to(device, non_blocking=True)
        t_labels = batch["teacher_labels"].to(device, non_blocking=True)
        h_labels = batch["human_labels"].to(device, non_blocking=True)
        s_mask   = batch["src_key_padding_mask"].to(device, non_blocking=True)
        t_mask   = batch["tgt_key_padding_mask"].to(device, non_blocking=True)

        lr = get_lr(global_step, cfg["warmup_steps"], total_steps,
                    cfg["peak_lr"], cfg["min_lr"])
        set_lr(optimizer, lr)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in,
                           src_key_padding_mask=s_mask,
                           tgt_key_padding_mask=t_mask)

            B, T, V = logits.shape   # T = teacher sequence length

            # human_labels may be a different length than teacher sequence —
            # truncate or pad to T so all three tensors share the same shape
            h = h_labels[:, :T]                          # truncate if longer
            if h.size(1) < T:                            # pad if shorter
                pad = torch.full((B, T - h.size(1)), cfg["pad_id"],
                                 dtype=torch.long, device=device)
                h = torch.cat([h, pad], dim=1)

            N = B * T
            loss, t_loss, h_loss = loss_fn(
                logits.reshape(N, V),
                t_labels[:, :T].reshape(N),
                h.reshape(N),
            )
            loss = loss / cfg["grad_accum_steps"]

        if not torch.isfinite(loss):
            if is_main(rank):
                print(f"  ⚠️  Non-finite loss at step {global_step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (local_step + 1) % cfg["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main(rank) and global_step % cfg["log_every"] == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  step {global_step:6d}  "
                    f"loss {loss.item()*cfg['grad_accum_steps']:.4f}  "
                    f"(teacher {t_loss.item():.4f}  human {h_loss.item():.4f})  "
                    f"lr {lr:.2e}  gnorm {grad_norm:.3f}  "
                    f"{(cfg['log_every']*cfg['batch_size']*cfg.get('world_size',1)/elapsed):,.0f} tok/s"
                )
                t0 = time.perf_counter()

        total_loss    += loss.item() * cfg["grad_accum_steps"]
        total_teacher += t_loss.item()
        total_human   += h_loss.item()
        n_steps       += 1

        pbar.set_postfix(
            loss=f"{loss.item()*cfg['grad_accum_steps']:.4f}",
            t=f"{t_loss.item():.4f}",
            h=f"{h_loss.item():.4f}",
        )

    s = max(n_steps, 1)
    return total_loss / s, total_teacher / s, total_human / s, global_step


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_dtype, rank):
    model.eval()
    total_loss = 0.0
    n_steps    = 0

    for batch in tqdm(loader, desc="  Eval", disable=not is_main(rank), leave=False):
        if batch is None:
            continue

        src      = batch["input_ids"].to(device, non_blocking=True)
        dec_in   = batch["decoder_input_ids"].to(device, non_blocking=True)
        t_labels = batch["teacher_labels"].to(device, non_blocking=True)
        h_labels = batch["human_labels"].to(device, non_blocking=True)
        s_mask   = batch["src_key_padding_mask"].to(device, non_blocking=True)
        t_mask   = batch["tgt_key_padding_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in,
                           src_key_padding_mask=s_mask,
                           tgt_key_padding_mask=t_mask)

            B, T, V = logits.shape
            h = h_labels[:, :T]
            if h.size(1) < T:
                pad = torch.full((B, T - h.size(1)), 0,
                                 dtype=torch.long, device=device)
                h = torch.cat([h, pad], dim=1)

            N = B * T
            loss, _, _ = loss_fn(
                logits.reshape(N, V),
                t_labels[:, :T].reshape(N),
                h.reshape(N),
            )

        if torch.isfinite(loss):
            total_loss += loss.item()
            n_steps    += 1

    return total_loss / max(n_steps, 1)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sequence-level distillation for Mini NLLB")
    parser.add_argument("--extra-epochs", type=int, default=None,
                        help="How many MORE epochs to train on top of existing distill checkpoint.")
    args, _ = parser.parse_known_args()

    rank, world = setup_ddp()
    cfg = dict(CFG)
    cfg["world_size"] = world

    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)

    device    = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    amp_dtype = torch.bfloat16

    if is_main(rank):
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Device: {device}  |  World: {world}  |  Alpha: {cfg['alpha']}")

    # ── tokeniser ───────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    vocab_size = sp.vocab_size()
    if is_main(rank):
        print(f"Vocab size: {vocab_size:,}")

    # ── dataset ─────────────────────────────────────────────
    if is_main(rank):
        print("\nLoading teacher datasets …")

    train_ds = BilingualDistillDataset(
        TEACHER_KM, TEACHER_EN,
        split="train", valid_frac=cfg["valid_split"], seed=cfg["seed"],
    )
    valid_ds = BilingualDistillDataset(
        TEACHER_KM, TEACHER_EN,
        split="valid", valid_frac=cfg["valid_split"], seed=cfg["seed"],
    )

    collate_fn    = make_distill_collate(sp, cfg["pad_id"], cfg["bos_id"], cfg["max_len_tokens"])
    train_sampler = DistributedSampler(train_ds, shuffle=True)  if world > 1 else None
    valid_sampler = DistributedSampler(valid_ds, shuffle=False) if world > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size         = cfg["batch_size"],
        sampler            = train_sampler,
        shuffle            = (train_sampler is None),
        collate_fn         = collate_fn,
        num_workers        = cfg["num_workers"],
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size         = cfg["batch_size"],
        sampler            = valid_sampler,
        shuffle            = False,
        collate_fn         = collate_fn,
        num_workers        = 2,
        pin_memory         = True,
        drop_last          = False,
        persistent_workers = True,
    )

    # ── model ───────────────────────────────────────────────
    model_raw = MiniNLLB(
        vocab_size = vocab_size,
        d_model    = cfg["d_model"],
        enc_layers = cfg["enc_layers"],
        dec_layers = cfg["dec_layers"],
        n_heads    = cfg["n_heads"],
        ffn_dim    = cfg["ffn_dim"],
        max_len    = cfg["max_len"],
        dropout    = cfg["dropout"],
    ).to(device)

    if cfg["compile_model"] and hasattr(torch, "compile"):
        model_raw = torch.compile(model_raw)
        if is_main(rank):
            print("torch.compile() enabled")

    model     = DDP(model_raw, device_ids=[rank]) if world > 1 else model_raw
    unwrapped = model.module if world > 1 else model_raw

    if is_main(rank):
        print(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")

    # ── loss + optimiser ────────────────────────────────────
    loss_fn = DistillationLoss(
        vocab_size = vocab_size,
        pad_id     = cfg["pad_id"],
        smoothing  = cfg["label_smoothing"],
        alpha      = cfg["alpha"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["peak_lr"],
        betas        = (0.9, 0.98),
        eps          = 1e-9,
        weight_decay = cfg["weight_decay"],
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── checkpoint / weight loading ─────────────────────────
    start_epoch = 0
    global_step = 0
    best_val    = float("inf")

    latest_distill = find_latest_checkpoint(CKPT_DIR)

    if latest_distill is not None:
        # Resume distillation run
        if is_main(rank):
            print(f"\nResuming distillation from {latest_distill} …")
        resumed_epoch, global_step, best_val = load_checkpoint(
            latest_distill, unwrapped, optimizer, scaler, device
        )
        start_epoch = resumed_epoch + 1

        # Warm restart — reset the step counter so the LR schedule
        # starts fresh from warmup rather than continuing the old cosine tail.
        if cfg.get("reset_schedule", False):
            global_step = 0
            if is_main(rank):
                print(f"  ⚡ reset_schedule=True — LR schedule restarted from step 0")
                print(f"     New peak_lr={cfg['peak_lr']:.1e}  warmup={cfg['warmup_steps']} steps")

        if is_main(rank):
            print(f"  Resumed at epoch {start_epoch}  step {global_step}  best_val {best_val:.4f}")

    elif BASE_CKPT.exists():
        # Fresh distillation — initialise from pretrain weights
        if is_main(rank):
            print(f"\nInitialising from pretrain checkpoint: {BASE_CKPT}")
        load_weights_only(BASE_CKPT, unwrapped, device)

    else:
        if is_main(rank):
            print(f"\n⚠️  Pretrain checkpoint not found at {BASE_CKPT}")
            print("   Update BASE_CKPT in the script to point to your checkpoint.")
            print("   Distilling from random init.")

    # ── epoch target ────────────────────────────────────────
    if args.extra_epochs is not None:
        total_epochs = start_epoch + args.extra_epochs
        if is_main(rank):
            print(f"--extra-epochs {args.extra_epochs}  →  epochs {start_epoch+1}…{total_epochs}")
    else:
        total_epochs = cfg["epochs"]

    if start_epoch >= total_epochs:
        if is_main(rank):
            print(f"Already at epoch {start_epoch}, target is {total_epochs}. Use --extra-epochs N.")
        cleanup_ddp(world)
        return

    steps_per_epoch = len(train_loader) // cfg["grad_accum_steps"]
    total_steps     = steps_per_epoch * total_epochs
    if is_main(rank):
        print(f"Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}  |  Epochs left: {total_epochs - start_epoch}\n")

    # ── load existing log ────────────────────────────────────
    log_rows: list[dict] = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    log_rows.append(json.loads(line))

    # ── training loop ───────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main(rank):
            print(f"\n{'='*60}")
            print(f"DISTILL EPOCH {epoch+1} / {total_epochs}")
            print(f"{'='*60}")

        train_loss, t_loss, h_loss, global_step = train_one_epoch(
            epoch, model, train_loader, loss_fn, optimizer, scaler,
            device, amp_dtype, cfg, global_step, total_steps, rank,
        )
        val_loss = evaluate(model, valid_loader, loss_fn, device, amp_dtype, rank)

        if is_main(rank):
            print(f"\nEpoch {epoch+1}  "
                  f"train={train_loss:.4f}  "
                  f"(teacher={t_loss:.4f}  human={h_loss:.4f})  "
                  f"val={val_loss:.4f}")

            if cfg["sample_every_epoch"]:
                print_translation_samples(
                    unwrapped, sp, valid_loader, device,
                    cfg["bos_id"], cfg["eos_id"], cfg["pad_id"],
                    n=cfg["n_samples"],
                )

            ckpt_path = CKPT_DIR / f"epoch_{epoch+1:02d}"
            save_checkpoint(ckpt_path, unwrapped, optimizer, scaler,
                            epoch, global_step, val_loss, cfg, vocab_size)
            print(f"Checkpoint → {ckpt_path}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(CKPT_DIR / "best", unwrapped, optimizer, scaler,
                                epoch, global_step, val_loss, cfg, vocab_size)
                print(f"🔥 New best distill val loss: {best_val:.4f}")

            log_rows.append({
                "epoch":        epoch + 1,
                "step":         global_step,
                "train_loss":   round(train_loss, 5),
                "teacher_loss": round(t_loss,     5),
                "human_loss":   round(h_loss,     5),
                "val_loss":     round(val_loss,   5),
            })
            with open(LOG_FILE, "w") as f:
                for row in log_rows:
                    f.write(json.dumps(row) + "\n")

    cleanup_ddp(world)


if __name__ == "__main__":
    # Fresh distillation (loads BASE_CKPT automatically):
    #   python 10_distill.py
    #
    # Resume distillation + 5 more epochs:
    #   python 10_distill.py --extra-epochs 5
    #
    # Dual GPU:
    #   torchrun --nproc_per_node=2 10_distill.py
    #   torchrun --nproc_per_node=2 10_distill.py --extra-epochs 5
    main()