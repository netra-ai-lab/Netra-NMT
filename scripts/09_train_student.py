"""
train.py  —  Mini NLLB training script
Optimised for dual RTX 4070 (2 × 12 GB VRAM).

Key upgrades vs. the original:
  • Padding masks built and passed to the model (src + tgt)
  • Linear warmup → cosine decay (replaces bare CosineAnnealingLR)
  • Label-smoothed cross-entropy
  • DDP via torchrun for dual-GPU (falls back to single-GPU / CPU cleanly)
  • Gradient accumulation for effectively larger batches
  • Sanity checks: overfit-one-batch probe + per-epoch translation samples
  • torch.compile() for ~15-20 % throughput boost on Ampere+
  • Structured checkpoint: saves config + weights + optimizer state
  • Proper AMP usage (torch.amp.autocast replaces the deprecated cuda.amp API)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

from model_mini_nllb import MiniNLLB


# ============================================================
# DISTRIBUTED HELPERS
# ============================================================

def setup_ddp():
    """Initialise DDP if launched via torchrun, otherwise no-op."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
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
TRAIN_FILE     = ROOT / "data/processed/bilingual_train.jsonl"
VALID_FILE     = ROOT / "data/processed/bilingual_valid.jsonl"
TOKENIZER_PATH = ROOT / "tokenizer/spm_32k.model"

# Output directory for THIS run's checkpoints — kept separate from
# checkpoints_distill/ so the two training histories never collide.
CKPT_DIR       = ROOT / "checkpoints_human_finetune"
LOG_FILE       = ROOT / "train_log_human_finetune.jsonl"

# Weights-only initialisation source. Used ONLY when CKPT_DIR is empty
# (i.e. the first run of this fine-tuning phase). Ignored once
# CKPT_DIR has its own checkpoints to resume from.
BASE_CKPT      = ROOT / "checkpoints_distill/best/checkpoint.pt"


# ============================================================
# CONFIG  (edit here)
# ============================================================

CFG = dict(
    # model
    d_model    = 512,
    enc_layers = 6,
    dec_layers = 6,
    n_heads    = 8,
    ffn_dim    = 2048,
    max_len    = 128,
    dropout    = 0.1,

    # training
    max_len_tokens   = 128,   # truncation length
    batch_size       = 32,    # per GPU
    grad_accum_steps = 2,     # effective batch = batch_size * world * grad_accum
    epochs           = 10,

    # ── LR schedule — tuned for FINE-TUNING from an already-trained model ──
    # peak_lr is much lower than the 3e-4 used for pretraining from scratch.
    # warmup is shorter since the model already has good gradient estimates.
    warmup_steps     = 500,
    peak_lr          = 2e-5,
    min_lr           = 1e-6,
    weight_decay     = 0.01,
    grad_clip        = 1.0,
    label_smoothing  = 0.1,

    # special token ids (match your spm model)
    pad_id = 0,
    bos_id = 2,
    eos_id = 3,

    # misc
    seed               = 42,
    num_workers        = 4,
    compile_model      = True,   # torch.compile — set False if PyTorch < 2.0
    log_every          = 50,     # steps
    sample_every_epoch = True,   # print translation samples after each epoch
    n_samples          = 10,     # how many validation sentences to translate

    # ── warm restart ──────────────────────────────────────────
    # Set True to reset global_step/optimizer when first switching from
    # distillation weights to human-data fine-tuning. After the first
    # successful run on this dataset, set to False so further
    # --extra-epochs calls continue the schedule normally.
    reset_schedule     = True,
)


# ============================================================
# LR SCHEDULE  —  linear warmup then cosine decay
# ============================================================

def get_lr(step: int, warmup: int, total: int, peak: float, min_lr: float) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (peak - min_lr) * cosine


def set_lr(optimizer: AdamW, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================================================
# LABEL-SMOOTHED CROSS-ENTROPY
# ============================================================

class LabelSmoothedCE(nn.Module):
    """
    Cross-entropy with label smoothing, padding-index ignore,
    and an optional EOS weight boost so the model learns to stop.
    """

    def __init__(self, vocab_size: int, pad_id: int,
                 smoothing: float = 0.1,
                 eos_id: int | None = None,
                 eos_weight: float = 2.0):
        super().__init__()
        self.pad_id    = pad_id
        self.smoothing = smoothing
        self.vocab     = vocab_size
        self.eos_id    = eos_id
        self.eos_weight = eos_weight   # multiply EOS loss by this factor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, V)   targets: (N,)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (self.vocab - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth[:, self.pad_id] = 0.0
            mask = (targets == self.pad_id)
            smooth[mask] = 0.0

        loss = -(smooth * log_probs).sum(dim=-1)    # (N,)

        # Upweight EOS positions so the model learns to stop reliably.
        # EOS appears ~once per sentence vs many content tokens, so without
        # this boost its gradient is drowned out.
        if self.eos_id is not None:
            eos_mask = (targets == self.eos_id)
            loss = torch.where(eos_mask, loss * self.eos_weight, loss)

        non_pad = (~mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ============================================================
# COLLATE  —  builds padding masks alongside padded tensors
# ============================================================

def make_collate(pad_id: int, bos_id: int, eos_id: int, max_len: int):
    def collate(batch):
        src_seqs, dec_in_seqs, label_seqs = [], [], []

        for b in batch:
            src = b["src"].tolist()
            tgt = b["tgt"].tolist()
            if not src or not tgt:
                continue

            src_seqs.append(src[:max_len])

            # Truncate to max_len-1 then always append EOS, so the model
            # is always trained to predict EOS at the final position even
            # when the sentence was truncated.
            tgt_truncated = tgt[:max_len - 1]
            tgt_with_eos  = tgt_truncated + [eos_id]

            # Decoder input: [BOS] + tgt_with_eos[:-1]  (right-shifted)
            dec_in_seqs.append([bos_id] + tgt_with_eos[:-1])
            label_seqs.append(tgt_with_eos)

        if not src_seqs:
            return None

        def pad_batch(seqs):
            max_l = max(len(s) for s in seqs)
            padded = torch.tensor(
                [s + [pad_id] * (max_l - len(s)) for s in seqs],
                dtype=torch.long,
            )
            mask = padded == pad_id
            return padded, mask

        src_ids,    src_mask = pad_batch(src_seqs)
        dec_in_ids, tgt_mask = pad_batch(dec_in_seqs)
        labels,     _        = pad_batch(label_seqs)

        return {
            "input_ids":            src_ids,
            "src_key_padding_mask": src_mask,
            "decoder_input_ids":    dec_in_ids,
            "tgt_key_padding_mask": tgt_mask,
            "labels":               labels,
        }

    return collate


# ============================================================
# SANITY CHECK 1  —  overfit a single batch
# ============================================================

@torch.no_grad()
def _check_batch_not_nan(batch, model, loss_fn, device, amp_dtype):
    """Returns True if the forward pass on this batch is finite."""
    src    = batch["input_ids"].to(device)
    dec_in = batch["decoder_input_ids"].to(device)
    labels = batch["labels"].to(device)
    s_mask = batch["src_key_padding_mask"].to(device)
    t_mask = batch["tgt_key_padding_mask"].to(device)

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        logits = model(src, dec_in,
                       src_key_padding_mask=s_mask,
                       tgt_key_padding_mask=t_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    return torch.isfinite(loss).item(), loss.item()


def overfit_one_batch(model, batch, loss_fn, device, amp_dtype, steps=50):
    """
    Trains on a single batch for `steps` gradient updates.
    A healthy model should drive loss below ~1.0 on this tiny task.
    Returns final loss.
    """
    print("\n[Sanity] Overfit-one-batch probe …")
    probe_opt = AdamW(model.parameters(), lr=1e-3)
    scaler    = torch.amp.GradScaler("cuda")

    src    = batch["input_ids"].to(device)
    dec_in = batch["decoder_input_ids"].to(device)
    labels = batch["labels"].to(device)
    s_mask = batch["src_key_padding_mask"].to(device)
    t_mask = batch["tgt_key_padding_mask"].to(device)

    model.train()
    for step in range(1, steps + 1):
        probe_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in,
                           src_key_padding_mask=s_mask,
                           tgt_key_padding_mask=t_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(probe_opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(probe_opt)
        scaler.update()
        if step % 10 == 0:
            print(f"  step {step:3d}  loss = {loss.item():.4f}")

    final = loss.item()
    status = "✅ PASS" if final < 2.0 else "❌ FAIL (model may not be learning)"
    print(f"[Sanity] Final loss after {steps} steps: {final:.4f}  {status}\n")
    return final


# ============================================================
# SANITY CHECK 2  —  translation samples during validation
# ============================================================

@torch.no_grad()
def print_translation_samples(
    model_raw,          # unwrapped (non-DDP) model
    sp,
    valid_loader,
    device,
    bos_id,
    eos_id,
    n: int = 3,
    max_new_tokens: int = 60,
):
    """
    Grabs the first `n` sentences from the validation set,
    generates translations greedily, and prints source / reference / prediction.
    """
    model_raw.eval()
    collected = 0
    print("\n" + "─" * 64)
    print("Translation samples (greedy)")
    print("─" * 64)

    for batch in valid_loader:
        if batch is None:
            continue
        B = batch["input_ids"].size(0)

        for i in range(min(B, n - collected)):
            src_ids  = batch["input_ids"][i : i + 1].to(device)
            s_mask   = batch["src_key_padding_mask"][i : i + 1].to(device)
            ref_ids  = batch["labels"][i].tolist()

            # Remove padding from display
            src_tokens = batch["input_ids"][i].tolist()
            src_tokens = [t for t in src_tokens if t != 0]
            ref_tokens = [t for t in ref_ids   if t not in (0, bos_id, eos_id)]

            src_text = sp.decode(src_tokens)
            ref_text = sp.decode(ref_tokens)

            gen_ids = model_raw.generate(
                src_ids, bos_token_id=bos_id, eos_token_id=eos_id,
                max_new_tokens=max_new_tokens, src_key_padding_mask=s_mask,
            )
            gen_tokens = gen_ids[0].tolist()
            gen_tokens = [t for t in gen_tokens if t not in (0, bos_id, eos_id)]
            gen_text   = sp.decode(gen_tokens)

            print(f"  SRC : {src_text}")
            print(f"  REF : {ref_text}")
            print(f"  HYP : {gen_text}")
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

def train_one_epoch(
    epoch, model, loader, loss_fn, optimizer, scaler,
    device, amp_dtype, cfg, global_step, total_steps, rank,
):
    model.train()
    total_loss = 0.0
    n_steps    = 0
    t0         = time.perf_counter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not is_main(rank))

    optimizer.zero_grad(set_to_none=True)

    for local_step, batch in enumerate(pbar):
        if batch is None:
            continue

        src    = batch["input_ids"].to(device, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        s_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        t_mask = batch["tgt_key_padding_mask"].to(device, non_blocking=True)

        # Update LR every step
        lr = get_lr(global_step, cfg["warmup_steps"], total_steps,
                    cfg["peak_lr"], cfg["min_lr"])
        set_lr(optimizer, lr)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in,
                           src_key_padding_mask=s_mask,
                           tgt_key_padding_mask=t_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            # Scale by grad_accum so effective loss magnitude stays consistent
            loss = loss / cfg["grad_accum_steps"]

        if not torch.isfinite(loss):
            if is_main(rank):
                print(f"  ⚠️  Non-finite loss at step {global_step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        # Gradient accumulation: only step every N micro-batches
        if (local_step + 1) % cfg["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main(rank) and global_step % cfg["log_every"] == 0:
                elapsed = time.perf_counter() - t0
                tps     = (cfg["log_every"] * cfg["batch_size"]
                           * cfg.get("world_size", 1) / elapsed)
                print(
                    f"  step {global_step:6d}  "
                    f"loss {loss.item() * cfg['grad_accum_steps']:.4f}  "
                    f"lr {lr:.2e}  "
                    f"grad_norm {grad_norm:.3f}  "
                    f"{tps:,.0f} tok/s"
                )
                t0 = time.perf_counter()

        total_loss += loss.item() * cfg["grad_accum_steps"]
        n_steps    += 1
        pbar.set_postfix(loss=f"{loss.item() * cfg['grad_accum_steps']:.4f}", lr=f"{lr:.2e}")

    return total_loss / max(n_steps, 1), global_step


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_dtype, rank):
    model.eval()
    total_loss = 0.0
    n_steps    = 0

    for batch in tqdm(loader, desc="  Eval", disable=not is_main(rank), leave=False):
        if batch is None:
            continue

        src    = batch["input_ids"].to(device, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        s_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        t_mask = batch["tgt_key_padding_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(src, dec_in,
                           src_key_padding_mask=s_mask,
                           tgt_key_padding_mask=t_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if torch.isfinite(loss):
            total_loss += loss.item()
            n_steps    += 1

    return total_loss / max(n_steps, 1)


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(path: Path, model_raw, optimizer, scaler, epoch, global_step,
                    val_loss, cfg, vocab_size):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "global_step":  global_step,
        "val_loss":     val_loss,
        "model":        model_raw.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scaler":       scaler.state_dict(),
        "cfg":          cfg,
        "vocab_size":   vocab_size,
    }, path / "checkpoint.pt")


def load_checkpoint(path: Path, model_raw, optimizer, scaler, device):
    ckpt = torch.load(path / "checkpoint.pt", map_location=device)
    sd   = ckpt["model"]

    # Normalise to plain keys first
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # Re-add prefix if the target compiled model expects it
    model_keys = set(dict(model_raw.named_parameters()).keys())
    if any(k.startswith("_orig_mod.") for k in model_keys):
        sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

    model_raw.load_state_dict(sd)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["val_loss"]


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """
    Scans ckpt_dir for folders named epoch_NN and returns the one with
    the highest N that actually contains a checkpoint.pt file.
    Falls back to None if nothing is found.
    """
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
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]   # path with the highest epoch number


def load_weights_only(path: Path, model_raw, device):
    """
    Load just model weights from another checkpoint (e.g. distillation best)
    — no optimizer/scaler/step state. Used to initialise a new fine-tuning
    phase on a different dataset, where Adam momentum and the LR schedule
    should both start fresh.
    """
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model", ckpt)

    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    model_keys = set(dict(model_raw.named_parameters()).keys())
    if any(k.startswith("_orig_mod.") for k in model_keys):
        sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

    missing, unexpected = model_raw.load_state_dict(sd, strict=False)
    missing    = [k for k in missing    if "lm_head" not in k]
    unexpected = [k for k in unexpected if "lm_head" not in k]
    if missing:
        print(f"  [warn] Missing keys:    {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("  Weights loaded cleanly ✓")


# ============================================================
# MAIN
# ============================================================

def main():
    # ── CLI ─────────────────────────────────────────────────
    # Parse before DDP init so all ranks see the same args.
    parser = argparse.ArgumentParser(description="Mini NLLB trainer")
    parser.add_argument(
        "--extra-epochs", type=int, default=None,
        help=(
            "How many MORE epochs to train on top of whatever checkpoint "
            "already exists. E.g. --extra-epochs 3 after epoch_01 trains "
            "epochs 2, 3, 4. Omit to use CFG['epochs'] as the total target."
        ),
    )
    args, _ = parser.parse_known_args()

    rank, world = setup_ddp()
    cfg = dict(CFG)           # copy so we don't mutate the module-level dict
    cfg["world_size"] = world

    # ── reproducibility ─────────────────────────────────────
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)

    device    = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    amp_dtype = torch.bfloat16

    if is_main(rank):
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Device: {device}  |  World size: {world}  |  AMP dtype: {amp_dtype}")

    # ── tokeniser ───────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    vocab_size = sp.vocab_size()
    if is_main(rank):
        print(f"Vocab size: {vocab_size}")

    # ── dataset ─────────────────────────────────────────────
    raw = load_dataset("json", data_files={
        "train": str(TRAIN_FILE),
        "valid": str(VALID_FILE),
    })

    def preprocess(ex):
        return {
            "src": sp.encode(ex["source"], out_type=int)[:cfg["max_len_tokens"]],
            "tgt": sp.encode(ex["target"], out_type=int)[:cfg["max_len_tokens"]],
        }

    dataset = raw.map(preprocess, remove_columns=raw["train"].column_names)
    dataset.set_format(type="torch")

    collate_fn    = make_collate(cfg["pad_id"], cfg["bos_id"], cfg["eos_id"], cfg["max_len_tokens"])
    train_sampler = DistributedSampler(dataset["train"], shuffle=True)  if world > 1 else None
    valid_sampler = DistributedSampler(dataset["valid"], shuffle=False) if world > 1 else None

    train_loader = DataLoader(
        dataset["train"],
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
        dataset["valid"],
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
        n_params = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}")

    # ── loss + optimiser ────────────────────────────────────
    loss_fn = LabelSmoothedCE(
        vocab_size  = vocab_size,
        pad_id      = cfg["pad_id"],
        smoothing   = cfg["label_smoothing"],
        eos_id      = cfg["eos_id"],
        eos_weight  = 2.0,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["peak_lr"],
        betas        = (0.9, 0.98),
        eps          = 1e-9,
        weight_decay = cfg["weight_decay"],
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── resume / initialise ─────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val    = float("inf")

    latest_ckpt = find_latest_checkpoint(CKPT_DIR)

    if latest_ckpt is not None:
        # Resuming a run that already exists in CKPT_DIR
        if is_main(rank):
            print(f"Resuming from {latest_ckpt} …")
        resumed_epoch, global_step, best_val = load_checkpoint(
            latest_ckpt, unwrapped, optimizer, scaler, device
        )
        start_epoch = resumed_epoch + 1   # epoch N is done; start at N+1

        if cfg.get("reset_schedule", False):
            global_step = 0
            best_val    = float("inf")
            if is_main(rank):
                print(f"  ⚡ reset_schedule=True — LR schedule and best_val reset")

        if is_main(rank):
            print(f"  Resumed at epoch {start_epoch}  |  global_step {global_step}  |  best_val {best_val:.4f}")

    elif BASE_CKPT.exists():
        # First run of this fine-tuning phase — load weights only,
        # fresh optimizer/scaler/schedule
        if is_main(rank):
            print(f"Initialising weights from {BASE_CKPT} (fresh optimizer + schedule)")
        load_weights_only(BASE_CKPT, unwrapped, device)

    else:
        if is_main(rank):
            print("No checkpoint found and BASE_CKPT does not exist — starting from random init.")

    # ── resolve total epoch target ───────────────────────────
    # --extra-epochs N  →  train for N more epochs from wherever we are now
    # no flag           →  train until CFG["epochs"] total (original behaviour)
    if args.extra_epochs is not None:
        total_epochs = start_epoch + args.extra_epochs
        if is_main(rank):
            print(f"--extra-epochs {args.extra_epochs}  →  will train epochs {start_epoch+1} … {total_epochs}")
    else:
        total_epochs = cfg["epochs"]
        if is_main(rank) and start_epoch >= total_epochs:
            print(
                f"Already completed {start_epoch} epoch(s) which meets the "
                f"CFG target of {total_epochs}. "
                f"Use --extra-epochs N to train more."
            )

    if start_epoch >= total_epochs:
        cleanup_ddp(world)
        return

    # ── recalculate schedule over the FULL intended run ─────
    # total_steps covers the entire run (including already-done epochs) so
    # the cosine tail lands in the right place when resuming mid-way.
    steps_per_epoch = len(train_loader) // cfg["grad_accum_steps"]
    total_steps     = steps_per_epoch * total_epochs
    if is_main(rank):
        print(f"Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}  |  Epochs left: {total_epochs - start_epoch}")

    # ── SANITY CHECK 1: overfit probe (fresh starts only) ───
    if is_main(rank) and start_epoch == 0:
        probe_batch = next(iter(train_loader))
        ok, init_loss = _check_batch_not_nan(probe_batch, unwrapped, loss_fn, device, amp_dtype)
        if not ok:
            raise RuntimeError("Initial forward pass produced NaN/Inf — check model init!")
        print(f"[Sanity] Initial loss: {init_loss:.4f}  (expected ~{math.log(vocab_size):.2f})")
        import copy
        probe_model = copy.deepcopy(unwrapped)
        overfit_one_batch(probe_model, probe_batch, loss_fn, device, amp_dtype, steps=60)
        del probe_model

    # ── load existing log so we append rather than overwrite ─
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
            print(f"EPOCH {epoch + 1} / {total_epochs}")
            print(f"{'='*60}")

        train_loss, global_step = train_one_epoch(
            epoch, model, train_loader, loss_fn, optimizer, scaler,
            device, amp_dtype, cfg, global_step, total_steps, rank,
        )
        val_loss = evaluate(model, valid_loader, loss_fn, device, amp_dtype, rank)

        if is_main(rank):
            print(f"\nEpoch {epoch+1}  train={train_loss:.4f}  val={val_loss:.4f}")

            # ── SANITY CHECK 2: translation samples ──────────
            if cfg["sample_every_epoch"]:
                print_translation_samples(
                    unwrapped, sp, valid_loader, device,
                    cfg["bos_id"], cfg["eos_id"],
                    n=cfg["n_samples"],
                )

            # ── checkpoint ───────────────────────────────────
            ckpt_path = CKPT_DIR / f"epoch_{epoch+1:02d}"
            save_checkpoint(
                ckpt_path, unwrapped, optimizer, scaler,
                epoch, global_step, val_loss, cfg, vocab_size,
            )
            print(f"Checkpoint saved → {ckpt_path}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    CKPT_DIR / "best",
                    unwrapped, optimizer, scaler,
                    epoch, global_step, val_loss, cfg, vocab_size,
                )
                print(f"🔥 New best val loss: {best_val:.4f}")

            # ── JSON training log ─────────────────────────────
            log_rows.append({
                "epoch":      epoch + 1,
                "step":       global_step,
                "train_loss": round(train_loss, 5),
                "val_loss":   round(val_loss,   5),
            })
            with open(LOG_FILE, "w") as f:
                for row in log_rows:
                    f.write(json.dumps(row) + "\n")

    cleanup_ddp(world)


if __name__ == "__main__":
    # Fresh start, single GPU:    python train.py
    # Fresh start, dual GPU:      torchrun --nproc_per_node=2 train.py
    # Resume + 3 more epochs:     python train.py --extra-epochs 3
    # Resume + 3 more, dual GPU:  torchrun --nproc_per_node=2 train.py --extra-epochs 3
    main()