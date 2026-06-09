from pathlib import Path

import torch
import sentencepiece as spm

from model_mini_nllb import MiniNLLB

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = ROOT / "tokenizer/spm_32k.model"

CHECKPOINT_PATH = (
    ROOT / "checkpoints/best/checkpoint.pt"
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ============================================================
# TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))

vocab_size = sp.vocab_size()

print("Tokenizer vocab:", vocab_size)

# ============================================================
# LOAD CHECKPOINT
# ============================================================

ckpt = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

print("\nCheckpoint Info")
print("Epoch:", ckpt["epoch"])
print("Global Step:", ckpt["global_step"])
print("Val Loss:", ckpt["val_loss"])

cfg = ckpt["cfg"]

print("\nModel Config:")
print(cfg)

# ============================================================
# BUILD MODEL
# ============================================================

model = MiniNLLB(
    vocab_size=ckpt["vocab_size"],
    d_model=cfg["d_model"],
    enc_layers=cfg["enc_layers"],
    dec_layers=cfg["dec_layers"],
    n_heads=cfg["n_heads"],
    ffn_dim=cfg["ffn_dim"],
    max_len=cfg["max_len"]
)

missing, unexpected = model.load_state_dict(
    ckpt["model"],
    strict=False
)

print("\nLoad Check:")
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.to(DEVICE)
model.eval()

# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

MAX_GENERATION_LEN = 128

# ============================================================
# TRANSLATE
# ============================================================

@torch.no_grad()
def translate(text):

    src_ids = sp.encode(text, out_type=int)

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    decoder_ids = [BOS_ID]

    for _ in range(MAX_GENERATION_LEN):

        dec = torch.tensor(
            [decoder_ids],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(
            input_ids=src,
            decoder_input_ids=dec
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1
        ).item()

        if next_token == EOS_ID:
            break

        decoder_ids.append(next_token)

    output_ids = decoder_ids[1:]

    return sp.decode(output_ids)

# ============================================================
# INTERACTIVE
# ============================================================

print("\nModel loaded successfully")
print("Type exit to quit\n")

while True:

    text = input("Input: ").strip()

    if text.lower() in ["exit", "quit"]:
        break

    translation = translate(text)

    print()
    print("Output:")
    print(translation)
    print()