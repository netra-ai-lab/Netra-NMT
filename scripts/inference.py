"""
translate.py  —  Inference script for Mini NLLB
================================================

Usage examples
--------------
# Interactive REPL (type sentences one by one):
    python translate.py

# Translate a single sentence from the command line:
    python translate.py --text "សួស្តី, តើអ្នកសុខសប្បាយទេ?"

# Translate every line of a file, write results to output.txt:
    python translate.py --file input.txt --output output.txt

# Use beam search (default: greedy):
    python translate.py --text "Hello" --mode beam --beam-size 5

# Use nucleus (top-p) sampling:
    python translate.py --text "Hello" --mode sample --top-p 0.95 --temperature 0.8

# Load a specific checkpoint instead of the best one:
    python translate.py --checkpoint /path/to/checkpoints/epoch_05/checkpoint.pt

# Show the top-5 token candidates at each decoding step (debug):
    python translate.py --text "Hello" --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model_mini_nllb import MiniNLLB


# ============================================================
# PATHS  (mirror train.py layout)
# ============================================================

ROOT           = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "tokenizer/spm_32k.model"
CKPT_DIR       = ROOT / "checkpoints_distill"
DEFAULT_CKPT   = CKPT_DIR / "epoch_20" / "checkpoint.pt"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(ckpt_path: Path, device: torch.device) -> tuple[MiniNLLB, dict]:
    """
    Load a checkpoint saved by train.py.
    Returns (model, cfg_dict).
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first, or pass --checkpoint <path>."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg        = ckpt["cfg"]
    vocab_size = ckpt["vocab_size"]

    model = MiniNLLB(
        vocab_size = vocab_size,
        d_model    = cfg["d_model"],
        enc_layers = cfg["enc_layers"],
        dec_layers = cfg["dec_layers"],
        n_heads    = cfg["n_heads"],
        ffn_dim    = cfg["ffn_dim"],
        max_len    = cfg["max_len"],
        dropout    = 0.0,           # disable dropout at inference
    ).to(device)

    # torch.compile() wraps keys with "_orig_mod." prefix — strip it if present
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    trained_epochs = ckpt.get("epoch", "?")
    val_loss       = ckpt.get("val_loss", float("nan"))
    print(f"  Trained epochs : {trained_epochs + 1 if isinstance(trained_epochs, int) else trained_epochs}")
    print(f"  Val loss       : {val_loss:.4f}")
    print(f"  Vocab size     : {vocab_size:,}")
    print(f"  Parameters     : {sum(p.numel() for p in model.parameters()):,}")

    return model, cfg


# ============================================================
# TOKENISE / DETOKENISE HELPERS
# ============================================================

def encode(sp: spm.SentencePieceProcessor, text: str, max_len: int, device: torch.device):
    """
    Encode a single string → (1, S) input_ids tensor + (1, S) padding mask.
    No padding needed for a single sentence; mask is all-False.
    """
    ids = sp.encode(text, out_type=int)[:max_len]
    if not ids:
        raise ValueError("Input text encoded to zero tokens — is it empty?")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask  = torch.zeros_like(input_ids, dtype=torch.bool)   # no padding
    return input_ids, src_mask


def decode_ids(sp: spm.SentencePieceProcessor, ids: list[int],
               bos_id: int, eos_id: int, pad_id: int) -> str:
    """Strip special tokens and decode to a string."""
    clean = [t for t in ids if t not in (bos_id, eos_id, pad_id)]
    return sp.decode(clean)


# ============================================================
# DECODING STRATEGIES
# ============================================================

@torch.no_grad()
def greedy_decode(
    model: MiniNLLB,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    verbose: bool = False,
    sp: spm.SentencePieceProcessor | None = None,
) -> list[int]:
    """Standard greedy decoding. Fastest, deterministic."""
    gen = model.generate(
        input_ids,
        bos_token_id      = bos_id,
        eos_token_id      = eos_id,
        max_new_tokens    = max_new_tokens,
        src_key_padding_mask = src_mask,
    )
    tokens = gen[0].tolist()

    if verbose and sp is not None:
        print("\n  [Greedy token trace]")
        for i, t in enumerate(tokens):
            print(f"    step {i:3d}  id={t:6d}  piece={sp.id_to_piece(t)!r}")

    return tokens


@torch.no_grad()
def beam_search(
    model: MiniNLLB,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    beam_size: int = 5,
    length_penalty: float = 0.6,
    verbose: bool = False,
    sp: spm.SentencePieceProcessor | None = None,
) -> list[int]:
    """
    Beam search decoding.
    Each beam is a tuple of (log_prob, token_id_list).
    length_penalty: >1 favours longer outputs, <1 favours shorter.
    """
    device = input_ids.device

    # Encode source once; repeat enc_out beam_size times
    enc_out = model.encoder(input_ids, src_key_padding_mask=src_mask)   # (1, S, D)
    enc_out  = enc_out.repeat(beam_size, 1, 1)                          # (B, S, D)
    src_mask = src_mask.repeat(beam_size, 1)                            # (B, S)

    # Initialise beams: (cumulative_log_prob, [token_ids], finished)
    beams: list[tuple[float, list[int], bool]] = [(0.0, [bos_id], False)]

    import torch.nn as nn

    for step in range(max_new_tokens):
        # Collect all unfinished beams to run in a single batched forward pass
        active_idx    = [i for i, (_, _, done) in enumerate(beams) if not done]
        finished_only = all(done for _, _, done in beams)
        if finished_only:
            break

        # Build decoder input from current beam sequences
        max_t   = max(len(beams[i][1]) for i in active_idx)
        dec_ids = torch.zeros(len(active_idx), max_t, dtype=torch.long, device=device)
        for j, i in enumerate(active_idx):
            seq = beams[i][1]
            dec_ids[j, :len(seq)] = torch.tensor(seq, device=device)

        # Only use enc_out rows matching active beams
        enc_slice  = enc_out[:len(active_idx)]
        mask_slice = src_mask[:len(active_idx)]

        # Forward through decoder
        pos        = torch.arange(max_t, device=device).unsqueeze(0)
        x          = model.decoder.dropout(
            model.decoder.embed(dec_ids) + model.decoder.pos_embed(pos)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            max_t, device=device, dtype=x.dtype
        )
        for layer in model.decoder.layers:
            x = layer(x, enc_slice, tgt_mask=causal_mask,
                      memory_key_padding_mask=mask_slice)
        x = model.decoder.norm(x)

        # Log-probs over vocab at the last position of each active beam
        logits    = model.lm_head(x[:, -1, :])           # (active, V)

        # Repetition penalty per beam
        for j, i in enumerate(active_idx):
            for prev_tok in set(beams[i][1]):
                logits[j, prev_tok] /= 1.3

        log_probs = F.log_softmax(logits, dim=-1)         # (active, V)

        if verbose and sp is not None and step < 3:
            top_vals, top_ids = log_probs[0].topk(5)
            print(f"\n  [Beam step {step}  top-5 tokens from beam 0]")
            for v, tid in zip(top_vals.tolist(), top_ids.tolist()):
                print(f"    {sp.id_to_piece(tid)!r:20s}  log_p={v:.3f}")

        # Expand beams
        candidates: list[tuple[float, list[int], bool]] = []

        # Keep finished beams as-is
        for i, (score, seq, done) in enumerate(beams):
            if done:
                candidates.append((score, seq, True))

        # Expand each active beam
        V          = log_probs.size(-1)
        top_k_vals, top_k_ids = log_probs.topk(beam_size, dim=-1)  # (active, beam_size)

        for j, i in enumerate(active_idx):
            score, seq, _ = beams[i]
            for k in range(beam_size):
                new_token  = top_k_ids[j, k].item()
                new_score  = score + top_k_vals[j, k].item()
                new_seq    = seq + [new_token]
                is_done    = (new_token == eos_id)
                candidates.append((new_score, new_seq, is_done))

        # Keep the best beam_size candidates (length-penalty normalised score for selection)
        def norm_score(cand):
            s, seq, _ = cand
            lp = ((5 + len(seq)) / 6) ** length_penalty
            return s / lp

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_size]

    # Return the sequence with the best normalised score
    def final_score(cand):
        s, seq, _ = cand
        lp = ((5 + len(seq)) / 6) ** length_penalty
        return s / lp

    best = max(beams, key=final_score)
    return best[1]


@torch.no_grad()
def sample_decode(
    model: MiniNLLB,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    verbose: bool = False,
    sp: spm.SentencePieceProcessor | None = None,
) -> list[int]:
    """
    Nucleus (top-p) sampling with temperature.
    temperature < 1  →  sharper, more confident
    temperature > 1  →  flatter, more creative
    top_p            →  only sample from tokens whose cumulative prob ≥ top_p
    """
    import torch.nn as nn

    device = input_ids.device
    enc_out = model.encoder(input_ids, src_key_padding_mask=src_mask)

    generated = [bos_id]

    for step in range(max_new_tokens):
        T       = len(generated)
        dec_ids = torch.tensor([generated], dtype=torch.long, device=device)
        pos     = torch.arange(T, device=device).unsqueeze(0)

        x = model.decoder.dropout(
            model.decoder.embed(dec_ids) + model.decoder.pos_embed(pos)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device, dtype=x.dtype
        )
        for layer in model.decoder.layers:
            x = layer(x, enc_out, tgt_mask=causal_mask,
                      memory_key_padding_mask=src_mask)
        x = model.decoder.norm(x)

        logits = model.lm_head(x[0, -1, :]) / temperature   # (V,)

        # Repetition penalty — discourages repeating tokens already generated
        for prev_tok in set(generated):
            logits[prev_tok] /= 1.3

        # Nucleus filtering
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens beyond the top-p nucleus
        remove = cum_probs > top_p
        remove[0] = False                          # always keep the top token
        sorted_logits[remove] = float("-inf")

        # Map back and sample
        filtered_logits                    = torch.full_like(logits, float("-inf"))
        filtered_logits[sorted_idx]        = sorted_logits
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1).item()

        if verbose and sp is not None and step < 5:
            top_vals, top_ids = F.softmax(logits * temperature, dim=-1).topk(5)
            print(f"\n  [Sample step {step}  top-5 candidates]")
            for v, tid in zip(top_vals.tolist(), top_ids.tolist()):
                print(f"    {sp.id_to_piece(tid)!r:20s}  p={v:.3f}")
            print(f"    → sampled: {sp.id_to_piece(next_token)!r}")

        generated.append(next_token)
        if next_token == eos_id:
            break

    return generated


# ============================================================
# TRANSLATE  (unified entry point)
# ============================================================

def translate(
    text: str,
    model: MiniNLLB,
    sp: spm.SentencePieceProcessor,
    cfg: dict,
    device: torch.device,
    mode: str       = "greedy",
    beam_size: int  = 5,
    length_penalty: float = 0.6,
    temperature: float    = 1.0,
    top_p: float          = 0.95,
    max_new_tokens: int   = 128,
    verbose: bool         = False,
) -> str:
    bos_id = cfg["bos_id"]
    eos_id = cfg["eos_id"]
    pad_id = cfg["pad_id"]
    max_len = cfg["max_len"]

    input_ids, src_mask = encode(sp, text.strip(), max_len, device)

    if mode == "greedy":
        tokens = greedy_decode(
            model, input_ids, src_mask, bos_id, eos_id,
            max_new_tokens, verbose=verbose, sp=sp,
        )
    elif mode == "beam":
        tokens = beam_search(
            model, input_ids, src_mask, bos_id, eos_id,
            max_new_tokens, beam_size=beam_size,
            length_penalty=length_penalty, verbose=verbose, sp=sp,
        )
    elif mode == "sample":
        tokens = sample_decode(
            model, input_ids, src_mask, bos_id, eos_id,
            max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose, sp=sp,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose greedy | beam | sample.")

    return decode_ids(sp, tokens, bos_id, eos_id, pad_id)


# ============================================================
# INTERACTIVE REPL
# ============================================================

REPL_HELP = """
Commands:
  :mode greedy | beam | sample   — switch decoding strategy
  :beam N                        — set beam size (default 5)
  :temp F                        — set sampling temperature (default 1.0)
  :topp F                        — set top-p (default 0.95)
  :maxlen N                      — set max output tokens (default 128)
  :verbose                       — toggle token-level debug output
  :help                          — show this message
  :quit  or  Ctrl-D              — exit
"""

def interactive_repl(model, sp, cfg, device, args):
    mode           = args.mode
    beam_size      = args.beam_size
    temperature    = args.temperature
    top_p          = args.top_p
    max_new_tokens = args.max_new_tokens
    verbose        = args.verbose

    print("\n" + "="*60)
    print("  Mini NLLB  —  Interactive Translation")
    print("="*60)
    print(f"  Mode: {mode}  |  Type :help for commands")
    print("="*60 + "\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue

        # ── commands ────────────────────────────────────────
        if text.startswith(":"):
            parts = text.split()
            cmd   = parts[0]

            if cmd == ":quit":
                print("Bye.")
                break
            elif cmd == ":help":
                print(REPL_HELP)
            elif cmd == ":verbose":
                verbose = not verbose
                print(f"  verbose = {verbose}")
            elif cmd == ":mode" and len(parts) == 2:
                mode = parts[1]
                print(f"  mode = {mode}")
            elif cmd == ":beam" and len(parts) == 2:
                beam_size = int(parts[1])
                print(f"  beam_size = {beam_size}")
            elif cmd == ":temp" and len(parts) == 2:
                temperature = float(parts[1])
                print(f"  temperature = {temperature}")
            elif cmd == ":topp" and len(parts) == 2:
                top_p = float(parts[1])
                print(f"  top_p = {top_p}")
            elif cmd == ":maxlen" and len(parts) == 2:
                max_new_tokens = int(parts[1])
                print(f"  max_new_tokens = {max_new_tokens}")
            else:
                print(f"  Unknown command: {cmd}  (type :help)")
            continue

        # ── translate ────────────────────────────────────────
        try:
            t0     = time.perf_counter()
            result = translate(
                text, model, sp, cfg, device,
                mode=mode, beam_size=beam_size,
                length_penalty=0.6, temperature=temperature,
                top_p=top_p, max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            elapsed = time.perf_counter() - t0
            print(f"<<< {result}")
            print(f"    ({elapsed*1000:.0f} ms)\n")
        except Exception as e:
            print(f"  Error: {e}\n")


# ============================================================
# FILE BATCH MODE
# ============================================================

def translate_file(input_path: Path, output_path: Path,
                   model, sp, cfg, device, args):
    lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]

    print(f"Translating {len(lines)} lines from {input_path} …")

    results = []
    for i, line in enumerate(lines, 1):
        try:
            out = translate(
                line, model, sp, cfg, device,
                mode           = args.mode,
                beam_size      = args.beam_size,
                length_penalty = 0.6,
                temperature    = args.temperature,
                top_p          = args.top_p,
                max_new_tokens = args.max_new_tokens,
                verbose        = False,
            )
        except Exception as e:
            out = f"[ERROR: {e}]"

        results.append(out)

        if i % 10 == 0 or i == len(lines):
            print(f"  {i}/{len(lines)}")

    output_path.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"Done. Results written to {output_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Translate text with a trained Mini NLLB checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    g = p.add_mutually_exclusive_group()
    g.add_argument("--text",  type=str, default=None,
                   help="Translate a single sentence and exit.")
    g.add_argument("--file",  type=Path, default=None,
                   help="Path to a plain-text file (one sentence per line).")

    p.add_argument("--output", type=Path, default=Path("output.txt"),
                   help="Output file when using --file (default: output.txt).")

    # Model
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                   help=f"Path to checkpoint.pt (default: {DEFAULT_CKPT}).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda | cpu (default: auto-detect).")

    # Decoding
    p.add_argument("--mode", choices=["greedy", "beam", "sample"],
                   default="greedy",
                   help="Decoding strategy (default: greedy).")
    p.add_argument("--beam-size",    type=int,   default=5,
                   help="Number of beams for beam search (default: 5).")
    p.add_argument("--temperature",  type=float, default=1.0,
                   help="Softmax temperature for sampling (default: 1.0).")
    p.add_argument("--top-p",        type=float, default=0.95,
                   help="Nucleus top-p for sampling (default: 0.95).")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Maximum output tokens to generate (default: 128).")

    # Debug
    p.add_argument("--verbose", action="store_true",
                   help="Print top token candidates at each decoding step.")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokeniser
    if not TOKENIZER_PATH.exists():
        sys.exit(f"Tokeniser not found: {TOKENIZER_PATH}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    # Load model
    model, cfg = load_model(args.checkpoint, device)

    # Dispatch
    if args.text is not None:
        # Single sentence
        t0     = time.perf_counter()
        result = translate(
            args.text, model, sp, cfg, device,
            mode           = args.mode,
            beam_size      = args.beam_size,
            length_penalty = 0.6,
            temperature    = args.temperature,
            top_p          = args.top_p,
            max_new_tokens = args.max_new_tokens,
            verbose        = args.verbose,
        )
        elapsed = time.perf_counter() - t0
        print(f"\nInput  : {args.text}")
        print(f"Output : {result}")
        print(f"Time   : {elapsed*1000:.0f} ms")

    elif args.file is not None:
        # Batch file
        if not args.file.exists():
            sys.exit(f"Input file not found: {args.file}")
        translate_file(args.file, args.output, model, sp, cfg, device, args)

    else:
        # Interactive REPL
        interactive_repl(model, sp, cfg, device, args)


if __name__ == "__main__":
    main()