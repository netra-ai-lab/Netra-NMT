"""
netra_nmt.cli — command-line interface for the netra-nmt translator.

Examples
--------
    # Single sentence (English → Khmer is the default direction):
    netra-translate --text "Hello, how are you?"

    # Khmer → English:
    netra-translate --text "សួស្តី, តើអ្នកសុខសប្បាយទេ?" --direction km2en

    # Translate every line of a file:
    netra-translate --file input.txt --output output.txt --direction en2km

    # Beam search:
    netra-translate --text "Hello" --mode beam --beam-size 5

    # Experimental non-autoregressive diffusion model (faster CPU decoding,
    # currently lower quality) — needs a local export, see
    # scripts/export_diffusion_checkpoint.py:
    netra-translate --text "Hello" --model-type diffusion --local-dir export_diffusion

    # Interactive REPL (no --text / --file):
    netra-translate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import DEFAULT_DIRECTION, DIRECTIONS
from .diffusion_translator import DEFAULT_DECODE_STEPS, DiffusionTranslator
from .translator import NetraTranslator

REPL_HELP = """
Commands:
  :dir en2km | km2en             — switch translation direction
  :mode greedy | beam | sample   — switch decoding strategy (AR model only)
  :beam N                        — set beam size (AR model only, default 5)
  :temp F                        — set sampling temperature (AR model only)
  :topp F                        — set top-p (AR model only)
  :maxlen N                      — set max output tokens (AR model only)
  :steps N                       — set denoising passes K (diffusion model only)
  :help                          — show this message
  :quit  or  Ctrl-D              — exit
"""


def _translate_one(translator, text: str, direction: str, args):
    """Dispatch to the right translate() signature for the loaded model type."""
    if isinstance(translator, DiffusionTranslator):
        return translator.translate(text, direction=direction, steps=args.decode_steps)
    return translator.translate(
        text, direction=direction, mode=args.mode, beam_size=args.beam_size,
        temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
    )


def _interactive_repl(translator, args):
    direction = args.direction
    mode = args.mode
    beam_size = args.beam_size
    temperature = args.temperature
    top_p = args.top_p
    max_new_tokens = args.max_new_tokens
    decode_steps = args.decode_steps
    is_diffusion = isinstance(translator, DiffusionTranslator)

    print("\n" + "=" * 60)
    print("  netra-nmt  —  Interactive Translation")
    print("=" * 60)
    if is_diffusion:
        print(f"  Model: diffusion (experimental)  |  Direction: {direction}  |  "
              f"Steps: {decode_steps}  |  Type :help for commands")
    else:
        print(f"  Model: autoregressive  |  Direction: {direction}  |  Mode: {mode}  |  "
              f"Type :help for commands")
    print("=" * 60 + "\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue

        if text.startswith(":"):
            parts = text.split()
            cmd = parts[0]
            if cmd == ":quit":
                print("Bye.")
                break
            elif cmd == ":help":
                print(REPL_HELP)
            elif cmd == ":dir" and len(parts) == 2:
                if parts[1] in DIRECTIONS:
                    direction = parts[1]
                    print(f"  direction = {direction}")
                else:
                    print(f"  Unknown direction: {parts[1]}")
            elif cmd == ":mode" and len(parts) == 2:
                mode = parts[1]
                args.mode = mode
                print(f"  mode = {mode}")
            elif cmd == ":beam" and len(parts) == 2:
                beam_size = int(parts[1])
                args.beam_size = beam_size
                print(f"  beam_size = {beam_size}")
            elif cmd == ":temp" and len(parts) == 2:
                temperature = float(parts[1])
                args.temperature = temperature
                print(f"  temperature = {temperature}")
            elif cmd == ":topp" and len(parts) == 2:
                top_p = float(parts[1])
                args.top_p = top_p
                print(f"  top_p = {top_p}")
            elif cmd == ":maxlen" and len(parts) == 2:
                max_new_tokens = int(parts[1])
                args.max_new_tokens = max_new_tokens
                print(f"  max_new_tokens = {max_new_tokens}")
            elif cmd == ":steps" and len(parts) == 2:
                decode_steps = int(parts[1])
                args.decode_steps = decode_steps
                print(f"  decode_steps = {decode_steps}")
            else:
                print(f"  Unknown command: {cmd}  (type :help)")
            continue

        try:
            t0 = time.perf_counter()
            result = _translate_one(translator, text, direction, args)
            elapsed = time.perf_counter() - t0
            print(f"<<< {result}")
            print(f"    ({elapsed * 1000:.0f} ms)\n")
        except Exception as e:  # noqa: BLE001 — surface errors in the REPL
            print(f"  Error: {e}\n")


def _translate_file(translator, in_path: Path, out_path: Path, args):
    lines = [l for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Translating {len(lines)} lines from {in_path} …")

    results = []
    for i, line in enumerate(lines, 1):
        try:
            out = _translate_one(translator, line, args.direction, args)
        except Exception as e:  # noqa: BLE001
            out = f"[ERROR: {e}]"
        results.append(out)
        if i % 10 == 0 or i == len(lines):
            print(f"  {i}/{len(lines)}")

    out_path.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"Done. Results written to {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="netra-translate",
        description="Translate text with the netra-nmt EN↔KM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--text", type=str, default=None,
                   help="Translate a single sentence and exit.")
    g.add_argument("--file", type=Path, default=None,
                   help="Path to a plain-text file (one sentence per line).")

    p.add_argument("--output", type=Path, default=Path("output.txt"),
                   help="Output file when using --file (default: output.txt).")

    p.add_argument("--direction", choices=sorted(DIRECTIONS), default=DEFAULT_DIRECTION,
                   help=f"Translation direction (default: {DEFAULT_DIRECTION}).")

    # Model choice
    p.add_argument("--model-type", choices=["ar", "diffusion"], default="ar",
                   help="'ar' = autoregressive (default, released model). "
                        "'diffusion' = experimental non-autoregressive model — same size, "
                        "trades some quality for much faster CPU decoding. Requires "
                        "--local-dir pointed at scripts/export_diffusion_checkpoint.py's "
                        "output (or --repo-id, if you've published diffusion weights).")
    p.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS,
                   help=f"Diffusion only: number of Mask-Predict denoising passes "
                        f"(default: {DEFAULT_DECODE_STEPS}). Higher = better quality, still "
                        f"a fixed, parallel cost.")

    # Model source
    p.add_argument("--repo-id", type=str, default=None,
                   help="Hugging Face repo id to load weights from.")
    p.add_argument("--local-dir", type=Path, default=None,
                   help="Local export directory to load weights from (skips the Hub).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda | cpu (default: auto-detect).")

    # Decoding (autoregressive model only)
    p.add_argument("--mode", choices=["greedy", "beam", "sample"], default="greedy",
                   help="AR model only. Decoding strategy (default: greedy).")
    p.add_argument("--beam-size", type=int, default=5,
                   help="AR model only. Number of beams for beam search (default: 5).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="AR model only. Softmax temperature for sampling (default: 1.0).")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="AR model only. Nucleus top-p for sampling (default: 0.95).")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="AR model only. Maximum output tokens to generate (default: 128).")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.model_type == "diffusion":
        translator = DiffusionTranslator(
            repo_id=args.repo_id, local_dir=args.local_dir, device=args.device,
            steps=args.decode_steps,
        )
    else:
        translator = NetraTranslator(
            repo_id=args.repo_id, local_dir=args.local_dir, device=args.device
        )

    if args.text is not None:
        t0 = time.perf_counter()
        result = _translate_one(translator, args.text, args.direction, args)
        elapsed = time.perf_counter() - t0
        print(f"Input  : {args.text}")
        print(f"Output : {result}")
        print(f"Time   : {elapsed * 1000:.0f} ms")

    elif args.file is not None:
        if not args.file.exists():
            sys.exit(f"Input file not found: {args.file}")
        _translate_file(translator, args.file, args.output, args)

    else:
        _interactive_repl(translator, args)


if __name__ == "__main__":
    main()
