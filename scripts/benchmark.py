"""Speed of the package per stage, on this machine.

    python scripts/benchmark.py --device cpu
    python scripts/benchmark.py --device cuda --threads 16 --out results/speed_onnx.json

EN->KM on the 20 sentences of results/speed_hw.json (3 warm-up), greedy and
beam-5: model load, encoder, one decoder step, then whole translations as
output tokens per second and median ms per sentence. The numbers are for
comparing versions and devices on the same machine; say which machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

SENTENCES = [
    "Hello, how are you?",
    "I love Cambodia.",
    "The government announced a new policy yesterday.",
    "Scientists have discovered a new species of frog in the rainforest.",
    "She bought three apples and a loaf of bread at the market.",
    "The meeting has been rescheduled to next Monday afternoon.",
    "Climate change is one of the most pressing issues of our time.",
    "He quickly ran to catch the last bus home.",
    "The new hospital will open its doors to patients in June.",
    "Please remember to submit your report before the deadline.",
    "The children played happily in the park until sunset.",
    "Our company reported record profits in the last quarter.",
    "The ancient temple attracts thousands of tourists every year.",
    "Reading books every day can improve your vocabulary significantly.",
    "The football match was cancelled because of heavy rain.",
    "A balanced diet and regular exercise keep you healthy.",
    "The engineers worked overnight to fix the power failure.",
    "Education is the key to a brighter future for everyone.",
    "The river flooded several villages after days of continuous rain.",
    "Technology has transformed the way we communicate with each other.",
]


def timed(fn, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return (time.perf_counter() - start) / repeat * 1000, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads (default: ORT's choice)")
    parser.add_argument("--local-dir", help="a model folder (default: the pinned release / NETRA_NMT_BUNDLES)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import numpy as np

    from netra_nmt import NetraTranslator
    from netra_nmt.inference import session

    if args.threads:   # applied to every Session the translator opens
        original = session.Session.__init__
        session.Session.__init__ = lambda self, path, device, threads=None: original(self, path, device, args.threads)

    load_ms, t = timed(lambda: NetraTranslator(local_dir=args.local_dir, device=args.device))
    m = t.model
    ids = m.encode_text("the government announced a new policy yesterday.", "km")
    enc_ms, (ck, cv, cm) = timed(lambda: m._encode(ids, False), repeat=20)
    past = np.zeros((m.layers, 1, m.heads, 10, m.head_dim), np.float32)
    step_ms, _ = timed(lambda: m._step([5], past, past, ck, cv, cm, False), repeat=50)

    row = {"model": "Netra-NMT-77M (ONNX)", "device": t.device, "threads": args.threads,
           "cpu": platform.processor() or platform.machine(), "load_ms": round(load_ms),
           "encoder_ms": round(enc_ms, 2), "decoder_step_ms": round(step_ms, 2)}
    for mode in ("greedy", "beam"):
        for s in SENTENCES[:3]:
            t.translate(s, mode=mode)
        times, toks = [], 0
        for s in SENTENCES:
            start = time.perf_counter()
            out = t.translate(s, mode=mode, beam_size=5, max_new_tokens=128)
            times.append(time.perf_counter() - start)
            toks += len(m.sp.encode(out))
        row[f"{mode}_tok_s"] = round(toks / sum(times), 1)
        row[f"{mode}_ms_sent"] = round(statistics.median(times) * 1000, 1)
    for k, v in row.items():
        print(f"{k:16s} {v}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2) + "\n")
        print(f"-> {args.out}")


if __name__ == "__main__":
    os.environ.setdefault("NETRA_NMT_DEVICE", "cpu")
    main()
