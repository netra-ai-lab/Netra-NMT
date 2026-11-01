"""Score models on parallel test sets, both directions.

    netra-nmt-train evaluate published
    netra-nmt-train evaluate published training/runs/NAME/bundles/model --test alt my_test.tsv
    netra-nmt-train evaluate training/export_rope --mode beam --limit 200

MODEL is one of
    published               the model the package ships, run exactly as the package runs it
    a model folder          (netra-model.json) from export.sh, run by the package
    a release folder        (model.safetensors + config.json) or a checkpoint.pt, run in PyTorch

Metrics (sacreBLEU): spBLEU (FLORES-200 tokenizer), chrF++ (word_order=2) and
TER (on khmercut-segmented text for Khmer targets), as in recipe/evaluate_alt.py,
which also adds COMET and BERTScore. Predictions (CSV) and summary.json go to
training/runs/eval-<date-time>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from ..hub import RUNS_DIR

DIRECTIONS = (("en2km", 0, 1), ("km2en", 1, 0))


def translator_for(spec: str, device: str):
    """A function (texts, direction, mode) -> outputs for any MODEL form."""
    from netra_nmt import NetraTranslator
    path = Path(spec)
    if spec == "published" or (path / "netra-model.json").exists():
        t = NetraTranslator(local_dir=None if spec == "published" else path, device=device)
        return lambda texts, direction, mode: t.translate_batch(texts, direction=direction, mode=mode)

    import torch

    from netra_nmt.inference.nmt import NMTModel  # noqa: F401  (tokenizer helpers below)
    from netra_nmt.khseg import desegment_km, segment_km
    from netra_nmt.textnorm import normalize_en_source

    from .. import decoding
    from ..checkpoint import DEFAULT_SPM, load
    import sentencepiece as spm

    model, config, sp_path = load(spec)
    torch_device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    model = model.to(torch_device)
    sp = spm.SentencePieceProcessor(model_file=str(sp_path or DEFAULT_SPM))
    markers = config.get("lang_markers", {"en": "<2en>", "km": "<2km>"})
    bos, eos = config.get("bos_id", 2), config.get("eos_id", 3)

    def run(texts, direction, mode):
        out = []
        for text in texts:
            tgt = "km" if direction == "en2km" else "en"
            text = normalize_en_source(text) if direction == "en2km" else segment_km(text)
            ids = sp.encode(f"{markers[tgt]} {text.strip()}", out_type=int)[: config.get("max_len", 256)]
            x = torch.tensor([ids], device=torch_device)
            mask = torch.zeros_like(x, dtype=torch.bool)
            with torch.no_grad():
                if mode == "beam":
                    toks = decoding.beam_search(model, x, mask, bos, eos, 128)
                else:
                    toks = decoding.greedy_decode(model, x, mask, bos, eos, 128)
            s = sp.decode([t for t in toks if t not in (0, bos, eos)])
            out.append(desegment_km(s) if tgt == "km" else s)
        return out
    return run


def score(preds: list[str], refs: list[str], target: str) -> dict:
    from sacrebleu.metrics import BLEU, CHRF, TER

    from netra_nmt.khseg import segment_km
    result = {
        "spbleu": round(BLEU(tokenize="flores200").corpus_score(preds, [refs]).score, 2),
        "chrf_pp": round(CHRF(word_order=2).corpus_score(preds, [refs]).score, 2),
    }
    if target == "km":
        result["ter"] = round(TER().corpus_score([segment_km(p) for p in preds],
                                                 [[segment_km(r) for r in refs]]).score, 2)
    else:
        result["ter"] = round(TER().corpus_score(preds, [refs]).score, 2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--test", nargs="+", default=["alt"], help="test sets (default: alt)")
    parser.add_argument("--mode", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--limit", type=int, help="score only the first N pairs of each test set")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from tqdm import tqdm

    from ..data import load_pairs

    out = args.out or RUNS_DIR / f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in args.models:
        run = translator_for(spec, args.device)
        for test in args.test:
            pairs = load_pairs(test, args.limit)
            for direction, src_col, ref_col in DIRECTIONS:
                srcs = [p[src_col] for p in pairs]
                refs = [p[ref_col] for p in pairs]
                start, preds = time.perf_counter(), []
                for i in tqdm(range(0, len(srcs), 32), desc=f"{Path(spec).name} {test} {direction}",
                              unit="batch", dynamic_ncols=True, leave=False):
                    preds += run(srcs[i:i + 32], direction, args.mode)
                seconds = time.perf_counter() - start
                row = {"model": spec, "test": test, "direction": direction, "mode": args.mode,
                       "n_pairs": len(pairs), **score(preds, refs, direction[-2:]),
                       "seconds": round(seconds, 1)}
                summary.append(row)
                name = f"{Path(spec).name}_{Path(test).stem.replace(':', '_')}_{direction}.csv"
                with open(out / name, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["source", "reference", "prediction"])
                    w.writerows(zip(srcs, refs, preds))
                print(f"{spec} | {test} | {direction} | spBLEU {row['spbleu']:6.2f} | chrF++ {row['chrf_pp']:6.2f} "
                      f"| TER {row['ter']:6.2f} | {len(pairs)} pairs, {seconds:.0f}s")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n| Model | Test | Direction | spBLEU | chrF++ | TER |\n|---|---|---|---:|---:|---:|")
    for r in summary:
        print(f"| {r['model']} | {r['test']} | {r['direction']} | {r['spbleu']} | {r['chrf_pp']} | {r['ter']} |")
    print(f"\nPredictions and summary.json: {out}")


if __name__ == "__main__":
    main()
