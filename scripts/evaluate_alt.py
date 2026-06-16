"""
evaluate_alt.py — Evaluate Mini NLLB on mutiyama/alt (Asian Language Treebank)
==============================================================================

Runs batched greedy translation in both directions (EN→KH and KH→EN) and
computes spBLEU, chrF++, kmBLEU (khmercut word BLEU), TER, COMET, and
BERTScore against the references.

Usage
-----
# Quick smoke test (first 50 pairs, CPU):
    python scripts/evaluate_alt.py --limit 50

# Full evaluation on test split (GPU recommended):
    python scripts/evaluate_alt.py --device cuda

# Use a different checkpoint:
    python scripts/evaluate_alt.py --checkpoint checkpoints/best/checkpoint.pt

Dependencies (pip install if missing):
    sacrebleu>=2.3.1
    bert-score>=0.3.13
    unbabel-comet>=2.2.0   (optional; skip with --no-comet)
    khmercut               (optional; skip with --no-khmercut)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

# ── path bootstrap: makes scripts/ importable from any CWD ──────────────────
SCRIPTS_DIR    = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
ROOT           = SCRIPTS_DIR.parent
TOKENIZER_PATH = ROOT / "tokenizer" / "spm_32k.model"
DEFAULT_CKPT   = ROOT / "checkpoints_human_finetune" / "epoch_13" / "checkpoint.pt"

from inference import load_model, beam_search   # load_model handles _orig_mod. stripping
from model_mini_nllb import MiniNLLB           # noqa: F401 (needed by load_model)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Mini NLLB on mutiyama/alt using BLEU, chrF, BERTScore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint",     type=Path,  default=DEFAULT_CKPT,
                   help=f"Path to checkpoint.pt (default: {DEFAULT_CKPT}).")
    p.add_argument("--batch-size",     type=int,   default=32,
                   help="Source sentences per forward pass (default: 32).")
    p.add_argument("--max-new-tokens", type=int,   default=120,
                   help="Max decoder steps. Auto-capped to cfg['max_len']-1 (default: 120).")
    p.add_argument("--device",         type=str,   default=None,
                   help="cuda | cpu (default: auto-detect).")
    p.add_argument("--limit",          type=int,   default=None,
                   help="Evaluate only the first N valid pairs (quick smoke test).")
    p.add_argument("--dataset",         type=str,   default="alt",
                   choices=["alt", "rinabuoy"],
                   help="Which dataset to evaluate on: 'alt' (mutiyama/alt) or 'rinabuoy' (rinabuoy/khmer-english-parallel). (default: alt).")
    p.add_argument("--split",          type=str,   default="test",
                   choices=["train", "validation", "test"],
                   help="Dataset split to evaluate on (default: test).")
    p.add_argument("--beam-size",       type=int,   default=1,
                   help="Beam size for decoding. 1 = greedy (fast), 4-5 = beam search (better). (default: 1).")
    p.add_argument("--output",         type=Path,  default=None,
                   help="Where to write the JSON results (default: eval_results_<dataset>.json).")
    p.add_argument("--no-comet",       action="store_true",
                   help="Skip the COMET metric (avoids the ~2GB model download; faster smoke tests).")
    p.add_argument("--no-khmercut",    action="store_true",
                   help="Skip khmercut Khmer word segmentation (disables kmBLEU and Khmer TER).")
    return p.parse_args()


# ── dataset loading ───────────────────────────────────────────────────────────

def load_alt_dataset(split: str) -> tuple[list[str], list[str]]:
    """
    Load mutiyama/alt and return (en_sentences, kh_sentences).

    The dataset stores each row as {"translation": {lang_code: text, ...}}.
    Language codes are detected at runtime to be resilient to schema changes.
    Rows where either language is None or blank are silently dropped.
    """
    print(f"Loading mutiyama/alt [{split}] from HuggingFace …")
    ds = load_dataset("mutiyama/alt", split=split, trust_remote_code=True)

    # Detect column names from the first row's translation dict
    sample = ds[0]["translation"]
    keys   = set(sample.keys())
    print(f"  Available language keys: {sorted(keys)}")

    en_key = next((k for k in ["en", "english", "eng"] if k in keys), None)
    kh_key = next((k for k in ["khm", "km", "kh", "khmer"] if k in keys), None)

    if en_key is None or kh_key is None:
        raise RuntimeError(
            f"Could not find English or Khmer column.\n"
            f"  English candidates tried : ['en', 'english', 'eng']\n"
            f"  Khmer candidates tried   : ['khm', 'km', 'kh', 'khmer']\n"
            f"  Keys available           : {sorted(keys)}"
        )

    print(f"  English key : '{en_key}'")
    print(f"  Khmer key   : '{kh_key}'")

    en_sents, kh_sents = [], []
    dropped = 0
    for row in ds:
        en = row["translation"].get(en_key)
        kh = row["translation"].get(kh_key)
        if not en or not kh or not en.strip() or not kh.strip():
            dropped += 1
            continue
        en_sents.append(en.strip())
        kh_sents.append(kh.strip())

    print(f"  Loaded {len(en_sents):,} pairs  ({dropped} dropped — missing or null)")
    return en_sents, kh_sents


def load_rinabuoy_dataset(split: str) -> tuple[list[str], list[str]]:
    """
    Load rinabuoy/khmer-english-parallel and return (en_sentences, kh_sentences).
    Column names are auto-detected from common candidates.
    """
    print(f"Loading rinabuoy/khmer-english-parallel [{split}] from HuggingFace …")
    try:
        ds = load_dataset("rinabuoy/khmer-english-parallel", split=split, trust_remote_code=True)
    except ValueError:
        # Dataset may only have a train split
        print(f"  Split '{split}' not found — falling back to 'train'.")
        ds = load_dataset("rinabuoy/khmer-english-parallel", split="train", trust_remote_code=True)

    cols = ds.column_names
    print(f"  Available columns: {cols}")

    en_key = next((k for k in ["en", "english", "eng", "English"] if k in cols), None)
    kh_key = next((k for k in ["km", "kh", "khmer", "khm", "Khmer"] if k in cols), None)

    if en_key is None or kh_key is None:
        raise RuntimeError(
            f"Could not find English or Khmer column.\n"
            f"  English candidates tried: ['en', 'english', 'eng', 'English']\n"
            f"  Khmer candidates tried  : ['km', 'kh', 'khmer', 'khm', 'Khmer']\n"
            f"  Columns available       : {cols}"
        )

    print(f"  English key : '{en_key}'")
    print(f"  Khmer key   : '{kh_key}'")

    en_sents, kh_sents = [], []
    dropped = 0
    for row in ds:
        en = row.get(en_key)
        kh = row.get(kh_key)
        if not en or not kh or not str(en).strip() or not str(kh).strip():
            dropped += 1
            continue
        en_sents.append(str(en).strip())
        kh_sents.append(str(kh).strip())

    print(f"  Loaded {len(en_sents):,} pairs  ({dropped} dropped — missing or null)")
    return en_sents, kh_sents


# ── batched encode / decode ───────────────────────────────────────────────────

def encode_batch(
    sp: spm.SentencePieceProcessor,
    texts: list[str],
    max_len: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a list of strings and right-pad them to the same length.

    Returns
    -------
    input_ids : (B, S_max)  long tensor
    src_mask  : (B, S_max)  bool tensor — True at pad positions
    """
    encoded = [sp.encode(t, out_type=int)[:max_len] for t in texts]
    # Degenerate guard: if a sentence encodes to nothing, use a single pad token
    encoded = [e if e else [pad_id] for e in encoded]

    max_s  = max(len(e) for e in encoded)
    padded = [e + [pad_id] * (max_s - len(e)) for e in encoded]

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    src_mask  = (input_ids == pad_id)   # True where padding — matches training convention
    return input_ids, src_mask


def decode_batch(
    sp: spm.SentencePieceProcessor,
    gen_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
) -> list[str]:
    """
    Decode a (B, T) generate() output tensor to strings.

    generate() runs until ALL sequences in the batch have emitted EOS, so
    short sequences accumulate junk tokens after their first EOS.  We must:
      1. Remove the leading BOS token (index 0).
      2. Truncate at the FIRST EOS occurrence.
    This differs from decode_ids() in inference.py which strips all instances
    of special tokens — correct for single-sentence decoding, wrong here.
    """
    results = []
    for row in gen_ids.tolist():
        seq = row[1:]                           # drop leading BOS
        if eos_id in seq:
            seq = seq[: seq.index(eos_id)]     # truncate at first EOS
        results.append(sp.decode(seq))
    return results


# ── core translation loop ─────────────────────────────────────────────────────

@torch.no_grad()
def translate_batch(
    sentences: list[str],
    prefix: str,
    model: MiniNLLB,
    sp: spm.SentencePieceProcessor,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    beam_size: int = 1,
) -> list[str]:
    """
    Translate a list of raw source sentences.

    beam_size=1  → batched greedy decoding (fast, ~32 sentences at once)
    beam_size>1  → per-sentence beam search (better quality, slower)
    """
    pad_id  = cfg["pad_id"]    # 0
    bos_id  = cfg["bos_id"]    # 2
    eos_id  = cfg["eos_id"]    # 3
    max_len = cfg["max_len"]   # 128

    all_translations: list[str] = []
    model.eval()

    if beam_size == 1:
        # ── batched greedy ────────────────────────────────────────────────────
        for start in tqdm(
            range(0, len(sentences), batch_size),
            desc=f"  Greedy ({prefix})",
            unit="batch",
        ):
            batch    = sentences[start : start + batch_size]
            prefixed = [f"{prefix} {s}" for s in batch]
            input_ids, src_mask = encode_batch(sp, prefixed, max_len, pad_id, device)
            gen_ids = model.generate(
                input_ids,
                bos_token_id         = bos_id,
                eos_token_id         = eos_id,
                max_new_tokens       = max_new_tokens,
                src_key_padding_mask = src_mask,
            )
            all_translations.extend(decode_batch(sp, gen_ids, bos_id, eos_id))

    else:
        # ── per-sentence beam search ──────────────────────────────────────────
        for sent in tqdm(sentences, desc=f"  Beam={beam_size} ({prefix})", unit="sent"):
            prefixed  = f"{prefix} {sent}"
            input_ids = torch.tensor(
                [sp.encode(prefixed, out_type=int)[:max_len]], dtype=torch.long, device=device
            )
            src_mask  = torch.zeros_like(input_ids, dtype=torch.bool)   # no padding (single sentence)

            tokens = beam_search(
                model, input_ids, src_mask,
                bos_id, eos_id,
                max_new_tokens = max_new_tokens,
                beam_size      = beam_size,
                length_penalty = 0.6,
            )
            # beam_search returns a list of token ids including BOS; strip special tokens
            clean = [t for t in tokens if t not in (bos_id, eos_id, pad_id)]
            all_translations.append(sp.decode(clean))

    return all_translations


# ── metric computation ────────────────────────────────────────────────────────

def segment_khmer(texts: list[str], segmenter) -> list[str]:
    """
    Word-segment Khmer strings with khmercut and return space-joined tokens,
    suitable for sacreBLEU with tokenize='none'. khmercut returns whitespace as
    separate tokens, so we drop pure-whitespace tokens before joining.
    """
    out = []
    for t in texts:
        toks = [w for w in segmenter(t) if w.strip()]
        out.append(" ".join(toks) if toks else t)
    return out


def compute_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    direction: str,
    comet_model=None,
    khmer_segmenter=None,
) -> dict:
    """
    Compute spBLEU, chrF++, kmBLEU, TER, COMET, and BERTScore F1.

    BLEU:
    - spBLEU = sacreBLEU with the FLORES-200 SentencePiece tokenizer
      ('flores200'). This is the NLLB-paper standard and is valid for BOTH
      Khmer and English, so the two directions are directly comparable.
      (The old 'char' tokenizer inflated EN→KH; '13a' was harsh on Khmer.)

    chrF++:
    - chrF with word bigrams (word_order=2) — the WMT-preferred variant.

    kmBLEU (Khmer-target only):
    - Word-level BLEU on khmercut-segmented text (tokenize='none'). Gives an
      interpretable word BLEU for Khmer, complementing spBLEU. None unless the
      target is Khmer and khmercut is available.

    TER:
    - Translation Edit Rate (lower is better). For English targets uses the
      default 'tercom' tokenizer; for Khmer targets uses khmercut segmentation
      (Khmer has no whitespace and is outside TER's CJK ranges, so without
      segmentation TER collapses to one token and explodes >100).

    COMET:
    - Unbabel/wmt22-comet-da, a neural reference-based metric that needs the
      source sentence. Loaded once by the caller and passed in via comet_model.
      Skipped gracefully (None) if comet_model is None or prediction fails.

    BERTScore:
    - xlm-roberta-base covers both English and Khmer script.
    """
    from sacrebleu.metrics import BLEU, CHRF, TER
    from bert_score import score as bert_score_fn

    refs_wrapped = [references]   # sacrebleu expects list[list[str]]

    # Use the metric-class API so get_signature() is available for reproducibility.
    bleu_metric = BLEU(tokenize="flores200")            # spBLEU
    chrf_metric = CHRF(word_order=2)                    # chrF++

    bleu = bleu_metric.corpus_score(hypotheses, refs_wrapped)
    chrf = chrf_metric.corpus_score(hypotheses, refs_wrapped)

    target_is_en = direction.split("→")[-1].strip().upper() == "EN"

    # Pre-segment once with khmercut when the target is Khmer (reused for kmBLEU + TER).
    use_khmercut = (not target_is_en) and (khmer_segmenter is not None)
    if use_khmercut:
        seg_hyps = segment_khmer(hypotheses, khmer_segmenter)
        seg_refs = [segment_khmer(references, khmer_segmenter)]

    # ── kmBLEU: word-level Khmer BLEU on khmercut segments ─────────────────────
    kmbleu_score = None
    if use_khmercut:
        kmbleu_metric = BLEU(tokenize="none")
        kmbleu        = kmbleu_metric.corpus_score(seg_hyps, seg_refs)
        kmbleu_score  = round(kmbleu.score, 2)

    # ── TER ────────────────────────────────────────────────────────────────────
    if target_is_en:
        ter_metric  = TER()
        ter         = ter_metric.corpus_score(hypotheses, refs_wrapped)
        ter_score   = round(ter.score, 2)
        ter_tokenize = "tercom"
    elif use_khmercut:
        # Default TER (normalized=False) just lowercases + whitespace-splits, so on
        # khmercut-segmented input it operates on Khmer words.
        ter_metric  = TER()
        ter         = ter_metric.corpus_score(seg_hyps, seg_refs)
        ter_score   = round(ter.score, 2)
        ter_tokenize = "khmercut"
    else:
        ter_score   = None
        ter_tokenize = None

    # ── COMET (neural, needs source) ──────────────────────────────────────────
    comet_score = None
    if comet_model is not None:
        try:
            print(f"  Running COMET ({direction}) with wmt22-comet-da …")
            data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)
            ]
            gpus = 1 if torch.cuda.is_available() else 0
            out  = comet_model.predict(data, batch_size=32, gpus=gpus)
            comet_score = round(out["system_score"] * 100.0, 2)
        except Exception as e:
            print(f"  [warn] COMET failed ({direction}): {e}")
            comet_score = None

    print(f"  Running BERTScore ({direction}) with xlm-roberta-base …")
    _, _, f1 = bert_score_fn(
        hypotheses,
        references,
        model_type = "xlm-roberta-base",
        verbose    = False,
    )
    bertscore_f1 = f1.mean().item() * 100.0

    return {
        "direction":       direction,
        "spbleu":          round(bleu.score, 2),
        "bleu_signature":  str(bleu_metric.get_signature()),
        "chrf_pp":         round(chrf.score, 2),
        "chrf_signature":  str(chrf_metric.get_signature()),
        "kmbleu":          kmbleu_score,
        "kmbleu_tokenize": "khmercut" if kmbleu_score is not None else None,
        "ter":             ter_score,
        "ter_tokenize":    ter_tokenize,
        "comet":           comet_score,
        "comet_model":     "Unbabel/wmt22-comet-da" if comet_model is not None else None,
        "bertscore_f1":    round(bertscore_f1, 2),
        "bertscore_model": "xlm-roberta-base",
        "n_pairs":         len(hypotheses),
        "beam_size":       1,   # overwritten by caller
    }


# ── COMET model loading ───────────────────────────────────────────────────────

def load_comet():
    """
    Download (cached after first run) and load Unbabel/wmt22-comet-da.
    Returns the loaded model, or None if COMET is unavailable.
    """
    try:
        from comet import download_model, load_from_checkpoint
        print("Loading COMET (Unbabel/wmt22-comet-da) — downloads ~2GB on first run …")
        path = download_model("Unbabel/wmt22-comet-da")
        return load_from_checkpoint(path)
    except Exception as e:
        print(f"  [warn] Could not load COMET ({e}). Skipping COMET metric.")
        return None


# ── khmercut loading ──────────────────────────────────────────────────────────

def load_khmer_segmenter():
    """
    Return khmercut's tokenize function, or None if khmercut is unavailable.
    Used for word-level Khmer BLEU/TER on Khmer-target output.
    """
    try:
        from khmercut import tokenize
        print("Loaded khmercut for Khmer word segmentation (kmBLEU / Khmer TER).")
        return tokenize
    except Exception as e:
        print(f"  [warn] khmercut unavailable ({e}). Khmer word-BLEU/TER will be n/a.")
        return None


# ── results display ───────────────────────────────────────────────────────────

def print_results_table(results: list[dict], dataset_label: str = "mutiyama/alt") -> None:
    col = {"dir": 12, "bleu": 8, "chrf": 8, "km": 8, "ter": 8, "comet": 8, "bert": 14, "n": 7}

    def fmt(val, width):
        return f"{'n/a':>{width}}" if val is None else f"{val:>{width}.2f}"

    header = (
        f"{'Direction':<{col['dir']}} "
        f"{'spBLEU':>{col['bleu']}} "
        f"{'chrF++':>{col['chrf']}} "
        f"{'kmBLEU':>{col['km']}} "
        f"{'TER':>{col['ter']}} "
        f"{'COMET':>{col['comet']}} "
        f"{'BERTScore F1':>{col['bert']}} "
        f"{'N':>{col['n']}}"
    )
    sep = "─" * len(header)
    print()
    print("═" * len(header))
    _mode = f"beam={results[0].get('beam_size', 1)}" if results and results[0].get('beam_size', 1) > 1 else "greedy"
    print(f"  Evaluation on {dataset_label}  ({_mode} decoding)")
    print("═" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['direction']:<{col['dir']}} "
            f"{fmt(r['spbleu'], col['bleu'])} "
            f"{fmt(r['chrf_pp'], col['chrf'])} "
            f"{fmt(r['kmbleu'], col['km'])} "
            f"{fmt(r['ter'], col['ter'])} "
            f"{fmt(r['comet'], col['comet'])} "
            f"{fmt(r['bertscore_f1'], col['bert'])} "
            f"{r['n_pairs']:>{col['n']},}"
        )
    print("═" * len(header))
    print()
    print("  spBLEU: sacreBLEU 'flores200' tokenizer (both directions) — NLLB standard")
    print("  kmBLEU: word BLEU on khmercut-segmented Khmer (KH target only)")
    print("  chrF++: chrF with word bigrams   |   TER: lower is better (khmercut for KH)")
    print("  COMET : Unbabel/wmt22-comet-da   |   BERTScore: xlm-roberta-base")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Tokenizer
    if not TOKENIZER_PATH.exists():
        sys.exit(f"Tokenizer not found: {TOKENIZER_PATH}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    # Model (load_model from inference.py already strips _orig_mod. prefix)
    model, cfg = load_model(args.checkpoint, device)

    # Cap decoder steps to stay within positional embedding range (0 … max_len-1)
    max_new_tokens = min(args.max_new_tokens, cfg["max_len"] - 1)
    if max_new_tokens < args.max_new_tokens:
        print(
            f"  [warn] max_new_tokens capped {args.max_new_tokens} → {max_new_tokens}"
            f" (positional embedding limit: {cfg['max_len']})"
        )

    # Dataset
    if args.dataset == "rinabuoy":
        dataset_label = "rinabuoy/khmer-english-parallel"
        en_sents, kh_sents = load_rinabuoy_dataset(args.split)
    else:
        dataset_label = "mutiyama/alt"
        en_sents, kh_sents = load_alt_dataset(args.split)

    if args.output is None:
        args.output = ROOT / f"eval_results_{args.dataset}.json"

    if args.limit is not None:
        en_sents = en_sents[: args.limit]
        kh_sents = kh_sents[: args.limit]
        print(f"  Limiting to first {len(en_sents)} pairs (--limit).")

    # COMET model (loaded once, reused for both directions)
    comet_model = None if args.no_comet else load_comet()

    # khmercut segmenter (for Khmer-target word BLEU / TER)
    khmer_segmenter = None if args.no_khmercut else load_khmer_segmenter()

    results = []

    beam_size = args.beam_size
    mode_str  = f"beam={beam_size}" if beam_size > 1 else "greedy"
    print(f"\nDecoding mode: {mode_str}")

    # ── EN → KH ──────────────────────────────────────────────────────────────
    print("\n[1/2] EN → KH")
    enkh_hyps    = translate_batch(en_sents, "<2km>", model, sp, cfg, device,
                                   args.batch_size, max_new_tokens, beam_size)
    enkh_metrics = compute_metrics(enkh_hyps, kh_sents, en_sents, "EN→KH",
                                   comet_model, khmer_segmenter)
    enkh_metrics["beam_size"] = beam_size
    results.append(enkh_metrics)

    # ── KH → EN ──────────────────────────────────────────────────────────────
    print("\n[2/2] KH → EN")
    khen_hyps    = translate_batch(kh_sents, "<2en>", model, sp, cfg, device,
                                   args.batch_size, max_new_tokens, beam_size)
    khen_metrics = compute_metrics(khen_hyps, en_sents, kh_sents, "KH→EN",
                                   comet_model, khmer_segmenter)
    khen_metrics["beam_size"] = beam_size
    results.append(khen_metrics)

    # ── Output ────────────────────────────────────────────────────────────────
    print_results_table(results, dataset_label)

    output_data = {
        "checkpoint":      str(args.checkpoint),
        "dataset":         dataset_label,
        "split":           args.split,
        "batch_size":      args.batch_size,
        "max_new_tokens":  max_new_tokens,
        "results":         results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
