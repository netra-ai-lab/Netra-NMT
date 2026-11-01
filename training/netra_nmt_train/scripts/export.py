"""Export a trained model as the ONNX model folder the package runs.

    netra-nmt-train export --model export_rope --out bundles/model
    netra-nmt-train export --model checkpoints_rope/best/checkpoint.pt --out bundles/model --check alt

--model is a release folder (model.safetensors + config.json + spm_32k.model),
a training checkpoint.pt, or "published". The folder gets encoder.onnx,
decoder_step.onnx (weights stored as fp16, computed in fp32), the tokenizer
and netra-model.json.

--check runs the folder through netra_nmt's own inference code and compares
the generated tokens with the PyTorch model, greedy and beam, both directions.
It exits non-zero below 99% identical.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

OPSET = 18


def export_graphs(model, out: Path, heads: int, head_dim: int, layers: int) -> None:
    from ..onnx_graphs import DecoderStep, Encoder

    Dim = torch.export.Dim
    B, S, P = Dim("B"), Dim("S", min=1, max=4096), Dim("P", min=0, max=4096)
    tokens = torch.tensor([[5, 6, 7], [8, 9, 10]])
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        cross_k, cross_v = Encoder(model)(tokens, mask)
        torch.onnx.export(
            Encoder(model), (tokens, mask), out / "encoder.onnx",
            input_names=["tokens", "mask"], output_names=["cross_k", "cross_v"],
            dynamic_shapes={"tokens": {0: B, 1: S}, "mask": {0: B, 1: S}},
            opset_version=OPSET, dynamo=True, external_data=False,
        )
        past = torch.randn(layers, 2, heads, 4, head_dim)
        torch.onnx.export(
            DecoderStep(model),
            (torch.tensor([[5], [6]]), past, past.clone(), cross_k, cross_v, torch.zeros(2, 3)),
            out / "decoder_step.onnx",
            input_names=["step_ids", "past_k", "past_v", "cross_k", "cross_v", "cross_mask"],
            output_names=["logits", "present_k", "present_v"],
            dynamic_shapes={"step_ids": {0: B}, "past_k": {1: B, 3: P}, "past_v": {1: B, 3: P},
                            "cross_k": {1: B, 3: S}, "cross_v": {1: B, 3: S}, "cross_mask": {0: B, 1: S}},
            opset_version=OPSET, dynamo=True, external_data=False,
        )


def folder_config(config: dict, source: str) -> dict:
    from netra_nmt import __version__
    heads = config["n_heads"]
    return {
        "name": "netra-nmt",
        "version": __version__,
        "task": "translation",
        "languages": ["en", "km"],
        "graphs": {"encoder": "encoder.onnx", "decoder_step": "decoder_step.onnx"},
        "components": {
            "preprocess": {
                "type": "nmt-source",
                "lang_markers": config.get("lang_markers", {"en": "<2en>", "km": "<2km>"}),
                "max_len": config.get("max_len", 256),
                "en_source": "nfc-collapse-whitespace-lowercase",
                "km_source": "khmercut-segment",
            },
            "decoder": {
                "type": "kv-cache", "layers": config["dec_layers"], "heads": heads,
                "head_dim": config["d_model"] // heads, "vocab_size": config["vocab_size"],
                "pad_id": config.get("pad_id", 0), "bos_id": config.get("bos_id", 2), "eos_id": config.get("eos_id", 3),
            },
            "tokenizer": {"type": "sentencepiece", "model": "spm_32k.model"},
            "postprocess": [{"type": "km-desegment"}],
        },
        "metadata": {
            "source_checkpoint": source,
            "architecture": f"encoder-decoder, {config['enc_layers']} enc / {config['dec_layers']} dec layers, "
                            f"d_model {config['d_model']}, RoPE, RMSNorm, SwiGLU, tied embeddings",
            "license": "MIT",
        },
    }


def parity(folder: Path, model, spec: str, limit: int | None, device: str, beam_size: int) -> float:
    from netra_nmt.inference.nmt import NMTModel
    from netra_nmt.khseg import segment_km
    from netra_nmt.textnorm import normalize_en_source

    from .. import decoding
    from ..data import load_pairs

    pairs = load_pairs(spec, limit)
    onnx_model = NMTModel(folder, device=device)
    worst = 1.0
    for direction, tgt, column, prep in (("en2km", "km", 0, normalize_en_source), ("km2en", "en", 1, segment_km)):
        inputs = [onnx_model.encode_text(prep(p[column]), tgt) for p in pairs]
        for mode in ("greedy", "beam"):
            same, t_torch, t_onnx, diffs = 0, 0.0, 0.0, []
            for ids in inputs:
                t0 = time.perf_counter()
                x = torch.tensor([ids])
                mask = torch.zeros_like(x, dtype=torch.bool)
                with torch.no_grad():
                    if mode == "greedy":
                        want = decoding.greedy_decode(model, x, mask, onnx_model.bos_id, onnx_model.eos_id, 128)
                    else:
                        want = decoding.beam_search(model, x, mask, onnx_model.bos_id, onnx_model.eos_id, 128,
                                                    beam_size=beam_size)
                t1 = time.perf_counter()
                have = onnx_model.greedy(ids) if mode == "greedy" else onnx_model.beam(ids, beam_size=beam_size)
                t2 = time.perf_counter()
                t_torch += t1 - t0
                t_onnx += t2 - t1
                if want == have:
                    same += 1
                elif len(diffs) < 3:
                    diffs.append((onnx_model.decode_ids(want), onnx_model.decode_ids(have)))
            share = same / len(inputs)
            worst = min(worst, share)
            print(f"{direction} {mode:6s}: identical tokens on {same}/{len(inputs)} ({share:.1%})  "
                  f"torch {t_torch:.1f}s, onnx {t_onnx:.1f}s")
            for want, have in diffs:
                print(f"    torch: {want}\n    onnx:  {have}")
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help='release folder, checkpoint.pt or "published"')
    parser.add_argument("--out", required=True, type=Path, help="the model folder to write")
    parser.add_argument("--spm", type=Path, help="tokenizer (default: the one beside --model, else training/tokenizer/spm_32k_kmseg.model)")
    parser.add_argument("--check", default="", help='test set for the parity check, e.g. "alt" (empty: skip)')
    parser.add_argument("--limit", type=int, help="check only the first N pairs")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="device for the ONNX side of the check")
    parser.add_argument("--keep-fp32", action="store_true", help="don't store the weights as fp16")
    args = parser.parse_args(argv)

    from ..checkpoint import DEFAULT_SPM, load
    from ..export_utils import store_fp16, write_config

    model, config, spm = load(args.model)
    spm = args.spm or spm or DEFAULT_SPM
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    heads = config["n_heads"]
    print(f"Exporting {args.model} -> {out}")
    export_graphs(model, out, heads, config["d_model"] // heads, config["dec_layers"])
    if not args.keep_fp32:
        # The released weights are fp16 already, so fp16 storage loses nothing.
        for name in ("encoder.onnx", "decoder_step.onnx"):
            print(f"  {name}: {store_fp16(out / name)}")
    shutil.copyfile(spm, out / "spm_32k.model")
    for stale in out.glob("*.onnx.data"):
        stale.unlink()
    write_config(out, folder_config(config, Path(str(args.model)).name))
    for f in sorted(out.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:8.1f} MB")

    if args.check:
        share = parity(out, model, args.check, args.limit, args.device, args.beam_size)
        if share < 0.99:
            print(f"FAILED: only {share:.1%} identical (need 99%)")
            sys.exit(1)
        print("Parity OK")


if __name__ == "__main__":
    main()
