# Architecture of the package

How `netra-nmt` turns a sentence into a translation, and what is in a model folder.
The model itself (layers, sizes, training) is described in the README §3.

## From text to text

```
text ── textnorm / khseg ── "<2km> source" ── SentencePiece ── ids
     ── encoder.onnx ── cross_k, cross_v (every decoder layer, once per sentence)
     ── decoder_step.onnx × N (greedy / beam / sampling in numpy, KV cache in and out)
     ── SentencePiece decode ── desegment Khmer ── text
```

| Step | Where | What |
| --- | --- | --- |
| Source preparation | `netra_nmt/textnorm.py`, `khseg.py` | en2km: NFC, whitespace collapsed, lowercased. km2en: khmercut word segmentation. Exactly the rules the training data was built with. |
| Tokens | `netra_nmt/inference/nmt.py` `encode_text` | the target-language marker (`<2km>` / `<2en>`) + the source, SentencePiece, cut at 256 tokens |
| Encoder | `encoder.onnx` | 12 layers; outputs the cross-attention keys and values of both decoder layers, so the encoder states never leave the graph |
| Decoding | `nmt.py` `greedy` / `beam` / `sample` | ports of `training/netra_nmt_train/decoding.py`: same stopping rule, GNMT length penalty `((5+len)/6)^0.6`, top-p filter |
| Output | `nmt.py` `decode_ids`, `khseg.desegment_km` | special tokens dropped; for Khmer, the spaces between Khmer characters that segmentation inserted are removed |

## The two graphs

Exported by `training/netra_nmt_train/onnx_graphs.py` (`netra-nmt-train export`), opset 18, dynamic batch and lengths.
L = 2 decoder layers, H = 8 heads, hd = 64.

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `encoder.onnx` | `tokens` (B, S) int64, `mask` (B, S) int64 (1 = token) | `cross_k`, `cross_v` (L, B, H, S, hd) |
| `decoder_step.onnx` | `step_ids` (B, 1), `past_k`, `past_v` (L, B, H, P, hd), `cross_k`, `cross_v`, `cross_mask` (B, S) float (0 or -inf) | `logits` (B, 32000), `present_k`, `present_v` (L, B, H, P+1, hd) |

The RoPE position of the new token is the cache length `P`, as `pos_offset=step` in the PyTorch model, so the step needs no position input. The first step gets an empty cache (`P = 0`).

Weights are **stored** as fp16 behind a `Cast` and computed in fp32. The released weights were fp16 already, so storing them as fp16 loses nothing.

## The model folder

```
model/
├── netra-model.json     settings + sha256 of every file
├── encoder.onnx         136 MB
├── decoder_step.onnx     52 MB
└── spm_32k.model        SentencePiece (Khmer-segmented, 32K)
```

`netra-model.json` holds everything the code needs, so nothing is hard-coded beside it:

```json
{
  "format": "netra-bundle", "format_version": 1,
  "name": "netra-nmt", "task": "translation", "languages": ["en", "km"],
  "graphs": {"encoder": "encoder.onnx", "decoder_step": "decoder_step.onnx"},
  "components": {
    "preprocess": {"type": "nmt-source", "lang_markers": {"en": "<2en>", "km": "<2km>"}, "max_len": 256,
                   "en_source": "nfc-collapse-whitespace-lowercase", "km_source": "khmercut-segment"},
    "decoder": {"type": "kv-cache", "layers": 2, "heads": 8, "head_dim": 64, "vocab_size": 32000,
                "pad_id": 0, "bos_id": 2, "eos_id": 3},
    "tokenizer": {"type": "sentencepiece", "model": "spm_32k.model"},
    "postprocess": [{"type": "km-desegment"}]
  },
  "files": {"...": "sha256"}
}
```

## Where the model comes from

`netra_nmt/models.py`: `hf://Netra-Lab/Netra-NMT/model@<MODEL_REVISION>`, a pinned commit, downloaded once into the Hugging Face cache. Overrides:

| Variable | Effect |
| --- | --- |
| `NETRA_NMT_BUNDLES=<dir>` | use `<dir>/model/` (e.g. `training/bundles` from `export.sh`); without it, Hugging Face |
| `NETRA_NMT_DEVICE` | `cpu`, `cuda`, `cuda:N`, `tensorrt`, `openvino`; default: CUDA if usable, else CPU |
| `NETRA_NMT_ORT_SPIN=1` | let ONNX Runtime's idle threads spin (off by default) |
| `HF_HUB_OFFLINE=1` | never touch the network once the model is cached |

On CUDA, greedy decoding keeps the cross-attention and KV-cache tensors on the GPU between steps (`Session.run(..., keep=...)`); beam search and sampling copy the cache to the host to reorder it.
