# Third-party licences

Netra NMT's code and weights are MIT (see [LICENSE](LICENSE)). It depends on, or
redistributes parts of, the following.

## Runtime dependencies of `netra-nmt`

| Package | Licence | Used for |
| --- | --- | --- |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (`onnxruntime`, `onnxruntime-gpu`) | MIT | running the model |
| [SentencePiece](https://github.com/google/sentencepiece) | Apache-2.0 | the tokenizer |
| [khmercut](https://pypi.org/project/khmercut/) (and `python-crfsuite`) | Apache-2.0 (khmercut) | Khmer word segmentation |
| [NumPy](https://numpy.org) | BSD-3-Clause | decoding |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | downloading the model |
| [FastAPI](https://github.com/fastapi/fastapi), [uvicorn](https://github.com/encode/uvicorn) (extra `server`) | MIT, BSD-3-Clause | the web app and REST API |

## Code from other Netra Lab projects

`netra_nmt/inference/session.py` and `hub.py` come from [Netra-OCR](https://github.com/netra-ai-lab/Netra-OCR) (MIT, same author).

## Brand marks

`assets/symbol.png`, `assets/wordmark-latin.png` and the web UI's `symbol.svg` / `wordmark-latin.svg` are the Netra Lab marks. Use them unedited, and only to refer to Netra Lab.

## Data

- **Training data:** the four sources listed in the README (§2). Each keeps its own terms; `SeyhaLite/Translate-English-Khmer-All` is Apache-2.0. <!-- TODO: licences of Darayut/khmer-english-pairs-raw, lyfeyvutha/nllb-en-km-316K and KrorngAI/ParaCrawl-English-Khmer-v2 (their cards state none) -->
- **Evaluation data and `training/examples/pairs.tsv`:** the Asian Language Treebank ([mutiyama/alt](https://huggingface.co/datasets/mutiyama/alt), NICT), CC BY 4.0.
