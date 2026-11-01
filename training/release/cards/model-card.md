---
license: mit
language:
- en
- km
library_name: onnx
pipeline_tag: translation
tags:
- translation
- khmer
- english
- onnx
- netra-lab
datasets:
- Netra-Lab/bilingual-en-km
---

# Netra NMT

A **77M-parameter** English↔Khmer translation model: one encoder-decoder transformer for both directions, trained from scratch on 220 million tokens of parallel text. It runs on ONNX Runtime with the `netra-nmt` package, on a CPU, without PyTorch.

**Code:** [github.com/netra-ai-lab/Netra-NMT](https://github.com/netra-ai-lab/Netra-NMT), with the inference package and the full training recipe.

## Results

ALT test split (`mutiyama/alt`, 1,018 pairs), greedy decoding; spBLEU with the FLORES-200 tokenizer, higher is better.

| Model | Params | EN → KM | KM → EN |
|---|---:|---:|---:|
| NLLB-350M (EN-KM) | 350M | 18.89 | 0.26 |
| NLLB-200 distilled 600M | 600M | 9.24 | 31.28 |
| NLLB-200 distilled 1.3B | 1.3B | 16.62 | 35.15 |
| MADLAD-400 | 3B | 16.19 | 36.13 |
| Google Translate | — | 28.93 | 34.86 |
| **Netra NMT** | **77M** | **25.83** | **19.47** |

- **Where it wins:** English→Khmer, the best spBLEU after Google Translate and the best TER of all systems (60.47), and on CPU 65.6 ms per sentence greedy against 550.8 for NLLB-350M.
- **Where it loses:** Khmer→English, 15.7 spBLEU behind NLLB-200 1.3B.

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-NMT/resolve/main/assets/metrics_benchmark.png" width="100%" alt="spBLEU, chrF++, TER, COMET and BERTScore on ALT"></p>

*spBLEU, chrF++, TER, COMET (wmt22-comet-da) and BERTScore F1 (xlm-roberta-base) on ALT, both directions.*

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-NMT/resolve/main/assets/speed_benchmark.png" width="100%" alt="Speed against NLLB on CPU and GPU"></p>

*English→Khmer, median ms per sentence over 20 sentences, Intel i7-13700KF (16 threads) and RTX 3060, PyTorch.*

## Architecture

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-NMT/resolve/main/assets/model_architecture.png" width="70%" alt="Encoder-decoder architecture"></p>

- **Shape:** 12 encoder layers, 2 decoder layers (deep encoder, shallow decoder), d_model 512, 8 heads, FFN 2048, vocabulary 32K.
- **Components:** rotary position embeddings, RMSNorm (pre-norm), SwiGLU, one embedding table shared by the encoder, the decoder and the tied output head, a KV cache.
- **Input:** a target-language marker (`<2km>` or `<2en>`) then the source. English sources are NFC-normalised and lowercased; Khmer is word-segmented with khmercut. The package does this for you.

## How it was made

1. **Data.** About 2.4M unique pairs from four sources (synthetic and ParaCrawl), deduplicated, length-filtered, then used in both directions: 4.2M examples.
2. **Tokenizer.** A 32K SentencePiece unigram model shared by both languages, trained on Khmer segmented with khmercut.
3. **Training.** From scratch, with label smoothing and a warm-up + cosine learning-rate schedule (settings in `training/netra_nmt_train/scripts/recipe/09_train_student.py`); the checkpoint with the best validation chrF is released.
4. **Export.** The encoder and a one-step decoder with a KV cache as ONNX graphs, weights stored as fp16. The exported folder generates the same tokens as the PyTorch model on the ALT test set (greedy and beam-5, both directions).

## Usage

```bash
pip install "netra-nmt[cpu]"        # or [gpu]
```

```python
from netra_nmt import translate

translate("Hello, how are you?")                         # English -> Khmer
translate("ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា។", direction="km2en")    # -> "I love Cambodia."
```

```bash
netra-nmt --text "Hello, how are you?" --mode beam
netra-nmt serve                       # web app + REST API (extra: server)
```

The package downloads `model/` from this repo once, at the commit it was tested with.

## Files

| File | Contents |
|---|---|
| `model/netra-model.json` | settings: graphs, decoder shape, special ids, language markers, sha256 of every file |
| `model/encoder.onnx` | tokens → cross-attention keys/values for every decoder layer (136 MB) |
| `model/decoder_step.onnx` | one decoding step with the self-attention KV cache (52 MB) |
| `model/spm_32k.model` | the SentencePiece tokenizer |
| `pytorch/model.safetensors` | fp16 weights for fine-tuning (`lm_head` tied to the embeddings, not stored) |
| `pytorch/config.json`, `pytorch/spm_32k.model` | architecture and tokenizer for the PyTorch model |
| `assets/` | the figures on this card |

To fine-tune from `pytorch/`: `bash training/finetune.sh my_pairs.tsv` in the GitHub repo.

## License and attribution

- **Weights:** MIT, trained from scratch (no pretrained weights).
- **Training data:** see the [dataset card](https://huggingface.co/datasets/Netra-Lab/bilingual-en-km) and its sources; check their terms before commercial use.
- **Evaluation data:** ALT (NICT), CC BY 4.0.
