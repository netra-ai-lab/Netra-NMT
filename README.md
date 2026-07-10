<div align="center">
  <img
    src="https://raw.githubusercontent.com/netra-ai-lab/Netra-NMT/main/assets/logo.png"
    width="20%"
    alt="Netra Lab"
    style="vertical-align: middle; margin-right: 30px;"
  />
  <img
    src="https://raw.githubusercontent.com/netra-ai-lab/Netra-NMT/main/assets/wordmark.png"
    width="20%"
    alt="Netra Lab"
    style="vertical-align: middle;"
  />
</div>

<hr>

<p align="center">
 <a href="https://github.com/netra-ai-lab/Netra-NMT"><b>GitHub</b></a> |
  <a href="https://huggingface.co/Darayut/netra-nmt-small"><b>Model Download</b></a> |
    <a href="https://huggingface.co/datasets/Darayut/bilingual-en-km"><b>Dataset Download</b></a> |
    <a href="https://huggingface.co/spaces/Darayut/Netra-NMT"><b>Inference Space</b></a> |
</p>

<h2>
<p align="center">
  <a href="">A Compact Bidirectional Encoder-Decoder Transformer-Based Model for English-Khmer Translation</a>
</p>
</h2>

<p align="center">
<img src="https://raw.githubusercontent.com/netra-ai-lab/Netra-NMT/main/assets/speed_benchmark.png" style="width: 1000px" align=center>
</p>

<p align="center">
<a href="">Inference Speed Benchmark on CPU (Intel i7-13700KF) and GPU (RTX 3060) using Greedy and Beam-5 decoding</a>       
</p>

## 1. Abstract
This repository presents Netra-NMT, a compact **77M-parameter** encoder-decoder transformer-based model trained from scratch on **220 million tokens** of English-Khmer parallel text (4.2M bidirectional examples). The encoder uses bidirectional self-attention, much like BERT, to capture global contextual representation. The decoder performs autoregressive generation through causal self-attention and encoder-decoder cross-attention.

Netra-NMT adopts a modern transformer recipe: **rotary position embeddings (RoPE)** in self-attention (no learned position table), **RMSNorm** with Pre-Normalization for stable optimization, **SwiGLU** feed-forward networks, and a **single embedding table shared** across the encoder input, decoder input, and tied output projection. It uses a **deep-encoder / shallow-decoder** layout (12 encoder layers, 2 decoder layers) which — combined with a **KV cache** for O(T) incremental decoding — delivers autoregressive quality at a fraction of the decoding latency (Kasai et al., 2021). Khmer text is word-segmented with **khmercut** before tokenization, so the shared 32K SentencePiece vocabulary sees real word boundaries.

## 2. Dataset

Netra-NMT was trained on **220 million tokens** drawn from approximately **2.4 million unique English-Khmer sentence pairs** (4.2 million examples after bidirectional augmentation). The corpus combines LLM-generated synthetic data with web-crawled parallel text, spanning legal, literary, medical, technical, and conversational domains.

### 2.1 Sources

| Dataset | Type | Pairs | Domains |
|---------|------|------:|---------|
| [Darayut/khmer-english-pairs-raw](https://huggingface.co/datasets/Darayut/khmer-english-pairs-raw) | Synthetic | 200K | Legal, Literary, Governmental |
| [lyfeyvutha/nllb-en-km-316K](https://huggingface.co/datasets/lyfeyvutha/nllb-en-km-316K) | Synthetic | 316K | General |
| [KrorngAI/ParaCrawl-English-Khmer-v2](https://huggingface.co/datasets/KrorngAI/ParaCrawl-English-Khmer-v2) | Web crawl (ParaCrawl) | 1.5M | Web / general |
| [SeyhaLite/Translate-English-Khmer-All](https://huggingface.co/datasets/SeyhaLite/Translate-English-Khmer-All) | --- | 366K | General |
| **Total** | | **2.4M** | |

### 2.2 Preprocessing

Raw data was cleaned through the following pipeline:

1. **Deduplication**: exact duplicate pairs removed across all sources.
2. **Length filtering**: pairs with extreme source/target length mismatches were discarded.
3. **Empty/null removal**: pairs where either side was empty or below a minimum token count were dropped.
4. **English source case-normalization**: English is NFC-normalized, whitespace-collapsed, and lowercased **only when it is the `en2km` source**, so `"I love Cambodia"` and `"i love cambodia"` map to identical model inputs. English kept as a `km2en` *target* retains its natural casing.
5. **Khmer word segmentation**: Khmer (which is written without spaces) is segmented with **khmercut** wherever it appears — as the `en2km` target and the `km2en` source — matching the segmented tokenizer. Khmer output is de-segmented back to natural text at inference time.

After cleaning, each surviving pair is duplicated in both directions (`EN→KM` and `KM→EN`) with a direction prefix token (`<2km>` / `<2en>`), yielding ~4.2 million training examples.

## 3. Model Architecture

<div align="center">
  <img src="https://raw.githubusercontent.com/netra-ai-lab/Netra-NMT/main/assets/model_architecture.png" width="70%" alt="Netra-NMT Architecture" />
  <p><em>Figure 1: Overview of the Netra-NMT encoder-decoder architecture. The encoder (left) processes the source sentence with bidirectional self-attention; the decoder (right) generates the target sentence autoregressively via causal self-attention and cross-attention over the encoder output. Both sides share a 32K SentencePiece tokenizer.</em></p>
</div>

Netra-NMT follows an encoder-decoder transformer architecture modernized for training stability, parameter efficiency, and low-latency decoding.

**Encoder** takes the source sentence tokenized by the shared 32K SentencePiece tokenizer (Khmer is word-segmented with khmercut beforehand), and passes the sequence through **12** transformer layers with *bidirectional* self-attention (every token attends to every other token, similar to BERT). Positions are encoded with **rotary embeddings (RoPE)** applied to the queries and keys inside self-attention — there is no learned position table. A final RMSNorm is applied to the encoder output before it is passed to the decoder via cross-attention.

**Decoder** takes the (partially generated) target sentence through the same tokenizer and passes it through only **2** transformer layers. Each decoder layer applies three sub-layers in order: (1) *causal* (masked) self-attention with RoPE over previously generated tokens, backed by a **KV cache** for O(T) incremental decoding, (2) cross-attention over the full encoder output, and (3) a SwiGLU feed-forward block. A final RMSNorm feeds into the tied projection head. The **deep-encoder / shallow-decoder** split (12 vs 2) follows Kasai et al. (2021): most of the modeling capacity lives in the parallelizable encoder, while the thin decoder — the part that runs sequentially at inference — keeps per-step cost low without sacrificing quality.

**Architectural improvements over the vanilla transformer:**

| Feature | Detail |
|---------|--------|
| Rotary Position Embeddings (RoPE) | Positions encoded by rotating Q/K inside self-attention — no learned position table, and generalizes past the training length |
| RMSNorm (Pre-Norm) | Root-mean-square normalization applied *before* each sub-layer; as stable as LayerNorm and cheaper |
| SwiGLU FFN | Feed-forward blocks use the SwiGLU activation instead of ReLU, providing richer representational capacity |
| Shared embeddings | One embedding table is shared across the encoder input, decoder input, and the tied output projection head |
| Deep encoder / shallow decoder | 12-layer encoder + 2-layer decoder (Kasai et al., 2021) — retains quality while sharply cutting sequential decoding cost |
| KV cache | Incremental decoding caches past keys/values, making generation O(T) instead of O(T²) |
| Khmer word segmentation | Khmer is segmented with khmercut before SentencePiece so the tokenizer learns real word boundaries |

**Hyperparameters:**

| | |
|---|---|
| d_model | 512 |
| Encoder / Decoder layers | 12 / 2 |
| Attention heads | 8 |
| FFN hidden size | 2048 |
| Position encoding | Rotary (RoPE) |
| Normalization | RMSNorm (Pre-Norm) |
| Vocabulary | 32K (SentencePiece unigram, shared, Khmer-segmented) |
| Total parameters | ~77M |

## 4. Evaluation Results

<p align="center">
<img src="https://raw.githubusercontent.com/netra-ai-lab/Netra-NMT/main/assets/metrics_benchmark.png" style="width: 1000px" align=center>
</p>

## Install

```bash
pip install netra-nmt              # core (Python API + CLI)
pip install "netra-nmt[web]"       # + FastAPI web app & REST API
```

Or from source:

```bash
git clone https://github.com/NDarayut/netra-nmt
cd netra-nmt
pip install -e ".[web]"
```

The first translation downloads the weights (~180 MB fp16) from the Hugging Face Hub and caches them
under `~/.cache/huggingface`.

## Usage

### 1. Python API

```python
from netra_nmt import NetraTranslator

t = NetraTranslator()                       # auto-detect GPU/CPU; downloads weights once
t.translate("Hello, how are you?", direction="en2km")   # → "សួស្តីសុខសប្បាយអត់?"
t.translate("ខ្ញុំស្រឡាញ់ប្រទេសរបស់ខ្ញុំ។", direction="km2en")

# Batch + decoding options
t.translate_batch(["Good morning.", "See you tomorrow."], direction="en2km")
t.translate("Good morning, my friend.", direction="en2km", mode="beam", beam_size=5)
```

One-shot helper (caches a default translator):

```python
from netra_nmt import translate
translate("Hello", direction="en2km")
```

`direction` is `"en2km"` (English→Khmer) or `"km2en"` (Khmer→English).
`mode` is `"greedy"` (default), `"beam"`, or `"sample"`.

### 2. CLI

```bash
# Single sentence (default direction en2km):
netra-translate --text "Hello, how are you?"

# Khmer → English with beam search:
netra-translate --text "សួស្តី, តើអ្នកសុខសប្បាយទេ?" --direction km2en --mode beam

# Translate a file (one sentence per line):
netra-translate --file input.txt --output output.txt --direction en2km

# Interactive REPL (omit --text / --file):
netra-translate
```

### 3. Web app + REST API (FastAPI)

```bash
netra-web                      # serves the web UI + API at http://127.0.0.1:8000
netra-web --port 8080 --device cpu
netra-web --local-dir export   # load weights from a local export dir
```

A two-pane translation site (source left, output right, EN⇄KM swap button) and a JSON API:

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you?", "direction": "en2km"}'
# {"translation": "...", "direction": "en2km"}
```

Requires the `web` extra (`pip install "netra-nmt[web]"`).