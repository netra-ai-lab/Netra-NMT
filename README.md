# netra-nmt

A compact, **from-scratch** encoder-decoder model for **English ↔ Khmer** machine translation.

`netra-nmt` is a ~90M-parameter transformer (SwiGLU FFN, weight-tied decoder, 32k SentencePiece
vocab) trained on **~222 million tokens** of EN–KM parallel text (≈4.2M bidirectional examples).
It is *not* a fine-tune of NLLB — the architecture and
tokenizer are custom and the weights are trained from scratch. This release is the **`small`**
variant (`Darayut/netra-nmt-small` on the Hub); larger variants are planned under the same package. The package ships with a clean Python
API, a CLI, and an optional web demo. Model weights are hosted on the Hugging Face Hub and downloaded
automatically on first use.

## Results

Evaluated on the test splits of `mutiyama/alt` (ALT) and `rinabuoy/khmer-english-parallel`
(greedy decoding, beam size 1):

| Dataset   | Direction | spBLEU | chrF++ | COMET | BERTScore F1 |
|-----------|-----------|:------:|:------:|:-----:|:------------:|
| ALT       | EN→KM     | 25.39  | 37.15  | 77.22 | 89.92        |
| ALT       | KM→EN     | 18.85  | 45.73  | 76.04 | 91.65        |
| rinabuoy  | EN→KM     | 11.43  | 23.53  | 72.80 | 87.85        |
| rinabuoy  | KM→EN     | 14.63  | 39.54  | 73.66 | 90.03        |

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
t.translate("Hello, how are you?", direction="en2km")   # → "សួស្តី សុខសប្បាយអត់?"
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

A two-pane translation site (source left, output right, EN⇄KM swap button) plus a JSON API:

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you?", "direction": "en2km"}'
# {"translation": "...", "direction": "en2km"}
```

Requires the `web` extra (`pip install "netra-nmt[web]"`).

## Model details

| | |
|---|---|
| Architecture | encoder-decoder transformer, Pre-LN, SwiGLU FFN |
| Size | d_model 512, 6 enc / 6 dec layers, 8 heads, ffn 2048 |
| Parameters | ~89.7M |
| Tokenizer | SentencePiece unigram, 32k vocab (bundled with the package) |
| Direction control | source is prefixed with `<2km>` (→Khmer) or `<2en>` (→English) |
| Released weights | fp16 `safetensors`, ~180 MB, on the Hugging Face Hub |

## Repository layout

```
netra_nmt/                 # the installable package (model + API + CLI + web)
scripts/
  export_checkpoint.py     # checkpoint.pt -> fp16 safetensors + config.json (+ HF upload)
  training/                # full data + training + evaluation pipeline (research code)
results/                   # evaluation results + training logs
```

## Re-exporting / re-training

To regenerate the release artifacts from a training checkpoint:

```bash
python scripts/export_checkpoint.py                       # -> export/
python scripts/export_checkpoint.py --push Darayut/netra-nmt-small   # also upload to the Hub
```

The end-to-end data preparation, training, and evaluation scripts live in `scripts/training/`
(numbered `01_*` … `11_*`, plus `09_train_student.py` and `evaluate_alt.py`). They require the
training extras: `pip install -e ".[train]"`.

## License

MIT.
