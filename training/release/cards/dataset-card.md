---
license: cc-by-4.0
task_categories:
  - translation
language:
  - en
  - km
tags:
  - translation
  - machine-translation
  - khmer
  - english
  - bilingual
  - nllb
pretty_name: Bilingual English–Khmer Parallel Corpus
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
---

# Bilingual English–Khmer Parallel Corpus

The training corpus of [Netra NMT](https://huggingface.co/Netra-Lab/Netra-NMT) (77M parameters,
English↔Khmer). Code and the full recipe: [github.com/netra-ai-lab/Netra-NMT](https://github.com/netra-ai-lab/Netra-NMT).

A combined, deduplicated English↔Khmer parallel corpus for training sequence-to-sequence
machine-translation models. It merges four publicly available English–Khmer datasets into a
single bidirectional training set, formatted with NLLB-style direction tokens so a single
model can be trained to translate in **both** directions.

## Dataset summary

| Split        | Pairs       | Files                          |
|--------------|-------------|--------------------------------|
| `train`      | 4,214,606   | `data/train-*.parquet`         |
| `validation` | 42,572      | `data/validation-*.parquet`    |

Each example is bidirectional: the corpus contains both EN→KM and KM→EN directions, indicated
by a leading direction token on the `source` field.

## Format

The data is stored as Parquet with two string columns, `source` and `target`. Each example
looks like:

```json
{"source": "<2km> A year later, the scientists were able to detect these waves.", "target": "មួយឆ្នាំក្រោយមក អ្នកវិទ្យាសាស្ត្រអាចរកឃើញរលកទាំងនេះ។"}
{"source": "<2en> សីតុណ្ហភាពប្រតិបត្តិការ -20 ~ 55 ℃", "target": "Operating Temperature -20~55℃"}
```

- **`source`** — the input text, prefixed with a direction token:
  - `<2km>` → translate the following English text **into Khmer**
  - `<2en>` → translate the following Khmer text **into English**
- **`target`** — the reference translation in the target language.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Netra-Lab/bilingual-en-km")
print(ds["train"][0])
# {'source': '<2km> ...', 'target': '...'}
```

## How Netra NMT used it

The rows here keep their natural casing and unsegmented Khmer. Before training, Netra NMT's
recipe (`training/netra_nmt_train/scripts/recipe/08c_build_casenorm_dataset.py` and
`08d_build_kmseg_from_casenorm.py`) lowercases English *sources* and word-segments Khmer with
khmercut on both sides. `netra-nmt-train prepare` applies the same rules to your own pairs.

The model is scored on a separate set, the ALT test split (`mutiyama/alt`, 1,018 pairs):

```bash
bash training/evaluate.sh published --test alt
```

## Source datasets

This corpus is a combination of the following four datasets. Please cite the original
sources if you use this data, and consult each source for its own licensing terms.

- [lyfeyvutha/nllb-en-km-316K](https://huggingface.co/datasets/lyfeyvutha/nllb-en-km-316K)
- [Darayut/khmer-english-pairs-raw](https://huggingface.co/datasets/Darayut/khmer-english-pairs-raw)
- [KrorngAI/ParaCrawl-English-Khmer-v2](https://huggingface.co/datasets/KrorngAI/ParaCrawl-English-Khmer-v2)
- [SeyhaLite/Translate-English-Khmer-All](https://huggingface.co/datasets/SeyhaLite/Translate-English-Khmer-All)

## Licensing

This combined dataset card is released under **CC-BY-4.0**. The underlying text originates
from the four source datasets listed above, each of which carries its own license and terms.
Users are responsible for reviewing and complying with the license of each source dataset
before redistribution or commercial use.

## Citation

If you use this dataset, please cite this repository along with the four source datasets:

```bibtex
@misc{bilingual_en_km,
  title        = {Bilingual English--Khmer Parallel Corpus},
  author       = {Nhem Darayut},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Netra-Lab/bilingual-en-km}},
  note         = {Combined from lyfeyvutha/nllb-en-km-316K, Darayut/khmer-english-pairs-raw,
                  KrorngAI/ParaCrawl-English-Khmer-v2, and SeyhaLite/Translate-English-Khmer-All}
}
```
