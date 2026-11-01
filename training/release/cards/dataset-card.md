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
  - netra-lab
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

<p align="center">
  <img src="https://huggingface.co/datasets/Netra-Lab/bilingual-en-km/resolve/main/assets/header.png" width="260" alt="Netra Lab" />
</p>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Netra-NMT"><b>GitHub</b></a> |
  <a href="https://huggingface.co/Netra-Lab/Netra-NMT"><b>Model</b></a> |
  <a href="https://huggingface.co/spaces/Darayut/Netra-NMT"><b>Demo</b></a>
</p>

# Bilingual English–Khmer Parallel Corpus

4,257,178 English↔Khmer translation examples, in both directions: the training corpus of
[Netra NMT](https://huggingface.co/Netra-Lab/Netra-NMT), a 77M-parameter translation model. Four public
English–Khmer datasets were merged, deduplicated and length-filtered (about 2.4M unique pairs), and every
pair appears once per direction. One row is one example: a source with a direction marker and its target.

**Code:** [github.com/netra-ai-lab/Netra-NMT](https://github.com/netra-ai-lab/Netra-NMT), with the recipe
that built it (`training/netra_nmt_train/scripts/recipe/`).

| Split | Examples | Files |
| --- | ---: | --- |
| `train` | 4,214,606 | `data/train-*.parquet` |
| `validation` | 42,572 | `data/validation-*.parquet` |

| Column | |
| --- | --- |
| `source` | the input, after a direction marker: `<2km>` (English, translate into Khmer) or `<2en>` (Khmer, translate into English) |
| `target` | the reference translation |

```json
{"source": "<2km> A year later, the scientists were able to detect these waves.", "target": "មួយឆ្នាំក្រោយមក អ្នកវិទ្យាសាស្ត្រអាចរកឃើញរលកទាំងនេះ។"}
{"source": "<2en> សីតុណ្ហភាពប្រតិបត្តិការ -20 ~ 55 ℃", "target": "Operating Temperature -20~55℃"}
```

## Use

```python
from datasets import load_dataset
ds = load_dataset("Netra-Lab/bilingual-en-km", split="train")
```

## How Netra NMT used it

The rows keep their natural casing and unsegmented Khmer. Before training, the recipe
(`08c_build_casenorm_dataset.py`, `08d_build_kmseg_from_casenorm.py`) lowercases English sources and
word-segments Khmer with khmercut on both sides. `netra-nmt-train prepare` applies the same rules to your
own pairs.

The model is scored on a separate set, the ALT test split (`mutiyama/alt`, 1,018 pairs), not on this
validation split:

```bash
bash training/evaluate.sh published --test alt
```

## Sources

| Source | License |
| --- | --- |
| [lyfeyvutha/nllb-en-km-316K](https://huggingface.co/datasets/lyfeyvutha/nllb-en-km-316K) | see the original repo |
| [Netra-Lab/khmer-english-pairs-raw](https://huggingface.co/datasets/Netra-Lab/khmer-english-pairs-raw) | MIT |
| [KrorngAI/ParaCrawl-English-Khmer-v2](https://huggingface.co/datasets/KrorngAI/ParaCrawl-English-Khmer-v2) | see the original repo |
| [SeyhaLite/Translate-English-Khmer-All](https://huggingface.co/datasets/SeyhaLite/Translate-English-Khmer-All) | Apache-2.0 |

## License

This card is CC BY 4.0. The text keeps the terms of the source it came from (table above); check them
before redistribution or commercial use, and cite the sources.

```bibtex
@misc{bilingual_en_km,
  title        = {Bilingual English--Khmer Parallel Corpus},
  author       = {Nhem Darayut},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Netra-Lab/bilingual-en-km}},
  note         = {Combined from lyfeyvutha/nllb-en-km-316K, Netra-Lab/khmer-english-pairs-raw,
                  KrorngAI/ParaCrawl-English-Khmer-v2, and SeyhaLite/Translate-English-Khmer-All}
}
```
