---
license: mit
task_categories:
- translation
language:
- km
- en
tags:
- translation
- khmer
- english
- synthetic
- netra-lab
pretty_name: Khmer-English Pairs (raw)
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: kh
    dtype: string
  - name: en
    dtype: string
  splits:
  - name: train
    num_bytes: 78595459
    num_examples: 200000
  download_size: 31554694
  dataset_size: 78595459
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Khmer-English Pairs (raw)

200,000 Khmer-English sentence pairs, LLM-generated synthetic data in legal, literary and
governmental registers. One of the four training sources of
[Netra NMT](https://huggingface.co/Netra-Lab/Netra-NMT); the combined, deduplicated corpus is
[Netra-Lab/bilingual-en-km](https://huggingface.co/datasets/Netra-Lab/bilingual-en-km).

**Code:** [github.com/netra-ai-lab/Netra-NMT](https://github.com/netra-ai-lab/Netra-NMT)
(the recipe that cleans and combines it: `training/netra_nmt_train/scripts/recipe/`).

## Columns

| Column | Contents |
| --- | --- |
| `kh` | Khmer sentence (natural text, not word-segmented) |
| `en` | English sentence |

```python
from datasets import load_dataset

ds = load_dataset("Netra-Lab/khmer-english-pairs-raw", split="train")
print(ds[0])
# {'kh': 'ព្រះរាជរដ្ឋាភិបាលកម្ពុជាប្តេជ្ញាចិត្តខ្ពស់ក្នុងការលើកកម្ពស់នីតិរដ្ឋ និងយុត្តិធម៌សង្គម។',
#  'en': 'The Royal Government of Cambodia is highly committed to promoting the rule of law and social justice.'}
```

"Raw" means as generated: no deduplication or length filtering (6,738 of the 200,000 pairs
are exact duplicates). The Netra NMT recipe does both before training.

The Netra NMT training scripts read it by name:

```
$ bash training/check_data.sh Netra-Lab/khmer-english-pairs-raw:train
Netra-Lab/khmer-english-pairs-raw:train: 200,000 pairs
  en: median 18 tokens, max 67, 0 over 256 (cut)
  km: median 29 tokens, max 107, 0 over 256 (cut)
  duplicates: 6,738   km side without Khmer: 0   looks swapped: 0

$ bash training/finetune.sh Netra-Lab/khmer-english-pairs-raw:train --name my-model
```

## License

MIT. Earlier copy: [Darayut/khmer-english-pairs-raw](https://huggingface.co/datasets/Darayut/khmer-english-pairs-raw).
