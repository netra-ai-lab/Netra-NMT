# Example data

`pairs.tsv`: 50 English-Khmer sentence pairs, one per line, `en<TAB>km`, no header.
This is the layout `check_data.sh`, `finetune.sh` and `evaluate.sh` read.

The pairs are the first 50 of the validation split of the Asian Language
Treebank ([mutiyama/alt](https://huggingface.co/datasets/mutiyama/alt), NICT),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are
not part of the ALT test split the model is scored on.

```bash
bash training/check_data.sh training/examples/pairs.tsv
bash training/evaluate.sh published --test training/examples/pairs.tsv
bash training/finetune.sh training/examples/pairs.tsv --name example   # 1 epoch
```
