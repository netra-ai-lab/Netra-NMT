# Training, fine-tuning and evaluation

Everything to reproduce Netra NMT, fine-tune it on your own sentence pairs, score it, and
export it for the package. This is its own package, `netra-nmt-train`, with PyTorch; the
`netra-nmt` package never imports it.

```
training/
├── setup.sh  _env.sh                      a .venv with netra-nmt (editable) + netra-nmt-train
├── check_data.sh  finetune.sh  train.sh   one job per script; --help prints the usage
├── evaluate.sh  export.sh
├── examples/pairs.tsv                     50 ALT validation pairs to try every script
├── tokenizer/spm_32k_kmseg.model          the model's tokenizer (32K, Khmer-segmented)
├── netra_nmt_train/                       model.py, decoding.py, data.py, checkpoint.py, onnx_graphs.py, cli.py
│   └── scripts/                           check_data, prepare, evaluate, export
│       └── recipe/                        the full pipeline that built the model (01 … 10)
├── release/                               HF upload script, model / dataset cards, the Space
└── runs/  bundles/  data/  checkpoints*/  git-ignored outputs and local data
```

## 1. Set up

```bash
bash training/setup.sh            # .venv in the repository root (uv if installed, else venv + pip)
```

It installs the package with the ONNX Runtime build that fits the machine (GPU when
`nvidia-smi` sees one) and the training package with PyTorch. Every other script activates
that environment (`VENV=...` to use another).

## 2. Your data

Parallel data is a list of English-Khmer sentence pairs, in any of these forms
([`netra_nmt_train/data.py`](netra_nmt_train/data.py)):

```
my_pairs.tsv              en<TAB>km, one pair per line, no header
my_pairs.csv / .jsonl     columns "en" and "km"
user/name[:split]         a Hugging Face dataset with en/km columns (or a "translation" dict)
alt                       the ALT test split (mutiyama/alt), also alt:validation
```

Check it first. Output from `examples/pairs.tsv`:

```
$ bash training/check_data.sh training/examples/pairs.tsv
training/examples/pairs.tsv: 50 pairs
  en: median 24 tokens, max 69, 0 over 256 (cut)
  km: median 38 tokens, max 128, 0 over 256 (cut)
  duplicates: 0   km side without Khmer: 0   looks swapped: 0
```

You don't segment or lowercase anything yourself: `netra-nmt-train prepare` (run by
`finetune.sh`) writes both directions with the markers, English sources lowercased and Khmer
word-segmented, exactly like the original training data.

## 3. Fine-tune

```bash
bash training/finetune.sh my_pairs.tsv --name my-model
bash training/finetune.sh my_pairs.tsv --name my-model --peak-lr 1e-5 --extra-epochs 3
```

It starts from the published weights (`--init published`, or a release folder / checkpoint),
holds out 5% of the pairs for validation, and runs
[`recipe/09_train_student.py`](netra_nmt_train/scripts/recipe/09_train_student.py) with a
fresh optimizer: peak LR 3e-5, 200 warm-up steps, cosine to 1e-6, one epoch. Each epoch
prints train/validation loss, sample translations and a validation chrF; the checkpoint with
the best chrF is kept. Output from a 20-step run on the example pairs (RTX 3060):

```
48 train pairs -> 96 rows, 2 validation pairs -> 4 rows (both directions) in training/runs/smoke-finetune/data
Device: cuda:0  |  World size: 1  |  AMP dtype: torch.bfloat16
Parameters: 77,217,792
Initialising weights from training/export_rope (fresh optimizer + schedule)
Epoch 1  train=3.6210  val=3.4140
```

The run goes to `training/runs/NAME/`: `data/`, `checkpoints/best/checkpoint.pt`,
`checkpoints/step_latest/`, `train_log.jsonl`.

## 4. Train from scratch

The recipe, in order (`training/netra_nmt_train/scripts/recipe/`):

| Step | Scripts | Output |
| --- | --- | --- |
| Download and clean the sources | `01_extract_*`, `02_extract_*`, `02b_clean_new_sources` | `data/raw/`, `data/processed/` |
| Deduplicate, filter, audit | `03_deduplicate`, `04_quick_filter`, `05_audit` | cleaned pairs |
| Tokenizer | `06_build_tokenizer_corpus`, `07_train_tokenizer` | `tokenizer/spm_32k_kmseg.model` |
| Bidirectional dataset | `08_build_bilingual_dataset`, `08b_build_combined_dataset`, `08c_build_casenorm_dataset`, `08d_build_kmseg_from_casenorm` | `data/processed/combined_casenorm_kmseg_{train,valid}.jsonl` |
| Train | `09_train_student` (via `train.sh`) | `runs/NAME/checkpoints/` |
| Distillation (optional) | `teacher/01..03`, `10_distill` | |

```bash
bash training/train.sh --name scratch                         # one GPU
TORCHRUN=1 NPROC=2 bash training/train.sh --name scratch      # two GPUs (DDP)
bash training/train.sh --name scratch --resume-from training/runs/scratch/checkpoints/epoch_01
```

The recipe also needs `pip install -e "training[recipe]"` (pandas, pyarrow, langdetect, fasttext).

## 5. Evaluate

```bash
bash training/evaluate.sh published                                  # the shipped model, on ALT
bash training/evaluate.sh published training/runs/my-model/checkpoints/best/checkpoint.pt --test my_test.tsv
bash training/evaluate.sh training/bundles/model --mode beam --limit 200
```

A model is `published`, an exported model folder (run by the package, exactly as users run
it) or a PyTorch checkpoint / release folder. It reports spBLEU (FLORES-200 tokenizer),
chrF++ and TER per direction, and writes the predictions and `summary.json` to
`training/runs/eval-<date-time>/`. Output on the example pairs:

```
| Model | Test | Direction | spBLEU | chrF++ | TER |
|---|---|---|---:|---:|---:|
| training/bundles/model | training/examples/pairs.tsv | en2km | 27.01 | 39.49 | 60.78 |
| training/bundles/model | training/examples/pairs.tsv | km2en | 18.07 | 46.07 | 65.91 |
```

COMET and BERTScore, as in the README table:
`python training/netra_nmt_train/scripts/recipe/evaluate_alt.py` (needs `training[metrics]`).

## 6. Export for the package

```bash
bash training/export.sh training/runs/my-model/checkpoints/best/checkpoint.pt
NETRA_NMT_BUNDLES=training/runs/my-model/checkpoints/best/bundles netra-nmt --text "Hello"
```

It writes the model folder (`encoder.onnx`, `decoder_step.onnx`, `spm_32k.model`,
`netra-model.json`; see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)), then runs it through
the package on the ALT test set and compares the generated tokens with PyTorch, greedy and
beam-5, both directions. It fails below 99% identical. `LIMIT=50` checks fewer pairs,
`CHECK=` skips the check. <!-- PARITY_OUTPUT -->
