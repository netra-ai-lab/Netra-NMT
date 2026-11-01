#!/usr/bin/env bash
# Fine-tune the published model on your own parallel data.
#
#   bash training/finetune.sh DATA [DATA ...] [--name NAME] [--init MODEL] [trainer options]
#
# DATA takes the forms check_data.sh accepts; check it first. The pairs are
# prepared like the original training data (both directions, English source
# lowercased, Khmer word-segmented), 5% held out for validation. MODEL is
# "published" (default), a release folder or a checkpoint.pt.
#
# Defaults: peak LR 3e-5 (warm-up 200 steps, cosine to 1e-6), 1 epoch. Any
# trainer option overrides them, e.g. --train-steps 5000 or --peak-lr 1e-5
# (see: python -m netra_nmt_train.scripts.recipe.09_train_student --help).
# The run goes to training/runs/NAME/ (default: finetune-<date-time>):
#     data/  checkpoints/best/checkpoint.pt (best validation chrF)  train_log.jsonl
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }
data=() name="finetune-$(date +%Y%m%d-%H%M%S)" init="published" extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --init) init="$2"; shift 2 ;;
    --*) extra=("$@"); break ;;
    *) data+=("$1"); shift ;;
  esac
done
[[ ${#data[@]} -gt 0 ]] || { echo "Give at least one DATA file or dataset id" >&2; exit 1; }
run="$TRAINING_DIR/runs/$name"
netra-nmt-train prepare "${data[@]}" --out "$run/data"
python "$TRAINING_DIR/netra_nmt_train/scripts/recipe/09_train_student.py" \
  --init-weights "$init" --train-file "$run/data/train.jsonl" --valid-file "$run/data/valid.jsonl" \
  --ckpt-dir "$run/checkpoints" --log-file "$run/train_log.jsonl" \
  --peak-lr 3e-5 --min-lr 1e-6 --warmup-steps 200 --extra-epochs 1 "${extra[@]}"
cat <<MSG

Fine-tuned model: $run/checkpoints/best/checkpoint.pt
Compare it with the published model:
  bash training/evaluate.sh published $run/checkpoints/best/checkpoint.pt
Then export it for the package:
  bash training/export.sh $run/checkpoints/best/checkpoint.pt
MSG
