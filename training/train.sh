#!/usr/bin/env bash
# Train from scratch (or continue a run) on the prepared training data.
#
#   bash training/train.sh [--name NAME] [trainer options]
#   bash training/train.sh --name NAME --resume-from training/runs/NAME/checkpoints/epoch_01
#
# Default data: training/data/processed/combined_casenorm_kmseg_{train,valid}.jsonl,
# built by the recipe (netra_nmt_train/scripts/recipe/01..08d, see
# training/README.md). Pass --train-file/--valid-file for other data (made with
# `netra-nmt-train prepare`). Multi-GPU: TORCHRUN=1 NPROC=2 bash training/train.sh
# The run goes to training/runs/NAME/ (checkpoints/, train_log.jsonl).
# Check the training data's licences before using a model commercially.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { usage; exit 0; }
name="train-$(date +%Y%m%d-%H%M%S)" extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
run="$TRAINING_DIR/runs/$name"
trainer="$TRAINING_DIR/netra_nmt_train/scripts/recipe/09_train_student.py"
args=(--ckpt-dir "$run/checkpoints" --log-file "$run/train_log.jsonl" "${extra[@]}")
if [[ -n "${TORCHRUN:-}" ]]; then
  torchrun --nproc_per_node="${NPROC:-2}" "$trainer" "${args[@]}"
else
  python "$trainer" "${args[@]}"
fi
echo
echo "Model: $run/checkpoints/best/checkpoint.pt. Score it with: bash training/evaluate.sh $run/checkpoints/best/checkpoint.pt"
