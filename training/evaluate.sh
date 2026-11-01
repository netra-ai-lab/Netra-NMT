#!/usr/bin/env bash
# Score models on parallel test sets, both directions (spBLEU, chrF++, TER).
#
#   bash training/evaluate.sh MODEL [MODEL ...] [--test SET [SET ...]] [--mode beam] [--limit N]
#
# MODEL is one of
#   published                                   the model the package ships, run as the package runs it
#   training/runs/NAME/bundles/model            a model folder from export.sh
#   training/runs/NAME/checkpoints/best/checkpoint.pt, or a release folder   (PyTorch)
#
# SET is any form check_data.sh accepts (default: alt, the ALT test split).
# Predictions (CSV) and summary.json go to training/runs/eval-<date-time>/.
# COMET and BERTScore: netra_nmt_train/scripts/recipe/evaluate_alt.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }
netra-nmt-train evaluate "$@"
