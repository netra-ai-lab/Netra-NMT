#!/usr/bin/env bash
# Check parallel English-Khmer data before training on it.
#
#   bash training/check_data.sh DATA [DATA ...]
#
# DATA is any of (see netra_nmt_train/data.py):
#   alt                          the ALT test set (mutiyama/alt), also alt:validation
#   my_pairs.tsv                 en<TAB>km per line
#   my_pairs.csv / .jsonl        columns "en" and "km"
#   user/name[:split]            a Hugging Face dataset with en/km columns
#
# For each one it prints the number of pairs, duplicates, token lengths (the
# model cuts sources at 256), pairs that look swapped and a few examples.
# Exits with 1 when a DATA can't be read.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }
netra-nmt-train check "$@"
