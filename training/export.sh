#!/usr/bin/env bash
# Export a model as the ONNX model folder the package runs (no PyTorch needed).
#
#   bash training/export.sh MODEL [OUT]
#
# MODEL is a checkpoint.pt, a release folder (model.safetensors + config.json)
# or "published". Writes OUT/model/ (default OUT: a "bundles" folder next to
# MODEL), then checks on the ALT test set that the folder generates the same
# tokens as PyTorch (greedy and beam-5, both directions; it fails below 99%).
# CHECK=DATA picks another test set, CHECK= skips the check, LIMIT=N checks
# only N pairs.
#
# Use it:
#   NETRA_NMT_BUNDLES=OUT netra-nmt --text "Hello"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }

model="$1"
if [[ -d "$model" ]]; then default_out="$model/../bundles"; else default_out="$(dirname "$model")/bundles"; fi
out="${2:-$default_out}"
check="${CHECK-alt}"

args=(--model "$model" --out "$out/model")
[[ -n "$check" ]] && args+=(--check "$check") || echo "Skipping the parity check (CHECK is empty)."
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")
netra-nmt-train export "${args[@]}"

cat <<MSG

Model folder: $out/model
Use it:   NETRA_NMT_BUNDLES=$out netra-nmt --text "Hello"
Score it: bash training/evaluate.sh $out/model
MSG
