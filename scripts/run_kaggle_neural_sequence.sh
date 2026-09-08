#!/usr/bin/env bash
# Kaggle SSH users are recreated after kernel restarts. Keep the GPU runtime
# libraries in the launch path instead of relying on interactive-shell state.
set -euo pipefail

export PATH="/opt/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

exec python scripts/run_neural_sequence_comparison.py "$@"
