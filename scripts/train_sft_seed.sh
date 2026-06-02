#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m distillkit_r.training.sft_seed \
    --config configs/sft_seed.yaml \
    "$@"
