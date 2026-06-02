#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m distillkit_r.training.opd_trainer \
    --config configs/gkd_opd.yaml \
    "$@"
