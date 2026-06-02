#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m distillkit_r.evaluation.eval_harness \
    --checkpoints checkpoints/sft_seed checkpoints/opd_final \
    --labels "sft_seed" "opd_final" \
    --tasks math::math_500 math::gsm8k \
    --output results/eval_results.json
