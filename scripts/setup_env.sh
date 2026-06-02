#!/usr/bin/env bash
set -euo pipefail
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.3.post1 --no-build-isolation
pip install -e ".[train,eval,dev]"
cp .env.example .env
pip install pre-commit
pre-commit install
detect-secrets scan > .secrets.baseline
echo "Setup complete. Edit .env before running training."
