# distillkit-r

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![torch 2.5.1](https://img.shields.io/badge/torch-2.5.1-orange.svg)](https://pytorch.org/)
[![TRL 0.12](https://img.shields.io/badge/TRL-0.12.2-purple.svg)](https://github.com/huggingface/trl)

On-policy distillation of a small reasoning model from a large frozen teacher.
A Qwen3-1.7B student is trained by generating its own chain-of-thought rollouts,
which are then scored by a frozen Qwen3-8B teacher using per-token reverse KL supervision.
This reproduces the post-training recipe adopted in Qwen3, MiMo-V2, and Thinking Machines Lab (Oct 2025),
implemented via TRL's `GKDTrainer` on a single H200 GPU.

---

## Architecture

```
Qwen3-8B-Instruct  (teacher — frozen, float16)
        │
        │   reverse KL supervision per token  (beta=1.0)
        ▼
Qwen3-1.7B-Instruct + LoRA r64  (student — bfloat16)
        │
        │   generates own rollouts  (lmbda=1.0, on-policy)
        ▼
OpenThoughts3-1.2M (math) + DeepMath-103K  →  lighteval (MATH-500, GSM8K, AIME 2024)

Stage 1 — SFT seed   : trl.SFTTrainer on teacher traces,  3 epochs
Stage 2 — OPD        : trl.experimental.gkd.GKDTrainer,  500 steps
```

---

## Requirements

- NVIDIA GPU, ≥ 40 GB VRAM (H100/H200/A100-80G)
- CUDA 12.4
- Python 3.11
- Ubuntu 22.04+

---

## Installation

```bash
git clone https://github.com/TirthBorasaniya/distillkit-r
cd distillkit-r

bash scripts/setup_env.sh
# then edit .env — set HF_TOKEN and MLFLOW_TRACKING_URI
```

---

## Training

```bash
source .venv/bin/activate

# stage 1: SFT seed on teacher traces (~4 h on H200)
bash scripts/train_sft_seed.sh

# stage 2: on-policy distillation (~2.5 h on H200)
bash scripts/train_opd.sh
```

---

## Evaluation

```bash
bash scripts/run_eval.sh
# writes results/eval_results.json and results/compute_efficiency_curve.png
# all metrics also logged to MLflow at MLFLOW_TRACKING_URI
```

---

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "TirthBorasaniya/distillkit-r-qwen3-1.7b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("TirthBorasaniya/distillkit-r-qwen3-1.7b")

messages = [
    {"role": "system", "content": "Think step by step."},
    {"role": "user",   "content": "Solve: x^4 - 5x^2 + 4 = 0"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids  = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=512, temperature=0.6, do_sample=True)

print(tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Configuration

All hyperparameters live in `configs/`. Pass `--config` to override:

```bash
python -m distillkit_r.training.opd_trainer \
    --config configs/gkd_opd.yaml \
    --learning_rate 1e-5 \
    --max_steps 300
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `beta` | `1.0` | KL direction: `0.0` = forward, `1.0` = reverse |
| `lmbda` | `1.0` | On-policy fraction: `1.0` = fully on-policy |
| `temperature` | `0.9` | Student generation temperature |
| `max_new_tokens` | `512` | Max rollout length |
| `lora_rank` | `64` | LoRA rank for student |

---

## Design decisions

See [DECISIONS.md](DECISIONS.md) for rationale on: KL direction, on-policy fraction,
tokenizer family constraint, and ablation results.

---

## Security

Model weights are published to the Hugging Face Hub — not stored in this repository.
All credentials are loaded from environment variables; see `.env.example` for the
required variables. The hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml)
enforce a 500 KB file-size limit and scan for secrets and private keys on every
commit — run `pre-commit install` after cloning to activate them.

---

## Citation

```bibtex
@misc{distillkitr2026,
  title        = {distillkit-r: On-Policy Distillation of a Reasoning Model},
  author       = {Tirth Borasaniya},
  year         = {2026},
  howpublished = {\url{https://github.com/TirthBorasaniya/distillkit-r}},
  note         = {Implements GKD (arXiv 2306.13649) as adopted by Thinking Machines Lab, Oct 2025}
}
```
