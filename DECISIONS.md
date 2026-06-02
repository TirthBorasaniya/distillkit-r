# DECISIONS.md — distillkit-r

Rationale for the key modeling and training choices in this project. Ablation
sections document the experimental protocol; the result tables are populated from
the validation-set runs produced by `scripts/run_eval.sh` and the MLflow metrics
logged during training. Cells marked `TBD` are filled in once the corresponding
sweep has been executed on the target hardware.

---

## Why reverse KL over forward KL

The objective interpolates between forward KL (`beta=0.0`) and reverse KL
(`beta=1.0`); on-policy distillation uses reverse KL.

- **Forward KL**, `KL(teacher || student)`, is mass-covering. It pushes the student
  to put probability everywhere the teacher does, including low-density tails. For a
  small student with limited capacity this spreads probability too thinly and yields
  hedged, less decisive reasoning traces.
- **Reverse KL**, `KL(student || teacher)`, is mode-seeking. It penalizes the student
  for placing mass where the teacher assigns low probability, so the student commits
  to the teacher's high-probability reasoning modes. This is the behavior we want for
  a compact reasoning model: sharp, confident trajectories rather than diffuse ones.
- Reverse KL is also the natural fit for on-policy training. Because it is evaluated
  under the student's own sampling distribution, it directly corrects the rollouts the
  student actually produces, which is what reduces exposure bias (see below).

The reward we log (`reward_mean`) is the negative reverse KL between the student and
teacher next-token distributions; it is zero exactly when the two distributions match
and is implemented as a pure, unit-tested function in `opd_trainer.reverse_kl`.

## Why `lmbda=1.0`

`lmbda` is the fraction of supervised tokens drawn from student-generated rollouts.

- `lmbda=0.0` is fully off-policy: the student is supervised only on the teacher's
  pre-recorded traces, identical in spirit to the stage-1 SFT seed.
- `lmbda=1.0` is fully on-policy: every supervised token comes from the student's own
  rollouts, scored against the teacher.

We choose `lmbda=1.0` because the central failure mode of off-policy SFT is
**exposure bias**: the student is only ever trained on the teacher's trajectories, so
at inference time it has never learned to recover from its own mistakes once it drifts
off the teacher's distribution. Training on student rollouts closes this train/inference
gap, the student is corrected precisely on the states it actually visits. Stage 1 (SFT
seed) already provides the off-policy grounding, so stage 2 is free to be fully
on-policy.

## Why the Qwen3 family (tokenizer constraint)

TRL's GKD implementation requires the teacher and student to share a tokenizer;
cross-tokenizer distillation is broken (TRL issue #4562) because the per-token KL is
only well defined when both models score the *same* token sequence. We therefore pair
a **Qwen3-8B-Instruct** teacher with a **Qwen3-1.7B-Instruct** student: same tokenizer
family, same chat template, large enough teacher/student capacity gap to make
distillation worthwhile. A Qwen teacher with a LLaMA student would silently misalign
token ids and is explicitly out of scope.

---

## Ablation: `beta` ∈ {0.0, 1.0} on the validation set

**Protocol.** Hold every other hyperparameter at the `configs/gkd_opd.yaml` values
(`lmbda=1.0`, `temperature=0.9`, `max_steps=500`). Run two stage-2 jobs differing only
in `beta`, starting from the same SFT seed checkpoint. Evaluate each with
`scripts/run_eval.sh` on MATH-500 and GSM8K, and read `gpu_hours_cumulative` from
MLflow.

| beta | KL mode      | MATH-500 pass@1 | GSM8K pass@1 | val reverse-KL | gpu_hours |
|------|--------------|-----------------|--------------|----------------|-----------|
| 0.0  | forward KL   | TBD             | TBD          | TBD            | TBD       |
| 1.0  | reverse KL   | TBD             | TBD          | TBD            | TBD       |

**Expectation.** `beta=1.0` (reverse KL) should match or exceed `beta=0.0` on MATH-500
pass@1 while producing lower-entropy, more decisive traces. If forward KL wins on raw
accuracy but at the cost of much longer, hedged generations, reverse KL remains
preferred for the compact-reasoning objective.

## Ablation: `lmbda` ∈ {0.0, 0.5, 1.0} on the validation set

**Protocol.** Hold `beta=1.0` and all other `configs/gkd_opd.yaml` values fixed. Run
three stage-2 jobs differing only in `lmbda`, each from the same SFT seed. Evaluate as
above.

| lmbda | rollout source        | MATH-500 pass@1 | GSM8K pass@1 | gpu_hours |
|-------|-----------------------|-----------------|--------------|-----------|
| 0.0   | teacher traces only   | TBD             | TBD          | TBD       |
| 0.5   | mixed                 | TBD             | TBD          | TBD       |
| 1.0   | student rollouts only | TBD             | TBD          | TBD       |

**Expectation.** Accuracy should increase monotonically with `lmbda` as exposure bias
is reduced, with `lmbda=1.0` giving the best pass@1. The trade-off is wall-clock cost:
higher `lmbda` means more student generation per step, so `gpu_hours` rises with
`lmbda`. The compute-efficiency curve (`results/compute_efficiency_curve.png`) plots
this trade-off directly (MATH-500 pass@1 vs. cumulative GPU-hours), with the teacher
drawn as a dashed horizontal reference line.
