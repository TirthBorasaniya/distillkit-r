"""distillkit-r: on-policy distillation of a small reasoning model.

Two-stage pipeline: stage 1 SFT seeding on teacher traces, stage 2 on-policy
distillation with TRL GKDTrainer under reverse KL supervision.
"""

__version__ = "0.1.0"
