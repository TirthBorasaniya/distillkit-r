"""Stage 1: supervised fine-tuning to seed the student on teacher traces.

A LoRA adapter is fitted on the Qwen3-1.7B student so that stage-2 on-policy
distillation starts from a checkpoint that already produces coherent reasoning
rollouts. Every hyperparameter is supplied by ``configs/sft_seed.yaml``; this
module contains no tunable numeric literals in its training path.

Heavy dependencies (torch, transformers, trl, peft, mlflow) are imported lazily
inside the functions that use them so the module can be imported for config
handling and CLI parsing without the full training stack installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import yaml

from distillkit_r.utils.checkpointing import save_and_optionally_upload
from distillkit_r.utils.logging_setup import configure_logging

if TYPE_CHECKING:
    from datasets import DatasetDict

logger = logging.getLogger(__name__)

ATTN_IMPLEMENTATION = "flash_attention_2"
LORA_TASK_TYPE = "CAUSAL_LM"
STAGE_NAME = "sft_seed"
MLFLOW_REPORT_TARGET = "mlflow"


@dataclass
class SFTSeedConfig:
    model_name: str = "Qwen/Qwen3-1.7B-Instruct"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None  # set in __post_init__
    per_device_train_batch: int = 4
    gradient_accumulation: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 4096
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    output_dir: str = "checkpoints/sft_seed"

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


def load_config(config_path: str) -> SFTSeedConfig:
    """Load ``configs/sft_seed.yaml`` and map it onto ``SFTSeedConfig``.

    YAML keys that are not declared dataclass fields (for example ``bf16`` and
    ``gradient_checkpointing``) are preserved on the returned object under the
    ``extra_args_dict`` attribute and forwarded to the trainer, so no
    hyperparameter is dropped and none is hardcoded in the training path.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    SFTSeedConfig
        Populated configuration with an attached ``extra_args_dict``.
    """
    with open(config_path, encoding="utf-8") as handle:
        raw_dict = yaml.safe_load(handle)

    field_name_set = {f.name for f in fields(SFTSeedConfig)}
    known_dict = {k: v for k, v in raw_dict.items() if k in field_name_set}
    extra_dict = {k: v for k, v in raw_dict.items() if k not in field_name_set}

    cfg = SFTSeedConfig(**known_dict)
    cfg.extra_args_dict = extra_dict
    logger.info("Loaded SFT seed config from %s", config_path)
    return cfg


def run_sft_seed(
    cfg: SFTSeedConfig,
    dataset_dict: DatasetDict,
    mlflow_run_id: str | None = None,
) -> str:
    """Run stage-1 SFT on teacher-generated traces. Returns checkpoint path.

    Parameters
    ----------
    cfg : SFTSeedConfig
        Training configuration (loaded from YAML; do not hardcode values).
    dataset_dict : DatasetDict
        Output of formatter.apply_template; must have 'prompt' and 'completion' columns.
    mlflow_run_id : str | None
        Parent MLflow run ID for nested run logging.

    Returns
    -------
    checkpoint_path : str
        Absolute path to the saved adapter checkpoint directory.
    """
    import mlflow
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    extra_args_dict = getattr(cfg, "extra_args_dict", {})

    logger.info("Loading student model %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=LORA_TASK_TYPE,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_seq_length=cfg.max_seq_length,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        report_to=[MLFLOW_REPORT_TARGET],
        **extra_args_dict,
    )

    def _formatting_func(batch: dict) -> list[str]:
        """Concatenate prompt and completion into the full SFT target text."""
        return [p + c for p, c in zip(batch["prompt"], batch["completion"])]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        formatting_func=_formatting_func,
        callbacks=[_MLflowLossCallback()],
    )

    with _nested_mlflow_run(mlflow, mlflow_run_id):
        mlflow.log_params(_loggable_params(cfg))
        logger.info("Starting SFT seed training")
        trainer.train()
        eval_metrics_dict = trainer.evaluate()
        metrics_dict = {
            "eval_loss": float(eval_metrics_dict.get("eval_loss", float("nan"))),
        }
        checkpoint_path = save_and_optionally_upload(
            peft_model=model,
            cfg_dict=_loggable_params(cfg),
            stage=STAGE_NAME,
            step=int(trainer.state.global_step),
            metrics_dict=metrics_dict,
        )

    logger.info("SFT seed complete; checkpoint at %s", checkpoint_path)
    return checkpoint_path


def _loggable_params(cfg: SFTSeedConfig) -> dict:
    """Return all config fields plus extras as a flat, MLflow-loggable dict.

    Parameters
    ----------
    cfg : SFTSeedConfig
        Configuration to flatten.

    Returns
    -------
    dict
        Field values joined with any forwarded extra arguments.
    """
    params_dict = asdict(cfg)
    params_dict.update(getattr(cfg, "extra_args_dict", {}))
    return params_dict


class _MLflowLossCallback:
    """Trainer callback that logs ``train/loss`` and ``eval/loss`` to MLflow.

    Implemented without subclassing ``transformers.TrainerCallback`` at module
    import time so the file stays importable without transformers installed; the
    Trainer only requires the ``on_log`` hook to be present.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        """Forward loss values from the trainer log dict to MLflow."""
        import mlflow

        if not logs:
            return
        step = int(state.global_step)
        if "loss" in logs:
            mlflow.log_metric("train/loss", float(logs["loss"]), step=step)
        if "eval_loss" in logs:
            mlflow.log_metric("eval/loss", float(logs["eval_loss"]), step=step)


def _nested_mlflow_run(mlflow, mlflow_run_id: str | None):
    """Return a context manager for a (possibly nested) MLflow run.

    Parameters
    ----------
    mlflow
        The imported mlflow module.
    mlflow_run_id : str | None
        Parent run id; when provided the training run is nested beneath it.

    Returns
    -------
    contextlib.AbstractContextManager
        A context manager that yields the active run.
    """
    import contextlib

    @contextlib.contextmanager
    def _runner():
        if mlflow_run_id is not None:
            mlflow.start_run(run_id=mlflow_run_id)
            try:
                with mlflow.start_run(nested=True) as run:
                    yield run
            finally:
                mlflow.end_run()
        else:
            with mlflow.start_run() as run:
                yield run

    return _runner()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: load config, build the dataset, and run SFT seeding.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    None
    """
    configure_logging()
    parser = argparse.ArgumentParser(description="Stage-1 SFT seed training")
    parser.add_argument("--config", required=True, help="Path to sft_seed.yaml")
    parser.add_argument("--mlflow-run-id", default=None, help="Parent MLflow run id")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load_config(args.config)

    from datasets import DatasetDict
    from transformers import AutoTokenizer

    from distillkit_r.data.formatter import apply_template
    from distillkit_r.data.loader import load_and_split

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset_dict = load_and_split()
    formatted_dict = DatasetDict(
        {split: apply_template(dataset_dict[split], tokenizer) for split in dataset_dict}
    )
    run_sft_seed(cfg, formatted_dict, mlflow_run_id=args.mlflow_run_id)


if __name__ == "__main__":
    main()
