"""Stage 2: on-policy distillation with TRL ``GKDTrainer`` under reverse KL.

The student (loaded from the stage-1 LoRA checkpoint) generates rollouts that are
scored against a frozen Qwen3-8B teacher. With ``lmbda=1.0`` every supervised token
comes from the student's own rollouts (fully on-policy) and ``beta=1.0`` selects the
reverse KL objective. All hyperparameters come from ``configs/gkd_opd.yaml``.

Heavy dependencies (torch, transformers, trl, peft, mlflow) are imported lazily so
this module is importable without the training stack; the pure reward helpers below
have no such dependency and are unit-tested directly.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import yaml

from distillkit_r.utils.checkpointing import save_and_optionally_upload
from distillkit_r.utils.logging_setup import configure_logging

if TYPE_CHECKING:
    from datasets import DatasetDict

logger = logging.getLogger(__name__)

ATTN_IMPLEMENTATION = "flash_attention_2"
STAGE_NAME = "opd_final"
MLFLOW_REPORT_TARGET = "mlflow"
SECONDS_PER_HOUR = 3600.0


@dataclass
class OPDConfig:
    teacher_model_name: str = "Qwen/Qwen3-8B-Instruct"
    sft_checkpoint_path: str = "checkpoints/sft_seed"
    lmbda: float = 1.0
    beta: float = 1.0
    temperature: float = 0.9
    max_new_tokens: int = 512
    per_device_train_batch: int = 2
    gradient_accumulation: int = 4
    max_steps: int = 500
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    save_steps: int = 100
    logging_steps: int = 25
    output_dir: str = "checkpoints/opd_final"


def reverse_kl(student_logprobs: list[float], teacher_logprobs: list[float]) -> float:
    """Compute the reverse KL divergence ``KL(student || teacher)`` for one position.

    Reverse KL weights the divergence by the *student's* probability mass, which is
    the supervision signal on-policy distillation optimizes: it penalizes student
    mass placed where the teacher assigns low probability.

    Parameters
    ----------
    student_logprobs : list[float]
        Log-probabilities of the student distribution over the vocabulary.
    teacher_logprobs : list[float]
        Log-probabilities of the teacher distribution over the same support.

    Returns
    -------
    float
        ``sum_x p_student(x) * (log p_student(x) - log p_teacher(x))`` in nats.

    Raises
    ------
    ValueError
        If the two distributions do not share the same support length.
    """
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError("student and teacher distributions must share support length")

    divergence = 0.0
    for student_lp, teacher_lp in zip(student_logprobs, teacher_logprobs):
        prob = math.exp(student_lp)
        divergence += prob * (student_lp - teacher_lp)
    return divergence


def reverse_kl_reward(student_logprobs: list[float], teacher_logprobs: list[float]) -> float:
    """Return the per-position reward used for logging: negative reverse KL.

    Higher reward means the student distribution is closer to the teacher; the
    reward is exactly zero when the two distributions are identical.

    Parameters
    ----------
    student_logprobs : list[float]
        Student log-probabilities.
    teacher_logprobs : list[float]
        Teacher log-probabilities.

    Returns
    -------
    float
        ``-KL(student || teacher)``.
    """
    return -reverse_kl(student_logprobs, teacher_logprobs)


def load_config(config_path: str) -> OPDConfig:
    """Load ``configs/gkd_opd.yaml`` and map it onto ``OPDConfig``.

    YAML keys not declared on ``OPDConfig`` (for example ``bf16``) are preserved on
    the returned object under ``extra_args_dict`` and forwarded to ``GKDConfig``.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    OPDConfig
        Populated configuration with an attached ``extra_args_dict``.
    """
    with open(config_path, encoding="utf-8") as handle:
        raw_dict = yaml.safe_load(handle)

    field_name_set = {f.name for f in fields(OPDConfig)}
    known_dict = {k: v for k, v in raw_dict.items() if k in field_name_set}
    extra_dict = {k: v for k, v in raw_dict.items() if k not in field_name_set}

    cfg = OPDConfig(**known_dict)
    cfg.extra_args_dict = extra_dict
    logger.info("Loaded OPD config from %s", config_path)
    return cfg


def run_opd(
    cfg: OPDConfig,
    dataset_dict: DatasetDict,
    mlflow_run_id: str | None = None,
) -> str:
    """Run stage-2 on-policy distillation using TRL GKDTrainer. Returns checkpoint path.

    Parameters
    ----------
    cfg : OPDConfig
    dataset_dict : DatasetDict
        Same dataset as SFT seed stage; GKDTrainer uses 'prompt' for rollout generation.
    mlflow_run_id : str | None

    Returns
    -------
    checkpoint_path : str

    Notes
    -----
    Teacher is loaded in torch.float16 with device_map="cuda"; frozen
    (requires_grad=False on all params). Student loads from sft_checkpoint_path as a
    PeftModel; trained in bfloat16. Logs gpu_hours (computed from time.perf_counter
    deltas) to MLflow as a metric.
    """
    import mlflow
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    extra_args_dict = getattr(cfg, "extra_args_dict", {})

    logger.info("Loading frozen teacher %s", cfg.teacher_model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    logger.info("Loading student from SFT checkpoint %s", cfg.sft_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint_path)
    student = AutoPeftModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    gkd_config = GKDConfig(
        output_dir=cfg.output_dir,
        teacher_model_name_or_path=cfg.teacher_model_name,
        lmbda=cfg.lmbda,
        beta=cfg.beta,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        per_device_train_batch_size=cfg.per_device_train_batch,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        report_to=[MLFLOW_REPORT_TARGET],
        **extra_args_dict,
    )

    metrics_callback = _OPDMetricsCallback(device_count=torch.cuda.device_count())

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=gkd_config,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        callbacks=[metrics_callback],
    )

    with _nested_mlflow_run(mlflow, mlflow_run_id):
        mlflow.log_params(_loggable_params(cfg))
        logger.info("Starting OPD training (lmbda=%s, beta=%s)", cfg.lmbda, cfg.beta)
        trainer.train()
        metrics_dict = {
            "gpu_hours_cumulative": metrics_callback.gpu_hours_cumulative,
        }
        checkpoint_path = save_and_optionally_upload(
            peft_model=student,
            cfg_dict=_loggable_params(cfg),
            stage=STAGE_NAME,
            step=int(trainer.state.global_step),
            metrics_dict=metrics_dict,
        )

    logger.info("OPD complete; checkpoint at %s", checkpoint_path)
    return checkpoint_path


def _loggable_params(cfg: OPDConfig) -> dict:
    """Flatten config fields plus forwarded extras into an MLflow-loggable dict.

    Parameters
    ----------
    cfg : OPDConfig
        Configuration to flatten.

    Returns
    -------
    dict
        Field values joined with any forwarded extra arguments.
    """
    params_dict = asdict(cfg)
    params_dict.update(getattr(cfg, "extra_args_dict", {}))
    return params_dict


class _OPDMetricsCallback:
    """Trainer callback logging KL, reward, and cumulative GPU-hours to MLflow.

    GPU-hours accumulate wall-clock time between log events scaled by the visible
    device count. The callback avoids importing transformers at module import time
    so the file stays importable without the training stack.
    """

    def __init__(self, device_count: int) -> None:
        self.device_count = max(device_count, 1)
        self.gpu_hours_cumulative = 0.0
        self._last_perf_counter: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        """Start the wall-clock accumulator."""
        self._last_perf_counter = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        """Accumulate GPU-hours and forward KL / reward metrics to MLflow."""
        import mlflow

        now = time.perf_counter()
        if self._last_perf_counter is not None:
            elapsed_hours = (now - self._last_perf_counter) / SECONDS_PER_HOUR
            self.gpu_hours_cumulative += elapsed_hours * self.device_count
        self._last_perf_counter = now

        step = int(state.global_step)
        mlflow.log_metric("gpu_hours_cumulative", self.gpu_hours_cumulative, step=step)

        if not logs:
            return
        for log_key, metric_name in (("kl", "kl_div"), ("reward", "reward_mean")):
            if log_key in logs:
                mlflow.log_metric(metric_name, float(logs[log_key]), step=step)


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
    """CLI entry point: load config, build the dataset, and run OPD.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    None
    """
    configure_logging()
    parser = argparse.ArgumentParser(description="Stage-2 on-policy distillation")
    parser.add_argument("--config", required=True, help="Path to gkd_opd.yaml")
    parser.add_argument("--mlflow-run-id", default=None, help="Parent MLflow run id")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load_config(args.config)

    from datasets import DatasetDict
    from transformers import AutoTokenizer

    from distillkit_r.data.formatter import apply_template
    from distillkit_r.data.loader import load_and_split

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint_path)
    dataset_dict = load_and_split()
    formatted_dict = DatasetDict(
        {
            split: apply_template(dataset_dict[split], tokenizer, o_add_generation_prompt=True)
            for split in dataset_dict
        }
    )
    run_opd(cfg, formatted_dict, mlflow_run_id=args.mlflow_run_id)


if __name__ == "__main__":
    main()
