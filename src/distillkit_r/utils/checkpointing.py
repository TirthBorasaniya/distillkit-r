"""Persist LoRA adapters (or merged models) and optionally publish to the HF Hub.

A PEFT ``save_pretrained`` writes only the adapter weights. When a standalone model
is needed (for example for lighteval), the LoRA is merged into the base weights via
``merge_and_unload`` before saving. The HF token is read from the environment and is
never logged.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = "checkpoints"
METADATA_FILENAME = "checkpoint_metadata.json"


def save_and_optionally_upload(
    peft_model,
    cfg_dict: dict[str, Any],
    stage: str,
    step: int,
    metrics_dict: dict[str, float],
    o_merge_before_save: bool = False,
    o_upload_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> str:
    """Save adapter (or merged model) and optionally push to the HF Hub.

    Saves to ``checkpoints/<stage>/step_<step>/``. When ``o_merge_before_save`` is
    set, the LoRA adapter is merged into the base weights first so the directory
    holds a standalone model rather than an adapter.

    Parameters
    ----------
    peft_model
        A ``peft.PeftModel`` wrapping the trained student.
    cfg_dict : dict[str, Any]
        Full training configuration, persisted alongside the weights.
    stage : str
        One of ``"sft_seed"`` or ``"opd_final"``.
    step : int
        Global training step at which the checkpoint is taken.
    metrics_dict : dict[str, float]
        Metrics to record in the checkpoint metadata.
    o_merge_before_save : bool
        If True, merge LoRA weights into the base model before saving.
    o_upload_to_hub : bool
        If True, push the saved artifact to the HF Hub.
    hub_repo_id : str | None
        Destination repo id; required when ``o_upload_to_hub`` is True.

    Returns
    -------
    save_path : str
        Absolute path to the saved checkpoint directory.
    """
    save_path = os.path.abspath(os.path.join(CHECKPOINT_ROOT, stage, f"step_{step}"))
    os.makedirs(save_path, exist_ok=True)

    if o_merge_before_save:
        logger.info("Merging LoRA adapter into base weights before save")
        model_to_save = peft_model.merge_and_unload()
    else:
        model_to_save = peft_model

    logger.info("Saving %s checkpoint to %s", stage, save_path)
    model_to_save.save_pretrained(save_path)

    _write_metadata(save_path, cfg_dict, stage, step, metrics_dict)

    if o_upload_to_hub:
        _upload_to_hub(model_to_save, save_path, hub_repo_id)

    return save_path


def _write_metadata(
    save_path: str,
    cfg_dict: dict[str, Any],
    stage: str,
    step: int,
    metrics_dict: dict[str, float],
) -> None:
    """Write a JSON sidecar describing the checkpoint.

    Parameters
    ----------
    save_path : str
        Directory the checkpoint was saved to.
    cfg_dict : dict[str, Any]
        Training configuration.
    stage : str
        Training stage label.
    step : int
        Global training step.
    metrics_dict : dict[str, float]
        Recorded metrics.

    Returns
    -------
    None
    """
    metadata_dict = {
        "stage": stage,
        "step": step,
        "metrics": metrics_dict,
        "config": cfg_dict,
    }
    metadata_path = os.path.join(save_path, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata_dict, handle, indent=2, sort_keys=True)
    logger.info("Wrote checkpoint metadata to %s", metadata_path)


def _upload_to_hub(model_to_save, save_path: str, hub_repo_id: str | None) -> None:
    """Push a saved checkpoint to the HF Hub using the environment token.

    Parameters
    ----------
    model_to_save
        The model (adapter or merged) exposing ``push_to_hub``.
    save_path : str
        Local checkpoint directory (used only for logging).
    hub_repo_id : str | None
        Destination repo id.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``hub_repo_id`` is missing or ``HF_TOKEN`` is not set.
    """
    if not hub_repo_id:
        logger.error("o_upload_to_hub is set but hub_repo_id is None")
        raise ValueError("hub_repo_id is required when o_upload_to_hub is True")

    if "HF_TOKEN" not in os.environ:
        logger.error("o_upload_to_hub is set but HF_TOKEN is not in the environment")
        raise ValueError("HF_TOKEN must be set in the environment to upload")

    logger.info("Uploading checkpoint at %s to hub repo %s", save_path, hub_repo_id)
    model_to_save.push_to_hub(hub_repo_id, token=os.environ["HF_TOKEN"], private=True)
    logger.info("Upload to %s complete", hub_repo_id)
