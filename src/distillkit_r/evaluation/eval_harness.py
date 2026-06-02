"""Evaluate checkpoints with lighteval and build the compute-efficiency report.

``evaluate_checkpoint`` shells out to lighteval, parses its JSON results, and maps
each task to a pass@1 accuracy. ``compare_checkpoints`` evaluates several
checkpoints, prints a table, writes ``results/eval_results.json``, and renders
``results/compute_efficiency_curve.png`` (gpu_hours vs. MATH-500 accuracy).
"""

import glob
import json
import logging
import os
import subprocess

from distillkit_r.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
EVAL_RESULTS_JSON = os.path.join(RESULTS_DIR, "eval_results.json")
EFFICIENCY_CURVE_PNG = os.path.join(RESULTS_DIR, "compute_efficiency_curve.png")
LIGHTEVAL_OUTPUT_DIR = os.path.join(RESULTS_DIR, "lighteval_raw")
CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.json"

MATH500_TASK = "math::math_500"
TEACHER_LABEL = "teacher"
GPU_HOURS_METRIC = "gpu_hours_cumulative"

# Preference order when a task reports several metrics; we want pass@1 accuracy.
SCORE_METRIC_PREFERENCE = ("math_pass@1", "pass@1", "qem", "acc", "exact_match")


def evaluate_checkpoint(
    checkpoint_path: str,
    task_list: list[str],
    o_log_to_mlflow: bool = True,
    mlflow_run_id: str | None = None,
) -> dict[str, float]:
    """Run lighteval on a checkpoint and return task -> score mapping.

    Parameters
    ----------
    checkpoint_path : str
        Path to merged model directory or HF Hub repo ID.
    task_list : list[str]
        lighteval task identifiers, e.g. ["math::math_500", "math::gsm8k"].
    o_log_to_mlflow : bool
    mlflow_run_id : str | None

    Returns
    -------
    results_dict : dict[str, float]
        Maps task name to pass@1 accuracy (float in [0, 1]).
    """
    os.makedirs(LIGHTEVAL_OUTPUT_DIR, exist_ok=True)
    tasks_arg = ",".join(task_list)
    command_list = [
        "lighteval",
        "accelerate",
        f"--model_args=pretrained={checkpoint_path}",
        f"--tasks={tasks_arg}",
        f"--output_dir={LIGHTEVAL_OUTPUT_DIR}",
    ]
    logger.info("Running lighteval: %s", " ".join(command_list))
    subprocess.run(command_list, check=True)

    raw_results_dict = _load_latest_lighteval_results()
    results_dict = {task: _extract_score(task, raw_results_dict) for task in task_list}
    logger.info("Parsed scores for %s: %s", checkpoint_path, results_dict)

    if o_log_to_mlflow:
        _log_scores_to_mlflow(results_dict, mlflow_run_id)

    return results_dict


def _load_latest_lighteval_results() -> dict:
    """Load the most recent lighteval ``results_*.json`` under the output dir.

    Returns
    -------
    dict
        The parsed ``results`` mapping ``{task_name: {metric: value}}``.

    Raises
    ------
    FileNotFoundError
        If no lighteval results file is present.
    """
    pattern = os.path.join(LIGHTEVAL_OUTPUT_DIR, "**", "results*.json")
    candidate_list = glob.glob(pattern, recursive=True)
    if not candidate_list:
        logger.error("No lighteval results found under %s", LIGHTEVAL_OUTPUT_DIR)
        raise FileNotFoundError("lighteval produced no results file")

    latest_path = max(candidate_list, key=os.path.getmtime)
    with open(latest_path, encoding="utf-8") as handle:
        payload_dict = json.load(handle)
    return payload_dict.get("results", payload_dict)


def _extract_score(task: str, raw_results_dict: dict) -> float:
    """Extract a pass@1 accuracy for ``task`` from a lighteval results mapping.

    lighteval keys are not always identical to the requested task id (it may append
    a suffix), so we match by substring and then pick the most pass@1-like metric.

    Parameters
    ----------
    task : str
        Requested task identifier.
    raw_results_dict : dict
        lighteval ``results`` mapping.

    Returns
    -------
    float
        Score in ``[0, 1]``; ``0.0`` when the task or a numeric metric is missing.
    """
    leaf_name = task.split("::")[-1]
    for result_key, metric_dict in raw_results_dict.items():
        if leaf_name in result_key and isinstance(metric_dict, dict):
            for metric_name in SCORE_METRIC_PREFERENCE:
                if metric_name in metric_dict:
                    return float(metric_dict[metric_name])
            numeric_value_list = [
                float(v) for v in metric_dict.values() if isinstance(v, int | float)
            ]
            if numeric_value_list:
                return numeric_value_list[0]
    logger.warning("No score found for task %s", task)
    return 0.0


def _log_scores_to_mlflow(results_dict: dict[str, float], mlflow_run_id: str | None) -> None:
    """Log per-task scores to MLflow (best-effort; never fatal to evaluation).

    Parameters
    ----------
    results_dict : dict[str, float]
        Task -> score mapping.
    mlflow_run_id : str | None
        Run to log under; a new run is started when None.

    Returns
    -------
    None
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping eval metric logging")
        return

    with mlflow.start_run(run_id=mlflow_run_id, nested=mlflow_run_id is not None):
        for task, score in results_dict.items():
            mlflow.log_metric(f"eval/{task}", score)


def compare_checkpoints(
    checkpoint_path_list: list[str],
    label_list: list[str],
    task_list: list[str],
) -> None:
    """Evaluate multiple checkpoints, print comparison table, save results/eval_results.json,
    and save results/compute_efficiency_curve.png (X=gpu_hours from MLflow, Y=math_500 acc).
    """
    if len(checkpoint_path_list) != len(label_list):
        raise ValueError("checkpoint_path_list and label_list must be the same length")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparison_dict: dict[str, dict[str, float]] = {}
    for checkpoint_path, label in zip(checkpoint_path_list, label_list):
        logger.info("Evaluating %s (%s)", label, checkpoint_path)
        scores_dict = evaluate_checkpoint(checkpoint_path, task_list)
        scores_dict[GPU_HOURS_METRIC] = _read_gpu_hours(checkpoint_path, label)
        comparison_dict[label] = scores_dict

    _print_comparison_table(comparison_dict, task_list)

    with open(EVAL_RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(comparison_dict, handle, indent=2, sort_keys=True)
    logger.info("Wrote %s", EVAL_RESULTS_JSON)

    _plot_efficiency_curve(comparison_dict)


def _read_gpu_hours(checkpoint_path: str, label: str) -> float:
    """Read cumulative GPU-hours for a checkpoint, preferring MLflow.

    MLflow is the authoritative source (the training run logs
    ``gpu_hours_cumulative``); when it is unavailable we fall back to the
    ``checkpoint_metadata.json`` sidecar written at save time.

    Parameters
    ----------
    checkpoint_path : str
        Checkpoint directory.
    label : str
        Run name used to look the value up in MLflow.

    Returns
    -------
    float
        Cumulative GPU-hours, or ``0.0`` when unknown.
    """
    mlflow_value = _gpu_hours_from_mlflow(label)
    if mlflow_value is not None:
        return mlflow_value

    metadata_path = os.path.join(checkpoint_path, CHECKPOINT_METADATA_FILENAME)
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata_dict = json.load(handle)
        return float(metadata_dict.get("metrics", {}).get(GPU_HOURS_METRIC, 0.0))

    logger.warning("No gpu_hours found for %s; defaulting to 0.0", label)
    return 0.0


def _gpu_hours_from_mlflow(label: str) -> float | None:
    """Look up the last ``gpu_hours_cumulative`` value for a run named ``label``.

    Parameters
    ----------
    label : str
        MLflow run name.

    Returns
    -------
    float | None
        The metric value, or None when MLflow is unavailable or the run is absent.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    try:
        client = MlflowClient()
        for experiment in client.search_experiments():
            run_list = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{label}'",
            )
            if run_list:
                history = client.get_metric_history(run_list[0].info.run_id, GPU_HOURS_METRIC)
                if history:
                    return float(history[-1].value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow gpu_hours lookup failed for %s: %s", label, exc)
    return None


def _print_comparison_table(
    comparison_dict: dict[str, dict[str, float]], task_list: list[str]
) -> None:
    """Print a fixed-width comparison table to stdout.

    Parameters
    ----------
    comparison_dict : dict[str, dict[str, float]]
        Label -> {task: score, gpu_hours_cumulative: hours}.
    task_list : list[str]
        Tasks to include as columns.

    Returns
    -------
    None
    """
    header_list = ["label", *task_list, GPU_HOURS_METRIC]
    column_width = max(len(col) for col in header_list) + 2
    header_row = "".join(col.ljust(column_width) for col in header_list)
    print(header_row)
    print("-" * len(header_row))
    for label, scores_dict in comparison_dict.items():
        cell_list = [label]
        for task in task_list:
            cell_list.append(f"{scores_dict.get(task, 0.0):.4f}")
        cell_list.append(f"{scores_dict.get(GPU_HOURS_METRIC, 0.0):.4f}")
        print("".join(cell.ljust(column_width) for cell in cell_list))


def _plot_efficiency_curve(comparison_dict: dict[str, dict[str, float]]) -> None:
    """Render the compute-efficiency curve to ``results/compute_efficiency_curve.png``.

    X is cumulative GPU-hours and Y is MATH-500 pass@1. A ``teacher`` label, when
    present, is drawn as a dashed horizontal reference line and excluded from the
    scatter of student checkpoints.

    Parameters
    ----------
    comparison_dict : dict[str, dict[str, float]]
        Label -> {task: score, gpu_hours_cumulative: hours}.

    Returns
    -------
    None
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping efficiency curve plot")
        return

    figure, axis = plt.subplots()
    for label, scores_dict in comparison_dict.items():
        math_acc = scores_dict.get(MATH500_TASK, 0.0)
        if label == TEACHER_LABEL:
            axis.axhline(math_acc, linestyle="--", color="gray", label="teacher (MATH-500)")
            continue
        gpu_hours = scores_dict.get(GPU_HOURS_METRIC, 0.0)
        axis.scatter(gpu_hours, math_acc)
        axis.annotate(label, (gpu_hours, math_acc))

    axis.set_xlabel("Cumulative GPU-hours")
    axis.set_ylabel("MATH-500 pass@1")
    axis.set_title("Compute-efficiency curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(EFFICIENCY_CURVE_PNG)
    plt.close(figure)
    logger.info("Wrote %s", EFFICIENCY_CURVE_PNG)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point mirroring ``scripts/run_eval.sh``.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    None
    """
    import argparse
    import sys

    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate and compare checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", default=EVAL_RESULTS_JSON)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    compare_checkpoints(args.checkpoints, args.labels, args.tasks)


if __name__ == "__main__":
    main()
