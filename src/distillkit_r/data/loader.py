"""Load and merge the two source corpora into a single train/validation split.

OpenThoughts3 (math subset) supplies multi-domain reasoning traces; DeepMath-103K
supplies math-only problems with reference solutions. Both are normalized to the
shared ``{"messages", "source", "subject"}`` schema, concatenated, and split 95/5.
"""

import logging
import os

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

OPENTHOUGHTS_ID = "open-thoughts/OpenThoughts3-1.2M"
DEEPMATH_ID = "zwhe99/DeepMath-103K"
MATH_FILTER = {"subject": "math"}
SPLIT_SEED = 42
VAL_FRACTION = 0.05

UNIFIED_COLUMNS = ["messages", "source", "subject"]

logger = logging.getLogger(__name__)


def _normalize_openthoughts(example: dict) -> dict:
    """Map one OpenThoughts3 row to the unified schema.

    The corpus stores turns under either ``messages`` or ``conversations`` and the
    domain under ``domain`` or ``subject``. We coerce both into the shared schema
    without assuming a single fixed layout.

    Parameters
    ----------
    example : dict
        A single raw row.

    Returns
    -------
    dict
        Row with ``messages``, ``source``, and ``subject`` keys.
    """
    raw_turns = example.get("messages") or example.get("conversations") or []
    messages_list = [_coerce_turn(turn) for turn in raw_turns]
    subject = example.get("subject") or example.get("domain") or "unknown"
    return {"messages": messages_list, "source": "openthoughts3", "subject": subject}


def _normalize_deepmath(example: dict) -> dict:
    """Map one DeepMath-103K row to the unified schema.

    DeepMath rows carry a ``question`` plus one or more reference solutions; we use
    the first available solution as the assistant turn, falling back to the final
    answer when no full solution is present. Every DeepMath row is math.

    Parameters
    ----------
    example : dict
        A single raw row.

    Returns
    -------
    dict
        Row with ``messages``, ``source``, and ``subject`` keys.
    """
    question = example.get("question") or example.get("problem") or ""
    solution = (
        example.get("r1_solution_1") or example.get("solution") or example.get("final_answer") or ""
    )
    messages_list = [
        {"role": "user", "content": str(question)},
        {"role": "assistant", "content": str(solution)},
    ]
    return {"messages": messages_list, "source": "deepmath", "subject": "math"}


def _coerce_turn(turn: dict) -> dict[str, str]:
    """Coerce a single conversation turn into ``{"role", "content"}``.

    Handles both the OpenAI-style ``{"role", "content"}`` layout and the ShareGPT
    ``{"from", "value"}`` layout used by some OpenThoughts exports.

    Parameters
    ----------
    turn : dict
        A single turn in either supported layout.

    Returns
    -------
    dict[str, str]
        Turn with ``role`` and ``content`` string keys.
    """
    role = turn.get("role") or turn.get("from") or "user"
    content = turn.get("content") or turn.get("value") or ""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return {"role": role_map.get(role, role), "content": str(content)}


def _project_to_unified(dataset: Dataset, normalizer) -> Dataset:
    """Apply a normalizer and drop every column not in the unified schema.

    Parameters
    ----------
    dataset : Dataset
        Raw source dataset.
    normalizer : callable
        Per-row mapping function returning the unified schema.

    Returns
    -------
    Dataset
        Dataset whose columns are exactly ``UNIFIED_COLUMNS``.
    """
    columns_to_remove = list(dataset.column_names)
    return dataset.map(normalizer, remove_columns=columns_to_remove, num_proc=4)


def load_and_split(
    o_run_from_scratch: bool = False,
    cache_path: str = "data/dataset_cache",
) -> DatasetDict:
    """Load and merge OpenThoughts3 (math subset) and DeepMath-103K, apply 95/5 split.

    Parameters
    ----------
    o_run_from_scratch : bool
        If True, bypass disk cache and re-download.
    cache_path : str
        Directory for saving the merged DatasetDict.

    Returns
    -------
    dataset_dict : DatasetDict
        Keys 'train' and 'validation'; columns: 'messages', 'source', 'subject'.
    """
    if os.path.isdir(cache_path) and not o_run_from_scratch:
        logger.info("Loading cached DatasetDict from %s", cache_path)
        return load_from_disk(cache_path)

    logger.info("Downloading %s", OPENTHOUGHTS_ID)
    openthoughts_raw = load_dataset(OPENTHOUGHTS_ID, split="train")
    openthoughts = _project_to_unified(openthoughts_raw, _normalize_openthoughts)

    subject_key = next(iter(MATH_FILTER))
    target_subject = MATH_FILTER[subject_key]
    openthoughts = openthoughts.filter(lambda row: row[subject_key] == target_subject, num_proc=4)
    logger.info("OpenThoughts3 math subset: %d rows", openthoughts.num_rows)

    logger.info("Downloading %s", DEEPMATH_ID)
    deepmath_raw = load_dataset(DEEPMATH_ID, split="train")
    deepmath = _project_to_unified(deepmath_raw, _normalize_deepmath)
    logger.info("DeepMath-103K: %d rows", deepmath.num_rows)

    merged = concatenate_datasets([openthoughts, deepmath])
    split = merged.train_test_split(test_size=VAL_FRACTION, seed=SPLIT_SEED)
    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})

    _log_split_summary(dataset_dict)

    logger.info("Saving merged DatasetDict to %s", cache_path)
    dataset_dict.save_to_disk(cache_path)
    return dataset_dict


def _log_split_summary(dataset_dict: DatasetDict) -> None:
    """Log train/validation sizes and per-source counts.

    Parameters
    ----------
    dataset_dict : DatasetDict
        The split to summarize.

    Returns
    -------
    None
    """
    train_n = dataset_dict["train"].num_rows
    val_n = dataset_dict["validation"].num_rows
    logger.info("Split sizes: train=%d, validation=%d", train_n, val_n)

    source_count_dict: dict[str, int] = {}
    for source in dataset_dict["train"]["source"]:
        source_count_dict[source] = source_count_dict.get(source, 0) + 1
    logger.info("Train source distribution: %s", source_count_dict)
