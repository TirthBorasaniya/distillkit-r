"""Tests for ``distillkit_r.data.loader.load_and_split``.

HF downloads are mocked: ``load_dataset`` is patched to return small in-memory
datasets so we exercise filtering, concatenation, the split ratio, and caching
without touching the network.
"""

from unittest.mock import patch

import pytest

datasets = pytest.importorskip("datasets")

from distillkit_r.data import loader  # noqa: E402


def _fake_openthoughts() -> "datasets.Dataset":
    """Build a fake OpenThoughts3 dataset: 90 math rows + 10 non-math rows."""
    rows = []
    for i in range(90):
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": f"math q{i}"},
                    {"from": "gpt", "value": f"math a{i}"},
                ],
                "domain": "math",
            }
        )
    for i in range(10):
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": f"sci q{i}"},
                    {"from": "gpt", "value": f"sci a{i}"},
                ],
                "domain": "science",
            }
        )
    return datasets.Dataset.from_list(rows)


def _fake_deepmath() -> "datasets.Dataset":
    """Build a fake DeepMath-103K dataset: 10 math rows."""
    rows = [{"question": f"deep q{i}", "r1_solution_1": f"deep a{i}"} for i in range(10)]
    return datasets.Dataset.from_list(rows)


def _fake_load_dataset(dataset_id: str, split: str = "train"):
    """Route a dataset id to the matching fake builder."""
    if dataset_id == loader.OPENTHOUGHTS_ID:
        return _fake_openthoughts()
    if dataset_id == loader.DEEPMATH_ID:
        return _fake_deepmath()
    raise AssertionError(f"unexpected dataset id: {dataset_id}")


def test_columns_and_schema(tmp_path):
    """Output columns are exactly the unified schema, with coerced turns."""
    cache = str(tmp_path / "cache")
    with patch.object(loader, "load_dataset", side_effect=_fake_load_dataset):
        dataset_dict = loader.load_and_split(o_run_from_scratch=True, cache_path=cache)

    for split in ("train", "validation"):
        assert set(dataset_dict[split].column_names) == {"messages", "source", "subject"}

    first = dataset_dict["train"][0]
    assert {turn["role"] for turn in first["messages"]} <= {"user", "assistant", "system"}
    assert first["source"] in {"openthoughts3", "deepmath"}


def test_non_math_openthoughts_filtered_out(tmp_path):
    """Science rows from OpenThoughts3 are dropped before concatenation."""
    cache = str(tmp_path / "cache")
    with patch.object(loader, "load_dataset", side_effect=_fake_load_dataset):
        dataset_dict = loader.load_and_split(o_run_from_scratch=True, cache_path=cache)

    all_subjects = dataset_dict["train"]["subject"] + dataset_dict["validation"]["subject"]
    assert set(all_subjects) == {"math"}
    # 90 math (openthoughts) + 10 (deepmath) = 100, science dropped.
    total = dataset_dict["train"].num_rows + dataset_dict["validation"].num_rows
    assert total == 100


def test_split_ratio_within_one_percent(tmp_path):
    """Train/validation split is within 1 percent of 95/5."""
    cache = str(tmp_path / "cache")
    with patch.object(loader, "load_dataset", side_effect=_fake_load_dataset):
        dataset_dict = loader.load_and_split(o_run_from_scratch=True, cache_path=cache)

    train_n = dataset_dict["train"].num_rows
    val_n = dataset_dict["validation"].num_rows
    total = train_n + val_n
    assert abs(train_n / total - 0.95) < 0.01
    assert abs(val_n / total - 0.05) < 0.01


def test_cache_is_reused(tmp_path):
    """When the cache exists and ``o_run_from_scratch`` is False, no re-download."""
    cache = str(tmp_path / "cache")
    with patch.object(loader, "load_dataset", side_effect=_fake_load_dataset):
        loader.load_and_split(o_run_from_scratch=True, cache_path=cache)

    # Second call must not invoke load_dataset at all.
    with patch.object(loader, "load_dataset", side_effect=AssertionError("no download")) as mocked:
        dataset_dict = loader.load_and_split(o_run_from_scratch=False, cache_path=cache)
        mocked.assert_not_called()

    total = dataset_dict["train"].num_rows + dataset_dict["validation"].num_rows
    assert total == 100
