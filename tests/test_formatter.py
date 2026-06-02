"""Tests for ``distillkit_r.data.formatter.apply_template``.

Uses the ``FakeChatTokenizer`` fixture so behavior is checked without loading a
real Qwen3 tokenizer.
"""

import pytest

datasets = pytest.importorskip("datasets")

from distillkit_r.data import formatter  # noqa: E402
from tests.conftest import GEN_PROMPT_SUFFIX  # noqa: E402


def _dataset_from(messages_list: list[list[dict[str, str]]]) -> "datasets.Dataset":
    """Wrap raw message threads in a Dataset with a ``messages`` column."""
    return datasets.Dataset.from_list([{"messages": m} for m in messages_list])


def test_adds_prompt_and_completion_columns(fake_tokenizer, sample_messages_list):
    """Both new columns exist and every row is non-empty."""
    dataset = _dataset_from(sample_messages_list)
    formatted = formatter.apply_template(dataset, fake_tokenizer)

    assert "prompt" in formatted.column_names
    assert "completion" in formatted.column_names
    for row in formatted:
        assert row["prompt"].strip()
        assert row["completion"].strip()


def test_completion_is_last_assistant_turn(fake_tokenizer, sample_messages_list):
    """Completion equals the content of the final assistant turn."""
    dataset = _dataset_from(sample_messages_list)
    formatted = formatter.apply_template(dataset, fake_tokenizer)

    assert formatted[0]["completion"] == "Step by step: 2 + 2 = 4."
    assert formatted[1]["completion"] == "x^2 / 2 + C."
    # The completion text must not leak into the prompt.
    assert "2 + 2 = 4" not in formatted[0]["prompt"]


def test_generation_prompt_suffix_toggle(fake_tokenizer, sample_messages_list):
    """Prompt ends with the generation suffix only when the flag is set."""
    dataset = _dataset_from(sample_messages_list)

    with_suffix = formatter.apply_template(dataset, fake_tokenizer, o_add_generation_prompt=True)
    without_suffix = formatter.apply_template(
        dataset, fake_tokenizer, o_add_generation_prompt=False
    )

    for row in with_suffix:
        assert row["prompt"].endswith(GEN_PROMPT_SUFFIX)
    for row in without_suffix:
        assert not row["prompt"].endswith(GEN_PROMPT_SUFFIX)


def test_default_system_prompt_injected(fake_tokenizer):
    """A thread without a system turn gets the default system prompt prepended."""
    dataset = _dataset_from(
        [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]]
    )
    formatted = formatter.apply_template(dataset, fake_tokenizer)
    assert formatter.SYSTEM_PROMPT in formatted[0]["prompt"]


def test_no_truncation_error_on_max_length_input(fake_tokenizer):
    """A very long input is templated without raising (text mode does not truncate)."""
    long_content = "x = 1; " * formatter.MAX_SEQ_LEN
    dataset = _dataset_from(
        [
            [
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": long_content},
            ]
        ]
    )
    formatted = formatter.apply_template(dataset, fake_tokenizer)
    assert formatted[0]["prompt"]
    assert formatted[0]["completion"]
