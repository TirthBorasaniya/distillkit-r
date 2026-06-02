"""Apply the Qwen3 chat template and derive the GKD ``prompt``/``completion`` columns.

GKDTrainer reads ``prompt`` (everything up to the model's turn) for on-policy
rollout generation and ``completion`` (the reference assistant turn) for the
off-policy target. Both columns are produced here in a single ``map`` pass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from datasets import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step before giving your final answer."
)
MAX_SEQ_LEN = 4096

logger = logging.getLogger(__name__)


def _last_assistant_index(messages_list: list[dict[str, str]]) -> int:
    """Return the index of the final assistant turn.

    Parameters
    ----------
    messages_list : list[dict[str, str]]
        Conversation turns with ``role`` and ``content`` keys.

    Returns
    -------
    int
        Index of the last turn whose role is ``assistant``.

    Raises
    ------
    ValueError
        If the thread contains no assistant turn.
    """
    for idx in range(len(messages_list) - 1, -1, -1):
        if messages_list[idx]["role"] == "assistant":
            return idx
    raise ValueError("messages thread contains no assistant turn")


def _ensure_system_prompt(messages_list: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prepend the default system prompt when the thread lacks one.

    Parameters
    ----------
    messages_list : list[dict[str, str]]
        Conversation turns.

    Returns
    -------
    list[dict[str, str]]
        Turns guaranteed to begin with a system turn.
    """
    if messages_list and messages_list[0]["role"] == "system":
        return messages_list
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages_list]


def apply_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    o_add_generation_prompt: bool = False,
) -> Dataset:
    """Apply Qwen3 chat template; add 'prompt' and 'completion' columns.

    Parameters
    ----------
    dataset : Dataset
        Must have 'messages' column of type list[dict[str, str]].
    tokenizer : PreTrainedTokenizer
        Qwen3 tokenizer with chat_template set.
    o_add_generation_prompt : bool
        Whether to append the model's generation prompt suffix.

    Returns
    -------
    formatted : Dataset
        Original columns plus 'prompt' (str) and 'completion' (str).
    """

    def _format_row(example: dict) -> dict:
        messages_list = _ensure_system_prompt(example["messages"])
        assistant_idx = _last_assistant_index(messages_list)

        prompt_messages = messages_list[:assistant_idx]
        completion = messages_list[assistant_idx]["content"]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=o_add_generation_prompt,
        )
        return {"prompt": prompt, "completion": completion}

    logger.info("Applying chat template to %d rows", dataset.num_rows)
    formatted = dataset.map(_format_row, num_proc=4)
    return formatted
