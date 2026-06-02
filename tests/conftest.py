"""Shared pytest fixtures for the distillkit-r test suite.

Heavy runtime dependencies (torch, transformers, trl, peft) are not imported here;
fixtures provide lightweight fakes so the data-layer tests run without a GPU stack.
"""

import pytest

GEN_PROMPT_SUFFIX = "<|im_start|>assistant\n"


class FakeChatTokenizer:
    """Minimal stand-in for a Qwen3 tokenizer exposing ``apply_chat_template``.

    The template mirrors the structural contract the formatter relies on: each turn
    is wrapped in ``<|im_start|>{role}\\n{content}<|im_end|>`` and, when requested,
    a trailing assistant generation-prompt suffix is appended.
    """

    model_max_length = 4096

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render messages to a single string (text mode only)."""
        if tokenize:
            raise NotImplementedError("FakeChatTokenizer only supports tokenize=False")
        rendered = "".join(
            f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in messages
        )
        if add_generation_prompt:
            rendered += GEN_PROMPT_SUFFIX
        return rendered


@pytest.fixture
def fake_tokenizer() -> FakeChatTokenizer:
    """Return a fresh fake chat tokenizer."""
    return FakeChatTokenizer()


@pytest.fixture
def sample_messages_list() -> list[list[dict[str, str]]]:
    """Return a few message threads covering single and multi-turn cases."""
    return [
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "Step by step: 2 + 2 = 4."},
        ],
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Integrate x dx."},
            {"role": "assistant", "content": "x^2 / 2 + C."},
        ],
    ]
