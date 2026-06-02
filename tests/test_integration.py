"""Integration tests for checkpoint production and the config loaders.

Two layers:

* Always-on: a fake PEFT model exercises ``save_and_optionally_upload`` and asserts
  the saved directory carries ``adapter_config.json`` and ``adapter_model.safetensors``
  (and the merged-model path), plus the YAML config loaders round-trip correctly.
* Opt-in: the real 2-step ``run_sft_seed`` smoke test runs only when the GPU training
  stack is installed and ``RUN_HEAVY_INTEGRATION=1`` is set, since it downloads a
  multi-gigabyte base model.
"""

import json
import os

import pytest

from distillkit_r.utils import checkpointing


class _FakePeftModel:
    """Stand-in for a PeftModel that writes adapter files on ``save_pretrained``."""

    def __init__(self) -> None:
        self.pushed_to = None

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as h:
            json.dump({"peft_type": "LORA"}, h)
        # Stand-in for the safetensors blob the real PEFT save would write.
        with open(os.path.join(path, "adapter_model.safetensors"), "wb") as h:
            h.write(b"\x00")

    def merge_and_unload(self) -> "_FakeMergedModel":
        return _FakeMergedModel()

    def push_to_hub(self, repo_id: str, token: str, private: bool = True) -> None:
        self.pushed_to = repo_id


class _FakeMergedModel:
    """Stand-in for a merged standalone model."""

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as h:
            json.dump({"model_type": "qwen3"}, h)
        with open(os.path.join(path, "model.safetensors"), "wb") as h:
            h.write(b"\x00")

    def push_to_hub(self, repo_id: str, token: str, private: bool = True) -> None:
        pass


def test_save_adapter_produces_expected_files(tmp_path, monkeypatch):
    """Saving an adapter yields adapter_config.json + adapter_model.safetensors."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checkpointing, "CHECKPOINT_ROOT", "checkpoints")

    save_path = checkpointing.save_and_optionally_upload(
        peft_model=_FakePeftModel(),
        cfg_dict={"lora_rank": 64},
        stage="sft_seed",
        step=2,
        metrics_dict={"eval_loss": 1.23},
    )

    assert os.path.isfile(os.path.join(save_path, "adapter_config.json"))
    assert os.path.isfile(os.path.join(save_path, "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(save_path, checkpointing.METADATA_FILENAME))
    assert save_path.endswith(os.path.join("sft_seed", "step_2"))

    with open(os.path.join(save_path, checkpointing.METADATA_FILENAME)) as handle:
        metadata_dict = json.load(handle)
    assert metadata_dict["stage"] == "sft_seed"
    assert metadata_dict["step"] == 2
    assert metadata_dict["metrics"]["eval_loss"] == 1.23


def test_merge_before_save_writes_full_model(tmp_path, monkeypatch):
    """The merge path writes a standalone model rather than an adapter."""
    monkeypatch.chdir(tmp_path)
    save_path = checkpointing.save_and_optionally_upload(
        peft_model=_FakePeftModel(),
        cfg_dict={},
        stage="opd_final",
        step=10,
        metrics_dict={"gpu_hours_cumulative": 4.0},
        o_merge_before_save=True,
    )
    assert os.path.isfile(os.path.join(save_path, "config.json"))
    assert os.path.isfile(os.path.join(save_path, "model.safetensors"))


def test_upload_requires_repo_id_and_token(tmp_path, monkeypatch):
    """Uploading without a repo id or token raises before any push."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="hub_repo_id is required"):
        checkpointing.save_and_optionally_upload(
            peft_model=_FakePeftModel(),
            cfg_dict={},
            stage="sft_seed",
            step=1,
            metrics_dict={},
            o_upload_to_hub=True,
            hub_repo_id=None,
        )

    with pytest.raises(ValueError, match="HF_TOKEN"):
        checkpointing.save_and_optionally_upload(
            peft_model=_FakePeftModel(),
            cfg_dict={},
            stage="sft_seed",
            step=1,
            metrics_dict={},
            o_upload_to_hub=True,
            hub_repo_id="org/model",
        )


def test_sft_config_loader_captures_extras():
    """The SFT loader keeps declared fields and forwards unknown YAML keys."""
    from distillkit_r.training.sft_seed import SFTSeedConfig, load_config

    cfg = load_config("configs/sft_seed.yaml")
    assert isinstance(cfg, SFTSeedConfig)
    assert cfg.lora_rank == 64
    assert cfg.target_modules[0] == "q_proj"
    assert cfg.extra_args_dict["bf16"] is True
    assert cfg.extra_args_dict["gradient_checkpointing"] is True


def test_opd_config_loader_captures_extras():
    """The OPD loader keeps declared fields and forwards unknown YAML keys."""
    from distillkit_r.training.opd_trainer import OPDConfig, load_config

    cfg = load_config("configs/gkd_opd.yaml")
    assert isinstance(cfg, OPDConfig)
    assert cfg.lmbda == 1.0
    assert cfg.beta == 1.0
    assert cfg.extra_args_dict["bf16"] is True


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_INTEGRATION") != "1",
    reason="set RUN_HEAVY_INTEGRATION=1 to run the real 2-step SFT smoke test",
)
def test_two_step_sft_smoke(tmp_path, monkeypatch):
    """Real smoke test: a 2-step SFT run produces a valid adapter checkpoint."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("mlflow")

    import mlflow
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    from distillkit_r.data.formatter import apply_template
    from distillkit_r.training.sft_seed import SFTSeedConfig, run_sft_seed

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")

    model_name = "Qwen/Qwen3-1.7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rows = [
        {
            "messages": [
                {"role": "user", "content": f"What is {i} + {i}?"},
                {"role": "assistant", "content": f"{i} + {i} = {2 * i}."},
            ]
        }
        for i in range(10)
    ]
    base = Dataset.from_list(rows)
    formatted = apply_template(base, tokenizer)
    dataset_dict = DatasetDict({"train": formatted, "validation": formatted})

    cfg = SFTSeedConfig(model_name=model_name, output_dir=str(tmp_path / "ckpt"))
    cfg.extra_args_dict = {"max_steps": 2, "save_strategy": "no"}

    checkpoint_path = run_sft_seed(cfg, dataset_dict)

    assert os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json"))
    assert os.path.isfile(os.path.join(checkpoint_path, "adapter_model.safetensors"))
