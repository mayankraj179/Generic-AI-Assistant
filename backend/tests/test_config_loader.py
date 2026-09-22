from pathlib import Path

import pytest

from app.config.loader import ConfigLoadError, load_all_assistant_configs, load_assistant_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_loads_sample_hr_assistant_config():
    config = load_assistant_config(CONFIGS_DIR / "hr_assistant.yaml")
    assert config.assistant_id == "hr_assistant"
    assert config.model.provider == "google_adk"
    assert config.retrieval is not None
    assert config.retrieval.collection_name == "hr_policy_docs"
    assert config.enabled_tools == []


def test_missing_file_raises_config_load_error(tmp_path):
    with pytest.raises(ConfigLoadError):
        load_assistant_config(tmp_path / "does_not_exist.yaml")


def test_invalid_yaml_raises_config_load_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("assistant_id: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigLoadError):
        load_assistant_config(bad_file)


def test_schema_violation_raises_config_load_error(tmp_path):
    bad_file = tmp_path / "bad_schema.yaml"
    bad_file.write_text("assistant_id: 'has spaces here'\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError):
        load_assistant_config(bad_file)


def test_load_all_configs_from_directory():
    configs = load_all_assistant_configs(CONFIGS_DIR)
    assert "hr_assistant" in configs


def test_duplicate_assistant_id_raises(tmp_path):
    (tmp_path / "a.yaml").write_text(
        _minimal_config_yaml("dup_id"), encoding="utf-8"
    )
    (tmp_path / "b.yaml").write_text(
        _minimal_config_yaml("dup_id"), encoding="utf-8"
    )
    with pytest.raises(ConfigLoadError):
        load_all_assistant_configs(tmp_path)


def _minimal_config_yaml(assistant_id: str) -> str:
    return f"""
assistant_id: {assistant_id}
display_name: Test Assistant
description: A test assistant.
tenant_id: test-tenant
model:
  provider: azure_openai
  model_name: gpt-4.1
system_prompt: You are a test assistant.
"""
