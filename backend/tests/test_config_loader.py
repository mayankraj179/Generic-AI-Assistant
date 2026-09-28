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


def test_only_gemini_assistants_are_active():
    # configs/examples/ is not scanned (non-recursive glob), so the parked
    # OpenRouter example must not be loaded as a live assistant.
    configs = load_all_assistant_configs(CONFIGS_DIR)
    assert set(configs) == {"hr_assistant", "finance_assistant"}
    assert all(c.model.provider == "google_adk" for c in configs.values())


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


@pytest.mark.parametrize("assistant_id", ["hr_assistant", "finance_assistant"])
def test_gemini_assistants_stay_fully_on_gemini(assistant_id):
    # Guards against these long-verified assistants being moved to another
    # provider as a side effect of unrelated work (e.g. a missing API key).
    config = load_assistant_config(CONFIGS_DIR / f"{assistant_id}.yaml")
    assert config.model.provider == "google_adk"
    assert config.model.model_name == "gemini-2.5-flash"
    assert config.retrieval is not None
    assert config.retrieval.embedding_provider == "gemini"
    assert config.retrieval.embedding_model is None


def test_parked_openrouter_example_is_fully_on_openrouter():
    config = load_assistant_config(CONFIGS_DIR / "examples" / "finance_assistant_openrouter.yaml")
    assert config.model.provider == "openrouter"
    assert config.model.model_name == "nvidia/nemotron-3-super-120b-a12b:free"
    assert config.retrieval is not None
    assert config.retrieval.embedding_provider == "openrouter"
    assert config.retrieval.embedding_model == "nvidia/nemotron-3-embed-1b:free"
    assert "experiment" in config.display_name.lower()
