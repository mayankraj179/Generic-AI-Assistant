from pathlib import Path

import pytest

from app.config.loader import ConfigLoadError, load_all_assistant_configs, load_assistant_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_loads_sample_hr_assistant_config():
    config = load_assistant_config(CONFIGS_DIR / "hr_assistant.yaml")
    assert config.assistant_id == "hr_assistant"
    assert config.model.provider == "openrouter"
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


def test_active_assistants_and_their_providers():
    # configs/examples/ is not scanned (non-recursive glob), so the parked
    # OpenRouter example must not be loaded as a live assistant.
    configs = load_all_assistant_configs(CONFIGS_DIR)
    assert {aid: c.model.provider for aid, c in configs.items()} == {
        "hr_assistant": "openrouter",
        "finance_assistant": "openrouter",
        "hr_assistant_grok": "xai",
        "finance_assistant_grok": "xai",
    }


def test_hr_grok_assistant_uses_grok_chat_and_temp_gemini_embeddings():
    grok = load_assistant_config(CONFIGS_DIR / "hr_assistant_grok.yaml")
    hr = load_assistant_config(CONFIGS_DIR / "hr_assistant.yaml")
    assert grok.model.provider == "xai"
    assert grok.model.model_name == "grok-4.3"
    assert grok.retrieval is not None and hr.retrieval is not None
    # TEMP (2026-09-29): Grok configs use Gemini embeddings with their own
    # measured threshold, to avoid OpenRouter's daily quota. Prompt and
    # collection still match hr_assistant.
    assert grok.retrieval.embedding_provider == "gemini"
    assert grok.retrieval.embedding_model is None
    assert grok.retrieval.min_similarity == 0.66
    assert grok.retrieval.collection_name == hr.retrieval.collection_name
    assert grok.system_prompt == hr.system_prompt


def test_finance_grok_assistant_mirrors_finance_assistant_except_chat_model():
    grok = load_assistant_config(CONFIGS_DIR / "finance_assistant_grok.yaml")
    finance = load_assistant_config(CONFIGS_DIR / "finance_assistant.yaml")
    assert grok.model.provider == "xai"
    assert grok.model.model_name == "grok-4.3"
    assert grok.retrieval is not None and finance.retrieval is not None
    # TEMP (2026-09-29): Gemini embeddings, own measured threshold.
    assert grok.retrieval.embedding_provider == "gemini"
    assert grok.retrieval.embedding_model is None
    assert grok.retrieval.min_similarity == 0.645
    assert grok.retrieval.collection_name == finance.retrieval.collection_name
    assert grok.system_prompt == finance.system_prompt
    assert grok.enabled_tools == finance.enabled_tools


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
def test_active_assistants_stay_fully_on_openrouter(assistant_id):
    # Guards against these assistants being moved to another provider as a
    # side effect of unrelated work (e.g. a missing API key). Switched back
    # from Gemini to OpenRouter deliberately on 2026-09-30; switching again
    # means updating this test along with the configs and re-ingesting.
    config = load_assistant_config(CONFIGS_DIR / f"{assistant_id}.yaml")
    assert config.model.provider == "openrouter"
    assert config.model.model_name == "nvidia/nemotron-3-super-120b-a12b:free"
    assert config.retrieval is not None
    assert config.retrieval.embedding_provider == "openrouter"
    assert config.retrieval.embedding_model == "nvidia/nemotron-3-embed-1b:free"


def test_parked_openrouter_example_is_fully_on_openrouter():
    config = load_assistant_config(CONFIGS_DIR / "examples" / "finance_assistant_openrouter.yaml")
    assert config.model.provider == "openrouter"
    assert config.model.model_name == "nvidia/nemotron-3-super-120b-a12b:free"
    assert config.retrieval is not None
    assert config.retrieval.embedding_provider == "openrouter"
    assert config.retrieval.embedding_model == "nvidia/nemotron-3-embed-1b:free"
    assert "experiment" in config.display_name.lower()
