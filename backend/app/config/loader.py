from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.config.assistant_config import AssistantConfig


class ConfigLoadError(Exception):
    """Raised when an assistant config file fails to parse or validate."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"failed to load assistant config from {path}: {reason}")


def load_assistant_config(path: str | Path) -> AssistantConfig:
    """Load and validate a single AssistantConfig YAML file.

    This is the sole mechanism for onboarding a new assistant use case —
    no core code changes should be required beyond adding a file here.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigLoadError(path, "file does not exist")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(path, f"invalid YAML: {exc}") from exc

    if raw is None:
        raise ConfigLoadError(path, "file is empty")

    try:
        return AssistantConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(path, f"schema validation failed: {exc}") from exc


def load_all_assistant_configs(directory: str | Path) -> dict[str, AssistantConfig]:
    """Load every *.yaml/*.yml assistant config in a directory.

    Fails fast on the first invalid file rather than silently skipping it —
    a broken config should never be dropped without anyone noticing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigLoadError(directory, "not a directory")

    configs: dict[str, AssistantConfig] = {}
    for pattern in ("*.yaml", "*.yml"):
        for file_path in sorted(directory.glob(pattern)):
            config = load_assistant_config(file_path)
            if config.assistant_id in configs:
                raise ConfigLoadError(
                    file_path,
                    f"duplicate assistant_id '{config.assistant_id}' "
                    f"already loaded from another file",
                )
            configs[config.assistant_id] = config

    return configs
