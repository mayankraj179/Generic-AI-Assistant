"""Connector presets: reviewed operation bundles for known systems, one YAML
file each in app/connectors/presets/. A config's ``connectors[].type`` is
either ``rest`` (operations written inline) or one of these. Presets are
validated when this module loads, so a broken preset fails startup rather
than a chat turn."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.connectors.spec import PresetSpec

REST_TYPE = "rest"
PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def load_presets(directory: Path = PRESETS_DIR) -> dict[str, PresetSpec]:
    presets: dict[str, PresetSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        preset = PresetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if preset.type in presets:
            raise ValueError(f"duplicate connector preset type '{preset.type}' in {path.name}")
        presets[preset.type] = preset
    return presets


CONNECTOR_PRESETS: dict[str, PresetSpec] = load_presets()
