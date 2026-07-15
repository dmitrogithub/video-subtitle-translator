"""YAML configuration loading for the subtitle command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.exceptions import ConfigurationError
from app.models import SubtitleTranslatorSettings


def load_settings(config_path: Path | None = None) -> SubtitleTranslatorSettings:
    """Load validated YAML settings, or return the documented defaults."""
    if config_path is None:
        return SubtitleTranslatorSettings()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file was not found: {config_path}")
    try:
        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML configuration: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a YAML mapping.")
    try:
        return SubtitleTranslatorSettings.model_validate(payload)
    except ValidationError as error:
        raise ConfigurationError(f"Invalid subtitle configuration: {error}") from error
