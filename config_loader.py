from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    endpoint: str
    subscription_key: str
    timeout_seconds: float = 30.0
    api_version: str | None = None


@dataclass(frozen=True)
class Settings:
    providers: dict[str, ProviderConfig]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        # Use utf-8-sig so JSON files with a BOM (common on Windows) still parse.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON config file: {path}") from exc


def load_settings() -> Settings:
    config_path = Path(os.getenv("APIM_CONFIG_PATH", "config.json"))
    payload = _load_json(config_path)

    providers_raw = payload.get("providers")
    if not isinstance(providers_raw, dict):
        raise ConfigError("Config must include a 'providers' object")

    providers: dict[str, ProviderConfig] = {}
    for name, conf in providers_raw.items():
        if not isinstance(conf, dict):
            raise ConfigError(f"Provider '{name}' config must be an object")

        endpoint = conf.get("endpoint")
        subscription_key = conf.get("subscription_key")

        if not endpoint or not isinstance(endpoint, str):
            raise ConfigError(f"Provider '{name}' is missing valid 'endpoint'")
        if not subscription_key or not isinstance(subscription_key, str):
            raise ConfigError(
                f"Provider '{name}' is missing valid 'subscription_key'"
            )

        providers[name] = ProviderConfig(
            endpoint=endpoint,
            subscription_key=subscription_key,
            timeout_seconds=float(conf.get("timeout_seconds", 30)),
            api_version=conf.get("api_version"),
        )

    return Settings(providers=providers)
