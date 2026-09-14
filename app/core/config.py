"""Configuration system for Pixel v2.

Loads from TOML files with environment variable overrides (PIXEL_ prefix).
Uses pydantic for validation. Works with sensible defaults if no config file exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import ConfigError

# Python 3.11+ has tomllib in stdlib
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redefine]


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"  # "console" or "json"
    file: str = ""


class CloudProviderConfig(BaseModel):
    enabled: bool = False
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key: str = ""
    model: str = "openrouter/free"
    default_model: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    http_referer: str = ""
    x_title: str = "Pixel"

    def get_model(self) -> str:
        """Return the effective model name."""
        return self.model or self.default_model or "openrouter/free"

    def get_api_key(self) -> str:
        """Retrieve API key securely from explicit setting or environment variable."""
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        if self.api_key_env:
            return os.getenv(self.api_key_env, "").strip()
        return ""


class OllamaProviderConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"


class FallbackProviderConfig(BaseModel):
    enabled: bool = True


class ProvidersConfig(BaseModel):
    cloud: CloudProviderConfig = CloudProviderConfig()
    ollama: OllamaProviderConfig = OllamaProviderConfig()
    fallback: FallbackProviderConfig = FallbackProviderConfig()


class AIConfig(BaseModel):
    default_provider: str = "fallback"
    provider_priority: list[str] = Field(default_factory=lambda: ["cloud", "ollama", "fallback"])
    request_timeout_seconds: int = 30
    max_retries: int = 2
    providers: ProvidersConfig = ProvidersConfig()


class RouterConfig(BaseModel):
    direct_confidence_threshold: float = 0.7
    fast_max_tokens: int = 256
    standard_max_tokens: int = 2048
    heavy_max_tokens: int = 8192


class SecurityConfig(BaseModel):
    auto_allow_risk_levels: list[str] = Field(default_factory=lambda: ["READ", "LOW"])
    prompt_risk_levels: list[str] = Field(default_factory=lambda: ["MEDIUM"])
    deny_risk_levels: list[str] = Field(default_factory=lambda: ["HIGH", "CRITICAL"])


class IntentConfig(BaseModel):
    greeting_response: str = "Hello! I'm Pixel, your personal assistant. How can I help?"


class RuntimeConfig(BaseModel):
    name: str = "Pixel"
    version: str = "2.0.0-alpha"
    debug: bool = False


class PixelConfig(BaseModel):
    """Top-level Pixel configuration."""
    runtime: RuntimeConfig = RuntimeConfig()
    logging: LoggingConfig = LoggingConfig()
    ai: AIConfig = AIConfig()
    router: RouterConfig = RouterConfig()
    security: SecurityConfig = SecurityConfig()
    intent: IntentConfig = IntentConfig()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_ENV_PREFIX = "PIXEL_"


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides with PIXEL_ prefix.

    Format: PIXEL_SECTION__KEY=value  (double underscore for nesting)
    Example: PIXEL_LOGGING__LEVEL=DEBUG
    """
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = key[len(_ENV_PREFIX):].lower().split("__")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def load_config(path: Path | str | None = None) -> PixelConfig:
    """Load configuration from TOML file with env overrides.

    Args:
        path: Path to TOML config file. If None, uses defaults only.

    Returns:
        Validated PixelConfig instance.
    """
    data: dict[str, Any] = {}

    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    data = tomllib.load(f)
            except Exception as exc:
                raise ConfigError(
                    f"Failed to load config from {config_path}",
                    context={"path": str(config_path), "error": str(exc)},
                ) from exc
        else:
            raise ConfigError(
                f"Config file not found: {config_path}",
                context={"path": str(config_path)},
            )

    data = _apply_env_overrides(data)

    try:
        return PixelConfig(**data)
    except Exception as exc:
        raise ConfigError(
            "Invalid configuration",
            context={"error": str(exc)},
        ) from exc
