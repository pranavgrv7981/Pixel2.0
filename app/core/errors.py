"""Pixel v2 error hierarchy.

Every error carries structured context so failures are traceable
through logs and diagnostics.
"""

from __future__ import annotations

from typing import Any


class PixelError(Exception):
    """Base exception for all Pixel errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{details}]"
        return base


class ConfigError(PixelError):
    """Configuration loading or validation failure."""


class StateError(PixelError):
    """Invalid state transition."""


class ProviderError(PixelError):
    """AI provider failure."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        stage: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx.update({"provider": provider, "model": model, "stage": stage})
        super().__init__(message, context=ctx)
        self.provider = provider
        self.model = model
        self.stage = stage


class ActionError(PixelError):
    """Action execution failure."""

    def __init__(
        self,
        message: str,
        *,
        action: str = "",
        stage: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx.update({"action": action, "stage": stage})
        super().__init__(message, context=ctx)
        self.action = action
        self.stage = stage


class SecurityError(PixelError):
    """Permission denied or security policy violation."""

    def __init__(
        self,
        message: str,
        *,
        action: str = "",
        risk_level: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx.update({"action": action, "risk_level": risk_level})
        super().__init__(message, context=ctx)
        self.action = action
        self.risk_level = risk_level


class IntentError(PixelError):
    """Intent detection failure."""


class EventError(PixelError):
    """Event bus error (e.g. cascade limit exceeded)."""
