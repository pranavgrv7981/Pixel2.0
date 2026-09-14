"""Shared type definitions and enums for Pixel v2."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """Request routing classification."""
    DIRECT = "DIRECT"
    FAST = "FAST"
    STANDARD = "STANDARD"
    HEAVY = "HEAVY"


# ---------------------------------------------------------------------------
# Pixel State
# ---------------------------------------------------------------------------

class PixelStateEnum(str, Enum):
    """Runtime state machine states."""
    IDLE = "IDLE"
    AWAKENING = "AWAKENING"
    LISTENING = "LISTENING"
    UNDERSTANDING = "UNDERSTANDING"
    THINKING = "THINKING"
    RESPONDING = "RESPONDING"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Risk Levels
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Action risk classification."""
    READ = "READ"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Intent Types
# ---------------------------------------------------------------------------

class DirectIntent(str, Enum):
    """Deterministically recognizable intents."""
    TIME = "TIME"
    DATE = "DATE"
    CALCULATOR = "CALCULATOR"
    BATTERY = "BATTERY"
    SYSTEM_INFO = "SYSTEM_INFO"
    OPEN_APPLICATION = "OPEN_APPLICATION"
    OPEN_WEBSITE = "OPEN_WEBSITE"
    LOCK_PC = "LOCK_PC"
    GREETING = "GREETING"
    MEMORY_STORE = "MEMORY_STORE"
    MEMORY_RECALL = "MEMORY_RECALL"


# ---------------------------------------------------------------------------
# Provider Capabilities
# ---------------------------------------------------------------------------

class ProviderCapability(str, Enum):
    """Capabilities an AI provider may support."""
    CONVERSATION = "CONVERSATION"
    TOOL_CALLING = "TOOL_CALLING"
    REASONING = "REASONING"
    VISION = "VISION"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"
    STREAMING = "STREAMING"


# ---------------------------------------------------------------------------
# Permission Decisions
# ---------------------------------------------------------------------------

class PermissionDecision(str, Enum):
    """Result of a permission check."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    PROMPT = "PROMPT"  # Future: ask user


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class RequestTelemetry:
    """Lightweight telemetry record for a single request."""

    __slots__ = (
        "request_id", "timestamp", "route", "provider", "model",
        "input_tokens", "output_tokens", "ttft_ms", "generation_ms",
        "total_ms", "success", "fallback_used", "error",
    )

    def __init__(self, request_id: str, **kwargs: Any) -> None:
        self.request_id = request_id
        for k, v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)
        # Set missing attributes to None
        for slot in self.__slots__:
            if not hasattr(self, slot):
                setattr(self, slot, None)

    def to_dict(self) -> dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}
