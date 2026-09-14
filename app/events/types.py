"""Typed event definitions for the Pixel v2 event bus.

All events inherit from PixelEvent and carry event_id, timestamp, source.
Uses dataclass with kw_only=True to avoid field ordering issues with inheritance.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.types import PixelStateEnum, Route


@dataclass(kw_only=True)
class PixelEvent:
    """Base event class for all system events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


@dataclass(kw_only=True)
class StateChanged(PixelEvent):
    """Emitted when the runtime state changes."""
    old_state: PixelStateEnum = PixelStateEnum.IDLE
    new_state: PixelStateEnum = PixelStateEnum.IDLE


@dataclass(kw_only=True)
class UserMessageReceived(PixelEvent):
    """Emitted when the user sends a text message."""
    text: str = ""
    session_id: str = ""


@dataclass(kw_only=True)
class VoiceInputStarted(PixelEvent):
    """Emitted when voice capture begins."""
    pass


@dataclass(kw_only=True)
class VoiceInputReceived(PixelEvent):
    """Emitted when STT produces text from voice input."""
    text: str = ""
    confidence: float = 0.0


@dataclass(kw_only=True)
class IntentDetected(PixelEvent):
    """Emitted when an intent is recognized."""
    intent: str = ""
    confidence: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class AIRequestStarted(PixelEvent):
    """Emitted when an AI request is sent to a provider."""
    provider: str = ""
    model: str = ""
    route: Route = Route.STANDARD


@dataclass(kw_only=True)
class AIFirstToken(PixelEvent):
    """Emitted when the first token is received from an AI provider."""
    provider: str = ""
    latency_ms: float = 0.0


@dataclass(kw_only=True)
class AIResponseCompleted(PixelEvent):
    """Emitted when an AI response is fully received."""
    provider: str = ""
    model: str = ""
    response: str = ""
    latency_ms: float = 0.0


@dataclass(kw_only=True)
class ActionRequested(PixelEvent):
    """Emitted when an action is requested."""
    action: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ActionStarted(PixelEvent):
    """Emitted when an action begins execution."""
    action: str = ""


@dataclass(kw_only=True)
class ActionCompleted(PixelEvent):
    """Emitted when an action completes successfully."""
    action: str = ""
    success: bool = True
    output: str = ""


@dataclass(kw_only=True)
class ActionFailed(PixelEvent):
    """Emitted when an action fails."""
    action: str = ""
    error: str = ""


@dataclass(kw_only=True)
class MemoryCreated(PixelEvent):
    """Emitted when a memory entry is created."""
    category: str = ""
    key: str = ""


@dataclass(kw_only=True)
class MemoryRetrieved(PixelEvent):
    """Emitted when a memory entry is retrieved."""
    category: str = ""
    key: str = ""


@dataclass(kw_only=True)
class BehaviorObserved(PixelEvent):
    """Emitted when a user behavior is observed."""
    behavior: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class BehaviorPatternDetected(PixelEvent):
    """Emitted when a repeated behavior pattern is detected."""
    pattern: str = ""
    confidence: float = 0.0


@dataclass(kw_only=True)
class TaskStarted(PixelEvent):
    """Emitted when a task begins."""
    task_id: str = ""
    description: str = ""


@dataclass(kw_only=True)
class TaskCompleted(PixelEvent):
    """Emitted when a task completes."""
    task_id: str = ""


@dataclass(kw_only=True)
class PixelActivated(PixelEvent):
    """Emitted when Pixel is activated (startup, wake)."""
    pass


@dataclass(kw_only=True)
class PixelDeactivated(PixelEvent):
    """Emitted when Pixel is deactivated (shutdown, sleep)."""
    pass
