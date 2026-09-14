"""Hardening tests for PixelRuntime lifecycle, error recovery, and fallback execution."""

from __future__ import annotations

import pytest
from typing import AsyncIterator
from unittest.mock import patch

from app.runtime.runtime import PixelRuntime
from app.runtime.state import PixelStateMachine
from app.core.config import PixelConfig
from app.core.types import PixelStateEnum
from app.core.errors import ProviderError
from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.ai.registry import ProviderRegistry


class FailingProvider(AIProvider):
    """Provider that is available but fails when chat() is called."""

    def __init__(self, name: str = "failing_primary") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        raise ProviderError("Connection timeout to primary AI", provider=self.name)

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        raise ProviderError("Connection timeout", provider=self.name)
        yield ""

    async def is_available(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(conversation=True, streaming=True)


class WorkingFallbackProvider(AIProvider):
    """Reliable fallback provider."""

    def __init__(self, name: str = "backup_fallback") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        return ChatResponse(content="Recovered via backup provider", model="backup-m", provider=self.name)

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        yield "Recovered via backup provider"

    async def is_available(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(conversation=True, streaming=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repeated_start_is_safe():
    """Calling start() multiple times logs a warning and does not corrupt runtime state."""
    runtime = PixelRuntime(PixelConfig())
    await runtime.start()
    assert runtime.running is True
    first_session = runtime.session_id

    # Second start call
    await runtime.start()
    assert runtime.running is True
    assert runtime.session_id == first_session
    assert runtime.state == PixelStateEnum.IDLE

    await runtime.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repeated_stop_is_safe():
    """Calling stop() multiple times terminates cleanly without error."""
    runtime = PixelRuntime(PixelConfig())
    await runtime.start()
    await runtime.stop()
    assert runtime.running is False

    # Second stop call
    await runtime.stop()
    assert runtime.running is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_input_when_stopped():
    """handle_input safely rejects execution when the runtime is not running."""
    runtime = PixelRuntime(PixelConfig())
    response = await runtime.handle_input("what time is it")
    assert response == "Pixel is not running."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subsystem_init_failure_cleans_up():
    """If subsystem initialization fails during start(), resources are cleaned up cleanly."""
    runtime = PixelRuntime(PixelConfig())

    with patch.object(PixelStateMachine, "transition", side_effect=RuntimeError("Subsystem boot error")):
        with pytest.raises(RuntimeError):
            # Patch StateMachine init or start step to simulate failure
            with patch("app.runtime.runtime.PixelStateMachine", side_effect=RuntimeError("Failed to init state machine")):
                await runtime.start()

    # After failure, runtime must not be in a half-running state
    assert runtime.running is False
    assert runtime.state_machine is None
    assert runtime.event_bus is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_fallback_on_primary_failure():
    """When the primary AI provider fails at runtime, PixelRuntime automatically falls back to secondary."""
    reg = ProviderRegistry(priority=["failing_primary", "backup_fallback"])
    reg.register(FailingProvider("failing_primary"))
    reg.register(WorkingFallbackProvider("backup_fallback"))

    runtime = PixelRuntime(PixelConfig(), provider_registry=reg)
    await runtime.start()

    # Send a request that routes to AI (not direct intent)
    response = await runtime.handle_input("explain the architecture of transformers")
    assert response == "Recovered via backup provider"
    assert runtime.state == PixelStateEnum.IDLE

    await runtime.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_recovery_transitions_through_error_state():
    """An unhandled error in request handling transitions through ERROR before resetting to IDLE."""
    runtime = PixelRuntime(PixelConfig())
    await runtime.start()

    # Force intent_router to raise an unexpected error during handle_input
    with patch.object(runtime.intent_router, "route", side_effect=ValueError("Intent crash")):
        response = await runtime.handle_input("test input")

    assert "unexpected error" in response.lower() or "wrong" in response.lower()
    assert runtime.state == PixelStateEnum.IDLE

    # History must show transition to ERROR then back to IDLE
    history_states = [new for old, new in runtime.state_machine.history]
    assert PixelStateEnum.ERROR in history_states

    await runtime.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_system_info_action_via_direct_intent():
    """System info request executes registered system_info action and returns platform details."""
    runtime = PixelRuntime(PixelConfig())
    await runtime.start()

    response = await runtime.handle_input("system info")
    assert response is not None
    assert "Python:" in response or "OS:" in response
    assert runtime.state == PixelStateEnum.IDLE

    await runtime.stop()
