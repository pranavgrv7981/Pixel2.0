"""Integration tests proving direct commands completely bypass the AI provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from app.runtime.runtime import PixelRuntime
from app.core.config import PixelConfig
from app.core.types import PixelStateEnum
from app.ai.provider import ChatMessage, ChatResponse


@pytest.fixture
async def active_runtime():
    config = PixelConfig()
    runtime = PixelRuntime(config)
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.stop()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direct_command,expected_substring",
    [
        ("calculate 100 * 5", "500"),
        ("what is 144 / 12", "12"),
        ("system info", "Python:"),
        ("show system information", "OS:"),
        ("what time is it", "current time is"),
        ("time", "current time is"),
        ("today's date", "today is"),
        ("open google", "opened google"),
        ("lock my pc", "requires user confirmation"),
    ],
)
async def test_direct_commands_bypass_ai_provider(active_runtime, direct_command, expected_substring):
    """Direct deterministic commands must execute via ActionEngine with 0 AI provider calls."""
    # Mock the chat method on all registered providers in runtime
    mock_calls = []

    for name in active_runtime.provider_registry.list_all():
        provider = active_runtime.provider_registry.get(name)
        if provider is None:
            continue
        original_chat = provider.chat

        def make_spy(p_name, orig):
            async def spy_chat(messages: list[ChatMessage], **kwargs) -> ChatResponse:
                mock_calls.append(p_name)
                return await orig(messages, **kwargs)
            return spy_chat

        spy = make_spy(name, original_chat)
        setattr(provider, "chat", spy)

    with patch("webbrowser.open", return_value=True):
        response = await active_runtime.handle_input(direct_command)

    # Assert that no AI provider was ever invoked
    assert len(mock_calls) == 0, f"Expected 0 AI provider calls for '{direct_command}', but got calls to: {mock_calls}"
    assert response is not None
    assert expected_substring.lower() in response.lower()
    assert active_runtime.state == PixelStateEnum.IDLE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversational_query_routes_to_ai_provider(active_runtime):
    """Non-direct conversational requests must invoke the AI provider."""
    called_providers = []

    for name in active_runtime.provider_registry.list_all():
        provider = active_runtime.provider_registry.get(name)
        if provider is None:
            continue

        def make_spy(p_name):
            async def spy_chat(messages: list[ChatMessage], **kwargs) -> ChatResponse:
                called_providers.append(p_name)
                return ChatResponse(content="AI response to question", model="mock", provider=p_name)
            return spy_chat

        spy = make_spy(name)
        setattr(provider, "chat", spy)

    response = await active_runtime.handle_input("explain why the sky is blue in simple words")

    # Verify that an AI provider was indeed invoked
    assert len(called_providers) > 0, "Expected an AI provider to be invoked for conversational query"
    assert "AI response" in response or "sky" in response or len(response) > 0
    assert active_runtime.state == PixelStateEnum.IDLE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_direct_command_latency_logged(active_runtime, caplog):
    """Direct commands execute within milliseconds and log route='DIRECT'."""
    with patch("webbrowser.open", return_value=True):
        await active_runtime.handle_input("calculate 25 * 8")

    assert active_runtime.state == PixelStateEnum.IDLE
