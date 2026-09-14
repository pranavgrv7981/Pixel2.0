"""Tests for AI providers and registry."""

from __future__ import annotations

import pytest
from app.ai.registry import ProviderRegistry
from app.ai.providers.fallback import FallbackProvider
from app.ai.provider import ChatMessage, ChatResponse


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_always_available():
    provider = FallbackProvider()
    assert await provider.is_available() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_chat():
    provider = FallbackProvider()
    messages = [ChatMessage(role="user", content="hello")]
    response = await provider.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.unit
def test_registry_register_and_get():
    reg = ProviderRegistry()
    provider = FallbackProvider()
    reg.register(provider)
    got = reg.get(provider.name)
    assert got is provider


@pytest.mark.unit
def test_registry_get_nonexistent():
    reg = ProviderRegistry()
    assert reg.get("nonexistent") is None


@pytest.mark.unit
def test_registry_list_all():
    reg = ProviderRegistry()
    reg.register(FallbackProvider())
    names = reg.list_all()
    assert "fallback" in names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_get_available():
    reg = ProviderRegistry()
    reg.register(FallbackProvider())
    available = await reg.get_available()
    assert len(available) > 0
    assert available[0].name == "fallback"


@pytest.mark.unit
def test_registry_remove():
    reg = ProviderRegistry()
    provider = FallbackProvider()
    reg.register(provider)
    reg.remove(provider.name)
    assert reg.get(provider.name) is None
