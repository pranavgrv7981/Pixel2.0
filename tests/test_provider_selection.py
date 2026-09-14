"""Tests for deterministic AI provider selection, priorities, and fallback chains."""

from __future__ import annotations

import pytest
from typing import AsyncIterator

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.ai.registry import ProviderRegistry, create_provider_registry
from app.core.config import AIConfig, ProvidersConfig, CloudProviderConfig, OllamaProviderConfig, FallbackProviderConfig
from app.core.types import ProviderCapability


class MockProvider(AIProvider):
    """Configurable mock AI provider for testing selection logic."""

    def __init__(
        self,
        name: str,
        available: bool = True,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self._name = name
        self._available = available
        self._caps = capabilities or ProviderCapabilities(conversation=True, streaming=True)

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        return ChatResponse(content=f"Response from {self._name}", model=f"{self._name}-m", provider=self._name)

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        yield f"Response from {self._name}"

    async def is_available(self) -> bool:
        return self._available

    def capabilities(self) -> ProviderCapabilities:
        return self._caps


@pytest.mark.unit
@pytest.mark.asyncio
async def test_priority_order_determines_selection():
    """Provider selection strictly honors the configured priority list, not registration order."""
    # Register in order: provider_b, provider_a
    reg = ProviderRegistry(priority=["provider_a", "provider_b"])
    b = MockProvider("provider_b", available=True)
    a = MockProvider("provider_a", available=True)

    reg.register(b)
    reg.register(a)

    selected = await reg.select_provider()
    assert selected is not None
    assert selected.name == "provider_a"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_order_invariance():
    """Registering providers in different orders produces identical, deterministic selection."""
    reg1 = ProviderRegistry(priority=["alpha", "beta", "gamma"])
    reg1.register(MockProvider("gamma"))
    reg1.register(MockProvider("alpha"))
    reg1.register(MockProvider("beta"))

    reg2 = ProviderRegistry(priority=["alpha", "beta", "gamma"])
    reg2.register(MockProvider("beta"))
    reg2.register(MockProvider("gamma"))
    reg2.register(MockProvider("alpha"))

    sel1 = await reg1.select_provider()
    sel2 = await reg2.select_provider()

    assert sel1 is not None and sel2 is not None
    assert sel1.name == sel2.name == "alpha"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_preferred_provider_evaluated_first():
    """Specifying preferred_provider overrides default priority order if available."""
    reg = ProviderRegistry(priority=["provider_a", "provider_b"])
    reg.register(MockProvider("provider_a", available=True))
    reg.register(MockProvider("provider_b", available=True))

    selected = await reg.select_provider(preferred_provider="provider_b")
    assert selected is not None
    assert selected.name == "provider_b"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unavailable_higher_priority_falls_back():
    """When a higher-priority provider is unavailable, next in priority chain is selected."""
    reg = ProviderRegistry(priority=["cloud", "ollama", "fallback"])
    reg.register(MockProvider("cloud", available=False))
    reg.register(MockProvider("ollama", available=False))
    reg.register(MockProvider("fallback", available=True))

    selected = await reg.select_provider()
    assert selected is not None
    assert selected.name == "fallback"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capability_filtering():
    """Only providers satisfying all requested capabilities are candidates."""
    reg = ProviderRegistry(priority=["fast_provider", "vision_provider"])
    fast = MockProvider(
        "fast_provider",
        available=True,
        capabilities=ProviderCapabilities(conversation=True, vision=False),
    )
    vision = MockProvider(
        "vision_provider",
        available=True,
        capabilities=ProviderCapabilities(conversation=True, vision=True),
    )
    reg.register(fast)
    reg.register(vision)

    # When VISION is required, fast_provider must be skipped even though it has higher priority
    selected = await reg.select_provider(required_capabilities={ProviderCapability.VISION})
    assert selected is not None
    assert selected.name == "vision_provider"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_chain_order():
    """Candidate chain returns ordered fallback sequence of all available eligible providers."""
    reg = ProviderRegistry(priority=["first", "second", "third", "unavailable"])
    reg.register(MockProvider("third", available=True))
    reg.register(MockProvider("first", available=True))
    reg.register(MockProvider("unavailable", available=False))
    reg.register(MockProvider("second", available=True))

    chain = await reg.select_candidate_chain()
    names = [p.name for p in chain]
    assert names == ["first", "second", "third"]


@pytest.mark.unit
def test_create_provider_registry_factory():
    """Factory creates registry from config without hardcoded provider coupling in runtime."""
    cfg = AIConfig(
        default_provider="fallback",
        provider_priority=["cloud", "ollama", "fallback"],
        providers=ProvidersConfig(
            cloud=CloudProviderConfig(enabled=False),
            ollama=OllamaProviderConfig(enabled=False),
            fallback=FallbackProviderConfig(enabled=True),
        ),
    )
    reg = create_provider_registry(cfg)
    assert reg.priority == ["cloud", "ollama", "fallback"]
    assert "cloud" in reg.list_all()
    assert "ollama" in reg.list_all()
    assert "fallback" in reg.list_all()
