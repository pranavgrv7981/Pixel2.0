from __future__ import annotations

from typing import Set

from .provider import AIProvider
from app.core.types import ProviderCapability
from app.core.logging import get_logger

logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for managing AI providers."""

    def __init__(self):
        self._providers: dict[str, AIProvider] = {}

    def register(self, provider: AIProvider) -> None:
        """Register a new provider."""
        self._providers[provider.name] = provider
        logger.debug(f"Registered AI provider: {provider.name}")

    def get(self, name: str) -> AIProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    async def get_available(self) -> list[AIProvider]:
        """Get all currently available providers."""
        available = []
        for provider in self._providers.values():
            if await provider.is_available():
                available.append(provider)
        return available

    async def get_best(self, required_capabilities: Set[ProviderCapability] | None = None) -> AIProvider | None:
        """Get the best available provider that meets required capabilities."""
        cap_map = {
            ProviderCapability.CONVERSATION: "conversation",
            ProviderCapability.TOOL_CALLING: "tool_calling",
            ProviderCapability.REASONING: "reasoning",
            ProviderCapability.VISION: "vision",
            ProviderCapability.STRUCTURED_OUTPUT: "structured_output",
            ProviderCapability.STREAMING: "streaming",
        }

        available = await self.get_available()
        
        if not required_capabilities:
            return available[0] if available else None

        for provider in available:
            caps = provider.capabilities()
            meets_all = True
            for req in required_capabilities:
                attr_name = cap_map.get(req)
                if attr_name and not getattr(caps, attr_name, False):
                    meets_all = False
                    break
            if meets_all:
                return provider
                
        return None

    def list_all(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def remove(self, name: str) -> None:
        """Remove a provider from the registry."""
        if name in self._providers:
            del self._providers[name]
            logger.debug(f"Removed AI provider: {name}")
