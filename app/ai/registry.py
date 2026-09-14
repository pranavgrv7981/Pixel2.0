"""Provider registry and deterministic provider selection for Pixel v2.

Manages AI providers, verifies capabilities and availability, and implements
an explicit, deterministic provider selection policy.
"""

from __future__ import annotations

from typing import Set

from .provider import AIProvider
from .providers.cloud import CloudProvider
from .providers.ollama import OllamaProvider
from .providers.fallback import FallbackProvider

from app.core.config import AIConfig
from app.core.types import ProviderCapability
from app.core.logging import get_logger

logger = get_logger("pixel.ai.registry")

_CAPABILITY_MAP: dict[ProviderCapability, str] = {
    ProviderCapability.CONVERSATION: "conversation",
    ProviderCapability.TOOL_CALLING: "tool_calling",
    ProviderCapability.REASONING: "reasoning",
    ProviderCapability.VISION: "vision",
    ProviderCapability.STRUCTURED_OUTPUT: "structured_output",
    ProviderCapability.STREAMING: "streaming",
}


class ProviderRegistry:
    """Registry for managing AI providers with deterministic selection policy."""

    def __init__(
        self,
        priority: list[str] | None = None,
        default_provider: str = "fallback",
    ) -> None:
        self._providers: dict[str, AIProvider] = {}
        self._priority: list[str] = list(priority or ["cloud", "ollama", "fallback"])
        self._default_provider: str = default_provider

    # -- Configuration -------------------------------------------------------

    @property
    def priority(self) -> list[str]:
        """Ordered list of provider names defining preference order."""
        return list(self._priority)

    def set_priority(self, priority: list[str]) -> None:
        """Set the priority order for provider selection."""
        self._priority = list(priority)

    @property
    def default_provider(self) -> str:
        """Default provider name."""
        return self._default_provider

    def set_default_provider(self, name: str) -> None:
        """Set the default provider name."""
        self._default_provider = name

    # -- Registration --------------------------------------------------------

    def register(self, provider: AIProvider) -> None:
        """Register a new provider."""
        self._providers[provider.name] = provider
        logger.debug("registered_ai_provider", provider=provider.name)

    def get(self, name: str) -> AIProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def list_all(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def remove(self, name: str) -> None:
        """Remove a provider from the registry."""
        if name in self._providers:
            del self._providers[name]
            logger.debug("removed_ai_provider", provider=name)

    # -- Availability & Capabilities -----------------------------------------

    async def get_available(self) -> list[AIProvider]:
        """Get all currently available providers (ordered by priority)."""
        ordered = self._order_providers()
        available: list[AIProvider] = []
        for provider in ordered:
            if await provider.is_available():
                available.append(provider)
        return available

    def meets_capabilities(
        self,
        provider: AIProvider,
        required_capabilities: Set[ProviderCapability] | None = None,
    ) -> bool:
        """Check whether a provider satisfies the required capabilities."""
        if not required_capabilities:
            return True

        caps = provider.capabilities()
        for req in required_capabilities:
            attr_name = _CAPABILITY_MAP.get(req)
            if not attr_name or not getattr(caps, attr_name, False):
                return False
        return True

    # -- Explicit Deterministic Selection Policy -----------------------------

    def _order_providers(
        self,
        preferred_provider: str | None = None,
    ) -> list[AIProvider]:
        """Sort registered providers deterministically according to explicit priority.

        1. Preferred provider (if specified and registered) is evaluated first.
        2. Providers appearing in self._priority appear in exact priority list order.
        3. Any other registered providers appear in alphabetical order.
        """
        priority_indices: dict[str, int] = {
            name: idx for idx, name in enumerate(self._priority)
        }

        def sort_key(provider: AIProvider) -> tuple[int, int, str]:
            # Priority group 0: preferred_provider
            if preferred_provider and provider.name == preferred_provider:
                return (0, 0, provider.name)
            # Priority group 1: in self._priority list
            if provider.name in priority_indices:
                return (1, priority_indices[provider.name], provider.name)
            # Priority group 2: registered but not in priority list (alphabetical)
            return (2, 0, provider.name)

        return sorted(self._providers.values(), key=sort_key)

    async def select_candidate_chain(
        self,
        required_capabilities: Set[ProviderCapability] | None = None,
        preferred_provider: str | None = None,
    ) -> list[AIProvider]:
        """Return an ordered candidate chain of available providers meeting capabilities.

        The chain order represents the fallback sequence:
        Chain[0] = Primary choice
        Chain[1] = First fallback
        Chain[2] = Second fallback, etc.
        """
        ordered = self._order_providers(preferred_provider=preferred_provider)
        candidates: list[AIProvider] = []

        for provider in ordered:
            if not self.meets_capabilities(provider, required_capabilities):
                continue
            if await provider.is_available():
                candidates.append(provider)

        return candidates

    async def select_provider(
        self,
        required_capabilities: Set[ProviderCapability] | None = None,
        preferred_provider: str | None = None,
    ) -> AIProvider | None:
        """Deterministically select the best available provider."""
        candidates = await self.select_candidate_chain(
            required_capabilities=required_capabilities,
            preferred_provider=preferred_provider,
        )
        return candidates[0] if candidates else None

    async def get_best(
        self,
        required_capabilities: Set[ProviderCapability] | None = None,
    ) -> AIProvider | None:
        """Backwards-compatible alias for select_provider."""
        return await self.select_provider(
            required_capabilities=required_capabilities,
            preferred_provider=self._default_provider,
        )


def create_provider_registry(config: AIConfig) -> ProviderRegistry:
    """Factory creating and configuring the ProviderRegistry from AIConfig.

    Decouples PixelRuntime from concrete AI provider classes.
    """
    registry = ProviderRegistry(
        priority=config.provider_priority,
        default_provider=config.default_provider,
    )

    # Register providers with their configuration
    registry.register(CloudProvider(config=config.providers.cloud))
    registry.register(OllamaProvider(config=config.providers.ollama))
    registry.register(FallbackProvider(config=config.providers.fallback))

    return registry
