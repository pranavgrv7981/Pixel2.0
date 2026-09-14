from __future__ import annotations

from typing import AsyncIterator

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.core.errors import ProviderError


from app.core.config import OllamaProviderConfig


class OllamaProvider(AIProvider):
    """Ollama AI provider (stub)."""

    def __init__(self, config: OllamaProviderConfig | None = None) -> None:
        self.config = config or OllamaProviderConfig()

    @property
    def name(self) -> str:
        return "ollama"

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        raise ProviderError("Ollama provider not configured", provider=self.name)

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        raise ProviderError("Ollama provider not configured", provider=self.name)
        yield ""  # For type checking

    async def is_available(self) -> bool:
        return bool(self.config.enabled)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            conversation=True,
            streaming=True,
            context_capacity=8192
        )
