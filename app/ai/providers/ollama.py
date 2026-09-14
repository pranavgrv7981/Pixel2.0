from __future__ import annotations

from typing import AsyncIterator

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.core.errors import ProviderError


class OllamaProvider(AIProvider):
    """Ollama AI provider (stub)."""

    @property
    def name(self) -> str:
        return "ollama"

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        raise ProviderError("Ollama provider not configured")

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        raise ProviderError("Ollama provider not configured")
        yield ""  # For type checking

    async def is_available(self) -> bool:
        return False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            conversation=True,
            streaming=True,
            context_capacity=8192
        )
