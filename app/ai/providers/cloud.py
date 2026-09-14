from __future__ import annotations

from typing import AsyncIterator

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.core.errors import ProviderError


class CloudProvider(AIProvider):
    """Cloud AI provider (stub)."""

    @property
    def name(self) -> str:
        return "cloud"

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        raise ProviderError("Cloud provider not implemented")

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        raise ProviderError("Cloud provider not implemented")
        yield ""  # For type checking

    async def is_available(self) -> bool:
        return False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            conversation=True,
            tool_calling=True,
            reasoning=True,
            vision=True,
            structured_output=True,
            streaming=True,
            context_capacity=128000
        )
