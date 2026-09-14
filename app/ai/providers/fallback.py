from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities


from app.core.config import FallbackProviderConfig


class FallbackProvider(AIProvider):
    """Fallback AI provider."""

    def __init__(self, config: FallbackProviderConfig | None = None) -> None:
        self.config = config or FallbackProviderConfig()

    @property
    def name(self) -> str:
        return "fallback"
        
    def _canned_message(self) -> str:
        return "I am running in fallback mode. AI providers are not configured. I can still help with direct commands like time, date, and calculations."

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        return ChatResponse(
            content=self._canned_message(),
            model="fallback-model",
            provider=self.name,
        )

    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        msg = self._canned_message()
        words = msg.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)

    async def is_available(self) -> bool:
        return bool(self.config.enabled)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            conversation=True,
            streaming=True,
            context_capacity=2048
        )
