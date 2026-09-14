from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator
from pydantic import BaseModel


class ProviderCapabilities(BaseModel):
    """Capabilities supported by an AI provider."""
    conversation: bool = False
    tool_calling: bool = False
    reasoning: bool = False
    vision: bool = False
    structured_output: bool = False
    streaming: bool = False
    context_capacity: int = 4096


class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: str
    content: str


class ChatResponse(BaseModel):
    """Response from an AI provider."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider."""
        pass

    @abstractmethod
    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def stream_chat(self, messages: list[ChatMessage], **kwargs) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is currently available."""
        pass

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Get the capabilities of the provider."""
        pass
