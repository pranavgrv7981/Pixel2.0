"""Optional live OpenRouter API smoke test.

Only executed when explicitly requested with:
    pytest tests/ -m smoke -v
and when the OPENROUTER_API_KEY environment variable is present.
"""

from __future__ import annotations

import os
import pytest

from app.ai.provider import ChatMessage, ChatResponse
from app.ai.providers.cloud import CloudProvider
from app.core.config import CloudProviderConfig

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not OPENROUTER_API_KEY,
        reason="OPENROUTER_API_KEY environment variable not set; skipping live cloud smoke tests",
    ),
]


@pytest.mark.asyncio
async def test_live_openrouter_chat_smoke():
    """Verify live chat generation using OpenRouter API and default free model."""
    config = CloudProviderConfig(
        enabled=True,
        api_key_env="OPENROUTER_API_KEY",
        model="openrouter/free",
        timeout_seconds=30.0,
    )
    provider = CloudProvider(config=config)
    assert await provider.is_available() is True

    messages = [
        ChatMessage(role="user", content="Reply with exactly: PIXEL_PROVIDER_OK"),
    ]
    response = await provider.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.content is not None
    assert "PIXEL_PROVIDER_OK" in response.content.upper()
    assert response.provider == "cloud"
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_live_openrouter_streaming_smoke():
    """Verify incremental SSE streaming from live OpenRouter API."""
    config = CloudProviderConfig(
        enabled=True,
        api_key_env="OPENROUTER_API_KEY",
        model="openrouter/free",
        timeout_seconds=30.0,
    )
    provider = CloudProvider(config=config)

    messages = [
        ChatMessage(role="user", content="Count from 1 to 3, one number per line."),
    ]

    chunks: list[str] = []
    async for chunk in provider.stream_chat(messages):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "1" in full_text or "2" in full_text or "3" in full_text
