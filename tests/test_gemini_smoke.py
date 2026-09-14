"""Optional live cloud API smoke test for Gemini integration.

Only executed when explicitly requested with:
    pytest tests/ -m smoke -v
and when the GEMINI_API_KEY environment variable is present.
"""

from __future__ import annotations

import os
import pytest

from app.ai.provider import ChatMessage, ChatResponse
from app.ai.providers.cloud import CloudProvider
from app.core.config import CloudProviderConfig

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not GEMINI_API_KEY,
        reason="GEMINI_API_KEY environment variable not set; skipping live cloud smoke tests",
    ),
]


@pytest.mark.asyncio
async def test_live_gemini_chat_smoke():
    """Verify end-to-end chat with live Gemini API."""
    config = CloudProviderConfig(
        enabled=True,
        api_key_env="GEMINI_API_KEY",
        model="gemini-2.5-flash",
        timeout_seconds=20.0,
    )
    provider = CloudProvider(config=config)
    assert await provider.is_available() is True

    messages = [
        ChatMessage(role="system", content="You are a terse assistant."),
        ChatMessage(role="user", content="Reply with exactly the word PONG."),
    ]
    response = await provider.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.content is not None
    assert "PONG" in response.content.upper()
    assert response.provider == "cloud"
    assert response.input_tokens > 0
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_live_gemini_streaming_smoke():
    """Verify incremental streaming from live Gemini API."""
    config = CloudProviderConfig(
        enabled=True,
        api_key_env="GEMINI_API_KEY",
        model="gemini-2.5-flash",
        timeout_seconds=20.0,
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
    assert "1" in full_text and "2" in full_text and "3" in full_text
