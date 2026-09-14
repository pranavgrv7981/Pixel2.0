"""Unit tests for OpenRouter CloudProvider with offline mocked HTTP transport."""

from __future__ import annotations

import asyncio
import json
import os
import pytest
import httpx

from app.ai.provider import ChatMessage, ChatResponse
from app.ai.providers.cloud import CloudProvider
from app.core.config import CloudProviderConfig, PixelConfig
from app.core.errors import ProviderError
from app.ai.registry import ProviderRegistry
from app.runtime.runtime import PixelRuntime


def make_mock_client(handler) -> httpx.AsyncClient:
    """Create an AsyncClient with custom MockTransport."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


# ---------------------------------------------------------------------------
# 1. API Key & Availability Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_disabled_unavailable():
    """Provider is unavailable when enabled=False even if API key exists."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENROUTER_API_KEY", "sk-or-fake-key")
        config = CloudProviderConfig(enabled=False)
        provider = CloudProvider(config=config)
        assert await provider.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_api_key_unavailable():
    """Provider is unavailable when OPENROUTER_API_KEY is not set."""
    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("OPENROUTER_API_KEY", raising=False)
        config = CloudProviderConfig(enabled=True, api_key="")
        provider = CloudProvider(config=config)
        assert await provider.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_availability():
    """Provider is available when enabled=True and API key is set."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENROUTER_API_KEY", "sk-or-fake-key")
        config = CloudProviderConfig(enabled=True)
        provider = CloudProvider(config=config)
        assert await provider.is_available() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_key_chat_raises_auth_error():
    """Attempting chat without key raises ProviderError with stage='auth'."""
    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("OPENROUTER_API_KEY", raising=False)
        config = CloudProviderConfig(enabled=True, api_key="")
        provider = CloudProvider(config=config)
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat([ChatMessage(role="user", content="hi")])
        assert exc_info.value.stage == "auth"
        assert "api key" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 2. Chat & Response Conversion Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_non_streaming_response():
    """Successful chat() parses choices, content, usage telemetry, and model."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer sk-test-key"
        assert request.headers.get("content-type") == "application/json"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "openrouter/free"
        assert body["messages"] == [{"role": "user", "content": "Hello Pixel"}]

        resp_body = {
            "id": "gen-12345",
            "model": "meta-llama/llama-3-8b-instruct:free",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello! I am ready to help."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }
        return httpx.Response(200, json=resp_body)

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="sk-test-key", model="openrouter/free")
    provider = CloudProvider(config=config, http_client=client)

    response = await provider.chat([ChatMessage(role="user", content="Hello Pixel")])

    assert isinstance(response, ChatResponse)
    assert response.content == "Hello! I am ready to help."
    assert response.model == "meta-llama/llama-3-8b-instruct:free"
    assert response.provider == "cloud"
    assert response.input_tokens == 12
    assert response.output_tokens == 8
    assert response.finish_reason == "stop"
    assert response.latency_ms > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_conversion_missing_usage_safe():
    """chat() handles responses lacking usage telemetry cleanly (defaults to 0)."""
    def handler(request: httpx.Request) -> httpx.Response:
        resp_body = {
            "choices": [{"message": {"content": "Answer without usage"}}],
        }
        return httpx.Response(200, json=resp_body)

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="sk-test-key")
    provider = CloudProvider(config=config, http_client=client)

    response = await provider.chat([ChatMessage(role="user", content="hi")])
    assert response.content == "Answer without usage"
    assert response.input_tokens == 0
    assert response.output_tokens == 0


@pytest.mark.unit
def test_message_conversion_roles():
    """Conversion supports system, user, assistant/model roles into OpenAI format."""
    config = CloudProviderConfig(enabled=True, api_key="test")
    provider = CloudProvider(config=config)

    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="model", content="Hi!"),
        ChatMessage(role="assistant", content="How can I help?"),
    ]
    converted = provider._convert_messages(messages)

    assert converted == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "assistant", "content": "How can I help?"},
    ]


# ---------------------------------------------------------------------------
# 3. Streaming Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_chunks_and_done_handling():
    """stream_chat() parses SSE chunks, ignores comments, and stops on [DONE]."""
    sse_data = (
        ": keep-alive\n\n"
        'data: {"choices": [{"delta": {"content": "Pixel "}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "is "}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "streaming!"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        return httpx.Response(200, text=sse_data, headers={"content-type": "text/event-stream"})

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key")
    provider = CloudProvider(config=config, http_client=client)

    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role="user", content="stream test")]):
        chunks.append(chunk)

    assert chunks == ["Pixel ", "is ", "streaming!"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_malformed_chunks_handled_safely():
    """stream_chat() ignores malformed JSON SSE lines without crashing."""
    sse_data = (
        'data: {"choices": [{"delta": {"content": "Valid part 1"}}]}\n\n'
        "data: {not valid json\n\n"
        'data: {"choices": [{"delta": {"content": " Valid part 2"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_data, headers={"content-type": "text/event-stream"})

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key")
    provider = CloudProvider(config=config, http_client=client)

    chunks = [c async for c in provider.stream_chat([ChatMessage(role="user", content="test")])]
    assert chunks == ["Valid part 1", " Valid part 2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_http_error():
    """Non-200 streaming response raises ProviderError with stage='stream'."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Rate limit exceeded")

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key")
    provider = CloudProvider(config=config, http_client=client)

    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role="user", content="hi")]):
            pass

    assert exc_info.value.stage == "rate_limit"


# ---------------------------------------------------------------------------
# 4. Timeout & Error Handling Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_raises_provider_error():
    """Connection timeout raises ProviderError with stage='timeout'."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out after 30 seconds")

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key", max_retries=0)
    provider = CloudProvider(config=config, http_client=client)

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat([ChatMessage(role="user", content="timeout test")])

    assert exc_info.value.stage == "timeout"
    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_network_failure():
    """Network connection failure raises ProviderError with stage='chat'."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect to openrouter.ai")

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key", max_retries=0)
    provider = CloudProvider(config=config, http_client=client)

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat([ChatMessage(role="user", content="net fail")])

    assert exc_info.value.stage == "chat"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_429_retries_and_succeeds():
    """HTTP 429 rate limit triggers transient retry and succeeds on next attempt."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "Rate limit reached"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Success on retry"}}]})

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key", max_retries=2)
    provider = CloudProvider(config=config, http_client=client)

    response = await provider.chat([ChatMessage(role="user", content="retry test")])
    assert attempts == 2
    assert response.content == "Success on retry"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_500_retries_and_exhausts():
    """HTTP 500 server error retries up to max_retries then raises ProviderError."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "Internal server error"}})

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key", max_retries=2)
    provider = CloudProvider(config=config, http_client=client)

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="500 test")])

    assert attempts == 3  # 1 initial + 2 retries


@pytest.mark.unit
@pytest.mark.asyncio
async def test_authentication_failure_no_retry():
    """HTTP 401 fails immediately without retrying and marks _auth_failed=True."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="invalid-key", max_retries=2)
    provider = CloudProvider(config=config, http_client=client)

    assert await provider.is_available() is True
    with pytest.raises(ProviderError) as exc_info:
        await provider.chat([ChatMessage(role="user", content="auth test")])

    assert attempts == 1  # No retries on 401!
    assert exc_info.value.stage == "auth"
    assert await provider.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_model_request_no_retry():
    """HTTP 400 Bad Request fails on first attempt without retrying."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"message": "Model not supported"}})

    client = make_mock_client(handler)
    config = CloudProviderConfig(enabled=True, api_key="test-key", max_retries=2)
    provider = CloudProvider(config=config, http_client=client)

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="invalid model")])

    assert attempts == 1


@pytest.mark.unit
def test_no_credential_leakage():
    """Sanitizer redacts secret API key from exception messages."""
    secret_key = "sk-or-v1-secret-123456789abcdef"
    config = CloudProviderConfig(enabled=True, api_key=secret_key)
    provider = CloudProvider(config=config)

    raw_exc = Exception(f"Connection failed using key: {secret_key}")
    sanitized = provider._sanitize_error(raw_exc, model="openrouter/free", stage="chat")

    err_str = str(sanitized)
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


# ---------------------------------------------------------------------------
# 5. Registry & Runtime Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_priority_and_order_independence():
    """Registry priority is respected regardless of whether cloud was registered first or second."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "cloud answer"}}]})

    cloud = CloudProvider(
        config=CloudProviderConfig(enabled=True, api_key="test-key"),
        http_client=make_mock_client(handler),
    )

    from app.ai.providers.fallback import FallbackProvider
    fallback = FallbackProvider()

    # Order 1: fallback then cloud
    reg1 = ProviderRegistry(priority=["cloud", "fallback"])
    reg1.register(fallback)
    reg1.register(cloud)

    # Order 2: cloud then fallback
    reg2 = ProviderRegistry(priority=["cloud", "fallback"])
    reg2.register(cloud)
    reg2.register(fallback)

    sel1 = await reg1.select_provider()
    sel2 = await reg2.select_provider()

    assert sel1.name == sel2.name == "cloud"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_fallback_when_cloud_fails():
    """When OpenRouter raises an exception, PixelRuntime falls back to FallbackProvider."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("OpenRouter unreachable")

    cloud = CloudProvider(
        config=CloudProviderConfig(enabled=True, api_key="test-key", max_retries=0),
        http_client=make_mock_client(handler),
    )

    reg = ProviderRegistry(priority=["cloud", "fallback"])
    reg.register(cloud)
    from app.ai.providers.fallback import FallbackProvider
    reg.register(FallbackProvider())

    runtime = PixelRuntime(PixelConfig(), provider_registry=reg)
    await runtime.start()

    response = await runtime.handle_input("explain black holes in detail")
    assert "fallback mode" in response.lower()
    assert runtime.state.value == "IDLE"

    await runtime.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_continues_when_cloud_unavailable():
    """When cloud is disabled / missing key, Pixel starts normally and uses fallback."""
    cfg = PixelConfig()
    cfg.ai.providers.cloud.enabled = False

    runtime = PixelRuntime(cfg)
    await runtime.start()

    assert runtime.running is True
    assert runtime.state.value == "IDLE"

    response = await runtime.handle_input("what is quantum computing?")
    assert "fallback mode" in response.lower()

    await runtime.stop()
