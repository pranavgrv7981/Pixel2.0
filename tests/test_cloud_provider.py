"""Unit tests for CloudProvider (Google Gemini SDK) with offline mocked client."""

from __future__ import annotations

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.provider import ChatMessage, ChatResponse
from app.ai.providers.cloud import CloudProvider
from app.core.config import CloudProviderConfig, AIConfig, ProvidersConfig
from app.core.errors import ProviderError
from app.ai.registry import ProviderRegistry, create_provider_registry
from app.runtime.runtime import PixelRuntime


class MockUsageMetadata:
    def __init__(self, prompt_tokens: int = 15, response_tokens: int = 25) -> None:
        self.prompt_token_count = prompt_tokens
        self.response_token_count = response_tokens


class MockCandidate:
    def __init__(self, finish_reason: str = "STOP") -> None:
        self.finish_reason = finish_reason


class MockGenerateContentResponse:
    def __init__(
        self,
        text: str = "Hello from mocked Gemini!",
        prompt_tokens: int = 15,
        response_tokens: int = 25,
        finish_reason: str = "STOP",
    ) -> None:
        self.text = text
        self.usage_metadata = MockUsageMetadata(prompt_tokens, response_tokens)
        self.candidates = [MockCandidate(finish_reason)]


class MockStreamChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class MockAsyncStream:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = [MockStreamChunk(c) for c in chunks]

    def __aiter__(self):
        self._iter = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def create_mock_client(
    generate_result: Any = None,
    stream_chunks: list[str] | None = None,
    generate_side_effect: Any = None,
    stream_side_effect: Any = None,
) -> MagicMock:
    """Helper creating a mock google-genai Client."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()

    if generate_side_effect is not None:
        client.aio.models.generate_content = AsyncMock(side_effect=generate_side_effect)
    else:
        client.aio.models.generate_content = AsyncMock(
            return_value=generate_result or MockGenerateContentResponse()
        )

    if stream_side_effect is not None:
        client.aio.models.generate_content_stream = AsyncMock(side_effect=stream_side_effect)
    else:
        client.aio.models.generate_content_stream = AsyncMock(
            return_value=MockAsyncStream(stream_chunks or ["Hello", " ", "from", " ", "stream!"])
        )

    return client


# ---------------------------------------------------------------------------
# 1. API Key & Availability Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_missing_unavailable():
    """Provider is unavailable when GEMINI_API_KEY is not set."""
    with patch.dict(os.environ, {}, clear=True):
        config = CloudProviderConfig(enabled=True, api_key="", api_key_env="GEMINI_API_KEY")
        provider = CloudProvider(config=config)
        assert await provider.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_present_and_enabled_available():
    """Provider is available when enabled and API key is present."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}):
        config = CloudProviderConfig(enabled=True, api_key_env="GEMINI_API_KEY")
        provider = CloudProvider(config=config)
        assert await provider.is_available() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_disabled_unavailable():
    """Provider is unavailable when enabled=False even if API key exists."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}):
        config = CloudProviderConfig(enabled=False, api_key_env="GEMINI_API_KEY")
        provider = CloudProvider(config=config)
        assert await provider.is_available() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_client_raises_when_key_missing():
    """Attempting chat without an API key raises ProviderError with stage='auth'."""
    with patch.dict(os.environ, {}, clear=True):
        config = CloudProviderConfig(enabled=True, api_key="")
        provider = CloudProvider(config=config)
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat([ChatMessage(role="user", content="hello")])
        assert exc_info.value.stage == "auth"
        assert "api key" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 2. Message & Response Conversion Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_message_conversion_system_and_roles():
    """System messages become system_instruction, user and assistant roles map to user/model."""
    config = CloudProviderConfig(enabled=True, api_key="fake")
    provider = CloudProvider(config=config)

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
        ChatMessage(role="assistant", content="4"),
        ChatMessage(role="user", content="Thanks!"),
    ]

    system_instruction, contents = provider._convert_messages(messages)

    assert system_instruction == "You are a helpful assistant."
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_successful_response():
    """chat() returns properly populated ChatResponse with latency and usage telemetry."""
    mock_client = create_mock_client(
        MockGenerateContentResponse(
            text="Simulated answer",
            prompt_tokens=18,
            response_tokens=32,
            finish_reason="STOP",
        )
    )
    config = CloudProviderConfig(enabled=True, api_key="fake", model="gemini-2.5-flash")
    provider = CloudProvider(config=config, client=mock_client)

    response = await provider.chat([ChatMessage(role="user", content="hello")])

    assert isinstance(response, ChatResponse)
    assert response.content == "Simulated answer"
    assert response.model == "gemini-2.5-flash"
    assert response.provider == "cloud"
    assert response.input_tokens == 18
    assert response.output_tokens == 32
    assert response.finish_reason == "stop"
    assert response.latency_ms > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_missing_usage_metadata_safe():
    """chat() handles responses where usage_metadata is None without crashing."""
    mock_resp = MockGenerateContentResponse(text="No tokens response")
    mock_resp.usage_metadata = None
    mock_client = create_mock_client(mock_resp)

    config = CloudProviderConfig(enabled=True, api_key="fake")
    provider = CloudProvider(config=config, client=mock_client)

    response = await provider.chat([ChatMessage(role="user", content="hello")])
    assert response.content == "No tokens response"
    assert response.input_tokens == 0
    assert response.output_tokens == 0


# ---------------------------------------------------------------------------
# 3. Streaming Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_yields_incremental_chunks():
    """stream_chat() yields text chunks incrementally as they arrive."""
    chunks = ["Pixel ", "is ", "intelligent."]
    mock_client = create_mock_client(stream_chunks=chunks)

    config = CloudProviderConfig(enabled=True, api_key="fake")
    provider = CloudProvider(config=config, client=mock_client)

    received = []
    async for chunk in provider.stream_chat([ChatMessage(role="user", content="hi")]):
        received.append(chunk)

    assert received == chunks


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_skips_empty_chunks():
    """stream_chat() skips empty text chunks without yielding empty strings."""
    chunks = ["first", "", "second", None, "third"]
    mock_stream = MockAsyncStream(["first", "", "second"])
    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream)

    config = CloudProviderConfig(enabled=True, api_key="fake")
    provider = CloudProvider(config=config, client=mock_client)

    received = [c async for c in provider.stream_chat([ChatMessage(role="user", content="hi")])]
    assert received == ["first", "second"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_error_wrapped_in_provider_error():
    """An exception during streaming is converted into ProviderError with stage='stream'."""
    async def failing_stream():
        yield MockStreamChunk("part1")
        raise RuntimeError("Network dropped mid-stream")

    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=failing_stream())

    config = CloudProviderConfig(enabled=True, api_key="fake")
    provider = CloudProvider(config=config, client=mock_client)

    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role="user", content="hi")]):
            pass
    assert exc_info.value.stage == "stream"


# ---------------------------------------------------------------------------
# 4. Timeout & Retry Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_timeout_protection():
    """A hanging request times out after timeout_seconds and raises ProviderError."""
    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(2.0)
        return MockGenerateContentResponse()

    mock_client = create_mock_client(generate_side_effect=slow_generate)
    config = CloudProviderConfig(enabled=True, api_key="fake", timeout_seconds=0.05, max_retries=0)
    provider = CloudProvider(config=config, client=mock_client)

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat([ChatMessage(role="user", content="slow")])

    assert exc_info.value.stage == "timeout"
    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transient_error_retries_and_succeeds():
    """Transient errors (e.g. 503) retry up to max_retries and succeed when resolved."""
    class Mock503Error(Exception):
        status_code = 503

    attempts = 0

    async def transient_generate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise Mock503Error("Service Temporarily Unavailable")
        return MockGenerateContentResponse(text="Success after retry")

    mock_client = create_mock_client(generate_side_effect=transient_generate)
    config = CloudProviderConfig(enabled=True, api_key="fake", max_retries=2)
    provider = CloudProvider(config=config, client=mock_client)

    response = await provider.chat([ChatMessage(role="user", content="retry test")])
    assert attempts == 2
    assert response.content == "Success after retry"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_permanent_error_fails_immediately():
    """Non-transient errors (400 Bad Request) fail on first attempt without retrying."""
    class Mock400Error(Exception):
        status_code = 400

    attempts = 0

    async def bad_request_generate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise Mock400Error("Invalid argument / malformed request")

    mock_client = create_mock_client(generate_side_effect=bad_request_generate)
    config = CloudProviderConfig(enabled=True, api_key="fake", max_retries=2)
    provider = CloudProvider(config=config, client=mock_client)

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="bad request")])

    assert attempts == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auth_error_marks_unavailable():
    """A 401 Authentication error marks _auth_failed=True, disabling subsequent availability."""
    class Mock401Error(Exception):
        status_code = 401

    mock_client = create_mock_client(generate_side_effect=Mock401Error("API_KEY_INVALID"))
    config = CloudProviderConfig(enabled=True, api_key="bad_key", max_retries=0)
    provider = CloudProvider(config=config, client=mock_client)

    assert await provider.is_available() is True
    with pytest.raises(ProviderError) as exc_info:
        await provider.chat([ChatMessage(role="user", content="test")])

    assert exc_info.value.stage == "auth"
    # Availability is now marked False to fail fast
    assert await provider.is_available() is False


@pytest.mark.unit
def test_no_credential_leakage():
    """Sanitizer ensures secret API key is redacted from exception messages."""
    secret_key = "AIzaSySecretApiKey12345"
    config = CloudProviderConfig(enabled=True, api_key=secret_key)
    provider = CloudProvider(config=config)

    raw_exc = Exception(f"HTTP connection failed with key: {secret_key}")
    sanitized = provider._sanitize_error(raw_exc, model="gemini", stage="test")

    err_str = str(sanitized)
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


# ---------------------------------------------------------------------------
# 5. Registry & Runtime Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_selects_cloud_when_available():
    """In PixelRuntime, when cloud is configured and available, it is chosen over fallback."""
    mock_client = create_mock_client(MockGenerateContentResponse(text="Live cloud answer"))
    cloud_provider = CloudProvider(
        config=CloudProviderConfig(enabled=True, api_key="valid_key"),
        client=mock_client,
    )

    reg = ProviderRegistry(priority=["cloud", "ollama", "fallback"])
    reg.register(cloud_provider)

    from app.ai.providers.fallback import FallbackProvider
    reg.register(FallbackProvider())

    selected = await reg.select_provider()
    assert selected is not None
    assert selected.name == "cloud"

    from app.core.config import PixelConfig
    runtime = PixelRuntime(PixelConfig(), provider_registry=reg)
    await runtime.start()

    response = await runtime.handle_input("explain machine learning")
    assert response == "Live cloud answer"
    await runtime.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_falls_back_when_cloud_fails():
    """In PixelRuntime, when cloud raises an exception, execution automatically falls back to secondary."""
    mock_client = create_mock_client(generate_side_effect=RuntimeError("Cloud service down"))
    cloud_provider = CloudProvider(
        config=CloudProviderConfig(enabled=True, api_key="valid_key", max_retries=0),
        client=mock_client,
    )

    reg = ProviderRegistry(priority=["cloud", "fallback"])
    reg.register(cloud_provider)

    from app.ai.providers.fallback import FallbackProvider
    reg.register(FallbackProvider())

    from app.core.config import PixelConfig
    runtime = PixelRuntime(PixelConfig(), provider_registry=reg)
    await runtime.start()

    # Cloud fails, runtime catches it and falls back to FallbackProvider
    response = await runtime.handle_input("explain black holes")
    assert "fallback mode" in response.lower()
    assert runtime.state.value == "IDLE"

    await runtime.stop()
