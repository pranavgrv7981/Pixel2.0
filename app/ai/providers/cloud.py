"""Cloud AI provider implementation using OpenRouter HTTP API.

Implements the AIProvider interface for OpenRouter with:
- Direct, lightweight HTTP integration via httpx
- Async chat generation and SSE streaming
- Multi-turn conversation conversion
- Bounded transient retries and explicit timeouts
- Non-expensive availability checks
- Token and latency telemetry
- Zero credential leakage
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.core.config import CloudProviderConfig
from app.core.errors import ProviderError
from app.core.logging import get_logger

logger = get_logger("pixel.ai.cloud")


class CloudProvider(AIProvider):
    """OpenRouter Cloud AI provider."""

    def __init__(
        self,
        config: CloudProviderConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or CloudProviderConfig()
        self._http_client = http_client
        self._auth_failed: bool = False

    @property
    def name(self) -> str:
        return "cloud"

    def capabilities(self) -> ProviderCapabilities:
        """Declared capabilities of the OpenRouter provider.

        Conservative for Phase 1: conversation and streaming.
        """
        return ProviderCapabilities(
            conversation=True,
            tool_calling=False,
            reasoning=False,
            vision=False,
            structured_output=False,
            streaming=True,
            context_capacity=128000,
        )

    # -- Availability --------------------------------------------------------

    async def is_available(self) -> bool:
        """Fast, non-expensive availability check.

        Does not perform network requests on application startup.
        """
        if not self.config.enabled:
            return False
        if self._auth_failed:
            return False
        api_key = self.config.get_api_key()
        return bool(api_key and api_key.strip())

    # -- HTTP Client & Headers -----------------------------------------------

    def _get_headers(self) -> dict[str, str]:
        """Construct request headers with authorization."""
        api_key = self.config.get_api_key()
        if not api_key:
            self._auth_failed = True
            raise ProviderError(
                "Cloud provider API key is missing or not configured",
                provider=self.name,
                model=self.config.get_model(),
                stage="auth",
                context={"error_type": "MissingCredentials", "env_var": self.config.api_key_env},
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.x_title:
            headers["X-Title"] = self.config.x_title

        return headers

    def _get_endpoint_url(self) -> str:
        """Get the full completions endpoint URL."""
        base = (self.config.base_url or "https://openrouter.ai/api/v1").rstrip("/")
        return f"{base}/chat/completions"

    def _get_client(self) -> httpx.AsyncClient:
        """Return injected client or new AsyncClient."""
        if self._http_client is not None:
            return self._http_client
        return httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout_seconds))

    # -- Message Conversion --------------------------------------------------

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert Pixel ChatMessage sequence into OpenAI-compatible message dictionaries."""
        payload_messages: list[dict[str, str]] = []

        for msg in messages:
            role = (msg.role or "user").lower().strip()
            content = msg.content or ""

            if role in ("assistant", "model", "bot"):
                target_role = "assistant"
            elif role in ("system", "developer"):
                target_role = "system"
            else:
                target_role = "user"

            payload_messages.append({"role": target_role, "content": content})

        if not payload_messages:
            payload_messages.append({"role": "user", "content": " "})

        return payload_messages

    # -- Error Classification & Safety ---------------------------------------

    def _is_transient_error(self, exc: Exception) -> bool:
        """Determine if an error is transient and eligible for retry."""
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code in (400, 401, 403, 404):
                return False
            if code in (429, 500, 502, 503, 504):
                return True

        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True

        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status_code in (400, 401, 403, 404):
            return False
        if status_code in (429, 500, 502, 503, 504):
            return True

        return False

    def _sanitize_error(self, exc: Exception, model: str, stage: str) -> ProviderError:
        """Create a safe ProviderError without leaking credentials."""
        api_key = self.config.get_api_key()
        raw_msg = str(exc)
        if api_key and api_key in raw_msg:
            raw_msg = raw_msg.replace(api_key, "[REDACTED_API_KEY]")

        status_code = None
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
        else:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)

        error_type = type(exc).__name__

        if status_code in (401, 403) or "unauthenticated" in raw_msg.lower() or "unauthorized" in raw_msg.lower():
            self._auth_failed = True
            stage = "auth"
            safe_msg = f"Cloud AI authentication failed: {raw_msg}"
        elif status_code == 429 or "rate limit" in raw_msg.lower():
            stage = "rate_limit"
            safe_msg = f"Cloud AI rate limit or quota exceeded: {raw_msg}"
        elif isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
            stage = "timeout"
            safe_msg = f"Cloud AI request timed out after {self.config.timeout_seconds}s."
        else:
            safe_msg = f"Cloud AI request failed during {stage}: {raw_msg}"

        return ProviderError(
            safe_msg,
            provider=self.name,
            model=model,
            stage=stage,
            context={
                "error_type": error_type,
                "status_code": status_code,
                "timeout_seconds": self.config.timeout_seconds,
            },
        )

    # -- Chat Generation -----------------------------------------------------

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Execute a non-streaming chat request with timeout and transient retries."""
        model = kwargs.get("model") or self.config.get_model()
        url = self._get_endpoint_url()
        headers = self._get_headers()
        payload_messages = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
        }
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        max_retries = self.config.max_retries
        timeout = httpx.Timeout(self.config.timeout_seconds)
        start_time = time.perf_counter()

        client = self._get_client()
        should_close_client = self._http_client is None

        try:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                    response.raise_for_status()

                    data = response.json()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    # Extract content
                    choices = data.get("choices", [])
                    content = choices[0].get("message", {}).get("content", "") if choices else ""

                    # Extract token telemetry
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0) or 0
                    output_tokens = usage.get("completion_tokens", 0) or 0

                    finish_reason = "stop"
                    if choices:
                        raw_reason = choices[0].get("finish_reason")
                        if raw_reason is not None:
                            finish_reason = str(raw_reason).lower()

                    resolved_model = data.get("model", model)

                    logger.info(
                        "cloud_ai_completed",
                        provider=self.name,
                        model=resolved_model,
                        duration_ms=round(elapsed_ms, 2),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        finish_reason=finish_reason,
                        attempts=attempt + 1,
                    )

                    return ChatResponse(
                        content=content,
                        model=resolved_model,
                        provider=self.name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=elapsed_ms,
                        finish_reason=finish_reason,
                    )

                except Exception as exc:
                    is_transient = self._is_transient_error(exc)
                    has_retry = is_transient and (attempt < max_retries)

                    logger.warning(
                        "cloud_ai_attempt_failed",
                        provider=self.name,
                        model=model,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(exc),
                        will_retry=has_retry,
                    )

                    if has_retry:
                        delay = min(0.25 * (2 ** attempt), 2.0)
                        await asyncio.sleep(delay)
                        continue

                    raise self._sanitize_error(exc, model=model, stage="chat") from exc

            raise ProviderError("Maximum retries exhausted", provider=self.name, model=model, stage="chat")
        finally:
            if should_close_client:
                await client.aclose()

    # -- Streaming Chat ------------------------------------------------------

    async def stream_chat(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream incremental text response chunks from OpenRouter SSE."""
        model = kwargs.get("model") or self.config.get_model()
        url = self._get_endpoint_url()
        headers = self._get_headers()
        payload_messages = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
        }
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        timeout = httpx.Timeout(self.config.timeout_seconds)
        client = self._get_client()
        should_close_client = self._http_client is None

        try:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=timeout) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    err_msg = f"HTTP {response.status_code}: {body.decode('utf-8', errors='replace')}"
                    raise httpx.HTTPStatusError(err_msg, request=response.request, response=response)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        # Skip empty lines or SSE ping/comments
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: "):].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                text_chunk = delta.get("content")
                                if text_chunk:
                                    yield text_chunk
                        except json.JSONDecodeError:
                            # Skip unparseable chunk
                            continue

        except Exception as exc:
            logger.error("cloud_ai_stream_failed", provider=self.name, model=model, error=str(exc))
            raise self._sanitize_error(exc, model=model, stage="stream") from exc
        finally:
            if should_close_client:
                await client.aclose()
