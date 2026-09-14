"""Cloud AI provider implementation using Google Gemini SDK (google-genai).

Implements the AIProvider interface for Google Gemini models with:
- Async chat generation and streaming
- Multi-turn conversation conversion
- System instruction support
- Bounded transient retries and explicit timeouts
- Non-expensive availability checks
- Token and latency telemetry
- Zero credential leakage
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, AsyncIterator

from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.core.config import CloudProviderConfig
from app.core.errors import ProviderError
from app.core.logging import get_logger

logger = get_logger("pixel.ai.cloud")


class CloudProvider(AIProvider):
    """Google Gemini cloud AI provider."""

    def __init__(
        self,
        config: CloudProviderConfig | None = None,
        client: Any = None,
    ) -> None:
        self.config = config or CloudProviderConfig()
        self._client = client
        self._auth_failed: bool = False

    @property
    def name(self) -> str:
        return "cloud"

    def capabilities(self) -> ProviderCapabilities:
        """Declared capabilities of the Gemini cloud provider."""
        return ProviderCapabilities(
            conversation=True,
            tool_calling=True,
            reasoning=True,
            vision=True,
            structured_output=True,
            streaming=True,
            context_capacity=1048576,
        )

    # -- Availability --------------------------------------------------------

    async def is_available(self) -> bool:
        """Fast, non-expensive availability check.

        Does not perform live generation requests during health check.
        """
        if not self.config.enabled:
            return False
        if self._auth_failed:
            return False
        api_key = self.config.get_api_key()
        return bool(api_key and api_key.strip())

    # -- Client Management ---------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily initialize and return the google-genai Client."""
        if self._client is not None:
            return self._client

        api_key = self.config.get_api_key()
        if not api_key:
            self._auth_failed = True
            raise ProviderError(
                "Cloud provider API key is not configured",
                provider=self.name,
                model=self.config.get_model(),
                stage="auth",
                context={"error_type": "MissingCredentials", "env_var": self.config.api_key_env},
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception as exc:
            logger.error("cloud_client_init_failed", error=str(exc))
            raise ProviderError(
                "Failed to initialize cloud AI client",
                provider=self.name,
                model=self.config.get_model(),
                stage="init",
                context={"error_type": type(exc).__name__},
            ) from exc

    # -- Message Conversion --------------------------------------------------

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[Any]]:
        """Convert Pixel ChatMessage sequence into Gemini system instruction and contents.

        Returns:
            (system_instruction, contents_list)
        """
        from google.genai import types

        system_parts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            role = (msg.role or "user").lower().strip()
            content_text = msg.content or ""

            if role in ("system", "developer"):
                if content_text.strip():
                    system_parts.append(content_text.strip())
            elif role in ("assistant", "model", "bot"):
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content_text)],
                    )
                )
            else:
                # user / human / default
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content_text)],
                    )
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        # Gemini requires at least one user content item if contents list is empty
        if not contents:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=" ")],
                )
            )

        return system_instruction, contents

    # -- Error Classification & Safety ---------------------------------------

    def _is_transient_error(self, exc: Exception) -> bool:
        """Classify whether an exception is a transient error eligible for retry."""
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

        # Permanent status codes — NEVER retry
        if status_code in (400, 401, 403, 404):
            return False

        # Transient status codes
        if status_code in (429, 500, 502, 503, 504):
            return True

        exc_str = str(exc).lower()
        if any(term in exc_str for term in ["rate limit", "resource_exhausted", "unavailable", "deadline", "timeout"]):
            return True

        error_type = type(exc).__name__.lower()
        if any(term in error_type for term in ["timeout", "connect", "network", "servererror"]):
            return True

        return False

    def _sanitize_error(self, exc: Exception, model: str, stage: str) -> ProviderError:
        """Create a safe ProviderError with structured diagnostics without leaking credentials."""
        api_key = self.config.get_api_key()
        raw_msg = str(exc)
        if api_key and api_key in raw_msg:
            raw_msg = raw_msg.replace(api_key, "[REDACTED_API_KEY]")

        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        error_type = type(exc).__name__

        if status_code in (401, 403) or "unauthenticated" in raw_msg.lower():
            self._auth_failed = True
            stage = "auth"
            safe_msg = f"Cloud AI authentication failed: {raw_msg}"
        elif status_code == 429 or "resource_exhausted" in raw_msg.lower():
            stage = "rate_limit"
            safe_msg = f"Cloud AI rate limit or quota exceeded: {raw_msg}"
        elif isinstance(exc, asyncio.TimeoutError):
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
        from google.genai import types

        model = kwargs.get("model") or self.config.get_model()
        client = self._get_client()
        system_instruction, contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=kwargs.get("temperature", 0.7),
        )
        if "max_output_tokens" in kwargs:
            config.max_output_tokens = kwargs["max_output_tokens"]

        max_retries = self.config.max_retries
        timeout_seconds = self.config.timeout_seconds
        start_time = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                # Wrap API call with explicit timeout
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=timeout_seconds,
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Extract content
                content_text = getattr(response, "text", None) or ""

                # Extract token usage telemetry safely
                input_tokens = 0
                output_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = (
                        getattr(usage, "response_token_count", 0)
                        or getattr(usage, "candidates_token_count", 0)
                        or 0
                    )

                # Extract finish reason
                finish_reason = "stop"
                candidates = getattr(response, "candidates", None)
                if candidates and len(candidates) > 0:
                    cand = candidates[0]
                    raw_reason = getattr(cand, "finish_reason", None)
                    if raw_reason is not None:
                        finish_reason = str(raw_reason).lower()

                logger.info(
                    "cloud_ai_completed",
                    provider=self.name,
                    model=model,
                    duration_ms=round(elapsed_ms, 2),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=finish_reason,
                    attempts=attempt + 1,
                )

                return ChatResponse(
                    content=content_text,
                    model=model,
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
                    # Bounded exponential backoff
                    delay = min(0.25 * (2 ** attempt), 2.0)
                    await asyncio.sleep(delay)
                    continue

                # Not retryable or retries exhausted
                raise self._sanitize_error(exc, model=model, stage="chat") from exc

        # Safeguard fallback
        raise ProviderError("Maximum retries exhausted", provider=self.name, model=model, stage="chat")

    # -- Streaming Chat ------------------------------------------------------

    async def stream_chat(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream incremental text response chunks from Gemini."""
        from google.genai import types

        model = kwargs.get("model") or self.config.get_model()
        client = self._get_client()
        system_instruction, contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=kwargs.get("temperature", 0.7),
        )
        if "max_output_tokens" in kwargs:
            config.max_output_tokens = kwargs["max_output_tokens"]

        timeout_seconds = self.config.timeout_seconds

        try:
            # Obtain async generator stream
            stream = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_seconds,
            )

            async for chunk in stream:
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    yield chunk_text

        except Exception as exc:
            logger.error("cloud_ai_stream_failed", provider=self.name, model=model, error=str(exc))
            raise self._sanitize_error(exc, model=model, stage="stream") from exc
