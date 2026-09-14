"""PixelRuntime — Central coordinator for Pixel v2.

Owns the lifecycle of all subsystems. Delegates all business logic
to specialized components. Does not contain feature implementation itself.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.core.config import PixelConfig
from app.core.errors import PixelError
from app.core.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
    setup_logging,
)
from app.core.types import PixelStateEnum, Route

from app.events.bus import EventBus
from app.events.types import (
    AIResponseCompleted,
    IntentDetected,
    PixelActivated,
    PixelDeactivated,
    UserMessageReceived,
)

from app.runtime.state import PixelStateMachine

from app.ai.providers.fallback import FallbackProvider
from app.ai.registry import ProviderRegistry
from app.ai.router import ModelRouter
from app.ai.provider import ChatMessage

from app.intents.direct import DirectIntentEngine
from app.intents.router import IntentRouter

from app.actions.engine import ActionEngine
from app.actions.registry import ActionRegistry
from app.actions.types import ActionRequest

from app.security.permissions import SecurityLayer

logger = get_logger("pixel.runtime")


class PixelRuntime:
    """Central Pixel v2 runtime.

    Coordinates all subsystems:
    - EventBus: decoupled inter-subsystem communication
    - StateMachine: explicit runtime state
    - ProviderRegistry: AI provider management
    - ModelRouter: request routing
    - IntentRouter: deterministic + AI intent detection
    - ActionEngine: safe action execution
    - SecurityLayer: centralized permission checks
    """

    def __init__(self, config: PixelConfig | None = None) -> None:
        self._config = config or PixelConfig()
        self._running = False
        self._session_id = ""

        # -- Subsystems (created on start) --
        self.event_bus: EventBus | None = None
        self.state_machine: PixelStateMachine | None = None
        self.provider_registry: ProviderRegistry | None = None
        self.model_router: ModelRouter | None = None
        self.intent_router: IntentRouter | None = None
        self.action_engine: ActionEngine | None = None
        self.action_registry: ActionRegistry | None = None
        self.security: SecurityLayer | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> PixelConfig:
        return self._config

    @property
    def running(self) -> bool:
        return self._running

    @property
    def state(self) -> PixelStateEnum:
        if self.state_machine is None:
            return PixelStateEnum.IDLE
        return self.state_machine.state

    @property
    def session_id(self) -> str:
        return self._session_id

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Initialize all subsystems and enter IDLE state."""
        if self._running:
            logger.warning("runtime_already_running")
            return

        # Configure logging first
        setup_logging(
            level=self._config.logging.level,
            fmt=self._config.logging.format,
            log_file=self._config.logging.file,
        )

        self._session_id = str(uuid.uuid4())[:8]
        logger.info(
            "runtime_starting",
            version=self._config.runtime.version,
            session_id=self._session_id,
        )

        # Initialize subsystems in dependency order
        self.event_bus = EventBus()
        self.state_machine = PixelStateMachine(event_bus=self.event_bus)
        self.security = SecurityLayer(self._config.security)
        self.action_registry = ActionRegistry()
        self.action_engine = ActionEngine(
            registry=self.action_registry,
            security=self.security,
        )

        # AI providers
        self.provider_registry = ProviderRegistry()
        self.provider_registry.register(FallbackProvider())

        # Router
        self.model_router = ModelRouter(
            registry=self.provider_registry,
            config=self._config.router,
        )

        # Intent detection
        direct_engine = DirectIntentEngine(self._config.intent)
        self.intent_router = IntentRouter(
            direct_engine=direct_engine,
            model_router=self.model_router,
        )

        self._running = True

        # Emit activation event
        await self.event_bus.emit(PixelActivated(source="runtime"))

        logger.info(
            "runtime_started",
            state=self.state.value,
            providers=self.provider_registry.list_all(),
        )

    async def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        if not self._running:
            return

        logger.info("runtime_stopping", session_id=self._session_id)

        # Reset state
        if self.state_machine:
            self.state_machine.reset()

        # Emit deactivation
        if self.event_bus:
            await self.event_bus.emit(PixelDeactivated(source="runtime"))
            self.event_bus.clear()

        self._running = False
        logger.info("runtime_stopped")

    # -- Input handling ------------------------------------------------------

    async def handle_input(self, text: str) -> str:
        """Process a user text input through the full pipeline.

        Returns:
            Response string for the user.
        """
        if not self._running:
            return "Pixel is not running."

        request_id = str(uuid.uuid4())[:12]
        bind_request_context(request_id, session=self._session_id)
        start_time = time.perf_counter()

        try:
            # Emit input event
            await self.event_bus.emit(  # type: ignore[union-attr]
                UserMessageReceived(
                    source="user",
                    text=text,
                    session_id=self._session_id,
                )
            )

            # Transition to UNDERSTANDING
            self.state_machine.transition(PixelStateEnum.UNDERSTANDING)  # type: ignore[union-attr]

            # Route through intent system
            result = await self.intent_router.route(text)  # type: ignore[union-attr]

            # Emit intent detection event
            await self.event_bus.emit(  # type: ignore[union-attr]
                IntentDetected(
                    source="intent_router",
                    intent=result.intent or "unknown",
                    confidence=result.confidence,
                    parameters=result.parameters,
                )
            )

            # --- Direct intent with response ---
            if result.direct and result.response is not None:
                self.state_machine.transition(PixelStateEnum.EXECUTING)  # type: ignore[union-attr]
                self.state_machine.transition(PixelStateEnum.SUCCESS)  # type: ignore[union-attr]
                self.state_machine.transition(PixelStateEnum.IDLE)  # type: ignore[union-attr]

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "request_completed",
                    request_id=request_id,
                    route="DIRECT",
                    duration_ms=round(elapsed, 2),
                )
                return result.response

            # --- Direct intent needing action ---
            if result.direct and result.intent:
                self.state_machine.transition(PixelStateEnum.EXECUTING)  # type: ignore[union-attr]
                response = await self._execute_intent_action(result.intent, result.parameters)
                if response:
                    self.state_machine.transition(PixelStateEnum.SUCCESS)  # type: ignore[union-attr]
                else:
                    response = f"I recognized the command '{result.intent}' but the action handler is not yet implemented."
                    self.state_machine.transition(PixelStateEnum.SUCCESS)  # type: ignore[union-attr]
                self.state_machine.transition(PixelStateEnum.IDLE)  # type: ignore[union-attr]
                return response

            # --- AI routing ---
            self.state_machine.transition(PixelStateEnum.THINKING)  # type: ignore[union-attr]

            response = await self._handle_ai_request(text, request_id)

            self.state_machine.transition(PixelStateEnum.RESPONDING)  # type: ignore[union-attr]
            self.state_machine.transition(PixelStateEnum.IDLE)  # type: ignore[union-attr]

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                "request_completed",
                request_id=request_id,
                route=result.route_decision.route.value if result.route_decision else "AI",
                duration_ms=round(elapsed, 2),
            )
            return response

        except Exception as exc:
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            # Try to recover state
            try:
                if self.state_machine and self.state_machine.state != PixelStateEnum.IDLE:
                    self.state_machine.reset()
            except Exception:
                pass

            return self._format_error(exc)

        finally:
            clear_request_context()

    # -- Internal ------------------------------------------------------------

    async def _handle_ai_request(self, text: str, request_id: str) -> str:
        """Send text to an AI provider via the router."""
        if not self.provider_registry:
            return "No AI providers available."

        # Get best available provider
        provider = await self.provider_registry.get_best()
        if provider is None:
            return "No AI providers are currently available."

        try:
            messages = [ChatMessage(role="user", content=text)]
            response = await provider.chat(messages)

            await self.event_bus.emit(  # type: ignore[union-attr]
                AIResponseCompleted(
                    source="ai",
                    provider=provider.name,
                    model=response.model,
                    response=response.content,
                    latency_ms=response.latency_ms,
                )
            )
            return response.content

        except PixelError as exc:
            logger.warning(
                "ai_provider_failed",
                provider=provider.name,
                error=str(exc),
            )
            return f"I'm having trouble connecting to my AI backend. {self._format_error(exc)}"

    async def _execute_intent_action(
        self, intent: str, parameters: dict[str, Any]
    ) -> str | None:
        """Execute an action for a recognized intent."""
        if not self.action_engine or not self.action_registry:
            return None

        action_name = intent.lower()
        if not self.action_registry.has(action_name):
            return None

        request = ActionRequest(action=action_name, arguments=parameters)
        result = await self.action_engine.execute(request)

        if result.success:
            return result.output
        return None

    def _format_error(self, exc: Exception) -> str:
        """Format an error for user display (no technical details)."""
        if isinstance(exc, PixelError):
            # User-friendly messages based on error type
            from app.core.errors import ProviderError, ActionError, SecurityError

            if isinstance(exc, ProviderError):
                return "I'm having trouble with my AI connection right now. Try a simpler request, or try again in a moment."
            if isinstance(exc, SecurityError):
                return "That action requires permissions that aren't currently available."
            if isinstance(exc, ActionError):
                return "I wasn't able to complete that action. Please try again."
            return "Something went wrong. Please try again."
        return "An unexpected error occurred. Please try again."

    # -- Dispatch/Event shortcuts -------------------------------------------

    async def dispatch_event(self, event: Any) -> None:
        """Dispatch an event on the event bus."""
        if self.event_bus:
            await self.event_bus.emit(event)

    async def request_action(self, action: str, arguments: dict | None = None) -> Any:
        """Execute a named action through the action engine."""
        if not self.action_engine:
            return None
        request = ActionRequest(action=action, arguments=arguments or {})
        return await self.action_engine.execute(request)
