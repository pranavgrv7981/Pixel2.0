"""Action execution engine with permission validation and event emission."""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.core.errors import ActionError
from app.core.types import PermissionDecision
from app.security.permissions import SecurityLayer
from .registry import ActionRegistry
from .types import ActionRequest, ActionResult
from app.events.types import (
    ActionRequested,
    ActionStarted,
    ActionCompleted,
    ActionFailed,
)

if TYPE_CHECKING:
    from app.events.bus import EventBus

logger = get_logger("pixel.actions.engine")


class ActionEngine:
    """Executes actions with security validation and audit event emissions."""

    def __init__(
        self,
        registry: ActionRegistry,
        security: SecurityLayer,
        event_bus: EventBus | None = None,
    ) -> None:
        self.registry = registry
        self.security = security
        self.event_bus = event_bus

    async def _emit_event(self, event: Any) -> None:
        """Helper to emit event on event_bus if available."""
        if self.event_bus is not None:
            try:
                await self.event_bus.emit(event)
            except Exception as exc:
                logger.warning("action_event_emission_failed", error=str(exc))

    async def execute(self, request: ActionRequest) -> ActionResult:
        """Execute an action request through the permission & verification pipeline.

        Pipeline:
        1. Validate request and handler registration.
        2. Permission check via SecurityLayer (replaces spoofed risk levels).
        3. Emit audit events.
        4. Execute handler (supports sync and async handlers).
        5. Verify result and calculate duration.
        """
        logger.info("executing_action", action=request.action, request_id=request.request_id)
        start_time = time.perf_counter()

        await self._emit_event(
            ActionRequested(action=request.action, arguments=request.arguments)
        )

        try:
            # 1. Validation
            handler_info = self.registry.get(request.action)
            if not handler_info:
                err_msg = f"Action '{request.action}' not found"
                logger.error("action_not_found", action=request.action)
                await self._emit_event(ActionFailed(action=request.action, error=err_msg))
                return ActionResult(success=False, error=err_msg)

            handler, registered_risk = handler_info

            # 2. Permission check (override request risk level to prevent caller spoofing)
            request.risk_level = registered_risk
            policy = self.security.check_permission(request)

            if policy.decision == PermissionDecision.DENY:
                logger.warning("action_permission_denied", action=request.action, reason=policy.reason)
                await self._emit_event(ActionFailed(action=request.action, error="Permission denied"))
                return ActionResult(success=False, error="Permission denied")

            if policy.decision == PermissionDecision.PROMPT:
                logger.warning("action_requires_prompt", action=request.action)

            # 3. Execution
            await self._emit_event(ActionStarted(action=request.action))

            if inspect.iscoroutinefunction(handler):
                result = await handler(request)
            else:
                result = handler(request)

            duration_ms = (time.perf_counter() - start_time) * 1000
            result.duration_ms = duration_ms

            # 4. Result reporting
            if result.success:
                await self._emit_event(
                    ActionCompleted(action=request.action, success=True, output=result.output)
                )
                logger.info("action_executed", action=request.action, success=True, duration_ms=duration_ms)
            else:
                await self._emit_event(ActionFailed(action=request.action, error=result.error))
                logger.warning("action_handler_failed", action=request.action, error=result.error)

            return result

        except ActionError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error("action_error", action=request.action, error=str(e))
            await self._emit_event(ActionFailed(action=request.action, error=str(e)))
            return ActionResult(success=False, error=str(e), duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("action_unexpected_error", action=request.action, error=str(e))
            await self._emit_event(ActionFailed(action=request.action, error=str(e)))
            return ActionResult(success=False, error=f"Unexpected error: {str(e)}", duration_ms=duration_ms)
