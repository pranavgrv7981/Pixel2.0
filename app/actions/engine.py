from __future__ import annotations

import time

from app.core.logging import get_logger
from app.core.errors import ActionError
from app.core.types import PermissionDecision
from app.security.permissions import SecurityLayer
from .registry import ActionRegistry
from .types import ActionRequest, ActionResult

logger = get_logger(__name__)

class ActionEngine:
    """Executes actions with security validation."""

    def __init__(self, registry: ActionRegistry, security: SecurityLayer):
        self.registry = registry
        self.security = security

    async def execute(self, request: ActionRequest) -> ActionResult:
        """Execute an action request."""
        logger.info("Executing action", action=request.action, request_id=request.request_id)
        start_time = time.perf_counter()

        try:
            handler_info = self.registry.get(request.action)
            if not handler_info:
                logger.error("Action not found in registry", action=request.action)
                return ActionResult(success=False, error=f"Action '{request.action}' not found")
            
            handler, risk_level = handler_info
            
            # Override request risk level with registry's risk level to prevent spoofing
            request.risk_level = risk_level
            
            policy = self.security.check_permission(request)
            if policy.decision == PermissionDecision.DENY:
                logger.warning("Action permission denied", action=request.action, reason=policy.reason)
                return ActionResult(success=False, error="Permission denied")
            
            # If prompt, we treat as ALLOW for now but log a warning
            if policy.decision == PermissionDecision.PROMPT:
                logger.warning("Action requires prompt, treating as ALLOW for now", action=request.action)
                
            result = await handler(request)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            result.duration_ms = duration_ms
            
            logger.info("Action executed", action=request.action, success=result.success, duration_ms=duration_ms)
            return result
            
        except ActionError as e:
            logger.error("ActionError during execution", action=request.action, error=str(e))
            return ActionResult(success=False, error=str(e))
        except Exception as e:
            logger.exception("Unexpected error during action execution", action=request.action, error=str(e))
            return ActionResult(success=False, error=f"Unexpected error: {str(e)}")
