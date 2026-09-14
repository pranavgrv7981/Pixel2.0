from __future__ import annotations

from pydantic import BaseModel

from app.core.types import RiskLevel, PermissionDecision
from app.core.config import SecurityConfig
from app.actions.types import ActionRequest
from app.core.logging import get_logger

logger = get_logger(__name__)

class PermissionPolicy(BaseModel):
    """Result of a permission check."""
    action: str
    risk_level: RiskLevel
    decision: PermissionDecision
    reason: str = ''

class SecurityLayer:
    """Centralized security and permissions layer."""

    def __init__(self, config: SecurityConfig):
        self.config = config

    def check_permission(self, request: ActionRequest) -> PermissionPolicy:
        """Check if an action request is allowed."""
        
        decision = PermissionDecision.DENY
        reason = "Default deny policy"
        
        if request.risk_level.value in self.config.auto_allow_risk_levels:
            decision = PermissionDecision.ALLOW
            reason = "Risk level auto-allowed"
        elif request.risk_level.value in self.config.prompt_risk_levels:
            decision = PermissionDecision.PROMPT
            reason = "Risk level requires user prompt"
        elif request.risk_level.value in self.config.deny_risk_levels:
            decision = PermissionDecision.DENY
            reason = "Risk level explicitly denied"

        policy = PermissionPolicy(
            action=request.action,
            risk_level=request.risk_level,
            decision=decision,
            reason=reason
        )
        
        logger.info("Permission check", action=request.action, risk_level=request.risk_level.name, decision=decision.name, reason=reason)
        
        return policy

    def is_auto_allowed(self, risk_level: RiskLevel) -> bool:
        """Check if a risk level is auto-allowed."""
        return risk_level.value in self.config.auto_allow_risk_levels
