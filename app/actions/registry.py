from __future__ import annotations

from typing import Callable, Awaitable

from .types import ActionRequest, ActionResult
from app.core.types import RiskLevel

ActionHandler = Callable[[ActionRequest], Awaitable[ActionResult]]

class ActionRegistry:
    """Registry for action handlers."""

    def __init__(self):
        self._handlers: dict[str, tuple[ActionHandler, RiskLevel]] = {}

    def register(self, action_name: str, handler: ActionHandler, risk_level: RiskLevel = RiskLevel.LOW) -> None:
        """Register a new action handler."""
        self._handlers[action_name] = (handler, risk_level)

    def get(self, action_name: str) -> tuple[ActionHandler, RiskLevel] | None:
        """Get an action handler and its risk level."""
        return self._handlers.get(action_name)

    def list_actions(self) -> list[str]:
        """List all registered action names."""
        return list(self._handlers.keys())

    def has(self, action_name: str) -> bool:
        """Check if an action is registered."""
        return action_name in self._handlers
