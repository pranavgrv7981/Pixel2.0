"""Tests for ActionEngine security, built-in actions, event emissions, and handler isolation."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.actions.engine import ActionEngine
from app.actions.registry import ActionRegistry
from app.actions.types import ActionRequest, ActionResult
from app.actions.builtin import register_builtin_actions, handle_echo, handle_system_info
from app.security.permissions import SecurityLayer
from app.core.config import SecurityConfig
from app.core.types import RiskLevel
from app.core.errors import ActionError
from app.events.bus import EventBus
from app.events.types import (
    ActionRequested,
    ActionStarted,
    ActionCompleted,
    ActionFailed,
)


@pytest.fixture
def action_setup():
    event_bus = EventBus()
    registry = ActionRegistry()
    register_builtin_actions(registry)
    security = SecurityLayer(SecurityConfig())
    engine = ActionEngine(registry=registry, security=security, event_bus=event_bus)
    return engine, registry, security, event_bus


@pytest.mark.unit
@pytest.mark.asyncio
async def test_builtin_echo_action(action_setup):
    """Built-in echo action safely returns provided message."""
    engine, _, _, _ = action_setup
    req = ActionRequest(action="echo", arguments={"message": "foundation test"})
    result = await engine.execute(req)

    assert result.success is True
    assert result.output == "foundation test"
    assert result.verification is not None
    assert result.verification.verified is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_builtin_system_info_action(action_setup):
    """Built-in system_info action safely returns non-sensitive platform inspection."""
    engine, _, _, _ = action_setup
    req = ActionRequest(action="system_info")
    result = await engine.execute(req)

    assert result.success is True
    assert "OS:" in result.output
    assert "Python:" in result.output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_action_event_emissions(action_setup):
    """ActionEngine emits ActionRequested, ActionStarted, and ActionCompleted events."""
    engine, _, _, event_bus = action_setup

    requested_handler = AsyncMock()
    started_handler = AsyncMock()
    completed_handler = AsyncMock()

    event_bus.subscribe(ActionRequested, requested_handler)
    event_bus.subscribe(ActionStarted, started_handler)
    event_bus.subscribe(ActionCompleted, completed_handler)

    req = ActionRequest(action="echo", arguments={"message": "event test"})
    await engine.execute(req)

    requested_handler.assert_called_once()
    started_handler.assert_called_once()
    completed_handler.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_action_emits_failed_event(action_setup):
    """Requesting an unknown action emits ActionFailed and returns an error result."""
    engine, _, _, event_bus = action_setup

    failed_handler = AsyncMock()
    event_bus.subscribe(ActionFailed, failed_handler)

    req = ActionRequest(action="nonexistent_action")
    result = await engine.execute(req)

    assert result.success is False
    assert "not found" in result.error.lower()
    failed_handler.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_action_handler_supported(action_setup):
    """ActionEngine supports synchronous (non-coroutine) handlers seamlessly."""
    engine, registry, _, _ = action_setup

    def sync_handler(request: ActionRequest) -> ActionResult:
        return ActionResult(success=True, output=f"sync: {request.arguments.get('val')}")

    registry.register("sync_action", sync_handler, RiskLevel.LOW)
    req = ActionRequest(action="sync_action", arguments={"val": 42})
    result = await engine.execute(req)

    assert result.success is True
    assert result.output == "sync: 42"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_risk_level_spoofing_prevented(action_setup):
    """Caller cannot bypass security by setting risk_level=READ on a HIGH risk action."""
    engine, registry, _, event_bus = action_setup

    async def dangerous_handler(req: ActionRequest) -> ActionResult:
        return ActionResult(success=True, output="dangerous executed")

    # Registered with HIGH risk (default denied in config)
    registry.register("reboot_system", dangerous_handler, RiskLevel.HIGH)

    # Caller tries to spoof risk_level as READ
    spoofed_req = ActionRequest(action="reboot_system", risk_level=RiskLevel.READ)
    result = await engine.execute(spoofed_req)

    assert result.success is False
    assert "permission denied" in result.error.lower()
    # Risk level on the request was corrected to HIGH by the engine
    assert spoofed_req.risk_level == RiskLevel.HIGH


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handler_exception_isolated(action_setup):
    """An action handler that raises an unhandled exception returns ActionResult with error, no crash."""
    engine, registry, _, _ = action_setup

    async def faulty_handler(req: ActionRequest) -> ActionResult:
        raise RuntimeError("Disk full simulation")

    registry.register("faulty_action", faulty_handler, RiskLevel.LOW)
    req = ActionRequest(action="faulty_action")
    result = await engine.execute(req)

    assert result.success is False
    assert "Disk full simulation" in result.error
