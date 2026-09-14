"""Tests for the action engine."""

from __future__ import annotations

import pytest
from app.actions.engine import ActionEngine
from app.actions.registry import ActionRegistry
from app.actions.types import ActionRequest, ActionResult
from app.security.permissions import SecurityLayer
from app.core.config import SecurityConfig
from app.core.types import RiskLevel


@pytest.fixture
def engine():
    registry = ActionRegistry()
    security = SecurityLayer(SecurityConfig())
    return ActionEngine(registry, security)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_registered_action(engine):
    async def dummy_handler(req: ActionRequest) -> ActionResult:
        return ActionResult(success=True, output="done")

    engine.registry.register("dummy", dummy_handler, RiskLevel.LOW)
    req = ActionRequest(action="dummy")
    res = await engine.execute(req)
    assert res.success is True
    assert res.output == "done"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_unregistered_action(engine):
    req = ActionRequest(action="unknown")
    res = await engine.execute(req)
    assert res.success is False
    assert "not found" in res.error.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_permission_denied(engine):
    async def dummy_handler(req: ActionRequest) -> ActionResult:
        return ActionResult(success=True, output="done")

    engine.registry.register("dangerous", dummy_handler, RiskLevel.HIGH)
    req = ActionRequest(action="dangerous")
    res = await engine.execute(req)
    assert res.success is False
    assert "denied" in res.error.lower()


@pytest.mark.unit
def test_action_result_fields():
    res = ActionResult(success=True, output="test", error="none")
    assert res.success is True
    assert res.output == "test"
    assert res.error == "none"
