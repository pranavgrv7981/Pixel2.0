"""Tests for the security layer."""

from __future__ import annotations

import pytest
from app.security.permissions import SecurityLayer
from app.core.config import SecurityConfig
from app.core.types import RiskLevel, PermissionDecision
from app.actions.types import ActionRequest


@pytest.fixture
def security():
    return SecurityLayer(SecurityConfig())


@pytest.mark.unit
def test_read_auto_allowed(security):
    req = ActionRequest(action="test", risk_level=RiskLevel.READ)
    policy = security.check_permission(req)
    assert policy.decision == PermissionDecision.ALLOW


@pytest.mark.unit
def test_low_auto_allowed(security):
    req = ActionRequest(action="test", risk_level=RiskLevel.LOW)
    policy = security.check_permission(req)
    assert policy.decision == PermissionDecision.ALLOW


@pytest.mark.unit
def test_high_denied(security):
    req = ActionRequest(action="test", risk_level=RiskLevel.HIGH)
    policy = security.check_permission(req)
    assert policy.decision == PermissionDecision.DENY


@pytest.mark.unit
def test_critical_denied(security):
    req = ActionRequest(action="test", risk_level=RiskLevel.CRITICAL)
    policy = security.check_permission(req)
    assert policy.decision == PermissionDecision.DENY


@pytest.mark.unit
def test_medium_prompted(security):
    req = ActionRequest(action="test", risk_level=RiskLevel.MEDIUM)
    policy = security.check_permission(req)
    assert policy.decision == PermissionDecision.PROMPT
