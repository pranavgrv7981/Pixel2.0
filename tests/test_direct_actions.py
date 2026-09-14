"""Tests for Phase 2 direct actions layer and controlled registries."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from app.actions.builtin import (
    handle_time,
    handle_date,
    handle_greeting,
    register_builtin_actions,
)
from app.actions.apps import (
    AppRegistry,
    AppDefinition,
    handle_open_app,
    DEFAULT_APP_REGISTRY,
)
from app.actions.websites import (
    WebsiteRegistry,
    handle_open_website,
    DEFAULT_WEBSITE_REGISTRY,
)
from app.actions.system import handle_lock_pc
from app.actions.registry import ActionRegistry
from app.actions.engine import ActionEngine
from app.actions.types import ActionRequest, ActionResult
from app.security.permissions import SecurityLayer
from app.core.config import SecurityConfig
from app.core.types import RiskLevel


@pytest.fixture
def engine():
    registry = ActionRegistry()
    security = SecurityLayer(SecurityConfig())
    register_builtin_actions(registry)
    return ActionEngine(registry, security)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_time_action(engine):
    """Time action returns local time formatted with verification."""
    req = ActionRequest(action="time")
    result = await engine.execute(req)
    assert result.success is True
    assert "current time is" in result.output.lower()
    assert result.verification is not None
    assert result.verification.method == "local_clock"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_date_action(engine):
    """Date action returns local date formatted with verification."""
    req = ActionRequest(action="date")
    result = await engine.execute(req)
    assert result.success is True
    assert "today is" in result.output.lower()
    assert result.verification is not None
    assert result.verification.method == "local_clock"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_greeting_action(engine):
    """Greeting action returns user-facing greeting."""
    req = ActionRequest(action="greeting")
    result = await engine.execute(req)
    assert result.success is True
    assert "pixel" in result.output.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_open_app_known_app(engine):
    """Known application in registry executes without shell=True."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch("app.actions.apps.resolve_executable", return_value="notepad.exe"):
        req = ActionRequest(action="open_app", arguments={"name": "notepad"})
        result = await engine.execute(req)
        assert result.success is True
        assert "launched notepad" in result.output.lower()
        mock_popen.assert_called_once()
        # Verify shell is explicitly False
        _, kwargs = mock_popen.call_args
        assert kwargs.get("shell") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_open_app_unknown_app_fails_safely(engine):
    """Unknown application fails safely without attempting execution."""
    with patch("subprocess.Popen") as mock_popen:
        req = ActionRequest(action="open_app", arguments={"name": "arbitrary_unknown_binary"})
        result = await engine.execute(req)
        assert result.success is False
        assert "not in the recognized application registry" in result.error
        mock_popen.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_open_app_rejects_shell_injection(engine):
    """Attempts to inject shell commands into app launch are safely rejected."""
    with patch("subprocess.Popen") as mock_popen:
        req = ActionRequest(action="open_app", arguments={"name": "notepad && calc.exe"})
        result = await engine.execute(req)
        assert result.success is False
        assert "not in the recognized application registry" in result.error
        mock_popen.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_open_website_whitelisted(engine):
    """Whitelisted website shortcut dispatches exact URL via browser."""
    with patch("webbrowser.open", return_value=True) as mock_browser:
        req = ActionRequest(action="open_website", arguments={"name": "youtube"})
        result = await engine.execute(req)
        assert result.success is True
        assert "opened youtube" in result.output.lower()
        mock_browser.assert_called_once_with("https://www.youtube.com", new=2)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_open_website_unknown_fails_safely(engine):
    """Unknown website is rejected safely."""
    with patch("webbrowser.open") as mock_browser:
        req = ActionRequest(action="open_website", arguments={"name": "some_random_domain"})
        result = await engine.execute(req)
        assert result.success is False
        assert "not in the approved website registry" in result.error
        mock_browser.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_open_website_rejects_arbitrary_urls(engine):
    """Direct URLs (http/https/javascript) are rejected to enforce controlled whitelist."""
    with patch("webbrowser.open") as mock_browser:
        req = ActionRequest(action="open_website", arguments={"name": "https://malicious.example.com"})
        result = await engine.execute(req)
        assert result.success is False
        assert "not in the approved website registry" in result.error
        mock_browser.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lock_pc_requires_confirmation(engine):
    """Lock PC has MEDIUM risk and requires confirmation before executing."""
    req = ActionRequest(action="lock_pc")
    result = await engine.execute(req)
    assert result.success is False
    assert "requires user confirmation" in result.error.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lock_pc_executes_when_confirmed(engine):
    """Lock PC executes when confirmed=True is passed."""
    mock_lock = MagicMock(return_value=1)
    with patch("ctypes.windll.user32.LockWorkStation", mock_lock):
        req = ActionRequest(action="lock_pc", arguments={"confirmed": True})
        result = await engine.execute(req)
        assert result.success is True
        assert "locked" in result.output.lower()
