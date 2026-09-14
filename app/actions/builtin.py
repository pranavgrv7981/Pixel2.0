"""Built-in safe actions for Pixel v2 foundation.

Provides minimal, deterministic, and safe action handlers that operate
within strict permission boundaries without executing arbitrary OS commands.
"""

from __future__ import annotations

import datetime
import platform
from typing import Any

from .registry import ActionRegistry
from .types import ActionRequest, ActionResult, ActionVerification
from .calculator import handle_calculate
from .apps import handle_open_app
from .websites import handle_open_website
from .system import handle_lock_pc
from app.core.types import RiskLevel


async def handle_echo(request: ActionRequest) -> ActionResult:
    """Safe echo action handler for testing and verification."""
    args = request.arguments
    if not isinstance(args, dict):
        return ActionResult(
            success=False,
            error="Malformed arguments: expected dictionary",
        )
    message = args.get("message", "")
    return ActionResult(
        success=True,
        output=str(message),
        verification=ActionVerification(
            verified=True,
            method="direct",
            details="Argument echoed directly",
        ),
    )


async def handle_system_info(request: ActionRequest) -> ActionResult:
    """Safe system info action returning non-sensitive OS platform information."""
    os_name = platform.system()
    release = platform.release()
    arch = platform.machine()
    py_ver = platform.python_version()

    output = f"OS: {os_name} {release} ({arch}) | Python: {py_ver}"
    return ActionResult(
        success=True,
        output=output,
        verification=ActionVerification(
            verified=True,
            method="platform_inspect",
            details="Retrieved via standard library platform inspection",
        ),
    )


async def handle_time(request: ActionRequest) -> ActionResult:
    """Safe time action returning the current local time."""
    now = datetime.datetime.now()
    time_str = f"The current time is {now.strftime('%I:%M %p')}."
    return ActionResult(
        success=True,
        output=time_str,
        verification=ActionVerification(
            verified=True,
            method="local_clock",
            details="Retrieved from local system clock",
        ),
    )


async def handle_date(request: ActionRequest) -> ActionResult:
    """Safe date action returning the current date."""
    today = datetime.datetime.now()
    date_str = f"Today is {today.strftime('%A, %B %d, %Y')}."
    return ActionResult(
        success=True,
        output=date_str,
        verification=ActionVerification(
            verified=True,
            method="local_clock",
            details="Retrieved from local system clock",
        ),
    )


async def handle_greeting(request: ActionRequest) -> ActionResult:
    """Safe greeting response action."""
    msg = request.arguments.get("response", "Hello! I'm Pixel, your personal assistant. How can I help?")
    return ActionResult(
        success=True,
        output=str(msg),
        verification=ActionVerification(
            verified=True,
            method="static",
            details="Standard greeting",
        ),
    )


def register_builtin_actions(registry: ActionRegistry) -> None:
    """Register all built-in safe actions into the ActionRegistry."""
    registry.register("echo", handle_echo, RiskLevel.READ)
    registry.register("system_info", handle_system_info, RiskLevel.READ)
    registry.register("time", handle_time, RiskLevel.READ)
    registry.register("get_time", handle_time, RiskLevel.READ)
    registry.register("date", handle_date, RiskLevel.READ)
    registry.register("get_date", handle_date, RiskLevel.READ)
    registry.register("calculate", handle_calculate, RiskLevel.READ)
    registry.register("greeting", handle_greeting, RiskLevel.READ)
    registry.register("open_app", handle_open_app, RiskLevel.LOW)
    registry.register("open_application", handle_open_app, RiskLevel.LOW)
    registry.register("open_website", handle_open_website, RiskLevel.LOW)
    registry.register("lock_pc", handle_lock_pc, RiskLevel.MEDIUM)
