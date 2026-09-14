"""Built-in safe actions for Pixel v2 foundation.

Provides minimal, deterministic, and safe action handlers that operate
within strict permission boundaries without executing arbitrary OS commands.
"""

from __future__ import annotations

import platform
from typing import Any

from .registry import ActionRegistry
from .types import ActionRequest, ActionResult, ActionVerification
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


def register_builtin_actions(registry: ActionRegistry) -> None:
    """Register foundation built-in safe actions into the ActionRegistry."""
    registry.register("echo", handle_echo, RiskLevel.READ)
    registry.register("system_info", handle_system_info, RiskLevel.READ)
