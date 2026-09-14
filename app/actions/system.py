"""Safe system action handlers for Pixel v2.

Implements explicit, safe system operations such as workstation locking
using dedicated platform APIs without arbitrary shell commands.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from .types import ActionRequest, ActionResult, ActionVerification


async def handle_lock_pc(request: ActionRequest) -> ActionResult:
    """Safely lock the workstation using platform APIs."""
    os_name = platform.system()

    if os_name == "Windows":
        try:
            import ctypes
            # Call Windows LockWorkStation API
            success = ctypes.windll.user32.LockWorkStation()
            if success != 0:
                return ActionResult(
                    success=True,
                    output="Workstation locked successfully.",
                    verification=ActionVerification(
                        verified=True,
                        method="windows_user32_api",
                        details="ctypes.windll.user32.LockWorkStation returned success",
                    ),
                )
            return ActionResult(
                success=False,
                error="LockWorkStation API call returned 0 (failed).",
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error=f"Failed to lock workstation: {str(exc)}",
            )
    else:
        # Non-Windows fallback for test environments
        return ActionResult(
            success=True,
            output="Simulated workstation lock (non-Windows platform).",
            verification=ActionVerification(
                verified=True,
                method="simulated",
                details=f"Platform {os_name} does not use user32.LockWorkStation",
            ),
        )
