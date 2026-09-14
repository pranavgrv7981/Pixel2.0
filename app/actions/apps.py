"""Controlled application registry and launcher for Pixel v2.

Safely resolves and launches strictly whitelisted applications without shell execution.
Rejects arbitrary executable paths, command strings, and unknown applications.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Sequence
from pydantic import BaseModel, Field

from app.core.types import RiskLevel
from .types import ActionRequest, ActionResult, ActionVerification


class AppDefinition(BaseModel):
    """Definition of a safe, whitelisted application."""
    name: str
    aliases: list[str] = Field(default_factory=list)
    executable_candidates: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


class AppRegistry:
    """Registry of approved, safe applications."""

    def __init__(self, apps: Sequence[AppDefinition] | None = None) -> None:
        self._apps: dict[str, AppDefinition] = {}
        self._alias_map: dict[str, str] = {}

        if apps is None:
            self._register_default_apps()
        else:
            for app in apps:
                self.register(app)

    def _register_default_apps(self) -> None:
        """Register default approved applications for Windows."""
        default_apps = [
            AppDefinition(
                name="chrome",
                aliases=["chrome", "google chrome"],
                executable_candidates=[
                    "chrome.exe",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
                ],
            ),
            AppDefinition(
                name="edge",
                aliases=["edge", "msedge", "microsoft edge"],
                executable_candidates=[
                    "msedge.exe",
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                ],
            ),
            AppDefinition(
                name="notepad",
                aliases=["notepad", "text editor"],
                executable_candidates=[
                    "notepad.exe",
                    r"C:\Windows\System32\notepad.exe",
                    r"C:\Windows\notepad.exe",
                ],
            ),
            AppDefinition(
                name="calculator",
                aliases=["calculator", "calc"],
                executable_candidates=[
                    "calc.exe",
                    r"C:\Windows\System32\calc.exe",
                ],
            ),
            AppDefinition(
                name="explorer",
                aliases=["explorer", "file explorer", "files"],
                executable_candidates=[
                    "explorer.exe",
                    r"C:\Windows\explorer.exe",
                ],
            ),
        ]
        for app in default_apps:
            self.register(app)

    def register(self, app: AppDefinition) -> None:
        """Register an application definition."""
        canonical_name = app.name.lower().strip()
        self._apps[canonical_name] = app
        self._alias_map[canonical_name] = canonical_name
        for alias in app.aliases:
            self._alias_map[alias.lower().strip()] = canonical_name

    def find(self, query: str) -> AppDefinition | None:
        """Look up an application by name or alias."""
        normalized = query.lower().strip()
        canonical_name = self._alias_map.get(normalized)
        if canonical_name and canonical_name in self._apps:
            return self._apps[canonical_name]
        return None

    def list_apps(self) -> list[str]:
        """Return canonical names of all registered applications."""
        return sorted(self._apps.keys())


# Global default instance
DEFAULT_APP_REGISTRY = AppRegistry()


def resolve_executable(app_def: AppDefinition) -> str | None:
    """Find the first candidate executable that exists on the system."""
    for candidate in app_def.executable_candidates:
        if not candidate:
            continue
        # Direct file check
        if os.path.isabs(candidate) and os.path.isfile(candidate):
            return candidate
        # PATH check
        found = shutil.which(candidate)
        if found:
            return found
    # Fallback to first candidate name if standard system binary
    if app_def.executable_candidates:
        return app_def.executable_candidates[0]
    return None


async def handle_open_app(
    request: ActionRequest,
    registry: AppRegistry | None = None,
) -> ActionResult:
    """Action handler to launch a registered application safely."""
    app_registry = registry or DEFAULT_APP_REGISTRY
    app_query = (
        request.arguments.get("name")
        or request.arguments.get("app")
        or ""
    )

    if not app_query or not isinstance(app_query, str):
        return ActionResult(
            success=False,
            error="Missing 'name' argument for open_app action",
        )

    app_def = app_registry.find(app_query)
    if not app_def:
        return ActionResult(
            success=False,
            error=f"Application '{app_query}' is not in the recognized application registry.",
            verification=ActionVerification(
                verified=True,
                method="registry_lookup",
                details="Lookup failed: application unknown",
            ),
        )

    exe_path = resolve_executable(app_def)
    if not exe_path:
        return ActionResult(
            success=False,
            error=f"Application '{app_def.name}' could not be located on this system.",
        )

    try:
        # Launch without shell=True, passing explicit executable list
        proc = subprocess.Popen([exe_path], shell=False)
        return ActionResult(
            success=True,
            output=f"Launched {app_def.name}.",
            verification=ActionVerification(
                verified=True,
                method="process_spawn",
                details=f"Process spawned successfully (pid={proc.pid}, exe={exe_path})",
            ),
        )
    except Exception as exc:
        return ActionResult(
            success=False,
            error=f"Failed to launch '{app_def.name}': {str(exc)}",
        )
