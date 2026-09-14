"""Controlled website shortcut launcher for Pixel v2.

Safely opens approved websites via a strict whitelist without shell execution.
Rejects arbitrary URLs, raw user URLs, and unauthorized domains.
"""

from __future__ import annotations

import webbrowser
from typing import Mapping

from .types import ActionRequest, ActionResult, ActionVerification


# Strict whitelist mapping of approved website shortcuts to fixed HTTPS URLs
APPROVED_WEBSITES: dict[str, str] = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://www.github.com",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "gmail": "https://mail.google.com",
}

# Aliases mapping to canonical names
WEBSITE_ALIASES: dict[str, str] = {
    "google": "google",
    "google search": "google",
    "youtube": "youtube",
    "yt": "youtube",
    "github": "github",
    "git": "github",
    "wikipedia": "wikipedia",
    "wiki": "wikipedia",
    "reddit": "reddit",
    "gmail": "gmail",
    "google mail": "gmail",
}


class WebsiteRegistry:
    """Registry of whitelisted websites."""

    def __init__(
        self,
        sites: Mapping[str, str] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._sites = dict(sites or APPROVED_WEBSITES)
        self._aliases = dict(aliases or WEBSITE_ALIASES)

    def find(self, query: str) -> tuple[str, str] | None:
        """Resolve a query string to (canonical_name, url) if whitelisted."""
        normalized = query.lower().strip()
        # Reject raw URLs or URI schemes
        if "://" in normalized or normalized.startswith("www."):
            return None

        canonical = self._aliases.get(normalized)
        if canonical and canonical in self._sites:
            return canonical, self._sites[canonical]
        return None

    def list_sites(self) -> list[str]:
        """Return all approved website names."""
        return sorted(self._sites.keys())


DEFAULT_WEBSITE_REGISTRY = WebsiteRegistry()


async def handle_open_website(
    request: ActionRequest,
    registry: WebsiteRegistry | None = None,
) -> ActionResult:
    """Action handler to open an approved website shortcut."""
    reg = registry or DEFAULT_WEBSITE_REGISTRY
    site_query = (
        request.arguments.get("name")
        or request.arguments.get("site")
        or ""
    )

    if not site_query or not isinstance(site_query, str):
        return ActionResult(
            success=False,
            error="Missing 'name' argument for open_website action",
        )

    resolved = reg.find(site_query)
    if not resolved:
        return ActionResult(
            success=False,
            error=f"Website '{site_query}' is not in the approved website registry.",
            verification=ActionVerification(
                verified=True,
                method="whitelist_check",
                details="Rejected: not in approved website whitelist",
            ),
        )

    canonical_name, target_url = resolved

    try:
        # Open in default browser via standard library webbrowser
        opened = webbrowser.open(target_url, new=2)
        return ActionResult(
            success=True,
            output=f"Opened {canonical_name}.",
            verification=ActionVerification(
                verified=opened,
                method="browser_dispatch",
                details=f"Dispatched URL {target_url}",
            ),
        )
    except Exception as exc:
        return ActionResult(
            success=False,
            error=f"Failed to open website '{canonical_name}': {str(exc)}",
        )
