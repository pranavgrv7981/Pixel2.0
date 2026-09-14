from __future__ import annotations

from pydantic import BaseModel
from app.core.types import Route
from app.core.errors import PixelError


class RequestProfile(BaseModel, frozen=True):
    """Configuration profile for a model request."""
    name: str
    max_tokens: int
    temperature: float
    context_window: int
    tools_enabled: bool
    reasoning_enabled: bool
    stream: bool


FAST_PROFILE = RequestProfile(
    name="fast",
    max_tokens=256,
    temperature=0.3,
    context_window=2048,
    tools_enabled=False,
    reasoning_enabled=False,
    stream=True
)

STANDARD_PROFILE = RequestProfile(
    name="standard",
    max_tokens=2048,
    temperature=0.7,
    context_window=8192,
    tools_enabled=True,
    reasoning_enabled=False,
    stream=True
)

HEAVY_PROFILE = RequestProfile(
    name="heavy",
    max_tokens=8192,
    temperature=0.7,
    context_window=32768,
    tools_enabled=True,
    reasoning_enabled=True,
    stream=True
)


def profile_for_route(route: Route) -> RequestProfile:
    """Get the appropriate request profile for a given route."""
    if route == Route.FAST:
        return FAST_PROFILE
    elif route == Route.STANDARD:
        return STANDARD_PROFILE
    elif route == Route.HEAVY:
        return HEAVY_PROFILE
    elif route == Route.DIRECT:
        return FAST_PROFILE
    raise PixelError(f"No profile defined for route {route}")
