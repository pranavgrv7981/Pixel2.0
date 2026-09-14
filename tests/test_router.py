"""Tests for the model router."""

from __future__ import annotations

import pytest
from app.ai.router import ModelRouter
from app.ai.registry import ProviderRegistry
from app.ai.providers.fallback import FallbackProvider
from app.core.config import RouterConfig
from app.core.types import Route


@pytest.fixture
def router():
    reg = ProviderRegistry()
    reg.register(FallbackProvider())
    return ModelRouter(registry=reg, config=RouterConfig())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_short_message_routes_fast(router):
    decision = await router.route("hi")
    assert decision.route == Route.FAST


@pytest.mark.unit
@pytest.mark.asyncio
async def test_greeting_routes_fast(router):
    decision = await router.route("hello there")
    assert decision.route == Route.FAST


@pytest.mark.unit
@pytest.mark.asyncio
async def test_medium_message_routes_standard(router):
    decision = await router.route("explain how recursion works")
    assert decision.route == Route.STANDARD


@pytest.mark.unit
@pytest.mark.asyncio
async def test_heavy_keywords_route_heavy(router):
    decision = await router.route("analyze this entire codebase and refactor it")
    assert decision.route == Route.HEAVY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_route_has_provider(router):
    decision = await router.route("hi")
    assert decision.provider is not None
    assert len(decision.provider) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_route_has_reason(router):
    decision = await router.route("hi")
    assert decision.reason is not None
    assert len(decision.reason) > 0
