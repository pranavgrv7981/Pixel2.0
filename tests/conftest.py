from __future__ import annotations

import pytest
from app.core.config import PixelConfig, load_config
from app.events.bus import EventBus
from app.events.types import PixelEvent
from app.ai.provider import AIProvider, ChatMessage, ChatResponse, ProviderCapabilities
from app.ai.registry import ProviderRegistry
from app.ai.providers.fallback import FallbackProvider
from app.actions.types import ActionRequest, ActionResult
from app.actions.registry import ActionRegistry
from app.security.permissions import SecurityLayer

@pytest.fixture
def config():
    return PixelConfig()

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture
def provider_registry():
    reg = ProviderRegistry()
    reg.register(FallbackProvider())
    return reg

@pytest.fixture
def security_layer(config):
    return SecurityLayer(config.security)

@pytest.fixture
def action_registry():
    return ActionRegistry()
