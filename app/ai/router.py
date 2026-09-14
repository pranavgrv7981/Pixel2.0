from __future__ import annotations

import re
from pydantic import BaseModel
from typing import Any

from app.core.types import Route
from app.core.config import RouterConfig
from .registry import ProviderRegistry


class RouteDecision(BaseModel):
    """Decision made by the model router."""
    route: Route
    provider: str
    model: str
    reason: str
    confidence: float
    requires_tools: bool = False
    requires_memory: bool = False
    requires_rag: bool = False
    requires_reasoning: bool = False


class ModelRouter:
    """Deterministic model/request router."""

    def __init__(self, registry: ProviderRegistry, config: RouterConfig):
        self.registry = registry
        self.config = config

    async def route(self, text: str, context: dict[str, Any] | None = None) -> RouteDecision:
        """Determine the route and provider for a given input."""
        length = len(text)
        
        # Default fallback
        provider_name = "fallback"
        available = await self.registry.get_available()
        if available:
            provider_name = available[0].name
        
        model_name = "default_model"

        heavy_keywords = ["analyze", "explain in detail", "plan", "refactor", "codebase"]
        greeting_prefixes = ("hi ", "hi!", "hello", "hey ", "hey!", "good morning", "good afternoon", "good evening")
        normalized = text.lower().strip()
        
        is_greeting = normalized in ("hi", "hello", "hey") or normalized.startswith(greeting_prefixes)
        
        if length < 15 or is_greeting:
            return RouteDecision(
                route=Route.FAST,
                provider=provider_name,
                model=model_name,
                reason="Short message or greeting",
                confidence=0.9
            )
            
        is_heavy = any(kw in text.lower() for kw in heavy_keywords) or length > 500
        
        if is_heavy:
            return RouteDecision(
                route=Route.HEAVY,
                provider=provider_name,
                model=model_name,
                reason="Long message or heavy keyword detected",
                confidence=0.8,
                requires_reasoning=True,
                requires_tools=True
            )
            
        return RouteDecision(
            route=Route.STANDARD,
            provider=provider_name,
            model=model_name,
            reason="Medium message without heavy keywords",
            confidence=0.7,
            requires_tools=True
        )
