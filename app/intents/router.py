from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from .direct import DirectIntentEngine
from app.ai.router import ModelRouter, RouteDecision

class IntentResult(BaseModel):
    """Result of intent detection."""
    matched: bool
    direct: bool
    intent: str | None = None
    confidence: float = 0.0
    parameters: dict[str, Any] = {}
    response: str | None = None
    route_decision: RouteDecision | None = None

class IntentRouter:
    """Orchestrates intent detection and routing."""

    def __init__(self, direct_engine: DirectIntentEngine, model_router: ModelRouter):
        self.direct_engine = direct_engine
        self.model_router = model_router

    async def route(self, text: str) -> IntentResult:
        """Route the input text to the appropriate intent or model."""
        match = self.direct_engine.detect(text)
        
        if match:
            # If match with direct response or action
            return IntentResult(
                matched=True,
                direct=True,
                intent=match.intent.name if hasattr(match.intent, 'name') else str(match.intent),
                confidence=match.confidence,
                parameters=match.parameters,
                response=match.response
            )
            
        # If no match, route via AI
        decision = await self.model_router.route(text)
        return IntentResult(
            matched=False,
            direct=False,
            route_decision=decision
        )
