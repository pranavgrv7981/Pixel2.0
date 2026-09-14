from __future__ import annotations

import uuid
from typing import Any
from pydantic import BaseModel, Field

from app.core.types import RiskLevel

class ActionVerification(BaseModel):
    """Details of action verification."""
    verified: bool
    method: str
    details: str = ''

class ActionRequest(BaseModel):
    """Request to execute an action."""
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    source: str = 'user'
    risk_level: RiskLevel = RiskLevel.LOW
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ActionResult(BaseModel):
    """Result of an action execution."""
    success: bool
    output: str = ''
    error: str = ''
    verification: ActionVerification | None = None
    duration_ms: float = 0.0
