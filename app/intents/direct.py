from __future__ import annotations

import datetime
import re
from typing import Any
from pydantic import BaseModel

from app.core.types import DirectIntent
from app.core.config import IntentConfig

class IntentMatch(BaseModel):
    """Result of a direct intent match."""
    intent: DirectIntent
    confidence: float
    parameters: dict[str, Any] = {}
    response: str | None = None

class DirectIntentEngine:
    """Deterministic intent detection engine."""

    def __init__(self, config: IntentConfig):
        self.config = config

    def detect(self, text: str) -> IntentMatch | None:
        """Detect direct intents using pattern matching."""
        normalized_text = text.lower().strip()

        # TIME patterns
        time_patterns = ['what time', 'current time', 'time is it', 'tell me the time']
        if any(p in normalized_text for p in time_patterns):
            now = datetime.datetime.now()
            response = f"The current time is {now.strftime('%I:%M %p')}."
            return IntentMatch(intent=DirectIntent.TIME, confidence=0.95, response=response)

        # DATE patterns
        date_patterns = ['what date', "today's date", 'what day', 'current date']
        if any(p in normalized_text for p in date_patterns):
            today = datetime.datetime.now()
            response = f"Today is {today.strftime('%A, %B %d, %Y')}."
            return IntentMatch(intent=DirectIntent.DATE, confidence=0.95, response=response)

        # CALCULATOR patterns
        calc_pattern = r"(?:calculate|what is)?\s*([0-9\+\-\*\/\(\)\.\s]+)"
        if any(w in normalized_text for w in ['calculate', 'what is', '+', '-', '*', '/']):
            match = re.search(calc_pattern, normalized_text)
            if match:
                expr = match.group(1).strip()
                if expr and re.fullmatch(r"^[0-9\+\-\*\/\(\)\.\s]+$", expr):
                    try:
                        # safe eval because we restricted characters via regex
                        result = eval(expr, {"__builtins__": None}, {})
                        response = str(result)
                        return IntentMatch(intent=DirectIntent.CALCULATOR, confidence=0.9, response=response)
                    except Exception:
                        pass

        # GREETING patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(normalized_text.startswith(p) for p in greeting_patterns):
            return IntentMatch(intent=DirectIntent.GREETING, confidence=0.85, response=self.config.greeting_response)

        # OPEN_APPLICATION patterns
        app_patterns = [r'open\s+(.+)', r'launch\s+(.+)', r'start\s+(.+)']
        for pattern in app_patterns:
            match = re.search(pattern, normalized_text)
            if match:
                app_name = match.group(1).strip()
                return IntentMatch(
                    intent=DirectIntent.OPEN_APPLICATION,
                    confidence=0.9,
                    parameters={'name': app_name}
                )

        # BATTERY patterns
        battery_patterns = ['battery', 'battery level', 'power level', 'charge level']
        if any(p in normalized_text for p in battery_patterns):
            return IntentMatch(intent=DirectIntent.BATTERY, confidence=0.9)

        # SYSTEM_INFO patterns
        sysinfo_patterns = ['system info', 'cpu usage', 'memory usage', 'disk space']
        if any(p in normalized_text for p in sysinfo_patterns):
            return IntentMatch(intent=DirectIntent.SYSTEM_INFO, confidence=0.85)

        return None
