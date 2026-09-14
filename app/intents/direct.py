"""Deterministic direct intent engine for Pixel v2.

Matches common, unambiguous user requests using exact patterns and safe parsers,
resolving them in milliseconds without an LLM.
"""

from __future__ import annotations

import datetime
import re
from typing import Any
from pydantic import BaseModel, Field

from app.core.types import DirectIntent
from app.core.config import IntentConfig


class IntentMatch(BaseModel):
    """Result of a direct intent match."""
    intent: DirectIntent
    confidence: float
    parameters: dict[str, Any] = Field(default_factory=dict)
    response: str | None = None


class DirectIntentEngine:
    """Deterministic intent detection engine."""

    def __init__(self, config: IntentConfig) -> None:
        self.config = config

    def detect(self, text: str) -> IntentMatch | None:
        """Detect direct intents using bounded pattern matching."""
        normalized_text = text.lower().strip()
        if not normalized_text:
            return None

        # -------------------------------------------------------------------
        # 1. TIME patterns (asking for current time)
        # -------------------------------------------------------------------
        time_phrases = [
            "what time is it",
            "what's the time",
            "what is the time",
            "current time",
            "tell me the time",
            "what time",
        ]
        if any(p in normalized_text for p in time_phrases):
            # Guard against event questions ("what time does the train leave")
            if not any(w in normalized_text for w in ["does", "did", "will", "starts", "ends", "leaves"]):
                now = datetime.datetime.now()
                response = f"The current time is {now.strftime('%I:%M %p')}."
                return IntentMatch(intent=DirectIntent.TIME, confidence=0.95, response=response)

        # -------------------------------------------------------------------
        # 2. DATE patterns (asking for current date)
        # -------------------------------------------------------------------
        date_phrases = [
            "today's date",
            "todays date",
            "current date",
            "what date is it",
            "what is today's date",
            "what's today's date",
            "what is the date",
            "what's the date",
            "what day is it today",
            "what day is today",
            "what day is it",
        ]
        if any(p in normalized_text for p in date_phrases):
            # Guard against specific calendar events ("what day is christmas")
            if not any(w in normalized_text for w in ["christmas", "halloween", "easter", "super bowl", "thanksgiving", "born", "birthday", "released"]):
                today = datetime.datetime.now()
                response = f"Today is {today.strftime('%A, %B %d, %Y')}."
                return IntentMatch(intent=DirectIntent.DATE, confidence=0.95, response=response)

        # -------------------------------------------------------------------
        # 3. CALCULATOR patterns (arithmetic expressions)
        # -------------------------------------------------------------------
        cleaned = normalized_text.rstrip("?").strip()
        is_explicit_calc = cleaned.startswith("calculate ")
        is_what_is = cleaned.startswith("what is ")

        if is_explicit_calc:
            raw_expr = cleaned[len("calculate "):].strip()
        elif is_what_is:
            raw_expr = cleaned[len("what is "):].strip()
        else:
            raw_expr = cleaned

        if raw_expr and re.fullmatch(r"^[0-9\+\-\*\/\(\)\.\s]+$", raw_expr):
            has_operator = any(op in raw_expr for op in ["+", "-", "*", "/"])
            has_digits = bool(re.search(r"\d", raw_expr))
            if has_digits and (has_operator or is_explicit_calc):
                try:
                    result = eval(raw_expr, {"__builtins__": None}, {})
                    return IntentMatch(
                        intent=DirectIntent.CALCULATOR,
                        confidence=0.9,
                        response=str(result),
                    )
                except ZeroDivisionError:
                    return IntentMatch(
                        intent=DirectIntent.CALCULATOR,
                        confidence=0.9,
                        response="Division by zero is undefined.",
                    )
                except Exception:
                    pass

        # -------------------------------------------------------------------
        # 4. GREETING patterns
        # -------------------------------------------------------------------
        greeting_words = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening")
        if normalized_text in greeting_words or any(
            normalized_text.startswith(f"{w} ")
            or normalized_text.startswith(f"{w}!")
            or normalized_text.startswith(f"{w},")
            for w in greeting_words
        ):
            return IntentMatch(
                intent=DirectIntent.GREETING,
                confidence=0.85,
                response=self.config.greeting_response,
            )

        # -------------------------------------------------------------------
        # 5. OPEN_APPLICATION patterns
        # -------------------------------------------------------------------
        app_match = re.fullmatch(r"(?:open|launch|start)\s+([a-zA-Z0-9_\-\.\s]+)", normalized_text)
        if app_match:
            app_name = app_match.group(1).strip()
            return IntentMatch(
                intent=DirectIntent.OPEN_APPLICATION,
                confidence=0.9,
                parameters={"name": app_name},
            )

        # -------------------------------------------------------------------
        # 6. BATTERY patterns
        # -------------------------------------------------------------------
        battery_phrases = ["battery level", "battery status", "battery percentage", "power level", "charge level"]
        if any(p in normalized_text for p in battery_phrases) or normalized_text == "battery":
            return IntentMatch(intent=DirectIntent.BATTERY, confidence=0.9)

        # -------------------------------------------------------------------
        # 7. SYSTEM_INFO patterns
        # -------------------------------------------------------------------
        sysinfo_phrases = ["system info", "system information", "cpu usage", "memory usage", "disk space"]
        if any(p in normalized_text for p in sysinfo_phrases):
            return IntentMatch(intent=DirectIntent.SYSTEM_INFO, confidence=0.85)

        return None
