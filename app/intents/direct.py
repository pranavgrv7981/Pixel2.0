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
from app.actions.calculator import safe_calculate, CalculatorError
from app.actions.websites import DEFAULT_WEBSITE_REGISTRY


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
        if not text or not isinstance(text, str):
            return None

        normalized_text = text.lower().strip()
        if not normalized_text:
            return None

        # Clean trailing punctuation for phrase matching
        cleaned = normalized_text.rstrip("?!.").strip()

        # -------------------------------------------------------------------
        # 1. LOCK PC patterns
        # -------------------------------------------------------------------
        lock_phrases = ["lock my pc", "lock pc", "lock computer", "lock screen", "lock workstation", "lock the pc"]
        if cleaned in lock_phrases:
            return IntentMatch(
                intent=DirectIntent.LOCK_PC,
                confidence=0.95,
            )

        # -------------------------------------------------------------------
        # 2. TIME patterns (asking for current time)
        # -------------------------------------------------------------------
        time_phrases = [
            "what time is it",
            "what's the time",
            "what is the time",
            "current time",
            "tell me the time",
            "what time",
        ]
        is_exact_time = cleaned in ("time", "current time", "the time")
        is_time_phrase = any(p in normalized_text for p in time_phrases)

        if is_exact_time or is_time_phrase:
            # Guard against event questions ("what time does the train leave", "tell me a story about time")
            event_words = ["does", "did", "will", "starts", "ends", "leaves", "story", "book", "movie"]
            if not any(w in normalized_text.split() for w in event_words):
                now = datetime.datetime.now()
                response = f"The current time is {now.strftime('%I:%M %p')}."
                return IntentMatch(
                    intent=DirectIntent.TIME,
                    confidence=0.95,
                    response=response,
                )

        # -------------------------------------------------------------------
        # 3. DATE patterns (asking for current date)
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
        is_exact_date = cleaned in ("date", "current date", "today's date", "todays date")
        is_date_phrase = any(p in normalized_text for p in date_phrases)

        if is_exact_date or is_date_phrase:
            # Guard against specific calendar events ("what day is christmas")
            event_words = ["christmas", "halloween", "easter", "super bowl", "thanksgiving", "born", "birthday", "released"]
            if not any(w in normalized_text for w in event_words):
                today = datetime.datetime.now()
                response = f"Today is {today.strftime('%A, %B %d, %Y')}."
                return IntentMatch(
                    intent=DirectIntent.DATE,
                    confidence=0.95,
                    response=response,
                )

        # -------------------------------------------------------------------
        # 4. CALCULATOR patterns (safe arithmetic evaluation)
        # -------------------------------------------------------------------
        is_explicit_calc = cleaned.startswith("calculate ")
        is_what_is = cleaned.startswith("what is ")

        if is_explicit_calc:
            raw_expr = cleaned[len("calculate "):].strip()
        elif is_what_is:
            raw_expr = cleaned[len("what is "):].strip()
        else:
            raw_expr = cleaned

        # Arithmetic character whitelist: numbers, operators (+, -, *, /, %, **), parentheses, dots, spaces
        if raw_expr and re.fullmatch(r"^[0-9\+\-\*\/\(\)\.\s\%]+$", raw_expr):
            has_operator = any(op in raw_expr for op in ["+", "-", "*", "/", "%"])
            has_digits = bool(re.search(r"\d", raw_expr))
            if has_digits and (has_operator or is_explicit_calc):
                try:
                    val = safe_calculate(raw_expr)
                    return IntentMatch(
                        intent=DirectIntent.CALCULATOR,
                        confidence=0.9,
                        parameters={"expression": raw_expr},
                        response=str(val),
                    )
                except ZeroDivisionError:
                    return IntentMatch(
                        intent=DirectIntent.CALCULATOR,
                        confidence=0.9,
                        parameters={"expression": raw_expr},
                        response="Division by zero is undefined.",
                    )
                except CalculatorError:
                    pass
                except Exception:
                    pass

        # -------------------------------------------------------------------
        # 5. GREETING patterns
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
                parameters={"response": self.config.greeting_response},
                response=self.config.greeting_response,
            )

        # -------------------------------------------------------------------
        # 6. SYSTEM_INFO patterns
        # -------------------------------------------------------------------
        sysinfo_phrases = [
            "system info",
            "system information",
            "show system info",
            "show system information",
            "what system am i running",
            "what system is this",
            "cpu usage",
            "memory usage",
            "disk space",
        ]
        if any(cleaned == p or normalized_text.startswith(p) for p in sysinfo_phrases):
            return IntentMatch(
                intent=DirectIntent.SYSTEM_INFO,
                confidence=0.9,
            )

        # -------------------------------------------------------------------
        # 7. OPEN APPLICATION or OPEN WEBSITE patterns
        # -------------------------------------------------------------------
        open_match = re.fullmatch(r"^(?:open|launch|start|visit|browse|go to)\s+([a-zA-Z0-9_\-\.\s]+)$", normalized_text)
        if open_match:
            target = open_match.group(1).strip()
            # First check if the target is a whitelisted website
            website_found = DEFAULT_WEBSITE_REGISTRY.find(target)
            if website_found:
                canonical_site, url = website_found
                return IntentMatch(
                    intent=DirectIntent.OPEN_WEBSITE,
                    confidence=0.95,
                    parameters={"name": canonical_site, "site": canonical_site, "url": url},
                )

            # Otherwise treat as application launch request
            return IntentMatch(
                intent=DirectIntent.OPEN_APPLICATION,
                confidence=0.9,
                parameters={"name": target},
            )

        # -------------------------------------------------------------------
        # 8. BATTERY patterns (preserves Phase 0 definition)
        # -------------------------------------------------------------------
        battery_phrases = ["battery level", "battery status", "battery percentage", "power level", "charge level"]
        if any(p in normalized_text for p in battery_phrases) or normalized_text == "battery":
            return IntentMatch(intent=DirectIntent.BATTERY, confidence=0.9)

        return None
