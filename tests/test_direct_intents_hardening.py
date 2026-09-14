"""Tests verifying DirectIntentEngine false-positive immunity and bounded pattern precision."""

from __future__ import annotations

import pytest
from app.intents.direct import DirectIntentEngine
from app.core.config import IntentConfig
from app.core.types import DirectIntent


@pytest.fixture
def engine():
    return DirectIntentEngine(IntentConfig())


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "what is the capital of France?",
        "what is the meaning of life",
        "what is 2024 in roman numerals",
        "what is machine learning",
        "what is Python used for",
        "what day is Christmas this year",
        "what day is Halloween",
        "what day is the super bowl",
        "what time does the train leave",
        "what time does the store open",
        "tell me a story about time",
        "open-ended questions are great",
        "",
        "   ",
    ],
)
def test_false_positives_avoided(engine, text):
    """General knowledge and conversational questions must NOT falsely trigger direct intents."""
    match = engine.detect(text)
    assert match is None, f"Expected '{text}' to return None, got {match}"


@pytest.mark.unit
def test_calculator_division_by_zero(engine):
    """Division by zero returns controlled error response instead of crashing or silent failure."""
    match = engine.detect("calculate 100 / 0")
    assert match is not None
    assert match.intent == DirectIntent.CALCULATOR
    assert "division by zero" in match.response.lower()


@pytest.mark.unit
def test_calculator_with_question_mark(engine):
    """Calculator pattern works with trailing question mark."""
    match = engine.detect("what is 50 + 25?")
    assert match is not None
    assert match.intent == DirectIntent.CALCULATOR
    assert match.response == "75"


@pytest.mark.unit
def test_calculator_pure_expression(engine):
    """Pure mathematical expression resolves correctly."""
    match = engine.detect("25 * 4")
    assert match is not None
    assert match.intent == DirectIntent.CALCULATOR
    assert match.response == "100"


@pytest.mark.unit
def test_open_application_multiword(engine):
    """Open application extracts full multi-word application name."""
    match = engine.detect("open visual studio code")
    assert match is not None
    assert match.intent == DirectIntent.OPEN_APPLICATION
    assert match.parameters.get("name") == "visual studio code"


@pytest.mark.unit
def test_greeting_bounded(engine):
    """Greeting matching respects word boundaries (e.g. 'hibernate' is not 'hi')."""
    assert engine.detect("hibernate mode") is None
    assert engine.detect("historic event") is None
    assert engine.detect("hi there") is not None
    assert engine.detect("hello") is not None
