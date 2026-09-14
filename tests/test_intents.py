"""Tests for direct intent detection."""

from __future__ import annotations

import pytest
from app.intents.direct import DirectIntentEngine
from app.core.config import IntentConfig
from app.core.types import DirectIntent


@pytest.fixture
def intent_engine():
    return DirectIntentEngine(IntentConfig())


@pytest.mark.unit
def test_time_detection(intent_engine):
    result = intent_engine.detect("what time is it")
    assert result is not None
    assert result.intent == DirectIntent.TIME
    assert result.response is not None


@pytest.mark.unit
def test_date_detection(intent_engine):
    result = intent_engine.detect("what is today's date")
    assert result is not None
    assert result.intent == DirectIntent.DATE
    assert result.response is not None


@pytest.mark.unit
def test_calculator(intent_engine):
    result = intent_engine.detect("calculate 25*4")
    assert result is not None
    assert result.intent == DirectIntent.CALCULATOR
    assert result.response == "100"


@pytest.mark.unit
def test_calculator_division(intent_engine):
    result = intent_engine.detect("what is 100/4")
    assert result is not None
    assert result.intent == DirectIntent.CALCULATOR
    assert result.response in ["25", "25.0"]


@pytest.mark.unit
def test_greeting(intent_engine):
    result = intent_engine.detect("hello")
    assert result is not None
    assert result.intent == DirectIntent.GREETING


@pytest.mark.unit
def test_open_app(intent_engine):
    result = intent_engine.detect("open chrome")
    assert result is not None
    assert result.intent == DirectIntent.OPEN_APPLICATION
    assert result.parameters == {"name": "chrome"}


@pytest.mark.unit
def test_unknown_input(intent_engine):
    result = intent_engine.detect("explain quantum physics")
    assert result is None


@pytest.mark.unit
def test_confidence_threshold(intent_engine):
    result = intent_engine.detect("what time is it")
    assert result is not None
    assert result.confidence >= 0.7
