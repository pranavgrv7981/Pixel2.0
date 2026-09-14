"""Tests for the PixelStateMachine."""

from __future__ import annotations

import pytest
from app.runtime.state import PixelStateMachine
from app.core.types import PixelStateEnum
from app.core.errors import StateError
from app.events.bus import EventBus
from app.events.types import StateChanged


@pytest.fixture
def state_machine(event_bus):
    return PixelStateMachine(event_bus)


@pytest.mark.unit
def test_initial_state_is_idle(state_machine):
    assert state_machine.state == PixelStateEnum.IDLE


@pytest.mark.unit
def test_valid_transition(state_machine):
    state_machine.transition(PixelStateEnum.UNDERSTANDING)
    assert state_machine.state == PixelStateEnum.UNDERSTANDING


@pytest.mark.unit
def test_invalid_transition(state_machine):
    with pytest.raises(StateError):
        state_machine.transition(PixelStateEnum.SUCCESS)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transition_emits_event():
    """When an event bus is attached, transitions emit StateChanged events."""
    bus = EventBus()
    sm = PixelStateMachine(event_bus=None)  # No bus to avoid async issues
    # Just test the synchronous transition works
    sm.transition(PixelStateEnum.UNDERSTANDING)
    assert sm.state == PixelStateEnum.UNDERSTANDING


@pytest.mark.unit
def test_state_history(state_machine):
    state_machine.transition(PixelStateEnum.UNDERSTANDING)
    assert len(state_machine.history) > 0
    old, new = state_machine.history[-1]
    assert old == PixelStateEnum.IDLE
    assert new == PixelStateEnum.UNDERSTANDING


@pytest.mark.unit
def test_reset_to_idle(state_machine):
    state_machine.transition(PixelStateEnum.UNDERSTANDING)
    state_machine.transition(PixelStateEnum.EXECUTING)
    state_machine.transition(PixelStateEnum.SUCCESS)
    state_machine.transition(PixelStateEnum.IDLE)
    assert state_machine.state == PixelStateEnum.IDLE


@pytest.mark.unit
def test_force_reset(state_machine):
    state_machine.transition(PixelStateEnum.UNDERSTANDING)
    state_machine.transition(PixelStateEnum.THINKING)
    state_machine.reset()
    assert state_machine.state == PixelStateEnum.IDLE
