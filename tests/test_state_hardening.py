"""Tests verifying explicit state machine transitions, illegal transition rejection, and error recovery."""

from __future__ import annotations

import pytest
from app.runtime.state import PixelStateMachine
from app.core.types import PixelStateEnum
from app.core.errors import StateError
from app.events.bus import EventBus


@pytest.fixture
def sm():
    return PixelStateMachine()


@pytest.mark.unit
def test_valid_error_transitions_from_processing_states():
    """All active processing states must have a valid transition to ERROR and back to IDLE."""
    # 1. UNDERSTANDING -> ERROR -> IDLE
    sm1 = PixelStateMachine()
    sm1.transition(PixelStateEnum.UNDERSTANDING)
    sm1.transition(PixelStateEnum.ERROR)
    assert sm1.state == PixelStateEnum.ERROR
    sm1.transition(PixelStateEnum.IDLE)
    assert sm1.state == PixelStateEnum.IDLE

    # 2. UNDERSTANDING -> THINKING -> ERROR -> IDLE
    sm2 = PixelStateMachine()
    sm2.transition(PixelStateEnum.UNDERSTANDING)
    sm2.transition(PixelStateEnum.THINKING)
    sm2.transition(PixelStateEnum.ERROR)
    assert sm2.state == PixelStateEnum.ERROR
    sm2.transition(PixelStateEnum.IDLE)
    assert sm2.state == PixelStateEnum.IDLE

    # 3. UNDERSTANDING -> EXECUTING -> ERROR -> IDLE
    sm3 = PixelStateMachine()
    sm3.transition(PixelStateEnum.UNDERSTANDING)
    sm3.transition(PixelStateEnum.EXECUTING)
    sm3.transition(PixelStateEnum.ERROR)
    assert sm3.state == PixelStateEnum.ERROR
    sm3.transition(PixelStateEnum.IDLE)
    assert sm3.state == PixelStateEnum.IDLE

    # 4. THINKING -> RESPONDING -> ERROR -> IDLE
    sm4 = PixelStateMachine()
    sm4.transition(PixelStateEnum.UNDERSTANDING)
    sm4.transition(PixelStateEnum.THINKING)
    sm4.transition(PixelStateEnum.RESPONDING)
    sm4.transition(PixelStateEnum.ERROR)
    assert sm4.state == PixelStateEnum.ERROR
    sm4.transition(PixelStateEnum.IDLE)
    assert sm4.state == PixelStateEnum.IDLE

    # 5. AWAKENING -> ERROR -> IDLE
    sm5 = PixelStateMachine()
    sm5.transition(PixelStateEnum.AWAKENING)
    sm5.transition(PixelStateEnum.ERROR)
    assert sm5.state == PixelStateEnum.ERROR
    sm5.transition(PixelStateEnum.IDLE)
    assert sm5.state == PixelStateEnum.IDLE

    # 6. LISTENING -> ERROR -> IDLE
    sm6 = PixelStateMachine()
    sm6.transition(PixelStateEnum.LISTENING)
    sm6.transition(PixelStateEnum.ERROR)
    assert sm6.state == PixelStateEnum.ERROR
    sm6.transition(PixelStateEnum.IDLE)
    assert sm6.state == PixelStateEnum.IDLE


@pytest.mark.unit
@pytest.mark.parametrize(
    "start_state,invalid_target",
    [
        (PixelStateEnum.IDLE, PixelStateEnum.SUCCESS),
        (PixelStateEnum.IDLE, PixelStateEnum.ERROR),
        (PixelStateEnum.IDLE, PixelStateEnum.RESPONDING),
        (PixelStateEnum.IDLE, PixelStateEnum.EXECUTING),
        (PixelStateEnum.IDLE, PixelStateEnum.THINKING),
        (PixelStateEnum.THINKING, PixelStateEnum.IDLE),
        (PixelStateEnum.THINKING, PixelStateEnum.SUCCESS),
        (PixelStateEnum.THINKING, PixelStateEnum.UNDERSTANDING),
        (PixelStateEnum.SUCCESS, PixelStateEnum.THINKING),
        (PixelStateEnum.SUCCESS, PixelStateEnum.ERROR),
        (PixelStateEnum.ERROR, PixelStateEnum.THINKING),
        (PixelStateEnum.ERROR, PixelStateEnum.EXECUTING),
    ],
)
def test_illegal_transitions_rejected(start_state, invalid_target):
    """Illegal transitions must raise StateError and leave state unmodified."""
    sm = PixelStateMachine()
    # Force to start state for testing
    sm._state = start_state
    with pytest.raises(StateError):
        sm.transition(invalid_target)
    assert sm.state == start_state


@pytest.mark.unit
@pytest.mark.parametrize(
    "state",
    [
        PixelStateEnum.IDLE,
        PixelStateEnum.AWAKENING,
        PixelStateEnum.LISTENING,
        PixelStateEnum.UNDERSTANDING,
        PixelStateEnum.THINKING,
        PixelStateEnum.RESPONDING,
        PixelStateEnum.EXECUTING,
        PixelStateEnum.SUCCESS,
        PixelStateEnum.ERROR,
    ],
)
def test_reset_recovers_to_idle_from_any_state(state):
    """Calling reset() from any state safely brings the state machine back to IDLE."""
    sm = PixelStateMachine()
    sm._state = state
    sm.reset()
    assert sm.state == PixelStateEnum.IDLE
