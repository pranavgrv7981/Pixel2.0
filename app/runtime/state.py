"""Runtime state machine for Pixel v2.

Manages the lifecycle state of the Pixel runtime with validated
transitions and optional event bus integration.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from app.core.errors import StateError
from app.core.logging import get_logger
from app.core.types import PixelStateEnum

if TYPE_CHECKING:
    from app.events.bus import EventBus

logger = get_logger("pixel.state")

# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[PixelStateEnum, set[PixelStateEnum]] = {
    PixelStateEnum.IDLE: {
        PixelStateEnum.AWAKENING,
        PixelStateEnum.LISTENING,
        PixelStateEnum.UNDERSTANDING,
    },
    PixelStateEnum.AWAKENING: {
        PixelStateEnum.LISTENING,
        PixelStateEnum.IDLE,
        PixelStateEnum.ERROR,
    },
    PixelStateEnum.LISTENING: {
        PixelStateEnum.UNDERSTANDING,
        PixelStateEnum.IDLE,
        PixelStateEnum.ERROR,
    },
    PixelStateEnum.UNDERSTANDING: {
        PixelStateEnum.THINKING,
        PixelStateEnum.EXECUTING,
        PixelStateEnum.IDLE,
        PixelStateEnum.ERROR,
    },
    PixelStateEnum.THINKING: {
        PixelStateEnum.RESPONDING,
        PixelStateEnum.ERROR,
    },
    PixelStateEnum.RESPONDING: {
        PixelStateEnum.EXECUTING,
        PixelStateEnum.IDLE,
        PixelStateEnum.ERROR,
    },
    PixelStateEnum.EXECUTING: {
        PixelStateEnum.SUCCESS,
        PixelStateEnum.ERROR,
    },
    PixelStateEnum.SUCCESS: {
        PixelStateEnum.IDLE,
    },
    PixelStateEnum.ERROR: {
        PixelStateEnum.IDLE,
    },
}


class PixelStateMachine:
    """Explicit state machine with validated transitions.

    Thread-safe. Optionally emits StateChanged events on an EventBus.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._state = PixelStateEnum.IDLE
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._history: list[tuple[PixelStateEnum, PixelStateEnum]] = []

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> PixelStateEnum:
        """Current state."""
        return self._state

    @property
    def history(self) -> list[tuple[PixelStateEnum, PixelStateEnum]]:
        """List of (old_state, new_state) transitions."""
        return list(self._history)

    # -- Transitions ---------------------------------------------------------

    def transition(self, new_state: PixelStateEnum) -> PixelStateEnum:
        """Transition to *new_state* if the transition is valid.

        Returns:
            The new state.

        Raises:
            StateError: If the transition is not allowed.
        """
        with self._lock:
            old = self._state
            allowed = _VALID_TRANSITIONS.get(old, set())
            if new_state not in allowed:
                raise StateError(
                    f"Invalid transition: {old.value} → {new_state.value}",
                    context={
                        "old_state": old.value,
                        "new_state": new_state.value,
                        "allowed": [s.value for s in allowed],
                    },
                )
            self._state = new_state
            self._history.append((old, new_state))
            logger.debug(
                "state_transition",
                old_state=old.value,
                new_state=new_state.value,
            )

        # Emit event outside the lock to avoid deadlocks
        if self._event_bus is not None:
            self._emit_state_changed(old, new_state)

        return new_state

    def reset(self) -> None:
        """Force-reset to IDLE regardless of current state.

        Used during shutdown or error recovery.
        """
        with self._lock:
            old = self._state
            self._state = PixelStateEnum.IDLE
            self._history.append((old, PixelStateEnum.IDLE))
            logger.info("state_reset", old_state=old.value)

    # -- Internals -----------------------------------------------------------

    def _emit_state_changed(
        self, old: PixelStateEnum, new: PixelStateEnum
    ) -> None:
        """Emit a StateChanged event (fire-and-forget for sync context)."""
        # Lazy import to avoid circular dependency
        import asyncio
        from app.events.types import StateChanged

        event = StateChanged(
            source="state_machine",
            old_state=old,
            new_state=new,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._event_bus.emit(event))  # type: ignore[union-attr]
        except RuntimeError:
            # No running loop — skip event emission
            pass
