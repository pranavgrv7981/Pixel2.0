"""Tests for EventBus hardening: non-function callables, subclass routing, and mixed handlers."""

from __future__ import annotations

import pytest
from app.events.bus import EventBus
from app.events.types import PixelEvent, StateChanged, UserMessageReceived
from app.core.types import PixelStateEnum


class CallableClassHandler:
    """A callable class instance (has no __name__ attribute) for testing handler robustness."""

    def __init__(self) -> None:
        self.called_with: list[PixelEvent] = []
        self.should_fail = False

    def __call__(self, event: PixelEvent) -> None:
        if self.should_fail:
            raise ValueError("Deliberate failure in callable class handler")
        self.called_with.append(event)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_callable_object_without_name_attribute():
    """EventBus works with instances of callable classes that do not possess __name__."""
    bus = EventBus()
    handler = CallableClassHandler()
    bus.subscribe(PixelEvent, handler)

    event = PixelEvent(source="test")
    await bus.emit(event)
    assert len(handler.called_with) == 1
    assert handler.called_with[0] == event

    # Also verify that if the callable object fails, bus logs error safely without AttributeError
    handler.should_fail = True
    await bus.emit(event)  # Must not crash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polymorphic_subclass_event_dispatch():
    """Subscribing to a parent event type dispatches all derived subclasses."""
    bus = EventBus()

    class CustomBaseEvent(PixelEvent):
        pass

    class SubEventA(CustomBaseEvent):
        pass

    class SubEventB(CustomBaseEvent):
        pass

    captured: list[PixelEvent] = []

    def base_handler(ev: PixelEvent) -> None:
        captured.append(ev)

    # Subscribe to CustomBaseEvent
    bus.subscribe(CustomBaseEvent, base_handler)

    ev_a = SubEventA(source="a")
    ev_b = SubEventB(source="b")
    ev_other = UserMessageReceived(source="user", text="hi")

    await bus.emit(ev_a)
    await bus.emit(ev_b)
    await bus.emit(ev_other)

    # Only ev_a and ev_b should have been dispatched to base_handler
    assert len(captured) == 2
    assert captured[0] == ev_a
    assert captured[1] == ev_b


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_sync_and_async_handlers_executed():
    """EventBus cleanly executes a combination of sync and async subscribers."""
    bus = EventBus()
    execution_order: list[str] = []

    def sync_sub(ev: PixelEvent) -> None:
        execution_order.append("sync")

    async def async_sub(ev: PixelEvent) -> None:
        execution_order.append("async")

    bus.subscribe(PixelEvent, sync_sub)
    bus.subscribe(PixelEvent, async_sub)

    await bus.emit(PixelEvent(source="test"))
    assert execution_order == ["sync", "async"]
