"""Tests for the EventBus."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from app.events.bus import EventBus
from app.events.types import PixelEvent, StateChanged
from app.core.types import PixelStateEnum


@pytest.fixture
def bus():
    return EventBus()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subscribe_and_emit(bus):
    handler = AsyncMock()
    bus.subscribe(PixelEvent, handler)
    event = PixelEvent(source="test")
    await bus.emit(event)
    handler.assert_called_once_with(event)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsubscribe(bus):
    handler = AsyncMock()
    bus.subscribe(PixelEvent, handler)
    bus.unsubscribe(PixelEvent, handler)
    event = PixelEvent(source="test")
    await bus.emit(event)
    handler.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_subscribers(bus):
    handler1 = AsyncMock()
    handler2 = AsyncMock()
    bus.subscribe(PixelEvent, handler1)
    bus.subscribe(PixelEvent, handler2)
    event = PixelEvent(source="test")
    await bus.emit(event)
    handler1.assert_called_once_with(event)
    handler2.assert_called_once_with(event)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base_event_subscription(bus):
    """Subscribing to PixelEvent receives all derived event types."""
    handler = AsyncMock()
    bus.subscribe(PixelEvent, handler)
    event = StateChanged(
        source="test",
        old_state=PixelStateEnum.IDLE,
        new_state=PixelStateEnum.UNDERSTANDING,
    )
    await bus.emit(event)
    handler.assert_called_once_with(event)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cascade_protection(bus):
    """Cascade protection limits recursive event emission depth."""

    class CascadingEvent(PixelEvent):
        pass

    emit_count = 0

    async def cascade_handler(event):
        nonlocal emit_count
        emit_count += 1
        await bus.emit(CascadingEvent(source=f"cascade_{emit_count}"))

    bus.subscribe(CascadingEvent, cascade_handler)

    try:
        await bus.emit(CascadingEvent(source="cascade_start"))
    except Exception:
        pass
    # Should have triggered some cascades but been stopped by max_cascade_depth
    assert emit_count > 0 and emit_count <= 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handler_exception_isolated(bus):
    """One handler failing doesn't prevent other handlers from running."""

    async def failing_handler(event):
        raise ValueError("deliberate failure")

    success_handler = AsyncMock()
    bus.subscribe(PixelEvent, failing_handler)
    bus.subscribe(PixelEvent, success_handler)
    event = PixelEvent(source="test")

    await bus.emit(event)
    success_handler.assert_called_once_with(event)


@pytest.mark.unit
def test_subscriber_count(bus):
    handler1 = AsyncMock()
    handler2 = AsyncMock()
    bus.subscribe(PixelEvent, handler1)
    assert bus.subscriber_count(PixelEvent) == 1
    bus.subscribe(PixelEvent, handler2)
    assert bus.subscriber_count(PixelEvent) == 2
    bus.unsubscribe(PixelEvent, handler1)
    assert bus.subscriber_count(PixelEvent) == 1


@pytest.mark.unit
def test_clear(bus):
    handler = AsyncMock()
    bus.subscribe(PixelEvent, handler)
    bus.clear()
    assert bus.subscriber_count(PixelEvent) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_handler(bus):
    handler = AsyncMock()
    bus.subscribe(PixelEvent, handler)
    event = PixelEvent(source="test")
    await bus.emit(event)
    handler.assert_called_once_with(event)
