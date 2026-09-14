"""Typed asynchronous event bus with cascade protection and exception isolation."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from typing import Any, Callable, Dict, List, Optional, Type

from app.core.errors import EventError
from app.core.logging import get_logger
from app.events.types import PixelEvent

logger = get_logger("pixel.events.bus")

# Context variable to track re-entry depth for cascade protection
_cascade_depth: contextvars.ContextVar[int] = contextvars.ContextVar("cascade_depth", default=0)


class EventBus:
    """Async typed event bus for routing system events."""

    def __init__(self, max_cascade_depth: int = 3) -> None:
        self.max_cascade_depth = max_cascade_depth
        self._subscribers: Dict[Type[PixelEvent], List[Callable]] = {}

    def subscribe(self, event_type: Type[PixelEvent], handler: Callable) -> None:
        """Subscribe a handler to an event type or base type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[PixelEvent], handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def emit(self, event: PixelEvent) -> None:
        """Emit an event to all subscribed handlers.

        Dispatches to exact type subscribers, base type subscribers, and PixelEvent subscribers.
        Protected against recursive cascades beyond max_cascade_depth.
        Isolates handler exceptions so one failure does not halt other handlers.
        """
        current_depth = _cascade_depth.get()
        if current_depth > self.max_cascade_depth:
            raise EventError(
                f"Max cascade depth ({self.max_cascade_depth}) exceeded when emitting {type(event).__name__}"
            )

        token = _cascade_depth.set(current_depth + 1)

        try:
            logger.debug(
                "emitting_event",
                event_type=type(event).__name__,
                event_id=event.event_id,
                source=event.source,
            )

            event_class = type(event)
            handlers: list[Callable] = []

            for reg_type, sub_list in self._subscribers.items():
                if issubclass(event_class, reg_type):
                    for h in sub_list:
                        if h not in handlers:
                            handlers.append(h)

            for handler in handlers:
                handler_name = getattr(handler, "__name__", repr(handler))
                try:
                    if inspect.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as exc:
                    logger.error(
                        "event_handler_failed",
                        handler=handler_name,
                        event_type=type(event).__name__,
                        error=str(exc),
                        exc_info=True,
                    )
        finally:
            _cascade_depth.reset(token)

    def subscriber_count(self, event_type: Optional[Type[PixelEvent]] = None) -> int:
        """Return subscriber count for a specific type or total distinct registrations."""
        if event_type is not None:
            return len(self._subscribers.get(event_type, []))
        return sum(len(handlers) for handlers in self._subscribers.values())

    def clear(self) -> None:
        """Clear all event subscriptions."""
        self._subscribers.clear()
