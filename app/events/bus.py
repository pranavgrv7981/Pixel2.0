from __future__ import annotations
import asyncio
import contextvars
import inspect
from typing import Callable, Dict, List, Type, Any, Optional

from app.core.errors import EventError
from app.core.logging import get_logger
from app.events.types import PixelEvent

logger = get_logger(__name__)

# Context variable to track re-entry depth for cascade protection
_cascade_depth: contextvars.ContextVar[int] = contextvars.ContextVar("cascade_depth", default=0)

class EventBus:
    """Async event bus for routing system events."""
    
    def __init__(self, max_cascade_depth: int = 3):
        self.max_cascade_depth = max_cascade_depth
        self._subscribers: Dict[Type[PixelEvent], List[Callable]] = {}
        
    def subscribe(self, event_type: Type[PixelEvent], handler: Callable) -> None:
        """Subscribe a handler to an event type."""
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
        """Emit an event to all subscribed handlers."""
        current_depth = _cascade_depth.get()
        if current_depth > self.max_cascade_depth:
            raise EventError(f"Max cascade depth ({self.max_cascade_depth}) exceeded when emitting {type(event).__name__}")
            
        token = _cascade_depth.set(current_depth + 1)
        
        try:
            logger.debug(f"Emitting event: {type(event).__name__}", event_id=event.event_id, source=event.source)
            
            # Find all handlers for this event type and the base PixelEvent type
            handlers = list(self._subscribers.get(type(event), []))
            if type(event) is not PixelEvent:
                handlers.extend(self._subscribers.get(PixelEvent, []))
                
            for handler in handlers:
                try:
                    if inspect.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(
                        f"Handler {handler.__name__} failed processing {type(event).__name__}",
                        error=str(e),
                        exc_info=True
                    )
        finally:
            _cascade_depth.reset(token)

    def subscriber_count(self, event_type: Optional[Type[PixelEvent]] = None) -> int:
        """Return the number of subscribers for an event type, or total if None."""
        if event_type is not None:
            return len(self._subscribers.get(event_type, []))
        return sum(len(handlers) for handlers in self._subscribers.values())
        
    def clear(self) -> None:
        """Clear all subscriptions."""
        self._subscribers.clear()
