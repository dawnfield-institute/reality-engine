"""EventBus — lightweight pub/sub for engine observability.

Operators emit events, dashboard and analyzers subscribe.
No coupling between producers and consumers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List


Callback = Callable[[Dict[str, Any]], None]


class EventBus:
    """Synchronous publish/subscribe event bus.

    Thread-safe enough for single-threaded tick loops.
    Dashboard integration uses async wrappers on top.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callback]] = defaultdict(list)
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 500

    def subscribe(self, event: str, callback: Callback) -> None:
        """Register *callback* to be called when *event* is emitted."""
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callback) -> None:
        """Remove *callback* from *event* subscribers."""
        try:
            self._subscribers[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event: str, data: Dict[str, Any] | None = None) -> None:
        """Emit *event* with optional *data* payload.

        All registered callbacks are invoked synchronously in subscription order.
        """
        payload = {"event": event, **(data or {})}
        self._history.append(payload)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        for cb in self._subscribers.get(event, []):
            cb(payload)

    def clear(self) -> None:
        """Remove all subscribers and history."""
        self._subscribers.clear()
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
