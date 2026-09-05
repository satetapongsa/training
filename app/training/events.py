import asyncio
from typing import Dict, Any, Set, Tuple
from app.core.logging import logger


class TrainingEventBus:
    """Pub/Sub event broadcaster for real-time WebSocket metrics and logs."""

    def __init__(self):
        self._subscribers: Set[Tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = set()

    async def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q = asyncio.Queue(maxsize=300)
        self._subscribers.add((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers = {item for item in self._subscribers if item[0] != q}

    def emit_threadsafe(self, event_type: str, data: Dict[str, Any]) -> None:
        """Thread-safe event broadcast to all subscribers on their native loops."""
        payload = {"event": event_type, "type": event_type, "data": data}
        if isinstance(data, dict):
            for k, v in data.items():
                if k not in payload:
                    payload[k] = v

        dead = set()
        for q, loop in list(self._subscribers):
            try:
                if loop.is_closed():
                    dead.add((q, loop))
                    continue

                def _safe_put(queue: asyncio.Queue, item: Any):
                    try:
                        queue.put_nowait(item)
                    except asyncio.QueueFull:
                        try:
                            queue.get_nowait()
                            queue.put_nowait(item)
                        except Exception:
                            pass

                loop.call_soon_threadsafe(_safe_put, q, payload)
            except Exception as e:
                dead.add((q, loop))

        for d in dead:
            self._subscribers.discard(d)

    async def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcasts event message to all connected clients."""
        self.emit_threadsafe(event_type, data)


event_bus = TrainingEventBus()
