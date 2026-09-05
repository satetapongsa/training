import asyncio
from typing import Dict, Any, Set
from app.core.logging import logger


class TrainingEventBus:
    """Pub/Sub event broadcaster for real-time WebSocket metrics and logs."""

    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    async def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcasts event message to all connected clients."""
        payload = {"event": event_type, "data": data}
        dead_queues = set()
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Discard oldest to keep stream fresh
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead_queues.add(q)
            except Exception:
                dead_queues.add(q)

        for dq in dead_queues:
            self._subscribers.discard(dq)


event_bus = TrainingEventBus()
