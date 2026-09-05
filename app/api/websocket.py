import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.training.events import event_bus
from app.core.logging import logger

router = APIRouter()


@router.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    Real-time WebSocket endpoint streaming training metrics, logs, and GPU telemetry.
    """
    await websocket.accept()
    queue = await event_bus.subscribe()
    logger.info("WebSocket client connected to live telemetry stream.")

    try:
        while True:
            # Wait for event from event_bus
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        event_bus.unsubscribe(queue)
