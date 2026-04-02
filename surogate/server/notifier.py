"""WebSocket-based notification hub.

Manages connected clients and broadcasts status-transition events so the
frontend can react immediately instead of polling.

Protocol (server -> client, JSON):
    {
        "type": "transition",
        "entity_type": "model" | "job" | "task",
        "entity_id": "<uuid>",
        "name": "<human name>",
        "old_status": "<status>",
        "new_status": "<status>",
        "data": { ... full entity (same shape as REST response) ... }
    }
"""

import json
from typing import Any

from fastapi import WebSocket

from surogate.utils.logger import get_logger

logger = get_logger()

_ALERT_STATUSES = {"failed", "failed_cleanup", "controller_failed", "error", "cancelled"}


class ConnectionManager:
    """Track active WebSocket connections and broadcast messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.debug("WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]
        logger.debug("WS client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        payload = json.dumps(message)
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


manager = ConnectionManager()


async def notify_transition(
    entity_type: str, entity_id: str, name: str,
    old_status: str, new_status: str,
    data: dict[str, Any],
) -> None:
    """Transition callback -- logs + broadcasts full entity to all WS clients."""
    if new_status in _ALERT_STATUSES:
        logger.warning(
            "ALERT: %s '%s' transitioned %s -> %s (id=%s)",
            entity_type, name, old_status, new_status, entity_id,
        )
    else:
        logger.info(
            "%s '%s': %s -> %s", entity_type, name, old_status, new_status,
        )

    msg = {
        "type": "transition",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "name": name,
        "old_status": old_status,
        "new_status": new_status,
        "data": data,
    }

    await manager.broadcast(msg)
