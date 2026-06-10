from typing import Any, Dict, Literal

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from ws.constants import APP_ERROR_WEB_SOCKET_CODE  # type: ignore

MessageType = Literal[
    "chunk",
    "status",
    "setCode",
    "error",
    "variantComplete",
    "variantError",
    "variantCount",
    "variantModels",
    "thinking",
    "assistant",
    "toolStart",
    "toolResult",
]


class WebSocketCommunicator:
    """Handles WebSocket communication with consistent error handling"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.is_closed = False

    async def accept(self) -> None:
        """Accept the WebSocket connection"""
        await self.websocket.accept()
        print("Incoming websocket connection...")

    async def send_message(
        self,
        type: MessageType,
        value: str | None,
        variantIndex: int,
        data: Dict[str, Any] | None = None,
        eventId: str | None = None,
    ) -> None:
        """Send a message to the client with debug logging"""
        if self.is_closed:
            return

        # Print for debugging on the backend
        if type == "error":
            print(f"Error (variant {variantIndex + 1}): {value}")
        elif type == "status":
            print(f"Status (variant {variantIndex + 1}): {value}")
        elif type == "variantComplete":
            print(f"Variant {variantIndex + 1} complete")
        elif type == "variantError":
            print(f"Variant {variantIndex + 1} error: {value}")

        try:
            payload: Dict[str, Any] = {"type": type, "variantIndex": variantIndex}
            if value is not None:
                payload["value"] = value
            if data is not None:
                payload["data"] = data
            if eventId is not None:
                payload["eventId"] = eventId
            await self.websocket.send_json(payload)
        except (ConnectionClosedOK, ConnectionClosedError):
            print(f"WebSocket closed by client, skipping message: {type}")
            self.is_closed = True

    async def throw_error(self, message: str) -> None:
        """Send an error message and close the connection"""
        print(message)
        if not self.is_closed:
            try:
                await self.websocket.send_json({"type": "error", "value": message})
                await self.websocket.close(APP_ERROR_WEB_SOCKET_CODE)
            except (ConnectionClosedOK, ConnectionClosedError):
                print("WebSocket already closed by client")
            self.is_closed = True

    async def receive_params(self) -> Dict[str, Any]:
        """Receive parameters from the client"""
        try:
            params: Dict[str, Any] = await self.websocket.receive_json()
            print("Received params")
            return params
        except (WebSocketDisconnect, ConnectionClosedOK, ConnectionClosedError):
            print("Client disconnected during parameter reception")
            self.is_closed = True
            raise

    async def close(self) -> None:
        """Close the WebSocket connection"""
        if not self.is_closed:
            try:
                await self.websocket.close()
            except (ConnectionClosedOK, ConnectionClosedError):
                pass  # Already closed by client
            self.is_closed = True
