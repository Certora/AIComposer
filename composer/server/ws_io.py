import asyncio
import json
from typing import Callable, Any
from composer.core.io import ComposerIO
from composer.diagnostics.stream import ProgressUpdate

class WebSocketComposerIO(ComposerIO):
    def __init__(self, websocket, loop: asyncio.AbstractEventLoop, response_queue: asyncio.Queue):
        self.websocket = websocket
        self.loop = loop
        self.response_queue = response_queue
        self.interrupt_requested = False

    def _send(self, msg_type: str, payload: Any) -> None:
        """Helper to send a JSON message over the websocket."""
        try:
            # Basic normalization for JSON serialization
            def serialize(obj):
                if hasattr(obj, "to_json"):
                    return obj.to_json()
                if hasattr(obj, "dict"):
                    return obj.dict()
                return str(obj)

            msg = json.dumps({
                "type": msg_type,
                "payload": payload
            }, default=serialize)
            
            asyncio.run_coroutine_threadsafe(self.websocket.send(msg), self.loop).result()
        except Exception as e:
            # We don't want to crash the workflow if a message fails to send,
            # but we should probably log it.
            print(f"[WS_IO] Failed to send websocket message ({msg_type}): {e}")

    def summarize_update(self, state: dict) -> None:
        self._send("summarize_update", state)

    def handle_user_update(self, update: ProgressUpdate) -> None:
        self._send("user_update", update)

    def next_checkpoint(self, checkpoint_id: str) -> None:
        self._send("checkpoint", {"checkpoint_id": checkpoint_id})

    def log_thread_id(self, thread_id: str) -> None:
        self._send("thread_id", {"thread_id": thread_id})

    def handle_human_interrupt(self, interrupt_data: dict, debug_thunk: Callable[[], None]) -> str:
        self._send("human_interrupt", interrupt_data)
        # Block until we receive a response from the queue
        future = asyncio.run_coroutine_threadsafe(self.response_queue.get(), self.loop)
        try:
            response = future.result()
            # Expecting response format: {"type": "human_response", "answer": "..."}
            return response.get("answer", "")
        except Exception as e:
            print(f"Error receiving human interrupt response: {e}")
            return "ERROR: Failed to receive response"

    def log_info(self, msg: str) -> None:
        self._send("info", {"message": msg})

    def log_error(self, msg: str) -> None:
        self._send("error", {"message": msg})

    def check_for_interrupt(self) -> bool:
        return self.interrupt_requested

    def reset_interrupt(self) -> None:
        self.interrupt_requested = False

