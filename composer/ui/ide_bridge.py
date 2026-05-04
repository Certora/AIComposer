"""WebSocket client for communicating with the VS Code extension.

Provides a high-level async API over a JSON-RPC 2.0 WebSocket connection.
The extension exposes workspace, editor, and results endpoints that let the
Python side read/write files, show diffs, display webviews, and manage
proposed-change previews inside VS Code.

Usage::

    bridge = await IDEBridge.connect()  # None if env vars are missing
    if bridge:
        root = await bridge.workspace_folder()
        await bridge.close()

Or as an async context manager::

    async with await IDEBridge.connect() as bridge:
        await bridge.show_file(content, "Token.sol", lang="solidity")
"""

import json
import os
from pathlib import Path
from types import TracebackType

import websockets
from websockets.asyncio.client import ClientConnection


class IDEBridgeError(Exception):
    """Raised when the extension returns a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: object = None):
        self.code = code
        self.rpc_message = message
        self.data = data
        super().__init__(f"IDE error {code}: {message}")


class IDEBridge:
    """Persistent WebSocket connection to the VS Code extension."""

    def __init__(self, ws: ClientConnection):
        self._ws = ws
        self._next_id = 0

    # -- construction --------------------------------------------------------

    @classmethod
    async def connect(cls) -> "IDEBridge | None":
        """Connect to the extension WebSocket server.

        Reads ``COMPOSER_WS_PORT`` and ``COMPOSER_AUTH_TOKEN`` from the
        environment.  Returns ``None`` if either variable is missing or the
        connection fails, so callers can gracefully degrade when VS Code is
        not available.
        """
        port = os.environ.get("COMPOSER_WS_PORT")
        token = os.environ.get("COMPOSER_AUTH_TOKEN")
        if not port or not token:
            return None

        uri = f"ws://127.0.0.1:{port}?token={token}"
        try:
            ws = await websockets.connect(uri)
        except (OSError, websockets.WebSocketException):
            return None
        return cls(ws)

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        await self._ws.close()

    # -- async context manager -----------------------------------------------

    async def __aenter__(self) -> "IDEBridge":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # -- JSON-RPC plumbing ---------------------------------------------------

    async def _call(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and return the ``result`` field.

        Raises :class:`IDEBridgeError` if the response contains an ``error``.
        """
        self._next_id += 1
        msg: dict = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params

        await self._ws.send(json.dumps(msg))
        raw = await self._ws.recv()
        resp = json.loads(raw)

        if "error" in resp:
            err = resp["error"]
            raise IDEBridgeError(err.get("code", -1), err.get("message", ""), err.get("data"))

        return resp.get("result", {})

    # -- public API ----------------------------------------------------------

    async def workspace_folder(self) -> Path:
        """Return the VS Code workspace root as a local path."""
        result = await self._call("workspace/getRoot")
        return Path(result["path"])

    async def show_file(
        self, content: str, path: str, lang: str | None = None
    ) -> None:
        """Open a read-only editor tab displaying *content*."""
        params: dict = {"content": content, "path": path}
        if lang is not None:
            params["lang"] = lang
        await self._call("workspace/showFile", params)

    async def show_diff(
        self, original: str, modified: str, title: str | None = None
    ) -> "DiffHandle":
        """Open a diff view comparing *original* and *modified* text.

        Returns a :class:`DiffHandle` whose ``close()`` method dismisses the
        diff tab.
        """
        params: dict = {"originalContent": original, "modifiedContent": modified}
        if title is not None:
            params["title"] = title
        result = await self._call("editor/showDiff", params)
        return DiffHandle(self, result["diffId"])

    async def show_webview(
        self,
        markdown: str,
        title: str | None = None,
        id: str | None = None,
    ) -> None:
        """Display a Markdown webview panel."""
        params: dict = {"markdown": markdown}
        if title is not None:
            params["title"] = title
        if id is not None:
            params["id"] = id
        await self._call("editor/showWebview", params)

    async def preview_results(self, files: dict[str, str]) -> str:
        """Show proposed file changes in the VS Code explorer.

        *files* maps relative paths to file contents.
        Returns the ``previewId`` needed for :meth:`accept_results` /
        :meth:`reject_results`.
        """
        result = await self._call("results/preview", {"files": files})
        return result.get("previewId", "")

    async def accept_results(self, preview_id: str) -> list[str]:
        """Accept a preview, writing the proposed files to the workspace.

        Returns the list of paths that were written.
        """
        result = await self._call("results/accept", {"previewId": preview_id})
        return result.get("writtenFiles", [])

    async def reject_results(self, preview_id: str) -> None:
        """Reject and discard a preview."""
        await self._call("results/reject", {"previewId": preview_id})


class DiffHandle:
    """Handle to a diff view opened via :meth:`IDEBridge.show_diff`."""

    def __init__(self, bridge: IDEBridge, diff_id: str):
        self._bridge = bridge
        self._diff_id = diff_id
        self._closed = False

    async def close(self) -> None:
        """Close the diff tab. Idempotent."""
        if self._closed:
            return
        self._closed = True
        await self._bridge._call("editor/closeDiff", {"diffId": self._diff_id})
