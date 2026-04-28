"""WebSocket client for communicating with the VS Code extension.

Provides a high-level async API over a JSON-RPC 2.0 WebSocket connection.
The extension exposes workspace, editor, and results endpoints that let the
Python side read/write files, show diffs, display webviews, and manage
proposed-change previews inside VS Code.

Usage — always as an async context manager. The yielded value is the
bridge, or ``None`` when the extension isn't running / env vars aren't
set, so callers can gracefully degrade::

    async with IDEBridge.connect() as bridge:
        if bridge is None:
            ...  # extension not available, fall back
        else:
            await bridge.show_file(content, "Token.sol", lang="solidity")

The context manager owns the WebSocket lifecycle: connection close fires
on exit (success, exception, or early return) so the underlying
``websockets`` transport is always shut down cleanly. There is no public
``close()`` and no plain ``await connect()`` form — the language-enforced
shape is the only shape, since callers were forgetting to close.
"""

import json
import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Any

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
    """Persistent WebSocket connection to the VS Code extension.

    Construct via :meth:`connect`, which is an async context manager that
    yields an ``IDEBridge`` (when the extension is reachable) or ``None``
    (when it isn't). Direct construction isn't part of the public API —
    we want exactly one lifecycle entry point so the WebSocket close
    can't be skipped."""

    class _ReaderDaemon():
        def __init__(self, _conn: ClientConnection):
            self._conn = _conn
            self.req_queue : dict[int, asyncio.Queue[dict[str, Any]]] = {}
        
        async def _send(self, id: int, data: dict[str, Any], cb: asyncio.Queue):
            assert id not in self.req_queue
            self.req_queue[id] = cb
            await self._conn.send(json.dumps(data))

        async def reader(
            self
        ):
            while True:
                try:
                    msg = await asyncio.wait_for(
                        self._conn.recv(),
                        timeout=0.1
                    )
                    try:
                        payload = json.loads(msg)
                        if "id" not in payload or not isinstance(payload["id"], str):
                            continue
                        id = payload["id"]
                        if id not in self.req_queue:
                            continue
                        self.req_queue[id].put_nowait(
                            item=payload
                        )
                        del self.req_queue[id]
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    return

    def __init__(self, ws: ClientConnection):
        self._ws = ws
        self._next_id = 0
        self._daemon = IDEBridge._ReaderDaemon(ws)

    # -- construction --------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def connect(cls) -> AsyncIterator["IDEBridge | None"]:
        """Connect to the extension WebSocket server for the duration of
        the ``async with`` block.

        Reads ``COMPOSER_WS_PORT`` and ``COMPOSER_AUTH_TOKEN`` from the
        environment. Yields ``None`` if either variable is missing or
        the WebSocket connection fails, so callers can branch on
        availability without adding a separate try/except. On exit the
        underlying connection (if any) is closed unconditionally.
        """
        port = os.environ.get("COMPOSER_WS_PORT")
        token = os.environ.get("COMPOSER_AUTH_TOKEN")
        if not port or not token:
            yield None
            return

        uri = f"ws://127.0.0.1:{port}?token={token}"
        try:
            ws = await websockets.connect(uri)
        except (OSError, websockets.WebSocketException):
            yield None
            return

        bridge = cls(ws)
        task = asyncio.create_task(
            bridge._daemon.reader()
        )
        try:
            yield bridge
        finally:
            task.cancel()
            await task
            await ws.close()

    # -- JSON-RPC plumbing ---------------------------------------------------

    async def _call(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and return the ``result`` field.

        Raises :class:`IDEBridgeError` if the response contains an ``error``.
        """
        self._next_id += 1
        this_id = self._next_id
        cb = asyncio.Queue(maxsize=1)
        msg: dict = {"jsonrpc": "2.0", "id": this_id, "method": method}
        if params is not None:
            msg["params"] = params
        await self._daemon._send(
            cb=cb,
            data=msg,
            id=this_id
        )
        resp = await asyncio.wait_for(cb.get(), 0.5)

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
