"""Msgpack WebSocket transport shared by the simulator and the model server.

The wire format is byte-compatible with the packaged GreenVLA server:
NumPy arrays are encoded as ``{b"__ndarray__": True, b"data": ..., b"dtype":
..., b"shape": ...}``, the server sends its metadata immediately after the
handshake, requests are ``{"obs": observation}`` and replies are
``{"actions_list": [...], "server_timing": {...}}``.

Nothing here knows about robots; it is a small, testable transport.
"""

from __future__ import annotations

import asyncio
import functools
import http
import logging
import threading
import time
import traceback
import urllib.request
from typing import Any, Mapping, Protocol

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames
import websockets.sync.client

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# NumPy-aware msgpack (same encoding as lerobot/scripts/obm_inference/msgpack_numpy.py)
# --------------------------------------------------------------------------- #


def pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def unpack_array(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy values so a metadata dict can be logged."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class Policy(Protocol):
    @property
    def metadata(self) -> dict: ...

    def step(self, observation: Mapping[str, Any]) -> dict: ...


class WebSocketPolicyServer:
    """Serve a stateful policy over the Green Challenge WebSocket protocol."""

    def __init__(
        self,
        policy: Policy,
        *,
        host: str = "0.0.0.0",
        port: int = 8999,
        health_path: str = "/healthz",
        max_size: int | None = None,
        ping_interval: float | None = None,
        log_requests: bool = True,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.health_path = health_path
        self.max_size = max_size
        self.ping_interval = ping_interval
        self.log_requests = log_requests
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: websocket_server.Server | None = None
        self._ready = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._server = await websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=self.max_size,
            ping_interval=self.ping_interval,
            process_request=self._process_request,
        )
        self._ready.set()
        LOGGER.info("policy server listening on %s:%d", self.host, self.port)
        try:
            await self._server.serve_forever()
        finally:
            self._server.close()

    def shutdown(self) -> None:
        """Ask the server to stop; safe to call from another thread."""
        loop, server = self._loop, self._server
        if loop is None or server is None:
            return
        loop.call_soon_threadsafe(server.close)

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    # ── HTTP health check (before the WebSocket upgrade) ───────────────────
    async def _process_request(
        self,
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> websocket_server.Response | None:
        if request.path == self.health_path:
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    # ── one client ─────────────────────────────────────────────────────────
    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        peer = websocket.remote_address
        LOGGER.info("connection opened from %s", peer)
        packer = Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        previous_total: float | None = None
        while True:
            started = time.monotonic()
            try:
                request = unpackb(await websocket.recv())
                if not isinstance(request, Mapping):
                    raise TypeError(f"request must be a mapping, got {type(request).__name__}")
                observation = request.get("obs", request)
                infer_started = time.monotonic()
                response = self.policy.step(observation)
                infer_ms = (time.monotonic() - infer_started) * 1000.0
                timing = {"infer_ms": infer_ms}
                if previous_total is not None:
                    timing["prev_total_ms"] = previous_total * 1000.0
                response["server_timing"] = timing
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - started
                if self.log_requests:
                    LOGGER.info(
                        "step t=%.3f rows=%d infer=%.1f ms previous_total=%.1f ms",
                        float(observation.get("t", -1.0)),
                        len(response.get("actions_list", ())),
                        infer_ms,
                        (previous_total or 0.0) * 1000.0,
                    )
            except websockets.ConnectionClosed:
                LOGGER.info("connection closed from %s", peer)
                return
            except Exception:
                text = traceback.format_exc()
                LOGGER.error("policy step failed:\n%s", text)
                try:
                    await websocket.send(text)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="internal error",
                    )
                except Exception:
                    pass
                return

# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class WebSocketPolicyClient:
    """Blocking client for a Green Challenge policy server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8998) -> None:
        self.uri = f"ws://{host}:{port}"
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._metadata: dict[str, Any] = {}
        self._packer = Packer()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def connect(self, *, retries: int = 1, delay: float = 1.0, timeout: float = 10.0) -> dict:
        last: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                ws = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=timeout,
                    ping_interval=None,
                    # The connection is long-lived, not scoped to a block.
                    legacy=True,
                )
                metadata = unpackb(ws.recv())
                self._ws = ws
                self._metadata = dict(metadata)
                return self._metadata
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                last = exc
                if attempt + 1 < max(1, retries):
                    LOGGER.info("upstream not ready (%s); retrying in %.1f s", exc, delay)
                    time.sleep(delay)
        raise ConnectionError(f"could not connect to {self.uri}: {last}")

    def step(self, observation: Mapping[str, Any]) -> dict:
        if self._ws is None:
            self.connect()
        assert self._ws is not None
        self._ws.send(self._packer.pack({"obs": observation}))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"policy server error:\n{reply}")
        response = unpackb(reply)
        if not isinstance(response, dict) or "actions_list" not in response:
            raise RuntimeError(f"malformed policy reply: {str(response)[:500]}")
        return response

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def wait_for_health(
    host: str,
    port: int,
    *,
    timeout: float = 120.0,
    process: Any | None = None,
    interval: float = 0.5,
) -> None:
    """Block until ``GET /healthz`` answers 200 (or the process dies)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"upstream exited before becoming healthy (returncode {process.returncode})"
            )
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        threading.Event().wait(interval)
    raise TimeoutError(f"no healthy endpoint on {host}:{port} after {timeout:.0f} s")
