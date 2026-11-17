"""Command lines and client for the packaged GreenVLA servers.

Only two facts are assumed about the image this package is copied into: the
``lerobot`` fork is importable, and it provides
``lerobot.scripts.obm_inference.serve_policy`` and ``serve_tracking``.  Both are
part of the organisers' own entrypoint, so they are the supported interface.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from .wire import WebSocketPolicyClient

LOGGER = logging.getLogger(__name__)

RAW_MODULE = "lerobot.scripts.obm_inference.serve_policy"
TRACKING_MODULE = "lerobot.scripts.obm_inference.serve_tracking"


def _python() -> str:
    # In the contest image this is the same interpreter that runs the
    # entrypoint, i.e. the venv that has torch and the fork installed.
    return os.environ.get("GC_PYTHON") or sys.executable


def raw_policy_command(
    *,
    checkpoint: str,
    data_config: str,
    host: str,
    port: int,
    device: str,
    feed_by_server: bool,
    row_dt: float | None,
    warmup_steps: int = 2,
    inference_steps: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """``serve_policy`` with an explicit feed decision.

    ``feed_by_server=False`` returns the untouched 50-row chunk, which is what
    the boundary stack schedules.  With ``feed_by_server=True`` the server owns
    scheduling and ``row_dt`` sets the playback rate: the reference 1/30 for
    1.0x, 1/39 for the measured 1.3x.
    """
    argv = [
        _python(),
        "-m",
        RAW_MODULE,
        "--checkpoint",
        str(checkpoint),
        "--data-config",
        str(data_config),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--device",
        str(device),
        "--warmup-steps",
        str(int(warmup_steps)),
    ]
    if feed_by_server:
        argv.append("--feed-by-server")
        if row_dt is not None:
            argv += ["--feed.row-dt", f"{float(row_dt):.17g}"]
    if inference_steps:
        argv += ["--inference-steps", str(int(inference_steps))]
    argv += list(extra_args or [])
    return argv


def tracking_command(
    *,
    upstream_port: int,
    port: int,
    calibration: Path | str,
    joint_calibration: Path | str,
    adapter: str = "none",
    neck_compensation: bool = False,
    log: str = "/tmp/greenchallenge/tracking.jsonl",
) -> list[str]:
    """The packaged wire boundary: echo decoding and relative finger targets."""
    argv = [
        _python(),
        "-m",
        TRACKING_MODULE,
        "--upstream-port",
        str(int(upstream_port)),
        "--port",
        str(int(port)),
        "--calibration",
        str(calibration),
        "--joint-calibration",
        str(joint_calibration),
        "--adapter",
        str(adapter),
        "--log",
        str(log),
    ]
    if neck_compensation:
        argv.append("--neck-compensation")
    return argv


class UpstreamClient:
    """Reconnect-capable wrapper around the msgpack WebSocket client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8998) -> None:
        self.host = host
        self.port = int(port)
        self._client = WebSocketPolicyClient(host, port)
        self.metadata: dict[str, Any] = {}

    def connect(self, *, retries: int = 60, delay: float = 1.0) -> dict:
        self.metadata = self._client.connect(retries=retries, delay=delay)
        return self.metadata

    def reconnect(self) -> None:
        self._client.close()
        self.connect(retries=10, delay=1.0)

    def step(self, observation: dict) -> dict:
        return self._client.step(observation)

    def close(self) -> None:
        self._client.close()
