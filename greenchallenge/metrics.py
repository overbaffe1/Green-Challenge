"""Small, dependency-free telemetry for a benchmark run.

The contest gives no access to the simulator, so the only way to debug a
submission is to log what the server decided.  One JSON object per request is
cheap (a few hundred bytes per 80 ms) and enough to reconstruct tempo, plan
liveness, clamping and instruction switches afterwards.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class JsonlLogger:
    """Thread-safe append-only JSONL writer that never kills the policy."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._broken = False
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.touch(exist_ok=True)
            except OSError as exc:
                LOGGER.warning("cannot open telemetry file %s: %s", self.path, exc)
                self._broken = True

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None or self._broken:
            return
        try:
            line = json.dumps(record, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                self._broken = True
                LOGGER.warning("telemetry disabled after write error: %s", exc)


class RateCounter:
    """Counts events and logs a summary every ``period`` seconds."""

    def __init__(self, name: str, period: float = 10.0, logger: logging.Logger | None = None) -> None:
        self.name = name
        self.period = float(period)
        self.logger = logger or LOGGER
        self._start = time.monotonic()
        self._last = self._start
        self.count = 0
        self.total = 0

    def tick(self, n: int = 1) -> None:
        self.count += n
        self.total += n
        now = time.monotonic()
        if now - self._last >= self.period:
            rate = self.count / max(now - self._last, 1e-9)
            self.logger.info("%s: %d events (%.2f/s, %d total)", self.name, self.count, rate, self.total)
            self.count = 0
            self._last = now
