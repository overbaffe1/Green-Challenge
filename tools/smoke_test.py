#!/usr/bin/env python3
"""Run the whole two-hop pipeline with a scripted policy.

    python tools/smoke_test.py --speed 1.3 --seconds 3

No checkpoint, GPU or Isaac Sim is needed: the scripted stand-in produces a
smooth plan, the real boundary schedules it, and a simulated client pretends to
be the 250 Hz simulator (echoing its last command, as ``PROPRIO_CMD=all``
does).  It exits non-zero on any protocol or timing violation, which makes it
usable as a pre-submission check of the transport layer.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # ``python tools/smoke_test.py`` puts tools/ on sys.path, not the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenchallenge import layout, wire
from greenchallenge.boundary import Boundary, BoundaryConfig
from greenchallenge.feed import FeedConfig
from greenchallenge.metrics import JsonlLogger
from greenchallenge.upstream import UpstreamClient
from tools.fake_upstream import ScriptedTestPolicy


CAPTURE_JOINTS = ("right_shoulder_pitch", "right_elbow_pitch")
TORSO_START = layout.WIRE_SLICES["torso_joint_pos"].start
CAPTURE_INDICES = tuple(layout.JOINT_NAMES_41.index(name) - TORSO_START for name in CAPTURE_JOINTS)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_health(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError(f"no /healthz on {port}")


def observation(t: float, last: dict | None, reset: bool) -> dict:
    state = {
        "legs_joint_pos": np.zeros(12),
        "torso_joint_pos": np.zeros(13),
        "finger_joint_pos": np.zeros(16),
        "base_lin_vel": np.zeros(3),
        "base_ang_vel": np.zeros(3),
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
        "root_height": 0.87,
        "root_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    if last is not None:
        state["torso_joint_pos"] = np.asarray(last["torso_joint_pos"], float)
        fingers = np.asarray(last["finger_joint_pos"], float)
        wrists = np.asarray(last["wrist_joint_pos"], float)
        state["finger_joint_pos"] = np.concatenate(
            [fingers[0:6], wrists[0:2], fingers[6:12], wrists[2:4]]
        )
    return {
        "state": state,
        "images": {
            "top_head": np.zeros((224, 224, 3), np.uint8),
            "hand_left": np.zeros((224, 224, 3), np.uint8),
            "hand_right": np.zeros((224, 224, 3), np.uint8),
        },
        "subtask": "Raise your right hand",
        "prompt": "Raise your right hand",
        "reset": reset,
        "t": round(t, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.3)
    parser.add_argument("--seconds", type=float, default=3.0, help="simulated seconds")
    parser.add_argument("--proprio", default="all", choices=("all", "measured", "auto"))
    parser.add_argument("--telemetry", default=None, type=Path)
    parser.add_argument("--capture", type=Path, help="write two emitted joint channels and feed counters to JSON")
    parser.add_argument("--keep-frozen", action="store_true", help="keep expired plans until blend lifetime ends")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.capture and args.telemetry and args.capture.resolve() == args.telemetry.resolve():
        parser.error("--capture and --telemetry must use different files")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    raw_port, policy_port = free_port(), free_port()
    fake = ScriptedTestPolicy()
    upstream = wire.WebSocketPolicyServer(fake, host="127.0.0.1", port=raw_port, log_requests=False)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    wait_health(raw_port)

    client = UpstreamClient("127.0.0.1", raw_port)
    client.connect(retries=5, delay=0.1)

    telemetry_path = args.telemetry
    temporary = None
    if telemetry_path is None:
        temporary = tempfile.NamedTemporaryFile(prefix="greenchallenge-", suffix=".jsonl", delete=False)
        telemetry_path = temporary.name
        temporary.close()

    feed = FeedConfig(speed=args.speed, expire_frozen=not args.keep_frozen)
    boundary = Boundary(
        client,
        config=BoundaryConfig(proprio=args.proprio),
        feed_config=feed,
        telemetry=JsonlLogger(telemetry_path),
    )
    server = wire.WebSocketPolicyServer(boundary, host="127.0.0.1", port=policy_port, log_requests=False)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    wait_health(policy_port)

    simulator = wire.WebSocketPolicyClient("127.0.0.1", policy_port)
    metadata = simulator.connect(retries=5, delay=0.1)
    assert metadata["feed_by_server"] is True, metadata
    assert metadata["action_horizon"] == 50, metadata

    tick = float(metadata["control_dt"])
    commands_per_request = int(metadata["greenchallenge"]["ticks_per_period"])
    period = commands_per_request * tick
    steps = max(1, int(round(args.seconds / period)))
    capture = [] if args.capture else None
    last = None
    started = time.monotonic()
    for step in range(steps):
        t = round(step * period, 6)
        reply = simulator.step(observation(t, last, reset=step == 0))
        rows = reply["actions_list"]
        if len(rows) != commands_per_request:
            raise SystemExit(f"reply {step} had {len(rows)} rows, expected {commands_per_request}")
        if not all(isinstance(row["finger_joint_pos"], list) for row in rows):
            raise SystemExit(f"reply {step} is not wire-encoded")
        if capture is not None:
            stats = boundary.feed.stats
            capture.append({
                "t": t,
                "live": stats.last_live,
                "frozen_dropped": stats.frozen_dropped,
                "commands": [[round(row["torso_joint_pos"][i], 6) for i in CAPTURE_INDICES] for row in rows],
            })
        last = rows[-1]

    elapsed = time.monotonic() - started
    print(f"requests:     {steps}")
    print(f"model calls:  {fake.calls}")
    print(f"wall time:    {elapsed:.2f} s")
    print(f"sim time:     {steps * period:.2f} s")
    print(f"speed:        {args.speed:.2f}x")
    print(f"live plans:   {boundary.feed.stats.live_peak} peak")
    print(f"frozen drops: {boundary.feed.stats.frozen_dropped}")
    print(f"telemetry:    {telemetry_path}")

    simulator.close()
    server.shutdown()
    upstream.shutdown()
    client.close()
    server_thread.join(timeout=5)
    upstream_thread.join(timeout=5)

    if capture is not None:
        recording = {
            "joint_names": CAPTURE_JOINTS,
            "control_dt": tick,
            "commands_per_request": commands_per_request,
            "rows_per_plan": metadata["action_horizon"],
            "speed": feed.speed,
            "row_step": feed.row_step(),
            "lead": feed.lead,
            "blend_life": feed.blend_life,
            "expire_frozen": feed.expire_frozen,
            "live_peak": boundary.feed.stats.live_peak,
            "model_calls": fake.calls,
            "requests": capture,
        }
        args.capture.write_text(json.dumps(recording, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        print(f"capture:      {args.capture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
