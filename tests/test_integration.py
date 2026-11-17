"""End-to-end: simulator -> boundary -> upstream -> back.

This exercises the real WebSocket/msgpack transport on both hops, the
observation decoder, the action feed and the reply encoder, with a scripted
policy standing in for the checkpoint.
"""

import threading
import time

import numpy as np

from greenchallenge import wire
from greenchallenge.boundary import Boundary, BoundaryConfig
from greenchallenge.feed import FeedConfig
from greenchallenge.upstream import UpstreamClient
from greenchallenge.wire import WebSocketPolicyServer
from tools.fake_upstream import ScriptedTestPolicy


def free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_health(port, timeout=10.0):
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError(port)


def test_two_hop_pipeline():
    raw_port, policy_port = free_port(), free_port()
    fake = ScriptedTestPolicy()
    upstream_server = WebSocketPolicyServer(fake, host="127.0.0.1", port=raw_port, log_requests=False)
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()
    wait_health(raw_port)

    client = UpstreamClient("127.0.0.1", raw_port)
    client.connect(retries=3, delay=0.05)
    boundary = Boundary(
        client,
        config=BoundaryConfig(proprio="all"),
        feed_config=FeedConfig(speed=1.0, blend=True, clamp=True),
    )
    policy_server = WebSocketPolicyServer(boundary, host="127.0.0.1", port=policy_port, log_requests=False)
    policy_thread = threading.Thread(target=policy_server.serve_forever, daemon=True)
    policy_thread.start()
    wait_health(policy_port)

    simulator = wire.WebSocketPolicyClient("127.0.0.1", policy_port)
    metadata = simulator.connect(retries=3, delay=0.05)
    assert metadata["feed_by_server"] is True
    assert metadata["greenchallenge"]["feed"]["speed"] == 1.0

    dt = 0.004
    replies = 0
    last = None
    t = 0.0
    for step in range(30):  # 2.4 s of simulated time
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
            # PROPRIO_CMD=all: the simulator echoes the previous wire command
            state["torso_joint_pos"] = np.asarray(last["torso_joint_pos"], float)
            fingers = np.asarray(last["finger_joint_pos"], float)
            wrists = np.asarray(last["wrist_joint_pos"], float)
            state["finger_joint_pos"] = np.concatenate(
                [fingers[0:6], wrists[0:2], fingers[6:12], wrists[2:4]]
            )
        observation = {
            "state": state,
            "images": {
                "top_head": np.zeros((224, 224, 3), np.uint8),
                "hand_left": np.zeros((224, 224, 3), np.uint8),
                "hand_right": np.zeros((224, 224, 3), np.uint8),
            },
            "subtask": "Raise your right hand",
            "prompt": "Raise your right hand",
            "reset": step == 0,
            "t": round(t, 6),
        }
        reply = simulator.step(observation)
        rows = reply["actions_list"]
        assert len(rows) == 20
        last = rows[-1]
        replies += 1
        t += 20 * dt

    assert replies == 30
    assert fake.calls == 30
    # the fake policy saw absolute fingers; the echo decoder restored them
    assert max(float(np.max(np.abs(f))) for f in fake.seen_fingers) > 0.0
    assert boundary.instruction_switches == 0
    assert boundary.resets == 1

    simulator.close()
    policy_server.shutdown()
    upstream_server.shutdown()
    client.close()
    policy_thread.join(timeout=5)
    upstream_thread.join(timeout=5)
