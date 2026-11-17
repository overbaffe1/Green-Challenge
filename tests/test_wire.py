import json
import socket
import threading
import time
import urllib.request

import numpy as np
import pytest

from greenchallenge import wire


def test_numpy_arrays_round_trip_through_msgpack():
    payload = {
        "obs": {
            "state": {
                "legs": np.arange(12, dtype=np.float32),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            "images": {"top_head": np.zeros((2, 3, 4), dtype=np.uint8)},
            "t": np.float64(1.25),
            "reset": True,
        }
    }
    back = wire.unpackb(wire.packb(payload))
    legs = back["obs"]["state"]["legs"]
    assert isinstance(legs, np.ndarray)
    assert legs.dtype == np.float32
    np.testing.assert_array_equal(legs, np.arange(12))
    assert back["obs"]["images"]["top_head"].shape == (2, 3, 4)
    assert float(back["obs"]["t"]) == 1.25
    assert back["obs"]["reset"] is True


def test_unsupported_dtypes_are_rejected():
    with pytest.raises(ValueError):
        wire.pack_array(np.zeros(3, dtype=object))
    with pytest.raises(ValueError):
        wire.pack_array(np.zeros(3, dtype=np.complex128))


def test_to_jsonable():
    value = {"a": np.arange(3), "b": {"c": np.float32(2.0)}, "d": [np.int64(4)]}
    assert wire.to_jsonable(value) == {"a": [0, 1, 2], "b": {"c": 2.0}, "d": [4]}


class EchoPolicy:
    metadata = {"policy": "test", "action_horizon": 3}

    def __init__(self):
        self.calls = 0

    def step(self, observation):
        self.calls += 1
        assert observation["t"] == 0.5
        row = {
            "legs_joint_pos": [0.0] * 12,
            "torso_joint_pos": [0.0] * 13,
            "finger_joint_pos": [0.0] * 12,
            "wrist_joint_pos": [0.0] * 4,
            "velocity": [0.0] * 6,
            "base_command": {"root_height": 0.87, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        }
        return {"actions_list": [row, row, row]}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_health(port, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(0.05)
    return False


def test_server_speaks_the_packaged_protocol():
    policy = EchoPolicy()
    port = free_port()
    server = wire.WebSocketPolicyServer(policy, host="127.0.0.1", port=port, log_requests=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    assert wait_for_health(port)
    client = wire.WebSocketPolicyClient("127.0.0.1", port)
    metadata = client.connect(retries=3, delay=0.1)
    assert metadata["policy"] == "test"
    reply = client.step({"t": 0.5})
    assert len(reply["actions_list"]) == 3
    assert reply["actions_list"][0]["finger_joint_pos"] == [0.0] * 12
    assert "infer_ms" in reply["server_timing"]
    client.close()
    server.shutdown()
    thread.join(timeout=5)


def test_boundary_metadata_is_jsonable():
    """Metadata is logged as JSON, so no NumPy scalar may leak into it."""
    from greenchallenge.boundary import Boundary
    from greenchallenge.feed import FeedConfig

    class Fake:
        metadata = {"policy": "fake", "action_horizon": 50}

        def step(self, observation):
            raise AssertionError("not used")

    boundary = Boundary(Fake(), feed_config=FeedConfig())
    json.dumps(wire.to_jsonable(boundary.metadata))
    assert boundary.metadata["greenchallenge"]["feed"]["speed"] == 1.3
