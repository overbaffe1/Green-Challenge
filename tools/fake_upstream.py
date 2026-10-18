"""A deterministic stand-in for the GreenVLA server.

Used only for transport/feed self-checks: it needs no GPU, no checkpoint and
no Isaac Sim, and it never ships as a submission backend.  The plan it returns
is a smooth right-arm raise with a grasp in the middle, so the timing of the
feed is visible in the numbers.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from greenchallenge import layout
from greenchallenge.contract import split_upper_body


def numeric_to_absolute_row(values: np.ndarray) -> dict:
    """Numeric wire row -> dict with *absolute* fingers (model-server side)."""
    values = np.asarray(values, dtype=np.float64)
    base = values[layout.WIRE_SLICES["base_command"]]
    return {
        "legs_joint_pos": values[layout.WIRE_SLICES["legs_joint_pos"]].tolist(),
        "torso_joint_pos": values[layout.WIRE_SLICES["torso_joint_pos"]].tolist(),
        "finger_joint_pos": values[layout.WIRE_SLICES["finger_joint_pos"]].tolist(),
        "wrist_joint_pos": values[layout.WIRE_SLICES["wrist_joint_pos"]].tolist(),
        "velocity": values[layout.WIRE_SLICES["velocity"]].tolist(),
        "base_command": {
            "root_height": float(base[0]),
            "roll": float(base[1]),
            "pitch": float(base[2]),
            "yaw": float(base[3]),
        },
    }


def _fingers_from_state(state: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(state.get("finger_joint_pos", np.zeros(12)), dtype=np.float64).reshape(-1)
    if raw.size == 16:
        return np.concatenate([raw[0:6], raw[8:14]])
    if raw.size == 12:
        return raw.copy()
    raise ValueError(f"unexpected finger state shape {raw.shape}")


class ScriptedTestPolicy:
    """A fake policy with a known, smooth 50-row plan."""

    def __init__(self, horizon: int = 50, fps: float = 30.0, grasp_at: float = 1.0) -> None:
        self.horizon = int(horizon)
        self.fps = float(fps)
        self.grasp_at = float(grasp_at)
        self.calls = 0
        self.seen_fingers: list[np.ndarray] = []
        self.seen_prompts: list[str] = []

    @property
    def metadata(self) -> dict:
        return {
            "policy": "scripted_test",
            "model_family": "fake",
            "robot": "fake_s0s1",
            "state_dim": layout.MODEL_DIM,
            "action_dim": layout.MODEL_DIM,
            "action_horizon": self.horizon,
            "action_sample_step": 1,
            "inference_steps": 1,
            "feed_by_server": False,
            "control_dt": 0.004,
        }

    def step(self, observation: Mapping[str, Any]) -> dict:
        self.calls += 1
        self.seen_prompts.append(str(observation.get("subtask") or observation.get("prompt") or ""))
        state = observation.get("state") or {}

        base = np.zeros(layout.WIRE_DIM, dtype=np.float64)
        if "torso_joint_pos" in state:
            torso, _, _, _, _ = split_upper_body(state)
            base[layout.WIRE_SLICES["torso_joint_pos"]] = torso
        observed_fingers = _fingers_from_state(state)
        self.seen_fingers.append(observed_fingers.copy())

        t = float(observation.get("t", 0.0))
        rows = []
        for j in range(self.horizon):
            row = base.copy()
            tj = t + (j + 1) / self.fps
            phase = max(0.0, min(1.0, (tj - 0.2) / 2.0))
            torso_out = row[layout.WIRE_SLICES["torso_joint_pos"]].copy()
            torso_out[6] = -0.5 * phase  # right shoulder pitch
            torso_out[9] = 0.7 * phase   # right elbow pitch
            row[layout.WIRE_SLICES["torso_joint_pos"]] = torso_out

            closed = 0.8 if tj >= self.grasp_at else 0.0
            row[layout.WIRE_SLICES["finger_joint_pos"]] = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, closed, closed, closed, closed, closed, closed]
            )
            row[layout.WIRE_SLICES["wrist_joint_pos"]] = 0.0
            row[layout.WIRE_SLICES["velocity"]] = 0.0
            row[47] = 0.87
            row[48] = 0.0
            row[49] = 0.0
            row[50] = 0.0
            rows.append(numeric_to_absolute_row(row))
        return {"actions_list": rows}
