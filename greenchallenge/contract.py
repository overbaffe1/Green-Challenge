"""Observation and action conversions for the Green Challenge wire contract.

Everything here is deliberately boring and heavily tested: a mismatch in the
hand ordering, the finger default offset or the quaternion convention does not
crash the server, it just makes the robot miss every grasp.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import layout

DEFAULT_ROOT_HEIGHT = 0.87

#: Values below this are treated as the same float across msgpack / numpy.
_ECHO_ATOL = 1e-6


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Wire <-> model calibration flags.

    Defaults match the packaged contest launcher: wrists swapped, thumb
    hardware mapping and inversions off.  The values are applied symmetrically
    to observations and commands, so a checkpoint trained under one setting is
    only meaningful under the same setting.
    """

    finger_defaults: np.ndarray = dataclasses.field(
        default_factory=lambda: layout.FINGER_DEFAULT_POSITION.copy()
    )
    swap_wrists: bool = True
    invert_r_wrist_roll: bool = False
    invert_r_thumb_yaw: bool = False
    thumb_hw: bool = False
    neck_bias: float = 0.0
    neck_limits: tuple[float, float] = (-0.524, 0.524)

    def __post_init__(self) -> None:
        defaults = np.asarray(self.finger_defaults, dtype=np.float64)
        if defaults.shape != (12,) or not np.isfinite(defaults).all():
            raise ValueError("finger_defaults must be 12 finite values")
        object.__setattr__(self, "finger_defaults", defaults)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def roll_pitch_yaw(state: Mapping[str, Any]) -> tuple[float, float, float]:
    """Body roll/pitch/yaw from ``root_quat`` (wxyz), else projected gravity."""
    quaternion = state.get("root_quat")
    if quaternion is not None:
        w, x, y, z = _vector(quaternion, name="state.root_quat", size=4)
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 0.0:
            raise ValueError("state.root_quat has zero norm")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    gravity = _vector(
        state.get("projected_gravity", (0.0, 0.0, -1.0)),
        name="state.projected_gravity",
        size=3,
    )
    roll = np.arctan2(gravity[1], -gravity[2])
    pitch = -np.arcsin(np.clip(gravity[0], -1.0, 1.0))
    yaw = float(state.get("root_yaw", 0.0))
    if not np.isfinite(yaw):
        raise ValueError("state.root_yaw must be finite")
    return float(roll), float(pitch), yaw


def split_upper_body(state: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    """Return ``(torso13, left_wrist2, right_wrist2, left_fingers6, right_fingers6)``.

    The simulator client packs fingers and wrists together in 16 channels:
    six left fingers, two left wrist dofs, six right fingers, two right wrist
    dofs.  Two older layouts are accepted because ``GreenSim`` changed shape
    more than once and a strict server is a server that does not start.
    """
    torso = np.asarray(state.get("torso_joint_pos"), dtype=np.float64).reshape(-1)
    fingers = np.asarray(state.get("finger_joint_pos"), dtype=np.float64).reshape(-1)
    wrists_value = state.get("wrist_joint_pos")

    if wrists_value is not None:
        torso = _vector(torso, name="state.torso_joint_pos", size=13)
        fingers = _vector(fingers, name="state.finger_joint_pos", size=12)
        wrists = _vector(wrists_value, name="state.wrist_joint_pos", size=4)
        left_fingers, right_fingers = fingers[:6], fingers[6:]
        left_wrists, right_wrists = wrists[:2], wrists[2:]
    elif torso.shape == (17,) and fingers.shape == (12,):
        wrists = torso[13:].copy()
        torso = torso[:13].copy()
        left_fingers, right_fingers = fingers[:6], fingers[6:]
        left_wrists, right_wrists = wrists[:2], wrists[2:]
    elif torso.shape == (13,) and fingers.shape == (16,):
        left_fingers = fingers[0:6]
        left_wrists = fingers[6:8]
        right_fingers = fingers[8:14]
        right_wrists = fingers[14:16]
    else:
        raise ValueError(
            "state must provide torso=13/fingers=12/wrists=4, "
            "torso=17/fingers=12, or torso=13/interleaved-fingers=16 "
            f"(got torso={torso.shape}, fingers={fingers.shape})"
        )

    for name, value in (
        ("torso", torso),
        ("left_fingers", left_fingers),
        ("right_fingers", right_fingers),
        ("left_wrists", left_wrists),
        ("right_wrists", right_wrists),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"state.{name} contains NaN or Inf")
    return torso, left_wrists, right_wrists, left_fingers, right_fingers


# --------------------------------------------------------------------------- #
# Observation: simulator dict -> raw51 / wire numeric
# --------------------------------------------------------------------------- #


def build_raw_state(state: Mapping[str, Any], calibration: Calibration | None = None) -> np.ndarray:
    """Convert a simulator observation mapping into the native raw-51 order."""
    calib = calibration or Calibration()
    if not isinstance(state, Mapping):
        return _vector(state, name="state", size=layout.RAW_STATE_DIM).copy()

    legs = _vector(state.get("legs_joint_pos"), name="state.legs_joint_pos", size=12)
    torso, left_wrists, right_wrists, left_fingers, right_fingers = split_upper_body(state)

    velocity = (
        _vector(state["velocity"], name="state.velocity", size=6)
        if "velocity" in state
        else np.concatenate(
            [
                _vector(state.get("base_lin_vel", (0.0, 0.0, 0.0)), name="state.base_lin_vel", size=3),
                _vector(state.get("base_ang_vel", (0.0, 0.0, 0.0)), name="state.base_ang_vel", size=3),
            ]
        )
    )
    root_height = float(state.get("root_height", DEFAULT_ROOT_HEIGHT))
    if not np.isfinite(root_height):
        raise ValueError("state.root_height must be finite")
    roll, pitch, yaw = roll_pitch_yaw(state)

    raw = np.zeros(layout.RAW_STATE_DIM, dtype=np.float64)
    raw[0:12] = legs
    raw[12:25] = torso
    raw[25:27] = left_wrists
    raw[27:29] = right_wrists
    raw[29:35] = left_fingers
    raw[35:41] = right_fingers
    raw[41:45] = (roll, pitch, root_height, yaw)
    raw[45:51] = velocity

    raw = apply_state_calibration(raw, calib)
    return raw


def apply_state_calibration(raw: np.ndarray, calibration: Calibration) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64).copy()
    if calibration.swap_wrists:
        raw[[25, 26]] = raw[[26, 25]]
        raw[[27, 28]] = raw[[28, 27]]
    if calibration.invert_r_wrist_roll:
        raw[27] = -raw[27]
    if calibration.invert_r_thumb_yaw:
        raw[40] = -raw[40]
    if calibration.thumb_hw:
        for index, (scale, offset) in layout.THUMB_HW.items():
            raw[index] = (float(raw[index]) - offset) / scale
    return raw


def state_to_wire_numeric(state: Mapping[str, Any], calibration: Calibration | None = None) -> np.ndarray:
    """The observation expressed as a wire command row.

    Used to seat the velocity/acceleration limiter on the pose the robot is
    holding, and to measure the tracking error between a command and the next
    observation.
    """
    raw = build_raw_state(state, calibration)
    return layout.raw_to_wire(raw)


# --------------------------------------------------------------------------- #
# Observation: command echo decoding (PROPRIO_CMD=all)
# --------------------------------------------------------------------------- #


def fingers_from_state(state: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    """Return ``(fingers, interleaved)`` from an observation state."""
    fingers = np.asarray(state.get("finger_joint_pos"), dtype=np.float64).reshape(-1)
    if fingers.shape == (16,):
        return fingers, True
    if fingers.shape == (12,):
        return fingers, False
    raise ValueError(f"unexpected finger state shape {fingers.shape}")


def echoed_operation_matches(
    state: Mapping[str, Any], last_wire: Mapping[str, Any] | None
) -> bool:
    """Whether ``state`` is the simulator echoing our last command.

    With ``PROPRIO_CMD=all`` GreenSim sends the policy its own last wire
    command as proprioception instead of the measured joints.  The torso is
    published absolute in both modes, so an exact 13-vector match is decisive;
    measured joints virtually never coincide to 1e-6 rad on all 13 channels.
    """
    if not last_wire:
        return False
    try:
        torso = np.asarray(state.get("torso_joint_pos"), dtype=np.float64).reshape(-1)
        previous = np.asarray(last_wire.get("torso_joint_pos"), dtype=np.float64).reshape(-1)
        fingers, interleaved = fingers_from_state(state)
        prev_fingers = np.asarray(last_wire.get("finger_joint_pos"), dtype=np.float64).reshape(-1)
        prev_wrists = np.asarray(last_wire.get("wrist_joint_pos"), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    if torso.shape != previous.shape or prev_fingers.shape != (12,) or prev_wrists.shape != (4,):
        return False
    if not np.allclose(torso, previous, atol=_ECHO_ATOL, rtol=0.0):
        return False
    if interleaved:
        expected = np.concatenate(
            [prev_fingers[0:6], prev_wrists[0:2], prev_fingers[6:12], prev_wrists[2:4]]
        )
    else:
        expected = prev_fingers
    return bool(np.allclose(fingers, expected, atol=_ECHO_ATOL, rtol=0.0))


def decode_proprioception(
    observation: Mapping[str, Any],
    *,
    proprio: str,
    calibration: Calibration,
    last_wire: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Undo the simulator's command echo before forwarding to the model.

    ``proprio="all"`` mirrors the packaged deployment (always decode when a
    previous command exists), ``"measured"`` disables decoding, and ``"auto"``
    detects the echo from an exact torso match.  On the first request of an
    episode there is no previous command, and the simulator itself sends the
    measured pose, so no decoding happens in any mode.
    """
    if proprio not in ("all", "measured", "auto"):
        raise ValueError(f"unknown proprio mode {proprio!r}")
    if proprio == "measured":
        return observation
    if last_wire is None:
        return observation
    if proprio == "auto" and not echoed_operation_matches(
        observation.get("state", {}), last_wire
    ):
        return observation
    if "state" not in observation:
        return observation

    state = dict(observation["state"])
    try:
        fingers, interleaved = fingers_from_state(state)
    except ValueError:
        return observation

    defaults = calibration.finger_defaults
    decoded = fingers.copy()
    if interleaved:
        decoded[np.r_[0:6, 8:14]] += defaults
    else:
        decoded += defaults
    state["finger_joint_pos"] = decoded

    torso = np.asarray(state.get("torso_joint_pos"), dtype=np.float64).reshape(-1)
    if torso.shape == (13,) and calibration.neck_bias:
        # Optional empirical neck correction is stored on the wire command;
        # remove it again so the model sees its own coordinates.
        torso = torso.copy()
        torso[12] -= calibration.neck_bias
        state["torso_joint_pos"] = torso

    result = dict(observation)
    result["state"] = state
    return result


# --------------------------------------------------------------------------- #
# Actions: model50 -> wire
# --------------------------------------------------------------------------- #


def model50_to_wire_numeric(
    action: np.ndarray,
    calibration: Calibration | None = None,
    *,
    zero_velocity: bool = False,
) -> np.ndarray:
    """Map one ``model50`` action to a numeric wire row in absolute coordinates."""
    calib = calibration or Calibration()
    model = _vector(action, name="action", size=layout.MODEL_DIM)
    raw = np.zeros(layout.RAW_STATE_DIM, dtype=np.float64)
    raw[layout.MODEL_TO_RAW_51] = model
    raw[44] = 0.0  # global root yaw: dropped during training, never predicted
    if zero_velocity:
        raw[45:51] = 0.0

    if calib.invert_r_thumb_yaw:
        raw[40] = -raw[40]
    if calib.thumb_hw:
        for index, (scale, offset) in layout.THUMB_HW.items():
            raw[index] = scale * float(raw[index]) + offset
    if calib.swap_wrists:
        raw[[25, 26, 27, 28]] = raw[[26, 25, 28, 27]]
    if calib.invert_r_wrist_roll:
        raw[28] = -raw[28]

    return layout.raw_to_wire(raw)


def model50_chunk_to_wire_numeric(
    actions: np.ndarray,
    calibration: Calibration | None = None,
    *,
    zero_velocity: bool = False,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != layout.MODEL_DIM:
        raise ValueError(f"actions must be (N, {layout.MODEL_DIM}), got {actions.shape}")
    rows = [model50_to_wire_numeric(a, calibration, zero_velocity=zero_velocity) for a in actions]
    return np.stack(rows, axis=0)


def numeric_to_wire_row(values: np.ndarray, calibration: Calibration | None = None) -> dict:
    """Numeric row (absolute coordinates) -> the dict the simulator executes.

    Finger commands are converted to *relative* targets here because the
    simulator adds ``default_joint_pos`` back inside its position action.
    """
    calib = calibration or Calibration()
    values = _vector(values, name="wire row", size=layout.WIRE_DIM)
    num_fingers = values[layout.WIRE_SLICES["finger_joint_pos"]] - calib.finger_defaults
    torso = values[layout.WIRE_SLICES["torso_joint_pos"]].copy()
    if calib.neck_bias:
        torso[12] = float(np.clip(torso[12] + calib.neck_bias, *calib.neck_limits))
    base = values[layout.WIRE_SLICES["base_command"]]
    return {
        "legs_joint_pos": values[layout.WIRE_SLICES["legs_joint_pos"]].tolist(),
        "torso_joint_pos": torso.tolist(),
        "finger_joint_pos": num_fingers.tolist(),
        "wrist_joint_pos": values[layout.WIRE_SLICES["wrist_joint_pos"]].tolist(),
        "velocity": values[layout.WIRE_SLICES["velocity"]].tolist(),
        "base_command": {
            "root_height": float(base[0]),
            "roll": float(base[1]),
            "pitch": float(base[2]),
            "yaw": float(base[3]),
        },
    }


def wire_row_to_numeric(row: Mapping[str, Any], *, require_base: bool = True) -> np.ndarray:
    """Inverse of :func:`numeric_to_wire_row` for a model/server row.

    Missing ``velocity`` or ``base_command`` entries are filled with zeros:
    some checkpoints emit only the 41 joint channels, and a partial row is
    still usable because the feed never clamps those channels.
    """
    values = np.zeros(layout.WIRE_DIM, dtype=np.float64)
    for name, size in layout.WIRE_FIELDS:
        source = row.get(name)
        if source is None and name in ("velocity",):
            continue
        if not isinstance(source, (list, tuple, np.ndarray)) or len(source) != size:
            raise ValueError(f"action row {name!r} must contain {size} values")
        values[layout.WIRE_SLICES[name]] = np.asarray(source, dtype=np.float64)
    base = row.get("base_command")
    if base is None:
        if require_base:
            raise ValueError("action row 'base_command' must be a mapping")
    elif not isinstance(base, Mapping):
        raise ValueError("action row 'base_command' must be a mapping")
    else:
        values[47] = float(base.get("root_height", 0.0))
        values[48] = float(base.get("roll", 0.0))
        values[49] = float(base.get("pitch", 0.0))
        values[50] = float(base.get("yaw", 0.0))
    if not np.isfinite(values).all():
        raise ValueError("action row contains NaN or Inf")
    return values


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def choose_prompt(
    observation: Mapping[str, Any],
    fallback: str | None,
    *,
    from_subtask: bool,
) -> str:
    prompt = str(observation.get("prompt") or fallback or "").strip()
    if from_subtask:
        subtask = observation.get("subtask")
        if subtask:
            prompt = str(subtask).strip()
    if not prompt:
        raise ValueError("a non-empty prompt or subtask is required")
    return prompt
