"""Joint layouts and wire constants for the Green Challenge humanoid.

Three coordinate systems appear in a Green Challenge submission and mixing
them up is the fastest way to score zero while the logs look healthy:

``raw51``
    ``observation.state`` in the LeRobot dataset, and the observation the
    simulator sends over the wire.  51 channels: 41 joints (legs, torso,
    arms, neck, wrists, fingers) + 4 root/odometry values + 6 velocities.

``model50``
    What GreenVLA v1.1 predicts.  Raw index 44 (global root yaw) is dropped
    because it was never learned; wrists and fingers are reordered so that
    each hand is contiguous (6 fingers + 2 wrists).

``wire``
    The per-tick command row the simulator expects: legs 12, torso 13,
    fingers 12, wrists 4, velocity 6, ``base_command`` 4.  The fingers are
    *relative* to the robot's default finger pose because the simulator's
    ``JointPositionActionCfg`` sets ``use_default_offset=True``.

The numeric encoding used inside this package flattens ``wire`` into 51
float64 columns in exactly that order, so a plan can be scheduled, blended
and limited without touching Python dictionaries.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #

RAW_STATE_DIM = 51
MODEL_DIM = 50

#: ``model50`` index -> ``raw51`` index.  Taken from the reference server
#: (``policy_adapter.MODEL_TO_RAW_51``); any change here changes the meaning
#: of every checkpoint, so it is pinned by a unit test.
MODEL_TO_RAW_51 = np.asarray(
    (
        *range(0, 25),          # legs, torso yaw, arms, neck
        43,                     # root height
        45, 46, 47, 48, 49, 50,  # base linear + angular velocity
        41, 42,                 # body roll, pitch
        29, 30, 31, 32, 33, 34,  # left fingers
        25, 26,                 # left wrist
        35, 36, 37, 38, 39, 40,  # right fingers
        27, 28,                 # right wrist
    ),
    dtype=np.int64,
)

#: ``raw51`` index -> ``model50`` index, ``-1`` where the channel is dropped.
RAW_TO_MODEL = np.full(RAW_STATE_DIM, -1, dtype=np.int64)
RAW_TO_MODEL[MODEL_TO_RAW_51] = np.arange(MODEL_DIM, dtype=np.int64)

# --------------------------------------------------------------------------- #
# Joint groups, in model50 coordinates
# --------------------------------------------------------------------------- #

MODEL_GROUPS: dict[str, slice] = {
    "legs": slice(0, 12),
    "torso_yaw": slice(12, 13),
    "left_arm": slice(13, 18),
    "right_arm": slice(18, 23),
    "neck": slice(23, 25),
    "root_height": slice(25, 26),
    "base_velocity": slice(26, 32),
    "body_attitude": slice(32, 34),
    "left_fingers": slice(34, 40),
    "left_wrist": slice(40, 42),
    "right_fingers": slice(42, 48),
    "right_wrist": slice(48, 50),
}

#: Channels the model predicts as a delta to the input state.  The rest
#: (root height, base twist, roll, pitch) are absolute.
MODEL_DELTA_SLICES: tuple[slice, ...] = (slice(0, 25), slice(34, 50))
MODEL_ABSOLUTE_SLICES: tuple[slice, ...] = (slice(25, 34),)


def delta_mask() -> np.ndarray:
    """Boolean mask over ``model50``: True where actions are deltas."""
    mask = np.zeros(MODEL_DIM, dtype=bool)
    for sl in MODEL_DELTA_SLICES:
        mask[sl] = True
    return mask


# --------------------------------------------------------------------------- #
# Wire command row (numeric encoding used by the feed)
# --------------------------------------------------------------------------- #

WIRE_FIELDS: tuple[tuple[str, int], ...] = (
    ("legs_joint_pos", 12),
    ("torso_joint_pos", 13),
    ("finger_joint_pos", 12),
    ("wrist_joint_pos", 4),
    ("velocity", 6),
)

WIRE_BASE_FIELDS: tuple[str, ...] = ("root_height", "roll", "pitch", "yaw")

WIRE_SLICES: dict[str, slice] = {}
_offset = 0
for _name, _size in WIRE_FIELDS:
    WIRE_SLICES[_name] = slice(_offset, _offset + _size)
    _offset += _size
WIRE_SLICES["base_command"] = slice(_offset, _offset + len(WIRE_BASE_FIELDS))
WIRE_DIM = _offset + len(WIRE_BASE_FIELDS)  # 51

# --------------------------------------------------------------------------- #
# Raw51 slices
# --------------------------------------------------------------------------- #

RAW_SLICES: dict[str, slice] = {
    "legs": slice(0, 12),
    "torso": slice(12, 25),
    "left_wrist": slice(25, 27),
    "right_wrist": slice(27, 29),
    "left_fingers": slice(29, 35),
    "right_fingers": slice(35, 41),
    "roll": slice(41, 42),
    "pitch": slice(42, 43),
    "root_height": slice(43, 44),
    "yaw": slice(44, 45),
    "velocity": slice(45, 51),
}

#: ``observation.state`` masks in the dataset: indices that are actually
#: measured for the scripted/teleop S0S1 embodiment.
STATE_VALID_INDICES = np.arange(RAW_STATE_DIM, dtype=np.int64)
ACTION_VALID_INDICES = np.arange(RAW_STATE_DIM, dtype=np.int64)

# --------------------------------------------------------------------------- #
# Hand calibration
# --------------------------------------------------------------------------- #

#: Default finger pose the simulator adds to relative finger actions
#: (``green_full.interface.json`` -> ``init_state.joint_pos``).  Only the
#: thumb yaw is non-zero; the rest is kept explicit for clarity.
FINGER_DEFAULT_POSITION = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
    dtype=np.float64,
)

#: S0/S1 hardware/simulator thumb calibration, ``sim = scale * raw + offset``.
#: Disabled by default; enabling it changes the meaning of the checkpoint.
THUMB_HW = {
    33: (0.998, 0.0),
    34: (0.6464, 0.2596),
    39: (0.672, 0.046),
    40: (0.6014, 0.2781),
}

JOINT_NAMES_41 = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "torso_yaw",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_yaw",
    "neck_yaw",
    "neck_pitch",
    "left_wrist_crank",
    "left_wrist_roll",
    "right_wrist_crank",
    "right_wrist_roll",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb_pitch",
    "left_thumb_yaw",
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb_pitch",
    "right_thumb_yaw",
)


def wire_to_raw(numeric: np.ndarray) -> np.ndarray:
    """Map ``(..., 51)`` numeric wire rows to ``raw51`` vectors."""
    numeric = np.asarray(numeric, dtype=np.float64)
    raw = np.zeros(numeric.shape[:-1] + (RAW_STATE_DIM,), dtype=np.float64)
    raw[..., 0:12] = numeric[..., WIRE_SLICES["legs_joint_pos"]]
    raw[..., 12:25] = numeric[..., WIRE_SLICES["torso_joint_pos"]]
    raw[..., 25:27] = numeric[..., 37:39]
    raw[..., 27:29] = numeric[..., 39:41]
    raw[..., 29:35] = numeric[..., 25:31]
    raw[..., 35:41] = numeric[..., 31:37]
    raw[..., 41] = numeric[..., 48]
    raw[..., 42] = numeric[..., 49]
    raw[..., 43] = numeric[..., 47]
    raw[..., 44] = numeric[..., 50]
    raw[..., 45:51] = numeric[..., WIRE_SLICES["velocity"]]
    return raw


def raw_to_wire(raw: np.ndarray) -> np.ndarray:
    """Map ``(..., 51)`` ``raw51`` vectors to numeric wire rows."""
    raw = np.asarray(raw, dtype=np.float64)
    numeric = np.zeros(raw.shape[:-1] + (WIRE_DIM,), dtype=np.float64)
    numeric[..., WIRE_SLICES["legs_joint_pos"]] = raw[..., 0:12]
    numeric[..., WIRE_SLICES["torso_joint_pos"]] = raw[..., 12:25]
    numeric[..., 25:31] = raw[..., 29:35]
    numeric[..., 31:37] = raw[..., 35:41]
    numeric[..., 37:39] = raw[..., 25:27]
    numeric[..., 39:41] = raw[..., 27:29]
    numeric[..., WIRE_SLICES["velocity"]] = raw[..., 45:51]
    numeric[..., 47] = raw[..., 43]
    numeric[..., 48] = raw[..., 41]
    numeric[..., 49] = raw[..., 42]
    numeric[..., 50] = raw[..., 44]
    return numeric


def model_to_wire(model: np.ndarray) -> np.ndarray:
    """Map ``(..., 50)`` model-space actions to wire rows (raw 44 neutral)."""
    model = np.asarray(model, dtype=np.float64)
    raw = np.zeros(model.shape[:-1] + (RAW_STATE_DIM,), dtype=np.float64)
    raw[..., MODEL_TO_RAW_51] = model
    raw[..., 44] = 0.0
    return raw_to_wire(raw)
