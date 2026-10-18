import numpy as np
import pytest

from greenchallenge import contract, layout


def sim_state(**overrides):
    """The observation state dictionary the simulator sends."""
    state = {
        "legs_joint_pos": np.linspace(-0.2, 0.2, 12),
        "torso_joint_pos": np.linspace(0.0, 0.5, 13),
        "finger_joint_pos": np.zeros(16),   # 6 left + 2 wrist + 6 right + 2 wrist
        "base_lin_vel": np.array([0.01, -0.02, 0.03]),
        "base_ang_vel": np.array([0.0, 0.1, 0.0]),
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
        "root_height": 0.9,
        "root_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    state.update(overrides)
    return state


def test_build_raw_state_matches_the_simulator_fields():
    state = sim_state()
    raw = contract.build_raw_state(state)
    np.testing.assert_array_equal(raw[0:12], state["legs_joint_pos"])
    np.testing.assert_array_equal(raw[12:25], state["torso_joint_pos"])
    np.testing.assert_array_equal(raw[29:35], np.zeros(6))
    assert raw[43] == pytest.approx(0.9)
    np.testing.assert_allclose(raw[41:43], 0.0, atol=1e-12)


def test_build_raw_state_accepts_split_wrist_layout():
    state = sim_state()
    state.pop("finger_joint_pos")
    state["wrist_joint_pos"] = np.array([0.1, 0.2, 0.3, 0.4])
    state["finger_joint_pos"] = np.full(12, 0.5)
    raw = contract.build_raw_state(state)
    np.testing.assert_allclose(raw[25:27], [0.2, 0.1])  # swapped wrists by default
    np.testing.assert_allclose(raw[27:29], [0.4, 0.3])
    np.testing.assert_allclose(raw[29:35], 0.5)


def test_build_raw_state_accepts_17_torso_layout():
    state = sim_state()
    state["torso_joint_pos"] = np.arange(17, dtype=float)
    state["finger_joint_pos"] = np.zeros(12)
    raw = contract.build_raw_state(state)
    np.testing.assert_array_equal(raw[12:25], np.arange(13, dtype=float))
    np.testing.assert_array_equal(raw[25:29], np.array([14.0, 13.0, 16.0, 15.0]))


def test_echo_decoding_adds_the_default_finger_pose():
    state = sim_state(subtask="Raise your right hand")
    state["finger_joint_pos"] = np.zeros(16)  # echo of "fully relative" command
    last = contract.numeric_to_wire_row(_absolute_row())
    decoded = contract.decode_proprioception(
        {"state": state}, proprio="all", calibration=contract.Calibration(), last_wire=last
    )
    fingers = np.asarray(decoded["state"]["finger_joint_pos"])
    expected = np.zeros(16)
    expected[[5, 13]] = -0.1
    np.testing.assert_allclose(fingers, expected)
    # torso passes through untouched when no neck bias is configured
    np.testing.assert_array_equal(decoded["state"]["torso_joint_pos"], state["torso_joint_pos"])
    # the caller's observation is not mutated
    assert np.allclose(state["finger_joint_pos"], 0.0)


def test_measured_mode_forwards_the_observation_unchanged():
    state = sim_state()
    last = contract.numeric_to_wire_row(_absolute_row())
    decoded = contract.decode_proprioception(
        {"state": state}, proprio="measured", calibration=contract.Calibration(), last_wire=last
    )
    assert decoded is not None
    assert np.array_equal(decoded["state"]["finger_joint_pos"], state["finger_joint_pos"])


def test_auto_mode_only_decodes_a_real_echo():
    last = contract.numeric_to_wire_row(_absolute_row())
    echoed = sim_state()
    echoed["torso_joint_pos"] = np.asarray(last["torso_joint_pos"], dtype=float)
    echoed["finger_joint_pos"] = _relative_fingers(last)
    decoded_echo = contract.decode_proprioception(
        {"state": echoed}, proprio="auto", calibration=contract.Calibration(), last_wire=last
    )
    assert not np.array_equal(decoded_echo["state"]["finger_joint_pos"], echoed["finger_joint_pos"])

    measured = sim_state()
    decoded_measured = contract.decode_proprioception(
        {"state": measured}, proprio="auto", calibration=contract.Calibration(), last_wire=last
    )
    assert np.array_equal(decoded_measured["state"]["finger_joint_pos"], measured["finger_joint_pos"])


def test_wire_row_numeric_round_trip():
    numeric = _absolute_row()
    row = contract.numeric_to_wire_row(numeric)
    # ``numeric_to_wire_row`` subtracts the finger defaults because the
    # simulator adds them back; the reverse direction consumes a model-server
    # row, which is already absolute.  Compare the absolute channels and the
    # finger channel with the offset restored.
    back = contract.wire_row_to_numeric(row)
    np.testing.assert_allclose(
        back[layout.WIRE_SLICES["legs_joint_pos"]],
        numeric[layout.WIRE_SLICES["legs_joint_pos"]],
    )
    np.testing.assert_allclose(
        back[layout.WIRE_SLICES["finger_joint_pos"]] + layout.FINGER_DEFAULT_POSITION,
        numeric[layout.WIRE_SLICES["finger_joint_pos"]],
    )
    np.testing.assert_allclose(
        row["finger_joint_pos"],
        numeric[layout.WIRE_SLICES["finger_joint_pos"]] - layout.FINGER_DEFAULT_POSITION,
    )


def test_relative_fingers_from_the_model_are_restored():
    numeric = _absolute_row()
    raw = layout.wire_to_raw(numeric)
    assert raw[29] == pytest.approx(numeric[25])


def test_model_chunk_mapping_shape_and_velocity_zeroing():
    chunk = np.zeros((50, layout.MODEL_DIM))
    numeric = contract.model50_chunk_to_wire_numeric(chunk, zero_velocity=True)
    assert numeric.shape == (50, layout.WIRE_DIM)
    np.testing.assert_allclose(numeric[:, layout.WIRE_SLICES["velocity"]], 0.0)
    np.testing.assert_allclose(numeric[:, layout.WIRE_SLICES["base_command"]][:, 0], 0.0)


def test_prompt_prefers_subtask_when_asked():
    observation = {"prompt": "Take the bowl", "subtask": "Raise your right hand"}
    assert contract.choose_prompt(observation, None, from_subtask=True) == "Raise your right hand"
    assert contract.choose_prompt(observation, None, from_subtask=False) == "Take the bowl"
    assert contract.choose_prompt({"subtask": ""}, "fallback", from_subtask=True) == "fallback"
    with pytest.raises(ValueError):
        contract.choose_prompt({}, None, from_subtask=True)


# --------------------------------------------------------------------------- #


def _absolute_row() -> np.ndarray:
    numeric = np.zeros(layout.WIRE_DIM)
    numeric[layout.WIRE_SLICES["legs_joint_pos"]] = np.linspace(0.0, 0.3, 12)
    numeric[layout.WIRE_SLICES["torso_joint_pos"]] = np.linspace(-0.1, 0.4, 13)
    numeric[layout.WIRE_SLICES["finger_joint_pos"]] = np.linspace(-0.2, 0.9, 12)
    numeric[layout.WIRE_SLICES["wrist_joint_pos"]] = np.array([0.1, -0.2, 0.3, -0.4])
    numeric[layout.WIRE_SLICES["velocity"]] = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.02])
    numeric[layout.WIRE_SLICES["base_command"]] = np.array([0.88, 0.01, -0.02, 0.03])
    return numeric


def _relative_fingers(row):
    fingers = np.asarray(row["finger_joint_pos"], dtype=float)
    wrists = np.asarray(row["wrist_joint_pos"], dtype=float)
    return np.concatenate([fingers[0:6], wrists[0:2], fingers[6:12], wrists[2:4]])
