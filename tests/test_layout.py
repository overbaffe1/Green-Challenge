import numpy as np
import pytest

from greenchallenge import layout


def test_dimensions():
    assert layout.WIRE_DIM == 51
    assert layout.RAW_STATE_DIM == 51
    assert layout.MODEL_DIM == 50
    assert sum(s.stop - s.start for s in layout.WIRE_SLICES.values()) == layout.WIRE_DIM


def test_model_to_raw_is_a_permutation_of_50_of_51():
    assert layout.MODEL_TO_RAW_51.shape == (50,)
    assert len(set(layout.MODEL_TO_RAW_51.tolist())) == 50
    assert 44 not in layout.MODEL_TO_RAW_51.tolist()
    np.testing.assert_array_equal(layout.RAW_TO_MODEL[layout.MODEL_TO_RAW_51],
                                  np.arange(layout.MODEL_DIM))
    assert layout.RAW_TO_MODEL[44] == -1


def test_wire_layout_matches_the_simulator():
    assert layout.WIRE_SLICES["legs_joint_pos"] == slice(0, 12)
    assert layout.WIRE_SLICES["torso_joint_pos"] == slice(12, 25)
    assert layout.WIRE_SLICES["finger_joint_pos"] == slice(25, 37)
    assert layout.WIRE_SLICES["wrist_joint_pos"] == slice(37, 41)
    assert layout.WIRE_SLICES["velocity"] == slice(41, 47)
    assert layout.WIRE_SLICES["base_command"] == slice(47, 51)


def test_wire_round_trip():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(6, 51))
    numeric = layout.raw_to_wire(raw)
    np.testing.assert_allclose(layout.wire_to_raw(numeric), raw, atol=0, rtol=0)


def test_raw_build_from_dataset_layout_order():
    raw = np.arange(51, dtype=float)
    numeric = layout.raw_to_wire(raw)
    np.testing.assert_array_equal(numeric[0:12], raw[0:12])       # legs
    np.testing.assert_array_equal(numeric[12:25], raw[12:25])     # torso
    np.testing.assert_array_equal(numeric[25:31], raw[29:35])     # left fingers
    np.testing.assert_array_equal(numeric[31:37], raw[35:41])     # right fingers
    np.testing.assert_array_equal(numeric[37:39], raw[25:27])     # left wrist
    np.testing.assert_array_equal(numeric[39:41], raw[27:29])     # right wrist
    np.testing.assert_array_equal(numeric[47:51], raw[[43, 41, 42, 44]])


def test_model_to_wire_drops_yaw_and_keeps_fingers_absolute():
    model = np.arange(50, dtype=float)
    numeric = layout.model_to_wire(model)
    raw = layout.wire_to_raw(numeric)
    np.testing.assert_array_equal(raw[layout.MODEL_TO_RAW_51], model)
    assert raw[44] == 0.0


def test_delta_mask_groups():
    mask = layout.delta_mask()
    assert mask[:25].all() and mask[34:50].all()
    assert not mask[25:34].any()


def test_finger_defaults_are_the_documented_pose():
    assert layout.FINGER_DEFAULT_POSITION.shape == (12,)
    assert layout.FINGER_DEFAULT_POSITION[5] == pytest.approx(-0.1)
    assert layout.FINGER_DEFAULT_POSITION[11] == pytest.approx(-0.1)
    assert np.count_nonzero(layout.FINGER_DEFAULT_POSITION) == 2
