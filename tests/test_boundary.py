import numpy as np
import pytest

from greenchallenge import contract, layout
from greenchallenge.boundary import Boundary, BoundaryConfig
from greenchallenge.feed import FeedConfig


class FakeUpstream:
    metadata = {"policy": "fake", "action_horizon": 50, "action_sample_step": 1}

    def __init__(self):
        self.observations = []

    def step(self, observation):
        self.observations.append(observation)
        return {"actions_list": [self.row(i) for i in range(50)]}

    @staticmethod
    def row(index):
        torso = [0.0] * 13
        torso[12] = 0.1  # neck pitch
        return {
            "legs_joint_pos": [0.0] * 12,
            "torso_joint_pos": torso,
            "finger_joint_pos": [0.5 + 0.001 * index] * 12,
            "wrist_joint_pos": [0.0] * 4,
            "velocity": [0.0] * 6,
            "base_command": {"root_height": 0.87, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        }


def observation(t=0.0, reset=False, fingers=None, torso=None, subtask="Raise your right hand"):
    state = {
        "legs_joint_pos": np.zeros(12),
        "torso_joint_pos": np.zeros(13) if torso is None else np.asarray(torso, float),
        "finger_joint_pos": np.zeros(16) if fingers is None else np.asarray(fingers, float),
        "base_lin_vel": np.zeros(3),
        "base_ang_vel": np.zeros(3),
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
        "root_height": 0.87,
        "root_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    return {
        "state": state,
        "images": {"top_head": np.zeros((2, 2, 3), np.uint8)},
        "subtask": subtask,
        "prompt": "Take the bowl",
        "reset": reset,
        "t": t,
    }


def make_boundary(proprio="all", **feed_kwargs):
    defaults = dict(speed=1.0, clamp=False, blend=False)
    defaults.update(feed_kwargs)
    upstream = FakeUpstream()
    boundary = Boundary(
        upstream,
        config=BoundaryConfig(proprio=proprio),
        feed_config=FeedConfig(**defaults),
    )
    return boundary, upstream


def test_boundary_returns_one_period_of_wire_rows():
    boundary, upstream = make_boundary()
    result = boundary.step(observation(t=0.0, reset=True))
    rows = result["actions_list"]
    assert len(rows) == 20
    for key, size in layout.WIRE_FIELDS:
        assert len(rows[0][key]) == size
    assert set(rows[0]["base_command"]) >= {"root_height", "roll", "pitch"}
    # the first tick plays row 0, and the simulator's finger action is
    # relative to the default pose
    np.testing.assert_allclose(
        rows[0]["finger_joint_pos"],
        np.asarray(upstream.row(0)["finger_joint_pos"]) - layout.FINGER_DEFAULT_POSITION,
    )
    assert boundary.last_wire == rows[-1]
    assert boundary.requests == 1
    assert upstream.observations[0]["subtask"] == "Raise your right hand"


def test_boundary_decodes_a_proprio_command_echo():
    boundary, upstream = make_boundary(proprio="all")
    first = boundary.step(observation(t=0.0, reset=True))
    last = first["actions_list"][-1]

    echoed = observation(t=0.08)
    echoed["state"]["torso_joint_pos"] = np.asarray(last["torso_joint_pos"], float)
    fingers = np.asarray(last["finger_joint_pos"], float)
    wrists = np.asarray(last["wrist_joint_pos"], float)
    echoed["state"]["finger_joint_pos"] = np.concatenate(
        [fingers[0:6], wrists[0:2], fingers[6:12], wrists[2:4]]
    )
    boundary.step(echoed)
    forwarded = upstream.observations[-1]["state"]["finger_joint_pos"]
    expected = np.asarray(echoed["state"]["finger_joint_pos"], float).copy()
    expected[[5, 13]] += -0.1
    np.testing.assert_allclose(forwarded, expected)
    # the observation the caller passed was copied, not mutated
    assert not np.array_equal(echoed["state"]["finger_joint_pos"], forwarded)


def test_auto_mode_does_not_decode_a_measured_pose():
    boundary, upstream = make_boundary(proprio="auto")
    boundary.step(observation(t=0.0, reset=True))
    measured = observation(t=0.08)
    measured["state"]["finger_joint_pos"] = np.full(16, 0.2)
    boundary.step(measured)
    np.testing.assert_allclose(upstream.observations[-1]["state"]["finger_joint_pos"], 0.2)


def test_auto_mode_decodes_an_exact_echo():
    boundary, upstream = make_boundary(proprio="auto")
    first = boundary.step(observation(t=0.0, reset=True))
    last = first["actions_list"][-1]
    echo = observation(t=0.08)
    echo["state"]["torso_joint_pos"] = np.asarray(last["torso_joint_pos"], float)
    fingers = np.asarray(last["finger_joint_pos"], float)
    wrists = np.asarray(last["wrist_joint_pos"], float)
    echo["state"]["finger_joint_pos"] = np.concatenate(
        [fingers[0:6], wrists[0:2], fingers[6:12], wrists[2:4]]
    )
    boundary.step(echo)
    assert not np.array_equal(
        upstream.observations[-1]["state"]["finger_joint_pos"],
        echo["state"]["finger_joint_pos"],
    )


def test_reset_restarts_the_episode_and_seeds_the_limiter():
    boundary, _ = make_boundary(clamp=True)
    boundary.step(observation(t=0.0, reset=True))
    boundary.step(observation(t=0.08))
    boundary.step(observation(t=0.0, reset=True))
    assert boundary.resets == 2
    assert boundary.feed.last_t_obs == 0.0


def test_instruction_change_drops_old_plans():
    boundary, _ = make_boundary()
    boundary.step(observation(t=0.0, reset=True, subtask="Raise your right hand"))
    assert boundary.instruction_switches == 0
    boundary.step(observation(t=0.08, subtask="Grasp the bowl"))
    assert boundary.instruction_switches == 1
    assert boundary.last_prompt == "Grasp the bowl"


def test_clock_rewind_is_treated_as_a_reset():
    boundary, _ = make_boundary()
    boundary.step(observation(t=1.0, reset=True))
    boundary.step(observation(t=0.0))
    assert boundary.resets == 2


def test_invalid_upstream_rows_are_rejected():
    class BadUpstream(FakeUpstream):
        def step(self, observation):
            self.observations.append(observation)
            rows = [self.row(0)] * 50
            rows[3]["finger_joint_pos"] = [0.0] * 11
            return {"actions_list": rows}

    boundary = Boundary(BadUpstream(), feed_config=FeedConfig(speed=1.0, clamp=False, blend=False))
    with pytest.raises(ValueError):
        boundary.step(observation(t=0.0, reset=True))


def test_nan_upstream_rows_are_rejected():
    class NanUpstream(FakeUpstream):
        def step(self, observation):
            rows = [self.row(i) for i in range(50)]
            rows[0]["legs_joint_pos"] = [float("nan")] * 12
            return {"actions_list": rows}

    boundary = Boundary(NanUpstream(), feed_config=FeedConfig(speed=1.0, clamp=False, blend=False))
    with pytest.raises(ValueError):
        boundary.step(observation(t=0.0, reset=True))


def test_prompt_falls_back_to_the_task_when_subtasks_are_disabled():
    upstream = FakeUpstream()
    boundary = Boundary(
        upstream,
        config=BoundaryConfig(proprio="all", prompt_from_subtask=False),
        feed_config=FeedConfig(speed=1.0, clamp=False, blend=False),
    )
    boundary.step(observation(t=0.0, reset=True))
    assert upstream.observations[0]["subtask"] == "Raise your right hand"
    # ``choose_prompt`` is what decides; the observation is forwarded as-is
    assert contract.choose_prompt(upstream.observations[0], None, from_subtask=False) == "Take the bowl"
