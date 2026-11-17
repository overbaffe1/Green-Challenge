import numpy as np
import pytest

from greenchallenge import feed, layout


def chunk(value=0.0, rows=50):
    rows_array = np.full((rows, layout.WIRE_DIM), value, dtype=np.float64)
    rows_array[:, layout.WIRE_SLICES["base_command"]] = (0.87, 0.0, 0.0, 0.0)
    return rows_array


def joints(commands):
    """The 41 joint channels, excluding velocity and the base command."""
    return commands[..., : layout.WIRE_SLICES["velocity"].start]


def test_constant_chunk_is_reproduced_after_the_limiter_settles():
    config = feed.FeedConfig(speed=1.0, clamp=False, blend=False, expire_frozen=False)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    numeric.reset(chunk(0.0)[0])
    commands = numeric.update(0.0, chunk(0.25))
    # before row 0 is due the feed can only offer row 0; after that it is a
    # constant plan, so every command equals the row
    np.testing.assert_allclose(joints(commands), 0.25, atol=1e-12)


def test_schedule_matches_the_reference_formula():
    """Row j is due at lead + (j+1)*row_step, linearly interpolated."""
    row_dt, lead, speed = 1 / 30, 0.030, 1.0
    config = feed.FeedConfig(row_dt=row_dt, speed=speed, lead=lead, clamp=False, blend=False)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    numeric.reset()
    rows = np.zeros((50, layout.WIRE_DIM))
    rows[:, layout.WIRE_SLICES["wrist_joint_pos"].start] = np.arange(50) * 0.01
    commands = numeric.update(0.0, rows)
    t = np.arange(20) * 0.004
    x = np.clip((t - lead) / (row_dt / speed) - 1.0, 0.0, 49.0)
    expected = np.interp(x, np.arange(50), np.arange(50) * 0.01)
    np.testing.assert_allclose(commands[:, layout.WIRE_SLICES["wrist_joint_pos"].start], expected, atol=1e-12)


def test_speed_compresses_row_timing_but_not_the_reply_length():
    slow = feed.NumericActionFeed(
        feed.FeedConfig(speed=1.0, clamp=False, blend=False), dt=0.004
    )
    fast = feed.NumericActionFeed(
        feed.FeedConfig(speed=1.3, clamp=False, blend=False), dt=0.004
    )
    rows = np.zeros((50, layout.WIRE_DIM))
    rows[:, layout.WIRE_SLICES["wrist_joint_pos"].start] = np.arange(50) * 0.01
    slow_commands = slow.update(0.0, rows)
    fast_commands = fast.update(0.0, rows)
    assert slow_commands.shape == fast_commands.shape == (20, layout.WIRE_DIM)
    # at the same wall time the faster feed is further along the plan
    assert fast_commands[-1, 37] > slow_commands[-1, 37]


def test_reply_length_follows_period_and_control_dt():
    config = feed.FeedConfig(period=0.080)
    assert feed.NumericActionFeed(config, dt=0.004).ticks_per_period == 20
    assert feed.NumericActionFeed(config, dt=0.005).ticks_per_period == 16


def test_clamp_limits_velocity_and_acceleration():
    config = feed.FeedConfig(speed=1.0, blend=False)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    numeric.reset(chunk(0.0)[0])
    target = chunk(0.0)
    target[:, layout.WIRE_SLICES["torso_joint_pos"]] = 1.0   # arms jump
    commands = numeric.update(0.0, target)
    arm = commands[:, layout.WIRE_SLICES["torso_joint_pos"]]
    steps = np.diff(arm, axis=0)
    assert np.max(np.abs(steps)) <= config.max_speed_arm * 0.004 + 1e-9
    accel = np.diff(arm, n=2, axis=0)
    assert np.max(np.abs(accel)) <= config.max_acceleration_arm * 0.004 ** 2 + 1e-9
    # and it never overshoots the target
    assert np.all(arm <= 1.0 + 1e-12)


def test_clamp_is_seeded_on_the_current_pose():
    config = feed.FeedConfig(speed=1.0, clamp=True, blend=False)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    seed = chunk(0.0)[0].copy()
    seed[layout.WIRE_SLICES["torso_joint_pos"]] = 0.5
    numeric.reset(chunk(0.5)[0])
    commands = numeric.update(0.0, chunk(0.5))
    np.testing.assert_allclose(joints(commands), 0.5, atol=1e-12)


def test_blend_matches_the_reference_weighted_mean():
    """Reimplement the documented formula independently and compare."""
    config = feed.FeedConfig(speed=1.0, blend=True, clamp=False, blend_tau=1.0, blend_life=1.2)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    numeric.reset()
    plans = []
    t_obs = 0.0
    errors = 0.0
    for step in range(40):
        rows = chunk(0.0)
        rows[:, layout.WIRE_SLICES["wrist_joint_pos"].start + 1] = np.linspace(0, 1, 50) * (0.1 * step)
        commands = numeric.update(t_obs, rows)
        # a fresh plan is *pending* until its first row comes due; only then
        # does it take part in the blend
        effective_at = t_obs + (config.resolved_latency() if plans else 0.0)
        plans.append((t_obs, effective_at, rows))
        for tick, command in enumerate(commands):
            t = t_obs + tick * 0.004
            live = [
                (o, r) for (o, effective_at, r) in plans
                if effective_at <= t + 1e-12 and t - o <= config.blend_life + 1e-9
            ]
            weights = [np.exp(-(t - o) / config.blend_tau) for (o, _) in live]
            forecasts = [_evaluate(r, t - o, config) for (o, r) in live]
            expected = sum(w * f for w, f in zip(weights, forecasts)) / sum(weights)
            errors = max(errors, np.max(np.abs(command - expected)))
        t_obs = round(t_obs / 0.004) * 0.004 + 20 * 0.004
    assert errors < 1e-9


def test_the_active_plan_never_expires():
    config = feed.FeedConfig(speed=1.0, blend=True, clamp=False, blend_life=0.05)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    numeric.reset()
    numeric.update(0.0, chunk(0.1))
    numeric.update(0.5, chunk(0.2))
    commands = numeric.update(0.6, chunk(0.2))
    # 0.5 s with life 0.05 s leaves only the new plan in the blend; the
    # command stream has also moved past the new plan's first row
    np.testing.assert_allclose(joints(commands), 0.2, atol=1e-12)


def test_frozen_plans_can_be_expired():
    config = feed.FeedConfig(speed=1.0, blend=True, clamp=False, blend_life=5.0, expire_frozen=True)
    numeric = feed.NumericActionFeed(config, dt=0.004)
    numeric.reset()
    stale = chunk(0.0)
    stale[:, layout.WIRE_SLICES["wrist_joint_pos"].start] = 1.0
    numeric.update(0.0, stale)
    fresh = chunk(0.0)
    numeric.update(2.0, fresh)   # stale plan is now past its 1.697 s horizon
    commands = numeric.update(2.08, fresh)
    assert np.max(np.abs(commands[:, 37])) < 1e-6
    assert numeric.stats.frozen_dropped > 0


def test_backwards_clock_resets_the_feed():
    numeric = feed.NumericActionFeed(feed.FeedConfig(speed=1.0, clamp=False, blend=False), dt=0.004)
    numeric.reset()
    numeric.update(1.0, chunk(0.3))
    commands = numeric.update(0.0, chunk(0.1))
    np.testing.assert_allclose(joints(commands), 0.1, atol=1e-12)


def test_invalid_chunks_are_rejected():
    numeric = feed.NumericActionFeed(feed.FeedConfig(), dt=0.004)
    numeric.reset()
    with pytest.raises(ValueError):
        numeric.update(0.0, np.zeros((0, layout.WIRE_DIM)))
    with pytest.raises(ValueError):
        numeric.update(0.0, np.zeros((5, 50)))
    bad = chunk(0.0)
    bad[3, 5] = np.nan
    with pytest.raises(ValueError):
        numeric.update(0.0, bad)


def test_speed_validation():
    with pytest.raises(ValueError):
        feed.FeedConfig(speed=0.0).validate()
    with pytest.raises(ValueError):
        feed.FeedConfig(adaptive_speed=0.5).validate()
    with pytest.raises(ValueError):
        feed.FeedConfig(clamp_channels="nonsense").validate()


def test_adaptive_detector_marks_grasp_rows_and_slows_them():
    config = feed.FeedConfig(
        speed=1.3, adaptive_speed=1.5, adaptive_detector="net", clamp=False, blend=False
    )
    rows = np.zeros((50, layout.WIRE_DIM))
    rows[20:30, layout.WIRE_SLICES["finger_joint_pos"].start + 6 : layout.WIRE_SLICES["finger_joint_pos"].stop] = 0.9
    steps, slow_rows = feed._row_steps(rows, config)
    assert slow_rows > 0
    assert steps[20] == pytest.approx(config.row_dt)
    assert steps[0] == pytest.approx(config.row_step())


# --------------------------------------------------------------------------- #


def _evaluate(rows, elapsed, config):
    row_dt = config.row_dt
    lead = config.lead
    if elapsed <= row_dt + lead - 1e-9:
        return rows[0].copy()
    x = (elapsed - lead) / row_dt - 1.0
    if x >= len(rows) - 1:
        return rows[-1].copy()
    j = int(x)
    return rows[j] + (x - j) * (rows[j + 1] - rows[j])
