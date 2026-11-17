import json

import pytest

from greenchallenge import config, upstream


def test_defaults_match_the_packaged_deployment():
    runtime = config.RuntimeConfig()
    assert runtime.stack == "official"
    assert runtime.feed.speed == 1.3
    assert runtime.feed.row_step() == pytest.approx(1 / 39)
    assert runtime.feed.period == 0.080
    assert runtime.boundary.prompt_from_subtask is True


def test_speed_scales_the_row_interval():
    runtime = config.RuntimeConfig()
    assert runtime.feed.row_step() == pytest.approx(1 / 30 / 1.3)
    fast = config.FeedConfig(row_dt=1 / 30, speed=1.5)
    assert fast.row_step() == pytest.approx(1 / 45)


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("POLICY_PORT", "9100")
    monkeypatch.setenv("POLICY_RAW_PORT", "9101")
    monkeypatch.setenv("GC_STACK", "boundary")
    monkeypatch.setenv("GC_SPEED", "1.5")
    monkeypatch.setenv("GC_ADAPTIVE_DETECTOR", "net")
    monkeypatch.setenv("GC_ADAPTIVE_SPEED", "1.8")
    runtime = config.RuntimeConfig.from_env()
    assert runtime.policy_port == 9100
    assert runtime.raw_port == 9101
    assert runtime.stack == "boundary"
    assert runtime.feed.speed == 1.5
    assert runtime.feed.adaptive_detector == "net"
    assert runtime.feed.adaptive_speed == 1.8
    json.dumps(runtime.describe())


def test_cli_overrides(monkeypatch):
    import os

    from greenchallenge.entrypoint import _apply_cli_overrides

    saved = {key: os.environ.get(key) for key in
             ("GC_DEVICE", "POLICY_PORT", "GC_SPEED", "GC_NECK_COMPENSATION")}
    try:
        _apply_cli_overrides(
            ["--device", "cpu", "--port", "9100", "--speed", "1.5",
             "--neck-compensation", "--not-a-real-flag", "x"]
        )
        assert os.environ["GC_DEVICE"] == "cpu"
        assert os.environ["POLICY_PORT"] == "9100"
        assert os.environ["GC_SPEED"] == "1.5"
        assert os.environ["GC_NECK_COMPENSATION"] == "1"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_bad_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("GC_SPEED", "fast")
    with pytest.raises(ValueError):
        config.RuntimeConfig.from_env()


def test_fixed_pose_overrides_are_refused(monkeypatch):
    for name, value in config._FORBIDDEN.items():
        neutral = next(iter(config._FORBIDDEN[name]))
        monkeypatch.setenv(name, neutral)
    assert config.compliance_violations() == []
    monkeypatch.setenv("S0S1_SQUAT_POSE_JSON", '{"left": [0.1]}')
    assert config.compliance_violations()
    with pytest.raises(RuntimeError):
        config.RuntimeConfig.from_env()


def test_raw_command_with_feed():
    argv = upstream.raw_policy_command(
        checkpoint="/ckpt",
        data_config="scripted_014814",
        host="127.0.0.1",
        port=8998,
        device="cuda:0",
        feed_by_server=True,
        row_dt=1 / 39,
        warmup_steps=2,
    )
    assert argv[:3] == [argv[0], "-m", "lerobot.scripts.obm_inference.serve_policy"]
    assert "--feed-by-server" in argv
    assert argv[argv.index("--feed.row-dt") + 1].startswith("0.025641")
    assert "--warmup-steps" in argv


def test_raw_command_without_feed_has_no_feed_flags():
    argv = upstream.raw_policy_command(
        checkpoint="/ckpt",
        data_config="scripted_014814",
        host="127.0.0.1",
        port=8998,
        device="cuda:0",
        feed_by_server=False,
        row_dt=None,
    )
    assert "--feed-by-server" not in argv
    assert "--feed.row-dt" not in argv


def test_tracking_command():
    argv = upstream.tracking_command(
        upstream_port=8998,
        port=8999,
        calibration="/app/assets/wire_calibration.json",
        joint_calibration="/app/assets/joint_tracking_calibration_v1.json",
        adapter="v1",
        neck_compensation=True,
        log="/tmp/tracking.jsonl",
    )
    assert "lerobot.scripts.obm_inference.serve_tracking" in argv
    assert argv[argv.index("--adapter") + 1] == "v1"
    assert "--neck-compensation" in argv
    assert argv[argv.index("--log") + 1] == "/tmp/tracking.jsonl"
