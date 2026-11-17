"""Container entrypoint.

Two stacks are available behind the same contract:

``official`` (default)
    The organisers' own two-process topology: ``serve_policy`` on loopback
    ``POLICY_RAW_PORT`` with ``--feed-by-server`` and a row interval tuned by
    ``GC_SPEED``, then ``serve_tracking`` on ``POLICY_PORT``.  This changes only
    inference parameters (playback rate and, optionally, the packaged adapter)
    and is the lowest-risk submission.

``boundary``
    ``serve_policy`` without a feed, plus this package's execution layer on the
    public port.  It adds the expiry and adaptive-speed experiments and is what
    the CPU tests exercise; use it only after an A/B run shows a gain.

Both stacks are supervised here: if either child exits, the container exits, so
a healthy proxy never hides a dead model.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from . import __version__
from .boundary import Boundary
from .config import RuntimeConfig
from .metrics import JsonlLogger
from .upstream import raw_policy_command, tracking_command, UpstreamClient
from .wire import WebSocketPolicyServer

LOGGER = logging.getLogger("greenchallenge")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def _assets_dir() -> Path:
    """Locate the packaged wire calibration shipped with the lerobot fork."""
    candidates = []
    try:
        import lerobot  # type: ignore

        candidates.append(Path(lerobot.__file__).resolve().parent)
    except Exception:  # pragma: no cover - only when the fork is absent
        pass
    candidates.append(Path("/app/lerobot"))
    for root in candidates:
        assets = root / "common/robot_safety/tracking/assets"
        if (assets / "wire_calibration.json").is_file():
            return assets
    return Path("/app/lerobot/common/robot_safety/tracking/assets")


def _wait_http(port: int, timeout: float, processes: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"child exited before readiness: {process.args}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        threading.Event().wait(0.5)
    raise TimeoutError(f"no healthy endpoint on 127.0.0.1:{port} after {timeout:.0f} s")


def _environment(config: RuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        # The checkpoint was trained with subtask instructions and swapped
        # wrists.  These are model-contract settings, not tuning knobs.
        "S0S1_PROMPT_FROM_SUBTASK": "1",
        "S0S1_SWAP_WRISTS": "1",
        "S0S1_GMM_GUARD_MODE": "off",
        "S0S1_DATA_CONFIG": config.data_config,
        "TRITON_CACHE_DIR": "/tmp/triton-cache",
        "XDG_CACHE_HOME": "/tmp/runtime-cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torch-cache",
        "PYTHONUNBUFFERED": "1",
        # The evaluation container has no internet; offline mode fails fast
        # on a missing asset instead of waiting on a socket.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for name, value in defaults.items():
        env.setdefault(name, value)
    return env


def _spawn(argv: list[str], env: dict[str, str], log_prefix: str) -> subprocess.Popen:
    LOGGER.info("starting %s: %s", log_prefix, " ".join(argv))
    return subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        shell=False,
        start_new_session=False,
    )


def _stop_all(processes: list[subprocess.Popen], timeout: float = 15.0) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_official(config: RuntimeConfig, stop: threading.Event) -> int:
    """The packaged topology, with the playback rate as the only change."""
    env = _environment(config)
    row_dt = config.feed.row_step()
    raw = raw_policy_command(
        checkpoint=config.checkpoint,
        data_config=config.data_config,
        host=config.raw_host,
        port=config.raw_port,
        device=config.device,
        feed_by_server=True,
        row_dt=row_dt,
        warmup_steps=2,
        inference_steps=config.inference_steps,
        extra_args=list(config.upstream_args),
    )
    assets = _assets_dir()
    tracking = tracking_command(
        upstream_port=config.raw_port,
        port=config.policy_port,
        calibration=assets / "wire_calibration.json",
        joint_calibration=assets / "joint_tracking_calibration_v1.json",
        adapter=config.adapter,
        neck_compensation=config.neck_compensation,
        log=config.telemetry_path or "/tmp/greenchallenge/tracking.jsonl",
    )
    LOGGER.info(
        "stack=official row_dt=%.6f (speed %.2fx) adapter=%s neck=%s",
        row_dt,
        config.feed.speed,
        config.adapter,
        config.neck_compensation,
    )
    processes: list[subprocess.Popen] = []
    try:
        processes.append(_spawn(raw, env, "raw policy"))
        _wait_http(config.raw_port, config.start_timeout, processes)
        processes.append(_spawn(tracking, env, "wire boundary"))
        _wait_http(config.policy_port, 120.0, processes)
        LOGGER.info("ready: ws://%s:%d", config.host, config.policy_port)
        while not stop.wait(0.5):
            for process in processes:
                code = process.poll()
                if code is not None:
                    LOGGER.error("child %s exited with code %s", process.args, code)
                    return 1
        return 0
    finally:
        _stop_all(processes)


def run_boundary(config: RuntimeConfig, stop: threading.Event) -> int:
    """Raw model on loopback, this package's action feed on the public port."""
    env = _environment(config)
    raw = raw_policy_command(
        checkpoint=config.checkpoint,
        data_config=config.data_config,
        host=config.raw_host,
        port=config.raw_port,
        device=config.device,
        feed_by_server=False,
        row_dt=None,
        warmup_steps=2,
        inference_steps=config.inference_steps,
        extra_args=list(config.upstream_args),
    )
    process = _spawn(raw, env, "raw policy")
    server = None
    client = None
    try:
        _wait_http(config.raw_port, config.start_timeout, [process])
        client = UpstreamClient(config.raw_host, config.raw_port)
        client.connect(retries=30, delay=1.0)
        telemetry = JsonlLogger(config.telemetry_path)
        boundary = Boundary(
            client,
            config=config.boundary,
            feed_config=config.feed,
            calibration=config.calibration,
            data_config=config.data_config,
            telemetry=telemetry,
            extra_metadata={"greenchallenge_version": __version__, "stack": "boundary"},
        )
        server = WebSocketPolicyServer(boundary, host=config.host, port=config.policy_port)
        thread = threading.Thread(target=server.serve_forever, name="policy-server", daemon=True)
        thread.start()
        if not server.wait_ready(30.0):
            raise RuntimeError("policy server did not bind within 30 s")
        LOGGER.info("ready: ws://%s:%d", config.host, config.policy_port)
        while not stop.wait(0.5):
            code = process.poll()
            if code is not None:
                LOGGER.error("raw policy exited with code %s", code)
                return 1
        return 0
    finally:
        if server is not None:
            server.shutdown()
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        _stop_all([process])


def _apply_cli_overrides(argv: list[str]) -> None:
    """Accept the subset of launcher flags the contest image documents.

    The harness runs the entrypoint without arguments; tolerating the packaged
    launcher's flags means an operator can reproduce a run without rebuilding.
    Unknown flags are logged and ignored rather than fatal, because the
    evaluation harness is not ours to change.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-config")
    parser.add_argument("--port", type=int)
    parser.add_argument("--raw-port", type=int)
    parser.add_argument("--adapter")
    parser.add_argument("--neck-compensation", action="store_true")
    parser.add_argument("--speed", type=float)
    known, unknown = parser.parse_known_args(argv)
    if unknown:
        LOGGER.warning("ignoring unrecognised arguments: %s", unknown)
    overrides = {
        "GC_DEVICE": known.device,
        "CHECKPOINT_PATH": known.checkpoint,
        "S0S1_DATA_CONFIG": known.data_config,
        "POLICY_PORT": known.port,
        "POLICY_RAW_PORT": known.raw_port,
        "GC_ADAPTER": known.adapter,
        "GC_SPEED": known.speed,
    }
    if known.port is not None:
        env_port = os.environ.get("POLICY_PORT")
        if env_port and env_port.strip() and int(env_port) != known.port:
            # The image healthcheck reads POLICY_PORT from the container
            # environment, not from this process, so a CLI port that disagrees
            # would leave the healthcheck watching the wrong socket.
            raise SystemExit(
                "--port must equal POLICY_PORT so the image healthcheck follows it"
            )
    for name, value in overrides.items():
        if value is not None:
            os.environ[name] = str(value)
    if known.neck_compensation:
        os.environ["GC_NECK_COMPENSATION"] = "1"


def main(argv: list[str] | None = None) -> int:
    _apply_cli_overrides(list(argv or []))
    config = RuntimeConfig.from_env()
    _configure_logging(config.log_level)
    LOGGER.info("greenchallenge %s", __version__)
    LOGGER.info("configuration: %s", config.describe())
    if config.stack not in ("official", "boundary"):
        LOGGER.error("GC_STACK must be official or boundary, got %r", config.stack)
        return 2

    stop = threading.Event()

    def _handle_signal(signum, _frame):
        LOGGER.info("received signal %s, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        if config.stack == "official":
            return run_official(config, stop)
        return run_boundary(config, stop)
    except Exception:
        LOGGER.exception("startup failed")
        return 1
    finally:
        LOGGER.info("shutdown complete")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
