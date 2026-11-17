"""Runtime configuration, entirely from environment variables.

The evaluation harness only guarantees ``POLICY_PORT`` and
``POLICY_RAW_PORT``; everything else is ours.  Environment variables were
chosen over a YAML file because a submission is a Docker image and changing
one number (``GC_SPEED``) should not require rebuilding it.

Every knob has a safe default matching the packaged baseline.  The
``compliance_violations`` check refuses to start if an environment variable
that substitutes fixed poses or clock-scheduled instructions is set - those
are disqualifying under the contest rules, and a submission should not be one
typo away from breaking them.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shlex
from typing import Any

from .boundary import BoundaryConfig
from .contract import Calibration
from .feed import FeedConfig

LOGGER = logging.getLogger(__name__)


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _tuple(name: str) -> tuple[str, ...]:
    value = os.environ.get(name, "")
    return tuple(part.strip() for part in value.split(",") if part.strip())


#: Settings of the reference server that replace model output with a fixed
#: pose or a clock-scheduled command.  Any non-neutral value is disqualifying
#: under the published rules, so the entrypoint refuses to start with them.
_FORBIDDEN = {
    "S0S1_TORSO_PITCH_OFFSET": {"", "0", "0.0"},
    "S0S1_ROOT_HEIGHT_OFFSET": {"", "0", "0.0"},
    "S0S1_LEG_SQUAT_FRAC": {"", "0", "0.0"},
    "S0S1_SQUAT_POSE_JSON": {""},
    "S0S1_ARM_OFFSETS": {""},
    "S0S1_FINGER_GAIN": {"", "1", "1.0"},
    "S0S1_THUMB_PITCH_MIN": {"", "0", "0.0"},
    "S0S1_PROMPT_SCHEDULE_JSON": {"", "[]"},
    "GC_SCRIPTED_TRAJECTORY": {""},
}


def compliance_violations(environ: dict[str, str] | None = None) -> list[tuple[str, str]]:
    environ = dict(os.environ if environ is None else environ)
    violations: list[tuple[str, str]] = []
    for name, neutral in _FORBIDDEN.items():
        if name in environ and environ[name].strip() not in neutral:
            violations.append((name, environ[name]))
    return violations


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    policy_port: int = 8999
    raw_port: int = 8998
    host: str = "0.0.0.0"
    raw_host: str = "127.0.0.1"
    checkpoint: str = "/ckpt"
    data_config: str = "scripted_014814"
    device: str = "cuda:0"
    stack: str = "official"
    adapter: str = "none"
    neck_compensation: bool = False
    upstream_args: tuple[str, ...] = ()
    inference_steps: int | None = None
    start_timeout: float = 600.0
    telemetry_path: str | None = "/tmp/greenchallenge/tracking.jsonl"
    log_level: str = "INFO"
    feed: FeedConfig = dataclasses.field(default_factory=FeedConfig)
    boundary: BoundaryConfig = dataclasses.field(default_factory=BoundaryConfig)
    calibration: Calibration = dataclasses.field(default_factory=Calibration)
    zero_velocity: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        violations = compliance_violations()
        if violations:
            rendered = ", ".join(f"{name}={value!r}" for name, value in violations)
            raise RuntimeError(
                "refusing to start: fixed-pose/scripted overrides are set and are "
                f"disqualifying under the contest rules ({rendered})"
            )

        data_config = _str("GC_DATA_CONFIG", _str("S0S1_DATA_CONFIG", "scripted_014814"))
        feed = FeedConfig(
            row_dt=_float("GC_ROW_DT", 1.0 / 30.0),
            speed=_float("GC_SPEED", 1.30),
            lead=_float("GC_LEAD", 0.030),
            period=_float("GC_PERIOD", 0.080),
            latency=_optional_float("GC_LATENCY"),
            blend=_bool("GC_BLEND", True),
            blend_tau=_float("GC_BLEND_TAU", 1.0),
            blend_life=_float("GC_BLEND_LIFE", 1.20),
            blend_skip=_tuple("GC_BLEND_SKIP"),
            clamp=_bool("GC_CLAMP", True),
            clamp_channels=_str("GC_CLAMP_CHANNELS", "both"),
            max_speed_fingers=_float("GC_MAX_SPEED_FINGERS", 10.1),
            max_speed_arm=_float("GC_MAX_SPEED_ARM", 3.8),
            max_speed_neck=_float("GC_MAX_SPEED_NECK", 4.5),
            max_speed_legs=_float("GC_MAX_SPEED_LEGS", 3.0),
            max_acceleration_fingers=_float("GC_MAX_ACCEL_FINGERS", 30.1),
            max_acceleration_arm=_float("GC_MAX_ACCEL_ARM", 3.8),
            max_acceleration_neck=_float("GC_MAX_ACCEL_NECK", 10.5),
            max_acceleration_legs=_float("GC_MAX_ACCEL_LEGS", 3.0),
            expire_frozen=_bool("GC_EXPIRE_FROZEN", True),
            adaptive_speed=_float("GC_ADAPTIVE_SPEED", 1.0),
            adaptive_detector=_str("GC_ADAPTIVE_DETECTOR", "level"),
            adaptive_threshold=_float("GC_ADAPTIVE_THRESHOLD", 0.15),
            adaptive_travel_threshold=_float("GC_ADAPTIVE_TRAVEL_THRESHOLD", 0.01),
            adaptive_net_span=_int("GC_ADAPTIVE_NET_SPAN", 4),
            adaptive_window=_int("GC_ADAPTIVE_WINDOW", 8),
            level_low=_float("GC_LEVEL_LOW", 0.35),
            level_high=_float("GC_LEVEL_HIGH", 0.65),
        ).validate()

        boundary = BoundaryConfig(
            proprio=_str("GC_PROPRIO", "auto"),
            prompt_from_subtask=_bool("GC_PROMPT_FROM_SUBTASK", True),
            reset_feed_on_instruction_change=_bool("GC_RESET_ON_INSTRUCTION", True),
            validate_upstream=_bool("GC_VALIDATE_UPSTREAM", True),
            max_retries=_int("GC_MAX_RETRIES", 2),
        )
        if boundary.proprio not in ("all", "measured", "auto"):
            raise ValueError("GC_PROPRIO must be all, measured or auto")

        calibration = Calibration(
            swap_wrists=_bool("GC_SWAP_WRISTS", True),
            invert_r_wrist_roll=_bool("GC_INVERT_R_WRIST_ROLL", False),
            invert_r_thumb_yaw=_bool("GC_INVERT_R_THUMB_YAW", False),
            thumb_hw=_bool("GC_THUMB_HW", False),
            neck_bias=_float("GC_NECK_BIAS", 0.0),
        )

        raw_args = _str("GC_UPSTREAM_ARGS", "")
        return cls(
            policy_port=_int("POLICY_PORT", 8999),
            raw_port=_int("POLICY_RAW_PORT", 8998),
            host=_str("GC_HOST", "0.0.0.0"),
            raw_host=_str("GC_RAW_HOST", "127.0.0.1"),
            checkpoint=_str("CHECKPOINT_PATH", "/ckpt"),
            data_config=data_config,
            device=_str("GC_DEVICE", "cuda:0"),
            stack=_str("GC_STACK", "official"),
            adapter=_str("GC_ADAPTER", "none"),
            neck_compensation=_bool("GC_NECK_COMPENSATION", False),
            upstream_args=tuple(shlex.split(raw_args)) if raw_args else (),
            inference_steps=(_int("GC_INFERENCE_STEPS", 0) or None),
            start_timeout=_float("GC_START_TIMEOUT", 600.0),
            telemetry_path=os.environ.get("GC_TELEMETRY", "/tmp/greenchallenge/tracking.jsonl"),
            log_level=_str("GC_LOG_LEVEL", "INFO"),
            feed=feed,
            boundary=boundary,
            calibration=calibration,
            zero_velocity=_bool("GC_ZERO_VELOCITY", False),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "ports": {"policy": self.policy_port, "raw": self.raw_port},
            "checkpoint": self.checkpoint,
            "data_config": self.data_config,
            "device": self.device,
            "stack": self.stack,
            "adapter": self.adapter,
            "speed": self.feed.speed,
            "row_dt": self.feed.row_dt,
            "period": self.feed.period,
            "blend_life": self.feed.blend_life,
            "adaptive_speed": self.feed.adaptive_speed,
            "expire_frozen": self.feed.expire_frozen,
            "proprio": self.boundary.proprio,
            "prompt_from_subtask": self.boundary.prompt_from_subtask,
            "telemetry": self.telemetry_path,
        }
