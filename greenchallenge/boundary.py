"""The public policy endpoint: simulator wire in, per-tick commands out.

Topology (the same two-stage shape the organisers ship, minus the parts that
would be non-compliant or unmeasured)::

    simulator  --ws-->  Boundary (POLICY_PORT)  --ws-->  model server (POLICY_RAW_PORT)
                            |                                  |
                            |  decode PROPRIO_CMD=all echo     |  raw 50-row chunk
                            |  schedule / blend / limit        |
                            +----------------------------------+

The boundary never invents a pose.  It forwards the observation to the model,
takes the model's chunk, and only decides *when* each predicted row is due,
which is the execution layer the contest's own baseline ships as well.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import contract, layout
from .feed import FeedConfig, NumericActionFeed
from .metrics import JsonlLogger

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BoundaryConfig:
    proprio: str = "auto"
    """``all`` (packaged deployment), ``measured`` or ``auto``.

    ``auto`` treats an exact torso match with the last published command as
    the simulator's ``PROPRIO_CMD=all`` echo.  It removes the need to know how
    the evaluation harness starts the simulator, which is not transmitted."""

    prompt_from_subtask: bool = True
    """Use the simulator's current atomic instruction.  The checkpoint was
    trained with one instruction per subtask, not the full task sentence."""

    reset_feed_on_instruction_change: bool = True
    """Drop old plans when the instruction changes.  Continuing to blend
    plans produced for the previous subtask is the one case where temporal
    ensembling is clearly wrong."""

    validate_upstream: bool = True
    """Reject chunks whose shape or finiteness is wrong instead of sending a
    plausible but corrupted command to 25 joints."""

    max_retries: int = 2


class Boundary:
    """Stateful adapter implementing the simulator's policy interface."""

    def __init__(
        self,
        upstream: Any,
        *,
        config: BoundaryConfig | None = None,
        feed_config: FeedConfig | None = None,
        calibration: contract.Calibration | None = None,
        control_dt: float = 0.004,
        data_config: str = "scripted_014814",
        telemetry: JsonlLogger | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.upstream = upstream
        self.config = config or BoundaryConfig()
        self.calibration = calibration or contract.Calibration()
        self.feed = NumericActionFeed(feed_config or FeedConfig(), control_dt)
        self.control_dt = float(control_dt)
        self.data_config = data_config
        self.telemetry = telemetry or JsonlLogger(None)
        self.extra_metadata = dict(extra_metadata or {})

        self.last_wire: dict | None = None
        self.last_prompt: str | None = None
        self.last_t: float | None = None
        self.tick = 0
        self.requests = 0
        self.resets = 0
        self.instruction_switches = 0
        self._last_command: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    @property
    def metadata(self) -> dict:
        upstream = {}
        try:
            upstream = dict(getattr(self.upstream, "metadata", {}) or {})
        except Exception:
            pass
        metadata = {
            "policy": "greenchallenge_boundary",
            "model_family": upstream.get("model_family", "GreenVLAv1.1"),
            "robot": upstream.get("robot", self.data_config),
            "state_dim": layout.MODEL_DIM,
            "action_dim": layout.MODEL_DIM,
            "raw_state_dim": layout.RAW_STATE_DIM,
            "action_horizon": int(upstream.get("action_horizon", 50) or 50),
            "action_sample_step": int(upstream.get("action_sample_step", 1) or 1),
            "feed_by_server": True,
            "control_dt": self.control_dt,
            "inference_steps": upstream.get("inference_steps"),
            "gmm_guard_mode": "off",
            "wire_boundary": {
                "proprio": self.config.proprio,
                "finger_defaults": self.calibration.finger_defaults.tolist(),
                "swap_wrists": self.calibration.swap_wrists,
                "neck_compensation": bool(self.calibration.neck_bias),
                "image_or_prompt_changes": False,
            },
            "greenchallenge": {
                "feed": dataclasses.asdict(self.feed.config),
                "ticks_per_period": self.feed.ticks_per_period,
                "prompt_from_subtask": self.config.prompt_from_subtask,
            },
        }
        metadata.update(self.extra_metadata)
        return metadata

    # ------------------------------------------------------------------ #
    # Episode / instruction boundaries
    # ------------------------------------------------------------------ #
    def _seed_row(self, observation: Mapping[str, Any]) -> np.ndarray | None:
        state = observation.get("state")
        if not isinstance(state, Mapping):
            return None
        try:
            return contract.state_to_wire_numeric(state, self.calibration)
        except (ValueError, TypeError):
            return None

    def _reset_episode(self, observation: Mapping[str, Any]) -> None:
        self.feed.reset(self._seed_row(observation))
        self.last_wire = None
        self.last_prompt = None
        self.last_t = None
        self.tick = 0
        self._last_command = None
        self.resets += 1
        LOGGER.info("episode reset: feed and echo history cleared")

    # ------------------------------------------------------------------ #
    # One request
    # ------------------------------------------------------------------ #
    def step(self, observation: Mapping[str, Any]) -> dict:
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation must be a mapping, got {type(observation).__name__}")

        reset_flag = bool(observation.get("reset", False))
        t = self._time(observation)
        reset = reset_flag or (self.last_t is not None and t < self.last_t - 1e-9)
        if reset:
            if not reset_flag:
                LOGGER.warning("clock moved backwards (%.3f -> %.3f); treating as reset", self.last_t, t)
            # At an episode start the simulator sends the measured pose, not an
            # echo of a command from the previous episode, so it must not be
            # decoded.  The seed row is what seats the limiter.
            decoded = observation
            self._reset_episode(decoded)
        else:
            decoded = contract.decode_proprioception(
                observation,
                proprio=self.config.proprio,
                calibration=self.calibration,
                last_wire=self.last_wire,
            )

        prompt = contract.choose_prompt(
            observation,
            self.last_prompt,
            from_subtask=self.config.prompt_from_subtask,
        )
        instruction_changed = self.last_prompt is not None and prompt != self.last_prompt
        if instruction_changed and self.config.reset_feed_on_instruction_change:
            LOGGER.info("instruction change: %r -> %r (old plans dropped)", self.last_prompt, prompt)
            # The state here is the echo of the last command; ``decoded`` is in
            # absolute coordinates, which is what the limiter expects.
            self.feed.reset(self._seed_row(decoded))
            self.instruction_switches += 1

        started = time.monotonic()
        response = self._infer(decoded)
        infer_ms = (time.monotonic() - started) * 1000.0
        rows = response["actions_list"]
        numeric = self._rows_to_numeric(rows)
        commands = self.feed.update(t, numeric)

        wire_rows = [
            contract.numeric_to_wire_row(row, self.calibration) for row in commands
        ]
        self.last_wire = wire_rows[-1]
        self._last_command = commands[-1].copy()
        self.last_prompt = prompt
        self.last_t = t
        self.tick = int(round(t / self.control_dt)) + len(wire_rows)
        self.requests += 1

        self._record(t, prompt, rows, numeric, commands, infer_ms, response)
        return {"actions_list": wire_rows}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _time(self, observation: Mapping[str, Any]) -> float:
        value = observation.get("t")
        if value is None:
            # A lenient fallback: the packaged simulator always sends t, but a
            # missing clock should degrade to our own tick counter rather than
            # crash the whole episode.
            t = self.tick * self.control_dt
        else:
            t = float(value)
        if not np.isfinite(t):
            raise ValueError("observation.t must be finite")
        return t

    def _infer(self, observation: Mapping[str, Any]) -> dict:
        last_error: Exception | None = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                response = self.upstream.step(observation)
                if "actions_list" not in response:
                    raise RuntimeError("upstream reply has no actions_list")
                return response
            except Exception as exc:
                last_error = exc
                LOGGER.warning("upstream step failed (%d/%d): %s", attempt + 1, self.config.max_retries, exc)
                reconnect = getattr(self.upstream, "reconnect", None)
                if reconnect is not None:
                    try:
                        reconnect()
                    except Exception as reconnect_exc:
                        LOGGER.warning("upstream reconnect failed: %s", reconnect_exc)
        raise RuntimeError(f"upstream inference failed: {last_error}")

    def _rows_to_numeric(self, rows: Any) -> np.ndarray:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("upstream returned an empty chunk")
        numeric = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"action row {index} is not a mapping")
            try:
                numeric.append(contract.wire_row_to_numeric(row))
            except ValueError as exc:
                raise ValueError(f"action row {index}: {exc}") from exc
        stacked = np.stack(numeric, axis=0)
        if self.config.validate_upstream:
            if stacked.shape[0] < 2:
                raise ValueError("chunk must contain at least two rows")
            if not np.isfinite(stacked).all():
                raise ValueError("chunk contains NaN or Inf")
        return stacked

    def _record(
        self,
        t: float,
        prompt: str,
        raw_rows: Any,
        numeric: np.ndarray,
        commands: np.ndarray,
        infer_ms: float,
        response: Mapping[str, Any],
    ) -> None:
        jump = 0.0
        if self._last_command is not None:
            # Command discontinuity at a plan seam, in radians, over the
            # channels the clamp does not already limit (base/velocity too).
            jump = float(np.max(np.abs(commands[0] - self._last_command)))
        self.telemetry.write(
            {
                "t": round(t, 4),
                "prompt": prompt[:160],
                "rows_in": len(raw_rows),
                "rows_out": len(commands),
                "live": self.feed.stats.last_live,
                "live_peak": self.feed.stats.live_peak,
                "frozen_dropped": self.feed.stats.frozen_dropped,
                "adaptive_rows": self.feed.stats.adaptive_rows,
                "row_step": round(self.feed.stats.last_row_step, 6),
                "speed": self.feed.config.speed,
                "infer_ms": round(infer_ms, 2),
                "upstream_ms": round(float(response.get("server_timing", {}).get("infer_ms", 0.0)), 2),
                "jump": round(jump, 5),
                "requests": self.requests,
                "resets": self.resets,
                "instruction_switches": self.instruction_switches,
            }
        )
