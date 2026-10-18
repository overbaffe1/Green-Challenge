"""Turn an action chunk into the per-tick commands the simulator executes.

The policy answers a request with a *plan*: 50 rows saying where the joints
should be at future moments.  It does not say which row is due on the next
4 ms tick, how to reconcile two plans that overlap, or how fast a joint may
move between ticks.  Those decisions are the action feed, and on this task
they are worth more than a fine-tuning round:

* the reference deployment scores about 70 % with its feed and 7 % without
  it (the feed is part of the baseline, not an optional extra);
* replaying the same plan 1.3x faster raised the public leaderboard score
  from 15.445 to 18.096 without touching a weight.

This module reimplements the feed with the same timing semantics and adds
three things the reference version leaves out: a uniform *speed* factor,
expiry of plans that have reached their last row, and an optional adaptive
speed that keeps grasp/release rows at the trained pace.

All arithmetic is in model seconds; ``control_dt`` is the simulator's tick.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Sequence

import numpy as np

from . import layout

LOGGER = logging.getLogger(__name__)

#: Channel indices (in the numeric wire row) for the clamp selections.
_FINGERS = tuple(range(25, 37))
_WRISTS = tuple(range(37, 41))
_TORSO = tuple(range(12, 25))
_ARM_RIGHT = tuple(12 + k for k in range(6, 11))
_ARM_LEFT = tuple(12 + k for k in range(1, 6))
_TORSO_YAW = (12,)
_NECK = (12 + 11, 12 + 12)
_LEGS = tuple(range(0, 12))

_GROUPS: dict[int, str] = {}
for _index in _FINGERS:
    _GROUPS[_index] = "fingers"
for _index in _WRISTS + _ARM_RIGHT + _ARM_LEFT + _TORSO_YAW:
    _GROUPS[_index] = "arm"
for _index in _NECK:
    _GROUPS[_index] = "neck"
for _index in _LEGS:
    _GROUPS[_index] = "legs"

_CHANNEL_SELECTIONS: dict[str, tuple[int, ...]] = {
    "rhand": _FINGERS[6:] + _WRISTS[2:],
    "hands": _FINGERS + _WRISTS,
    "arm": _ARM_RIGHT,
    "both": _FINGERS + _WRISTS + _ARM_RIGHT,
    "all": _LEGS + _TORSO_YAW + _ARM_LEFT + _ARM_RIGHT + _NECK + _WRISTS + _FINGERS,
}


@dataclasses.dataclass(frozen=True)
class FeedConfig:
    """Every knob of the action feed."""

    # ── scheduling ───────────────────────────────────────────────────────────
    row_dt: float = 1.0 / 30.0
    """Plan row step at 1x, model seconds."""

    speed: float = 1.30
    """Uniform playback speed.  1.30 is the measured public optimum for the
    baseline checkpoint (15.445 -> 18.096); 1.0 reproduces the reference feed.
    The learned chunk is replayed faster, not recomputed: rows keep their
    shape and only their timestamps are compressed."""

    lead: float = 0.030
    """How far ahead of the observation row 0 is scheduled."""

    period: float = 0.080
    """Observation-to-observation period; sets the number of rows per reply."""

    latency: float | None = None
    """When a fresh plan takes effect, measured from its own observation.
    ``None`` derives ``lead + row_dt / speed``, i.e. the moment row 0 is due."""

    # ── blending ─────────────────────────────────────────────────────────────
    blend: bool = True
    blend_tau: float = 1.0
    blend_life: float = 1.20
    blend_skip: tuple[str, ...] = ()
    """Numeric fields taken from the newest plan instead of the average,
    e.g. ``("finger_joint_pos",)``.  Averaging a short grasp impulse can
    average the grasp away."""

    # ── clamping ─────────────────────────────────────────────────────────────
    clamp: bool = True
    clamp_channels: str = "both"
    max_speed_fingers: float = 10.1
    max_speed_arm: float = 3.8
    max_speed_neck: float = 4.5
    max_speed_legs: float = 3.0
    max_acceleration_fingers: float = 30.1
    max_acceleration_arm: float = 3.8
    max_acceleration_neck: float = 10.5
    max_acceleration_legs: float = 3.0

    # ── improvements over the reference feed ─────────────────────────────────
    expire_frozen: bool = True
    """Drop plans from the blend once the current time is past their last
    row.  Without this a plan older than its horizon freezes on its final
    pose and pulls the setpoint backwards; it bites at speed > 1.42."""

    adaptive_speed: float = 1.0
    """Extra speed away from grasp/release events (SAIL-style).  ``1.0``
    disables it.  Events are detected on the plan itself."""

    adaptive_detector: str = "level"
    """``level`` (hysteresis around 0.5), ``net`` (centred finger travel) or
    ``travel`` (row-to-row travel; the reference knee-jerk detector, which on
    sampled flow plans fires on ~60 % of rows)."""

    adaptive_threshold: float = 0.15
    adaptive_travel_threshold: float = 0.01
    adaptive_net_span: int = 4
    adaptive_window: int = 8
    level_low: float = 0.35
    level_high: float = 0.65

    def speed_limit(self, group: str) -> float:
        return float(getattr(self, "max_speed_" + group))

    def acceleration_limit(self, group: str) -> float:
        return float(getattr(self, "max_acceleration_" + group))

    def row_step(self) -> float:
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        return float(self.row_dt) / float(self.speed)

    def resolved_latency(self) -> float:
        if self.latency is None:
            return float(self.lead) + self.row_step()
        value = float(self.latency)
        if value >= self.period:
            LOGGER.warning(
                "latency %.4f s >= period %.4f s: clamping to half a period",
                value,
                self.period,
            )
            value = self.period * 0.5
        return value

    def validate(self) -> "FeedConfig":
        if self.row_dt <= 0 or self.period <= 0 or self.lead < 0:
            raise ValueError("row_dt and period must be positive, lead non-negative")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if self.blend_life <= 0 or self.blend_tau <= 0:
            raise ValueError("blend_life and blend_tau must be positive")
        if self.clamp_channels not in _CHANNEL_SELECTIONS:
            raise ValueError(
                f"clamp_channels must be one of {sorted(_CHANNEL_SELECTIONS)}, "
                f"got {self.clamp_channels!r}"
            )
        if self.adaptive_detector not in ("level", "net", "travel"):
            raise ValueError("adaptive_detector must be level, net or travel")
        if not 0.0 < self.level_low < self.level_high < 1.0:
            raise ValueError("level_low/high must satisfy 0 < low < high < 1")
        if self.adaptive_speed < 1.0:
            raise ValueError("adaptive_speed below 1 would slow the plan down")
        return self


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Plan:
    rows: np.ndarray
    t_obs: float
    times: np.ndarray
    index: int
    adaptive_rows: int = 0

    @property
    def horizon(self) -> float:
        return float(self.times[-1])


def _event_rows(rows: np.ndarray, config: FeedConfig) -> np.ndarray:
    """Boolean mask over rows that belong to a grasp or release."""
    fingers = rows[:, _FINGERS]
    n = len(fingers)
    if n < 2:
        return np.zeros(n, dtype=bool)

    if config.adaptive_detector == "travel":
        travel = np.abs(np.diff(fingers, axis=0)).max(axis=1)
        events = np.zeros(n, dtype=bool)
        events[1:] = travel > config.adaptive_travel_threshold
        return events

    if config.adaptive_detector == "net":
        span = max(1, int(config.adaptive_net_span))
        padded = np.concatenate(
            [np.repeat(fingers[:1], span, axis=0), fingers, np.repeat(fingers[-1:], span, axis=0)]
        )
        travel = np.abs(padded[2 * span :] - padded[: -2 * span]).max(axis=1)
        return travel > config.adaptive_threshold

    # Hysteresis level crossing: a finger counts as closed above level_high
    # and open below level_low, so jitter around a single threshold does not
    # produce an event on every row.
    state = np.zeros(fingers.shape[1], dtype=np.int8)
    events = np.zeros(n, dtype=bool)
    for row in range(n):
        value = fingers[row]
        newly_closed = (state == 0) & (value > config.level_high)
        newly_open = (state == 0) & (value < config.level_low)
        events[row] = bool(newly_closed.any() or newly_open.any())
        state = np.where(newly_closed, 1, state)
        state = np.where(newly_open, -1, state)
    return events


def _row_steps(rows: np.ndarray, config: FeedConfig) -> tuple[np.ndarray, int]:
    """Per-row time steps and the number of rows kept at the trained pace."""
    n = len(rows)
    fast = config.row_step()
    steps = np.full(n, fast, dtype=np.float64)
    if config.adaptive_speed <= 1.0:
        return steps, 0

    events = _event_rows(rows, config)
    slow = np.zeros(n, dtype=bool)
    window = max(0, int(config.adaptive_window))
    for row in np.flatnonzero(events):
        slow[max(0, row - window) : min(n, row + window + 1)] = True
    steps[slow] = config.row_dt
    return steps, int(slow.sum())


def make_plan(rows: np.ndarray, t_obs: float, index: int, config: FeedConfig) -> Plan:
    steps, adaptive_rows = _row_steps(rows, config)
    times = config.lead + np.cumsum(steps)
    return Plan(rows=rows, t_obs=float(t_obs), times=times, index=index, adaptive_rows=adaptive_rows)


def evaluate(plan: Plan, t: float, config: FeedConfig) -> np.ndarray:
    """The plan's forecast for model time ``t`` (linear interpolation)."""
    elapsed = t - plan.t_obs
    if elapsed <= plan.times[0] + 1e-12:
        return plan.rows[0].copy()
    if elapsed >= plan.times[-1] - 1e-12:
        return plan.rows[-1].copy()
    j = int(np.searchsorted(plan.times, elapsed, side="right")) - 1
    j = min(max(j, 0), len(plan.rows) - 2)
    span = plan.times[j + 1] - plan.times[j]
    frac = 0.0 if span <= 0 else (elapsed - plan.times[j]) / span
    return plan.rows[j] + frac * (plan.rows[j + 1] - plan.rows[j])


# --------------------------------------------------------------------------- #
# Limiter
# --------------------------------------------------------------------------- #


class Clamp:
    """Velocity/acceleration limiter with discrete braking distance.

    State is kept on the command issued, never on a measured pose: the
    limiter reasons about what it asked for, not about how well the joint
    tracked it.
    """

    def __init__(self, config: FeedConfig, dt: float) -> None:
        self.config = config
        self.dt = float(dt)
        self.channels = _CHANNEL_SELECTIONS[config.clamp_channels]
        self.position = np.zeros(layout.WIRE_DIM, dtype=np.float64)
        self.velocity = np.zeros(layout.WIRE_DIM, dtype=np.float64)
        self.seated = np.zeros(layout.WIRE_DIM, dtype=bool)

    def seed(self, row: np.ndarray) -> int:
        row = np.asarray(row, dtype=np.float64)
        for index in self.channels:
            self.position[index] = float(row[index])
            self.velocity[index] = 0.0
            self.seated[index] = True
        return len(self.channels)

    def apply(self, row: np.ndarray) -> np.ndarray:
        out = np.asarray(row, dtype=np.float64).copy()
        amax_dt = {group: self.config.acceleration_limit(group) * self.dt for group in
                   ("fingers", "arm", "neck", "legs")}
        for index in self.channels:
            group = _GROUPS[index]
            vmax = self.config.speed_limit(group)
            amax = self.config.acceleration_limit(group)
            goal = float(out[index])
            if not self.seated[index]:
                self.position[index] = goal
                self.velocity[index] = 0.0
                self.seated[index] = True
                continue
            position = self.position[index]
            velocity = self.velocity[index]
            gap = goal - position
            if amax > 0:
                adt = amax_dt[group]
                brake = 0.5 * adt * (
                    math.sqrt(1.0 + 8.0 * abs(gap) / max(adt * self.dt, 1e-300)) - 1.0
                )
                brake = min(brake, vmax)
            else:
                brake = vmax
            wanted = gap / self.dt
            wanted = max(-vmax, min(vmax, wanted))
            wanted = max(-brake, min(brake, wanted))
            wanted = max(velocity - amax * self.dt, min(velocity + amax * self.dt, wanted))
            moved = position + wanted * self.dt
            if (goal - moved) * (goal - position) < 0.0:
                moved, wanted = goal, 0.0
            self.position[index] = moved
            self.velocity[index] = wanted
            out[index] = moved
        return out


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FeedStats:
    chunks: int = 0
    live_peak: int = 0
    frozen_dropped: int = 0
    adaptive_rows: int = 0
    last_live: int = 0
    last_row_step: float = 0.0


class NumericActionFeed:
    """Stateful translator from action chunks to per-tick commands."""

    def __init__(self, config: FeedConfig, dt: float) -> None:
        if dt <= 0:
            raise ValueError("control dt must be positive")
        self.config = config.validate()
        self.dt = float(dt)
        self.latency = config.resolved_latency()
        self.ticks_per_period = max(1, int(round(config.period / self.dt)))
        self._skip_slices = tuple(
            layout.WIRE_SLICES[name] for name in config.blend_skip
        )
        self.stats = FeedStats(last_row_step=config.row_step())
        self.reset()

    # ── episode boundary ────────────────────────────────────────────────────
    def reset(self, seed_row: np.ndarray | None = None) -> None:
        self._active: Plan | None = None
        self._pending: Plan | None = None
        self._pending_effective_at = 0.0
        self._n_chunks = 0
        self._live: list[Plan] = []
        self._last_t_obs: float | None = None
        self._clamp = Clamp(self.config, self.dt) if self.config.clamp else None
        if self._clamp is not None and seed_row is not None:
            seated = self._clamp.seed(seed_row)
            LOGGER.info("action feed: limiter seated on %d channels", seated)

    @property
    def last_t_obs(self) -> float | None:
        return self._last_t_obs

    # ── one request ─────────────────────────────────────────────────────────
    def update(self, t_obs: float, rows: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Register a fresh chunk and return the commands for the next ticks."""
        numeric = np.asarray(rows, dtype=np.float64)
        if numeric.ndim != 2 or numeric.shape[1] != layout.WIRE_DIM:
            raise ValueError(
                f"action rows must have shape (N, {layout.WIRE_DIM}), got {numeric.shape}"
            )
        if len(numeric) < 2:
            raise ValueError("the action feed was handed a chunk with fewer than two rows")
        if not np.isfinite(numeric).all():
            raise ValueError("action rows contain NaN or Inf")

        t_obs = float(t_obs)
        if self._last_t_obs is not None and t_obs < self._last_t_obs - 1e-9:
            LOGGER.warning(
                "episode clock moved backwards %.3f -> %.3f; resetting feed",
                self._last_t_obs,
                t_obs,
            )
            self.reset()
        self._last_t_obs = t_obs

        if self._pending is not None:
            self._promote_pending()

        self._n_chunks += 1
        plan = make_plan(numeric, t_obs, self._n_chunks, self.config)
        self._pending = plan
        self.stats.chunks = self._n_chunks
        self.stats.adaptive_rows = plan.adaptive_rows
        self._pending_effective_at = t_obs + (self.latency if self._active is not None else 0.0)

        first_tick = round(t_obs / self.dt)
        commands = np.empty((self.ticks_per_period, layout.WIRE_DIM), dtype=np.float64)
        for tick in range(self.ticks_per_period):
            t = (first_tick + tick) * self.dt
            if self._pending is not None and t >= self._pending_effective_at - 1e-9:
                self._promote_pending()
            if self._active is None:
                raise RuntimeError(f"the action feed has no active plan at {t:.3f} s")
            command = evaluate(self._active, t, self.config)
            if self.config.blend:
                command = self._blend(t, command)
            if self._clamp is not None:
                command = self._clamp.apply(command)
            commands[tick] = command
        return commands

    # ── internals ───────────────────────────────────────────────────────────
    def _promote_pending(self) -> None:
        self._active = self._pending
        self._pending = None
        if self._active is not None and self.config.blend:
            self._live.append(self._active)

    def _blend(self, t: float, active_command: np.ndarray) -> np.ndarray:
        life = self.config.blend_life
        tau = self.config.blend_tau
        keep: list[Plan] = []
        for plan in self._live:
            age = t - plan.t_obs
            if plan is not self._active and age > life + 1e-9:
                continue
            if (
                self.config.expire_frozen
                and plan is not self._active
                and age > plan.horizon + 1e-9
            ):
                self.stats.frozen_dropped += 1
                continue
            keep.append(plan)
        self._live = keep
        self.stats.last_live = len(keep)
        if len(keep) > self.stats.live_peak:
            self.stats.live_peak = len(keep)

        if len(keep) < 2:
            return active_command

        accumulated = active_command.copy()
        weight_sum = math.exp(-(t - self._active.t_obs) / max(tau, 1e-9))
        for plan in keep:
            if plan is self._active:
                continue
            age = t - plan.t_obs
            weight = math.exp(-age / max(tau, 1e-9))
            forecast = evaluate(plan, t, self.config)
            weight_sum += weight
            alpha = weight / weight_sum if weight_sum > 0 else 0.0
            accumulated += alpha * (forecast - accumulated)
        for field_slice in self._skip_slices:
            accumulated[field_slice] = active_command[field_slice]
        return accumulated
