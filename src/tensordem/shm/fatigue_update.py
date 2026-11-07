"""Streaming rainflow counting and Miner damage accumulation for the twin.

The twin must update the accumulated fatigue damage at every acquisition window,
i.e. without seeing the future.  The ASTM E1049-85 range-pair algorithm is
naturally incremental: cycles are *closed* as soon as a third turning point
satisfies ``Y >= X``, and the remaining stack is only a provisional residue.  So
the streaming counter keeps

* a gated turning-point detector with persistent direction state,
* the range-pair stack of the batch algorithm,
* the damage already delivered by the closed cycles,

and reports two damages:

* ``damage_closed`` — sum over closed cycles only.  Monotone in time and final,
  which is what an operator display must show (a residue half cycle that later
  gets closed must not make the damage *fall*).
* ``damage`` — closed cycles plus the current residue half cycles (the batch
  convention of :mod:`tensordem.whipping.rainflow`), used for the end-of-record
  comparison with a direct batch rainflow count.

Two safeguards are needed because the counted signal is an *inverted* stress:

* **gate (hysteresis).**  A strain inversion leaves a residual noise floor, and
  rainflow counting is famously unbounded on a noisy record: every noise
  wiggle becomes a half cycle.  A reversal is therefore accepted only after the
  signal retraces by more than ``gate`` from the running extremum, with ``gate``
  tied to the strain resolution of the acquisition chain (``1 MPa`` is 4.85 micro
  strain for hull steel).  With ``gate = 0`` the counter must reproduce the batch
  algorithm exactly, which is verified in the tests.
* **incremental damage.**  The damage of a closed cycle is added the moment it
  closes, so the per-window cost does not grow with the length of the record
  (re-deriving Miner's sum from all cycles each window would be quadratic and
  would miss the 50 ms latency budget).

The counting rule is verified against the batch implementation
(:func:`tensordem.whipping.rainflow.rainflow`) in the tests, so the streaming
version is not a second source of truth but a stateful evaluation of the same
algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..whipping.fatigue import DNV_CURVES, SNCurve, miner_damage

__all__ = ["StreamingRainflow", "FatigueAccumulator", "CycleRecord"]


@dataclass(frozen=True)
class CycleRecord:
    """One counted cycle: stress range [Pa], count, mean [Pa]."""

    range: float
    count: float
    mean: float


class StreamingRainflow:
    """Incremental, gated ASTM E1049-85 range-pair counter."""

    def __init__(self, gate: float = 0.0):
        if gate < 0.0:
            raise ValueError("gate must be nonnegative")
        self.gate = float(gate)
        self._stack: list[float] = []
        self._direction: int = 0
        self._extremum: float | None = None
        self._last: float | None = None
        self.closed: list[CycleRecord] = []
        self.n_samples = 0
        self.n_turning = 0

    # ---------------------------------------------------------------- feeding
    def _reduce(self, stack, value):
        """Append one turning point and apply the range-pair reduction rule."""
        stack = list(stack) + [float(value)]
        closed = []
        while len(stack) >= 3:
            x_range = abs(stack[-2] - stack[-3])
            y_range = abs(stack[-1] - stack[-2])
            if y_range >= x_range:
                closed.append(CycleRecord(range=x_range, count=1.0,
                                          mean=0.5 * (stack[-3] + stack[-2])))
                del stack[-3:-1]
            else:
                break
        return stack, closed

    def _push(self, value: float) -> None:
        self._stack, new = self._reduce(self._stack, value)
        self.closed.extend(new)
        self.n_turning += 1

    def update(self, signal) -> int:
        """Feed a chunk of samples; return the number of newly closed cycles."""
        x = np.asarray(signal, dtype=float).ravel()
        if x.size == 0:
            return 0
        if not np.isfinite(x).all():
            raise ValueError("signal must be finite")
        n_before = len(self.closed)
        for value in x:
            value = float(value)
            self.n_samples += 1
            if self._last is None:
                self._last = value
                self._extremum = value
                self._push(value)
                continue
            if value == self._last:
                continue
            if self._direction == 0:
                self._direction = 1 if value > self._last else -1
                self._extremum = value
            elif (value - self._last) * self._direction > 0:
                self._extremum = value                    # extend the extremum
            elif abs(value - self._extremum) > self.gate:
                self._push(self._extremum)                # confirmed reversal
                self._direction = -self._direction
                self._extremum = value
            self._last = value
        return len(self.closed) - n_before

    # --------------------------------------------------------------- residues
    def residues(self) -> tuple[list[CycleRecord], list[CycleRecord]]:
        """Return ``(closed_by_termination, half_cycles)`` of the current residue."""
        if self._extremum is None:
            return [], []
        stack, closed = self._reduce(self._stack, self._extremum)
        halves = [CycleRecord(range=abs(stack[i + 1] - stack[i]), count=0.5,
                              mean=0.5 * (stack[i + 1] + stack[i]))
                  for i in range(len(stack) - 1)]
        self._residue_closed = closed
        self._residue_half = halves
        return closed, halves

    def all_cycles(self) -> list[CycleRecord]:
        """Closed cycles + residue (the batch-equivalent cycle set)."""
        closed_at_end, halves = self.residues()
        return list(self.closed) + closed_at_end + halves

    def batch_equivalent(self, signal) -> list[CycleRecord]:
        """Count a complete record with :mod:`tensordem.whipping.rainflow`."""
        from ..whipping.rainflow import rainflow

        return [CycleRecord(range=c.range, count=c.count, mean=c.mean)
                for c in rainflow(signal)]


@dataclass
class FatigueAccumulator:
    """Incremental Miner damage accumulator fed by stress windows of the twin.

    ``curve`` is a DNV-RP-C203 S-N curve (``D_air`` by default, the free-air
    branch that applies to deck plating inside the hull); ``gate`` is the
    hysteresis gate of the turn-point detector in Pa; ``stress_scale`` converts
    the fed stress unit to Pa (the twin feeds Pa, so it stays at 1).
    """

    curve: SNCurve = DNV_CURVES["D_air"]
    stress_scale: float = 1.0
    gate: float = 1.0e6
    counter: StreamingRainflow = field(default=None)
    damage_closed: float = 0.0
    history: list = field(default_factory=list)   # (time_s, damage_closed)
    n_accounted: int = 0

    def __post_init__(self) -> None:
        if self.counter is None:
            self.counter = StreamingRainflow(gate=self.gate)

    # ---------------------------------------------------------------- feeding
    def _life(self, ranges: np.ndarray) -> np.ndarray:
        if ranges.size == 0:
            return np.zeros(0)
        return self.curve.cycles_to_failure(np.maximum(ranges, 1e-12))

    def update(self, stress, *, time_s: float | None = None) -> dict:
        """Feed a stress chunk and add the damage of the newly closed cycles."""
        x = np.asarray(stress, dtype=float) * self.stress_scale
        self.counter.update(x)
        new = self.counter.closed[self.n_accounted:]
        if new:
            ranges = np.array([c.range for c in new])
            counts = np.array([c.count for c in new])
            self.damage_closed += float(np.sum(counts / self._life(ranges)))
            self.n_accounted = len(self.counter.closed)
        if time_s is not None:
            self.history.append((float(time_s), self.damage_closed))
        return {"damage_closed": self.damage_closed,
                "cycles_closed": float(len(self.counter.closed)),
                "n_samples": self.counter.n_samples}

    # ------------------------------------------------------------- reporting
    def total_damage(self, *, include_residue: bool = True) -> float:
        """Damage including (default) or excluding the provisional residue."""
        if not include_residue:
            return self.damage_closed
        closed_at_end, halves = self.counter.residues()
        cycles = closed_at_end + halves
        if not cycles:
            return self.damage_closed
        ranges = np.array([c.range for c in cycles])
        counts = np.array([c.count for c in cycles])
        keep = ranges > 0.0
        if not np.any(keep):
            return self.damage_closed
        return self.damage_closed + float(np.sum(counts[keep] / self._life(ranges[keep])))

    def summary(self) -> dict:
        cycles = self.counter.all_cycles()
        if cycles:
            ranges = np.array([c.range for c in cycles])
            counts = np.array([c.count for c in cycles])
            batch = miner_damage(self.curve, ranges, counts)
        else:
            batch = {"damage": 0.0, "cycles": 0.0, "max_range_MPa": 0.0,
                     "equivalent_range_MPa": 0.0, "min_life_cycles": 0.0}
        return {
            "damage": self.total_damage(include_residue=True),
            "damage_closed": self.damage_closed,
            "damage_batch_equivalent": float(batch["damage"]),
            "cycles": float(batch["cycles"]),
            "cycles_closed": float(len(self.counter.closed)),
            "max_range_MPa": float(batch["max_range_MPa"]),
            "equivalent_range_MPa": float(batch["equivalent_range_MPa"]),
            "min_life_cycles": float(batch["min_life_cycles"]),
            "curve": self.curve.name,
            "gate_MPa": self.gate / 1.0e6,
            "n_samples": self.counter.n_samples,
            "n_turning_points": self.counter.n_turning,
        }