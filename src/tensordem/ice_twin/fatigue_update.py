"""Real-time (incremental) fatigue damage for the ice-zone digital twin.

The online path processes the stress history in bounded windows.  The ASTM
E1049-85 range-pair algorithm is *online*: the cycles it emits after seeing
the k-th turning point depend only on the prefix of turning points up to k.
We therefore stream the turning points through a persistent stack, using a
short sample *tail* carried over between windows as context so that turning
points near a window edge are detected exactly as in the single-pass
reference.  Every emitted cycle is converted to Miner damage immediately,

    dD = n_i / N_i,   N_i from the DNV-RP-C203 S-N curve,

and accumulated.  The offline reference is the single-pass
:func:`tensordem.whipping.rainflow.rainflow` over the whole record; the two
must agree within the campaign tolerance (10 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..whipping.fatigue import SNCurve, miner_damage
from ..whipping.rainflow import Cycle

__all__ = ["IncrementalFatigue", "turning_points_indexed"]


def turning_points_indexed(signal) -> tuple[np.ndarray, np.ndarray, bool]:
    """Like ``whipping.rainflow.turning_points`` but keeps the sample indices.

    Returns ``(idx, values, last_is_reversal)``: ``idx`` are positions in the
    input array of the alternating peaks/valleys (``idx[0] = 0`` always,
    mirroring the reference implementation).  ``last_is_reversal`` is True
    when the final entry was appended by a direction change (a genuine
    extremum) and False when it merely extends the final monotone run (the
    value of the last sample, not yet an extremum of the full record).
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("signal must contain at least two samples")
    idx = [0]
    vals = [float(x[0])]
    last_is_reversal = False
    for i in range(1, x.size):
        value = float(x[i])
        if value == vals[-1]:
            continue
        if len(vals) == 1:
            idx.append(i)
            vals.append(value)
            last_is_reversal = False
        elif (vals[-1] - vals[-2]) * (value - vals[-1]) < 0.0:
            idx.append(i)
            vals.append(value)
            last_is_reversal = True
        else:
            idx[-1] = i
            vals[-1] = value
            last_is_reversal = False
    return np.asarray(idx, dtype=int), np.asarray(vals, dtype=float), last_is_reversal


@dataclass
class IncrementalFatigue:
    """Windowed rainflow + Miner accumulation with bounded memory.

    ``tail_samples`` samples of the previous window are kept as context
    (constant memory, the property an embedded SHM unit has); the E1049
    stack itself is also bounded to ``max_stack`` entries.
    """

    curve: SNCurve
    tail_samples: int = 32
    max_stack: int = 32

    _stack: list = field(default_factory=list, init=False, repr=False)
    _tail: np.ndarray = field(default_factory=lambda: np.zeros(0),
                              init=False, repr=False)
    damage: float = field(default=0.0, init=False)
    cycles_counted: float = field(default=0.0, init=False)
    max_range: float = field(default=0.0, init=False)

    # ------------------------------------------------------------------ #
    def _emit(self, cycles: list[Cycle]) -> None:
        if not cycles:
            return
        ranges = np.array([c.range for c in cycles])
        counts = np.array([c.count for c in cycles])
        res = miner_damage(self.curve, ranges, counts)
        self.damage += res["damage"]
        self.cycles_counted += res["cycles"]
        self.max_range = max(self.max_range, res["max_range_MPa"])

    def _run_e1049(self, new_values) -> None:
        """Feed new turning points through the persistent E1049 stack."""
        for value in new_values:
            self._stack.append(float(value))
            while len(self._stack) >= 3:
                x_range = abs(self._stack[-2] - self._stack[-3])
                y_range = abs(self._stack[-1] - self._stack[-2])
                if y_range >= x_range:
                    self._emit([Cycle(range=x_range, count=1.0,
                                      mean=0.5 * (self._stack[-3] + self._stack[-2]))])
                    del self._stack[-3:-1]
                else:
                    break
            if len(self._stack) > self.max_stack:
                del self._stack[: len(self._stack) - self.max_stack]

    def update(self, sigma_chunk) -> float:
        """Consume one window of stress (Pa); returns the damage increment.

        Streaming invariant: after this call the E1049 stack equals the stack
        of the single-pass reference after it has consumed every turning
        point of the record up to the last sample of this chunk.  The tail
        of the previous window is context only; the end-of-chunk value is
        *not* a turning point of the full record (the monotone run continues
        into the next window) and is therefore not fed -- the true extremum
        is detected with full context in the next call.
        """
        chunk = np.asarray(sigma_chunk, dtype=float).ravel()
        if chunk.size == 0:
            return 0.0
        if self._tail.size:
            combined = np.concatenate([self._tail, chunk])
            boundary = self._tail.size
        else:
            combined, boundary = chunk, 0
        idx, vals, last_is_reversal = turning_points_indexed(combined)
        # the tail's last sample sits at idx = boundary - 1; it is the end of
        # a monotone run whose reversal is only visible with the next window's
        # context, so it is (re)fed here.  A duplicate value is a no-op for
        # the E1049 loop (it emits a zero-range cycle, filtered by the S-N
        # curve), so re-feeding a genuine reversal is harmless.
        new_vals = vals[idx >= boundary - 1]
        if (new_vals.size and idx[idx >= boundary - 1][-1] == combined.size - 1
                and not last_is_reversal):
            new_vals = new_vals[:-1]      # end-of-chunk extension, not a run end
        self._run_e1049(new_vals)
        self._tail = chunk[-self.tail_samples:]
        return self.damage

    def finish(self) -> float:
        """Close the record: feed the final sample, then residual -> half cycles."""
        if self._tail.size:
            self._run_e1049([float(self._tail[-1])])
        stack = self._stack
        for i in range(len(stack) - 1):
            self._emit([Cycle(range=abs(stack[i + 1] - stack[i]), count=0.5,
                           mean=0.5 * (stack[i + 1] + stack[i]))])
        self._stack = []
        self._tail = np.zeros(0)
        return self.damage

    @property
    def remaining_life_cycles(self) -> float:
        """Cycles to D = 1 at the current equivalent rate (inf if D = 0)."""
        if self.damage <= 0.0:
            return float("inf")
        return float(self.cycles_counted / self.damage)
