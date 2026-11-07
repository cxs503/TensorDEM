"""Rainflow cycle counting (ASTM E1049-85 three-point algorithm).

The input is a stress (or load) history; turning points are extracted first and
the standard "range pair" algorithm is then applied:

1. read the next turning point;
2. form the two most recent ranges ``X`` (older) and ``Y`` (newer);
3. if ``X < Y`` count ``X`` as one full cycle and discard the two points that
   bound it, then repeat; otherwise continue reading;
4. the points left on the stack form the residue and their consecutive ranges
   are counted as half cycles.

The result is a list of ``(range, count, mean)`` triples, with ``range`` the
stress range ``dSigma`` and ``count`` in cycles (1.0 for closed cycles, 0.5 for
residue half cycles).  For a closed turning-point sequence the counts satisfy
``sum(counts) = (n_turning_points - 1) / 2``, which is used as an invariant in
the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = ["Cycle", "turning_points", "rainflow", "range_histogram"]


@dataclass(frozen=True)
class Cycle:
    range: float
    count: float
    mean: float


def turning_points(signal) -> np.ndarray:
    """Reduce a sampled signal to its alternating peaks and valleys."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("signal must contain at least two samples")
    tp = [float(x[0])]
    for value in x[1:]:
        value = float(value)
        if value == tp[-1]:
            continue
        if len(tp) == 1:
            tp.append(value)
        elif (tp[-1] - tp[-2]) * (value - tp[-1]) < 0.0:
            tp.append(value)          # direction reversal -> new turning point
        else:
            tp[-1] = value            # still extending the current extremum
    return np.asarray(tp, dtype=float)


def rainflow(signal) -> list[Cycle]:
    """Count cycles in ``signal`` with the ASTM E1049-85 range-pair algorithm."""
    pts = turning_points(signal)
    if pts.size < 2:
        return []
    stack: list[float] = []
    cycles: list[Cycle] = []
    for value in pts:
        stack.append(float(value))
        while len(stack) >= 3:
            x_range = abs(stack[-2] - stack[-3])
            y_range = abs(stack[-1] - stack[-2])
            if y_range >= x_range:
                cycles.append(Cycle(range=x_range, count=1.0,
                                    mean=0.5 * (stack[-3] + stack[-2])))
                del stack[-3:-1]
            else:
                break
    for i in range(len(stack) - 1):
        cycles.append(Cycle(range=abs(stack[i + 1] - stack[i]), count=0.5,
                            mean=0.5 * (stack[i + 1] + stack[i])))
    return cycles


def range_histogram(cycles, bin_edges=None) -> dict:
    """Aggregate rainflow cycles into a stress-range histogram."""
    if not cycles:
        return {"bin_edges": [], "counts": []}
    ranges = np.array([c.range for c in cycles])
    counts = np.array([c.count for c in cycles])
    if bin_edges is None:
        top = float(ranges.max())
        bin_edges = np.linspace(0.0, top * 1.0001 if top > 0 else 1.0, 11)
    hist, edges = np.histogram(ranges, bins=bin_edges, weights=counts)
    return {"bin_edges": edges.tolist(), "counts": hist.tolist()}


def total_count(cycles) -> float:
    return float(sum(c.count for c in cycles))


def equivalent_stress_range(cycles, slope: float = 3.0) -> float:
    """Damage-equivalent stress range ``(sum n dS^m / sum n)^(1/m)``."""
    n = np.array([c.count for c in cycles])
    ds = np.array([c.range for c in cycles])
    if n.size == 0 or n.sum() <= 0:
        return 0.0
    return float((np.sum(n * ds ** slope) / n.sum()) ** (1.0 / slope))


def check_closed(cycles, n_turning: int) -> bool:
    """Invariant: ``sum(counts) = (n_turning - 1) / 2`` for a closed sequence."""
    return math.isclose(total_count(cycles), 0.5 * (n_turning - 1), rel_tol=1e-12)