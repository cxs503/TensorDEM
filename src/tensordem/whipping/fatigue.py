"""Miner linear damage accumulation with DNV-RP-C203 S-N curves.

The two-slope S-N relation of DNV-RP-C203 (Table 2-1) is

    N = a1 dSigma^-m1   for N <= N_knee (m1 = 3)
    N = a2 dSigma^-m2   for N >  N_knee (m2 = 5, knee at N = 1e7)

with the knee stress ``dSigma_1 = (a1/a2)^(1/(m2-m1))`` -- ``52.6 MPa`` for
curve D in air and ``47.0 MPa`` for curve D in seawater with cathodic
protection, both of which are asserted in the tests.  The coefficients are
entered from the standard's table (curve D is the usual choice for a hull
girder detail in air); the primary document is not part of this repository,
so the values are cross-checked against the published knee stresses.

Damage follows Miner's rule, ``D = sum n_i / N_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = ["SNCurve", "DNV_CURVES", "cycles_to_failure", "miner_damage"]


@dataclass(frozen=True)
class SNCurve:
    name: str
    log_a1: float
    m1: int
    log_a2: float
    m2: int
    n_knee: float = 1.0e7

    @property
    def a1(self) -> float:
        return 10.0 ** self.log_a1

    @property
    def a2(self) -> float:
        return 10.0 ** self.log_a2

    def knee_stress_MPa(self) -> float:
        # equality of the two branches at N = n_knee (dSigma in MPa)
        return (self.a2 / self.a1) ** (1.0 / (self.m2 - self.m1))

    def knee_stress(self) -> float:
        """Knee stress range in Pa (the tabulated value is in MPa)."""
        return self.knee_stress_MPa() * 1.0e6

    def cycles_to_failure(self, dsigma) -> np.ndarray:
        # the tabulated DNV coefficients take dSigma in MPa; the API is SI (Pa)
        s = np.asarray(dsigma, dtype=float) / 1.0e6
        if np.any(s <= 0.0):
            raise ValueError("stress ranges must be positive")
        out = np.empty_like(s)
        low = s >= self.knee_stress_MPa()
        out[low] = self.a1 * s[low] ** (-self.m1)
        out[~low] = self.a2 * s[~low] ** (-self.m2)
        return out

    def as_dict(self) -> dict:
        return {"name": self.name, "log_a1": self.log_a1, "m1": self.m1,
                "log_a2": self.log_a2, "m2": self.m2, "n_knee": self.n_knee,
                "knee_stress_MPa": self.knee_stress() / 1.0e6}


DNV_CURVES = {
    "D_air": SNCurve("DNV-RP-C203 D in air", 12.164, 3, 15.606, 5),
}


def cycles_to_failure(curve: SNCurve, dsigma) -> np.ndarray:
    return curve.cycles_to_failure(dsigma)


def miner_damage(curve: SNCurve, ranges, counts) -> dict:
    """Miner damage from a set of stress ranges and cycle counts."""
    ds = np.asarray(ranges, dtype=float).ravel()
    n = np.asarray(counts, dtype=float).ravel()
    if ds.size != n.size:
        raise ValueError("ranges and counts must have equal length")
    if ds.size == 0:
        return {"damage": 0.0, "cycles": 0.0, "max_range_MPa": 0.0,
                "equivalent_range_MPa": 0.0, "min_life_cycles": 0.0}
    keep = ds > 0.0
    ds, n = ds[keep], n[keep]
    if ds.size == 0:
        return {"damage": 0.0, "cycles": 0.0, "max_range_MPa": 0.0,
                "equivalent_range_MPa": 0.0, "min_life_cycles": 0.0}
    life = curve.cycles_to_failure(ds)
    dmg = float(np.sum(n / life))
    eq = float((np.sum(n * ds ** curve.m1) / np.sum(n)) ** (1.0 / curve.m1))
    return {
        "damage": dmg,
        "cycles": float(np.sum(n)),
        "max_range_MPa": float(ds.max()) / 1e6,
        "equivalent_range_MPa": eq / 1e6,
        "min_life_cycles": float(life.min()),
    }


def damage_share(d_total: float, d_part: float) -> float:
    """Fraction of the total damage contributed by a part (guards zero)."""
    if d_total <= 0.0:
        return 0.0
    return float(d_part / d_total)