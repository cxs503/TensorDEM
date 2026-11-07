"""Rigid-body / elastic (springing-whipping) separation of strain records.

Convention of the campaign specification: the low-frequency part of the
structural response (below ``cutoff_hz``, 0.5 Hz by default) is the
quasi-static rigid-body motion (slow heave/pitch bending of the whole hull),
the high-frequency part is the elastic vibration (springing/whipping).  The
split is a zero-phase Butterworth low-pass (no phase distortion of the
record) and the complementary high-pass, with the energy shares computed by
Parseval on the time domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..whipping.response import lowpass

__all__ = ["RigidElasticSplit", "rigid_elastic_split"]


@dataclass(frozen=True)
class RigidElasticSplit:
    rigid: np.ndarray
    elastic: np.ndarray
    energy_rigid: float
    energy_elastic: float
    frac_rigid: float
    frac_elastic: float

    @property
    def classification(self) -> str:
        if self.frac_rigid > 0.8:
            return "rigid_dominant"
        if self.frac_elastic > 0.7:
            return "elastic_dominant"
        return "mixed"


def rigid_elastic_split(signal, dt: float, cutoff_hz: float = 0.5,
                        order: int = 4) -> RigidElasticSplit:
    """Split a strain (or force) record into rigid and elastic parts."""
    s = np.asarray(signal, dtype=float).ravel()
    if s.size < 8:
        raise ValueError("record too short for a 4th-order filter")
    rigid = lowpass(s, dt, cutoff_hz, order=order)
    elastic = s - rigid
    e_r = float(np.trapezoid(rigid * rigid, dx=dt))
    e_e = float(np.trapezoid(elastic * elastic, dx=dt))
    tot = e_r + e_e
    return RigidElasticSplit(
        rigid=rigid, elastic=elastic,
        energy_rigid=e_r, energy_elastic=e_e,
        frac_rigid=e_r / tot if tot > 0 else 0.0,
        frac_elastic=e_e / tot if tot > 0 else 0.0,
    )
