"""Rigid-body / elastic separation of a strain or load record at 0.5 Hz.

A hull-girder strain record contains two physically different contributions:

* ``< 0.5 Hz``  load-following, quasi-static bending plus rigid-body motion of
  the hull as a whole.  At these frequencies the structure answers the ice load
  quasi-statically (Kelvin-Kirchhoff regime), so the signal follows the load
  envelope; a rigid body as such produces no bending strain, the band is therefore
  the *rigid / quasi-static* component of the measured record.
* ``> 0.5 Hz``  elastic vibration of the hull girder (springing and whipping),
  i.e. the response of the wet modes around and above the first wet natural
  frequency.

The split is a zero-phase Butterworth decomposition (no phase distortion, so the
two bands can be added back exactly).  The complement is used for the high band
so that the two band energies add up to the total variance — a hard invariant the
tests check — instead of relying on two independent filters whose transition
bands would leak energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = ["RigidElasticSplit", "split_rigid_elastic", "separation_criteria"]


@dataclass(frozen=True)
class RigidElasticSplit:
    """Band energies of a record split at ``cutoff_hz``.

    ``rigid + elastic == signal`` holds exactly; the energies are second moments
    of the two bands (the record mean is counted as quasi-static), so
    ``rigid_fraction + elastic_fraction`` is 1 up to the (small) cross term
    ``2 mean(rigid * elastic)``, reported as ``energy_closure``.
    """

    cutoff_hz: float
    total_energy: float          # second moment (DC included)
    rigid_energy: float
    elastic_energy: float
    rigid: np.ndarray            # < cutoff, quasi-static / rigid-body band
    elastic: np.ndarray          # >= cutoff, elastic vibration band

    # ---- AC (fluctuation) energies: what the classification actually uses -----
    @property
    def mean(self) -> float:
        """Signal mean (static offset, carried by the rigid band by definition)."""
        return float(self.rigid.mean())

    @property
    def total_energy_ac(self) -> float:
        return float(np.mean((self.rigid + self.elastic - self.mean) ** 2))

    @property
    def rigid_energy_ac(self) -> float:
        return float(np.mean((self.rigid - self.mean) ** 2))

    @property
    def elastic_energy_ac(self) -> float:
        return float(np.mean(self.elastic**2))

    @property
    def rigid_fraction(self) -> float:
        """AC energy fraction below the cut-off (classification quantity)."""
        total = self.total_energy_ac
        return 0.0 if total <= 0.0 else self.rigid_energy_ac / total

    @property
    def elastic_fraction(self) -> float:
        """AC energy fraction above the cut-off (classification quantity)."""
        total = self.total_energy_ac
        return 0.0 if total <= 0.0 else self.elastic_energy_ac / total

    def as_dict(self) -> dict:
        return {
            "cutoff_hz": self.cutoff_hz,
            "total_energy": self.total_energy,
            "rigid_energy": self.rigid_energy,
            "elastic_energy": self.elastic_energy,
            "rigid_fraction": self.rigid_fraction,
            "elastic_fraction": self.elastic_fraction,
            "rigid_fraction_dc_included": (self.rigid_energy / self.total_energy
                                           if self.total_energy > 0.0 else 0.0),
            "elastic_fraction_dc_included": (self.elastic_energy / self.total_energy
                                             if self.total_energy > 0.0 else 0.0),
            "mean_signal": self.mean,
            "energy_closure": (self.rigid_energy + self.elastic_energy) / self.total_energy
            if self.total_energy > 0.0 else 0.0,
        }


def split_rigid_elastic(signal, dt: float, *, cutoff_hz: float = 0.5,
                        order: int = 4) -> RigidElasticSplit:
    """Split a record into a quasi-static/rigid band and an elastic band."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 16:
        raise ValueError("record too short for a 0.5 Hz separation")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    fs = 1.0 / dt
    if not 0.0 < cutoff_hz < 0.5 * fs:
        raise ValueError("cutoff must lie strictly inside the Nyquist band")
    sos = butter(order, cutoff_hz / (0.5 * fs), btype="low", output="sos")
    rigid = sosfiltfilt(sos, x)
    elastic = x - rigid                      # exact complement, no band leakage
    # second moments without mean removal: the mean of the record is a static
    # (quasi-static) contribution and belongs to the rigid band, so the two band
    # energies close on the total second moment up to the cross term
    return RigidElasticSplit(
        cutoff_hz=cutoff_hz,
        total_energy=float(np.mean(x**2)),
        rigid_energy=float(np.mean(rigid**2)),
        elastic_energy=float(np.mean(elastic**2)),
        rigid=rigid,
        elastic=elastic,
    )


def separation_criteria(rigid_fraction: float, elastic_fraction: float, *,
                        rigid_threshold: float = 0.80,
                        elastic_threshold: float = 0.70) -> dict:
    """Acceptance verdicts of the rigid/elastic classification.

    A record is classified rigid when more than ``rigid_threshold`` of its energy
    sits below 0.5 Hz and elastic when more than ``elastic_threshold`` sits above
    it.  The thresholds are asymmetric on purpose: the quasi-static band always
    survives filtering of a vibration record (the load envelope is low frequency),
    so a *conclusive* elastic verdict needs a larger margin.
    """
    return {
        "rigid_fraction": float(rigid_fraction),
        "elastic_fraction": float(elastic_fraction),
        "rigid_dominated": bool(rigid_fraction > rigid_threshold),
        "elastic_dominated": bool(elastic_fraction > elastic_threshold),
        "rigid_threshold": rigid_threshold,
        "elastic_threshold": elastic_threshold,
    }