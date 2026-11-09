"""Freeze-bond (sintered ice-rubble) cohesive strength law.

Broken-ice rubble that has been left in contact under a negative temperature
gradient and pore-water salinity consolidates into *freeze bonds*: thin necks of
fresh ice that grow across the contact area.  The engineering community describes
their strength with a power law in three variables -- the temperature drop
``Delta T`` below the freezing point of the pore water, the pore-water salinity
``S`` and the consolidation (contact) time ``t`` (Timco & Weeks 2010 for the ice
property background; the freeze-bond kinetic form is the working hypothesis of
this module, not a measured law):

.. math::

    \\sigma_{fb}(\\Delta T, S, t)
        = A \\, (\\Delta T)^{\\alpha} \\, (1 + \\beta S)^{-1} \\, t^{\\gamma}

with ``A`` in Pa K^-alpha s^-gamma, ``beta`` in 1/ppt and ``t`` in seconds.  The
law is used in three places and never silently calibrated:

* ``sigma_fb`` is the **tensile** freeze-bond strength;
* the shear strength is ``sigma_s = kappa_s * sigma_fb`` with
  ``kappa_s = shear_to_tension_ratio``;
* the compressive branch is carried by the *bulk* ice strength
  ``sigma_c`` (= ``IceState.compressive_strength_pa`` in
  :mod:`tensordem.level_ice.formula`), because a freeze bond in compression
  crushes the surrounding ice rather than the bond neck.

Every failure-mode branch carries a fracture energy ``G_f`` (J/m^2) which fixes
the softening length ``delta_f = 2 G_f / sigma`` of the cohesive law in
:mod:`tensordem.freeze_bond.cohesive`.  ``G_f`` is a *declared* parameter here:
no claim is made that it equals the macroscopic ice fracture energy
(:mod:`tensordem.bond_fracture_energy` establishes that mapping for the
lattice-bond model, not for freeze bonds).

Evidence grade: the power-law *form* and its exponent ranges are
WITHHELD-as-physics (self-consistency only).  What this module does verify is
that the DEM bond element reproduces this closed form pointwise, and that the
fracture energy is honoured exactly by the softening branch -- see
``tests/test_freeze_bond_constitutive.py`` and ``tests/test_freeze_bond_energy.py``.

All quantities SI (Pa, K, ppt, s, J/m^2).  ``S`` and ``t`` are per-bond, so the
class is vectorised with :mod:`numpy` throughout: scalars in, scalars out; arrays
in, arrays out.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

__all__ = [
    "FreezeBondLaw",
    "FREEZE_BOND_LAW_REFERENCE",
]


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class FreezeBondLaw:
    """``sigma_fb = A * dT**alpha / (1 + beta*S) * t**gamma``.

    Parameters
    ----------
    coefficient_pa:
        ``A`` [Pa K^-alpha s^-gamma].
    temperature_exponent:
        ``alpha`` [1/K^alpha]. Must be positive: colder pore water -> stronger bond.
    salinity_coefficient_per_ppt:
        ``beta`` [1/ppt]. ``0`` switches the salinity dependence off.
    consolidation_exponent:
        ``gamma`` [1]. Must be positive: longer contact -> stronger bond.
    shear_to_tension_ratio:
        ``kappa_s = sigma_s / sigma_fb`` in ``(0, 1]``.
    fracture_energy_J_m2:
        Mode-I fracture energy of the bond neck, J/m^2 (drives softening).
    shear_fracture_energy_J_m2:
        Mode-II fracture energy; defaults to the mode-I value.
    crush_strength_pa:
        Compressive strength of the bond in the closing branch (bulk ice value).
    min_temperature_drop_K:
        Numerical floor on ``Delta T`` used only to keep the power law finite;
        a bond at ``Delta T <= 0`` (pore water at or above its freezing point)
        does not form at all, which :meth:`tension_strength` reports as zero.
    """

    coefficient_pa: float
    temperature_exponent: float = 1.0
    salinity_coefficient_per_ppt: float = 0.0
    consolidation_exponent: float = 1.0
    shear_to_tension_ratio: float = 0.5
    fracture_energy_J_m2: float = 3.0
    shear_fracture_energy_J_m2: float | None = None
    crush_strength_pa: float = 1.5e6
    min_temperature_drop_K: float = 1.0e-6
    softening_ratio: float | None = None

    # ------------------------------------------------------------- validation
    def __post_init__(self) -> None:
        _positive("coefficient_pa", self.coefficient_pa)
        _positive("temperature_exponent", self.temperature_exponent)
        _positive("consolidation_exponent", self.consolidation_exponent)
        if not math.isfinite(self.salinity_coefficient_per_ppt) \
                or self.salinity_coefficient_per_ppt < 0.0:
            raise ValueError("salinity_coefficient_per_ppt must be finite and >= 0")
        if not 0.0 < self.shear_to_tension_ratio <= 1.0:
            raise ValueError("shear_to_tension_ratio must be in (0, 1]")
        _positive("fracture_energy_J_m2", self.fracture_energy_J_m2)
        if self.shear_fracture_energy_J_m2 is not None:
            _positive("shear_fracture_energy_J_m2", self.shear_fracture_energy_J_m2)
        if self.softening_ratio is not None:
            if not math.isfinite(self.softening_ratio) or self.softening_ratio < 1.0:
                raise ValueError("softening_ratio must be finite and >= 1 "
                                 "(delta_f = softening_ratio * delta_0)")
        _positive("crush_strength_pa", self.crush_strength_pa)
        _positive("min_temperature_drop_K", self.min_temperature_drop_K)

    # ------------------------------------------------------------- the law
    @staticmethod
    def _check(name: str, value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        return arr

    def tension_strength(self, delta_T, salinity=0.0, consolidation_time_s=1.0):
        """Tensile freeze-bond strength ``sigma_fb`` [Pa] from the closed form.

        ``Delta T <= 0`` returns exactly ``0`` (no freezing, no bond), and
        ``t <= 0`` raises: a bond has no strength before it starts growing.
        """
        dT = self._check("delta_T", delta_T)
        s = self._check("salinity", salinity)
        t = self._check("consolidation_time_s", consolidation_time_s)
        if np.any(s < 0.0):
            raise ValueError("salinity must be >= 0 ppt")
        if np.any(t <= 0.0):
            raise ValueError("consolidation_time_s must be > 0 s")
        dT_eff = np.maximum(dT, 0.0)
        cold = np.maximum(dT_eff, self.min_temperature_drop_K)
        sigma = (
            self.coefficient_pa
            * cold ** self.temperature_exponent
            * (1.0 + self.salinity_coefficient_per_ppt * s) ** (-1.0)
            * t ** self.consolidation_exponent
        )
        sigma = np.where(dT_eff > 0.0, sigma, 0.0)
        return float(sigma) if np.ndim(sigma) == 0 else sigma

    def shear_strength(self, delta_T, salinity=0.0, consolidation_time_s=1.0):
        """Shear (mode-II) strength ``sigma_s = kappa_s sigma_fb`` [Pa]."""
        return self.shear_to_tension_ratio * self.tension_strength(
            delta_T, salinity, consolidation_time_s)

    def fracture_energy(self, mode: str = "tension") -> float:
        """Declared cohesive fracture energy [J/m^2] for the given mode."""
        if mode == "tension":
            return self.fracture_energy_J_m2
        if mode == "shear":
            return (self.fracture_energy_J_m2 if self.shear_fracture_energy_J_m2 is None
                    else self.shear_fracture_energy_J_m2)
        raise ValueError("mode must be 'tension' or 'shear'")

    def fracture_energy_at(self, strength_pa, stiffness_pa_per_m: float,
                           *, mode: str = "tension"):
        """Fracture energy actually used by the cohesive element [J/m^2].

        Two mutually exclusive declarations are supported:

        * an explicit ``G_f`` (:meth:`fracture_energy`), or
        * a ``softening_ratio = delta_f / delta_0 >= 1``, in which case
          ``G_f = 0.5 * sigma * delta_f = 0.5 * ratio * sigma^2 / k0``.

        The ratio form exists because an explicit ``G_f`` is only *admissible*
        when ``G_f >= 0.5 sigma^2 / k0`` (otherwise ``delta_f < delta_0``, the
        bond would have to snap inside its elastic branch and the cohesive law
        could not honour both its peak traction and its fracture energy).  With
        a softening ratio the element is energetically consistent for *every*
        bond strength, which matters here because the campaign sweeps
        ``sigma_fb`` over two decades.
        """
        strength = self._check("strength_pa", strength_pa)
        if self.softening_ratio is not None:
            out = 0.5 * self.softening_ratio * np.maximum(strength, 0.0) ** 2 \
                / _positive("stiffness_pa_per_m", float(stiffness_pa_per_m))
        else:
            out = np.full_like(np.asarray(strength, dtype=np.float64),
                               self.fracture_energy(mode))
        return float(out) if np.ndim(out) == 0 else out

    def characteristic_separation(self, strength_pa, fracture_energy_J_m2=None,
                                  *, mode: str = "tension"):
        """Softening length ``delta_f = 2 G_f / sigma`` [m] of the coherent law.

        ``0.5 * sigma * delta_f = G_f`` is the work spent per unit bond area when
        the traction is linear in the separation, which is the law implemented in
        :mod:`tensordem.freeze_bond.cohesive`.
        """
        sigma = self._check("strength_pa", strength_pa)
        if np.any(sigma <= 0.0):
            raise ValueError("strength_pa must be > 0")
        Gf = self.fracture_energy(mode) if fracture_energy_J_m2 is None \
            else _positive("fracture_energy_J_m2", float(fracture_energy_J_m2))
        out = 2.0 * Gf / sigma
        return float(out) if np.ndim(out) == 0 else out

    # ------------------------------------------------------- parameterisation
    @classmethod
    def from_reference_strength(
        cls,
        reference_strength_pa: float,
        *,
        reference_delta_T_K: float = 5.0,
        reference_salinity_ppt: float = 0.0,
        reference_consolidation_s: float = 3600.0,
        **kwargs: Any,
    ) -> "FreezeBondLaw":
        """Build the law so that ``sigma_fb(ref dT, S, t) == reference_strength_pa``.

        ``A = sigma_ref (1 + beta S_ref) t_ref^-gamma dT_ref^-alpha``.  This is the
        anchor used by the arching campaign: the single physical knob is the bond
        strength at the declared reference state, everything else is the
        dimensionless exponent set.
        """
        sigma_ref = _positive("reference_strength_pa", float(reference_strength_pa))
        dT_ref = _positive("reference_delta_T_K", float(reference_delta_T_K))
        t_ref = _positive("reference_consolidation_s", float(reference_consolidation_s))
        s_ref = float(reference_salinity_ppt)
        if s_ref < 0.0:
            raise ValueError("reference_salinity_ppt must be >= 0")
        probe = cls(coefficient_pa=1.0, **kwargs)
        A = (
            sigma_ref
            * (1.0 + probe.salinity_coefficient_per_ppt * s_ref)
            / (dT_ref ** probe.temperature_exponent * t_ref ** probe.consolidation_exponent)
        )
        return cls(coefficient_pa=A, **kwargs)

    # ---------------------------------------------------------------- report
    def as_dict(self) -> dict[str, float]:
        return {
            "coefficient_pa": self.coefficient_pa,
            "temperature_exponent": self.temperature_exponent,
            "salinity_coefficient_per_ppt": self.salinity_coefficient_per_ppt,
            "consolidation_exponent": self.consolidation_exponent,
            "shear_to_tension_ratio": self.shear_to_tension_ratio,
            "fracture_energy_J_m2": self.fracture_energy_J_m2,
            "shear_fracture_energy_J_m2": self.fracture_energy("shear"),
            "crush_strength_pa": self.crush_strength_pa,
        }


#: Declared reference law for the arching campaign.
#:
#: The exponent set ``alpha = 1``, ``beta = 0.05 1/ppt``, ``gamma = 0.5`` and
#: ``kappa_s = 0.5`` is a *declared* hypothesis (freeze-bond strength grows
#: linearly with supercooling, half as fast as the square root of the contact
#: time, and is halved per 20 ppt of pore-water salinity).  The anchor strength
#: is ``sigma_fb = 30 kPa`` at ``Delta T = 5 K``, ``S = 0``, ``t = 1 h``, i.e.
#: ``kappa = sigma_fb / sigma_bulk = 0.02`` against the 1.5 MPa bulk strength of
#: the Lindqvist anchor -- the middle of the freeze-bond literature range.
FREEZE_BOND_LAW_REFERENCE: FreezeBondLaw = FreezeBondLaw.from_reference_strength(
    3.0e4,
    reference_delta_T_K=5.0,
    reference_salinity_ppt=0.0,
    reference_consolidation_s=3600.0,
    temperature_exponent=1.0,
    salinity_coefficient_per_ppt=0.05,
    consolidation_exponent=0.5,
    shear_to_tension_ratio=0.5,
    fracture_energy_J_m2=3.0,
    crush_strength_pa=1.5e6,
    softening_ratio=10.0,
)