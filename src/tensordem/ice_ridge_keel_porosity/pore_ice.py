"""Pore-ice (freeze-bond) closure: the thermal half of keel consolidation.

Why a second mechanism is needed
--------------------------------
A DEM rearrangement of rigid spheres cannot take a keel from ``n = 0.4`` to
``n = 0.2``: random close packing of equal spheres sits at ``n ~ 0.36`` and even
the densest sphere packing (fcc) only reaches ``n = 0.2595``.  Polydispersity
and block crushing help (the gradation sweep quantifies how much) but they
cannot close the gap either.  The remaining pore space of a real keel is filled
by refrozen water and freeze bonds: field drilling of consolidated ridge keels
reports void fractions of 0.2-0.3 against 0.35-0.45 for fresh rubble.

This module models that closure with one physical time scale and one calibrated
capacity:

``tau_pore``
    first-mode conduction time across the pore scale,
    ``tau_pore = L_p^2 / (pi^2 * alpha_eff)`` with ``L_p`` the pore diameter
    obtained from the *DEM packing itself* through the hydraulic pore radius
    ``r_h = n*V_solid / ((1-n)*S_solid)`` and ``L_p = 2 r_h``.  ``alpha_eff`` is
    the thermal diffusivity of the ice/water/salt mixture (~1.3e-6 m2/s).  There
    is no free parameter here: the pore scale comes from the simulated packing.
    No latent-heat or brine-transport limit is applied (documented limitation).

``f_eq``
    the fraction of the pore space that ends up filled with ice.  It is *not*
    predicted: it is calibrated so that the asymptotic porosity equals the
    observed void fraction of a consolidated keel, ``n_cons`` (default 0.20,
    the centre of the 0.2-0.3 range reported for consolidated keels).  Stated
    as a calibration, not as a first-principles result.

The composite porosity evolution is

    n(t) = [n_m + (n_0 - n_m) exp(-t/tau_m)] * [1 - f_eq (1 - exp(-t/tau_p))]

i.e. mechanical rearrangement of the skeleton acts multiplicatively on the
pore space that thermal refreezing subsequently fills.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "PoreIceConfig",
    "hydraulic_pore_radius",
    "pore_scale",
    "pore_ice_time_scale",
    "filled_fraction",
    "pore_filling_fraction_for",
    "COMPOSITE_FORM",
]

COMPOSITE_FORM = "n(t) = [n_m + (n_0-n_m)exp(-t/tau_m)] * [1 - f_eq(1 - exp(-t/tau_p))]"


@dataclass(frozen=True)
class PoreIceConfig:
    """Thermal pore-filling parameters.

    Attributes
    ----------
    alpha_eff : float
        Thermal diffusivity of the ice/water/salt mixture (m2/s). Default 1.3e-6
        (ice 1.2e-6, water 0.14e-6; a pore filled with slush is closer to ice).
    n_consolidated : float
        Observed void fraction of a consolidated keel (calibration target).
        Default 0.20 (range 0.2-0.3).
    """

    alpha_eff: float = 1.3e-6
    n_consolidated: float = 0.20

    def __post_init__(self) -> None:
        if self.alpha_eff <= 0:
            raise ValueError("alpha_eff must be positive")
        if not 0.0 < self.n_consolidated < 1.0:
            raise ValueError("n_consolidated must be in (0, 1)")


def hydraulic_pore_radius(radii: np.ndarray, porosity: float) -> float:
    """Hydraulic radius of the pore space of a sphere packing (m).

    ``r_h = V_pore / S_solid = n*V_solid / ((1-n)*S_solid)``.  Volume is
    ``sum(4/3 pi r^3)`` and wetted area ``sum(4 pi r^2)``, so the result is
    ``(n/(1-n)) * <r^3>/(3<r^2>)``: it grows with block size and with porosity.
    """
    radii = np.asarray(radii, dtype=float)
    if len(radii) == 0:
        raise ValueError("empty packing")
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must be in (0, 1)")
    v_solid = float(np.sum(4.0 / 3.0 * math.pi * radii ** 3))
    s_solid = float(np.sum(4.0 * math.pi * radii ** 2))
    return porosity * v_solid / ((1.0 - porosity) * s_solid)


def pore_scale(radii: np.ndarray, porosity: float) -> float:
    """Characteristic pore *diameter* (m) = ``2 * r_h``."""
    return 2.0 * hydraulic_pore_radius(radii, porosity)


def pore_ice_time_scale(radii: np.ndarray, porosity: float,
                        config: PoreIceConfig | None = None) -> float:
    """``tau_pore = L_p^2 / (pi^2 alpha_eff)`` -- first-mode conduction time (s)."""
    cfg = config or PoreIceConfig()
    lp = pore_scale(radii, porosity)
    return lp ** 2 / (math.pi ** 2 * cfg.alpha_eff)


def filled_fraction(t: np.ndarray | float, tau_pore: float) -> np.ndarray | float:
    """``f(t) = 1 - exp(-t/tau_pore)``: progress of the pore-filling process."""
    return 1.0 - np.exp(-np.asarray(t, dtype=float) / tau_pore)


def pore_filling_fraction_for(n_mech_inf: float, n_consolidated: float) -> float:
    """Pore-filling capacity ``f_eq`` calibrated on the consolidated void ratio.

    ``f_eq = 1 - n_consolidated / n_mech_inf``.  Must lie in (0, 1), i.e. the
    mechanical (DEM) plateau has to be looser than the observed consolidated
    value; otherwise the calibration is inconsistent and an error is raised
    instead of returning a silently clipped number.
    """
    if n_mech_inf <= n_consolidated:
        raise ValueError(
            f"mechanical plateau {n_mech_inf:.4f} is not above the observed "
            f"consolidated porosity {n_consolidated:.4f}: the thermal closure "
            "has nothing to fill")
    f_eq = 1.0 - n_consolidated / n_mech_inf
    if not 0.0 < f_eq < 1.0:
        raise ValueError(f"non-physical pore-filling capacity f_eq={f_eq}")
    return f_eq
