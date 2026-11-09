"""Adfreeze -> DEM bond-key adapter (single-point coupling surface).

Why this module exists
======================
The platform adfreeze model lives in TensorPhysics
(``tensorphysics.ice.adfreeze``) and answers one question: *how strong is a
frozen joint between ice and a cold surface*, as

    ``tau_adfreeze = tau_ref(dT) * eta_freeze(t, dT) * phi_rough(Ra) * phi_salt(T, S)``
    ``dT = T_f(S) - T_i``                                                    (1)

with ``tau_ref`` the saturating Frederking-calibrated freshwater closure, ``eta``
the Stefan-growth frozen area fraction, ``phi_rough`` the interlocking
amplification and ``phi_salt = (1 - v_b)**n`` the brine penalty.

Every bonded DEM face in this library (ice rubble, ridge keels, floe fields,
ice-structure contacts) needs the *same* three-step translation of (1) into a
discrete element, and it is easy for one face to get it subtly wrong:

1. **temperature -> undercooling**: ``dT`` is measured against the liquidus of
   the pore water, ``T_f(S)``, not against 0 degC and not against the ice
   temperature.  A transposed liquidus table silently zeroes whole salinity
   ranges (see :data:`LIQUIDUS_NOTE`);
2. **strength -> force**: ``F_break = tau_adfreeze * A``, where ``A`` is the
   *frozen neck* area at the block-block contact.  Multiplying by the Hertzian
   contact patch of a soft DEM contact instead gives the wrong number by the
   ratio of those two areas;
3. **force -> envelope**: the shear threshold is a fixed fraction of the normal
   one (:data:`~tensordem.ice_ridge_keel_freezebond.KAPPA_SHEAR`), so a single
   scalar carries the whole coupled constitutive law.

This module is that translation, with no DEM state attached: a pure function
``(T, t, Ra, S, geometry) -> key`` usable by any campaign, plus a machine-checked
parity report against the TensorPhysics source when that tree is importable.

Evidence grades
---------------
* the adfreeze law itself: **WITHHELD** as physics (it is the platform model:
  calibrated + engineering closures, see ``ADFREEZE_PROVENANCE`` in
  :mod:`tensordem.ice_ridge_keel_freezebond`);
* the *mapping* (1)->DEM key and its unit/threshold chain: **REAL** -- audited
  against the analytic ``tau * A`` by the DEM twin-particle pull test in
  ``tests/test_ice_ridge_keel_freezebond.py``;
* the parity report: **REAL** when TensorPhysics is importable, otherwise
  ``available=False`` with an explicit reason (never a silent pass).

References
----------
- Frederking (1984); Frederking & Barker (1979): adfreeze strength data, -2..-20 degC
- Assur (1958); Cox & Weeks (1983): seawater liquidus / brine salinity
- Timco & Weeks (2010): ice thickness and rubble porosity review
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

from .ice_ridge_keel_freezebond import (
    AdfreezeClosure,
    KAPPA_SHEAR,
    NECK_AREA_FRACTION,
    neck_area_m2,
    seawater_freezing_point,
    stefan_frozen_thickness,
)

__all__ = [
    "LIQUIDUS_NOTE",
    "ADAPTER_PROVENANCE",
    "AdfreezeKeyParams",
    "AdfreezeKey",
    "tau_adfreeze_pa",
    "adfreeze_key",
    "adfreeze_key_table",
    "platform_parity",
]

#: Fingerprint of the liquidus anchors this adapter is consistent with.
#: The TensorDEM mirror of the platform liquidus had its temperature list sorted
#: ascending against a descending brine-salinity list, which mis-paired every
#: anchor (S_b = 78 g/kg read at -10 degC instead of -5 degC, S_b = 35 g/kg at
#: -15 degC instead of -2 degC).  Consequence: ``T_f(32 g/kg)`` came out at
#: -15.9 degC instead of -1.83 degC, so every saline joint warmer than that was
#: silently assigned zero adfreeze strength -- i.e. no freeze bonds at all in
#: any realistic (saline) ridge keel.  :func:`platform_parity` now audits this
#: against the upstream source whenever it is importable.
LIQUIDUS_NOTE = (
    "liquidus anchors: (T, S_b) = (-0.1, 2) (-1, 17.5) (-2, 35) (-5, 78) "
    "(-10, 143) (-15, 197) (-20, 240) (-23, 255) degC / g/kg, NaCl-H2O "
    "liquidus (Assur 1958, Cox & Weeks 1983); T_f(32 g/kg) = -1.83 degC"
)

ADAPTER_PROVENANCE = {
    "tau_adfreeze": "tensorphysics.ice.adfreeze (platform model), WITHHELD as physics",
    "liquidus": "literature_calibrated (Assur 1958 / Cox & Weeks 1983 phase diagram)",
    "neck_area_fraction": "engineering_estimate (+-50%, linear in keel cohesion)",
    "kappa_shear": "engineering_estimate (freeze-bond shear:tension ratio)",
    "f_break": "physics_derived: F_break = tau_adfreeze * A_neck",
}


# ---------------------------------------------------------------------------
# Parameters and key
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdfreezeKeyParams:
    """Inputs of one adfreeze joint state.

    ``temperature_c`` is the *ice / joint* temperature, ``salinity_gkg`` the bulk
    salinity of the ice at the joint (0 for freshwater ice), ``contact_time_s``
    the time the two bodies have been frozen together, ``roughness_um`` the
    arithmetic mean roughness of the counter-surface.
    """

    temperature_c: float
    contact_time_s: float = 3.6e3
    salinity_gkg: float = 0.0
    roughness_um: float = 2.5
    gap_scale_m: float = 5.0e-3

    def __post_init__(self) -> None:
        if self.contact_time_s < 0.0:
            raise ValueError("contact_time_s must be non-negative")
        if self.salinity_gkg < 0.0:
            raise ValueError("salinity_gkg must be non-negative")
        if self.roughness_um < 0.0:
            raise ValueError("roughness_um must be non-negative")
        if self.gap_scale_m <= 0.0:
            raise ValueError("gap_scale_m must be positive")

    def closure(self) -> AdfreezeClosure:
        return AdfreezeClosure(
            contact_time_s=self.contact_time_s,
            salinity_gkg=self.salinity_gkg,
            roughness_um=self.roughness_um,
            gap_scale_m=self.gap_scale_m,
        )


@dataclass(frozen=True)
class AdfreezeKey:
    """A DEM freeze-bond key derived from one adfreeze joint state."""

    params: AdfreezeKeyParams
    freezing_point_c: float
    undercooling_k: float
    tau_adfreeze_pa: float
    tau_reference_pa: float
    frozen_area_fraction: float
    roughness_factor: float
    salinity_factor: float
    neck_area_m2: float
    f_break_n: float
    f_shear_break_n: float
    frozen_thickness_m: float
    provenance: dict = field(default_factory=lambda: dict(ADAPTER_PROVENANCE))

    @property
    def frozen(self) -> bool:
        """``True`` when the joint is cold enough to carry a freeze bond."""
        return self.tau_adfreeze_pa > 0.0

    def as_dict(self) -> dict:
        out = asdict(self)
        out["frozen"] = self.frozen
        return out


# ---------------------------------------------------------------------------
# The coupling itself
# ---------------------------------------------------------------------------
def tau_adfreeze_pa(params: AdfreezeKeyParams) -> float:
    """Platform adfreeze joint strength [Pa] at ``params``.

    Zero when the joint is at or above the liquidus of its pore water
    (``dT <= 0``): no unfrozen-water driving force, no bond.  That is the
    ``dT = 0 -> F_break = 0`` negative control in one line.
    """
    p = params
    return 1.0e6 * p.closure().tau_mpa(p.temperature_c)


def adfreeze_key(
    params: AdfreezeKeyParams,
    radius_i: float,
    radius_j: float,
    *,
    neck_fraction: float = NECK_AREA_FRACTION,
    kappa_shear: float = KAPPA_SHEAR,
) -> AdfreezeKey:
    """Map one adfreeze joint state onto a DEM bond key.

    ``F_break = tau_adfreeze * A_neck`` with ``A_neck = pi (f r_min)^2`` the
    frozen neck cross-section inherited from
    :func:`tensordem.ice_ridge_keel_freezebond.neck_area_m2`; the neck area is
    a *material* property of the bond (fresh ice grown across the contact), not
    of the elastic contact patch.
    """
    if radius_i <= 0.0 or radius_j <= 0.0:
        raise ValueError("radii must be positive")
    if kappa_shear <= 0.0:
        raise ValueError("kappa_shear must be positive")
    p = params
    state = p.closure().state(p.temperature_c)
    area = neck_area_m2(radius_i, radius_j, neck_fraction)
    tau_pa = state.tau_adfreeze_mpa * 1.0e6
    f_break = tau_pa * area
    h_frozen = stefan_frozen_thickness(state.undercooling_k, p.contact_time_s)
    return AdfreezeKey(
        params=p,
        freezing_point_c=state.freezing_point_c,
        undercooling_k=state.undercooling_k,
        tau_adfreeze_pa=tau_pa,
        tau_reference_pa=state.tau_reference_mpa * 1.0e6,
        frozen_area_fraction=state.frozen_fraction,
        roughness_factor=state.roughness,
        salinity_factor=state.salinity,
        neck_area_m2=area,
        f_break_n=f_break,
        f_shear_break_n=kappa_shear * f_break,
        frozen_thickness_m=h_frozen,
    )


def adfreeze_key_table(
    params: Iterable[AdfreezeKeyParams],
    radius_i: float,
    radius_j: float,
    **kwargs,
) -> list:
    """Vector form of :func:`adfreeze_key` -- one dict per parameter set."""
    return [adfreeze_key(p, radius_i, radius_j, **kwargs).as_dict() for p in params]


# ---------------------------------------------------------------------------
# Parity against the platform source
# ---------------------------------------------------------------------------
DEFAULT_PARITY_GRID = dict(
    temperatures_c=(-2.0, -5.0, -10.0, -20.0),
    contact_times_s=(1.0e3, 3.6e3, 1.0e4),
    salinities_gkg=(0.0, 5.0, 32.0, 60.0),
    roughnesses_um=(2.5, 20.0),
)


def platform_parity(
    temperatures_c: Sequence[float] = DEFAULT_PARITY_GRID["temperatures_c"],
    contact_times_s: Sequence[float] = DEFAULT_PARITY_GRID["contact_times_s"],
    salinities_gkg: Sequence[float] = DEFAULT_PARITY_GRID["salinities_gkg"],
    roughnesses_um: Sequence[float] = DEFAULT_PARITY_GRID["roughnesses_um"],
) -> dict:
    """Audit the mirrored closure against ``tensorphysics.ice.adfreeze``.

    Returns a dict with ``available`` (whether the upstream tree could be
    imported), ``max_rel_error`` over the grid, the worst case and the number of
    compared points.  When TensorPhysics is absent the result is
    ``available=False`` + ``reason`` -- never an implicit pass.
    """
    if importlib.util.find_spec("tensorphysics") is None:
        return {
            "available": False,
            "reason": "tensorphysics is not importable in this environment",
            "max_rel_error": None,
            "n_points": 0,
            "worst": None,
        }
    from tensorphysics.ice import adfreeze as ref

    worst = None
    max_rel = 0.0
    n = 0
    for T in temperatures_c:
        for t in contact_times_s:
            for S in salinities_gkg:
                for Ra in roughnesses_um:
                    got = tau_adfreeze_pa(
                        AdfreezeKeyParams(temperature_c=T, contact_time_s=t,
                                          salinity_gkg=S, roughness_um=Ra)
                    )
                    want = 1.0e6 * ref.adfreeze_strength(T, t, S, Ra)
                    denom = abs(want) if want != 0.0 else 1.0
                    rel = abs(got - want) / denom
                    n += 1
                    if rel > max_rel:
                        max_rel = rel
                        worst = {"T_C": T, "t_s": t, "S_gkg": S, "Ra_um": Ra,
                                 "mirror_Pa": got, "platform_Pa": want}
    tf_mirror = seawater_freezing_point(32.0)
    tf_ref = ref.seawater_freezing_point(32.0)
    return {
        "available": True,
        "max_rel_error": float(max_rel),
        "n_points": int(n),
        "worst": worst,
        "liquidus_Tf_32gkg_mirror_C": float(tf_mirror),
        "liquidus_Tf_32gkg_platform_C": float(tf_ref),
        "liquidus_rel_error": float(abs(tf_mirror - tf_ref) / abs(tf_ref)),
        "liquidus_note": LIQUIDUS_NOTE,
        "parity_ok": bool(max_rel <= 1e-12 and abs(tf_mirror - tf_ref) <= 1e-12),
    }