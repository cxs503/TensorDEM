"""Attribution gates for bonded-DEM specimen strength tests.

Why this module exists
----------------------
``docs/ice_bond_specimen_rate_dependence.md`` (mission m6464) reported a 4x4x3
opposed-platen specimen whose fitted strength exponent (0.244 linear+creep,
0.198 maxwell_wiechert+creep) could not be compared with the analytic ductile
envelope ``1/(m+1) = 1/3``, because the measured strength was "7-9x the
rigid-plate bond bound" and the purely elastic negative control was not flat
either (fitted slope +0.117).  The campaign declared the specimen
*inertia-dominated* and was recorded as an honest FAIL.

Neither claim was measured, and the report contradicts itself: its own
limitation section says "the fastest rate is 3 % of the lattice wave speed, so
inertia is present but small".  Three things are wrong, and all three are
detectable from numbers the engine already records:

1. **The reference bound undercounted the load path.**  ``analytic_brittle_strength``
   assumed ``nx * ny = 16`` bonds cross the gauge section.  The specimen's bond
   network also carries face and body diagonals, so **100** bonds cross the
   plane, and the axial force a bond can carry at its cleavage strain is
   ``k_u * eps_b * dz`` with ``dz`` the bond's axial projection -- so the bound
   is ``100 * k_u * eps_b * L0 / A``, 6.25x the value used.
2. **The dependent variable was post-failure.**  Every "peak" stress in the
   campaign is reached after the bond network is (nearly) fully broken: a debris
   column compacted by the platens can carry arbitrary stress, and a
   pre-failure service stress is the only quantity the bond law controls.
3. **The inertia attribution used the wrong number.**  The *local*
   platen-speed / wave-speed ratio is not the quasi-staticity criterion; the
   specimen-scale one is the wave transit time over the loading time,
   ``Dyn = (H / c) / (eps_b / edot)``.

This module supplies the machine-checkable gates so the mistake cannot recur:

* :func:`bond_path_capacity` -- exact rigid-plate bound from the bond geometry
  (``n`` crossing bonds, axial projection ``dz``), no hand counting.
* :func:`inertia_numbers` -- both inertia numbers, side by side.
* :func:`classify_peak` -- pre-failure service strength versus post-failure
  jamming peak, with the capacity utilisation of each.
* :func:`load_path_integrity_ratio` -- platen reaction against the axial force
  crossing the gauge plane, so a hidden load path (or an accounting slip) is
  caught rather than argued about.

All are pure functions of recorded numbers, so they cost nothing and can gate
any future bonded-specimen campaign.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BondPathCapacity",
    "InertiaNumbers",
    "bond_path_capacity",
    "classify_peak",
    "inertia_numbers",
    "load_path_integrity_ratio",
    "particle_mass",
]


def particle_mass(radius: float, density: float) -> float:
    """Mass of one sphere (kg)."""
    if radius <= 0.0 or density <= 0.0:
        raise ValueError("radius and density must be positive")
    return density * 4.0 / 3.0 * math.pi * radius ** 3


@dataclass(frozen=True)
class BondPathCapacity:
    """Rigid-plate bound of the *intact* bond load path across a gauge plane.

    A bond of rest length ``L`` and axial projection ``dz`` that is stretched by
    the global axial strain ``eps`` carries ``k_u * eps * dz`` in the axial
    direction (its own strain is ``eps * dz / L``, and its axial component is
    ``cos = dz / L`` of the axial force).  It therefore fails when
    ``eps * dz / L = eps_b``, i.e. it carries ``k_u * eps_b * dz`` axially at
    that instant.  Summing over the bonds that cross the plane gives the bound;
    bonds are independent of the global strain at which each fails, so the sum
    is an upper bound for any single instant.
    """

    n_bonds: int
    axial_force_per_bond_N: tuple[float, ...]
    bond_stiffness_N_m: float
    breaking_strain: float
    gross_area_m2: float

    def __post_init__(self) -> None:
        if self.n_bonds <= 0:
            raise ValueError("n_bonds must be positive")
        if len(self.axial_force_per_bond_N) != self.n_bonds:
            raise ValueError("axial_force_per_bond_N must have n_bonds entries")
        if not (self.gross_area_m2 > 0.0 and math.isfinite(self.gross_area_m2)):
            raise ValueError("gross_area_m2 must be finite and positive")

    @property
    def capacity_force_N(self) -> float:
        return float(sum(self.axial_force_per_bond_N))

    @property
    def capacity_stress_Pa(self) -> float:
        return self.capacity_force_N / self.gross_area_m2


def bond_path_capacity(bond_dz, *, bond_stiffness_N_m: float,
                       breaking_strain: float, gross_area_m2: float) -> BondPathCapacity:
    """Build the bound from the axial projections ``dz`` of the crossing bonds.

    ``bond_dz`` is the (signed) axial projection of every bond that crosses the
    gauge plane; the sign is irrelevant and is taken absolutely.
    """
    if bond_stiffness_N_m <= 0.0 or breaking_strain <= 0.0:
        raise ValueError("bond stiffness and breaking strain must be positive")
    forces = tuple(bond_stiffness_N_m * breaking_strain * abs(float(dz))
                   for dz in bond_dz)
    return BondPathCapacity(
        n_bonds=len(forces),
        axial_force_per_bond_N=forces,
        bond_stiffness_N_m=bond_stiffness_N_m,
        breaking_strain=breaking_strain,
        gross_area_m2=gross_area_m2,
    )


@dataclass(frozen=True)
class InertiaNumbers:
    """Dimensionless inertia diagnostics for a constant-platen-speed test."""

    bond_wave_speed_m_s: float
    bond_period_s: float
    platen_speed_ratio: float
    loading_time_ratio: float
    transit_ratio: float
    quasi_static: bool


def inertia_numbers(*, strain_rate: float, breaking_strain: float,
                    specimen_height_m: float, rest_length_m: float,
                    bond_stiffness_N_m: float, radius_m: float,
                    density: float) -> InertiaNumbers:
    """Inertia diagnostics for a specimen loaded at ``strain_rate``.

    ``c = L0 * sqrt(k_u / m)`` is the axial bond wave speed, ``1 / omega`` the
    single-bond oscillation time, ``t_load = eps_b / edot`` the time to reach
    the cleavage strain and ``t_transit = H / c`` the axial wave transit over
    the specimen height.

    ``platen_speed_ratio`` is the *single-platen* speed of the opposed-platen
    protocol (``edot * H / 2``) over ``c`` -- the number the m6464 report quoted
    as "3 % of the lattice wave speed" -- and ``loading_time_ratio`` is
    ``t_load`` over the bond oscillation period.  ``transit_ratio =
    t_transit / t_load`` is the specimen-scale criterion: above 1 the load is
    applied faster than the specimen can equilibrate, whatever the two local
    ratios say.
    """
    if strain_rate <= 0.0:
        raise ValueError("strain_rate must be positive")
    mass = particle_mass(radius_m, density)
    omega = math.sqrt(bond_stiffness_N_m / mass)
    c = rest_length_m * omega
    t_load = breaking_strain / strain_rate
    t_transit = specimen_height_m / c
    transit_ratio = t_transit / t_load
    return InertiaNumbers(
        bond_wave_speed_m_s=c,
        bond_period_s=1.0 / omega,
        platen_speed_ratio=(0.5 * strain_rate * specimen_height_m) / c,
        loading_time_ratio=t_load / (1.0 / omega),
        transit_ratio=transit_ratio,
        quasi_static=transit_ratio < 1.0,
    )


def load_path_integrity_ratio(platen_reaction_N: float,
                              bond_axial_sum_N: float) -> float:
    """Platen reaction over the axial bond force crossing the gauge plane.

    A bonded specimen's platen reaction must be carried by the bonds, so the
    ratio is 1 (to the tolerance of the dynamics) while the load path is
    intact.  A systematic departure means either a second load path (contacts,
    lateral platens, friction) or an accounting slip in the reported stress.

    The engine reports the platen reaction and the bond force with opposite
    signs (compression positive for one, negative for the other), so the ratio
    is formed from magnitudes and is reported positive by construction.
    """
    if bond_axial_sum_N == 0.0:
        raise ValueError("no bond load path: ratio undefined")
    return abs(float(platen_reaction_N)) / abs(float(bond_axial_sum_N))


def classify_peak(rows, capacity_stress_Pa: float) -> dict:
    """Split a stress history into a pre-failure strength and a jamming peak.

    ``rows`` is any iterable of records carrying ``axial_stress_Pa``,
    ``axial_engineering_strain`` and ``broken_bonds``.  The *pre-failure
    strength* is the largest stress reached while no bond has yet broken; the
    *peak* is the largest stress anywhere in the history, which a column of
    debris can raise arbitrarily as the platens compact it.
    """
    history = list(rows)
    if not history:
        raise ValueError("empty history")
    if capacity_stress_Pa <= 0.0:
        raise ValueError("capacity_stress_Pa must be positive")
    peak = max(history, key=lambda r: r["axial_stress_Pa"])
    intact = [r for r in history if int(r["broken_bonds"]) == 0]
    pre = max(intact, key=lambda r: r["axial_stress_Pa"]) if intact else None
    peak_stress = float(peak["axial_stress_Pa"])
    pre_stress = None if pre is None else float(pre["axial_stress_Pa"])
    tolerance = 1.0 + 1e-9
    return {
        "peak_stress_Pa": peak_stress,
        "peak_strain": float(peak["axial_engineering_strain"]),
        "broken_bonds_at_peak": int(peak["broken_bonds"]),
        "peak_is_prefailure": int(peak["broken_bonds"]) == 0,
        "peak_over_capacity": peak_stress / capacity_stress_Pa,
        "prefailure_stress_Pa": pre_stress,
        "prefailure_strain": (
            None if pre is None else float(pre["axial_engineering_strain"])),
        "prefailure_over_capacity": (
            None if pre_stress is None else pre_stress / capacity_stress_Pa),
        "capacity_stress_Pa": capacity_stress_Pa,
        "prefailure_within_capacity": bool(
            pre_stress is not None and pre_stress <= capacity_stress_Pa * tolerance),
        "valid_strength_measurement": bool(
            pre_stress is not None and pre_stress <= capacity_stress_Pa * tolerance),
    }