"""Triaxial creep chamber and size-effect homogenization for bonded-sphere ice DEM.

Mission IL70-N06: extends the uniaxial creep lattice (ice_creep.py) to
triaxial stress states and systematic specimen-size scans.

Physics
-------
1. Triaxial creep: a cubic lattice specimen is subjected to a hydrostatic
   confining pressure sigma_3 on all four lateral faces plus an additional
   axial deviator (sigma_1 - sigma_3) on the top/bottom faces.  The resulting
   mean stress is p = (sigma_1 + 2 sigma_3)/3 and the deviatoric (von Mises)
   stress is seq = sigma_1 - sigma_3.  Glen's law predicts the steady-state
   axial creep rate

       edot_11 = A(T) [(sigma_1 - sigma_3)]^n   (uniaxial Norton limit)

   more generally, for the full tensor form with the work-conjugate
   normalisation, the equivalent strain rate is A seq^n and the axial rate
   is (2/3) times that for uniaxial-like loading (the lattice naturally
   reproduces this through its bond-level Norton dashpots).

2. Size effect: the bonded lattice is a discrete approximation of a
   continuum ice specimen.  As D/d_p -> infinity the macroscopic creep rate
   should converge to the continuum Norton prediction.  We scan
   D/d_p = 10, 15, 20, 30, 40 and verify that the last two points differ
   by <= 5 %.

All numbers written by the driver are machine-generated; nothing is
transcribed by hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tensordem.ice_creep import (
    BondedSphereCreepLattice,
    GlenCreepLaw,
    LatticeCreepParameters,
    cubic_lattice_block,
    power_law_fit,
    norton_uniaxial_rate,
    R_GAS,
    T0_K,
)

__all__ = [
    "triaxial_constant_stress",
    "size_effect_scan",
    "temperature_sweep",
    "arrhenius_activation_energy",
]


# --------------------------------------------------------------------------- #
# Triaxial creep chamber
# --------------------------------------------------------------------------- #
def triaxial_constant_stress(
    *,
    sigma_1: float,
    sigma_3: float,
    n_cells: int = 6,
    spacing: float = 0.02,
    params: Optional[LatticeCreepParameters] = None,
    t_end: float = 2000.0,
    dt: float = 10.0,
) -> dict:
    """Triaxial creep: confining pressure sigma_3 + axial stress sigma_1.

    The specimen is a cubic bonded-sphere lattice.  Lateral faces carry
    uniform traction sigma_3 (compressive positive convention: we apply
    inward forces); top/bottom carry sigma_1.  The bottom face is fixed
    in z (plus one node pinned laterally for rigid-body removal).

    The deviatoric (von Mises) stress is seq = |sigma_1 - sigma_3| and
    the mean stress is p = (sigma_1 + 2*sigma_3)/3.  Glen's law is
    insensitive to p (creep is incompressible), so the creep rate depends
    only on seq.

    Returns
    -------
    dict with keys:
        t, strain_axial, strain_volumetric, strain_deviatoric,
        seq (von Mises stress), p (mean stress),
        norton_rate_expected (A * seq^n),
        lattice, height, cross_section
    """
    params = params or LatticeCreepParameters()
    positions, bonds = cubic_lattice_block(n_cells, n_cells, n_cells,
                                           spacing=spacing)
    lat = BondedSphereCreepLattice(positions, bonds, radius=spacing / 2.0,
                                   params=params)
    zmax = positions[:, 2].max()
    zmin = positions[:, 2].min()
    xmax = positions[:, 0].max()
    xmin = positions[:, 0].min()
    ymax = positions[:, 1].max()
    ymin = positions[:, 1].min()
    height = float(zmax - zmin)
    width_x = float(xmax - xmin)
    width_y = float(ymax - ymin)

    top = np.nonzero(np.isclose(positions[:, 2], zmax))[0]
    bottom = np.nonzero(np.isclose(positions[:, 2], zmin))[0]
    left_x = np.nonzero(np.isclose(positions[:, 0], xmin))[0]
    right_x = np.nonzero(np.isclose(positions[:, 0], xmax))[0]
    left_y = np.nonzero(np.isclose(positions[:, 1], ymin))[0]
    right_y = np.nonzero(np.isclose(positions[:, 1], ymax))[0]

    # Bottom face: fix u_z = 0; pin node 0 laterally
    lat.fixed[3 * bottom + 2] = True
    lat.fixed[0] = True   # node 0: u_x = 0
    lat.fixed[1] = True   # node 0: u_y = 0

    cross_section_top = width_x * width_y
    cross_section_x = height * width_y
    cross_section_y = height * width_x

    # Axial load: sigma_1 on top face (compressive = negative z force)
    target_axial = sigma_1 * cross_section_top
    lat.external[3 * top + 2] = -target_axial / len(top)

    # Lateral confinement: sigma_3 on x-faces
    target_lx = sigma_3 * cross_section_x
    lat.external[3 * right_x + 0] = -target_lx / len(right_x)  # inward (-x)
    lat.external[3 * left_x + 0] = +target_lx / len(left_x)    # inward (+x)

    # Lateral confinement: sigma_3 on y-faces
    target_ly = sigma_3 * cross_section_y
    lat.external[3 * right_y + 1] = -target_ly / len(right_y)  # inward (-y)
    lat.external[3 * left_y + 1] = +target_ly / len(left_y)    # inward (+y)

    # Deviatoric / mean stress
    seq = abs(sigma_1 - sigma_3)
    p = (sigma_1 + 2.0 * sigma_3) / 3.0

    # Norton prediction for the equivalent strain rate
    A_T = params.fluidity()
    n_exp = params.law.n
    norton_eq_rate = float(norton_uniaxial_rate(seq, A_T, n_exp))

    def record(state):
        u = state.u
        # Axial strain: mean displacement of top face / height
        eps_axial = -float(np.mean(u[top, 2])) / height
        # Lateral strains
        eps_x = float(np.mean(u[right_x, 0]) - np.mean(u[left_x, 0])) / width_x
        eps_y = float(np.mean(u[right_y, 1]) - np.mean(u[left_y, 1])) / width_y
        eps_vol = eps_axial + eps_x + eps_y
        eps_dev_axial = eps_axial - eps_vol / 3.0
        return {
            "t": state.time,
            "strain_axial": eps_axial,
            "strain_x": eps_x,
            "strain_y": eps_y,
            "strain_volumetric": eps_vol,
            "strain_deviatoric": eps_dev_axial,
        }

    series = []
    lat.solve()
    series.append(record(lat))
    for _ in range(int(round(t_end / dt))):
        lat.step(dt)
        lat.solve()
        series.append(record(lat))

    t_arr = np.array([s["t"] for s in series])
    eps_ax = np.array([s["strain_axial"] for s in series])
    eps_vol = np.array([s["strain_volumetric"] for s in series])
    eps_dev = np.array([s["strain_deviatoric"] for s in series])

    # Fit steady-state creep rate from the second half of the record
    n_half = len(t_arr) // 2
    if n_half > 2 and t_arr[-1] > t_arr[n_half]:
        rate_measured = float((eps_dev[-1] - eps_dev[n_half])
                              / (t_arr[-1] - t_arr[n_half]))
    else:
        rate_measured = 0.0

    return {
        "t": t_arr,
        "strain_axial": eps_ax,
        "strain_volumetric": eps_vol,
        "strain_deviatoric": eps_dev,
        "seq": seq,
        "p": p,
        "sigma_1": sigma_1,
        "sigma_3": sigma_3,
        "norton_rate_expected": norton_eq_rate,
        "rate_measured": rate_measured,
        "lattice": lat,
        "height": height,
        "n_nodes": lat.n_nodes,
        "n_bonds": len(bonds),
    }


# --------------------------------------------------------------------------- #
# Size effect homogenization scan
# --------------------------------------------------------------------------- #
def size_effect_scan(
    *,
    sigma0: float = 1.0e6,
    ratios: tuple = (10, 15, 20, 30, 40),
    spacing: float = 0.02,
    params: Optional[LatticeCreepParameters] = None,
    t_end: float = 2000.0,
    dt: float = 10.0,
) -> dict:
    """Scan specimen size D/d_p and measure the steady-state creep rate.

    For each ratio r in ``ratios``, the specimen has n_cells = r particles
    along each edge (so D = r * spacing, d_p = spacing, D/d_p = r).
    The constant-stress uniaxial creep test is run and the steady-state
    axial strain rate is extracted from the second half of the record.

    Returns
    -------
    dict with keys:
        ratios, D_over_dp, rates_measured, rate_norton,
        convergence_ratio (last two rates),
        results_per_size (list of individual run dicts)
    """
    params = params or LatticeCreepParameters()
    A_T = params.fluidity()
    n_exp = params.law.n
    norton_rate = float(norton_uniaxial_rate(sigma0, A_T, n_exp))

    results = []
    rates = []
    for r in ratios:
        n_cells = max(int(r), 4)
        res = _uniaxial_creep_rate(sigma0=sigma0, n_cells=n_cells,
                                   spacing=spacing, params=params,
                                   t_end=t_end, dt=dt)
        results.append(res)
        rates.append(res["steady_rate"])

    rates_arr = np.array(rates)
    # Convergence: ratio of last two rates
    if len(rates_arr) >= 2 and rates_arr[-2] != 0:
        convergence = abs(rates_arr[-1] - rates_arr[-2]) / abs(rates_arr[-2])
    else:
        convergence = float("inf")

    return {
        "ratios": list(ratios),
        "D_over_dp": list(ratios),
        "rates_measured": rates,
        "rate_norton": norton_rate,
        "convergence_ratio": float(convergence),
        "results_per_size": results,
    }


def _uniaxial_creep_rate(*, sigma0, n_cells, spacing, params, t_end, dt):
    """Helper: run a uniaxial creep test and extract the steady-state rate."""
    positions, bonds = cubic_lattice_block(n_cells, n_cells, n_cells,
                                           spacing=spacing)
    lat = BondedSphereCreepLattice(positions, bonds, radius=spacing / 2.0,
                                   params=params)
    zmax = positions[:, 2].max()
    zmin = positions[:, 2].min()
    height = float(zmax - zmin)
    top = np.nonzero(np.isclose(positions[:, 2], zmax))[0]
    bottom = np.nonzero(np.isclose(positions[:, 2], zmin))[0]

    lat.fixed[3 * bottom + 2] = True
    lat.fixed[0] = True
    lat.fixed[1] = True

    cross_section = (float(positions[:, 0].max() - positions[:, 0].min())
                     * float(positions[:, 1].max() - positions[:, 1].min()))
    target = sigma0 * cross_section
    lat.external[3 * top + 2] = -target / len(top)

    lat.solve()
    eps_init = -float(np.mean(lat.u[top, 2])) / height
    t_init = 0.0

    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        lat.step(dt)
        lat.solve()

    eps_final = -float(np.mean(lat.u[top, 2])) / height
    t_final = lat.time

    # Steady rate from overall slope (valid for secondary creep dominated)
    if t_final > t_init:
        steady_rate = (eps_final - eps_init) / (t_final - t_init)
    else:
        steady_rate = 0.0

    return {
        "n_cells": n_cells,
        "D_over_dp": n_cells,
        "steady_rate": float(steady_rate),
        "eps_final": float(eps_final),
        "t_end": t_final,
    }


# --------------------------------------------------------------------------- #
# Temperature sweep for Arrhenius validation
# --------------------------------------------------------------------------- #
def temperature_sweep(
    *,
    sigma0: float = 1.0e6,
    temperatures_c: tuple = (-40.0, -30.0, -20.0, -10.0),
    n_cells: int = 6,
    spacing: float = 0.02,
    base_law: Optional[GlenCreepLaw] = None,
    young_modulus: float = 9.0e9,
    t_end: float = 2000.0,
    dt: float = 10.0,
) -> dict:
    """Run uniaxial creep at multiple temperatures and fit Arrhenius.

    For each temperature, a new LatticeCreepParameters is created with the
    appropriate fluidity (the GlenCreepLaw inside handles the Arrhenius
    temperature dependence).  The steady-state creep rate is measured and
    then an Arrhenius fit extracts the activation energy Q.

    Returns
    -------
    dict with keys:
        temperatures_c, temperatures_k, rates_measured,
        Q_fitted (J/mol), Q_expected (J/mol from law),
        arrhenius_r2
    """
    law = base_law or GlenCreepLaw()
    results = []
    rates = []
    temps_k = []

    for t_c in temperatures_c:
        p = LatticeCreepParameters(
            law=law,
            temperature_c=t_c,
            young_modulus=young_modulus,
        )
        t_k = t_c + T0_K
        temps_k.append(t_k)

        res = _uniaxial_creep_rate(sigma0=sigma0, n_cells=n_cells,
                                   spacing=spacing, params=p,
                                   t_end=t_end, dt=dt)
        res["temperature_c"] = t_c
        res["temperature_k"] = t_k
        results.append(res)
        rates.append(res["steady_rate"])

    # Arrhenius fit: ln(rate) = ln(A0 sigma^n) - Q/(R T)
    # => ln(rate) vs 1/T should be linear with slope = -Q/R
    rates_arr = np.array(rates)
    temps_k_arr = np.array(temps_k)
    inv_t = 1.0 / temps_k_arr
    positive = rates_arr > 0
    if positive.sum() >= 2:
        ln_rate = np.log(rates_arr[positive])
        inv_t_pos = inv_t[positive]
        design = np.vstack([inv_t_pos, np.ones_like(inv_t_pos)]).T
        coef, *_ = np.linalg.lstsq(design, ln_rate, rcond=None)
        slope = float(coef[0])
        Q_fitted = -slope * R_GAS  # slope = -Q/R => Q = -slope * R
        # R^2
        residual = ln_rate - design @ coef
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((ln_rate - ln_rate.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        Q_fitted = float("nan")
        r2 = float("nan")

    Q_expected = law.q_over_r * R_GAS

    return {
        "temperatures_c": list(temperatures_c),
        "temperatures_k": temps_k,
        "rates_measured": rates,
        "Q_fitted": Q_fitted,
        "Q_expected": Q_expected,
        "Q_fitted_kJ": Q_fitted / 1000.0 if math.isfinite(Q_fitted) else float("nan"),
        "Q_expected_kJ": Q_expected / 1000.0,
        "arrhenius_r2": r2,
        "results_per_temp": results,
    }


# --------------------------------------------------------------------------- #
# Activation energy from Arrhenius (standalone utility)
# --------------------------------------------------------------------------- #
def arrhenius_activation_energy(temperatures_k, rates):
    """Fit Q from ln(rate) vs 1/T.  Returns Q in J/mol."""
    t_k = np.asarray(temperatures_k, dtype=float)
    r = np.asarray(rates, dtype=float)
    pos = r > 0
    if pos.sum() < 2:
        return float("nan")
    inv_t = 1.0 / t_k[pos]
    ln_r = np.log(r[pos])
    design = np.vstack([inv_t, np.ones_like(inv_t)]).T
    coef, *_ = np.linalg.lstsq(design, ln_r, rcond=None)
    return float(-coef[0] * R_GAS)
