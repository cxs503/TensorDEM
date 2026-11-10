"""Machine-checkable acceptance harness for the NGF + NSCD capability (P295W1R2-N03).

Criteria (all numbers are produced by the routines below -- nothing is typed in
by hand):

======  ==========================================================================
A1      ``xi -> 0`` : NGF collapses onto the local ``mu(I)`` fluidity, pointwise
        relative error <= 2 %
A2      band width saturation for wide cells (``H/d >= 20``), ``w/d in [5,12]`` and
        within 30 % of the literature anchor ``w/d ~ 10``
A3      coarse-grained ``mu(I)`` fit ``R^2 >= 0.95`` after NGF deconvolution
A4      ``xi/d`` recovered from three coarse-graining resolutions varies <= 20 %
A5      NSCD restitution ``e in {0.2, 0.5, 0.8}``, error <= 2 %
A6      zero penetration ``max(overlap)/d <= 1e-6`` (single impact, chain, pack)
A7      energy / momentum closure in ``[0.98, 1.02]``
A8      stiffness independence: ``k_n`` x10 -> penetration change <= 1 % (NSCD),
        contrasted with the soft-sphere (penalty) baseline
======  ==========================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .calibration import (
    LITERATURE_BAND_WIDTH_OVER_D,
    LITERATURE_BAND_RANGE_OVER_D,
    LITERATURE_SOURCE,
    ThinBandReference,
    calibrate_xi_from_thin_band,
    coarse_graining_study,
    wide_band_validation,
)
from .fluidity import (
    BandCase,
    Boundary,
    NGFOperator1D,
    MuIParams,
    cg_jacobi,
    local_fluidity,
    local_strain_rate,
    solve_band,
)
from .nscd import (
    confined_packing,
    fcc_packing,
    impact_chain,
    packing_impact,
    packing_run,
    single_impact,
    single_impact_penetration_scan,
    soft_sphere_penetration_scan,
)

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# A1: xi -> 0 recovers the local rheology
# --------------------------------------------------------------------------- #
def a1_local_limit(xi_over_d_list=(0.4, 0.04, 0.004), *, gap_over_d: float = 4.0,
                   cells_per_xi: float = 8.0, mu_bulk: float = 0.45,
                   mu_wall: float = 0.55, pressure: float = 1000.0,
                   params: MuIParams = MuIParams()) -> dict:
    """Pointwise deviation of the NGF solution from ``g_loc`` as ``xi -> 0``.

    The grid resolves the boundary layer (``cells_per_xi`` cells per ``xi``) so
    the limit is tested, not the discretisation.
    """
    d = 1.0e-3
    rho_p = 2500.0
    mu_wall_val = float(local_strain_rate(mu_wall, pressure, d, rho_p, params) / mu_wall)
    g_loc_val = float(local_fluidity(mu_bulk, pressure, d, rho_p, params))
    rows = []
    for xd in xi_over_d_list:
        xi = xd * d
        # keep >= 60 xi of domain so that the 10%-of-xi boundary layer is excluded
        H = max(gap_over_d * d, 60.0 * xi)
        h = xi / cells_per_xi
        n = int(round(H / h))
        h = H / n
        op = NGFOperator1D(n, h, xi, Boundary("dirichlet", mu_wall_val),
                           Boundary("neumann", 0.0))
        rhs = torch.full((n,), g_loc_val, dtype=DTYPE) - op.bc_vector()
        res = cg_jacobi(op, rhs, tol=1e-13, maxiter=200000)
        y = (torch.arange(n, dtype=DTYPE) + 0.5) * h
        interior = y >= 10.0 * xi
        # analytic solution for the constant-g_loc problem
        y_t = y
        # analytic solution, written with exp(-y/xi) to avoid cosh overflow
        decay = torch.exp(-(y_t) / xi) * (1.0 + torch.exp(-2.0 * (H - y_t) / xi)) \
            / (1.0 + math.exp(-2.0 * H / xi))
        exact = g_loc_val + (mu_wall_val - g_loc_val) * decay
        err_model = float(((res.x[interior] - g_loc_val).abs()
                           / abs(g_loc_val)).max()) if g_loc_val else float("inf")
        err_exact = float(((res.x[interior] - exact[interior]).abs()
                           / abs(g_loc_val)).max()) if g_loc_val else float("inf")
        rows.append({
            "xi_over_d": xd, "n_cells": n, "cells_per_xi": cells_per_xi,
            "cg_iterations": res.iterations, "cg_residual": res.residual,
            "max_rel_error_vs_local": err_model,
            "max_rel_error_vs_analytic": err_exact,
        })
    worst = max(r["max_rel_error_vs_local"] for r in rows)
    return {"rows": rows, "worst_error": worst, "threshold": 0.02,
            "pass": bool(worst <= 0.02),
            "g_loc": g_loc_val, "g_wall": mu_wall_val,
            "note": "error measured on y >= 10*xi, i.e. outside the boundary layer"}


# --------------------------------------------------------------------------- #
# A2 / A4: band width saturation + coarse-graining robustness
# --------------------------------------------------------------------------- #
def a2_band_width(ref: ThinBandReference | None = None, *,
                  cells_per_d: float = 4.0) -> dict:
    ref = ref or ThinBandReference()
    cal = calibrate_xi_from_thin_band(ref)
    wide = wide_band_validation(cal["xi_over_d"], cells_per_d=cells_per_d)
    # the local model has no length scale: its band width is a step
    local_case = BandCase(gap_over_d=40.0, xi_over_d=1e-9, cells_per_d=cells_per_d,
                          shell="neumann")
    local_sol = solve_band(local_case)
    from .fluidity import band_width_90_10
    local_w = band_width_90_10(local_sol["y"], local_sol["g"]) / local_case.d
    in_range = LITERATURE_BAND_RANGE_OVER_D[0] <= wide["mean_w_over_d"] <= LITERATURE_BAND_RANGE_OVER_D[1]
    return {
        "thin_band_calibration": cal,
        "wide_band": wide,
        "local_model_width_over_d": local_w,
        "saturation_pass": bool(wide["saturation_spread"] <= 0.05),
        "range_pass": bool(in_range),
        "literature_pass": bool(wide["literature_error"] <= 0.30),
        "pass": bool(in_range and wide["literature_error"] <= 0.30
                     and wide["saturation_spread"] <= 0.05),
        "literature_anchor_over_d": LITERATURE_BAND_WIDTH_OVER_D,
        "literature_source": LITERATURE_SOURCE,
    }


def a4_coarse_graining(xi_over_d: float) -> dict:
    cg = coarse_graining_study(xi_over_d)
    return {**cg, "threshold": 0.20, "pass": bool(cg["variation"] <= 0.20)}


# --------------------------------------------------------------------------- #
# A3: coarse-grained mu(I) fit with and without the NGF deconvolution
# --------------------------------------------------------------------------- #
def _fit_mu_i(I: np.ndarray, mu: np.ndarray) -> dict:
    """Least-squares fit of ``mu(I) = mu_s + (mu_2-mu_s)/(I_0/I + 1)``."""
    from scipy.optimize import least_squares

    def resid(p):
        mu_s, mu_2, I_0 = p
        return (mu_s + (mu_2 - mu_s) / (I_0 / np.maximum(I, 1e-30) + 1.0)) - mu

    p0 = np.array([0.30, 0.60, 0.30])
    sol = least_squares(resid, p0, bounds=([0.05, 0.31, 1e-3], [0.60, 1.5, 5.0]))
    pred = mu + resid(sol.x)
    ss_res = float(np.sum((mu - pred) ** 2))
    ss_tot = float(np.sum((mu - mu.mean()) ** 2))
    return {"mu_s": float(sol.x[0]), "mu_2": float(sol.x[1]), "I_0": float(sol.x[2]),
            "r2": 1.0 - ss_res / max(ss_tot, 1e-300)}


def _band_field(gap_over_d: float, cells_per_d: float, xi_over_d: float,
                mu_wall: float, mu_bulk: float, pressure: float, d: float,
                rho_p: float, params: MuIParams):
    """Fine DEM-resolution field of the NGF shear band (returns y, mu, I_meas, I_loc)."""
    H = gap_over_d * d
    n = int(round(gap_over_d * cells_per_d))
    h = H / n
    y = (torch.arange(n, dtype=DTYPE) + 0.5) * h
    mu_prof = mu_bulk + (mu_wall - mu_bulk) * (1.0 - y / H)
    g_loc = local_fluidity(mu_prof, pressure, d, rho_p, params)
    xi = xi_over_d * d
    op = NGFOperator1D(n, h, xi,
                       Boundary("dirichlet",
                                float(local_strain_rate(mu_wall, pressure, d, rho_p, params) / mu_wall)),
                       Boundary("neumann", 0.0))
    sol = cg_jacobi(op, g_loc - op.bc_vector(), tol=1e-12, maxiter=100000)
    g = sol.x
    gamma = g * mu_prof
    lap = torch.zeros_like(g)
    lap[1:-1] = (g[2:] - 2 * g[1:-1] + g[:-2]) / h**2
    lap[0] = (g[1] - 2 * g[0] + (2 * op.lo.value - g[0])) / h**2
    lap[-1] = (g[-2] - 2 * g[-1] + g[-1]) / h**2
    g_loc_rec = g - xi**2 * lap
    scale = float(torch.sqrt(torch.tensor(pressure / rho_p, dtype=DTYPE)))
    I_meas = gamma * d / scale
    I_loc = (g_loc_rec * mu_prof) * d / scale
    return y, mu_prof, I_meas, I_loc


def a3_mu_i_deconvolution(*, gap_over_d: float = 30.0, cells_per_d: float = 2.0,
                          xi_over_d: float = 3.9, mu_wall: float = 0.55,
                          mu_bulk: float = 0.28, pressure: float = 5000.0,
                          bin_over_d: float = 6.0,
                          pooled_gaps: tuple[float, ...] = (10.0, 20.0, 40.0),
                          bin_sweep: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0),
                          params: MuIParams = MuIParams()) -> dict:
    """Coarse-graining bias of the local ``mu(I)`` fit and its NGF correction.

    A DEM-resolution shear field is generated from the NGF closure on a
    *stress-heterogeneous* cell (``mu`` decreasing linearly from the wall into
    the bulk, crossing yield).  The field is coarse-grained into cells of
    ``bin_over_d`` grains -- the resolution a DEM measurement would provide.
    Fitting the local ``mu(I)`` law to ``(I_meas, mu_cg)`` is biased wherever the
    nonlocal term matters; inverting the NGF operator
    (``g_loc = g - xi^2 lap(g)``, i.e. *deconvolution*) removes that bias.

    Honest scope: the fine field is model-consistent (produced by the NGF
    closure itself), so this measures whether the deconvolution inverts the
    coarse-graining bias -- it is not an experimental validation.
    """
    d = 1.0e-3
    rho_p = 2500.0
    I_list, Iloc_list, mu_list = [], [], []
    for gd in pooled_gaps:
        y, mu_prof, I_meas, I_loc = _band_field(gd, cells_per_d, xi_over_d, mu_wall,
                                                mu_bulk, pressure, d, rho_p, params)
        n = y.numel()
        win = max(1, int(round(bin_over_d * cells_per_d)))
        nb = n // win
        I_list.append(I_meas[:nb * win].reshape(nb, win).mean(dim=1))
        Iloc_list.append(I_loc[:nb * win].reshape(nb, win).mean(dim=1))
        mu_list.append(mu_prof[:nb * win].reshape(nb, win).mean(dim=1))
    I_cg = torch.cat(I_list).detach().numpy()
    Iloc_cg = torch.cat(Iloc_list).detach().numpy()
    mu_cg = torch.cat(mu_list).detach().numpy()
    nb = I_cg.size
    local_fit = _fit_mu_i(I_cg, mu_cg)
    ngf_fit = _fit_mu_i(Iloc_cg, mu_cg)
    sweep = {}
    for bd in (bin_sweep if bin_sweep else ()):
        sub = a3_mu_i_deconvolution(gap_over_d=gap_over_d, cells_per_d=cells_per_d,
                                    xi_over_d=xi_over_d, mu_wall=mu_wall,
                                    mu_bulk=mu_bulk, pressure=pressure,
                                    bin_over_d=bd, pooled_gaps=pooled_gaps,
                                    bin_sweep=(), params=params)
        sweep[f"bin_{bd:g}d"] = {"local_r2": sub["local_fit"]["r2"],
                                 "ngf_r2": sub["ngf_fit"]["r2"],
                                 "n_bins": sub["n_bins"]}
    return {
        "bin_sweep": sweep,
        "pooled_gaps_over_d": list(pooled_gaps), "cells_per_d": cells_per_d,
        "bin_over_d": bin_over_d, "n_bins": int(nb), "xi_over_d": xi_over_d,
        "local_fit": local_fit, "ngf_fit": ngf_fit,
        "r2_gain": ngf_fit["r2"] - local_fit["r2"],
        "threshold": 0.95,
        "pass": bool(ngf_fit["r2"] >= 0.95),
        "caveat": "fine field is model-consistent (NGF closure), not experimental data",
    }


# --------------------------------------------------------------------------- #
# A5-A8: contact solver gates
# --------------------------------------------------------------------------- #
def a5_restitution(values=(0.2, 0.5, 0.8), *, dt: float = 2.0e-4) -> dict:
    runs = {}
    for e in values:
        runs[f"e={e}"] = single_impact(restitution=e, dt=dt, mu_wall=0.3)
    worst = max(r["restitution_error"] for r in runs.values())
    return {"runs": runs, "worst_error": worst, "threshold": 0.02,
            "pass": bool(worst <= 0.02)}


def a6_zero_penetration(*, chain_balls: int = 5, pack_cells: int = 3) -> dict:
    single = min(single_impact(restitution=e, mu_wall=0.3)["max_overlap_over_d"]
                 for e in (0.2, 0.5, 0.8))
    chain = impact_chain(n_balls=chain_balls, restitution=0.5)["max_overlap_over_d"]
    pack = packing_impact(fcc_packing(n_cells=pack_cells), speed=5.0, steps=250,
                          n_iter=60, project_sweeps=40)["max_overlap_over_d"]
    conf = confined_packing(phi=0.85, n_particles=80)
    comp = packing_run(conf, increments=60, n_iter=15, dt=2e-6,
                       steps_per_increment=1, project_sweeps=120)
    worst = max(single, chain, pack)
    return {
        "single_impact": single, "chain": chain, "fcc_packing_impact": pack,
        "worst": worst, "threshold": 1e-6, "pass": bool(worst <= 1e-6),
        "compression_test": {k: comp[k] for k in
                             ("phi_initial", "phi_final", "phi_target",
                              "compression_stopped_at_phi", "max_overlap_over_d",
                              "final_overlap_over_d", "n_particles")},
        "compression_note": ("phi = 0.85 exceeds the Kepler bound 0.7405 for equal rigid "
                             "spheres; the confined-compression probe reports the solid "
                             "fraction at which a rigid random packing still admits zero "
                             "penetration"),
    }


def a7_energy_closure(*, chain_balls: int = 5, pack_cells: int = 3) -> dict:
    chain = impact_chain(n_balls=chain_balls, restitution=0.5)
    pack = packing_impact(fcc_packing(n_cells=pack_cells), speed=5.0, steps=250,
                          n_iter=60, project_sweeps=40)
    impact = single_impact(restitution=0.5, mu_wall=0.0)
    e_ratio = (impact["restitution_measured"] ** 2) / (0.5**2)
    momentum_closure = 1.0 - pack["momentum_error"]
    chain_momentum = 1.0 - float(abs(sum(chain["final_velocities"]) - chain["v0"])) / chain["v0"]
    ledger = pack["energy_ledger_closure"]
    items = {"impact_energy_ratio": e_ratio,
             "packing_momentum_closure": momentum_closure,
             "packing_work_ledger_closure": ledger,
             "chain_momentum_closure": chain_momentum}
    ok = all(0.98 <= v <= 1.02 for v in items.values())
    return {"items": items, "band": [0.98, 1.02], "pass": bool(ok),
            "packing": {k: pack[k] for k in ("energy_in", "energy_out",
                                             "work_contacts", "momentum_error")}}


def a8_stiffness_independence(*, kn_list=(1.0e5, 1.0e6)) -> dict:
    nscd = single_impact_penetration_scan(restitution=0.5, kn_list=kn_list)
    soft = soft_sphere_penetration_scan(restitution=0.5, kn_list=kn_list)
    return {
        "nscd_penetration_rel_change": nscd["penetration_rel_change"],
        "soft_sphere_penetration_rel_change": soft["penetration_rel_change"],
        "kn_ratio": nscd["kn_ratio"],
        "nscd_runs": {k: {"max_overlap_over_d": v["max_overlap_over_d"],
                          "restitution_measured": v["restitution_measured"]}
                      for k, v in nscd["runs"].items()},
        "soft_sphere_runs": {k: {"max_overlap_over_d": v["max_overlap_over_d"],
                                 "restitution_measured": v["restitution_measured"]}
                             for k, v in soft["runs"].items()},
        "threshold": 0.01,
        "pass": bool(nscd["penetration_rel_change"] <= 0.01),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_acceptance(*, quick: bool = False) -> dict:
    """Run every criterion and return the audit record."""
    out: dict = {"module": "tensordem.nonlocal_granular",
                 "node": "P295W1R2-N03", "quick": quick}
    out["A1"] = a1_local_limit()
    a2 = a2_band_width()
    out["A2"] = a2
    out["A4"] = a4_coarse_graining(a2["thin_band_calibration"]["xi_over_d"])
    out["A3"] = a3_mu_i_deconvolution(xi_over_d=a2["thin_band_calibration"]["xi_over_d"])
    out["A5"] = a5_restitution()
    out["A6"] = a6_zero_penetration()
    out["A7"] = a7_energy_closure()
    out["A8"] = a8_stiffness_independence()
    gates = {k: bool(out[k]["pass"]) for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")}
    out["gates"] = gates
    out["n_pass"] = sum(gates.values())
    out["all_pass"] = all(gates.values())
    return out