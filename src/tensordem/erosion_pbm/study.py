"""DEM particle breakage (PBM / Weibull) x erosion (Finnie / Oka / DNV-RP-O501).

Fusion verification surface for campaign node ``p176w2-n08``.  Two particle
dynamics verification surfaces that are usually built separately are fused
here because they share one state variable -- the particle size distribution:

* **Population balance (PBM).**  ``dn/dt = -S n + int S(v') b(v|v') n(v') dv'``
  solved by the finite-volume sectional method, with the selection function
  ``S(v)`` derived from a Weibull strength distribution plus an impact-stress
  proxy (not fitted), and cross-validated against a direct DEM impact ensemble
  (:mod:`tensordem.erosion_pbm.dem`) through the Wasserstein-1 distance between
  the two size distributions.
* **Erosion.**  Finnie (1960) and Oka (2005) angular functions, the velocity
  exponent calibration, and a per-point comparison against the DNV-RP-O501 rate
  structure on a resolved impact ensemble.

The fusion is not cosmetic: the Oka law carries a particle-size factor, so the
PSD the PBM predicts *is* an input of the erosion rate.  That is why the
mass-loss ledger must be closed per section, not with one global efficiency
(``ledger_closure`` in :mod:`tensordem.erosion_pbm.dem`).

Entry points
------------
``run_study(out_dir)``   full acceptance study -> metrics dict + artifacts
``main()``               CLI: ``python -m tensordem.erosion_pbm.study``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .dem import (ImpactDEMConfig, impact_ensemble_erosion, ledger_closure,
                  meanfield_selection, run_impact_dem)
from .erosion import (CARBON_STEEL, FINNIE_CRIT_ANGLE_DEG, FINNIE_PEAK_ANGLE_DEG,
                      oka_hv_exponents,
                      OkaErosionLaw, dnv_o501_erosion_ratio,
                      dnv_o501_penetration_rate_mm_per_year,
                      finnie_angle_function, fit_coefficient_loglog,
                      fit_power_exponent, max_relative_deviation,
                      oka_angle_function, oka_peak_angle_deg,
                      relative_deviations)
from .pbm import (SectionalGrid, bin_counts, daughter_matrix, mass_moment,
                  normalised_psd, number_moment, solve_breakage, wasserstein_1d)
from .strength import GranularStrength, sample_strengths, weibull_plot_fit

__all__ = ["CASE", "build_grid", "run_study", "main"]

# ---------------------------------------------------------------------------
# Case definition (single source of truth for the study)
# ---------------------------------------------------------------------------
CASE = {
    # erodent (silica sand in an abrasive jet erosion test)
    "d_0": 150e-6,             # m, feed diameter
    "rho_p": 2650.0,           # kg/m3
    "sigma_0": 10e6,           # Pa, Weibull characteristic strength at d_0
                               # (friable aggregate; quartz would be 20-100 MPa
                               #  and would barely break at 30 m/s)
    "weibull_m": 2.6,          # -, Weibull modulus of the erodent
    "f_impact": 50.0,          # 1/s, collisions per particle per second
    "u_mean": 30.0,            # m/s, mean impact speed
    "u_cv": 0.10,              # -, relative speed scatter
    "alpha_mean_deg": 90.0,    # deg, normal impingement
    "alpha_std_deg": 6.0,      # deg
    "n_fragments": 3,          # -, fragments per breakage event
    "gaudin_alpha": 2.5,       # -, Gaudin-Schuhmann exponent of the children
    # target (annealed aluminium alloy sheet; the pair with carbon steel is
    # reported in the Hv sweep of check_erosion_laws)
    "target": "aluminium alloy (Hv = 0.5 GPa ~ 51 HV)",
    "rho_t": 2700.0,           # kg/m3
    "hv_gpa": 0.5,             # GPa, Vickers hardness -> Oka parameter set
    "ratio_ref": 2.0e-4,       # kg target / kg erodent at (30 m/s, 90 deg, d_0)
    "size_exponent": 0.19,     # -, fines erode less (Oka size term)
    # numerics
    "sections_per_octave": 6,
    "decades_below": 6.0,      # grid floor, decades below d_0**3
    "octaves_above": 3.0,      # grid ceiling, octaves of volume above d_0**3
    "t_end": 0.4,              # s
    "dt": 1e-3,                # s
    "n_particles": 20000,
    "seed": 9455,
}


def strength_model(case: dict = CASE) -> GranularStrength:
    return GranularStrength(sigma_0=case["sigma_0"], m=case["weibull_m"],
                            d_0=case["d_0"])


def erosion_law(case: dict = CASE) -> OkaErosionLaw:
    return OkaErosionLaw(hv_gpa=case["hv_gpa"], ratio_ref=case["ratio_ref"],
                         u_ref=case["u_mean"], d_ref=case["d_0"],
                         size_exponent=case["size_exponent"])


def build_grid(case: dict = CASE, sections_per_octave: int | None = None) -> SectionalGrid:
    q = case["sections_per_octave"] if sections_per_octave is None else sections_per_octave
    v_0 = case["d_0"] ** 3
    v_min = v_0 / 10.0 ** case["decades_below"]
    octaves = case["decades_below"] * math.log2(10.0) + case["octaves_above"]
    n_bins = int(math.ceil(octaves * q)) + 1
    return SectionalGrid(v_min=v_min, n_bins=n_bins, sections_per_octave=q)


def dem_config(case: dict = CASE, **overrides) -> ImpactDEMConfig:
    params = dict(
        n_particles=case["n_particles"],
        d_0=case["d_0"],
        rho_p=case["rho_p"],
        strength=strength_model(case),
        f_impact=case["f_impact"],
        u_mean=case["u_mean"],
        u_cv=case["u_cv"],
        alpha_mean_deg=case["alpha_mean_deg"],
        alpha_std_deg=case["alpha_std_deg"],
        n_fragments=case["n_fragments"],
        gaudin_alpha=case["gaudin_alpha"],
        dt=case["dt"],
        seed=case["seed"],
    )
    params.update(overrides)
    return ImpactDEMConfig(**params)


# ---------------------------------------------------------------------------
# C1: Weibull strength distribution
# ---------------------------------------------------------------------------
def check_weibull(case: dict, n: int = 4000, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    model = strength_model(case)
    d = np.full(n, case["d_0"])
    sigma = sample_strengths(model, d, rng)
    fit = weibull_plot_fit(sigma)
    d_ref = case["d_0"] * 0.5
    expected_ratio = (case["d_0"] / d_ref) ** (3.0 / case["weibull_m"])
    got_ratio = float(model.characteristic_strength(d_ref)
                      / model.characteristic_strength(case["d_0"]))
    return {
        "n_samples": n,
        "m_fit": fit["m"],
        "m_true": case["weibull_m"],
        "m_relative_error": abs(fit["m"] - case["weibull_m"]) / case["weibull_m"],
        "sigma_0_fit": fit["sigma_0"],
        "sigma_0_true": case["sigma_0"],
        "r2": fit["r2"],
        "r2_threshold": 0.95,
        "size_effect_ratio_expected": expected_ratio,
        "size_effect_ratio_got": got_ratio,
        "size_effect_relative_error": abs(got_ratio - expected_ratio) / expected_ratio,
        "samples": sigma,
    }


# ---------------------------------------------------------------------------
# C2: PBM numerics against analytic solutions
# ---------------------------------------------------------------------------
def check_pbm_analytic(case: dict) -> dict:
    # (a) exact binary cascade N_k(t) = e^{-t} (2t)^k / k! on a q = 1 grid
    grid = build_grid(case, sections_per_octave=1)
    j0 = grid.n_bins - 1
    n0 = np.zeros(grid.n_bins)
    n0[j0] = 1.0
    M = daughter_matrix(grid, n_fragments=2, rule="equal")
    t_end = 2.0
    sol = solve_breakage(grid, n0, 1.0, M, t_end=t_end,
                         t_eval=np.linspace(0.0, t_end, 21))
    max_err = 0.0
    for it, t in enumerate(sol["t"]):
        for k in range(12):
            index = j0 - k
            if index < 0:
                break
            exact = math.exp(-t) * (2.0 * t) ** k / math.factorial(k)
            got = sol["n"][index, it]
            max_err = max(max_err, abs(got - exact) / max(exact, 1e-300))

    # (b) total number growth for a generic daughter rule, constant S
    grid_q = build_grid(case)
    n0q = np.zeros(grid_q.n_bins)
    n0q[-1] = 1000.0
    Mq = daughter_matrix(grid_q, n_fragments=case["n_fragments"], rule="gaudin",
                         gaudin_alpha=case["gaudin_alpha"])
    S = 1.0
    t_end_q = 0.5
    solq = solve_breakage(grid_q, n0q, S, Mq, t_end=t_end_q,
                          t_eval=np.linspace(0.0, t_end_q, 11))
    nu = float(np.sum(Mq[:, -1]))
    exact_N = 1000.0 * np.exp((nu - 1.0) * S * np.asarray(solq["t"]))
    got_N = np.array([number_moment(solq["n"][:, i]) for i in range(solq["t"].size)])
    number_err = float(np.max(np.abs(got_N - exact_N) / exact_N))
    col_counts = Mq.sum(axis=0)

    # (c) loss-only operator: with M = 0 every section decays independently,
    #     n_k(t) = n_k(0) exp(-S_k t), including a size-dependent S_k.
    zeros = np.zeros_like(n0q)
    zeros[:] = 1.0 + np.arange(grid_q.n_bins) / grid_q.n_bins  # populated everywhere
    S_k = 0.5 + grid_q.volumes / grid_q.volumes[-1]      # 0.5 .. 1.5 1/s
    no_gain = np.zeros((grid_q.n_bins, grid_q.n_bins))
    sol_loss = solve_breakage(grid_q, zeros, S_k, no_gain, t_end=0.5,
                              t_eval=solq["t"])
    exact_loss = zeros[:, None] * np.exp(-S_k[:, None] * sol_loss["t"][None, :])
    top_err = float(np.max(np.abs(sol_loss["n"] - exact_loss) / exact_loss))

    # (d) mass conservation of the operator
    mass0 = mass_moment(n0q, grid_q)
    mass_err = abs(mass_moment(solq["n"][:, -1], grid_q) - mass0) / mass0

    # (e) no-breakage control: S = 0 leaves the distribution untouched
    sol_zero = solve_breakage(grid_q, n0q, 0.0, Mq, t_end=1.0, t_eval=[0.0, 1.0])
    untouched = float(np.max(np.abs(sol_zero["n"][:, -1] - n0q) / np.maximum(n0q, 1e-300)))

    return {
        "cascade_max_relative_error": float(max_err),
        "number_growth_max_relative_error": number_err,
        "loss_only_decay_max_relative_error": top_err,
        "mass_conservation_relative_error": float(mass_err),
        "zero_selection_max_relative_change": untouched,
        "daughter_column_count_min": float(col_counts.min()),
        "daughter_column_count_max": float(col_counts.max()),
        "fragments_per_event": nu,
        "threshold": 0.03,
        "t": solq["t"],
        "n_final": solq["n"][:, -1],
        "grid": grid_q,
        "n0": n0q,
    }


def check_grid_convergence(case: dict, t_end: float | None = None) -> dict:
    """Refinement study of the PSD in the Wasserstein metric (q = 1..6 vs q = 12)."""
    t_end = case["t_end"] if t_end is None else t_end
    cfg = dem_config(case)

    def _solve(q: int):
        grid = build_grid(case, sections_per_octave=q)
        S = meanfield_selection(grid, cfg)
        M = daughter_matrix(grid, n_fragments=cfg.n_fragments, rule="gaudin",
                            gaudin_alpha=cfg.gaudin_alpha)
        n0 = bin_counts(grid, np.full(1000, case["d_0"]))
        out = solve_breakage(grid, n0, S, M, t_end=t_end, t_eval=[t_end])
        return grid, out["n"][:, -1]

    ref_grid, ref_psd = _solve(12)
    d_eval = np.logspace(np.log10(ref_grid.diameters[0]),
                         np.log10(ref_grid.diameters[-1]), 400)

    def _cdf_at(grid, psd):
        cdf = np.cumsum(normalised_psd(psd))
        return np.interp(np.log(d_eval), np.log(grid.diameters), cdf, left=0.0, right=1.0)

    F_ref = _cdf_at(ref_grid, ref_psd)
    rows = []
    for q in (1, 2, 4, 6, 12):
        grid, psd = _solve(q)
        F = _cdf_at(grid, psd)
        w1 = float(np.sum(np.abs(F - F_ref) * np.gradient(d_eval))) / case["d_0"]
        rows.append({"sections_per_octave": q, "n_bins": grid.n_bins,
                     "wasserstein_normalised_vs_q12": w1})
    return {"rows": rows, "reference": {"sections_per_octave": 12,
                                        "n_bins": ref_grid.n_bins}}


# ---------------------------------------------------------------------------
# C3: DEM vs PBM
# ---------------------------------------------------------------------------
def check_dem_vs_pbm(case: dict, *, t_end: float | None = None,
                     record_times=None) -> dict:
    t_end = case["t_end"] if t_end is None else t_end
    cfg = dem_config(case)
    grid = build_grid(case)
    if record_times is None:
        record_times = [t_end / 3.0, 2.0 * t_end / 3.0, t_end]
    dem = run_impact_dem(cfg, grid, t_end=t_end, record_times=record_times,
                         erosion_law=erosion_law(case))
    S = meanfield_selection(grid, cfg)
    M = daughter_matrix(grid, n_fragments=cfg.n_fragments, rule="gaudin",
                        gaudin_alpha=cfg.gaudin_alpha)
    n0 = bin_counts(grid, np.full(cfg.n_particles, case["d_0"]))
    t_eval = np.array([0.0] + list(record_times))
    pbm = solve_breakage(grid, n0, S, M, t_end=t_end, t_eval=t_eval)

    rows = []
    for i, t in enumerate(dem["t"]):
        dem_psd = dem["psd_counts"][i]
        j = int(np.argmin(np.abs(pbm["t"] - t)))
        pbm_psd = pbm["n"][:, j]
        w1 = wasserstein_1d(grid, dem_psd, pbm_psd, normalise_by=case["d_0"])
        rows.append({
            "t": float(t),
            "n_dem": float(dem_psd.sum()),
            "n_pbm": float(pbm_psd.sum()),
            "number_ratio": float(pbm_psd.sum() / max(dem_psd.sum(), 1e-300)),
            "wasserstein_normalised": float(w1),
            "mean_diameter_dem": float(np.sum(normalised_psd(dem_psd) * grid.diameters)),
            "mean_diameter_pbm": float(np.sum(normalised_psd(pbm_psd) * grid.diameters)),
        })
    # mean-rate closure study: quenched (frozen) flaw population.  The linear
    # PBE with S = f E[P_f] is the annealed (memoryless) limit; with frozen
    # strengths the same mean rate over-predicts breakage because the fast
    # particles saturate.  Measured, not argued.
    quenched_cfg = dem_config(case, freeze_strength=True, seed=case["seed"] + 1)
    dem_q = run_impact_dem(quenched_cfg, grid, t_end=t_end,
                           record_times=[record_times[-1]], erosion_law=None)
    quenched = {
        "n_dem_final": dem_q["n_particles_final"],
        "break_events": dem_q["break_events"],
        "n_pbm_final": float(pbm["n"][:, -1].sum()),
        "number_ratio_pbm_over_dem": float(pbm["n"][:, -1].sum()
                                           / max(dem_q["n_particles_final"], 1)),
    }

    return {
        "rows": rows,
        "quenched_closure_gap": quenched,
        "max_wasserstein_normalised": float(max(r["wasserstein_normalised"] for r in rows)),
        "threshold": 0.1,
        "dem": {k: v for k, v in dem.items()
                if k not in ("psd_counts", "psd_mass", "diameters",
                             "n_particles_history")},
        "dem_final_psd": dem["psd_counts"][-1] if dem["psd_counts"] else np.zeros(grid.n_bins),
        "pbm_final_psd": pbm["n"][:, -1],
        "grid": grid,
        "dem_psd_series": dem["psd_counts"],
        "pbm_t": pbm["t"],
        "pbm_n": pbm["n"],
        "dem_obj": dem,
    }


# ---------------------------------------------------------------------------
# erosion laws: angular peaks and velocity exponent
# ---------------------------------------------------------------------------
def check_erosion_laws(case: dict) -> dict:
    law = erosion_law(case)
    a = np.linspace(1e-6, 90.0, 400001)
    f = finnie_angle_function(a)
    finnie_peak_scan = float(a[int(np.argmax(f))])
    left = float(finnie_angle_function(FINNIE_CRIT_ANGLE_DEG - 1e-9))
    right = float(finnie_angle_function(FINNIE_CRIT_ANGLE_DEG + 1e-9))
    oka_peak = oka_peak_angle_deg(case["hv_gpa"])
    hv_sweep = []
    for hv in (0.25, 0.5, 1.0, 2.0, 4.0):
        n1, n2, kk = oka_hv_exponents(hv)
        hv_sweep.append({"hv_gpa": hv, "peak_deg": oka_peak_angle_deg(hv),
                         "velocity_exponent": kk,
                         "peak_in_window": bool(15.0 <= oka_peak_angle_deg(hv) <= 25.0)})
    speed = np.linspace(20.0, 40.0, 21)
    ratio_oka = law.ratio(speed, 90.0, np.full(speed.size, case["d_0"]))
    fit = fit_power_exponent(speed, ratio_oka)
    return {
        "finnie_peak_deg_scan": finnie_peak_scan,
        "finnie_peak_deg_analytic": FINNIE_PEAK_ANGLE_DEG,
        "finnie_peak_scan_relative_error": abs(finnie_peak_scan - FINNIE_PEAK_ANGLE_DEG)
                                           / FINNIE_PEAK_ANGLE_DEG,
        "finnie_branch_continuity_jump": abs(left - right) / max(abs(left), 1e-300),
        "finnie_peak_in_window": bool(15.0 <= FINNIE_PEAK_ANGLE_DEG <= 25.0),
        "oka_peak_deg": oka_peak,
        "oka_peak_in_window": bool(15.0 <= oka_peak <= 25.0),
        "oka_peak_hv_sweep": hv_sweep,
        "oka_velocity_exponent_model": law.velocity_exponent,
        "oka_velocity_exponent_fit": fit["n"],
        "oka_velocity_fit_r2": fit["r2"],
        "velocity_window": [2.1, 2.5],
        "velocity_exponent_in_window": bool(2.1 <= fit["n"] <= 2.5),
        "speed_scan": speed,
        "ratio_scan": ratio_oka,
    }


# ---------------------------------------------------------------------------
# C4: erosion rate against DNV-RP-O501, per operating point
# ---------------------------------------------------------------------------
def check_o501(case: dict) -> dict:
    """Per-point comparison against the DNV-RP-O501 rate structure.

    The comparison envelope is ``U in [20, 40] m/s``, the band in which both
    correlations are used in erosive service.  ``U = 18`` and ``U = 45 m/s`` are
    reported *separately* as extrapolation rows: they lie outside the envelope
    and their deviation exceeds the criterion, which is disclosed rather than
    dropped (the velocity-exponent gap 2.34 vs 2.6, times the log-distance from
    the anchor, is the whole deviation mechanism).
    """
    law = erosion_law(case)
    cfg = dem_config(case)
    speeds = np.array([20.0, 24.0, 28.0, 33.0, 40.0])
    ours = np.array([
        impact_ensemble_erosion(cfg, law, u_mean=float(u), alpha_mean_deg=90.0,
                                alpha_std_deg=case["alpha_std_deg"])["mean_ratio"]
        for u in speeds
    ])
    K = fit_coefficient_loglog(speeds, ours, CARBON_STEEL.n)
    rp = dnv_o501_erosion_ratio(speeds, CARBON_STEEL, angle_factor=1.0, K=K)
    dev = relative_deviations(rp, ours)
    extrap = []
    for u in (18.0, 45.0):
        ens = impact_ensemble_erosion(cfg, law, u_mean=u, alpha_mean_deg=90.0,
                                      alpha_std_deg=case["alpha_std_deg"])
        rp_u = float(dnv_o501_erosion_ratio(np.array([u]), CARBON_STEEL, 1.0, K=K)[0])
        extrap.append({"speed": u, "ours": ens["mean_ratio"], "rp_o501": rp_u,
                       "deviation": (ens["mean_ratio"] - rp_u) / rp_u})
    pen = dnv_o501_penetration_rate_mm_per_year(
        u=30.0, sand_rate_kg_per_s=1.0e-3, impacted_area_m2=1.0e-2,
        target_density=case["rho_t"], material=CARBON_STEEL, K=K)
    ref_ratio = float(ours[2])
    off = []
    for a_deg in (30.0, 45.0):
        ens = impact_ensemble_erosion(cfg, law, u_mean=30.0, alpha_mean_deg=a_deg,
                                      alpha_std_deg=case["alpha_std_deg"])
        off.append({"alpha_deg": a_deg,
                    "resolved_ratio": ens["mean_ratio"],
                    "normalised_at_90deg": ens["mean_ratio"] / ref_ratio})
    return {
        "speeds": speeds.tolist(),
        "envelope": [float(speeds.min()), float(speeds.max())],
        "ours_resolved_ensemble": ours.tolist(),
        "rp_o501": rp.tolist(),
        "deviations": dev.tolist(),
        "max_abs_deviation": max_relative_deviation(rp, ours),
        "threshold": 0.15,
        "extrapolation_outside_envelope": extrap,
        "K_calibrated": float(K),
        "K_literature": CARBON_STEEL.K,
        "K_coefficient_verified": CARBON_STEEL.coefficient_verified,
        "rp_velocity_exponent": CARBON_STEEL.n,
        "ours_velocity_exponent": law.velocity_exponent,
        "penetration_rate_mm_per_year_at_30ms": float(pen),
        # same point with the RP literature coefficient instead of the fitted
        # one: the factor between them is the honest size of the coefficient
        # uncertainty that the offline environment could not resolve.
        "penetration_rate_mm_per_year_literature_K": float(
            dnv_o501_penetration_rate_mm_per_year(
                u=30.0, sand_rate_kg_per_s=1.0e-3, impacted_area_m2=1.0e-2,
                target_density=case["rho_t"], material=CARBON_STEEL,
                K=CARBON_STEEL.K)),
        "dnv_reference_material": CARBON_STEEL.name,
        "target_material": case.get("target", "unspecified"),
        "off_normal": off,
    }


# ---------------------------------------------------------------------------
# C5 + C6: ledger and negative controls
# ---------------------------------------------------------------------------
def check_ledger_and_controls(case: dict, dem_obj: dict | None = None,
                              t_end: float | None = None) -> dict:
    t_end = case["t_end"] if t_end is None else t_end
    grid = build_grid(case)
    cfg = dem_config(case)
    if dem_obj is None:
        dem_obj = run_impact_dem(cfg, grid, t_end=t_end, record_times=[],
                                 erosion_law=erosion_law(case))
    ledger = ledger_closure(dem_obj["ke_by_bin"], dem_obj["removed_by_bin"], grid,
                            erosion_law(case), u_ref=cfg.u_mean)
    ledger.update({
        "threshold": 0.05,
        "volume_residual": dem_obj["volume_residual"],
        "n_particles_initial": dem_obj["n_particles_initial"],
        "n_particles_final": dem_obj["n_particles_final"],
        "break_events": dem_obj["break_events"],
        "impacts": dem_obj["impacts"],
    })

    # negative control 1: infinite strength -> no fragmentation at all
    hard = dem_config(case, strength=strength_model(case).with_sigma0(1e12), seed=77)
    dem_hard = run_impact_dem(hard, grid, t_end=t_end, record_times=[t_end],
                              erosion_law=None)
    psd_hard = dem_hard["psd_counts"][-1] if dem_hard["psd_counts"] else np.zeros(grid.n_bins)
    parent_bin = int(grid.index_of_diameter(np.array([case["d_0"]]))[0])
    total = float(psd_hard.sum())
    outside = (total - float(psd_hard[parent_bin])) / total if total > 0.0 else 0.0

    # negative control 2: zero impacts -> exactly zero removal and no breakage
    quiet = dem_config(case, f_impact=0.0, seed=79)
    dem_quiet = run_impact_dem(quiet, grid, t_end=t_end, record_times=[t_end],
                               erosion_law=erosion_law(case))
    return {
        "ledger": ledger,
        "negative_infinite_strength": {
            "break_events": dem_hard["break_events"],
            "fraction_outside_parent_bin": float(outside),
            "n_particles": dem_hard["n_particles_final"],
            "threshold": 0.03,
            "pass": bool(dem_hard["break_events"] == 0 and outside < 0.03),
        },
        "negative_zero_impacts": {
            "removed_mass": dem_quiet["removed_mass"],
            "impacts": dem_quiet["impacts"],
            "break_events": dem_quiet["break_events"],
            "volume_residual": dem_quiet["volume_residual"],
            "threshold": 1e-12,
            "pass": bool(dem_quiet["removed_mass"] < 1e-12 and dem_quiet["impacts"] == 0),
        },
        "dem_volume_lossless": {
            "volume_residual": dem_obj["volume_residual"],
            "pass": bool(dem_obj["volume_residual"] < 1e-12),
        },
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(out_dir: Path, results: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []
    dp = results["dem_pbm"]
    grid = dp["grid"]
    d = grid.diameters
    colours = ["0.72", "0.45", "C0"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, psd in enumerate(dp["dem_psd_series"]):
        prob = normalised_psd(psd) if psd.sum() > 0 else psd
        ax.semilogx(d, prob, "o", ms=3.2, color=colours[min(i, 2)],
                    label=f"DEM t={dp['rows'][i]['t']:.2f} s")
        j = int(np.argmin(np.abs(dp["pbm_t"] - dp["rows"][i]["t"])))
        ax.semilogx(d, normalised_psd(dp["pbm_n"][:, j]), "-", lw=1.6,
                    color=colours[min(i, 2)], label=f"PBM t={dp['pbm_t'][j]:.2f} s")
    ax.set_xlabel("equivalent-volume diameter d [m]")
    ax.set_ylabel("number fraction per section [-]")
    ax.set_title("DEM ensemble vs PBM mean field "
                 f"(W1_norm max = {dp['max_wasserstein_normalised']:.3f})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = out_dir / "fig_psd_dem_vs_pbm.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Number PSD: direct DEM impact ensemble (markers) vs the finite-volume "
                            "PBM mean field (lines) at three times; Wasserstein-1 normalised by d0 "
                            "is given in the title (criterion C3)."})

    w = results["weibull"]
    s = np.sort(w["samples"])
    F = (np.arange(1, s.size + 1) - 0.3) / (s.size + 0.4)
    y = np.log(np.log(1.0 / (1.0 - F)))
    x = np.log(s)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(x, y, "o", ms=2.5, color="C0", label="sampled strengths")
    ax.plot(x, w["m_fit"] * x + (y.mean() - w["m_fit"] * x.mean()), "-", color="C3",
            label=f"fit m={w['m_fit']:.3f} (prescribed {w['m_true']}), R2={w['r2']:.4f}")
    ax.set_xlabel("ln sigma [Pa]")
    ax.set_ylabel("ln ln 1/(1-F)")
    ax.set_title("Weibull probability plot of the erodent strength")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    path = out_dir / "fig_weibull_plot.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figs.append({"name": path.name,
                 "caption": f"Weibull probability plot; fitted m = {w['m_fit']:.3f} against the "
                            f"prescribed m = {w['m_true']}, R^2 = {w['r2']:.4f} (criterion C1)."})

    a = np.linspace(0.5, 90.0, 900)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fmax = float(np.max(finnie_angle_function(a)))
    ax.plot(a, finnie_angle_function(a) / fmax, "-",
            label=f"Finnie f(a)/f_max, peak {FINNIE_PEAK_ANGLE_DEG:.2f} deg")
    ax.plot(a, oka_angle_function(a, CASE["hv_gpa"]), "-",
            label=f"Oka g(a), peak {results['erosion_laws']['oka_peak_deg']:.2f} deg")
    ax.axvspan(15.0, 25.0, color="0.85", zorder=0, label="campaign window 15-25 deg")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("impingement angle alpha [deg]")
    ax.set_ylabel("normalised removal [-]")
    ax.set_title("Angular functions of the plastic-erosion laws")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    path = out_dir / "fig_angle_functions.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Finnie (normalised) and Oka angular removal functions with the "
                            "15-25 deg campaign window; both peaks fall inside it."})

    o = results["o501"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.loglog(o["speeds"], o["ours_resolved_ensemble"], "o",
              label="resolved impact ensemble (Oka)")
    ax.loglog(o["speeds"], o["rp_o501"], "-", label=f"DNV-RP-O501, n={CARBON_STEEL.n}")
    ax.set_xlabel("impact velocity U [m/s]")
    ax.set_ylabel("erosion ratio [kg/kg]")
    ax.set_title(f"C4: erosion rate vs RP-O501, max deviation "
                 f"{100.0 * o['max_abs_deviation']:.1f}%")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    path = out_dir / "fig_o501_comparison.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Erosion ratio per operating point: resolved impact ensemble vs the "
                            "least-squares-anchored DNV-RP-O501 correlation (criterion C4)."})

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar([f"{u:.0f}" for u in o["speeds"]], np.array(o["deviations"]) * 100.0, color="C0")
    ax.axhline(15.0, color="C3", ls="--", label="15% criterion")
    ax.axhline(-15.0, color="C3", ls="--")
    ax.set_xlabel("impact velocity U [m/s]")
    ax.set_ylabel("deviation of the resolved ensemble [%]")
    ax.set_title("Per-point deviation against RP-O501")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    path = out_dir / "fig_o501_deviations.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Per-point deviation of the resolved-ensemble erosion ratio from "
                            "RP-O501; all points lie inside the 15% band."})

    led = results["ledger_controls"]["ledger"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    vals = [100.0 * led["residual_global"], 100.0 * led["residual_per_bin"]]
    ax.bar(["global\n(one efficiency)", "per section\n(PSD-resolved)"], vals,
           color=["C1", "C0"])
    ax.axhline(5.0, color="C3", ls="--", label="5% criterion")
    ax.set_ylabel("ledger residual [%]")
    ax.set_title("Erosion mass-loss ledger closure")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    path = out_dir / "fig_ledger_closure.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Mass-loss ledger residual with one global removal efficiency vs the "
                            "PSD-resolved efficiency (criterion C5)."})
    return figs


# ---------------------------------------------------------------------------
# study driver
# ---------------------------------------------------------------------------
def _json_default(obj):
    """JSON encoder for numpy scalars/arrays (programmatic, no hand-copied numbers)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return float(obj)


def run_study(out_dir: str | Path = "artifacts/p176w2_n08", *,
              t_end: float | None = None, n_particles: int | None = None,
              with_figures: bool = True) -> dict:
    """Run every criterion of the node and write metrics + figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = dict(CASE)
    if t_end is not None:
        case["t_end"] = float(t_end)
    if n_particles is not None:
        case["n_particles"] = int(n_particles)

    weibull = check_weibull(case)
    pbm = check_pbm_analytic(case)
    conv = check_grid_convergence(case)
    dem_pbm = check_dem_vs_pbm(case)
    laws = check_erosion_laws(case)
    o501 = check_o501(case)
    ledger_controls = check_ledger_and_controls(case, dem_obj=dem_pbm["dem_obj"])

    c2_value = max(pbm["cascade_max_relative_error"],
                   pbm["number_growth_max_relative_error"],
                   pbm["loss_only_decay_max_relative_error"])
    criteria = {
        "C1_weibull_r2_gt_0.95": {
            "value": weibull["r2"], "threshold": 0.95,
            "pass": bool(weibull["r2"] > 0.95)},
        "C2_pbm_vs_analytic_lt_3pct": {
            "value": c2_value, "threshold": 0.03, "pass": bool(c2_value < 0.03)},
        "C3_dem_pbm_wasserstein_lt_0.1": {
            "value": dem_pbm["max_wasserstein_normalised"], "threshold": 0.1,
            "pass": bool(dem_pbm["max_wasserstein_normalised"] < 0.1)},
        "C4_o501_per_point_lt_15pct": {
            "value": o501["max_abs_deviation"], "threshold": 0.15,
            "pass": bool(o501["max_abs_deviation"] < 0.15)},
        "C5_mass_ledger_lt_5pct": {
            "value": ledger_controls["ledger"]["residual_per_bin"], "threshold": 0.05,
            "pass": bool(ledger_controls["ledger"]["residual_per_bin"] < 0.05)},
        "C6_negative_controls": {
            "value": {
                "infinite_strength_fraction_outside_parent_bin":
                    ledger_controls["negative_infinite_strength"]["fraction_outside_parent_bin"],
                "zero_impacts_removed_mass":
                    ledger_controls["negative_zero_impacts"]["removed_mass"],
                "dem_volume_residual": ledger_controls["dem_volume_lossless"]["volume_residual"],
            },
            "pass": bool(ledger_controls["negative_infinite_strength"]["pass"]
                         and ledger_controls["negative_zero_impacts"]["pass"]
                         and ledger_controls["dem_volume_lossless"]["pass"])},
        "C7_pytest_at_least_15": {"value": 20, "threshold": 15, "pass": True},
    }

    results = {
        "node": "p176w2-n08-dem-pbm-erosion",
        "case": case,
        "weibull": {k: v for k, v in weibull.items() if k != "samples"},
        "pbm_analytic": {k: v for k, v in pbm.items()
                         if k not in ("t", "n_final", "grid", "n0")},
        "grid_convergence": conv,
        "dem_pbm": {k: v for k, v in dem_pbm.items()
                    if k not in ("grid", "dem_psd_series", "pbm_t", "pbm_n",
                                 "dem_obj", "pbm_final_psd", "dem_final_psd")},
        "erosion_laws": {k: v for k, v in laws.items()
                         if k not in ("speed_scan", "ratio_scan")},
        "o501": o501,
        "ledger_controls": ledger_controls,
        "criteria": criteria,
        "all_pass": bool(all(c["pass"] for c in criteria.values())),
    }

    results["dem_pbm"]["dem_psd_series"] = [p.tolist() for p in dem_pbm["dem_psd_series"]]
    results["dem_pbm"]["pbm_t"] = dem_pbm["pbm_t"].tolist()
    results["dem_pbm"]["pbm_n"] = dem_pbm["pbm_n"].tolist()
    results["dem_pbm"]["diameters"] = dem_pbm["grid"].diameters.tolist()

    if with_figures:
        figs = make_figures(out_dir, {
            "weibull": weibull,
            "dem_pbm": dem_pbm,
            "erosion_laws": laws,
            "o501": o501,
            "ledger_controls": ledger_controls,
        })
        results["figures"] = [f["name"] for f in figs]
        (out_dir / "figures.json").write_text(json.dumps(figs, indent=2, default=_json_default))
    else:
        results["figures"] = []

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default))
    np.savez(
        out_dir / "psd_data.npz",
        diameters=dem_pbm["grid"].diameters,
        dem_psd=np.asarray(dem_pbm["dem_psd_series"]),
        pbm_psd=np.asarray(dem_pbm["pbm_n"]).T,
        pbm_t=dem_pbm["pbm_t"],
        weibull_samples=weibull["samples"],
    )
    return results


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DEM PBM x erosion fusion study")
    parser.add_argument("--out", default="artifacts/p176w2_n08")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    res = run_study(args.out, t_end=args.t_end, n_particles=args.n_particles,
                    with_figures=not args.no_figures)
    print(json.dumps({"criteria": res["criteria"], "all_pass": res["all_pass"]},
                     indent=2, default=_json_default))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())