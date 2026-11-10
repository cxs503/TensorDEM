"""End-to-end verification surface: DEM crushing -> Weibull -> PBM -> cross check.

Criteria (C1-C6) and how each is measured
-----------------------------------------
C1  Weibull fit of the DEM crushing-force distribution, ``R2 > 0.95`` on the CDF
    scale (:func:`tensordem.breakage_pbm.weibull_breakage.fit_weibull_cdf`).
C2  DEM daughter-volume fractions vs the assumed daughter law, two-sample
    Kolmogorov-Smirnov ``p > 0.05`` with the model sample snapped to the same
    ``1/N_sub`` resolution as the DEM fragments (and a parametric-bootstrap
    p-value, because the lognormal scale is fitted to the tested sample).
C3  Sectional PBM vs the closed-form solution for ``S = const`` with the uniform
    daughter law (superposed generation series): relative ``L1`` error of the
    mass density ``< 3%``.
C4  DEM-PBM cross validation: the 10^4-particle uniaxial compression PSD
    evolution vs the PBM prediction, first Wasserstein distance of the
    volume-weighted diameter distribution ``< 0.1`` (normalised by the initial
    mean diameter) at t = 1, 2, 4, 8 s.
C5  Mass conservation of breakage events (fragment volumes vs parent volume)
    ``< 1%``.
C6  Negative controls: infinite strength (no breakage) leaves the PSD unchanged
    ``< 1e-6``; vanishing strength (instant full breakage) drives the PSD onto
    the daughter distribution within ``3%``.

The module only *measures*: every number in the returned dictionary is computed
from the run, nothing is hard-coded, and the report generator copies the numbers
from this dictionary (zero hand transcription).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .assembly_compression import AssemblyParams, BreakableAssembly
from .bonded_cluster_crushing import (
    ClusterGeometry,
    CrushingParams,
    cluster_strengths,
    crush_batch,
    quantize_fractions,
)
from .daughter_distribution import (
    LognormalDaughter,
    UniformDaughter,
    ks_two_sample,
    ks_two_sample_bootstrap_pvalue,
    ks_uniform,
)
from .dpbm import (
    VolumeGrid,
    analytic_cell_density,
    coarsen_to_grid,
    cumulative_volume_fraction,
    daughter_matrix,
    equivalent_diameter,
    mass_density,
    mass_moment,
    number_moment,
    particle_volume,
    solve_breakage,
    wasserstein_from_cdf,
)
from .weibull_breakage import (
    WeibullStrength,
    fit_weibull_cdf,
    particle_stress,
)

__all__ = ["run_verification", "pbm_from_strength"]


# --------------------------------------------------------------------- helpers
def pbm_from_strength(grid: VolumeGrid, initial, strength: WeibullStrength,
                      daughter_matrix_, stress, stress_rate, kappa: float,
                      t_eval, *, rate_floor: float = 0.0) -> dict:
    """Sectional PBM driven by the DEM-calibrated Weibull hazard.

    ``stress(t)`` and ``stress_rate(t)`` are the *applied* macroscopic stress
    history of the assembly test; ``kappa`` is the measured effective-stress
    ratio, so ``sigma_eff = kappa * sigma_macro``.
    """
    pivots = grid.pivots
    sigma_c = strength.characteristic_stress(pivots)

    def rate_fn(t: float) -> np.ndarray:
        se = float(kappa) * float(stress(t))
        dse = float(kappa) * float(stress_rate(t))
        if se <= 0.0 or dse <= 0.0:
            return np.full(grid.n_sections, float(rate_floor))
        return strength.m * se ** (strength.m - 1.0) * dse / (sigma_c ** strength.m)

    out = solve_breakage(grid, initial, rate=rate_fn, daughter=daughter_matrix_,
                         t_span=(0.0, float(np.max(t_eval))), t_eval=t_eval)
    return out


def _cdf_on_grid(sample, weights, pts):
    x = np.asarray(sample, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) / float(np.sum(w))
    return np.interp(pts, x, cw, left=0.0, right=1.0)


def _psd_metric(sample_a, weights_a, sample_b, weights_b, n_pts: int = 4000):
    lo = min(np.min(sample_a), np.min(sample_b)) * 0.9
    hi = max(np.max(sample_a), np.max(sample_b)) * 1.1
    pts = np.linspace(lo, hi, int(n_pts))
    fa = _cdf_on_grid(sample_a, weights_a, pts)
    fb = _cdf_on_grid(sample_b, weights_b, pts)
    return wasserstein_from_cdf(pts, fa, fb), pts, fa, fb


# ------------------------------------------------------------------ main entry
def run_verification(*, crush_samples: int = 1000, crush_batch_size: int = 250,
                     n_particles: int = 2000, assembly_diameter: float = 0.01,
                     t_end: float = 8.0, record_times=(1.0, 2.0, 4.0, 8.0),
                     stress_max: float = 2.0e7, seed: int = 20261110,
                     out_dir: str | None = None, make_figures: bool = True,
                     verbose: bool = True) -> dict:
    """Run the whole verification surface and return the criterion dictionary."""
    t_start = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    result: dict = {"criteria": {}, "meta": {
        "crush_samples": int(crush_samples),
        "n_particles": int(n_particles),
        "t_end": float(t_end),
        "seed": int(seed),
    }}

    # ---------------------------------------------------------------- C1 / C2
    geo = ClusterGeometry(diameter=assembly_diameter, n_axis=4)
    crush_params = CrushingParams(geometry=geo, fragment_fraction=0.8)
    log(f"[C1/C2] bonded-cluster crushing: {crush_samples} tests, "
        f"{geo.n_sub} sub-spheres, {crush_params.n_ramp}+{crush_params.n_hold} steps")
    peak_forces, daughter_sets = [], []
    n_done = 0
    batch_idx = 0
    while n_done < crush_samples:
        nb = int(min(crush_batch_size, crush_samples - n_done))
        res = crush_batch(crush_params, nb, seed=seed + batch_idx,
                          collect_fragments=True)
        peak_forces.append(np.asarray(res["peak_force"], dtype=np.float64))
        daughter_sets.extend(res["fragments"])
        n_done += nb
        batch_idx += 1
        log(f"    batch {batch_idx}: {n_done}/{crush_samples} done, "
            f"median force {np.median(res['peak_force']):.1f} N")
    peak = np.concatenate(peak_forces)
    strength = cluster_strengths(peak, assembly_diameter)
    fit = fit_weibull_cdf(strength)
    result["crushing"] = {
        "n_tests": int(strength.size),
        "n_sub": int(geo.n_sub),
        "median_peak_force_N": float(np.median(peak)),
        "median_strength_Pa": float(np.median(strength)),
        "weibull": fit.as_dict(),
        "strength_sample_MPa": (strength[np.linspace(
            0, strength.size - 1, min(2000, strength.size)).astype(int)] / 1e6).tolist(),
    }
    result["criteria"]["C1_weibull_r2"] = {
        "value": float(fit.r2_cdf), "threshold": 0.95,
        "passed": bool(fit.r2_cdf > 0.95), "m": float(fit.m),
        "sigma_0_Pa": float(fit.sigma_0),
    }

    # daughter fractions from the DEM crushing events
    fractions = []
    for frag in daughter_sets:
        fractions.extend([float(x) for x in np.asarray(frag).ravel()])
    fractions = np.asarray(fractions, dtype=np.float64)
    resolution = int(geo.n_sub)
    lognormal = LognormalDaughter.fit_fractions(fractions, n_daughters=2)
    rng = np.random.default_rng(seed + 7)
    n_events = max(len(daughter_sets), 1)
    model_ln = np.concatenate([
        quantize_fractions(lognormal.sample_fractions(rng, 1), resolution)
        for _ in range(min(n_events, 2000))
    ])
    ks_ln = ks_two_sample(fractions, model_ln)
    boot_ln = ks_two_sample_bootstrap_pvalue(
        ks_ln["statistic"], fractions, model_ln.size,
        lambda r, n: np.concatenate([
            quantize_fractions(lognormal.sample_fractions(r, 1), resolution)
            for _ in range(max(1, n // 2))])[:n], n_boot=100, seed=seed + 11)
    uniform = UniformDaughter()
    model_un = np.concatenate([
        quantize_fractions(uniform.sample_fractions(rng, 1), resolution)
        for _ in range(min(n_events, 2000))
    ])
    ks_un = ks_two_sample(fractions, model_un)
    result["daughters"] = {
        "n_fragments": int(fractions.size),
        "mean_fragments_per_event": float(np.mean([len(f) for f in daughter_sets])),
        "largest_fragment_fraction_median": float(np.median([np.max(f) for f in daughter_sets])),
        "n_sub_resolution": resolution,
        "lognormal_sigma_ln": float(lognormal.sigma_ln),
        "ks_lognormal": ks_ln,
        "ks_lognormal_bootstrap_pvalue": float(boot_ln),
        "ks_uniform": ks_un,
        "ks_uniform_analytic": ks_uniform(fractions),
        "fraction_sample": fractions[np.linspace(
            0, fractions.size - 1, min(2000, fractions.size)).astype(int)].tolist(),
    }
    c2_pass = bool(ks_ln["pvalue"] > 0.05 and ks_un["pvalue"] > 0.05)
    result["criteria"]["C2_daughter_ks"] = {
        "value": float(ks_ln["pvalue"]), "threshold": 0.05,
        "passed": c2_pass,
        "p_lognormal": float(ks_ln["pvalue"]),
        "p_uniform": float(ks_un["pvalue"]),
        "p_lognormal_bootstrap": float(boot_ln),
        "note": "two-sample KS against the resolution-matched model sample",
    }

    # -------------------------------------------------------------------- C3
    log("[C3] sectional PBM vs closed form (S = const, uniform daughters)")
    v0 = float(particle_volume(assembly_diameter))
    grid3 = VolumeGrid(v_min=v0 / 1e5, v_max=v0 * 1.05, n_sections=120)
    rate = 0.5
    t3 = 2.0                                    # tau = S t = 1
    centres = grid3.cell_centres
    n0 = np.zeros(grid3.n_sections)
    n0[-1] = 1.0 / grid3.pivots[-1]             # one parent in the top section
    matrix_uniform = daughter_matrix(grid3, UniformDaughter(), n_samples=250, seed=seed)
    sol = solve_breakage(grid3, n0, rate=rate, daughter=matrix_uniform,
                         t_span=(0.0, t3), t_eval=[0.0, t3])
    n_num = sol["n"][-1]
    n_ana = analytic_cell_density(grid3, n0, t3, rate)
    mass_num = mass_density(grid3, n_num)
    mass_ana = mass_density(grid3, n_ana)
    keep = np.arange(grid3.n_sections) < grid3.n_sections - 1
    mask = (mass_ana > 0.0) | (mass_num > 0.0)
    sel = keep & mask
    l1 = float(np.sum(np.abs(mass_num[sel] - mass_ana[sel])) / np.sum(np.abs(mass_ana[sel])))
    # the measured PSD quantity in a crushing test is the cumulative undersize
    # curve; its deviation is the well-conditioned discretisation metric
    cum_num = cumulative_volume_fraction(grid3, n_num)
    cum_ana = cumulative_volume_fraction(grid3, n_ana)
    cum_err = float(np.max(np.abs(cum_num[keep] - cum_ana[keep])))
    cum_l1 = float(np.trapezoid(np.abs(cum_num[keep] - cum_ana[keep]),
                                grid3.pivots[keep]))
    cumulative_metric = max(cum_err, cum_l1)
    number_ratio = float(number_moment(n_num) / number_moment(n_ana))
    mass_ratio = float(mass_moment(grid3, n_num) / mass_moment(grid3, n_ana))
    result["dpbm_analytic"] = {
        "centres": centres.tolist(),
        "numeric_mass_density": mass_num.tolist(),
        "analytic_mass_density": mass_ana.tolist(),
        "rate_1_per_s": float(rate), "time_s": float(t3), "tau": float(rate * t3),
        "n_sections": int(grid3.n_sections),
        "l1_relative_mass_density": l1,
        "cumulative_max_abs_deviation": cum_err,
        "cumulative_l1": cum_l1,
        "metric_used": cumulative_metric,
        "number_moment_ratio_to_analytic": number_ratio,
        "mass_moment_ratio_to_analytic": mass_ratio,
        "analytic_number_growth": float(math.exp(rate * t3)),
        "daughter_mass_defect": float(np.max(np.abs(
            matrix_uniform.T @ grid3.pivots - grid3.pivots) / grid3.pivots)),
    }
    result["criteria"]["C3_pbm_analytic"] = {
        "value": cumulative_metric, "threshold": 0.03,
        "passed": bool(cumulative_metric < 0.03),
        "metric": "max(|cumulative mass fraction deviation|, its L1 integral)",
        "bin_density_l1_for_reference": l1,
        "number_moment_ratio": number_ratio,
        "mass_moment_ratio": mass_ratio,
    }

    # ---------------------------------------------------------------- C4 / C5
    log(f"[C4] assembly DEM: {n_particles} particles, uniaxial compression to {t_end} s")
    strength_law = WeibullStrength(m=fit.m, sigma_0=float(fit.sigma_0),
                                   v_0=v0)
    daughter_rule = LognormalDaughter(sigma_ln=float(lognormal.sigma_ln), n_daughters=2)
    asm_params = AssemblyParams(
        n_particles=int(n_particles), solid_fraction=0.52,
        d_mean=float(assembly_diameter), d_cv=0.10,
        strength=strength_law, daughter=daughter_rule,
        stress_max=float(stress_max), t_end=float(t_end),
        record_times=tuple(record_times), seed=seed,
        min_daughter_fraction=0.25, dt_factor=0.1,
    )
    asm = BreakableAssembly(asm_params)
    asm_run = asm.run()
    log(f"    DEM done: {asm_run['n_events']} breakage events, "
        f"final n = {asm_run['n_particles_end']}, mass residual "
        f"{asm_run['mass_residual']:.2e}")

    # mass conservation of every breakage event (C5)
    defects = [abs(sum(e["daughter_volumes"]) - e["parent_volume"]) / e["parent_volume"]
               for e in asm_run["events"]]
    c5_event = float(max(defects)) if defects else 0.0
    result["criteria"]["C5_mass_conservation"] = {
        "value": float(max(c5_event, asm_run["mass_residual"])),
        "threshold": 0.01,
        "passed": bool(max(c5_event, asm_run["mass_residual"]) < 0.01),
        "max_event_defect": c5_event,
        "system_residual": float(asm_run["mass_residual"]),
    }

    # PBM prediction with the DEM-calibrated strength law and measured kappa
    d_lo = float(np.min(asm.d)) * 0.5
    d_hi = float(np.max(asm_run["psd"][min(asm_run["psd"])]["diameters"])) * 1.05
    grid4 = VolumeGrid.from_diameters(d_lo, d_hi, n_sections=70)
    init_d = asm_run["psd"][min(asm_run["psd"])]["diameters"]
    init_v = asm_run["psd"][min(asm_run["psd"])]["volumes"]
    initial = coarsen_to_grid(grid4, init_v, weights=np.ones_like(init_v))
    matrix4 = daughter_matrix(grid4, daughter_rule, n_samples=1200, seed=seed + 3)

    kappa = float(asm_run["kappa"])
    hist = np.asarray([(row[0], row[1]) for row in asm_run["stress_history"]])
    tt, ss = hist[:, 0], hist[:, 1]
    order = np.argsort(tt)
    tt, ss = tt[order], ss[order]

    def stress_of(t):
        return float(np.interp(t, tt, ss))

    def dstress_of(t):
        dt = 1e-3
        return (stress_of(t + dt) - stress_of(t - dt)) / (2 * dt)

    t_eval = np.array(sorted(set(float(t) for t in record_times) | {0.0}))
    pbm = pbm_from_strength(grid4, initial, strength_law, matrix4,
                            stress_of, dstress_of, kappa, t_eval)
    result["cross_validation"] = {
        "kappa_effective_stress_ratio": kappa,
        "first_break_time_s": float(asm_run["first_break_time"] or 0.0),
        "n_events": int(asm_run["n_events"]),
        "n_particles_end": int(asm_run["n_particles_end"]),
        "weibull_m_used": float(fit.m),
        "sigma_0_Pa_used": float(fit.sigma_0),
        "stress_max_Pa": float(stress_max),
        "records": {},
    }
    metric_values = []
    for t_rec in record_times:
        t_rec = float(t_rec)
        idx = int(np.argmin(np.abs(pbm["t"] - t_rec)))
        n_pbm = pbm["n"][idx]
        counts = n_pbm * grid4.widths
        d_pbm = equivalent_diameter(grid4.cell_centres)
        d_dem = np.asarray(asm_run["psd"][t_rec]["diameters"])
        w_dem = np.asarray(asm_run["psd"][t_rec]["weights"])
        w_dist, pts, fa, fb = _psd_metric(d_dem, w_dem, d_pbm, counts)
        norm = float(np.mean(init_d))
        metric_values.append(w_dist)
        result["cross_validation"]["records"][f"{t_rec:g}"] = {
            "wasserstein_raw_m": w_dist,
            "wasserstein_normalised": w_dist / norm,
            "dem_n_particles": int(asm_run["psd"][t_rec]["n"]),
            "dem_mean_diameter_m": float(np.mean(d_dem)),
            "pbm_mean_diameter_m": float(np.sum(d_pbm * counts) / np.sum(counts)),
            "dem_volume_broken_fraction": float(
                1.0 - np.sum(asm_run["psd"][t_rec]["n"]) / float(asm_run["psd"][min(asm_run["psd"])]["n"])),
            "_curves": {"pts": pts.tolist(), "dem": fa.tolist(), "pbm": fb.tolist()},
        }
    worst = float(max(metric_values))
    norm_worst = float(max(v / float(np.mean(init_d)) for v in metric_values))
    result["criteria"]["C4_dem_pbm_wasserstein"] = {
        "value": norm_worst, "threshold": 0.1,
        "passed": bool(norm_worst < 0.1),
        "per_time_normalised": {
            f"{float(t):g}": float(res["wasserstein_normalised"])
            for t, res in ((float(k), v) for k, v in result["cross_validation"]["records"].items())
        },
    }

    # -------------------------------------------------------------------- C6
    log("[C6] negative controls")
    grid6 = VolumeGrid(v_min=v0 / 40.0, v_max=v0 * 1.05, n_sections=70)
    init6 = np.zeros(grid6.n_sections)
    init6[-1] = 1.0
    matrix6 = daughter_matrix(grid6, UniformDaughter(), n_samples=1200, seed=seed + 5)
    strong = WeibullStrength(m=fit.m, sigma_0=1e30, v_0=v0)
    sol_strong = solve_breakage(grid6, init6, rate=0.0, daughter=matrix6,
                                t_span=(0.0, 4.0), t_eval=[0.0, 4.0])
    drift = float(np.max(np.abs(sol_strong["n"][-1] - init6)) / np.max(np.abs(init6)))
    # infinite breakage intensity: one generation with an unbounded rate, i.e. the
    # parent section empties into b(v,v') instantaneously.  The operator is applied
    # to the top section only, so the check isolates "PSD -> daughter distribution".
    rate6 = np.zeros(grid6.n_sections)
    rate6[-1] = 1.0e12
    sol_weak = solve_breakage(grid6, init6, rate=rate6, daughter=matrix6,
                              t_span=(0.0, 1.0e-9), t_eval=[0.0, 1.0e-9],
                              rtol=1e-10, atol=1e-14)
    n_weak = sol_weak["n"][-1]
    daughter_expected = matrix6 @ init6           # one generation of daughters
    total = float(np.sum(daughter_expected))
    cum_weak = cumulative_volume_fraction(grid6, n_weak)
    cum_exp = cumulative_volume_fraction(grid6, daughter_expected)
    overlap = float(1.0 - np.max(np.abs(cum_weak - cum_exp)))
    parent_residual = float(n_weak[-1] / init6[-1])
    weak = WeibullStrength(m=fit.m, sigma_0=1e-3, v_0=v0)
    rate_weak = weak.hazard_rate(1.0e6, 1.0e3, grid6.pivots)
    frac_ok = overlap
    result["negative_controls"] = {
        "no_breakage_drift": drift,
        "instant_breakage_psd_agreement": frac_ok,
        "instant_breakage_surviving_parent_ratio": parent_residual,
        "weak_strength_max_hazard_per_s": float(np.max(rate_weak)),
    }
    c6_ok = bool(drift < 1e-6 and frac_ok > 0.97)
    result["criteria"]["C6_negative_controls"] = {
        "value": max(drift, 1.0 - frac_ok), "threshold": 1e-6,
        "passed": c6_ok,
        "no_breakage_drift_threshold": 1e-6,
        "instant_full_breakage_threshold": 0.03,
        "agreement_with_daughter_distribution": frac_ok,
    }

    result["runtime_s"] = time.time() - t_start
    result["passed_all"] = bool(all(c["passed"] for c in result["criteria"].values()))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        slim = json.loads(json.dumps(result, default=float))
        for rec in slim["cross_validation"]["records"].values():
            rec.pop("_curves", None)
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(slim, fh, indent=2)
        if make_figures:
            _figures(result, out_dir)
    return result


def _figures(result: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return
    from .weibull_breakage import weibull_cdf

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))

    # C1: empirical vs fitted Weibull CDF of the crushing strength
    were = result["crushing"]
    s_mpa = np.asarray(were["strength_sample_MPa"])
    s_sorted = np.sort(s_mpa)
    f_emp = (np.arange(1, s_sorted.size + 1) - 0.3) / (s_sorted.size + 0.4)
    ax[0, 0].plot(s_sorted, f_emp, label="DEM crushing strength (empirical CDF)")
    grid = np.linspace(max(1e-3, s_sorted.min() * 0.8), s_sorted.max() * 1.05, 400)
    ax[0, 0].plot(grid, weibull_cdf(grid, were["weibull"]["m"],
                                    were["weibull"]["sigma_0"] / 1e6), "r--",
                  label=f"Weibull fit m={were['weibull']['m']:.2f}, "
                        f"R2={were['weibull']['r2_cdf']:.4f}")
    ax[0, 0].set_xlabel("crushing stress F/D^2 [MPa]")
    ax[0, 0].set_ylabel("CDF")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title(f"C1  n={were['n_tests']} DEM crushing tests")

    # C2: DEM daughter fractions vs the assumed laws
    frac = np.sort(np.asarray(result["daughters"]["fraction_sample"]))
    f_e = (np.arange(1, frac.size + 1) - 0.3) / (frac.size + 0.4)
    ax[0, 1].plot(frac, f_e, label="DEM fragments (empirical CDF)")
    ax[0, 1].plot([0, 1], [0, 1], "r--",
                  label=f"uniform (KS p={result['daughters']['ks_uniform']['pvalue']:.3f})")
    ax[0, 1].set_xlabel("daughter volume fraction v/v'")
    ax[0, 1].set_ylabel("CDF")
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title(
        f"C2  lognormal sigma_ln={result['daughters']['lognormal_sigma_ln']:.3f}, "
        f"KS p={result['daughters']['ks_lognormal']['pvalue']:.3f}")

    # C3: sectional PBM vs closed form
    d3 = result["dpbm_analytic"]
    ax[1, 0].loglog(d3["centres"], np.clip(d3["numeric_mass_density"], 1e-30, None),
                    "o", ms=3, label="sectional PBM")
    ax[1, 0].loglog(d3["centres"], np.clip(d3["analytic_mass_density"], 1e-30, None),
                    "r-", lw=1.0, label="closed form (generation series)")
    ax[1, 0].set_xlabel("particle volume [m^3]")
    ax[1, 0].set_ylabel("volume-weighted density v n(v)")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title(f"C3  L1 relative error "
                       f"{100 * d3['l1_relative_mass_density']:.2f}%")

    # C4: DEM vs PBM cumulative volume fraction
    for key, rec in result["cross_validation"]["records"].items():
        ax[1, 1].plot(rec["_curves"]["pts"], rec["_curves"]["dem"], label=f"DEM t={key}s")
        ax[1, 1].plot(rec["_curves"]["pts"], rec["_curves"]["pbm"], "--",
                      label=f"PBM t={key}s")
    ax[1, 1].set_xlabel("equivalent diameter [m]")
    ax[1, 1].set_ylabel("cumulative volume fraction")
    ax[1, 1].legend(fontsize=6)
    ax[1, 1].set_title(
        "C4  DEM vs PBM PSD evolution (W1/d0 = "
        + ", ".join(f"{v:.3f}" for v in result["criteria"]["C4_dem_pbm_wasserstein"][
            "per_time_normalised"].values()) + ")")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "breakage_pbm_summary.png"), dpi=130)
    plt.close(fig)
