"""Acceptance gates (a)-(d) for mission P46W3-N02.

Each gate returns a plain dict with the measured numbers, the predeclared
threshold from ``data/gates_p46w3n02.md`` and a boolean ``pass``.  Nothing is
hard-coded as a result: every number comes out of a DEM run in this module.

Gates
-----
(a) Glen exponent: constant-stress DEM creep at four stresses -> fitted ``n``,
    window sensitivity, and a 2 % measurement-noise Monte-Carlo 95 % CI.
(b) Andrade curve: DEM chain creep vs ``eps_0 + eps_1 t^(1/3)``, plus a
    time-step convergence study (the error must shrink with ``dt``).
(c) Relaxation: single-bond Norton relaxation sweep ``tau_eff(A)``, fractional
    chain sweep ``tau_eff(eta_alpha)``, both with a log-log correlation
    statistic, plus the two exact limits (alpha = 1 -> exponential,
    alpha = 1/2 -> ``exp(x^2) erfc(x)``) and the honest quality of a single
    exponential for alpha = 1/3.
(d) Negative controls: ``A -> 0`` and ``eta_alpha -> inf`` collapse to the
    instantaneous elastic strain.

The functions are pure in their arguments (fixed seeds) so a run is
reproducible from the JSON that the driver writes.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc, gamma as _gamma

from tensordem.ice_creep import (
    GlenCreepLaw,
    LatticeCreepParameters,
    power_law_fit,
    punch_relaxation,
    uniaxial_constant_stress,
)
from tensordem.ice_creep_andrade import (
    ALPHA_ANDRADE,
    AndradeCreepLaw,
    FractionalDemChain,
    exponential_fit_quality,
    fit_andrade,
    fractional_relaxation_reference,
    relaxation_time_constant,
    tau_scale_correlation,
)

#: Predeclared thresholds (mirror of data/gates_p46w3n02.md).
THRESHOLDS = {
    "a1_glen_rel_error": 0.10,
    "a2_ci_contains_three": True,
    "b1_curve_max_rel_error": 0.05,
    "b2_alpha_rel_error": 0.10,
    "b3_eps0_rel_error": 0.02,
    "b4_convergence": "error must decrease monotonically with reducing dt",
    "c1_abs_r_norton": 0.90,
    "c2_slope_rel_error": 0.05,
    "c3_abs_r_fractional": 0.90,
    "c4_exp_limit_max_rel": 0.01,
    "c5_half_order_max_rel": 0.03,
    "d1_norton_no_creep_rel": 0.01,
    "d2_fractional_no_creep_rel": 0.01,
    "d3_creep_scaling_rel": 0.01,
}

YOUNG_ICE = 9.0e9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def andrade_law_for_retardation(alpha: float = ALPHA_ANDRADE,
                                retardation_s: float = 100.0,
                                young: float = YOUNG_ICE) -> AndradeCreepLaw:
    """Springpot viscosity placing the creep/elastic crossover at ``retardation_s``.

    ``eps_creep(t_r) / eps_elastic = 1`` requires
    ``eta_alpha = E t_r^alpha / Gamma(1 + alpha)``.
    """
    eta = young * retardation_s ** alpha / float(_gamma(1.0 + alpha))
    return AndradeCreepLaw(alpha=alpha, eta_alpha=eta)


def norton_bond_relaxation(*, fluidity: float, n: float = 3.0,
                           young: float = YOUNG_ICE, radius: float = 0.01,
                           rest: float = 0.02, strain0: float = 1.0e-3,
                           t_end: float = 2.0e4, n_steps: int = 4000) -> dict:
    """Simulated fixed-elongation relaxation of one Norton (Maxwell) bond.

    The bond obeys ``dF/dt = -C F^n`` with ``C = k A L0 A_b^-n`` (``A`` =
    fluidity).  The decay is stiff for large ``A``, so the history is integrated
    with **implicit Euler** (Newton per step, unconditionally stable) rather
    than an explicit scheme -- an explicit RK4 blows up as soon as
    ``dt C F^(n-1) > 1``.  The closed form in
    :func:`tensordem.ice_creep.single_bond_relaxation_analytic` is the reference
    for the relaxation *time*, so this function supplies an independent,
    integrator-based history rather than re-evaluating the formula.
    """
    area = math.pi * radius ** 2
    stiffness = young * area / rest
    f0 = stiffness * rest * strain0
    c_rate = stiffness * fluidity * rest * area ** (-n)

    dt = t_end / n_steps
    times = [0.0]
    forces = [f0]
    force = f0
    for step in range(1, n_steps + 1):
        # implicit Euler: F_{n+1} + dt C F_{n+1}^n = F_n   (root in (0, F_n])
        target = force
        guess = force
        for _ in range(60):
            g = guess + dt * c_rate * abs(guess) ** n - target
            dg = 1.0 + dt * c_rate * n * guess ** (n - 1.0)
            step_newton = g / dg
            guess -= step_newton
            if guess <= 0.0:
                guess = 0.5 * force
            if abs(step_newton) <= 1e-15 * max(abs(guess), 1e-300):
                break
        force = guess
        times.append(step * dt)
        forces.append(force)
        if force < 1e-6 * f0:
            break
    times = np.array(times)
    forces = np.array(forces)
    return {"t": times, "force": forces, "force_normalised": forces / f0,
            "f0": f0, "stiffness": stiffness, "area": area, "rest": rest,
            "fluidity": fluidity, "n": n, "dt": dt, "scheme": "implicit-euler"}


# --------------------------------------------------------------------------- #
# (a) Glen exponent from DEM constant-stress creep
# --------------------------------------------------------------------------- #
def gate_glen_exponent(*, n_true: float = 3.0, a_ref: float = 1.0e-24,
                       sigmas=(5.0e5, 1.0e6, 2.0e6, 4.0e6),
                       n_cells: int = 3, spacing: float = 0.02,
                       t_end: float = 200.0, dt: float = 10.0,
                       windows=((10.0, 40.0), (10.0, 60.0), (20.0, 80.0)),
                       noise: float = 0.02, n_mc: int = 200,
                       seed: int = 20261110,
                       resolution_cells=(3, 4, 5)) -> dict:
    """Criterion (a): recover the Glen exponent from simulated DEM creep curves."""
    params = LatticeCreepParameters(
        law=GlenCreepLaw(a_ref=a_ref, n=n_true), young_modulus=YOUNG_ICE)
    histories = {}
    for sigma0 in sigmas:
        out = uniaxial_constant_stress(sigma0=sigma0, n_cells=n_cells,
                                       spacing=spacing, params=params,
                                       t_end=t_end, dt=dt)
        histories[float(sigma0)] = {
            "t": out["t"], "strain": out["strain"],
            "creep_ratio_max": float(out["creep_ratio"].max()),
        }

    per_window = {}
    for lo, hi in windows:
        rates = []
        for sigma0, hist in histories.items():
            mask = (hist["t"] >= lo) & (hist["t"] <= hi)
            rate = float(np.polyfit(hist["t"][mask], hist["strain"][mask], 1)[0])
            rates.append(rate)
        fit = power_law_fit(np.array(list(histories)), np.array(rates))
        per_window[f"{lo:g}-{hi:g}s"] = {
            "exponent": fit["exponent"],
            "r2": fit["r2"],
            "rel_error_vs_true": abs(fit["exponent"] - n_true) / n_true,
            "rates": rates,
        }

    primary = per_window[f"{windows[0][0]:g}-{windows[0][1]:g}s"]

    # measurement-noise Monte Carlo: the same fit on rates perturbed by the
    # stated relative noise, to put an uncertainty on the exponent
    rng = np.random.default_rng(seed)
    base_rates = np.array(primary["rates"])
    sig = np.array(list(histories))
    mc = []
    for _ in range(n_mc):
        perturbed = base_rates * (1.0 + noise * rng.standard_normal(base_rates.size))
        mc.append(power_law_fit(sig, perturbed)["exponent"])
    mc = np.array(mc)
    lo_ci, hi_ci = (float(v) for v in np.percentile(mc, [2.5, 97.5]))
    rel_error = abs(primary["exponent"] - n_true) / n_true

    # resolution study: the lattice exponent is known to be biased low because
    # the bond section is the sphere section, not a Voronoi facet, so the axial
    # bond stress is not the macroscopic stress and load redistributes between
    # axial and diagonal bonds as the specimen creeps.  Quantify it.
    resolution = {}
    for cells in resolution_cells:
        if cells == n_cells:
            resolution[str(cells)] = {"exponent": primary["exponent"],
                                      "r2": primary["r2"]}
            continue
        res_params = LatticeCreepParameters(
            law=GlenCreepLaw(a_ref=a_ref, n=n_true), young_modulus=YOUNG_ICE)
        res_rates = []
        for sigma0 in sigmas:
            hist = uniaxial_constant_stress(sigma0=sigma0, n_cells=cells,
                                            spacing=spacing, params=res_params,
                                            t_end=t_end, dt=dt)
            mask = (hist["t"] >= windows[0][0]) & (hist["t"] <= windows[0][1])
            res_rates.append(float(np.polyfit(hist["t"][mask],
                                              hist["strain"][mask], 1)[0]))
        res_fit = power_law_fit(np.array(list(histories)), np.array(res_rates))
        resolution[str(cells)] = {"exponent": res_fit["exponent"],
                                  "r2": res_fit["r2"],
                                  "n_particles": cells ** 3}
    return {
        "gate": "a",
        "n_true": n_true,
        "sigmas_Pa": list(histories),
        "histories": histories,
        "per_window": per_window,
        "fitted_n": primary["exponent"],
        "fit_r2": primary["r2"],
        "rel_error_vs_literature": rel_error,
        "window_max_spread": max(v["exponent"] for v in per_window.values())
                             - min(v["exponent"] for v in per_window.values()),
        "noise_mc": {
            "noise_level": noise,
            "n_realisations": n_mc,
            "seed": seed,
            "mean": float(mc.mean()),
            "std": float(mc.std(ddof=1)),
            "ci95": [lo_ci, hi_ci],
            "ci_contains_three": bool(lo_ci <= n_true <= hi_ci),
        },
        "resolution_study": resolution,
        "resolution_bias_vs_n3": {
            key: abs(row["exponent"] - n_true) / n_true
            for key, row in resolution.items()},
        "error_budget_posthoc": {
            "note": "post-hoc diagnostic, NOT a predeclared gate: the Monte-Carlo "
                    "interval covers random measurement noise only, the rest of the "
                    "error is the systematic lattice bias measured by the resolution "
                    "study",
            "systematic_bias": rel_error,
            "statistical_halfwidth": 0.5 * (hi_ci - lo_ci),
            "total": rel_error + 0.5 * (hi_ci - lo_ci),
            "total_within_mission_10pct": bool(
                rel_error + 0.5 * (hi_ci - lo_ci) <= THRESHOLDS["a1_glen_rel_error"]),
        },
        "threshold_rel_error": THRESHOLDS["a1_glen_rel_error"],
        "mission_criterion_pass": bool(rel_error <= THRESHOLDS["a1_glen_rel_error"]),
        "statistical_ci_contains_three": bool(lo_ci <= n_true <= hi_ci),
        "pass": bool(rel_error <= THRESHOLDS["a1_glen_rel_error"]
                     and lo_ci <= n_true <= hi_ci),
    }


# --------------------------------------------------------------------------- #
# (b) Andrade creep curve vs eps_0 + eps_1 t^(1/3)
# --------------------------------------------------------------------------- #
def gate_andrade_curve(*, alpha: float = ALPHA_ANDRADE, sigma0: float = 1.0e6,
                       retardation_s: float = 100.0, t_end: float = 1000.0,
                       n_steps: int = 2000, young: float = YOUNG_ICE,
                       convergence_steps=(250, 500, 1000, 2000)) -> dict:
    """Criterion (b): DEM creep curve vs the analytic Andrade law."""
    law = andrade_law_for_retardation(alpha, retardation_s, young)
    chain = FractionalDemChain(n_bonds=8, young_modulus=young, law=law)
    out = chain.creep(sigma0, t_end, n_steps)

    rel = np.abs(out["strain"] - out["analytic_strain"]) / np.abs(out["analytic_strain"])
    fit_all = fit_andrade(out["t"], out["strain"])
    skip = min(10, n_steps // 10)
    fit_late = fit_andrade(out["t"][skip:], out["strain"][skip:])

    convergence = {}
    for steps in convergence_steps:
        run = chain.creep(sigma0, t_end, steps)
        err = float(np.max(np.abs(run["strain"] - run["analytic_strain"])
                           / np.abs(run["analytic_strain"])))
        convergence[str(steps)] = {"dt": run["dt"], "max_rel_error": err}

    ordered = [convergence[str(s)]["max_rel_error"] for s in convergence_steps]
    monotone = all(b < a for a, b in zip(ordered, ordered[1:]))

    return {
        "gate": "b",
        "alpha_true": alpha,
        "sigma0_Pa": sigma0,
        "young_modulus_Pa": young,
        "eta_alpha": law.eta_alpha,
        "beta_analytic": out["eps_1_analytic"],
        "eps_0_analytic": out["eps_0_analytic"],
        "t_end": t_end,
        "n_steps": n_steps,
        "dt": out["dt"],
        "max_rel_error_curve": float(rel[1:].max()),
        "rms_rel_error_curve": float(np.sqrt(np.mean(rel[1:] ** 2))),
        "fit_full": fit_all,
        "fit_after_startup": fit_late,
        "alpha_rel_error": abs(fit_late["alpha"] - alpha) / alpha,
        "eps0_rel_error": abs(fit_late["eps_0"] / out["eps_0_analytic"] - 1.0),
        "eps1_rel_error": abs(fit_late["eps_1"] / out["eps_1_analytic"] - 1.0),
        "convergence": convergence,
        "convergence_monotone": monotone,
        "history": {"t": out["t"], "strain": out["strain"],
                    "analytic": out["analytic_strain"],
                    "creep_strain": out["creep_strain"]},
        "thresholds": {
            "max_rel_error_curve": THRESHOLDS["b1_curve_max_rel_error"],
            "alpha_rel_error": THRESHOLDS["b2_alpha_rel_error"],
            "eps0_rel_error": THRESHOLDS["b3_eps0_rel_error"],
        },
        "pass": bool(float(rel[1:].max()) <= THRESHOLDS["b1_curve_max_rel_error"]
                     and abs(fit_late["alpha"] - alpha) / alpha
                     <= THRESHOLDS["b2_alpha_rel_error"]
                     and abs(fit_late["eps_0"] / out["eps_0_analytic"] - 1.0)
                     <= THRESHOLDS["b3_eps0_rel_error"]
                     and monotone),
    }


# --------------------------------------------------------------------------- #
# (c) relaxation: tau vs creep parameter, exact limits
# --------------------------------------------------------------------------- #
def gate_relaxation(*, n: float = 3.0, strain0: float = 1.0e-3,
                    fluidities=(1.0e-23, 3.16e-23, 1.0e-22, 3.16e-22, 1.0e-21),
                    eta_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
                    alpha: float = ALPHA_ANDRADE,
                    t_end_bond: float = 2.0e4,
                    t_end_chain: float = 2000.0, n_steps_chain: int = 1000,
                    lattice_fluidities=(1.0e-26, 3.16e-26, 1.0e-25, 3.16e-25),
                    lattice_cells: int = 4, lattice_level: float = 0.9,
                    lattice_t_end: float = 2.0e4) -> dict:
    """Criterion (c): ``F(t) = F_0 exp(-t/tau)`` and ``tau`` tied to the creep law."""
    # --- Norton single bond: tau_eff proportional to 1/A --------------------
    bond_sweep = []
    for fluidity in fluidities:
        # the decay time of a Norton bond is tau = 1/(k A L0 A_b^-n F0^(n-1)),
        # i.e. it spans ~2 decades over this sweep; each row is therefore run
        # over 20 analytic crossing times so that the transient is resolved
        law_probe = norton_bond_relaxation(fluidity=fluidity, n=n,
                                           t_end=1.0, n_steps=200)
        beta_probe = ((n - 1.0) * law_probe["stiffness"] * fluidity
                      * law_probe["rest"] * law_probe["area"] ** (-n)
                      * law_probe["f0"] ** (n - 1.0))
        t_end_row = min(max(20.0 * (math.exp(n - 1.0) - 1.0) / beta_probe, 1.0e-3),
                        t_end_bond)
        hist = norton_bond_relaxation(fluidity=fluidity, n=n,
                                      t_end=t_end_row, n_steps=4000)
        tau = relaxation_time_constant(hist["t"], hist["force_normalised"])
        # closed-form crossing time of F/F0 = (1 + beta t)^(-1/(n-1)):
        #   t_1/e = (e^(n-1) - 1) / beta with beta = (n-1) k A L0 A_b^-n F0^(n-1)
        beta = ((n - 1.0) * hist["stiffness"] * fluidity * hist["rest"]
                * hist["area"] ** (-n) * hist["f0"] ** (n - 1.0))
        analytic_crossing = (math.exp(n - 1.0) - 1.0) / beta
        bond_sweep.append({
            "fluidity": fluidity,
            "tau_eff_s": tau,
            "final_force_ratio": float(hist["force_normalised"][-1]),
            "analytic_tau_crossing": analytic_crossing,
            "analytic_rel_error": abs(tau / analytic_crossing - 1.0),
        })
    corr_norton = tau_scale_correlation(
        [row["fluidity"] for row in bond_sweep], [row["tau_eff_s"] for row in bond_sweep])
    tau_analytic_err = max(row["analytic_rel_error"] for row in bond_sweep)

    # --- fractional chain: tau_eff proportional to eta^(1/alpha) -----------
    base_law = andrade_law_for_retardation(alpha)
    chain_sweep = []
    for mult in eta_multipliers:
        law = AndradeCreepLaw(alpha=alpha, eta_alpha=base_law.eta_alpha * mult)
        chain = FractionalDemChain(n_bonds=8, law=law)
        # run six retardation times so that the 1/e crossing is always resolved
        t_end_chain = 6.0 * chain.retardation_time
        out = chain.relax(1.0e-3, t_end_chain, n_steps_chain)
        tau = relaxation_time_constant(out["t"], out["force_normalised"])
        # the 1/e crossing of the Mittag-Leffler reference (not the element
        # retardation time tau = (eta/E)^(1/alpha), which is a different scale)
        tau_ref = relaxation_time_constant(out["t"], out["reference_normalised"])
        chain_sweep.append({
            "eta_alpha": law.eta_alpha,
            "tau_eff_s": tau,
            "tau_crossing_reference_s": tau_ref,
            "tau_retardation_s": out["tau"],
            "crossing_rel_error": abs(tau / tau_ref - 1.0),
            "relaxation_max_rel_error": float(
                np.max(np.abs(out["force_normalised"] - out["reference_normalised"])
                       / out["reference_normalised"])),
        })
    corr_frac = tau_scale_correlation([row["eta_alpha"] for row in chain_sweep],
                                      [row["tau_eff_s"] for row in chain_sweep])
    frac_slope_rel = abs(corr_frac["exponent"] - 1.0 / alpha) / (1.0 / alpha)

    # --- exact limits and the honest quality of a single exponential -------
    law1 = andrade_law_for_retardation(1.0)
    chain1 = FractionalDemChain(n_bonds=8, law=law1)
    run1 = chain1.relax(1.0e-3, 500.0, 1000)
    exp_dev = float(np.max(np.abs(run1["force_normalised"]
                                  - np.exp(-run1["t"] / run1["tau"]))))

    law_half = andrade_law_for_retardation(0.5)
    chain_half = FractionalDemChain(n_bonds=8, law=law_half)
    run_half = chain_half.relax(1.0e-3, 500.0, 1000)
    x = np.sqrt(run_half["t"] / run_half["tau"])
    erfc_ref = np.exp(x * x) * erfc(x)
    half_dev = float(np.max(np.abs(run_half["force_normalised"] - erfc_ref)))

    law_third = andrade_law_for_retardation(alpha)
    chain_third = FractionalDemChain(n_bonds=8, law=law_third)
    run_third = chain_third.relax(1.0e-3, t_end_chain, n_steps_chain)
    exp_quality = exponential_fit_quality(run_third["t"], run_third["force_normalised"])

    # --- many-body DEM check: flat-punch relaxation of a bonded block ------
    lattice_sweep = []
    for fluidity in lattice_fluidities:
        params = LatticeCreepParameters(
            law=GlenCreepLaw(a_ref=fluidity, n=n), young_modulus=YOUNG_ICE)
        out = punch_relaxation(n_cells=lattice_cells, params=params,
                               depth_fraction=1.0e-4, t_end=lattice_t_end, dt=10.0)
        normalised = out["force"] / out["initial_force"]
        tau = relaxation_time_constant(out["t"], normalised, level=lattice_level)
        quality = exponential_fit_quality(out["t"], normalised, level=lattice_level)
        lattice_sweep.append({
            "fluidity": fluidity,
            "level": lattice_level,
            "tau_eff_s": tau,
            "force_ratio_end": float(normalised[-1]),
            "creep_ratio_max": float(out["creep_ratio"].max()),
            "initial_force_N": float(out["initial_force"]),
            "exponential_fit": quality,
        })
    corr_lattice = tau_scale_correlation(
        [row["fluidity"] for row in lattice_sweep],
        [row["tau_eff_s"] for row in lattice_sweep])
    lattice_history = punch_relaxation(
        n_cells=lattice_cells,
        params=LatticeCreepParameters(law=GlenCreepLaw(a_ref=lattice_fluidities[0], n=n),
                                      young_modulus=YOUNG_ICE),
        depth_fraction=1.0e-4, t_end=lattice_t_end, dt=10.0)

    threshold_c2 = THRESHOLDS["c2_slope_rel_error"]
    return {
        "gate": "c",
        "n_true": n,
        "alpha": alpha,
        "norton_bond_sweep": {
            "rows": bond_sweep,
            "correlation": corr_norton,
            "analytic_tau_max_rel_error": tau_analytic_err,
            "slope_rel_error_vs_minus_one": abs(corr_norton["exponent"] + 1.0),
        },
        "fractional_chain_sweep": {
            "rows": chain_sweep,
            "correlation": corr_frac,
            "slope_rel_error_vs_one_over_alpha": frac_slope_rel,
            "crossing_rel_error_max": max(r["crossing_rel_error"] for r in chain_sweep),
            "history": {"t": run_third["t"],
                        "force_normalised": run_third["force_normalised"],
                        "reference": run_third["reference_normalised"],
                        "tau": run_third["tau"]},
        },
        "lattice_punch_sweep": {
            "rows": lattice_sweep,
            "correlation": corr_lattice,
            "slope_rel_error_vs_minus_one": abs(corr_lattice["exponent"] + 1.0),
            "history": {"t": lattice_history["t"],
                        "force_normalised": (lattice_history["force"]
                                             / lattice_history["initial_force"])},
        },
        "exact_limits": {
            "alpha1_max_rel_deviation_from_exp": exp_dev,
            "alpha_half_max_rel_deviation_from_erfc": half_dev,
            "alpha_third_exponential_fit": exp_quality,
        },
        "thresholds": {
            "abs_r_norton": THRESHOLDS["c1_abs_r_norton"],
            "slope_rel_error": threshold_c2,
            "abs_r_fractional": THRESHOLDS["c3_abs_r_fractional"],
            "exp_limit_max_rel": THRESHOLDS["c4_exp_limit_max_rel"],
            "half_order_max_rel": THRESHOLDS["c5_half_order_max_rel"],
        },
        "pass": bool(corr_norton["abs_r"] >= THRESHOLDS["c1_abs_r_norton"]
                     and abs(corr_norton["exponent"] + 1.0) <= threshold_c2
                     and corr_frac["abs_r"] >= THRESHOLDS["c3_abs_r_fractional"]
                     and exp_dev <= THRESHOLDS["c4_exp_limit_max_rel"]
                     and half_dev <= THRESHOLDS["c5_half_order_max_rel"]
                     and corr_lattice["abs_r"] >= THRESHOLDS["c1_abs_r_norton"]),
    }


# --------------------------------------------------------------------------- #
# (d) negative controls
# --------------------------------------------------------------------------- #
def gate_negative_control(*, sigma0: float = 1.0e6, n_cells: int = 3,
                          t_end: float = 200.0, dt: float = 10.0,
                          alpha: float = ALPHA_ANDRADE) -> dict:
    """Criterion (d): no creep parameter -> no creep, only elastic strain."""
    # Norton branch: fluidity at the floor (A -> 0)
    params = LatticeCreepParameters(law=GlenCreepLaw(a_ref=1.0e-30, n=3.0),
                                    young_modulus=YOUNG_ICE)
    out = uniaxial_constant_stress(sigma0=sigma0, n_cells=n_cells, params=params,
                                   t_end=t_end, dt=dt)
    eps_el = sigma0 / YOUNG_ICE
    strain_end = float(out["strain"][-1])
    # lattice strain carries the discretisation factor of the bond section, so
    # the reference is the *initial* (elastic) strain of the same lattice
    strain_start = float(out["strain"][1])
    norton_rel = abs(strain_end - strain_start) / abs(strain_start)

    # fractional branch: eta_alpha -> very large
    law = AndradeCreepLaw(alpha=alpha, eta_alpha=1.0e16)
    chain = FractionalDemChain(n_bonds=8, law=law)
    run = chain.creep(sigma0, 1000.0, 500)
    frac_rel = float(abs(run["creep_strain"][-1]) / run["elastic_strain"][-1])

    # monotone collapse with the creep parameter
    nominal = FractionalDemChain(n_bonds=8, law=andrade_law_for_retardation(alpha))
    stiff = FractionalDemChain(n_bonds=8, law=AndradeCreepLaw(
        alpha=alpha, eta_alpha=nominal.law.eta_alpha * 1.0e4))
    c_nom = float(nominal.creep(sigma0, 1000.0, 500)["creep_strain"][-1])
    c_stiff = float(stiff.creep(sigma0, 1000.0, 500)["creep_strain"][-1])
    scaling_rel = abs(c_stiff / (c_nom * 1.0e-4) - 1.0)

    return {
        "gate": "d",
        "sigma0_Pa": sigma0,
        "young_modulus_Pa": YOUNG_ICE,
        "elastic_strain_analytic": eps_el,
        "norton": {
            "fluidity": 1.0e-30,
            "strain_initial": strain_start,
            "strain_end": strain_end,
            "rel_change": norton_rel,
        },
        "fractional": {
            "eta_alpha": law.eta_alpha,
            "creep_strain_end": float(run["creep_strain"][-1]),
            "elastic_strain": float(run["elastic_strain"][-1]),
            "creep_over_elastic": frac_rel,
        },
        "creep_scaling": {
            "nominal_creep": c_nom,
            "stiff_creep": c_stiff,
            "expected_ratio": 1.0e-4,
            "rel_error": scaling_rel,
        },
        "thresholds": {
            "norton_no_creep_rel": THRESHOLDS["d1_norton_no_creep_rel"],
            "fractional_no_creep_rel": THRESHOLDS["d2_fractional_no_creep_rel"],
            "creep_scaling_rel": THRESHOLDS["d3_creep_scaling_rel"],
        },
        "pass": bool(norton_rel <= THRESHOLDS["d1_norton_no_creep_rel"]
                     and frac_rel <= THRESHOLDS["d2_fractional_no_creep_rel"]
                     and scaling_rel <= THRESHOLDS["d3_creep_scaling_rel"]),
    }


def gate_summary(gates: dict) -> dict:
    """Roll the four gates up; ``pass`` is a single boolean."""
    keys = [k for k in ("a", "b", "c", "d") if k in gates]
    passed = {k: bool(gates[k]["pass"]) for k in keys}
    return {"gate_pass": passed,
            "all_gates_pass": all(passed.values()),
            "failed_gates": [k for k, v in passed.items() if not v]}