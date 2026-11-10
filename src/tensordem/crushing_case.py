"""Acceptance cases for Weibull-controlled particle crushing (P96W2-N03).

The module turns the criteria C1..C7 of the campaign node into runnable,
machine-checkable cases, and is used by both the CLI and ``tests/test_crushing.py``::

    tensordem-crushing --case verify      # one command, red/green, exit code

Cases
-----
C1  mass / momentum conservation over 500 parent -> children replacements
C2  Weibull inverse consistency: 500 samples -> MLE inversion of (m, sigma0)
C3  size effect: median strength vs diameter, log-log slope vs -3/m
C4  fragment spectrum: Gaudin-Schuhmann fit of the crushed fragments
C5  energy closure of the 1-D bidisperse bed over 1e4 steps
C6  negative controls: crushing-off, crushing-very-strong, bond/crushing
    orthogonality
C7  regression: ``dem3d`` trajectories unchanged (bit-for-bit checksum)

Every case returns a dict with ``passed``, the measured numbers and the
criterion it is scored against, so the report can be generated without any
hand-copied number.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crushing import (
    CrushingConfig,
    CrushingDEM,
    Fragmenter,
    MassMomentumLedger,
    Particle,
    WeibullStrength,
    fit_gaudin_schuhmann,
    size_effect_slope,
    sphere_mass,
)

#: Ice-like demonstration material.  ``sigma0`` is chosen so that the
#: displacement-controlled cases reach crushing inside the simulated window;
#: it is a *material input*, not a result.
DEFAULT_MATERIAL = WeibullStrength(d0=2.0e-3, sigma0=5.0e5, m=3.0)
DEFAULT_GAMMA = 0.1              # J/m^2, ice fracture surface energy
DEFAULT_SIZES = (1.0e-3, 2.0e-3, 4.0e-3)


def default_config(**overrides: Any) -> CrushingConfig:
    """Crushing configuration shared by the acceptance cases."""
    params: dict[str, Any] = dict(
        density=917.0,
        contact_stiffness=1.0e6,
        friction_coefficient=0.4,
        gamma=DEFAULT_GAMMA,
        weibull=DEFAULT_MATERIAL,
        fragmenter=Fragmenter(k=8, n=0.8),
        max_generation=1,
        dt_safety=0.1,
        record_every=50,
    )
    params.update(overrides)
    return CrushingConfig(**params)


# --------------------------------------------------------------------------
# C1 - mass / momentum conservation of the replacement operator
# --------------------------------------------------------------------------
def case_c1_replacement(n_events: int = 500, *, seed: int = 0,
                        tolerance: float = 1.0e-12) -> dict[str, Any]:
    """500 parent -> k children replacements, mass and momentum residuals."""
    rng = np.random.default_rng(seed)
    fragmenter = Fragmenter(k=8, n=0.8)
    ledger = MassMomentumLedger(gamma=DEFAULT_GAMMA)
    max_mass = 0.0
    max_momentum = 0.0
    for event in range(int(n_events)):
        diameter = float(rng.uniform(0.5e-3, 4.0e-3))
        velocity = rng.normal(size=3) * rng.uniform(0.1, 2.0)
        parent = Particle(position=rng.normal(size=3) * 1.0e-3, velocity=velocity,
                          radius=0.5 * diameter,
                          mass=sphere_mass(diameter, 917.0),
                          strength=DEFAULT_MATERIAL.median_strength(diameter),
                          lineage=event, generation=0)
        result = fragmenter.split(parent, density=917.0, rng=rng,
                                  separation_energy_J=float(rng.uniform(0.0, 1.0e-7)))
        ledger.record_replacement(result, released_elastic_energy_J=1.0e-7)
        max_mass = max(max_mass, result.mass_residual())
        max_momentum = max(max_momentum, result.momentum_residual())
    report = ledger.replacement_report()
    passed = max_mass <= tolerance and max_momentum <= tolerance
    return {
        "case": "C1",
        "criterion": f"mass and momentum residual <= {tolerance:g} over {n_events} replacements",
        "events": int(n_events),
        "max_mass_residual": max_mass,
        "max_momentum_residual": max_momentum,
        "ledger_max_mass_residual": report["max_mass_residual"],
        "ledger_max_momentum_residual": report["max_momentum_residual"],
        "tolerance": tolerance,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# C2 - Weibull inverse consistency
# --------------------------------------------------------------------------
def case_c2_weibull_inversion(n_samples: int = 500, *, seed: int = 1,
                              material: WeibullStrength = DEFAULT_MATERIAL,
                              tolerance: float = 0.10) -> dict[str, Any]:
    """Draw ``n_samples`` strengths and invert (m, sigma0) by MLE."""
    samples = material.sample(n_samples, material.d0, seed=seed)
    fit = material.mle_fit(samples, material.d0)
    passed = fit["m_error"] <= tolerance and fit["sigma0_error"] <= tolerance
    return {
        "case": "C2",
        "criterion": f"MLE inversion of m and sigma0 within {tolerance:.0%} of the input",
        "n_samples": int(n_samples),
        "input_m": material.m,
        "input_sigma0_Pa": material.sigma0,
        "m_hat": fit["m_hat"],
        "sigma0_hat_Pa": fit["sigma0_hat"],
        "m_relative_error": fit["m_error"],
        "sigma0_relative_error": fit["sigma0_error"],
        "log_likelihood": fit["log_likelihood"],
        "tolerance": tolerance,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# C3 - size effect slope
# --------------------------------------------------------------------------
def case_c3_size_effect(diameters: tuple[float, ...] = DEFAULT_SIZES, *,
                        n_samples: int = 2000, seed: int = 2,
                        material: WeibullStrength = DEFAULT_MATERIAL,
                        tolerance: float = 0.15) -> dict[str, Any]:
    """Median strength at d = 1/2/4 mm; log-log slope vs ``-3/m``."""
    medians: list[float] = []
    means: list[float] = []
    for index, diameter in enumerate(diameters):
        samples = material.sample(n_samples, diameter, seed=seed + 101 * index)
        medians.append(float(np.median(samples)))
        means.append(float(np.mean(samples)))
    fit_median = size_effect_slope(diameters, medians)
    fit_mean = size_effect_slope(diameters, means)
    theory = material.size_effect_exponent
    slope_error = abs(fit_median["slope"] - theory) / abs(theory)
    passed = slope_error <= tolerance
    return {
        "case": "C3",
        "criterion": f"fitted log-log slope within {tolerance:.0%} of -3/m = {theory:.3f}",
        "diameters_m": list(diameters),
        "n_samples_per_size": int(n_samples),
        "median_strength_Pa": medians,
        "mean_strength_Pa": means,
        "fitted_slope_median": fit_median["slope"],
        "fitted_slope_mean": fit_mean["slope"],
        "fit_r_squared": fit_median["r_squared"],
        "theory_slope": theory,
        "slope_relative_error": slope_error,
        "tolerance": tolerance,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# single-particle compression (C3 cross-check, C4 source, figure)
# --------------------------------------------------------------------------
def run_single_particle(diameter: float, *, seed: int,
                        platen_velocity: float = -0.05,
                        steps: int = 3000,
                        material: WeibullStrength = DEFAULT_MATERIAL,
                        config: CrushingConfig | None = None) -> dict[str, Any]:
    """Displacement-controlled compression of one particle between two platens."""
    cfg = config or default_config(seed=seed, weibull=material)
    engine = CrushingDEM.single_sphere(diameter, cfg, seed=seed)
    engine.platen_velocity = float(platen_velocity)
    assigned = float(engine.strengths[0])
    track_stress: list[float] = []
    track_displacement: list[float] = []
    initial_z = float(engine.top_platen_z)
    after_break = 0
    for _ in range(int(steps)):
        engine.step()
        track_stress.append(float((engine._force_pass().max_compressive_force_N
                                   / (math.pi * engine.radii ** 2)).max()))
        track_displacement.append(initial_z - float(engine.top_platen_z))
        if engine.crushing_events:
            # Keep a short tail after the first breakage for the figure, then
            # stop: the census only needs the first-generation fragments.
            after_break += 1
            if after_break > 20:
                break
    diagnostics = engine.diagnostics()
    measured = engine.breakage_stress_Pa[0] if engine.breakage_stress_Pa else float("nan")
    return {
        "diameter_m": float(diameter),
        "seed": int(seed),
        "assigned_strength_Pa": assigned,
        "measured_breakage_stress_Pa": measured,
        "strength_relative_error": (abs(measured - assigned) / assigned
                                    if engine.breakage_stress_Pa else float("nan")),
        "crushing_events": engine.crushing_events,
        "particle_count": engine.n,
        "first_generation_fragments_m": list(engine.first_generation_fragments),
        "stress_history_Pa": track_stress,
        "displacement_history_m": track_displacement,
        "energy_audit": diagnostics["energy_audit"],
        "replacement_residuals": diagnostics["replacement_residuals"],
    }


def case_c3_dem_cross_check(diameters: tuple[float, ...] = DEFAULT_SIZES, *,
                            seed: int = 21, tolerance: float = 0.05) -> dict[str, Any]:
    """The DEM criterion must fire at the *sampled* strength of each particle.

    This is the mechanism check for C3: the statistical size effect comes from
    the material sampling, but only if the time loop actually breaks a particle
    when its sampled strength is reached.  One compression run per diameter is
    enough to score the trigger accuracy.
    """
    rows = []
    for index, diameter in enumerate(diameters):
        run = run_single_particle(diameter, seed=seed + 7 * index, steps=4000)
        rows.append({
            "diameter_m": float(diameter),
            "assigned_strength_Pa": run["assigned_strength_Pa"],
            "measured_breakage_stress_Pa": run["measured_breakage_stress_Pa"],
            "relative_error": run["strength_relative_error"],
            "crushing_events": run["crushing_events"],
        })
    errors = [r["relative_error"] for r in rows if not math.isnan(r["relative_error"])]
    max_error = max(errors) if errors else float("nan")
    passed = bool(errors) and max_error <= tolerance and all(r["crushing_events"] > 0 for r in rows)
    return {
        "case": "C3-DEM",
        "criterion": f"DEM breakage stress equals the assigned strength within {tolerance:.0%}",
        "rows": rows,
        "max_relative_error": max_error,
        "tolerance": tolerance,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# C4 - fragment size spectrum
# --------------------------------------------------------------------------
def case_c4_fragment_spectrum(n_runs: int = 12, *, seed: int = 31,
                              diameter: float = 2.0e-3,
                              fragmenter: Fragmenter = Fragmenter(k=8, n=0.8,
                                                                 d_min_fraction=2.0e-3),
                              n_operator_draws: int = 80,
                              tolerance_n: float = 0.10,
                              tolerance_r2: float = 0.98) -> dict[str, Any]:
    """Pool the first-generation fragments of ``n_runs`` crushing events.

    Half of the census comes from the DEM replacement operator during a real
    single-particle compression run, half from direct draws of the same
    operator, so the fit is scored on a statistically meaningful sample rather
    than on the eight fragments of one event.
    """
    census: list[float] = []
    dem_runs = 0
    config = default_config(fragmenter=fragmenter)
    for index in range(int(n_runs)):
        run = run_single_particle(diameter, seed=seed + 13 * index, steps=4000,
                                  config=config)
        if run["first_generation_fragments_m"]:
            census.extend(run["first_generation_fragments_m"])
            dem_runs += 1
    rng = np.random.default_rng(seed)
    for _ in range(int(n_operator_draws)):
        parent = Particle(position=np.zeros(3), velocity=np.zeros(3),
                          radius=0.5 * diameter,
                          mass=sphere_mass(diameter, 917.0),
                          strength=DEFAULT_MATERIAL.median_strength(diameter))
        result = fragmenter.split(parent, density=917.0, rng=rng)
        census.extend(float(d) for d in result.diameters)
    fit = fit_gaudin_schuhmann(census)
    n_error = abs(fit["n"] - fragmenter.n) / fragmenter.n
    n_mle_error = abs(fit["n_mle"] - fragmenter.n) / fragmenter.n
    passed = fit["r_squared"] >= tolerance_r2 and n_error <= tolerance_n
    return {
        "case": "C4",
        "criterion": (f"Gaudin-Schuhmann fit R^2 >= {tolerance_r2} and exponent n within "
                      f"{tolerance_n:.0%} of {fragmenter.n}"),
        "dem_runs_used": int(dem_runs),
        "operator_draws": int(n_operator_draws),
        "fragments_in_census": len(census),
        "input_n": float(fragmenter.n),
        "fitted_n": fit["n"],
        "n_relative_error": n_error,
        "fitted_n_mle": fit["n_mle"],
        "n_mle_relative_error": n_mle_error,
        "r_squared": fit["r_squared"],
        "d_max_fit_m": fit["d_max"],
        "diameters_m": sorted(float(v) for v in census)[:400],
        "tolerance_n": tolerance_n,
        "tolerance_r2": tolerance_r2,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# C5 - energy closure of the 1-D bed
# --------------------------------------------------------------------------
def run_bed(*, seed: int = 11, steps: int = 10000, count: int = 24,
            diameter: float = 1.0e-3, size_ratio: float = 2.0,
            platen_velocity: float = -0.05,
            material: WeibullStrength = DEFAULT_MATERIAL,
            config: CrushingConfig | None = None) -> dict[str, Any]:
    """1-D bidisperse (1:2) bed under displacement control."""
    cfg = config or default_config(seed=seed, weibull=material)
    engine = CrushingDEM.column_bed(count, diameter, cfg, seed=seed,
                                    size_ratio=size_ratio)
    engine.platen_velocity = float(platen_velocity)
    initial_diameters = engine.diameters.copy()
    diagnostics = engine.run(int(steps))
    return {"engine": engine, "diagnostics": diagnostics,
            "initial_diameters_m": initial_diameters}


def case_c5_energy_closure(*, seed: int = 11, steps: int = 10000,
                           tolerance: float = 0.02,
                           material: WeibullStrength = DEFAULT_MATERIAL) -> dict[str, Any]:
    """Energy balance ``W_ext = dU + dK + E_diss + E_surface`` over 1e4 steps."""
    run = run_bed(seed=seed, steps=steps, material=material)
    engine, diagnostics = run["engine"], run["diagnostics"]
    audit = diagnostics["energy_audit"]
    residual = audit["residual_relative"]
    passed = residual <= tolerance and engine.crushing_events > 0
    return {
        "case": "C5",
        "criterion": (f"energy residual <= {tolerance:.0%} over {steps} steps "
                      f"with at least one crushing event"),
        "steps": int(steps),
        "crushing_events": engine.crushing_events,
        "particle_count": engine.n,
        "external_work_J": audit["external_work_J"],
        "delta_elastic_J": audit["delta_elastic_J"],
        "delta_kinetic_J": audit["delta_kinetic_J"],
        "dissipation_J": audit["dissipation_J"],
        "surface_energy_J": audit["surface_energy_J"],
        "unfunded_surface_energy_J": audit["unfunded_surface_energy_J"],
        "residual_J": audit["residual_J"],
        "residual_relative": residual,
        "replacement_residuals": diagnostics["replacement_residuals"],
        "history": engine.history,
        "tolerance": tolerance,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# C6 - negative controls
# --------------------------------------------------------------------------
def _bond_config(seed: int, *, bond_sigma0: float, crushing: bool,
                 material: WeibullStrength = DEFAULT_MATERIAL) -> CrushingConfig:
    return default_config(
        seed=seed, weibull=material, crushing_enabled=crushing,
        bond_stiffness=1.0e6,
        bond_weibull=WeibullStrength(d0=1.0e-3, sigma0=bond_sigma0, m=3.0),
        bond_tension_only=False,
    )


def case_c6_negative_controls(*, seed: int = 41, steps: int = 4000) -> dict[str, Any]:
    """Three end-member controls for the crushing mechanism."""
    velocity = -0.05
    # (a) sigma_f -> infinity: crushing off, bond-Weibull fracture retained.
    cfg_a = _bond_config(seed, bond_sigma0=2.0e5, crushing=False)
    run_a = run_bed(seed=seed, steps=steps, platen_velocity=velocity, config=cfg_a)
    engine_a = run_a["engine"]
    diameters_after_a = engine_a.diameters
    unchanged = bool(np.array_equal(diameters_after_a, run_a["initial_diameters_m"]))

    # (b) sigma_f -> 0: every particle must break, C1 must still hold.
    cfg_b = default_config(seed=seed,
                           weibull=WeibullStrength(d0=2.0e-3, sigma0=1.0e-9, m=3.0))
    engine_b = CrushingDEM.column_bed(6, 1.0e-3, cfg_b, seed=seed,
                                      initial_overlap=1.0e-6)
    engine_b.platen_velocity = 0.0
    engine_b.run(5)
    residuals_b = engine_b.ledger.replacement_report()
    all_broken = engine_b.crushing_events >= 6

    # (c) bond fracture off, crushing on: orthogonal paths.
    cfg_c = _bond_config(seed, bond_sigma0=1.0e12, crushing=True)
    run_c = run_bed(seed=seed, steps=steps, platen_velocity=velocity, config=cfg_c)
    engine_c = run_c["engine"]
    final_c = np.sort(engine_c.diameters)
    initial_c = np.sort(np.asarray(run_c["initial_diameters_m"], dtype=float))
    psd_evolved = bool(final_c.size != initial_c.size
                       or not np.allclose(final_c, initial_c))

    passed = (
        engine_a.crushing_events == 0 and unchanged and engine_a.bond_break_events > 0
        and all_broken and residuals_b["max_mass_residual"] <= 1.0e-12
        and residuals_b["max_momentum_residual"] <= 1.0e-12
        and engine_c.crushing_events > 0 and engine_c.bond_break_events == 0
        and engine_c.bond_derived_fragment_clusters() == 0 and psd_evolved)
    return {
        "case": "C6",
        "criterion": ("(a) crushing off: 0 crushing events, PSD bit-identical, bond fracture "
                      "active; (b) sigma_f->0: all particles break, mass/momentum residual "
                      "<=1e-12; (c) bond fracture off: 0 bond-derived clusters, PSD evolves"),
        "a": {
            "crushing_events": engine_a.crushing_events,
            "bond_break_events": engine_a.bond_break_events,
            "psd_bitwise_unchanged": unchanged,
            "particle_count": engine_a.n,
        },
        "b": {
            "crushing_events": engine_b.crushing_events,
            "particle_count": engine_b.n,
            "max_mass_residual": residuals_b["max_mass_residual"],
            "max_momentum_residual": residuals_b["max_momentum_residual"],
        },
        "c": {
            "crushing_events": engine_c.crushing_events,
            "bond_break_events": engine_c.bond_break_events,
            "bond_derived_fragment_clusters": engine_c.bond_derived_fragment_clusters(),
            "psd_evolved": psd_evolved,
            "particle_count": engine_c.n,
        },
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# C7 - regression: dem3d is untouched
# --------------------------------------------------------------------------
#: sha256 of ``dem3d`` positions+velocities after 1000 steps of a 4x3x3 lattice,
#: captured on the unmodified ``main`` checkout (0404ace3) before this node.
DEM3D_GOLDEN_SHA256 = "7008629abdb4435f40445065dc6dec268389b47a3defb444dd5fa0371faee12f"


def dem3d_trajectory_checksum(steps: int = 1000) -> dict[str, Any]:
    """Run the untouched ``dem3d`` engine and checksum the trajectory."""
    import hashlib

    from .dem3d import DEM3DConfig, IceDEM3D

    engine = IceDEM3D(DEM3DConfig(nx=4, ny=3, nz=3, radius=0.025))
    for _ in range(int(steps)):
        engine.step()
    digest = hashlib.sha256(engine.positions.numpy().tobytes()
                            + engine.velocities.numpy().tobytes()).hexdigest()
    return {"steps": int(steps), "sha256": digest,
            "positions_sum": float(engine.positions.sum()),
            "particle_count": int(engine.positions.shape[0])}


def case_c7_regression() -> dict[str, Any]:
    """The unbreakable ``dem3d`` path must stay bit-for-bit identical to main."""
    result = dem3d_trajectory_checksum(1000)
    passed = result["sha256"] == DEM3D_GOLDEN_SHA256
    return {
        "case": "C7",
        "criterion": "dem3d 1000-step trajectory sha256 equals the pre-change main baseline",
        "sha256": result["sha256"],
        "golden_sha256": DEM3D_GOLDEN_SHA256,
        "steps": result["steps"],
        "positions_sum": result["positions_sum"],
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def verify_all(*, quick: bool = False) -> dict[str, Any]:
    """Run every acceptance criterion and return the verdict dictionary.

    ``quick`` trims the sample counts / step counts for the unit-test suite;
    the full run is what the report quotes.
    """
    steps_c5 = 2000 if quick else 10000
    samples_c2 = 200 if quick else 500
    samples_c3 = 500 if quick else 2000
    runs_c4 = 4 if quick else 20
    draws_c4 = 10 if quick else 80
    results = [
        case_c1_replacement(60 if quick else 500),
        case_c2_weibull_inversion(samples_c2),
        case_c3_size_effect(n_samples=samples_c3),
        case_c3_dem_cross_check(),
        case_c4_fragment_spectrum(n_runs=runs_c4, n_operator_draws=draws_c4),
        case_c5_energy_closure(steps=steps_c5),
        case_c6_negative_controls(steps=1500 if quick else 2500),
        case_c7_regression(),
    ]
    by_case = {item["case"]: item for item in results}
    passed = all(item["passed"] for item in results)
    return {
        "node": "P96W2-N03",
        "criterion_set": "TensorDEM single-particle crushing / Weibull size effect",
        "quick": bool(quick),
        "passed": bool(passed),
        "cases": by_case,
    }


def _figure_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "weibull": output_dir / "c2_weibull_survival.png",
        "size_effect": output_dir / "c3_size_effect.png",
        "single": output_dir / "single_particle_compression.png",
        "psd": output_dir / "c4_fragment_psd.png",
        "energy": output_dir / "c5_energy_ledger.png",
    }


def make_figures(output_dir: Path, verdict: dict[str, Any], *,
                 material: WeibullStrength = DEFAULT_MATERIAL) -> dict[str, str]:
    """Render the report figures (matplotlib, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _figure_paths(output_dir)
    figures: dict[str, str] = {}

    # (1) Weibull survival curves + sampled strength histograms.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for index, diameter in enumerate(DEFAULT_SIZES):
        samples = material.sample(2000, diameter, seed=2 + 101 * index)
        sigma = np.linspace(1e4, 5.0 * material.characteristic_strength(diameter), 400)
        ax.plot(sigma, material.survival(sigma, diameter), lw=1.6,
                label=f"analytic d={diameter*1e3:.0f} mm")
        ax.hist(samples, bins=40, density=True, histtype="step", lw=1.0, alpha=0.8,
                label=f"sampled d={diameter*1e3:.0f} mm")
    ax.set_xlabel("particle strength $\\sigma_f$ (Pa)")
    ax.set_ylabel("survival probability / density")
    ax.set_title("C2: weakest-link Weibull strength, volume scaling $(d/d_0)^3$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["weibull"], dpi=150)
    plt.close(fig)
    figures["c2_weibull_survival"] = str(paths["weibull"])

    # (2) size effect
    c3 = verdict["cases"]["C3"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    d = np.array(c3["diameters_m"])
    med = np.array(c3["median_strength_Pa"])
    ax.loglog(d, med, "o-", label="median strength (sampled)")
    d_ref = np.logspace(np.log10(d.min() / 2), np.log10(d.max() * 2), 50)
    ax.loglog(d_ref, material.median_strength(d_ref[0]) * (d_ref / d_ref[0]) ** c3["theory_slope"],
              "k--", lw=1.0, label=f"theory slope $-3/m$ = {c3['theory_slope']:.2f}")
    fitted = med[0] * (d_ref / d[0]) ** c3["fitted_slope_median"]
    ax.loglog(d_ref, fitted, "r:", label=f"fitted slope = {c3['fitted_slope_median']:.3f}")
    ax.set_xlabel("particle diameter $d$ (m)")
    ax.set_ylabel("median crushing strength (Pa)")
    ax.set_title("C3: size effect of the single-particle strength")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(paths["size_effect"], dpi=150)
    plt.close(fig)
    figures["c3_size_effect"] = str(paths["size_effect"])

    # (3) single-particle compression stress-displacement
    run = run_single_particle(2.0e-3, seed=31, steps=3000)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    displacement = np.array(run["displacement_history_m"]) * 1e3
    stress = np.array(run["stress_history_Pa"]) / 1e6
    ax.plot(displacement, stress, lw=1.2, label="nominal stress $F/(\\pi R^2)$")
    ax.axhline(run["assigned_strength_Pa"] / 1e6, ls="--", lw=1.0,
               label="sampled strength $\\sigma_f$")
    ax.axvline(displacement[-1], color="0.6", ls=":", lw=1.0)
    ax.set_xlabel("platen displacement (mm)")
    ax.set_ylabel("nominal particle stress (MPa)")
    ax.set_title("C3 mechanism: single-particle compression, breakage at $\\sigma_f$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["single"], dpi=150)
    plt.close(fig)
    figures["single_particle_compression"] = str(paths["single"])

    # (4) fragment size distribution
    c4 = verdict["cases"]["C4"]
    diameters = np.array(sorted(c4["diameters_m"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    if diameters.size:
        cumulative = np.arange(1, diameters.size + 1) / (diameters.size + 1.0)
        ax.loglog(diameters, cumulative, "o", ms=3, label="DEM fragment census")
        d_ref = np.logspace(np.log10(diameters.min()), np.log10(diameters.max()), 50)
        ax.loglog(d_ref, (d_ref / c4["d_max_fit_m"]) ** c4["fitted_n"], "r-", lw=1.2,
                  label=f"GS fit n={c4['fitted_n']:.3f}, $R^2$={c4['r_squared']:.4f}")
        ax.loglog(d_ref, (d_ref / c4["d_max_fit_m"]) ** c4["input_n"], "k--", lw=1.0,
                  label=f"input n={c4['input_n']:.2f}")
    ax.set_xlabel("fragment diameter (m)")
    ax.set_ylabel("cumulative number fraction")
    ax.set_title("C4: Gaudin-Schuhmann fragment size distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(paths["psd"], dpi=150)
    plt.close(fig)
    figures["c4_fragment_psd"] = str(paths["psd"])

    # (5) energy ledger of the bed
    c5 = verdict["cases"]["C5"]
    history = c5.get("history") or []
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if history:
        time = np.array([row["time_s"] for row in history])
        for key, label in (("external_work_J", "$W_{ext}$"), ("dissipation_J", "$E_{diss}$"),
                           ("surface_energy_J", "$E_{surface}$"),
                           ("elastic_energy_J", "$U_{elastic}$"),
                           ("kinetic_energy_J", "$K$")):
            ax.plot(time * 1e6, np.array([row[key] for row in history]), lw=1.2, label=label)
        residue = np.array([row["external_work_J"] - row["dissipation_J"]
                            - row["surface_energy_J"] - row["elastic_energy_J"]
                            - row["kinetic_energy_J"] for row in history])
        ax.plot(time * 1e6, residue, lw=1.2, ls=":", label="closure residual")
    ax.set_xlabel("time ($\\mu$s)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"C5: 1-D bidisperse bed energy ledger (residual {c5['residual_relative']:.3%})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["energy"], dpi=150)
    plt.close(fig)
    figures["c5_energy_ledger"] = str(paths["energy"])

    return figures


def main(argv: list[str] | None = None) -> int:
    """CLI: run the acceptance cases, write JSON and figures.

    ``--case verify`` is the single command that judges the node red/green:
    it exits with status 0 only when every criterion passes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Weibull-controlled particle crushing acceptance cases (P96W2-N03)")
    parser.add_argument("--case", default="verify",
                        choices=("verify", "quick", "c1", "c2", "c3", "c3dem", "c4", "c5", "c6", "c7"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/crushing"),
                        help="directory for the verdict JSON and figures")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.case in ("verify", "quick"):
        verdict = verify_all(quick=args.case == "quick")
        payload: dict[str, Any] = verdict
    else:
        single = {
            "c1": lambda: case_c1_replacement(),
            "c2": lambda: case_c2_weibull_inversion(),
            "c3": lambda: case_c3_size_effect(),
            "c3dem": lambda: case_c3_dem_cross_check(),
            "c4": lambda: case_c4_fragment_spectrum(),
            "c5": lambda: case_c5_energy_closure(),
            "c6": lambda: case_c6_negative_controls(),
            "c7": lambda: case_c7_regression(),
        }[args.case]()
        verdict = {"node": "P96W2-N03", "passed": single["passed"],
                   "cases": {single["case"]: single}}
        payload = verdict

    if not args.no_figures:
        try:
            figures = make_figures(output, verdict)
            payload["figures"] = figures
        except Exception as error:  # pragma: no cover - plotting is optional
            payload["figures_error"] = f"{type(error).__name__}: {error}"

    verdict_path = output / ("crushing_verdict_quick.json" if args.case == "quick"
                             else "crushing_verdict.json")
    verdict_path.write_text(json.dumps(payload, indent=2, default=float))
    summary = " ".join(f"{name}={'PASS' if item['passed'] else 'FAIL'}"
                       for name, item in payload["cases"].items())
    print(f"P96W2-N03 crushing verdict: {'PASS' if payload['passed'] else 'FAIL'}  [{summary}]")
    print(f"verdict JSON: {verdict_path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
