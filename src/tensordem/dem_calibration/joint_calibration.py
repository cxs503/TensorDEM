"""Joint three-benchmark Bayesian calibration of DEM contact parameters.

Node P284W3-N08.  The calibrated vector is

    theta = [k_n, k_t, mu, e, sigma_bond]

and the experimental data are the responses of three macroscopic tests
(repose / direct shear / bonded uniaxial compression, see
:mod:`tensordem.dem_calibration.joint_benchmarks`).

Objective (campaign definition, sum of squared relative deviations grouped per
benchmark)::

    J(theta) = sum_b w_b * ||y_sim,b(theta) - y_exp,b||^2 / ||y_exp,b||^2

Search: Gaussian-process surrogate (Matern-5/2, anisotropic length scales) with
expected improvement, reusing :mod:`tensordem.dem_calibration.inverse` and
:mod:`tensordem.dem_calibration.surrogate`.  Uncertainty: random-walk Metropolis
on the surrogate response surface (Kennedy & O'Hagan emulator-error-inflated
likelihood), then a posterior-predictive coverage check of the measured data.

The experiment is a *twin* experiment (synthetic truth): the targets are DEM
responses at a declared truth parameter set.  What this validates is the
inversion machinery -- identifiability, convergence, posterior calibration --
not the ability of the forward model to reproduce a real soil.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .contact_engine import PARAMETER_BOUNDS, PARAMETER_NAMES, ContactParams, GrainMaterial
from .inverse import ParameterSpace, bayesian_optimize, posterior_sample
from .joint_benchmarks import ReposeCase, ShearCase, UniaxialCase, run_repose, run_shear, run_uniaxial
from .surrogate import GaussianProcess

__all__ = [
    "PARAMETER_SPACE",
    "THRESHOLDS",
    "BENCHMARK_WEIGHTS",
    "TRUTH",
    "ExperimentSpec",
    "make_targets",
    "forward_features",
    "objective_from_features",
    "posterior_predictive_coverage",
    "calibrate_material",
]

PARAMETER_SPACE = ParameterSpace(
    names=PARAMETER_NAMES,
    lower=tuple(PARAMETER_BOUNDS[n][0] for n in PARAMETER_NAMES),
    upper=tuple(PARAMETER_BOUNDS[n][1] for n in PARAMETER_NAMES),
)

#: Acceptance thresholds of node P284W3-N08 (criteria (a)-(e) of the node card).
THRESHOLDS = {
    "repose_error_deg": 3.0,        # (a) |theta_sim - theta_target| < 3 deg
    "shear_rmse_rel": 0.05,         # (b) curve RMSE / peak < 5 %
    "peak_stress_rel": 0.05,        # (c) peak stress error < 5 %
    "objective_drop": 0.90,         # (d) J drop > 90 % within 50 evaluations
    "max_evaluations": 50,          # (d) budget
    "coverage": 0.90,               # (e) posterior predictive coverage >= 90 %
    "credible_level": 0.90,         # (e) two-sided predictive band
}

BENCHMARK_WEIGHTS = {"repose": 0.25, "shear": 0.40, "uniaxial": 0.35}

#: Objective value assigned to a parameter set whose forward run fails to
#: produce a measurable response (the search must be able to reject it without
#: poisoning the surrogate with NaN).
FAILURE_PENALTY = 1.0e3

#: Shear-curve checkpoints (engineering shear strain) and uniaxial checkpoints
#: (axial strain) used as the response features.
SHEAR_STRAIN_POINTS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
UNIAXIAL_STRAIN_POINTS = (0.005, 0.010, 0.015, 0.020, 0.030, 0.050)

#: Declared synthetic truth per material class (the "experiment" of the twin
#: study).  Chosen inside the calibration box, not on its boundary.
TRUTH: dict[str, ContactParams] = {
    "sand": ContactParams(kn=800.0, kt=380.0, mu=0.52, restitution=0.35, sigma_bond=8.0e3),
    "glass_beads": ContactParams(kn=1500.0, kt=700.0, mu=0.30, restitution=0.60, sigma_bond=3.0e3),
    "cemented_gravel": ContactParams(kn=1200.0, kt=950.0, mu=0.65, restitution=0.20, sigma_bond=1.5e4),
}


@dataclass(frozen=True)
class ExperimentSpec:
    """Geometry/protocol of the three tests for one material class."""

    name: str
    material: GrainMaterial = field(default_factory=GrainMaterial)
    repose: ReposeCase = None
    shear: ShearCase = None
    uniaxial: UniaxialCase = None

    def __post_init__(self) -> None:
        mat = self.material
        # Particle size is part of each *test* definition (it sets the specimen
        # resolution relative to the device); the calibrated contact parameters
        # are shared.  The 6 mm repose grains give a dense, settled deposit with
        # a monotonically increasing theta(mu); 4 mm grains give a flatter,
        # non-monotone deposit at the same settling time.
        repose_mat = GrainMaterial(radius=0.006, density=mat.density,
                                   mu_rolling=mat.mu_rolling, gravity=mat.gravity)
        if self.repose is None:
            object.__setattr__(self, "repose", ReposeCase(
                material=repose_mat, block_width=7, block_layers=4,
                settle_steps=1200, seed=5))
        if self.shear is None:
            object.__setattr__(self, "shear", ShearCase(
                material=mat, box_width=0.060, box_depth=0.040, fill_height=0.030,
                shear_strain_target=0.10, shear_velocity=0.050,
                settle_steps=200, consol_steps=200, consolidation_strain=0.006,
                seed=7))
        if self.uniaxial is None:
            object.__setattr__(self, "uniaxial", UniaxialCase(
                material=mat, strain_target=0.10, loading_velocity=0.040,
                settle_steps=150, seed=3))

    def with_validation_seeds(self, offset: int = 100) -> "ExperimentSpec":
        """Same experiments, independent random packing/loading seeds."""
        from dataclasses import replace as _replace
        return ExperimentSpec(
            name=self.name, material=self.material,
            repose=_replace(self.repose, seed=self.repose.seed + offset),
            shear=_replace(self.shear, seed=self.shear.seed + offset),
            uniaxial=_replace(self.uniaxial, seed=self.uniaxial.seed + offset),
        )


def _interp(x: np.ndarray, y: np.ndarray, xp) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return np.interp(np.asarray(xp, dtype=float), x[order], y[order])


def forward_features(spec: ExperimentSpec, params: ContactParams) -> dict:
    """Run the three benchmarks and return the raw macroscopic responses."""
    t0 = time.time()
    rep = run_repose(spec.repose, params)
    shr = run_shear(spec.shear, params)
    uni = run_uniaxial(spec.uniaxial, params)
    return {
        "repose": rep,
        "shear": shr,
        "uniaxial": uni,
        "wall_seconds": time.time() - t0,
    }


def feature_vector(raw: dict) -> dict[str, float]:
    """Flatten the raw responses into the named scalar features used by the
    objective, the emulators and the predictive coverage check."""
    rep, shr, uni = raw["repose"], raw["shear"], raw["uniaxial"]
    feats: dict[str, float] = {
        "repose.theta": float(rep["theta_deg"]),
        "repose.phi": float(rep["phi_solid"]),
    }
    sigma_n = max(float(shr["sigma_n_measured"]), 1e-12)
    for g in SHEAR_STRAIN_POINTS:
        tau = float(_interp(shr["strain"], shr["tau_smooth"], g))
        feats[f"shear.tau@{g:.3f}"] = tau / sigma_n
    feats["shear.peak_over_residual"] = float(
        shr["peak_tau"] / max(shr["residual_tau"], 1e-12))
    peak = float(uni["peak_stress"])
    for e in UNIAXIAL_STRAIN_POINTS:
        feats[f"uniaxial.sigma@{e:.3f}"] = float(
            _interp(uni["strain"], uni["stress_smooth"], e) / max(peak, 1e-12))
    feats["uniaxial.peak"] = peak
    feats["uniaxial.modulus"] = float(uni["modulus"])
    return feats


def _group(name: str) -> str:
    return name.split(".", 1)[0]


def make_targets(spec: ExperimentSpec, truth: ContactParams) -> dict:
    """Twin-experiment targets: forward responses at the declared truth."""
    raw = forward_features(spec, truth)
    return {"features": feature_vector(raw), "raw": raw}


def objective_from_features(sim: dict[str, float], exp: dict[str, float],
                            weights: dict[str, float] | None = None) -> dict:
    """Grouped relative least-squares objective of the campaign node."""
    weights = weights or BENCHMARK_WEIGHTS
    parts: dict[str, float] = {}
    for group in weights:
        names = [n for n in exp if _group(n) == group]
        num = sum((sim[n] - exp[n]) ** 2 for n in names)
        den = sum(exp[n] ** 2 for n in names)
        parts[group] = num / den if den > 0 else float("inf")
    total = sum(weights[g] * parts[g] for g in weights)
    return {"total": float(total), "parts": parts}


def posterior_predictive_coverage(x_unit: np.ndarray, responses: dict[str, np.ndarray],
                                  targets: dict[str, float], sigmas: dict[str, float],
                                  samples_unit: np.ndarray, *,
                                  level: float = 0.90, seed: int = 0) -> dict:
    """Posterior predictive coverage of the measured data.

    Each feature gets its own GP emulator on the visited design; the predictive
    distribution at a parameter draw is ``N(m_i(theta), s_i(theta)^2 +
    sigma_i^2)``.  We evaluate it over the retained MCMC draws, take the central
    ``level`` band and count the measured values inside it.
    """
    names = [n for n in responses if n in targets]
    if not names:
        raise ValueError("no overlapping responses and targets")
    lo_q = 50.0 * (1.0 - level)
    hi_q = 100.0 - lo_q
    per_feature = {}
    inside = 0
    total = 0
    for name in names:
        y = np.asarray(responses[name], dtype=float)
        keep = np.isfinite(y)
        if int(keep.sum()) < 6:
            continue
        gp = GaussianProcess().fit(x_unit[keep], y[keep], optimise=True, seed=seed)
        mean, std = gp.predict(samples_unit)
        sigma = float(sigmas.get(name, 0.0))
        draw = np.random.default_rng(seed).normal(mean, np.hypot(std, sigma))
        lo, hi = np.percentile(draw, [lo_q, hi_q])
        hit = bool(lo <= targets[name] <= hi)
        inside += int(hit)
        total += 1
        per_feature[name] = {"target": float(targets[name]), "lo": float(lo),
                             "hi": float(hi), "covered": hit}
    return {"coverage": inside / total if total else float("nan"),
            "n_features": total, "level": level, "per_feature": per_feature}


def _curve_rmse_rel(sim: np.ndarray, exp: np.ndarray, normaliser: float) -> float:
    sim = np.asarray(sim, float)
    exp = np.asarray(exp, float)
    return float(math.sqrt(np.mean((sim - exp) ** 2)) / max(abs(normaliser), 1e-12))


def calibrate_material(name: str, *, spec: ExperimentSpec | None = None,
                       truth: ContactParams | None = None,
                       n_init: int = 12, n_iter: int = 38, seed: int = 0,
                       n_posterior: int = 3000, n_predictive: int = 600,
                       log: list | None = None) -> dict:
    """Full calibration of one material class (twin experiment)."""
    truth = truth or TRUTH[name]
    spec = spec or ExperimentSpec(name=name)
    if n_init + n_iter > THRESHOLDS["max_evaluations"]:
        raise ValueError("total evaluations must stay within 50")

    target = make_targets(spec, truth)
    exp_feat = target["features"]
    cache: dict[tuple, dict] = {}

    def evaluate(params: ContactParams) -> dict:
        key = tuple(round(float(params.as_dict()[n]), 10) for n in PARAMETER_NAMES)
        if key not in cache:
            raw = forward_features(spec, params)
            cache[key] = {"raw": raw, "features": feature_vector(raw)}
            if log is not None:
                log.append({"n": len(cache), "params": params.as_dict(),
                            "wall_seconds": raw["wall_seconds"]})
        return cache[key]

    def objective(values: dict) -> float:
        sim = evaluate(ContactParams.from_mapping(values))["features"]
        if not np.all(np.isfinite(list(sim.values()))):
            # a benchmark that cannot be measured (e.g. a deposit without a
            # flank) is a hard failure of the parameter set, not a NaN that
            # would poison the GP training set
            return FAILURE_PENALTY
        value = objective_from_features(sim, exp_feat)["total"]
        return value if math.isfinite(value) else FAILURE_PENALTY

    t0 = time.time()
    bo = bayesian_optimize(objective, PARAMETER_SPACE, n_init=n_init, n_iter=n_iter,
                           seed=seed)
    bo_seconds = time.time() - t0

    table = bo["table"]
    J_values = np.array([row["objective"] for row in table])
    finite_init = J_values[:n_init][J_values[:n_init] < FAILURE_PENALTY]
    # reference level of the drop criterion: the *median* of the initial
    # Latin-hypercube design (a max would be inflated by a single failed point
    # and would make the >90 % drop criterion trivially satisfiable)
    J_init = float(np.median(finite_init)) if finite_init.size else float("nan")
    J_init_max = float(J_values[:n_init].max())
    J_best = float(J_values.min())
    drop = 1.0 - J_best / J_init if np.isfinite(J_init) and J_init > 0 else float("nan")

    # --- posterior on the response surface ------------------------------- #
    x_unit = np.array([row["unit"] for row in table])
    rows = [cache[tuple(round(float(row["params"][k]), 10) for k in PARAMETER_NAMES)]
            ["features"] for row in table]
    ok = np.array([all(np.isfinite([r[n] for n in exp_feat])) for r in rows])
    if int(ok.sum()) < 8:
        raise RuntimeError("fewer than 8 measurable designs: posterior is undefined")
    x_unit = x_unit[ok]
    responses = {n: np.array([r[n] for r, keep in zip(rows, ok) if keep])
                 for n in exp_feat}
    sigmas = {n: max(0.01 * abs(exp_feat[n]), 1e-6) for n in exp_feat}
    post = posterior_sample(x_unit, responses, exp_feat, sigmas, PARAMETER_SPACE,
                            n_samples=n_posterior, burn_in=800, thin=3, seed=seed)
    sub = post["samples_unit"][:: max(len(post["samples_unit"]) // n_predictive, 1)]
    coverage = posterior_predictive_coverage(x_unit, responses, exp_feat, sigmas, sub,
                                            level=THRESHOLDS["credible_level"],
                                            seed=seed)

    # --- verification on independent seeds ------------------------------- #
    best = ContactParams.from_mapping(bo["best_params"])
    val_spec = spec.with_validation_seeds(100)
    val = forward_features(val_spec, best)
    val_feat = feature_vector(val)

    theta_target = float(exp_feat["repose.theta"])
    theta_val = float(val_feat["repose.theta"])
    repose_err = abs(theta_val - theta_target)

    exp_tau = np.array([exp_feat[f"shear.tau@{g:.3f}"] for g in SHEAR_STRAIN_POINTS])
    sim_tau = np.array([val_feat[f"shear.tau@{g:.3f}"] for g in SHEAR_STRAIN_POINTS])
    shear_rmse = _curve_rmse_rel(sim_tau, exp_tau, exp_feat["shear.tau@0.090"])

    peak_exp = float(exp_feat["uniaxial.peak"])
    peak_sim = float(val_feat["uniaxial.peak"])
    peak_err = abs(peak_sim - peak_exp) / max(peak_exp, 1e-12)

    verdicts = {
        "a_repose": {"value": repose_err, "threshold": THRESHOLDS["repose_error_deg"],
                     "pass": bool(repose_err < THRESHOLDS["repose_error_deg"])},
        "b_shear_rmse": {"value": shear_rmse, "threshold": THRESHOLDS["shear_rmse_rel"],
                         "pass": bool(shear_rmse < THRESHOLDS["shear_rmse_rel"])},
        "c_peak_stress": {"value": peak_err, "threshold": THRESHOLDS["peak_stress_rel"],
                          "pass": bool(peak_err < THRESHOLDS["peak_stress_rel"])},
        "d_objective_drop": {"value": drop, "threshold": THRESHOLDS["objective_drop"],
                             "pass": bool(drop > THRESHOLDS["objective_drop"]),
                             "evaluations": len(table), "J_initial": J_init,
                             "J_initial_max": J_init_max, "J_best": J_best},
        "e_coverage": {"value": coverage["coverage"], "threshold": THRESHOLDS["coverage"],
                       "pass": bool(coverage["coverage"] >= THRESHOLDS["coverage"]),
                       "n_features": coverage["n_features"]},
    }

    return {
        "material": name,
        "truth": truth.as_dict(),
        "targets": {k: float(v) for k, v in exp_feat.items()},
        "target_raw": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items() if kk != "positions"}
                       for k, v in target["raw"].items() if isinstance(v, dict)},
        "best_params": bo["best_params"],
        "best_objective": J_best,
        "objective_drop": drop,
        "evaluations": len(table),
        "design": [{"iteration": row["iteration"], "unit": row["unit"],
                    "params": row["params"], "objective": row["objective"]}
                   for row in table],
        "posterior": post["summary"],
        "posterior_acceptance_rate": post["acceptance_rate"],
        "coverage": coverage,
        "verification": {
            "features": val_feat,
            "theta_sim_deg": theta_val,
            "theta_target_deg": theta_target,
            "repose_error_deg": repose_err,
            "shear_curve_sim": sim_tau.tolist(),
            "shear_curve_target": exp_tau.tolist(),
            "shear_rmse_rel": shear_rmse,
            "peak_stress_sim": peak_sim,
            "peak_stress_target": peak_exp,
            "peak_stress_rel_err": peak_err,
            "shear_sigma_n": float(val["shear"]["sigma_n_measured"]),
            "shear_escaped": bool(val["shear"]["escaped"]),
            "uniaxial_intact_fraction": float(val["uniaxial"]["intact_fraction"]),
        },
        "verdicts": verdicts,
        "all_pass": all(v["pass"] for v in verdicts.values()),
        "wall_seconds": {"bayesopt": bo_seconds, "total": time.time() - t0},
        "n_unique_evaluations": len(cache),
    }