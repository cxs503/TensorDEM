"""Identifiability, inversion and uncertainty propagation on the forward map.

The forward map is ``u -> y(u)`` with ``u`` in the unit cube (so all
conditioning numbers are scale free) and ``y`` the five observables of
:mod:`tensordem.granular_identifiability.targets`.

Local information content follows the usual linearised-Gaussian reading and
re-uses the ecosystem machinery of
:mod:`tensordem.sensitivity.identifiability`:

``FIM = J^T Sigma^-1 J``, ``Sigma = diag(sigma_j^2)``,

so ``cond(FIM)`` measures how well the *parameter set* is separated by the
observable set at the stated measurement noise, and ``FIM^-1`` is the local
posterior covariance of the calibration.

Two practical points matter for a DEM map:

* the central-difference step must be *large* compared with the structural
  (rearrangement) jitter of a discrete simulation, so the Jacobian uses
  ``h = 0.05`` of the unit cube by default and its stability is re-checked at
  ``2h``;
* the same parameter vector is reached from several places in a study
  (base point, Jacobian probes, line search), so :class:`ForwardMap` memoises
  evaluations -- otherwise the Jacobian would silently cost more than the
  inversion it serves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..sensitivity.identifiability import condition_number, correlation_matrix, fisher_information
from .system import BudgetExceeded, CaseBudget, GranularSystemSpec, TEST_GRANULAR_01
from .targets import TARGET_NAMES, evaluate_targets, target_sigma, target_vector

__all__ = [
    "PARAMETER_SPECS",
    "PARAMETER_NAMES",
    "ForwardMap",
    "unit_from_params",
    "params_from_unit",
    "central_jacobian",
    "conditioning",
    "invert_parameters",
    "propagation_linearity",
    "noise_model",
]

#: (name, lower, upper, symbol, unit) of the three identified contact parameters
PARAMETER_SPECS = (
    ("friction", 0.15, 0.85, "mu", "-"),
    ("restitution", 0.15, 0.85, "e", "-"),
    ("stiffness", 6.0e3, 1.5e4, "k_n", "N/m"),
)
PARAMETER_NAMES = tuple(p[0] for p in PARAMETER_SPECS)


def _bounds(name: str) -> tuple[float, float]:
    for key, lo, hi, _, _ in PARAMETER_SPECS:
        if key == name:
            return lo, hi
    raise KeyError(name)


def unit_from_params(params: dict) -> np.ndarray:
    """Map physical parameters onto the unit cube."""
    u = []
    for name, lo, hi, _, _ in PARAMETER_SPECS:
        p = float(params[name])
        u.append((p - lo) / (hi - lo))
    return np.asarray(u, dtype=float)


def params_from_unit(u) -> dict:
    """Map a (possibly slightly out-of-box) unit-cube point to parameters."""
    u = np.asarray(u, dtype=float).ravel()
    if u.size != len(PARAMETER_SPECS):
        raise ValueError(f"expected {len(PARAMETER_SPECS)} unit-cube coordinates")
    if np.any(u < -1e-9) or np.any(u > 1.0 + 1e-9):
        raise ValueError("unit-cube coordinate outside [0, 1]")
    out = {}
    for (name, lo, hi, _, _), value in zip(PARAMETER_SPECS, u):
        out[name] = float(lo + min(max(value, 0.0), 1.0) * (hi - lo))
    return out


def noise_model(y_ref: np.ndarray) -> np.ndarray:
    """Per-observable measurement standard deviation at the reference vector."""
    return np.asarray(
        [target_sigma(name, float(value)) for name, value in zip(TARGET_NAMES, y_ref)],
        dtype=float,
    )


@dataclass
class ForwardMap:
    """Memoised ``u -> y`` map with budget accounting.

    Parameters
    ----------
    spec:
        Synthetic-system specification (use :meth:`fast` for a cheap variant).
    budget:
        Per-*evaluation* budget; a copy of the template is used for every call
        so that one expensive evaluation cannot consume another's allowance.
    cache_decimals:
        Cache key rounding.  ``12`` is exact enough for a float64 vector and
        still collapses the duplicated probes of a least-squares Jacobian.
    """

    spec: GranularSystemSpec = TEST_GRANULAR_01
    budget: CaseBudget | None = None
    cache_decimals: int = 12

    def __post_init__(self) -> None:
        self._cache: dict[tuple, dict] = {}
        self.calls = 0
        self.cache_hits = 0
        self.total_steps = 0
        self.total_wall_seconds = 0.0
        self.log: list[dict] = []

    # ------------------------------------------------------------------
    def __call__(self, u) -> np.ndarray:
        result = self.result(u)
        return target_vector(result)

    def result(self, u) -> dict:
        key = tuple(np.round(np.asarray(u, dtype=float).ravel(), self.cache_decimals))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        params = params_from_unit(u)
        template = self.budget or CaseBudget()
        budget = CaseBudget(
            max_particles=template.max_particles,
            max_steps=template.max_steps,
            max_wall_seconds=template.max_wall_seconds,
        ).start()
        try:
            result = evaluate_targets(params, spec=self.spec, budget=budget)
        except BudgetExceeded as exc:  # re-raise with the offending point
            raise BudgetExceeded(f"forward evaluation at u={np.round(key, 4)}: {exc}") from exc
        usage = result["usage"]
        self.calls += 1
        self.total_steps += int(usage["steps"])
        self.total_wall_seconds += float(usage["wall_seconds"])
        self.log.append({"u": list(key), "params": params, "targets": result["targets"],
                         "steps": usage["steps"], "wall_seconds": usage["wall_seconds"]})
        if any(not np.isfinite(value) for value in result["targets"].values()):
            raise FloatingPointError(f"non-finite observable at u={np.round(key, 4)}: "
                                     f"{result['targets']}")
        self._cache[key] = result
        return result

    @property
    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "total_steps": self.total_steps,
            "total_wall_seconds": round(self.total_wall_seconds, 3),
        }


def central_jacobian(forward, u0, *, h: float = 0.05) -> np.ndarray:
    """Central-difference Jacobian ``dy/du`` of shape ``(n_targets, n_params)``.

    Probes are kept inside the unit cube by mirroring the step onto the
    opposite side (the map is only defined on the box), which preserves the
    O(h^2) accuracy of the central difference at the faces.
    """
    u0 = np.asarray(u0, dtype=float).ravel()
    if np.any(u0 < 0.0) or np.any(u0 > 1.0):
        raise ValueError("u0 must lie in the unit cube")
    if h <= 0.0:
        raise ValueError("h must be positive")
    base = np.asarray(forward(u0), dtype=float)
    J = np.empty((base.size, u0.size))
    for i in range(u0.size):
        step = min(h, 0.5)
        plus = u0.copy()
        minus = u0.copy()
        plus[i] = min(1.0, u0[i] + step)
        minus[i] = max(0.0, u0[i] - step)
        y_plus = np.asarray(forward(plus), dtype=float)
        y_minus = np.asarray(forward(minus), dtype=float)
        J[:, i] = (y_plus - y_minus) / (plus[i] - minus[i])
    return J


def conditioning(J: np.ndarray, sigma: np.ndarray, *, names=PARAMETER_NAMES) -> dict:
    """Fisher information, its spectrum, condition number and correlations.

    The correlation matrix is built from the pseudo-inverse of the FIM, so a
    rank-deficient information matrix (fewer informative observables than
    parameters) yields a readable correlation structure and an explicit
    ``rank``/``deficient`` flag instead of infinities.
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    if np.any(sigma <= 0.0):
        raise ValueError("all measurement uncertainties must be positive")
    fim = fisher_information(J, sigma)
    eig = np.linalg.eigvalsh(fim)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cond = float(condition_number(fim))
    cov = np.linalg.pinv(fim, rcond=1e-12)
    scale = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        corr = np.clip(cov / np.outer(scale, scale), -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    rank = int(np.linalg.matrix_rank(fim, tol=1e-9 * max(1.0, float(np.max(eig)))))
    return {
        "fim": fim,
        "condition_number": cond,
        "rank": rank,
        "deficient": bool(rank < len(names)),
        "eigenvalues": [float(v) for v in eig],
        "smallest_eigenvalue": float(np.min(eig)),
        "largest_eigenvalue": float(np.max(eig)),
        "correlation": corr.tolist(),
        "names": list(names),
        "jacobian": J.tolist(),
        "sigma": sigma.tolist(),
        "info_per_parameter": {name: float(fim[i, i]) for i, name in enumerate(names)},
    }


def invert_parameters(
    forward,
    y_obs,
    sigma,
    u0,
    *,
    max_nfev: int = 60,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    diff_step: float = 0.02,
) -> dict:
    """Weighted least-squares inversion of synthetic observations.

    ``residual(u) = (y(u) - y_obs) / sigma`` is minimised on the unit cube with
    a bounded trust-region method; the parameter-space Jacobian of the solver
    is finite-differenced with ``diff_step`` of the unit cube (DEM maps are
    structurally noisy below ~1 %, so a tighter step would only fit jitter).
    """
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    if y_obs.size != sigma.size:
        raise ValueError("y_obs and sigma must have the same length")

    def residual(u):
        return (np.asarray(forward(u), dtype=float) - y_obs) / sigma

    u0 = np.clip(np.asarray(u0, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    sol = least_squares(
        residual,
        u0,
        bounds=(0.0, 1.0),
        x_scale=1.0,
        diff_step=diff_step,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
    )
    y_fit = np.asarray(forward(sol.x), dtype=float)
    return {
        "unit": sol.x.tolist(),
        "params": params_from_unit(sol.x),
        "y_fit": dict(zip(TARGET_NAMES, y_fit.tolist())),
        "chi2": float(np.sum(((y_fit - y_obs) / sigma) ** 2)),
        "dof": int(y_obs.size - sol.x.size),
        "nfev": int(sol.nfev),
        "status": int(sol.status),
        "cost": float(sol.cost),
        "residuals": ((y_fit - y_obs) / sigma).tolist(),
    }


def recovery(estimate: dict, truth: dict) -> dict:
    """Relative recovery error of every parameter, ``|est - true| / true``."""
    out = {}
    for name in PARAMETER_NAMES:
        est = float(estimate[name])
        ref = float(truth[name])
        out[name] = {
            "estimate": est,
            "truth": ref,
            "rel_error": abs(est - ref) / ref,
        }
    out["max_rel_error"] = max(v["rel_error"] for k, v in out.items() if isinstance(v, dict))
    return out


def propagation_linearity(
    forward,
    truth_params: dict,
    *,
    factors=(-0.10, -0.05, 0.0, 0.05, 0.10),
    min_relative_response: float = 1.0e-3,
    sigma=None,
) -> dict:
    """Uncertainty propagation: is the +-10 % response linear and resolvable?

    For each parameter the physical value is perturbed by the given relative
    factors (clipped to its admissible box) and the whole observable vector is
    re-evaluated.  Per (parameter, observable) a straight line is fitted to
    ``y`` versus the relative perturbation and ``R^2`` is reported.

    An observable counts as *resolvable* for a parameter only when the peak
    response exceeds the measurement noise of that observable (``sigma``) and
    ``min_relative_response``.  Below that gate the fit is fitting jitter, so
    ``R^2`` is reported for transparency but the pair is excluded from the
    linearity criterion -- which is stated in the report rather than hidden.
    Sensitivities ``dy/dp`` come from the same scan and give the propagated
    uncertainty ``sigma_y = |dy/dp| sigma_p``.
    """
    truth_params = {k: float(v) for k, v in truth_params.items()}
    base = np.asarray(forward(unit_from_params(truth_params)), dtype=float)
    rows: dict[str, dict] = {}
    worst = {"r2": 1.0}
    for name in PARAMETER_NAMES:
        lo, hi = _bounds(name)
        p0 = truth_params[name]
        used = []
        y = []
        for f in factors:
            p = min(hi, max(lo, p0 * (1.0 + f)))
            if not used or abs(p - used[-1][1]) > 0.0:
                used.append((f, p))
                y.append(
                    np.asarray(forward(unit_from_params({**truth_params, name: p})), dtype=float)
                )
        frac = np.asarray([(p - p0) / p0 for _, p in used])
        Y = np.asarray(y)
        entries = {}
        for j, target in enumerate(TARGET_NAMES):
            yj = Y[:, j]
            rel = float(np.max(np.abs(yj - base[j])) / max(abs(base[j]), 1e-30))
            noise = 0.0
            if sigma is not None and target in sigma:
                noise = float(sigma[target]) / max(abs(base[j]), 1e-30)
            gate = max(float(min_relative_response), noise)
            r2 = None
            slope = 0.0
            if np.ptp(frac) > 0.0:
                slope, intercept = np.polyfit(frac, yj, 1)
                pred = slope * frac + intercept
                ss_res = float(np.sum((yj - pred) ** 2))
                ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None
            resolvable = bool(rel >= gate and r2 is not None)
            entries[target] = {
                "sensitive": resolvable,
                "resolvable": resolvable,
                "relative_response": rel,
                "noise_gate": gate,
                "r2": r2,
                "sensitivity_per_unit_relative": float(slope),
            }
            if resolvable and r2 is not None and r2 < worst.get("r2", 1.0):
                worst = {"r2": r2, "parameter": name, "target": target}
        rows[name] = {
            "factors": [f for f, _ in used],
            "values": [p for _, p in used],
            "observables": entries,
            "y": Y.tolist(),
        }
    return {
        "base": dict(zip(TARGET_NAMES, base.tolist())),
        "parameters": rows,
        "worst": worst,
        "excluded": [
            {"parameter": name, "target": target,
             "relative_response": entry["relative_response"],
             "noise_gate": entry["noise_gate"], "r2_if_fitted": entry["r2"]}
            for name, row in rows.items()
            for target, entry in row["observables"].items()
            if not entry["resolvable"]
        ],
    }
