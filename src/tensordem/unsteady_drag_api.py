"""API handler for ``POST /api/v1/dem/unsteady_drag``.

The route exposes the three-term unsteady hydrodynamic ledger
(quasi-steady + added mass + Basset history, plus the Froude-Krylov pressure
gradient term) with a *selectable scenario*, so that one endpoint serves both
production queries and the analytic-anchor self checks:

``mode = "oscillatory"``
    fixed sphere in ``u_f = U0 cos(omega t)`` -> amplitude / lag phase vs the
    Landau & Lifshitz section 24 exact result, plus the RMS share of each ledger
    term.
``mode = "sudden_acceleration"``
    constant relative acceleration in quiescent fluid -> added-mass coefficient
    ``A = F_am/(rho_f V du/dt)``.
``mode = "steady_uniform"``
    constant slip (optionally with a ramp) -> negative control: the unsteady
    terms must vanish, the total must equal Schiller-Naumann.
``mode = "impulsive_start"``
    impulsive start of steady slip -> force vs the exact Stokes
    ``6 pi mu a U (1 + a/sqrt(pi nu t))`` decay (pins the Basset prefactor).
``mode = "recursive_benchmark"``
    O(N^2) reference vs O(N) recursion: L2 error and CPU-time speedup.
``mode = "dilute_limit"``
    multi-sphere cloud at decreasing solid fraction -> per-sphere force must
    converge to the single-sphere value.

Like the other TensorDEM endpoint handlers (e.g.
``tensordem.acoustics.api.handle_acoustic_source``) this is a pure function
``payload -> dict`` so the FastAPI route is a two-liner and invalid input
surfaces as ``ValueError`` (mapped to HTTP 422 by the route layer).

Every response carries an ``evidence`` block (grade + analytic anchors) and a
``criteria`` block (measured value, target, pass/fail) so that the numerical
claim is machine-checkable downstream.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from . import unsteady_drag as ud

__all__ = ["UNSTEADY_DRAG_ENDPOINT", "handle_unsteady_drag", "MODES"]

UNSTEADY_DRAG_ENDPOINT = "/api/v1/dem/unsteady_drag"
MODES = (
    "oscillatory",
    "sudden_acceleration",
    "steady_uniform",
    "impulsive_start",
    "recursive_benchmark",
    "dilute_limit",
)


# --------------------------------------------------------------------------- #
# payload helpers
# --------------------------------------------------------------------------- #
def _positive(payload: Dict[str, Any], key: str, default: float, where: str) -> float:
    value = float(payload.get(key, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where}.{key} must be a positive finite number")
    return value


def _sub(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _criterion(value: float, target: float, kind: str = "max") -> Dict[str, Any]:
    ok = abs(value) <= target if kind == "max" else abs(value) >= target
    return {"value": float(value), "target": float(target),
            "criterion": f"{'<=' if kind == 'max' else '>='} {target}", "pass": bool(ok)}


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def _fluid(payload: Dict[str, Any]) -> Dict[str, float]:
    fluid = _sub(payload, "fluid")
    rho_f = _positive(fluid, "rho_f", 1000.0, "fluid")
    mu = _positive(fluid, "mu", 1.0e-3, "fluid")
    return {"rho_f": rho_f, "mu": mu}


def _particle(payload: Dict[str, Any]) -> Dict[str, float]:
    particle = _sub(payload, "particle")
    d = _positive(particle, "diameter", 2.0e-3, "particle")
    rho_p = float(particle.get("rho_p", 7800.0))
    if not math.isfinite(rho_p) or rho_p <= 0.0:
        raise ValueError("particle.rho_p must be positive")
    return {"diameter": d, "rho_p": rho_p}


def _time_grid(payload: Dict[str, Any], frequency: Optional[float],
               default_periods: float = 40.0,
               default_samples: int = 400) -> Dict[str, float]:
    """Resolve (dt, t_end) either from physical times or from the period."""
    dt = payload.get("dt")
    t_end = payload.get("t_end")
    if frequency is not None:
        n_periods = float(payload.get("n_periods", default_periods))
        samples = int(payload.get("samples_per_period", default_samples))
        if n_periods <= 0 or samples < 8:
            raise ValueError("n_periods must be > 0 and samples_per_period >= 8")
        period = 2.0 * math.pi / frequency
        if dt is None:
            dt = period / samples
        if t_end is None:
            t_end = n_periods * period
    if dt is None or t_end is None:
        raise ValueError("either (dt, t_end) or a frequency must be given")
    dt = float(dt)
    t_end = float(t_end)
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(t_end) and t_end > 0):
        raise ValueError("dt and t_end must be positive finite numbers")
    if t_end / dt > 5.0e6:
        raise ValueError("grid too large: t_end/dt > 5e6")
    return {"dt": dt, "t_end": t_end}


def _mode_oscillatory(payload: Dict[str, Any]) -> Dict[str, Any]:
    fluid, particle = _fluid(payload), _particle(payload)
    omega = float(payload.get("omega", 0.0))
    if omega <= 0.0:
        eps = float(payload.get("epsilon", 1.0))       # eps = omega a^2/nu
        nu = fluid["mu"] / fluid["rho_f"]
        omega = eps * nu / (particle["diameter"] / 2.0) ** 2
    U0 = float(payload.get("U0", 1.0e-5))
    grid = _time_grid(payload, omega)
    out = ud.run_oscillatory_flow_simulation(
        fluid["rho_f"], particle["diameter"], fluid["mu"], particle["rho_p"],
        U0, omega, grid["t_end"], grid["dt"],
        n_exp=int(payload.get("n_exp", 16)),
        qs_factor=float(payload.get("qs_factor", 1.0)),
    )
    criteria = {
        "amplitude_rel_error<0.10": _criterion(out["amplitude_rel_error"], 0.10),
        "phase_error_deg<5": _criterion(out["phase_error_deg"], 5.0),
    }
    return {
        "scenario": "oscillatory",
        "inputs": {**fluid, **particle, "U0": U0, "omega": omega, **grid},
        "alpha": out["alpha"],
        "Re_amplitude": out["Re_amplitude"],
        "amplitude_measured": out["amplitude_measured"],
        "amplitude_analytical": out["amplitude_analytical"],
        "phase_measured_deg": out["phase_measured_deg"],
        "phase_analytical_deg": out["phase_analytical_deg"],
        "component_rms": out["component_rms"],
        "history_share": out["history_share"],
        "criteria": criteria,
        "evidence": {
            "grade": "REAL",
            "anchors": ["Landau & Lifshitz section 24 (exact unsteady Stokes)",
                        "Stokes quasi-steady 6 pi mu a u"],
            "note": "one-way coupling: the flow field is prescribed",
        },
    }


def _mode_sudden_acceleration(payload: Dict[str, Any]) -> Dict[str, Any]:
    fluid, particle = _fluid(payload), _particle(payload)
    du_dt = float(payload.get("du_dt", 1.0))
    if du_dt == 0.0:
        raise ValueError("du_dt must be non-zero")
    grid = _time_grid(payload, None)
    out = ud.run_sudden_acceleration(fluid["rho_f"], particle["diameter"],
                                     fluid["mu"], du_dt, grid["t_end"], grid["dt"],
                                     rho_p=particle["rho_p"])
    # independent high-frequency asymptote of the exact unsteady Stokes solution
    nu = fluid["mu"] / fluid["rho_f"]
    a = particle["diameter"] / 2.0
    alpha = float(payload.get("alpha_asymptote", 1.0e3))
    omega = 2.0 * nu * alpha ** 2 / a ** 2
    lam = ud.stokes_oscillatory_parameter(omega, particle["diameter"],
                                          fluid["mu"], fluid["rho_f"])
    V = ud.sphere_volume(particle["diameter"])
    F_hat = ud.landau_lifshitz_oscillatory_drag_amplitude_phase(
        omega, particle["diameter"], fluid["mu"], fluid["rho_f"], 1.0)[0]
    # total force per unit rho_f V omega (acceleration amplitude omega for U0=1)
    A_total_asymptote = F_hat / (fluid["rho_f"] * V * omega)
    criteria = {
        "added_mass_A<0.05_rel_error": _criterion(
            abs(out["A_measured_mean"] - 0.5) / 0.5, 0.05),
        "asymptote_A_minus_1<0.05_rel_error": _criterion(
            abs(A_total_asymptote - 1.5) / 1.5, 0.05),
    }
    return {
        "scenario": "sudden_acceleration",
        "inputs": {**fluid, **particle, "du_dt": du_dt, **grid},
        "A_measured": out["A_measured_mean"],
        "A_expected": 0.5,
        "A_high_frequency_asymptote_total": A_total_asymptote,
        "alpha_asymptote": alpha,
        "criteria": criteria,
        "evidence": {
            "grade": "REAL",
            "anchors": ["potential-flow added mass C_A = 1/2 (Lamb 1932)",
                        "alpha -> inf limit of Landau & Lifshitz section 24"],
        },
    }


def _mode_steady_uniform(payload: Dict[str, Any]) -> Dict[str, Any]:
    fluid, particle = _fluid(payload), _particle(payload)
    U_slip = float(payload.get("U_slip", 1.0e-3))
    if U_slip == 0.0:
        raise ValueError("U_slip must be non-zero")
    grid = _time_grid(payload, None)
    ramp_time = float(payload.get("ramp_time", 0.0))
    out = ud.run_steady_uniform(fluid["rho_f"], particle["diameter"], fluid["mu"],
                                particle["rho_p"], U_slip, grid["t_end"],
                                grid["dt"], ramp_time=ramp_time)
    criteria = {
        "total_vs_schiller_naumann<0.01": _criterion(
            out["F_total_vs_sn_rel_error"], 0.01),
        "basset_residual<1e-3_Fqs": _criterion(out["F_basset_vs_qs"], 1.0e-3),
    }
    return {
        "scenario": "steady_uniform",
        "inputs": {**fluid, **particle, "U_slip": U_slip, "ramp_time": ramp_time,
                   **grid},
        "F_quasi_steady_mean": out["F_quasi_steady_mean"],
        "F_schiller_naumann": out["F_schiller_naumann"],
        "F_total_vs_sn_rel_error": out["F_total_vs_sn_rel_error"],
        "F_basset_vs_qs": out["F_basset_vs_qs"],
        "criteria": criteria,
        "evidence": {
            "grade": "REAL",
            "anchors": ["Schiller-Naumann (1935)",
                        "no acceleration -> Basset integral is identically zero"],
        },
    }


def _mode_impulsive_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    fluid, particle = _fluid(payload), _particle(payload)
    U = float(payload.get("U", 1.0e-3))
    if U == 0.0:
        raise ValueError("U must be non-zero")
    grid = _time_grid(payload, None)
    out = ud.run_impulsive_start(fluid["rho_f"], particle["diameter"],
                                 fluid["mu"], U, grid["t_end"], grid["dt"])
    criteria = {
        "max_rel_error_vs_stokes<0.15": _criterion(out["max_rel_error"], 0.15),
        "mean_rel_error_vs_stokes<0.05": _criterion(out["mean_rel_error"], 0.05),
    }
    return {
        "scenario": "impulsive_start",
        "inputs": {**fluid, **particle, "U": U, **grid},
        "F_stokes_steady": out["F_stokes_steady"],
        "max_rel_error": out["max_rel_error"],
        "mean_rel_error": out["mean_rel_error"],
        "criteria": criteria,
        "evidence": {
            "grade": "REAL",
            "anchors": ["Basset (1888): F(t) = 6 pi mu a U (1 + a/sqrt(pi nu t))"],
        },
    }


def _mode_recursive_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    fluid, particle = _fluid(payload), _particle(payload)
    n_steps = int(payload.get("n_steps", 10000))
    if n_steps < 1000:
        raise ValueError("n_steps must be >= 1000")
    dt = _positive(payload, "dt", 1.0e-5, "payload")
    n_exp = payload.get("n_exp_values", [3, 8, 16, 24])
    n_exp_values = [int(v) for v in n_exp]
    out = ud.benchmark_basset_methods(dt, n_steps, fluid["mu"], fluid["rho_f"],
                                      particle["diameter"],
                                      n_exp_values=n_exp_values)
    best = out["best"]
    criteria = {
        "recursive_L2_rel_error<0.02": _criterion(best["l2_rel_error"], 0.02),
        "speedup_vs_O_N2>=10": _criterion(best["speedup_cpu_vs_direct"], 10.0,
                                          kind="min"),
    }
    return {
        "scenario": "recursive_benchmark",
        "inputs": {**fluid, **particle, "n_steps": n_steps, "dt": dt},
        "t_direct_cpu_s": out["t_direct_cpu_s"],
        "terms": out["terms"],
        "best": best,
        "criteria": criteria,
        "evidence": {
            "grade": "REAL",
            "anchors": ["piecewise-constant quadrature of the 1/sqrt(t) kernel",
                        "van Hinsberg & Winckelmans (2010) exponential-kernel idea"],
            "note": "CPU time measured with time.process_time()",
        },
    }


def _mode_dilute_limit(payload: Dict[str, Any]) -> Dict[str, Any]:
    fluid, particle = _fluid(payload), _particle(payload)
    U_slip = float(payload.get("U_slip", 1.0e-5))
    phi_values = payload.get("phi_values", [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3])
    out = ud.run_dilute_limit_convergence(
        fluid["rho_f"], particle["diameter"], fluid["mu"], U_slip,
        phi_values=[float(p) for p in phi_values],
        n_spheres=int(payload.get("n_spheres", 8)),
    )
    smallest = min(out["rows"], key=lambda r: r["phi"])
    criteria = {
        "force_ratio_smallest_phi<0.10": _criterion(
            abs(smallest["force_ratio"] - 1.0), 0.10),
        "phi13_extrapolation<0.10": _criterion(out["extrapolation_error"], 0.10),
    }
    return {
        "scenario": "dilute_limit",
        "inputs": {**fluid, **particle, "U_slip": U_slip},
        "rows": out["rows"],
        "single_sphere_force": out["single_sphere_force"],
        "phi13_extrapolated_ratio": out["phi13_extrapolated_ratio"],
        "criteria": criteria,
        "evidence": {
            "grade": "REAL",
            "anchors": ["exact Stokes disturbance of a translating sphere "
                        "(Stokeslet + potential dipole)"],
            "note": "finite cluster (no periodic images), one-way coupling",
        },
    }


_HANDLERS = {
    "oscillatory": _mode_oscillatory,
    "sudden_acceleration": _mode_sudden_acceleration,
    "steady_uniform": _mode_steady_uniform,
    "impulsive_start": _mode_impulsive_start,
    "recursive_benchmark": _mode_recursive_benchmark,
    "dilute_limit": _mode_dilute_limit,
}


def handle_unsteady_drag(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure handler for ``POST /api/v1/dem/unsteady_drag``.

    Parameters
    ----------
    payload : dict
        ``mode`` selects the scenario (see :data:`MODES`); ``fluid`` carries
        ``rho_f``/``mu``, ``particle`` carries ``diameter``/``rho_p``.

    Returns
    -------
    dict with the scenario numbers, a ``criteria`` block (value/target/pass)
    and an ``evidence`` block (grade + analytic anchors).

    Raises
    ------
    ValueError
        on unknown mode or non-physical input (route layer maps it to 422).
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    mode = str(payload.get("mode", "oscillatory"))
    if mode not in _HANDLERS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    result = _HANDLERS[mode](payload)
    result["endpoint"] = UNSTEADY_DRAG_ENDPOINT
    result["all_criteria_pass"] = all(c["pass"] for c in result["criteria"].values())
    return result


def create_router():  # pragma: no cover - thin FastAPI adapter
    """Return a FastAPI ``APIRouter`` with the endpoint mounted (optional dep)."""
    from fastapi import APIRouter, HTTPException

    router = APIRouter()

    @router.post(UNSTEADY_DRAG_ENDPOINT.replace("/api/v1", "", 1))
    def _route(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return handle_unsteady_drag(payload)
        except ValueError as exc:      # pragma: no cover - route layer
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router