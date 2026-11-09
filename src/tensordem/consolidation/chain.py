"""The closed consolidation chain: DEM micro-parameters -> curves -> gates.

Order of operations (each arrow is a separate, independently checkable step)::

    DEM oedometer          -> E_oed, n, Z, k_hyd          (dem_oedometer.py)
    E_oed, k_hyd           -> C_v (assembled, never fitted)
    C_v                    -> layered two-field column       (poroelastic.py)
    U(t), s(t)             -> C_v inverted, s_f inverted     (inversion.py)
    inverted vs assembled  -> C2 consistency gates
    column U(T_v) vs series-> C1 analytic gate               (terzaghi.py)
    refinement scan        -> C3 drift gate
    table + applicability  -> r_u(T_v) JSON hand-off         (ru_table.py)

A single call to :func:`run_chain` returns every number the report needs, with
the pass/fail verdict of every gate and the raw curves that produced it.  The
gates are deliberately stated as *numbers with thresholds*, not as prose: a
failure is reported, never repaired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from . import applicability as app
from . import inversion as inv
from . import ru_table
from .poroelastic import (ColumnParameters, PoroelasticColumn, WATER_UNIT_WEIGHT,
                          WATER_VISCOSITY)
from .terzaghi import TV_AT_U50, U_series, Tv_of_U

__all__ = [
    "PrototypeBed",
    "run_chain",
    "u_drift_pct",
    "column_refinement_scan",
    "prototype_pore_reynolds",
    "DEFAULT_TV_POINTS",
]

#: The five time factors of acceptance criterion C1.
DEFAULT_TV_POINTS = (0.05, 0.2, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class PrototypeBed:
    """The saturated bed the curves are published for (SI)."""

    height: float = 1.0                 # H (m), full bed
    stress_step: float = 10.0e3         # step surcharge d_sigma (Pa)
    drainage: str = "double"
    saturation: float = 1.0
    chemical_cementation: bool = False
    provenance: str = "granular_bed"
    gamma_w: float = WATER_UNIT_WEIGHT
    mu_w: float = WATER_VISCOSITY
    rho_w: float = 1000.0

    @property
    def drainage_path(self) -> float:
        return 0.5 * self.height if self.drainage == "double" else self.height


def prototype_pore_reynolds(
    *,
    k_hyd: float,
    grain_diameter: float,
    stress_step: float,
    drainage_path: float,
    rho_w: float = 1000.0,
    mu_w: float = WATER_VISCOSITY,
    gamma_w: float = WATER_UNIT_WEIGHT,
) -> dict:
    """Pore Reynolds number of the *prototype* bed (not of the DEM cell).

    The DEM cell is a millimetre-scale model; its own hydraulic gradient is
    meaningless.  The Darcy gate must be witnessed on the published geometry:
    ``i = d_sigma / (gamma_w H_dr)``, ``v_s = k_hyd i``,
    ``Re_p = rho_w v_s d / mu_w``.
    """
    gradient = stress_step / (gamma_w * drainage_path)
    v_s = k_hyd * gradient
    return {
        "gradient": gradient,
        "darcy_velocity": v_s,
        "Re_p": rho_w * v_s * grain_diameter / mu_w,
        "grain_diameter": grain_diameter,
        "definition": "Re_p = rho_w v_s d / mu_w, v_s = k i, i = d_sigma/(gamma_w H_dr)",
    }


def u_drift_pct(samples_a: Sequence[dict], samples_b: Sequence[dict]) -> float:
    """Max |U_a(T_v) - U_b(T_v)| in per cent of the reference value."""
    a = np.asarray([s["U_simulated"] for s in samples_a], dtype=np.float64)
    b = np.asarray([s["U_simulated"] for s in samples_b], dtype=np.float64)
    return float(np.max(np.abs(a - b)) * 100.0)


def column_refinement_scan(
    *, k_hyd, E_oed, prototype: PrototypeBed, n_cells: Iterable[int],
    Tv_points: Sequence[float] = DEFAULT_TV_POINTS,
) -> dict:
    """Grid-refinement scan of the column (acceptance criterion C3, fluid side)."""
    runs = []
    for n in n_cells:
        col = PoroelasticColumn(
            params=ColumnParameters(
                height=prototype.height, stress_step=prototype.stress_step,
                drainage=prototype.drainage, gamma_w=prototype.gamma_w,
            ),
            k_hyd=k_hyd, E_oed=E_oed, n_cells=int(n), mu_w=prototype.mu_w,
        )
        out = col.run(Tv_points)
        runs.append({
            "n_cells": int(n),
            "samples": out["samples"],
            "C_v_assembled": out["C_v_assembled"],
            "steps": int(col.steps),
        })
    ref = runs[-1]
    for run in runs:
        run["drift_vs_finest_pct"] = u_drift_pct(run["samples"], ref["samples"])
    worst = max(run["drift_vs_finest_pct"] for run in runs[:-1]) if len(runs) > 1 else 0.0
    return {"runs": runs, "max_drift_pct": worst}


def _subsample(hist: dict, max_points: int = 1500) -> dict:
    """Keep the raw curves, thinned uniformly (C5: the raw evidence is retained)."""
    n = int(hist["t"].size)
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)
    return {
        "t": np.asarray(hist["t"])[idx].tolist(),
        "U": np.asarray(hist["U"])[idx].tolist(),
        "settlement": np.asarray(hist["settlement"])[idx].tolist(),
        "r_u_average": np.asarray(hist["r_u_average"])[idx].tolist(),
        "thin_step": int(step),
    }


def run_chain(
    *,
    dem: dict,
    prototype: PrototypeBed,
    n_cells: int = 40,
    Tv_points: Sequence[float] = DEFAULT_TV_POINTS,
    refinement_cells: Iterable[int] = (20, 40, 80),
    Tv_grid: Sequence[float] | None = None,
    dem_ensemble: Sequence[dict] | None = None,
    tolerance: dict | None = None,
) -> dict:
    """Run every gate of the chain and return the full evidence bundle."""
    tol = {
        "C1_pct": 5.0,
        "C2_cv_pct": 10.0,
        "C2_sf_pct": 5.0,
        "C3_pct": 5.0,
    }
    if tolerance:
        tol.update(tolerance)

    E_oed = float(dem["E_oed"])
    k_hyd = float(dem["k_hyd"])
    diameter = float(dem["config"]["diameter"])
    porosity = float(dem["porosity"])
    C_v = k_hyd * E_oed / prototype.gamma_w

    Re = prototype_pore_reynolds(
        k_hyd=k_hyd, grain_diameter=diameter, stress_step=prototype.stress_step,
        drainage_path=prototype.drainage_path, rho_w=prototype.rho_w,
        mu_w=prototype.mu_w, gamma_w=prototype.gamma_w,
    )
    verdict = app.check_applicability(
        saturation=prototype.saturation,
        pore_reynolds=Re["Re_p"],
        porosity=porosity,
        chemical_cementation=prototype.chemical_cementation,
        provenance=prototype.provenance,
        strict=False,
    )

    params = ColumnParameters(
        height=prototype.height, stress_step=prototype.stress_step,
        drainage=prototype.drainage, gamma_w=prototype.gamma_w,
    )
    column = PoroelasticColumn(
        params=params, k_hyd=k_hyd, E_oed=E_oed, n_cells=n_cells, mu_w=prototype.mu_w,
    )
    run = column.run(Tv_points)

    # -- C1: simulated U(T_v) vs the independent Fourier series --------------
    c1_points = []
    for sample in run["samples"]:
        Tv = sample["T_v"]
        U_ana = float(U_series(Tv))
        dev = abs(sample["U_simulated"] - U_ana) / U_ana * 100.0
        c1_points.append({
            "T_v": Tv,
            "U_simulated": sample["U_simulated"],
            "U_analytic": U_ana,
            "deviation_pct": dev,
            "r_u_simulated": sample["r_u_average_simulated"],
            "r_u_analytic": 1.0 - U_ana,
        })
    c1_max = max(p["deviation_pct"] for p in c1_points)
    c1 = {
        "points": c1_points,
        "max_deviation_pct": c1_max,
        "threshold_pct": tol["C1_pct"],
        "pass": bool(c1_max <= tol["C1_pct"]),
    }

    # -- C2: back-analysis ---------------------------------------------------
    hist = run["history"]
    cv_inv = inv.invert_cv(hist["t"], hist["U"], params.drainage_path, method="t50")
    cv_fit = inv.invert_cv(hist["t"], hist["U"], params.drainage_path, method="least_squares")
    s_inv = inv.invert_settlement(hist["t"], hist["settlement"])
    s_f_analytic = prototype.stress_step * prototype.height / E_oed
    cv_dev = abs(cv_inv["C_v"] - C_v) / C_v * 100.0
    sf_dev = abs(s_inv["s_final"] - s_f_analytic) / s_f_analytic * 100.0
    c2 = {
        "C_v_assembled": C_v,
        "C_v_inverted_t50": cv_inv["C_v"],
        "C_v_inverted_fit": cv_fit["C_v"],
        "C_v_deviation_pct": cv_dev,
        "C_v_fit_deviation_pct": abs(cv_fit["C_v"] - C_v) / C_v * 100.0,
        "C_v_threshold_pct": tol["C2_cv_pct"],
        "s_f_simulated": s_inv["s_final"],
        "s_f_analytic": s_f_analytic,
        "s_f_deviation_pct": sf_dev,
        "s_f_threshold_pct": tol["C2_sf_pct"],
        "t50_inverted": cv_inv["t50"],
        "t90_inverted": inv.invert_cv(hist["t"], hist["U"], params.drainage_path,
                                      method="t90").get("t90"),
        "inversion_residual_rms": cv_inv["rms_residual"],
        "pass": bool(cv_dev <= tol["C2_cv_pct"] and sf_dev <= tol["C2_sf_pct"]),
    }

    # -- C3: refinement scans ------------------------------------------------
    col_scan = column_refinement_scan(
        k_hyd=k_hyd, E_oed=E_oed, prototype=prototype,
        n_cells=refinement_cells, Tv_points=Tv_points,
    )
    dem_scan = [] if dem_ensemble is None else list(dem_ensemble)
    curves: list[dict] = []
    if dem_scan:
        E_samples = [float(d["E_oed"]) for d in dem_scan]
        k_samples = [float(d["k_hyd"]) for d in dem_scan]
        cv_stats = inv.propagate_cv_uncertainty(E_samples, k_samples, prototype.gamma_w)
        # U drift induced by the DEM ensemble: fix the physical time axis of
        # the reference realisation and compare U at those times
        ref_cv = C_v
        times = [float(Tv) * params.drainage_path ** 2 / ref_cv for Tv in Tv_points]
        curves = []
        for d in dem_scan:
            cv_i = float(d["k_hyd"]) * float(d["E_oed"]) / prototype.gamma_w
            col_i = PoroelasticColumn(
                params=params, k_hyd=d["k_hyd"], E_oed=d["E_oed"], n_cells=n_cells,
                mu_w=prototype.mu_w,
            )
            Tv_i = [cv_i * t / params.drainage_path ** 2 for t in times]
            out_i = col_i.run(Tv_i)
            curves.append({
                "E_oed": float(d["E_oed"]),
                "k_hyd": float(d["k_hyd"]),
                "C_v": cv_i,
                "particle_count": int(d["config"]["particle_count"]),
                "porosity": float(d["porosity"]),
                "coordination": float(d.get("coordination", float("nan"))),
                "U_at_reference_times": [s["U_simulated"] for s in out_i["samples"]],
            })
        # the reference curve is the realisation the chain itself ran on
        ref_idx = next(
            (i for i, d in enumerate(dem_scan)
             if float(d["E_oed"]) == E_oed and float(d["k_hyd"]) == k_hyd),
            0,
        )
        for i, c in enumerate(curves):
            c["is_reference"] = (i == ref_idx)
        ref_curve = curves[ref_idx]["U_at_reference_times"]
        for c in curves:
            c["U_drift_vs_reference_pct"] = float(
                np.max(np.abs(np.asarray(c["U_at_reference_times"]) - np.asarray(ref_curve)))
                * 100.0
            )
        dem_drift = max(c["U_drift_vs_reference_pct"] for c in curves)
        E_stats = inv.ci95(E_samples)
        k_stats = inv.ci95(k_samples)
    else:
        cv_stats = {"n": 1, "mean": C_v, "se": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "confidence": 0.95, "relative_se": float("nan")}
        dem_drift = 0.0
        E_stats = inv.ci95([E_oed])
        k_stats = inv.ci95([k_hyd])

    drift = max(col_scan["max_drift_pct"], dem_drift)
    c3 = {
        "threshold_pct": tol["C3_pct"],
        "column_refinement": col_scan,
        "dem_particle_scan": {
            "curves": curves,
            "drift_pct": dem_drift,
            "E_oed_stats": E_stats,
            "k_hyd_stats": k_stats,
        },
        "max_drift_pct": drift,
        "pass": bool(drift <= tol["C3_pct"]),
    }

    # -- negative control: does the gate have discriminating power? ----------
    # A *wrong* C_v (hence wrong E_oed) is invisible in U-vs-T_v space, because
    # T_v is normalised with that same C_v.  The control therefore compares the
    # curves on a fixed *physical* time axis: t_k = T_v,k H_dr^2 / C_v,reference.
    # If a 20 % error in C_v still passed C1, the gate would be vacuous.
    negative = []
    for factor in (0.8, 1.2):
        col_n = PoroelasticColumn(
            params=params, k_hyd=k_hyd, E_oed=E_oed * factor, n_cells=n_cells,
            mu_w=prototype.mu_w,
        )
        cv_n = col_n.cv_assembled()
        Tv_shifted = [Tv * cv_n / C_v for Tv in Tv_points]
        out_n = col_n.run(Tv_shifted)
        devs = [
            abs(s["U_simulated"] - float(U_series(Tv))) / float(U_series(Tv)) * 100.0
            for s, Tv in zip(out_n["samples"], Tv_points)
        ]
        negative.append({
            "E_oed_factor": factor,
            "C_v": cv_n,
            "max_deviation_vs_anchor_pct": float(max(devs)),
            "would_fail_C1": bool(max(devs) > tol["C1_pct"]),
        })

    if Tv_grid is None:
        Tv_grid = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5,
                   0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
    table = ru_table.build_ru_table(
        Tv_grid=Tv_grid,
        drainage=prototype.drainage,
        drainage_path=prototype.drainage_path,
        C_v=C_v,
        C_v_ci=cv_stats,
        applicability=verdict.as_dict(),
        provenance={
            "source": "tensordem.consolidation",
            "E_oed_Pa": E_oed,
            "k_hyd_m_s": k_hyd,
            "porosity": porosity,
            "coordination": float(dem.get("coordination", float("nan"))),
            "grain_diameter_m": diameter,
            "dem_particles": int(dem["config"]["particle_count"]),
            "dem_kn": float(dem["config"]["kn"]),
            "C_v_assembled_m2_s": C_v,
            "Re_p_prototype": Re["Re_p"],
        },
        profile_Tv=(0.05, 0.2, 0.5, 1.0, 2.0),
        validation={"C1": {"max_deviation_pct": c1_max, "pass": c1["pass"]},
                    "C2": {"C_v_deviation_pct": cv_dev, "pass": c2["pass"]}},
    )

    return {
        "prototype": {
            "height": prototype.height,
            "stress_step": prototype.stress_step,
            "drainage": prototype.drainage,
            "drainage_path": prototype.drainage_path,
            "saturation": prototype.saturation,
            "gamma_w": prototype.gamma_w,
            "mu_w": prototype.mu_w,
        },
        "dem": {
            "E_oed": E_oed,
            "E_oed_mid": float(dem.get("E_oed_mid", float("nan"))),
            "k_hyd": k_hyd,
            "porosity": porosity,
            "coordination": float(dem.get("coordination", float("nan"))),
            "strain_increment": float(dem.get("strain_increment", float("nan"))),
            "grain_diameter": diameter,
            "particle_count": int(dem["config"]["particle_count"]),
            "steps": int(dem.get("total_steps", 0)),
        },
        "C_v_assembled": C_v,
        "C_v_confidence": cv_stats,
        "applicability": verdict.as_dict(),
        "reynolds": Re,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "negative_control": negative,
        "column": {
            "n_cells": int(n_cells),
            "settlement_final": run["settlement_final"],
            "samples": run["samples"],
            "snapshots": run["snapshots"],
            "history_points": int(hist["t"].size),
            "raw_curves": _subsample(hist),
        },
        "r_u_table": table,
    }