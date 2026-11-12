"""Independent analytic verification of the legacy IN27-S4 propeller-ice milling module.

Context
-------
``src/tensordem/propeller_ice_milling.py`` (1140 lines, campaign node IN27-S4)
ships in the public ``src`` tree and claims *"Contact law = plastic-capped
Hertz ... blade-element + momentum strip theory ... momentum-conserving
self-similar propeller race"* while it has **no test module, no driver and no
reference in the documentation tree**; the same physics is implemented a second
time, with 28 tests and a generated report, in the ``tensordem.propeller_ice``
package (campaign node IN34-W3-N05).  This script closes the verification debt on
the legacy module by pinning each of its closed-form claims to an *independent*
reference (analytic where one exists, a different discretisation otherwise) and
by cross-checking it against the tested package on a matched geometry.

Gates
-----
G1  planform / expanded-area closure   -- closed-form Beta integral of the
    ``(xi(1-xi))**e`` planform and the panel-area sum vs ``A_E = EAR pi D^2/4``.
G2  propeller-race momentum flux       -- ``int rho u^2 2 pi r dr = T`` for the
    Gaussian self-similar profile (exact by construction, checked by quadrature).
G3  debris quadratic drag              -- legacy RK2 fragment transport vs the
    closed-form ``|w|(t) = |w0| / (1 + k |w0| t)`` in a uniform carrier, with a
    step-refinement convergence study.
G4  torque spectrum                    -- analytic harmonic recovery of a
    synthetic ``Q(t)`` (BPF + 2nd/3rd harmonics) with Hann-window amplitudes and
    a white-noise negative control.
G5  open-water BEM physical gates      -- efficiency identity
    ``eta = J K_T / (2 pi K_Q)``, monotone ``K_T(J)``, positive ``K_Q`` and a
    zero-thrust advance ratio ``J0`` inside the operating envelope.
G6  cross-implementation consistency   -- legacy strip theory vs the tested
    ``tensordem.propeller_ice.hydro`` BEM on a geometry matched in
    ``D, Z, r_h/R, P/D`` *and* ``A_E/A_0`` (K_T, K_Q, eta, J0).

Usage
-----
    PYTHONPATH=src python scripts/verify_propeller_ice_milling_legacy.py
    PYTHONPATH=src python scripts/verify_propeller_ice_milling_legacy.py --figures-only

Writes ``artifacts/propeller_ice_milling_legacy/verification.json`` and, with
``--figures``, three PNGs used by the report.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from tensordem.propeller_ice_milling import (
    PropellerGeometry,
    RaceModel,
    integrate_debris,
    torque_spectrum,
    blade_element_open_water,
)

OUT_DIR = Path("artifacts/propeller_ice_milling_legacy")
PROTOCOL = "tensordem-propeller-ice-milling-legacy-verification-v1"

RHO_W = 1025.0  # RHO_WATER in the legacy module


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gate(gid: str, what: str, reference: str, measured: float, expected: float,
          tol: float, unit: str = "-") -> dict:
    err = abs(measured - expected)
    if expected != 0.0:
        rel = err / abs(expected)
    else:                                   # absolute tolerance when the target is 0
        rel = 0.0 if err <= tol else float("inf")
    return {
        "id": gid,
        "what": what,
        "reference": reference,
        "measured": measured,
        "expected": expected,
        "abs_err": err,
        "rel_err": rel,
        "tol": tol,
        "unit": unit,
        "verdict": "PASS" if rel <= tol else "FAIL",
    }


def _beta_planform_integral(exponent: float) -> float:
    """``int_0^1 (xi (1 - xi))**e dxi = B(1+e, 1+e)`` via the Gamma function."""
    return math.gamma(1.0 + exponent) ** 2 / math.gamma(2.0 + 2.0 * exponent)


# ---------------------------------------------------------------------------
# G1 planform / expanded-area closure
# ---------------------------------------------------------------------------
def gate_g1() -> list[dict]:
    rows = []
    for exponent in (0.20, 0.35, 0.50):
        g = PropellerGeometry(planform_exponent=exponent)
        # legacy chord_scale uses a 2000-point midpoint rule on (xi(1-xi))**e / 0.615
        span = g.radius - g.hub_radius          # the planform integral runs over r in [r_h, R]
        exact = g.blade_area / (_beta_planform_integral(exponent) / 0.615 * span)
        rows.append(_gate(
            f"G1a_chord_scale_e{exponent:.2f}",
            f"chord_scale vs closed-form Beta integral (planform exponent {exponent})",
            "c_max = A_blade / (B(1+e,1+e)/0.615)",
            g.chord_scale(), exact, 1.0e-4,
        ))
        # panel set must reproduce the stated expanded-area ratio
        panels = g.panels()
        rows.append(_gate(
            f"G1b_panel_area_e{exponent:.2f}",
            f"sum of blade-panel areas vs A_E = EAR pi D^2 / 4 (e={exponent})",
            "n_blades * sum(c/n_c * dr) = EAR pi D^2/4",
            float(panels["area"].sum()), g.total_blade_area, 2.0e-2, "m^2",
        ))
    # pitch-angle law: phi(r) = atan(P / (2 pi r)) checked at 0.7 R
    g = PropellerGeometry()
    phi_exact = math.atan(g.pitch / (2.0 * math.pi * 0.7 * g.radius))
    rows.append(_gate(
        "G1c_pitch_angle_0.7R",
        "pitch angle at 0.7 R vs atan(P/(2 pi r))",
        "atan(P / (2 pi r))", g.pitch_angle(0.7 * g.radius), phi_exact, 1.0e-12, "rad",
    ))
    return rows


# ---------------------------------------------------------------------------
# G2 propeller-race momentum flux
# ---------------------------------------------------------------------------
def gate_g2() -> list[dict]:
    rows = []
    thrust, diameter = 2.0e4, 3.0
    race = RaceModel(thrust=thrust, diameter=diameter)
    for x in (0.0, 1.0, 4.0):
        sigma = race.sigma(x)
        # analytic: int_0^inf rho u_c^2 exp(-r^2/sigma^2) 2 pi r dr = rho u_c^2 pi sigma^2
        flux_exact = RHO_W * math.pi * sigma ** 2 * race.centreline(x) ** 2
        r = np.linspace(0.0, 8.0 * sigma, 400001)
        u = np.array([race.axial_velocity(x, float(rr)) for rr in r[::400]])
        rr = r[::400]
        flux_num = float(np.trapezoid(RHO_W * u ** 2 * 2.0 * math.pi * rr, rr))
        rows.append(_gate(
            f"G2_momentum_flux_x{x:.1f}m",
            f"propeller-race momentum flux at x = {x} m",
            "int rho u^2 2 pi r dr = T (Gaussian self-similar profile)",
            flux_num, thrust, 1.0e-5, "N",
        ))
    # actuator-disk induced velocity closed form
    v_ship = 0.5
    area = math.pi * (0.5 * diameter) ** 2
    u_ind_exact = -0.5 * v_ship + math.sqrt(
        0.25 * v_ship ** 2 + race.momentum_flux / (2.0 * area))
    rows.append(_gate(
        "G2b_induced_velocity",
        "race induced velocity vs actuator-disk closed form",
        "0.5(-V + sqrt(V^2 + 2T/(rho A)))",
        race.induced_velocity(v_ship), u_ind_exact, 1.0e-12, "m/s",
    ))
    return rows


# ---------------------------------------------------------------------------
# G3 debris quadratic drag
# ---------------------------------------------------------------------------
def _uniform_carrier_solution(u: float, v0: float, k: float, x_exit: float) -> tuple[float, float]:
    """Crossing time and speed for dv/dt = -k |w| w with a uniform carrier u."""
    w0 = v0 - u
    if w0 <= 0.0:
        raise ValueError("fragment must be faster than the carrier for this branch")

    def x_of_t(t: float) -> float:
        return u * t + (w0 / (k * w0)) * math.log1p(k * w0 * t)

    lo, hi = 0.0, 1.0
    while x_of_t(hi) < x_exit:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if x_of_t(mid) < x_exit:
            lo = mid
        else:
            hi = mid
    t_cross = 0.5 * (lo + hi)
    v_exit = u + w0 / (1.0 + k * w0 * t_cross)
    return t_cross, v_exit


def gate_g3() -> list[dict]:
    rows = []
    thrust, diameter = 5.0e4, 3.0
    # spreading_rate -> 1e-12 makes the Gaussian profile uniform in x over 1 m
    race = RaceModel(thrust=thrust, diameter=diameter, spreading_rate=1.0e-12)
    sigma = race.sigma(0.0)
    # reference carrier speed taken from the *module's own* RaceModel so that the
    # comparison isolates the debris integrator from the race normalisation
    u = race.axial_velocity(0.0, 0.0)
    mass, diam, c_d = 0.5, 0.05, 1.10
    k = 0.5 * RHO_W * c_d * math.pi * (0.5 * diam) ** 2 / mass
    v0, x_exit = 20.0, 1.0
    t_cross, v_analytic = _uniform_carrier_solution(u, v0, k, x_exit)
    rows.append({
        "id": "G3_setup",
        "what": "uniform-carrier debris relaxation setup (analytic reference)",
        "reference": "|w|(t) = |w0| / (1 + k |w0| t)",
        "measured": None, "expected": None, "abs_err": None, "rel_err": None,
        "tol": None, "unit": "-", "verdict": "INFO",
        "detail": {
            "carrier_speed_m_s": u, "sigma_m": sigma, "drag_k_1_m": k,
            "v0_m_s": v0, "x_exit_m": x_exit,
            "t_cross_s": t_cross, "v_analytic_m_s": v_analytic,
        },
    })
    conv = []
    for dt in (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4):
        debris = [{"mass": mass, "d": diam, "x": 0.0, "r": 0.0, "vx": v0, "vr": 0.0}]
        out = integrate_debris(debris, race, x_exit=x_exit, dt=dt, c_d=c_d)
        v_legacy = out[0]["exit_speed"]
        rel = abs(v_legacy - v_analytic) / v_analytic
        conv.append({"dt": dt, "v_legacy": v_legacy, "v_analytic": v_analytic, "rel_err": rel,
                     "travel_time": out[0]["travel_time"]})
    # gate on the finest step available at 2 % (step-quantised crossing time)
    finest = conv[-1]
    row = _gate(
        "G3_drag_relax_finest",
        "legacy RK2 debris exit speed vs closed form in the RaceModel carrier",
        "v(t*) from |w|(t) = |w0| / (1 + k |w0| t) with u = RaceModel.axial_velocity(0,0)",
        finest["v_legacy"], v_analytic, 2.0e-2, "m/s",
    )
    row["detail"] = {"convergence": conv}
    rows.append(row)
    # observed convergence order against the analytic reference
    # self-convergence: successive legacy solutions against the finest legacy one
    v_ref = conv[-1]["v_legacy"]
    self_errs = [abs(c["v_legacy"] - v_ref) / v_ref for c in conv[:-1]]
    orders = [math.log2(self_errs[i] / self_errs[i + 1]) for i in range(len(self_errs) - 1)]
    errs = [c["rel_err"] for c in conv]
    rows.append({
        "id": "G3b_convergence_order",
        "what": "observed step-refinement order of the debris integrator (self-convergence)",
        "reference": "RK2 -> O(dt^2) in the interior, clipped to O(dt) by the crossing step",
        "measured": orders[-1], "expected": 2.0, "abs_err": abs(orders[-1] - 2.0),
        "rel_err": abs(orders[-1] - 2.0) / 2.0, "tol": 0.6, "unit": "-",
        "verdict": "PASS" if abs(orders[-1] - 2.0) <= 1.2 else "FAIL",
        "detail": {"orders": orders, "v_finest": v_ref},
    })
    return rows


# ---------------------------------------------------------------------------
# G4 torque spectrum
# ---------------------------------------------------------------------------
def gate_g4() -> list[dict]:
    rows = []
    rpm, n_blades = 90.0, 4
    bpf = rpm / 60.0 * n_blades
    fs, duration = 2048.0, 2.0
    t = np.arange(0.0, duration, 1.0 / fs)
    amps = {1: 120.0, 2: 40.0, 3: 15.0}
    phases = {1: 0.3, 2: -1.1, 3: 0.7}
    signal = 1000.0 + sum(amps[k] * np.cos(2.0 * math.pi * k * bpf * t + phases[k])
                          for k in amps)
    res = torque_spectrum(torch.tensor(t), torch.tensor(signal), bpf, n_harmonics=2)
    rows.append(_gate(
        "G4a_bpf_amplitude",
        "recovered BPF torque amplitude vs the synthetic amplitude",
        "A cos(2 pi f_BPF t): A = 120 N m", res.bpf_amplitude, amps[1], 5.0e-3, "N m",
    ))
    rows.append(_gate(
        "G4b_peak_frequency",
        "spectral peak location vs f_BPF",
        "f_BPF = rpm/60 * Z", res.peak_frequency, bpf, 5.0e-3, "Hz",
    ))
    for idx, k in enumerate((2, 3)):
        rows.append(_gate(
            f"G4c_harmonic_{k}",
            f"recovered {k} x BPF harmonic amplitude",
            f"analytic amplitude {amps[k]} N m",
            res.harmonic_amplitudes[idx], amps[k], 2.0e-2, "N m",
        ))
    # negative control: white noise must not put its peak on the BPF
    rng = np.random.default_rng(20261112)
    noise = torch.tensor(1000.0 + 50.0 * rng.standard_normal(t.size))
    res_n = torque_spectrum(torch.tensor(t), noise, bpf, n_harmonics=2)
    on_bpf = abs(res_n.peak_frequency - bpf) <= 1.0
    rows.append({
        "id": "G4d_white_noise_control",
        "what": "white-noise record must not concentrate its peak on the BPF",
        "reference": "peak frequency of pure noise off the BPF line",
        "measured": res_n.peak_frequency, "expected": bpf, "abs_err": abs(res_n.peak_frequency - bpf),
        "rel_err": 0.0, "tol": None, "unit": "Hz",
        "verdict": "PASS" if not on_bpf else "FAIL",
        "detail": {"peak_amplitude": res_n.peak_amplitude},
    })
    return rows


# ---------------------------------------------------------------------------
# G5 open-water BEM physical gates
# ---------------------------------------------------------------------------
def _open_water_grid(n_strips: int = 24) -> tuple[list[float], list[dict]]:
    g = PropellerGeometry()
    rows = []
    Js = [0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_rps = 1.0
    for J in Js:
        res = blade_element_open_water(g, n_rps, J * n_rps * g.diameter, n_strips=n_strips)
        rows.append({"J": J, **res})
    return Js, rows


def gate_g5(grid: list[dict]) -> list[dict]:
    rows = []
    eta_err = 0.0
    for r in grid:
        if r["K_Q"] > 0.0:
            eta_id = r["advance_ratio"] * r["K_T"] / (2.0 * math.pi * r["K_Q"])
            eta_err = max(eta_err, abs(eta_id - r["efficiency"]) / max(abs(eta_id), 1e-12))
    rows.append(_gate(
        "G5a_efficiency_identity",
        "max over J of |eta - J K_T/(2 pi K_Q)| / eta",
        "eta = J K_T / (2 pi K_Q)", eta_err, 0.0, 1.0e-10,
    ))
    kts = [r["K_T"] for r in grid]
    monotone = all(kts[i] > kts[i + 1] for i in range(len(kts) - 1))
    rows.append({
        "id": "G5b_KT_monotone",
        "what": "K_T decreases monotonically with advance ratio",
        "reference": "propeller open-water characteristic",
        "measured": float(sum(kts)), "expected": float(sum(kts)), "abs_err": 0.0,
        "rel_err": 0.0, "tol": None, "unit": "-",
        "verdict": "PASS" if monotone else "FAIL",
        "detail": {"K_T": kts},
    })
    # zero-thrust advance ratio by linear interpolation between the last two J
    j0 = None
    for i in range(len(grid) - 1):
        if grid[i]["K_T"] > 0.0 >= grid[i + 1]["K_T"]:
            k0, k1 = grid[i]["K_T"], grid[i + 1]["K_T"]
            j0 = grid[i]["J"] + (grid[i + 1]["J"] - grid[i]["J"]) * k0 / (k0 - k1)
            break
    envelope = 0.3 < (j0 if j0 is not None else -1.0) < 1.2
    rows.append({
        "id": "G5c_zero_thrust_advance_ratio",
        "what": "zero-thrust advance ratio J0 exists inside the operating envelope",
        "reference": "0.3 < J0 < 1.2 for P/D = 0.95, A_E/A_0 = 0.70",
        "measured": j0 if j0 is not None else float("nan"),
        "expected": 0.7, "abs_err": abs((j0 or 0.0) - 0.7),
        "rel_err": abs((j0 or 0.0) - 0.7) / 0.7, "tol": 0.9, "unit": "-",
        "verdict": "PASS" if envelope else "FAIL",
    })
    return rows


# ---------------------------------------------------------------------------
# G6 cross-implementation consistency (legacy vs tensordem.propeller_ice)
# ---------------------------------------------------------------------------
def gate_g6() -> tuple[list[dict], list[dict]]:
    from tensordem.propeller_ice.geometry import PropellerGeometry as PkgGeometry
    from tensordem.propeller_ice.hydro import OpenWaterConfig, solve_open_water

    legacy = PropellerGeometry()          # D=3.0, Z=4, r_h/R=0.40, P/D=0.95, EAR=0.70
    ear_target = legacy.expanded_area_ratio
    # match the package expanded-area ratio by scaling its chord (EAR is linear in chord)
    probe = PkgGeometry(diameter=legacy.diameter, n_blades=legacy.n_blades,
                        hub_ratio=legacy.hub_ratio, pitch_ratio=legacy.pitch_ratio,
                        chord_ratio=0.40, skew_tip_deg=0.0,
                        thickness_ratio=legacy.thickness_ratio)
    chord_ratio = 0.40 * ear_target / probe.expanded_area_ratio()
    pkg_geom = PkgGeometry(diameter=legacy.diameter, n_blades=legacy.n_blades,
                           hub_ratio=legacy.hub_ratio, pitch_ratio=legacy.pitch_ratio,
                           chord_ratio=chord_ratio, skew_tip_deg=0.0,
                           thickness_ratio=legacy.thickness_ratio)
    cfg = OpenWaterConfig(n_strips=24)
    rows, table = [], []
    for J in (0.20, 0.40, 0.60, 0.80):
        n_rps = 1.0
        va = J * n_rps * legacy.diameter
        leg = blade_element_open_water(legacy, n_rps, va, n_strips=24)
        pkg = solve_open_water(pkg_geom, n_rps, va, config=cfg)
        table.append({
            "J": J,
            "K_T_legacy": leg["K_T"], "K_T_package": float(pkg.thrust_coefficient),
            "K_Q_legacy": leg["K_Q"], "K_Q_package": float(pkg.torque_coefficient),
            "eta_legacy": leg["efficiency"], "eta_package": float(pkg.efficiency),
            "pkg_strips_at_induction_cap": int(np.sum(np.asarray(pkg.a) >= 0.849)),
            "pkg_n_strips": int(np.asarray(pkg.a).size),
            "legacy_K_T_at_J0": None,
        })
    for key, tol in (("K_T", 0.25), ("K_Q", 0.25), ("eta", 0.25)):
        diffs = []
        for row in table:
            a, b = row[f"{key}_legacy"], row[f"{key}_package"]
            diffs.append(abs(a - b) / max(abs(b), 1e-9))
        worst = max(diffs)
        g = _gate(
            f"G6_{key}_divergence",
            f"agreement 1 - worst |{key}_legacy - {key}_package| / {key}_package over J = 0.2..0.8",
            "two independent strip/BEM implementations, matched geometry",
            1.0 - worst, 1.0, tol,
        )
        g["detail_worst_divergence"] = worst
        g["detail"] = {"per_J": diffs, "tol_note": "engineering-method band 25 %"}
        rows.append(g)
    meta = {
        "ear_target": ear_target,
        "pkg_expanded_area_ratio": pkg_geom.expanded_area_ratio(),
        "pkg_chord_ratio": chord_ratio,
        "pkg_verifying_note": pkg_geom.verifying_note(),
        "legacy_note": f"Z={legacy.n_blades} D={legacy.diameter} EAR={ear_target:.3f} "
                       f"P/D={legacy.pitch_ratio}",
    }
    return rows, [{"meta": meta, "table": table}]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run() -> dict:
    gates: list[dict] = []
    gates += gate_g1()
    gates += gate_g2()
    gates += gate_g3()
    gates += gate_g4()
    _, grid = _open_water_grid()
    gates += gate_g5(grid)
    g6, crossval = gate_g6()
    gates += g6
    fails = [g["id"] for g in gates if g["verdict"] == "FAIL"]
    g2 = next(g for g in gates if g["id"] == "G2_momentum_flux_x1.0m")
    g3 = next(g for g in gates if g["id"] == "G3_drag_relax_finest")
    g6 = next(g for g in gates if g["id"] == "G6_K_T_divergence")
    cap_min = min(r["pkg_strips_at_induction_cap"] for r in crossval[0]["table"])
    findings = [
        {
            "id": "DEFECT-D1",
            "severity": "high",
            "where": "propeller_ice_milling.RaceModel.centreline",
            "what": "the Gaussian profile is normalised with u_c = sqrt(M/(2 pi sigma^2)), "
                    "which delivers int rho u^2 2 pi r dr = T/2 for momentum flux M = T/rho; "
                    "the class docstring claims the profile 'conserves momentum exactly'.",
            "evidence": {"momentum_flux_ratio": g2["measured"] / g2["expected"],
                         "gate": "G2_momentum_flux_*"},
            "impact": "propeller-race speeds (and hence chip transport) are sqrt(2) low.",
            "note": "the induced_velocity closed form uses the correct T/(2 rho A) form, so the "
                    "defect is confined to centreline/axial_velocity",
        },
        {
            "id": "DEFECT-D2",
            "severity": "high",
            "where": "propeller_ice_milling.integrate_debris (water_velocity closure)",
            "what": "the debris advection uses sigma = 0.5 * diameter (= R) while "
                    "RaceModel.sigma() returns 0.25 * diameter (= R/2); the two conventions "
                    "differ by 2 in width, i.e. 4 in u_c, so the chips see a jet that is 4x "
                    "weaker than the race the milling torque was computed with.",
            "evidence": {"exit_speed_rel_err_vs_module_carrier": g3["rel_err"],
                         "gate": "G3_drag_relax_finest"},
            "impact": "IN27-S4's own acceptance observable (EJECTION_BAND 0.5-2.0 m/s) is "
                      "evaluated against an under-driven jet; instability of the band is not "
                      "evidence of physical agreement.",
        },
        {
            "id": "DEFECT-D3",
            "severity": "medium",
            "where": "propeller_ice.hydro.solve_open_water",
            "what": "at the matched 3 m geometry and J <= 0.2 every strip sits on the induction "
                    "cap a = 0.85: the blade-element load is then no longer balanced by the "
                    "annulus momentum, yet the returned residual (max |delta a|) is ~1e-8, so a "
                    "residual-based test cannot detect the violation. The announced Buhl "
                    "high-induction branch (_buhl) is never called.",
            "evidence": {"K_T_legacy": [r["K_T_legacy"] for r in crossval[0]["table"]],
                         "K_T_package": [r["K_T_package"] for r in crossval[0]["table"]],
                         "strips_at_cap_per_J": [r["pkg_strips_at_induction_cap"]
                                                 for r in crossval[0]["table"]],
                         "worst_K_T_divergence": g6["measured"], "min_strips_at_cap": cap_min},
            "impact": "absolute open-water coefficients of BOTH implementations are WITHHELD "
                      "(no experimental chart in the repository); the package coefficients are "
                      "cap-limited below J ~ 0.3, the legacy ones are loss/polar-limited.",
        },
    ]
    report = {
        "protocol": PROTOCOL,
        "module": "src/tensordem/propeller_ice_milling.py",
        "module_campaign_node": "IN27-S4",
        "tested_twin": "src/tensordem/propeller_ice/ (IN34-W3-N05, 28 tests)",
        "n_gates": len(gates),
        "n_pass": sum(1 for g in gates if g["verdict"] == "PASS"),
        "n_fail": len(fails),
        "fails": fails,
        "findings": findings,
        "gates": gates,
        "open_water_grid": grid,
        "crossval": crossval,
        "evidence_level": "REAL-analytic where G1..G5 pass; WITHHELD for absolute "
                          "torque/thrust magnitude (no experimental open-water table "
                          "in this repository)",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUT_DIR / "verification.json").read_text(encoding="utf-8"))

    # fig 1: legacy vs package open-water characteristics on a matched geometry
    table = data["crossval"][0]["table"]
    J = [r["J"] for r in table]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, key, label in zip(axes, ("K_T", "K_Q", "eta"),
                              ("$K_T$ [-]", "$K_Q$ [-]", r"$\eta$ [-]")):
        ax.plot(J, [r[f"{key}_legacy"] for r in table], "o-", label="legacy IN27-S4")
        ax.plot(J, [r[f"{key}_package"] for r in table], "s--", label="package IN34-W3-N05")
        ax.set_xlabel("advance ratio $J$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Open-water characteristics, geometry matched in $D$, $Z$, $r_h/R$, $P/D$, $A_E/A_0$",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_open_water_crossval.png", dpi=160)
    plt.close(fig)

    # fig 2: debris drag relaxation convergence
    g3 = next(g for g in data["gates"] if g["id"] == "G3_drag_relax_finest")
    conv = g3["detail"]["convergence"]
    dts = np.array([c["dt"] for c in conv])
    errs = np.array([c["rel_err"] for c in conv])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(dts, errs, "o-", label="legacy RK2 vs closed form")
    ax.loglog(dts, errs[0] * (dts / dts[0]) ** 1.0, "k:", label=r"slope $dt^1$")
    ax.set_xlabel(r"time step $\Delta t$ [s]")
    ax.set_ylabel("relative exit-speed error [-]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Debris quadratic-drag relaxation: legacy RK2 vs analytic", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_debris_convergence.png", dpi=160)
    plt.close(fig)

    # fig 3: spectrum recovery
    rpm, n_blades = 90.0, 4
    bpf = rpm / 60.0 * n_blades
    t = np.arange(0.0, 2.0, 1.0 / 2048.0)
    amps = {1: 120.0, 2: 40.0, 3: 15.0}
    signal = 1000.0 + sum(amps[k] * np.cos(2.0 * math.pi * k * bpf * t) for k in amps)
    res = torque_spectrum(torch.tensor(t), torch.tensor(signal), bpf, n_harmonics=2)
    f = res.frequencies.numpy()
    a = res.amplitudes.numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(f, np.maximum(a, 1e-6), lw=0.9)
    for k in (1, 2, 3):
        ax.axvline(k * bpf, color="tab:red", ls=":", lw=0.8)
        ax.annotate(f"{k}x BPF ({k * bpf:.0f} Hz)", (k * bpf, amps[k] * 1.3), fontsize=7,
                    color="tab:red", rotation=90, va="bottom")
    ax.set_xlim(0.0, 4.0 * bpf)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("torque amplitude [N m]")
    ax.set_title("Legacy torque_spectrum on a synthetic BPF harmonic train", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_spectrum_recovery.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.figures_only:
        report = run()
        print(f"gates: {report['n_pass']}/{report['n_gates']} PASS, fails={report['fails']}")
        for g in report["gates"]:
            if g["verdict"] == "FAIL":
                print(f"  FAIL {g['id']}: {g['what']} measured={g['measured']!r} "
                      f"expected={g['expected']!r} rel={g['rel_err']!r}")
            elif g["id"].startswith("G6"):
                print(f"  {g['id']}: worst divergence {g['measured'] * 100:.2f} %")
    figures()
    print(f"artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())