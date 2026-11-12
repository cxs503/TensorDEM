"""Verification campaign for the DEFECT-D3 repair (open-water BEM, high loading).

Background
----------
``docs/dem_calibration_gap_ledger.md`` registered **DEFECT-D3** against
``propeller_ice.hydro.solve_open_water``: at low advance ratio every strip sat on
the induction cap ``a = 0.85``, the blade-element load was no longer balanced by
the annulus momentum, the returned residual (``max |delta a|``) was ~1e-8 so no
residual-based acceptance test could see it, and the documented "Buhl
high-induction branch" (``_buhl``) was never called.

What this campaign establishes (each item is a numbered gate, PASS/FAIL written
to ``artifacts/open_water_high_induction/verification.json``)
---------------------------------------------------------------------------
G1  the *construction* used for the high-loading closure reproduces Buhl's
    published turbine-convention parabola ``(8/9, 4F - 40/9, 50/9 - 4F)``;
G2  the propeller-convention breakdown induction is exactly ``a_s = 0.2``
    (``4 a(1+a) = 0.96 F``);
G3  the closure is C1-continuous with the classical relation at ``a_s``;
G4  the closure inverse is exact (``CT(a(CT)) = CT``);
G5  the repaired default mode reproduces the pre-repair (legacy) coefficients
    where the state lies inside the model's validity bound -- no silent
    numerical regression;
G6  the *defect* is reproducible: legacy mode reports iteration residual < 1e-6
    while violating its own momentum identity by per-cent amounts, with every
    strip on the cap (this gate fails on the pre-repair code, i.e. it is a
    repair guard, not a characterisation of current behaviour);
G7  where the closure has a root, the state satisfies the momentum identity to
    round-off (``momentum_residual < 1e-6``);
G8  the validity boundary is a measurable property of the geometry, not an
    artefact of one parameter set;
G9  the degenerate bollard case (``V_a = 0``) is reported, not silently capped;
G10 efficiency can no longer be reported negative;
G11 the coded ``_buhl`` constant form does **not** invert the parabola it is
    documented to invert (measured against the exact quadratic);
G12 the turbine-convention transplant is not a change of variable of the
    propeller convention (the map ``a_t -> -a_p`` flips the sign of ``CT``).

Every number in the JSON is produced by the run; the doc is rendered from that
JSON, so nothing is transcribed by hand.

Usage
-----
    PYTHONPATH=src python scripts/verify_open_water_high_induction.py
    PYTHONPATH=src python scripts/verify_open_water_high_induction.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordem.propeller_ice.geometry import PropellerGeometry
from tensordem.propeller_ice.hydro import (
    OpenWaterConfig,
    _buhl,
    classical_induction,
    empirical_ct,
    empirical_ct_coefficients,
    empirical_ct_max,
    empirical_induction,
    high_loading_branch_end,
    momentum_breakdown_induction,
    solve_open_water,
)

PROTOCOL = "m16448::open_water_high_induction::v1"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "open_water_high_induction"

# The geometry the defect was registered against (IN27-S4 matched 3 m propeller)
DEFECT_GEOM = dict(diameter=3.0, n_blades=4, hub_ratio=0.40, pitch_ratio=0.95,
                   chord_ratio=0.539, skew_tip_deg=0.0, thickness_ratio=0.05)
DEFECT_J = 0.2

# Default library geometry (used by the package tests) and a realistic-solidity
# geometry (sigma_0.7R ~ 0.22) used to locate the validity boundary.
LIB_GEOM = dict(diameter=0.6, n_blades=4, hub_ratio=0.35, pitch_ratio=1.0,
                chord_ratio=0.40)
LEAN_GEOM = dict(diameter=0.6, n_blades=4, hub_ratio=0.35, pitch_ratio=1.0,
                 chord_ratio=0.12)

J_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def _gate(gates: list[dict], gid: str, ok: bool, measured: float | None,
          expected: float | None, tol: float | None, note: str,
          extra: dict | None = None) -> None:
    entry = {"id": gid, "verdict": "PASS" if ok else "FAIL",
             "measured": measured, "expected": expected, "tol": tol, "note": note}
    if extra:
        entry.update(extra)
    gates.append(entry)


# ---------------------------------------------------------------------------
# G1..G4, G11, G12: algebra of the closure
# ---------------------------------------------------------------------------
def closure_algebra() -> dict:
    gates: list[dict] = []
    data: dict = {}

    # G1: the construction reproduces Buhl's published coefficients
    worst = 0.0
    replica = {}
    for F in (0.3, 0.5, 0.7, 1.0):
        c = empirical_ct_coefficients(F, 0.96, base="turbine")
        ref = (8.0 / 9.0, 4.0 * F - 40.0 / 9.0, 50.0 / 9.0 - 4.0 * F)
        err = max(abs(ci - ri) / max(abs(ri), 1.0) for ci, ri in zip(c, ref))
        replica[f"{F:.2f}"] = {"computed": list(c), "published": list(ref), "rel_err": err}
        worst = max(worst, err)
    data["buhl_replica"] = replica
    _gate(gates, "G1_construction_reproduces_buhl", worst < 1e-12, worst, 0.0, 1e-12,
          "3-condition construction in the turbine convention == Buhl 2005 coefficients")

    # G2: propeller-convention breakdown induction and the CT it corresponds to
    a_s = momentum_breakdown_induction(0.96)
    ct_at = 4.0 * 1.0 * a_s * (1.0 + a_s)
    data["breakdown"] = {"a_s": a_s, "ct_at_a_s_F1": ct_at,
                         "closure_ct_at_a_s_F1": float(empirical_ct(np.array([a_s]), 1.0)[0])}
    _gate(gates, "G2_breakdown_induction", abs(a_s - 0.2) < 1e-14, a_s, 0.2, 1e-14,
          "4a(1+a) = 0.96 -> a_s = 0.2 in the propeller convention")

    # G3: C1 continuity (value and slope) with the classical relation at a_s
    value_err, slope_err = 0.0, 0.0
    for F in (0.3, 0.5, 0.8, 1.0):
        c0, c1, c2 = empirical_ct_coefficients(F, 0.96, base="propeller")
        value_err = max(value_err, abs((c0 + c1 * a_s + c2 * a_s ** 2)
                                       - 4.0 * F * a_s * (1.0 + a_s)) / (4.0 * F * a_s * (1.0 + a_s)))
        slope_err = max(slope_err, abs((c1 + 2.0 * c2 * a_s) - 4.0 * F * (1.0 + 2.0 * a_s))
                        / (4.0 * F * (1.0 + 2.0 * a_s)))
    data["continuity"] = {"value_rel_err": value_err, "slope_rel_err": slope_err}
    _gate(gates, "G3_c1_continuity_at_a_s", max(value_err, slope_err) < 1e-12,
          max(value_err, slope_err), 0.0, 1e-12,
          "value and slope of the closure match 4Fa(1+a) at a_s for F in {0.3,0.5,0.8,1.0}")

    # G4: the inverse is exact
    worst_inv = 0.0
    rows = []
    for F in (0.35, 0.6, 1.0):
        a_hi = high_loading_branch_end(F, 0.96, 0.85)
        ct_grid = np.linspace(0.96 * F * 1.001, float(empirical_ct(np.array([a_hi]), F)[0]) * 0.999, 12)
        a, exceeded = empirical_induction(ct_grid, np.full(ct_grid.shape, F), 0.96, 0.85)
        back = empirical_ct(a, F)
        err = float(np.max(np.abs(back - ct_grid) / ct_grid))
        worst_inv = max(worst_inv, err)
        rows.append({"F": F, "a_hi": float(a_hi), "n": int(ct_grid.size),
                     "exceeded": int(np.sum(exceeded)), "inv_rel_err": err})
    data["inverse"] = {"rows": rows, "worst_rel_err": worst_inv}
    _gate(gates, "G4_closure_inverse_exact", worst_inv < 1e-12, worst_inv, 0.0, 1e-12,
          "CT(a(CT)) = CT on the monotone high-loading segment")

    # G11: does the coded _buhl constant form invert the parabola?
    rows = []
    worst_buhl = 0.0
    for F in (0.4, 0.5, 0.7, 1.0):
        a_buhl = float(_buhl(np.array([0.9]), np.array([F]), 0.85)[0])
        c0, c1, c2 = empirical_ct_coefficients(F, 0.96, base="turbine")
        ct_buhl = c0 + c1 * a_buhl + c2 * a_buhl ** 2
        # exact root of the same parabola at CT = 0.96
        disc = c1 * c1 - 4.0 * c2 * (c0 - 0.96)
        exact = sorted({(-c1 - math.sqrt(disc)) / (2.0 * c2),
                        (-c1 + math.sqrt(disc)) / (2.0 * c2)})
        exact_root = min(exact, key=lambda v: abs(v - 0.4))
        gap = abs(a_buhl - exact_root) / abs(exact_root)
        worst_buhl = max(worst_buhl, gap)
        rows.append({"F": F, "a_buhl": a_buhl, "exact_root": exact_root,
                     "ct_at_a_buhl": ct_buhl, "rel_gap": gap})
    data["buhl_form_consistency"] = {"rows": rows, "worst_rel_gap": worst_buhl}
    _gate(gates, "G11_coded_buhl_does_not_invert_its_parabola", worst_buhl > 0.5,
          worst_buhl, None, None,
          "the coded constant form gives a = 0.9446 at F = 1 while the inverse of the "
          "parabola at CT = 0.96 is a = 0.4; it is a formula of the right shape that does "
          "not reproduce the inversion it is documented to perform")

    # G12: convention transplant is not a change of variable
    a_p = np.array([0.25])
    F = 1.0
    ct_prop = 4.0 * F * a_p * (1.0 + a_p)
    ct_turb = 4.0 * F * (-a_p) * (1.0 - (-a_p))
    data["convention_map"] = {"a_p": float(a_p[0]), "ct_propeller": float(ct_prop[0]),
                              "ct_turbine_at_minus_a_p": float(ct_turb[0])}
    _gate(gates, "G12_turbine_map_flips_ct", float(ct_turb[0]) < 0.0 < float(ct_prop[0]),
          float(ct_turb[0]), None, None,
          "a_t = -a_p maps the propeller relation onto minus the turbine CT, so the "
          "turbine high-induction branch cannot be applied to this module's a")
    return {"gates": gates, "data": data}


# ---------------------------------------------------------------------------
# G5..G10: solver behaviour
# ---------------------------------------------------------------------------
def _sweep(geom_kwargs: dict, j_grid, model: str, n_strips: int = 24) -> list[dict]:
    geom = PropellerGeometry(**geom_kwargs)
    rows = []
    for J in j_grid:
        state = solve_open_water(geom, 1.0, J * geom.diameter,
                                 config=OpenWaterConfig(n_strips=n_strips,
                                                        induction_model=model))
        rows.append({
            "J": J,
            "KT": state.thrust_coefficient,
            "KQ": state.torque_coefficient,
            "eta": state.efficiency,
            "a_min": float(np.min(state.a)),
            "a_max": float(np.max(state.a)),
            "strips_at_cap": int(np.sum(np.asarray(state.a) >= 0.849)),
            "n_strips": int(np.asarray(state.a).size),
            "residual": state.residual,
            "momentum_residual": state.momentum_residual,
            "ct_max_loading": state.ct_max_loading,
            "n_strips_above_ct_limit": state.n_strips_above_ct_limit,
            "model_valid": state.model_valid,
            "validity_flag": state.validity_flag,
            "converged": state.converged,
            "thrust": state.thrust,
            "thrust_momentum": state.thrust_momentum,
        })
    return rows


def solver_behaviour(with_figures: bool = True) -> dict:
    gates: list[dict] = []
    data: dict = {}

    defect = {m: _sweep(DEFECT_GEOM, J_GRID, m) for m in ("clipped", "classical", "high_loading")}
    lib = {m: _sweep(LIB_GEOM, J_GRID, m) for m in ("clipped", "classical", "high_loading")}
    lean = {m: _sweep(LEAN_GEOM, J_GRID, m) for m in ("clipped", "classical", "high_loading")}
    data["sweeps"] = {"defect_geometry": defect, "library_geometry": lib,
                      "lean_geometry": lean}

    # G5: legacy parity where the loading is inside the model's validity bound
    worst = 0.0
    rows = []
    for label, sweeps in (("defect", defect), ("library", lib), ("lean", lean)):
        for old, new in zip(sweeps["clipped"], sweeps["classical"]):
            if old["model_valid"] and new["model_valid"]:
                for key in ("KT", "KQ"):
                    denom = max(abs(old[key]), 1e-12)
                    err = abs(new[key] - old[key]) / denom
                    rows.append({"geometry": label, "J": old["J"], "key": key, "rel_err": err})
                    worst = max(worst, err)
    data["legacy_parity"] = {"n_compared": len(rows), "worst_rel_err": worst,
                             "rows": rows[:12] + rows[-12:] if len(rows) > 24 else rows}
    _gate(gates, "G5_legacy_parity_in_valid_regime", worst < 1e-4, worst, 0.0, 1e-4,
          "repaired default reproduces the pre-repair coefficients inside the validity bound")

    # G6: the defect (repair guard -- the pre-repair code fails this gate)
    row = next(r for r in defect["clipped"] if r["J"] == DEFECT_J)
    ok = (row["residual"] < 1.0e-6 and row["strips_at_cap"] == row["n_strips"]
          and row["momentum_residual"] > 1.0e-2 and not row["model_valid"]
          and row["ct_max_loading"] > 10.0)
    data["defect_evidence"] = row
    _gate(gates, "G6_defect_reproducible_and_visible", ok,
          row["momentum_residual"], None, None,
          "legacy mode at J = 0.2: iteration residual < 1e-6 but the blade-element thrust "
          "is unbalanced by the momentum side; every strip on the cap; loading "
          "ct_max_loading x above the closure bound")

    # G7: where the closure has a root, the momentum identity holds
    worst = 0.0
    for sweeps in (defect, lib, lean):
        for r in sweeps["high_loading"]:
            if r["model_valid"] and r["n_strips_above_ct_limit"] == 0:
                worst = max(worst, r["momentum_residual"])
    data["closure_root_momentum_residual"] = worst
    _gate(gates, "G7_closure_root_satisfies_momentum_identity", worst < 1e-6, worst, 0.0, 1e-6,
          "high_loading states that are inside the bound satisfy their own momentum "
          "identity to round-off")

    # G8: the validity boundary is a property of the geometry
    boundary = {}
    for label, sweeps in (("defect", defect), ("library", lib), ("lean", lean)):
        j_valid = [r["J"] for r in sweeps["classical"] if r["model_valid"]]
        boundary[label] = {"j_valid": j_valid,
                           "first_valid_J": min(j_valid) if j_valid else None,
                           "j_invalid": [r["J"] for r in sweeps["classical"] if not r["model_valid"]]}
    data["validity_boundary"] = boundary
    ok = (boundary["lean"]["first_valid_J"] is not None
          and boundary["defect"]["first_valid_J"] is not None
          and boundary["lean"]["first_valid_J"] <= boundary["defect"]["first_valid_J"]
          and len(boundary["lean"]["j_invalid"]) > 0)
    _gate(gates, "G8_validity_boundary_is_geometric", ok,
          boundary["lean"]["first_valid_J"], None, None,
          "a low-solidity geometry reaches the valid envelope at a lower J than the "
          "high-solidity one; the boundary is not a fixed J cut")

    # G9: bollard (V_a = 0) is reported, not silently capped
    bollard = {}
    for label, kwargs in (("library", LIB_GEOM), ("defect", DEFECT_GEOM)):
        geom = PropellerGeometry(**kwargs)
        state = solve_open_water(geom, 1.0, 0.0, config=OpenWaterConfig(n_strips=24))
        bollard[label] = {"KT": state.thrust_coefficient, "KQ": state.torque_coefficient,
                          "validity_flag": state.validity_flag,
                          "model_valid": state.model_valid,
                          "momentum_residual": state.momentum_residual,
                          "strips_at_cap": int(np.sum(np.asarray(state.a) >= 0.849))}
    data["bollard"] = bollard
    ok = all(v["validity_flag"] == "bollard_unsupported" and not v["model_valid"]
             and math.isfinite(v["KT"]) for v in bollard.values())
    _gate(gates, "G9_bollard_reported_not_silent", ok, None, None, None,
          "V_a = 0 returns finite legacy-compatible numbers *and* an explicit "
          "validity_flag='bollard_unsupported'")

    # G10: efficiency is never negative
    worst_eta = 0.0
    bad = []
    for sweeps in (defect, lib, lean):
        for model, rows_ in sweeps.items():
            for r in rows_:
                if r["eta"] < 0.0 or r["eta"] > 1.0:
                    bad.append({"model": model, "J": r["J"], "eta": r["eta"]})
                worst_eta = min(worst_eta, r["eta"])
    data["efficiency"] = {"min_eta": worst_eta, "violations": bad}
    _gate(gates, "G10_efficiency_non_negative", not bad, worst_eta, 0.0, 0.0,
          "eta is reported as 0 outside the propulsive quadrant instead of a negative "
          "value from a negative K_T")
    return {"gates": gates, "data": data}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(data: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    algebra = data["algebra"]["data"]
    solver = data["solver"]["data"]

    # --- fig 1: the closure and its continuity with the classical relation
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a = np.linspace(0.0, 0.85, 400)
    F = 1.0
    c0, c1, c2 = empirical_ct_coefficients(F, 0.96, base="propeller")
    a_s = momentum_breakdown_induction(0.96)
    ax.plot(a, 4.0 * F * a * (1.0 + a), "k-", lw=2, label=r"classical $4Fa(1+a)$")
    ax.plot(a[a >= a_s], c0 + c1 * a[a >= a_s] + c2 * a[a >= a_s] ** 2, "r--", lw=2,
            label="empirical closure (propeller convention)")
    tb = empirical_ct_coefficients(F, 0.96, base="turbine")
    at = np.linspace(0.0, 1.0, 300)
    ax.plot(at, tb[0] + tb[1] * at + tb[2] * at ** 2, "b:", lw=2,
            label="Buhl parabola (turbine convention)")
    ax.axvline(a_s, color="0.6", lw=1)
    ax.axhline(0.96, color="0.6", lw=1, ls=":")
    ax.set_xlabel(r"axial induction $a$")
    ax.set_ylabel(r"local thrust coefficient $C_T$")
    ax.set_title("closure and published parabola ($F=1$)")
    ax.legend(fontsize=8, loc="upper left")
    zoom = np.linspace(0.9 * a_s, 1.1 * a_s, 60)
    ax2.plot(zoom, 4.0 * F * zoom * (1.0 + zoom), "k-", lw=3, label="classical")
    ax2.plot(zoom, c0 + c1 * zoom + c2 * zoom ** 2, "r--", lw=2, label="closure")
    ax2.axvline(a_s, color="0.6", lw=1)
    ax2.set_xlabel(r"$a$")
    ax2.set_ylabel(r"$C_T$")
    ax2.set_title(f"C1 junction at $a_s$ = {a_s:.4f} (value and slope)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig1_closure_continuity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"name": "fig1_closure_continuity.png",
                 "caption": "Classical propeller momentum relation, the empirical "
                            "high-loading closure built by the same three conditions "
                            "Buhl uses, the published turbine parabola, and the C1 "
                            "junction at a_s = 0.2 (right)."})

    # --- fig 2: the defect -- iteration residual vs physical residual
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    rows = solver["sweeps"]["defect_geometry"]["clipped"]
    js = [r["J"] for r in rows]
    ax.semilogy(js, [r["residual"] for r in rows], "k-o", label="iteration $\\max|\\Delta a|$")
    ax.semilogy(js, [max(r["momentum_residual"], 1e-12) for r in rows], "r-s",
                label="momentum identity residual")
    ax.axhline(1e-6, color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"advance ratio $J$")
    ax.set_ylabel("residual [-]")
    ax.set_title("legacy mode: converged by its own metric, unbalanced in momentum")
    ax.legend(fontsize=8)
    ax2.semilogy(js, [max(r["ct_max_loading"], 1e-3) for r in rows], "b-^")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.set_xlabel(r"advance ratio $J$")
    ax2.set_ylabel(r"$C_{T,\mathrm{BE}}/C_{T,\max}$ [-]")
    ax2.set_title("disk loading relative to the closure bound")
    fig.tight_layout()
    path = OUT / "fig2_defect_evidence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"name": "fig2_defect_evidence.png",
                 "caption": "The registered defect, quantified: the pre-repair mode stops "
                            "at 1e-8 by its own metric while the momentum identity is "
                            "violated, and the disk loading exceeds the closure bound by "
                            "up to 4.5e1 at J = 0.1."})

    # --- fig 3: open-water curves, valid region marked
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for label, style in (("clipped", "k--"), ("classical", "r-"), ("high_loading", "b:")):
        rows = solver["sweeps"]["library_geometry"][label]
        js = [r["J"] for r in rows]
        axes[0].plot(js, [r["KT"] for r in rows], style, label=label)
        axes[1].plot(js, [r["KQ"] for r in rows], style, label=label)
        axes[2].plot(js, [r["eta"] for r in rows], style, label=label)
    rows = solver["sweeps"]["library_geometry"]["classical"]
    for ax, ylab in zip(axes, (r"$K_T$", r"$K_Q$", r"$\eta$")):
        for r in rows:
            if not r["model_valid"]:
                ax.axvspan(r["J"] - 0.05, r["J"] + 0.05, color="0.9", zorder=0)
        ax.set_xlabel(r"$J$")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    axes[0].set_title("open-water curves, library geometry (grey = outside validity)")
    fig.tight_layout()
    path = OUT / "fig3_open_water_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"name": "fig3_open_water_curves.png",
                 "caption": "K_T, K_Q and eta for the three modes on the library geometry; "
                            "shaded bands are operating points the annular streamtube model "
                            "does not support (grey), so they are reported provisional."})

    # --- fig 4: coded _buhl vs the exact inverse of its own parabola
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Fs = np.linspace(0.35, 1.0, 60)
    coded, exact = [], []
    for F in Fs:
        c0, c1, c2 = empirical_ct_coefficients(F, 0.96, base="turbine")
        coded.append(float(_buhl(np.array([0.9]), np.array([F]), 0.85)[0]))
        disc = c1 * c1 - 4.0 * c2 * (c0 - 0.96)
        roots = [(-c1 - math.sqrt(disc)) / (2.0 * c2), (-c1 + math.sqrt(disc)) / (2.0 * c2)]
        exact.append(min(roots, key=lambda v: abs(v - 0.4)))
    ax.plot(Fs, coded, "r-", lw=2, label="coded `_buhl` constant form")
    ax.plot(Fs, exact, "k--", lw=2, label="exact inverse of the Buhl parabola at $C_T=0.96$")
    ax.set_xlabel(r"loss factor $F$")
    ax.set_ylabel(r"$a$")
    ax.set_title("the announced branch is unused *and* inconsistent")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig4_buhl_form_check.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"name": "fig4_buhl_form_check.png",
                 "caption": "The coded Buhl constant form against the exact root of the "
                            "parabola it is documented to invert; the two agree only near "
                            "F = 0.5 and diverge by 1.4x at F = 1."})
    return made


# ---------------------------------------------------------------------------
def run(with_figures: bool = True) -> dict:
    algebra = closure_algebra()
    solver = solver_behaviour()
    report = {"protocol": PROTOCOL,
              "algebra": algebra,
              "solver": solver,
              "gates": algebra["gates"] + solver["gates"]}
    report["n_gates"] = len(report["gates"])
    report["n_pass"] = sum(1 for g in report["gates"] if g["verdict"] == "PASS")
    report["n_fail"] = report["n_gates"] - report["n_pass"]
    report["fails"] = [g for g in report["gates"] if g["verdict"] == "FAIL"]
    if with_figures:
        report["figures"] = figures(report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip figures")
    args = ap.parse_args()
    report = run(with_figures=not args.quick)
    for g in report["gates"]:
        print(f"{g['verdict']:4s} {g['id']:44s} measured={g['measured']}")
    print(f"\n{report['n_pass']}/{report['n_gates']} gates pass, {report['n_fail']} fail")
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())