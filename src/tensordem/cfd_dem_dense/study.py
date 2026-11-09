"""Acceptance study for the dense CFD--DEM closure anchors.

Seven machine-checkable criteria (criterion numbers follow the campaign node
``p50w2-arb7250-n03-dense-dem-cfd``):

1. action--reaction momentum conservation at ``eps_s = 0.3`` and ``0.5``
   (residual normalised by ``sum |F|``, threshold ``1e-12``), including the
   cross-check that the discrete drag density integrates to the pressure drop
   the same ``beta`` predicts;
2. Richardson--Zaki: the *dynamically inverted* index ``n`` agrees with the
   tabulated correlation to 5% and ``eps(u)`` is monotonic;
3. Ergun: the fixed-bed pressure drop at both ends of the Re range agrees with
   the analytic porous-media relation to 5% (uniform and graded beds);
4. minimum fluidisation: the dynamic force balance agrees with the closed-form
   Ergun inversion to 8%, and the drifting motion changes sign across it;
5. grid convergence: halving ``dx`` reduces the graded-bed pressure-drop error
   by at least a factor 2.5;
6. conservation: solid mass, inter-phase power-pair and the discrete energy
   ledger identity close to ``1e-10``;
7. three negative controls: switching the particle->fluid reaction off changes
   the bed pressure drop by >20%, doubling the drag coefficient moves the
   terminal velocity by ``2**-0.5`` (2%), and switching the pressure-gradient
   force off breaks the submerged-weight balance by >20%.

Every number in the output JSON is produced by running the code in this
package; nothing is transcribed.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import column as col
from . import forces as fr

__all__ = [
    "NODE_ID",
    "THRESHOLDS",
    "run_acceptance_study",
    "write_results",
    "make_figures",
    "summary_markdown",
]

NODE_ID = "p50w2-arb7250-n03-dense-dem-cfd"

THRESHOLDS = {
    "momentum_residual": 1.0e-12,
    "rz_index_deviation": 0.05,
    "ergun_pressure_deviation": 0.05,
    "umf_deviation": 0.08,
    "grid_error_ratio_min": 2.5,
    "conservation": 1.0e-10,
    "nc1_reaction_off_deviation_min": 0.20,
    "nc2_terminal_velocity_deviation": 0.02,
    "nc3_pressure_gradient_off_deviation_min": 0.20,
}

# operating point: fine sand in water -- the regime where the Richardson--Zaki
# correlation itself was established (Re_t < 2) and where the Wen--Yu dilute
# branch is calibrated.
RZ_DIAMETER = 1.0e-4
RZ_SOLID_FRACTION_PACKED = 0.58
RZ_POROSITIES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

ERGUN_DIAMETER = 2.0e-4
ERGUN_SOLID_FRACTION = 0.50
ERGUN_VELOCITIES = (0.01, 0.30)  # Re_superficial = 2 and 60

RX2 = 2.0**-0.5


@dataclass(frozen=True)
class StudyOptions:
    """Sizing knobs so the pytest suite can run a reduced study."""

    rz_points: int = 9
    n_particles: int = 64
    n_cells: int = 64
    relax_t_end: float = 3.0e-3
    relax_dt: float = 5.0e-5
    grid_cells: tuple = (20, 40, 80, 160, 320)
    ledger_dts: tuple = (4.0e-5, 2.0e-5, 1.0e-5, 5.0e-6, 2.5e-6)
    sensitivity_closures: bool = True


def _rz_config(opts: StudyOptions, model: fr.ForceModel, solid_fraction: float) -> col.ColumnConfig:
    return col.ColumnConfig(
        diameter=RZ_DIAMETER,
        rho_particle=2500.0,
        rho_fluid=1000.0,
        mu=1.0e-3,
        length=0.02,
        n_cells=opts.n_cells,
        solid_fraction=solid_fraction,
        n_particles=opts.n_particles,
        model=model,
    )


def _relax_kwargs(opts: StudyOptions) -> dict:
    return {"t_end": opts.relax_t_end, "dt": opts.relax_dt}


# ---------------------------------------------------------------------------
# criterion 1
# ---------------------------------------------------------------------------
def criterion_momentum(opts: StudyOptions) -> dict:
    """Action--reaction conservation at two solid fractions.

    The particle velocities are deliberately heterogeneous (50% spread) so the
    per-particle forces differ: an audit where every force is identical would
    cancel exactly and could not detect a deposition-weight bug.  A fourth row
    runs the same assembly with the particle->fluid reaction switched off, which
    must fail the audit at O(1) -- that is the teeth of the check.
    """
    rows = []
    rng = np.random.default_rng(11)
    for solid_fraction in (0.30, 0.50):
        n_particles = 4000
        length = 5.0e-3
        z = col.jittered_lattice_positions(
            n_particles, length, jitter_fraction=1.0, seed=7
        )
        v = 4.0e-3 * (1.0 + 0.5 * (rng.random(n_particles) - 0.5))
        model = fr.ForceModel(drag_model="gidaspow")
        audit = fr.momentum_exchange_audit(
            z_particle=z,
            v_particle=v,
            n_cells=48,
            length=length,
            diameter=RZ_DIAMETER,
            rho_particle=2500.0,
            rho_fluid=1000.0,
            mu=1.0e-3,
            model=model,
            solid_fraction=solid_fraction,
            du_rel_dt=25.0,
            fluid_velocity=0.02,
            kernel_cells=2.0,
        )
        audit_off = fr.momentum_exchange_audit(
            z_particle=z,
            v_particle=v,
            n_cells=48,
            length=length,
            diameter=RZ_DIAMETER,
            rho_particle=2500.0,
            rho_fluid=1000.0,
            mu=1.0e-3,
            model=model.replace(fluid_reaction=False),
            solid_fraction=solid_fraction,
            du_rel_dt=25.0,
            fluid_velocity=0.02,
            kernel_cells=2.0,
        )
        # cross-check: the drag density integrated over the bed equals the
        # pressure drop the same beta predicts through the fluid balance
        cfg = col.ColumnConfig(
            diameter=RZ_DIAMETER,
            solid_fraction=solid_fraction,
            n_cells=opts.n_cells,
            length=0.02,
            n_particles=opts.n_particles,
            model=model,
        )
        u_rel = 0.02 - solid_fraction * 4.0e-3
        dens = fr.drag_density(
            u_rel,
            1.0 - solid_fraction,
            diameter=cfg.diameter,
            rho_fluid=cfg.rho_fluid,
            mu=cfg.mu,
            model=cfg.model,
        )
        dp_from_drag = float(dens) * cfg.length / (1.0 - solid_fraction)
        dp_from_balance = col.fixed_bed_pressure_drop(cfg, u_rel)["dp_total"] - (
            cfg.rho_fluid * cfg.gravity * cfg.length
        )
        rows.append(
            {
                **audit,
                "reaction_off_residual": audit_off["reaction_residual"],
                "dp_from_drag_density": dp_from_drag,
                "dp_from_fluid_balance": dp_from_balance,
                "drag_density_vs_pressure_relative": abs(dp_from_drag - dp_from_balance)
                / abs(dp_from_balance),
            }
        )
    worst = max(r["reaction_residual"] for r in rows)
    worst_dp = max(r["drag_density_vs_pressure_relative"] for r in rows)
    off_worst = max(r["reaction_off_residual"] for r in rows)
    return {
        "threshold": THRESHOLDS["momentum_residual"],
        "worst_reaction_residual": worst,
        "worst_drag_density_vs_pressure_relative": worst_dp,
        "reaction_off_worst_residual": off_worst,
        "rows": rows,
        "passed": bool(
            worst <= THRESHOLDS["momentum_residual"]
            and worst_dp <= THRESHOLDS["momentum_residual"]
            and off_worst > 0.5  # the audit detects a disabled reaction
        ),
    }


# ---------------------------------------------------------------------------
# criterion 2
# ---------------------------------------------------------------------------
def criterion_richardson_zaki(opts: StudyOptions) -> dict:
    """Richardson--Zaki index from the dynamic inversion vs the correlation."""
    model = fr.ForceModel(drag_model="wen_yu")
    packed = _rz_config(opts, model, RZ_SOLID_FRACTION_PACKED)
    porosities = np.linspace(RZ_POROSITIES[0], RZ_POROSITIES[-1], opts.rz_points)
    dynamic = col.richardson_zaki_sweep(
        packed, porosities, dynamic=True, **_relax_kwargs(opts)
    )
    algebraic = col.richardson_zaki_sweep(packed, porosities, dynamic=False)
    sensitivity = {}
    if opts.sensitivity_closures:
        for name in ("wen_yu", "di_felice", "gidaspow", "schiller_naumann"):
            alt = col.richardson_zaki_sweep(
                _rz_config(opts, fr.ForceModel(drag_model=name), RZ_SOLID_FRACTION_PACKED),
                porosities,
                dynamic=False,
            )
            sensitivity[name] = {
                "n_fit": alt["n_fit"],
                "n_deviation_relative": alt["n_deviation_relative"],
            }
    deviation = max(
        dynamic["n_deviation_relative"], algebraic["n_deviation_relative"]
    )
    return {
        "threshold": THRESHOLDS["rz_index_deviation"],
        "dynamic": dynamic,
        "algebraic": algebraic,
        "dynamic_vs_algebraic_relative": abs(
            dynamic["n_fit"] - algebraic["n_fit"]
        )
        / abs(dynamic["n_fit"]),
        "closure_sensitivity": sensitivity,
        "monotonic": bool(dynamic["monotonic_increasing"] and algebraic["monotonic_increasing"]),
        "worst_deviation": deviation,
        "passed": bool(
            deviation <= THRESHOLDS["rz_index_deviation"]
            and dynamic["monotonic_increasing"]
            and algebraic["monotonic_increasing"]
        ),
    }


# ---------------------------------------------------------------------------
# criterion 3
# ---------------------------------------------------------------------------
def criterion_ergun(opts: StudyOptions) -> dict:
    """Fixed-bed pressure drop at both ends of the Re range."""
    uniform_model = fr.ForceModel(drag_model="gidaspow")
    rows = []
    for u in ERGUN_VELOCITIES:
        cfg = col.ColumnConfig(
            diameter=ERGUN_DIAMETER,
            solid_fraction=ERGUN_SOLID_FRACTION,
            n_cells=opts.n_cells,
            length=0.2,
            n_particles=opts.n_particles,
            model=uniform_model,
        )
        uniform = col.fixed_bed_pressure_drop(cfg, u)
        graded = col.graded_bed_pressure_drop(
            n_cells=max(160, opts.n_cells * 2),
            length=0.2,
            peak_solid_fraction=0.55,
            diameter=ERGUN_DIAMETER,
            u_superficial=u,
        )
        rows.append(
            {
                "u_superficial": u,
                "reynolds_superficial": uniform["reynolds_superficial"],
                "inertial_fraction": uniform["inertial_fraction"],
                "uniform": uniform,
                "graded": graded,
                "worst_relative_error": max(
                    uniform["relative_error"], graded["relative_error"]
                ),
            }
        )
    worst = max(r["worst_relative_error"] for r in rows)
    regimes = [("viscous" if r["inertial_fraction"] < 0.5 else "inertial") for r in rows]
    return {
        "threshold": THRESHOLDS["ergun_pressure_deviation"],
        "rows": rows,
        "regimes_exercised": regimes,
        "worst_relative_error": worst,
        "passed": bool(
            worst <= THRESHOLDS["ergun_pressure_deviation"]
            and "viscous" in regimes
            and "inertial" in regimes
        ),
    }


# ---------------------------------------------------------------------------
# criterion 4
# ---------------------------------------------------------------------------
def criterion_minimum_fluidization(opts: StudyOptions) -> dict:
    """Dynamic force balance vs the closed-form Ergun inversion."""
    model = fr.ForceModel(drag_model="gidaspow")
    cfg = col.ColumnConfig(
        diameter=RZ_DIAMETER,
        solid_fraction=RZ_SOLID_FRACTION_PACKED,
        n_cells=opts.n_cells,
        length=0.02,
        n_particles=opts.n_particles,
        model=model,
    )
    study = col.minimum_fluidization_study(cfg, **_relax_kwargs(opts))
    closure_sensitivity = {}
    for name in ("wen_yu", "di_felice", "schiller_naumann"):
        alt = col.minimum_fluidization_study(
            cfg.with_(model=fr.ForceModel(drag_model=name)), **_relax_kwargs(opts)
        )
        closure_sensitivity[name] = {
            "u_mf_dynamic": alt["u_mf_dynamic"],
            "u_mf_deviation_relative": alt["u_mf_deviation_relative"],
        }
    return {
        "threshold": THRESHOLDS["umf_deviation"],
        "study": study,
        "closure_sensitivity": closure_sensitivity,
        "passed": bool(
            study["u_mf_deviation_relative"] <= THRESHOLDS["umf_deviation"]
            and study["two_sided"]
        ),
    }


# ---------------------------------------------------------------------------
# criterion 5
# ---------------------------------------------------------------------------
def criterion_grid(opts: StudyOptions) -> dict:
    """Halving study of the graded-bed pressure drop."""
    study = col.grid_convergence_study(
        n_cells=opts.grid_cells,
        length=0.2,
        diameter=ERGUN_DIAMETER,
        u_superficial=1.0e-3,
        peak_solid_fraction=0.55,
    )
    return {
        "threshold": THRESHOLDS["grid_error_ratio_min"],
        "study": study,
        "passed": bool(
            study["min_error_ratio"] >= THRESHOLDS["grid_error_ratio_min"]
            and study["all_errors_nonzero"]
        ),
    }


# ---------------------------------------------------------------------------
# criterion 6
# ---------------------------------------------------------------------------
def criterion_conservation(opts: StudyOptions) -> dict:
    """Mass, inter-phase power pair and discrete energy ledger."""
    mass_ok = col.solid_mass_advection_test(t_end=0.5, dt=1.0e-4)
    mass_teeth = col.solid_mass_advection_test(t_end=0.5, dt=1.0e-4, conservative=False)
    cfg = col.ColumnConfig(
        diameter=ERGUN_DIAMETER,
        solid_fraction=ERGUN_SOLID_FRACTION,
        n_cells=opts.n_cells,
        length=0.2,
        n_particles=opts.n_particles,
        model=fr.ForceModel(drag_model="gidaspow"),
    )
    ledger_rows = []
    for dt in opts.ledger_dts:
        led = col.relaxation_energy_ledger(cfg, 0.02, t_end=2.0e-3, dt=dt)
        ledger_rows.append(
            {
                "dt": dt,
                "mechanical_closure_relative": led["mechanical_closure_relative"],
                "ledger_close_rk4_relative": led["ledger_close_rk4_relative"],
                "ledger_close_potential_relative": led["ledger_close_potential_relative"],
                "power_pair_relative_residual": led["power_pair_relative_residual"],
                "exchange_power": led["exchange_power"],
                "dissipation_sign_ok": led["dissipation_sign_ok"],
                "m_added_fraction": led["m_added_fraction"],
            }
        )
    worst_exact = max(
        max(
            r["power_pair_relative_residual"],
            r["ledger_close_rk4_relative"],
            r["ledger_close_potential_relative"],
        )
        for r in ledger_rows
    )
    quadrature = min(r["mechanical_closure_relative"] for r in ledger_rows)
    ratios = [
        ledger_rows[i]["mechanical_closure_relative"]
        / ledger_rows[i + 1]["mechanical_closure_relative"]
        for i in range(len(ledger_rows) - 1)
    ]
    return {
        "threshold": THRESHOLDS["conservation"],
        "mass": mass_ok,
        "mass_nonconservative_variant": mass_teeth,
        "ledger": ledger_rows,
        "worst_exact_identity_residual": worst_exact,
        "best_quadrature_closure": quadrature,
        "quadrature_convergence_ratios": ratios,
        "dissipation_sign_ok": bool(all(r["dissipation_sign_ok"] for r in ledger_rows)),
        "passed": bool(
            mass_ok["relative_drift"] <= THRESHOLDS["conservation"]
            and worst_exact <= THRESHOLDS["conservation"]
            and all(r["dissipation_sign_ok"] for r in ledger_rows)
            and mass_teeth["relative_drift"] > 1.0e-2  # the check has teeth
        ),
    }


# ---------------------------------------------------------------------------
# criterion 7
# ---------------------------------------------------------------------------
def criterion_negative_controls(opts: StudyOptions) -> dict:
    """NC1 reaction off, NC2 doubled drag, NC3 pressure gradient off."""
    dense = fr.ForceModel(drag_model="gidaspow")
    cfg = col.ColumnConfig(
        diameter=RZ_DIAMETER,
        solid_fraction=RZ_SOLID_FRACTION_PACKED,
        n_cells=opts.n_cells,
        length=0.02,
        n_particles=opts.n_particles,
        model=dense,
    )

    # NC1: no particle -> fluid reaction
    u_probe = 4.0e-4
    dp_on = col.fixed_bed_pressure_drop(cfg, u_probe)
    dp_off = col.fixed_bed_pressure_drop(
        cfg.with_(model=dense.replace(fluid_reaction=False)), u_probe
    )
    nc1 = {
        "u_superficial": u_probe,
        "dp_with_reaction": dp_on["dp_total"],
        "dp_without_reaction": dp_off["dp_total"],
        "deviation_relative": abs(dp_off["dp_total"] - dp_on["dp_total"])
        / abs(dp_on["dp_total"]),
        "threshold": THRESHOLDS["nc1_reaction_off_deviation_min"],
    }
    nc1["passed"] = bool(nc1["deviation_relative"] >= nc1["threshold"])

    # NC2: doubled drag coefficient -> terminal velocity ratio 2**-0.5
    inertia = fr.ForceModel(drag_model="schiller_naumann", cd_model="newton")
    u_base = fr.terminal_velocity(
        diameter=5.0e-3, rho_particle=2500.0, rho_fluid=1000.0, mu=1.0e-3, model=inertia
    )
    u_double = fr.terminal_velocity(
        diameter=5.0e-3,
        rho_particle=2500.0,
        rho_fluid=1000.0,
        mu=1.0e-3,
        model=inertia.replace(cd_scale=2.0),
    )
    ratio = u_double / u_base
    stokes_base = fr.terminal_velocity(
        diameter=RZ_DIAMETER,
        rho_particle=2500.0,
        rho_fluid=1000.0,
        mu=1.0e-3,
        model=fr.ForceModel(drag_model="wen_yu"),
    )
    stokes_double = fr.terminal_velocity(
        diameter=RZ_DIAMETER,
        rho_particle=2500.0,
        rho_fluid=1000.0,
        mu=1.0e-3,
        model=fr.ForceModel(drag_model="wen_yu", cd_scale=2.0),
    )
    nc2 = {
        "regime": "Newton (constant C_D = 0.44, inertial limit, Re > 1000)",
        "u_terminal_base": u_base,
        "u_terminal_doubled_cd": u_double,
        "ratio": ratio,
        "expected_ratio": RX2,
        "deviation_relative": abs(ratio - RX2) / RX2,
        "threshold": THRESHOLDS["nc2_terminal_velocity_deviation"],
        "stokes_regime_cross_check": {
            "diameter": RZ_DIAMETER,
            "u_terminal_base": stokes_base,
            "u_terminal_doubled_cd": stokes_double,
            "ratio": stokes_double / stokes_base,
            "note": (
                "with the Re-dependent Schiller--Naumann coefficient the ratio is "
                "not 2**-0.5 (the Stokes limit gives 1/2); reported, not hidden"
            ),
        },
    }
    nc2["passed"] = bool(nc2["deviation_relative"] <= nc2["threshold"])

    # NC3: no pressure-gradient force
    umf = col.minimum_fluidization_study(cfg, **_relax_kwargs(opts))
    nc3 = {
        "u_mf_with_pressure_gradient": umf["u_mf_dynamic"],
        "u_mf_without_pressure_gradient": umf["u_mf_without_pressure_gradient"],
        "deviation_relative": umf["nc3_deviation_relative"],
        "threshold": THRESHOLDS["nc3_pressure_gradient_off_deviation_min"],
    }
    nc3["passed"] = bool(nc3["deviation_relative"] >= nc3["threshold"])

    return {
        "nc1_reaction_off": nc1,
        "nc2_drag_doubled": nc2,
        "nc3_pressure_gradient_off": nc3,
        "n_passed": int(sum(x["passed"] for x in (nc1, nc2, nc3))),
        "passed": bool(nc1["passed"] and nc2["passed"] and nc3["passed"]),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_acceptance_study(options: StudyOptions | None = None) -> dict:
    """Run all seven criteria and return the study dictionary."""
    opts = options or StudyOptions()
    started = time.time()
    criteria = {
        "1_momentum_conservation": criterion_momentum(opts),
        "2_richardson_zaki": criterion_richardson_zaki(opts),
        "3_ergun_pressure_drop": criterion_ergun(opts),
        "4_minimum_fluidization": criterion_minimum_fluidization(opts),
        "5_grid_convergence": criterion_grid(opts),
        "6_conservation": criterion_conservation(opts),
        "7_negative_controls": criterion_negative_controls(opts),
    }
    results = {
        "node": NODE_ID,
        "module": "tensordem.cfd_dem_dense",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": THRESHOLDS,
        "operating_points": {
            "richardson_zaki": {
                "diameter_m": RZ_DIAMETER,
                "rho_particle": 2500.0,
                "rho_fluid": 1000.0,
                "mu": 1.0e-3,
                "packed_solid_fraction": RZ_SOLID_FRACTION_PACKED,
                "porosities": list(RZ_POROSITIES),
                "closure": "wen_yu (Schiller--Naumann + eps^-2.65)",
            },
            "ergun": {
                "diameter_m": ERGUN_DIAMETER,
                "solid_fraction": ERGUN_SOLID_FRACTION,
                "bed_length_m": 0.2,
                "velocities_m_s": list(ERGUN_VELOCITIES),
            },
        },
        "force_model_switches": fr.ForceModel(drag_model="gidaspow").describe(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "options": {
            "rz_points": opts.rz_points,
            "n_particles": opts.n_particles,
            "n_cells": opts.n_cells,
            "relax_t_end": opts.relax_t_end,
            "relax_dt": opts.relax_dt,
            "grid_cells": list(opts.grid_cells),
            "ledger_dts": list(opts.ledger_dts),
        },
        "criteria": criteria,
    }
    results["all_criteria_passed"] = bool(all(c["passed"] for c in criteria.values()))
    results["wall_clock_s"] = time.time() - started
    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(results: dict, outdir) -> list:
    """Render the five study figures; returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})

    # --- fig 1: Richardson-Zaki ------------------------------------------------
    rz = results["criteria"]["2_richardson_zaki"]
    dyn = rz["dynamic"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    eps = np.array(dyn["porosity"])
    u = np.array(dyn["u_superficial"])
    ax[0].loglog(eps, u, "o-", label="dynamic inversion (coupled ODE)", lw=1.4, ms=4)
    ax[0].loglog(
        eps,
        np.exp(dyn["log_terminal_velocity"]) * eps ** dyn["n_fit"],
        "--",
        label=f"fit n = {dyn['n_fit']:.3f}",
    )
    ax[0].loglog(
        eps,
        np.exp(dyn["log_terminal_velocity"]) * eps ** dyn["n_correlation"],
        ":",
        label=f"RZ correlation n = {dyn['n_correlation']:.3f}",
        lw=1.6,
    )
    ax[0].set_xlabel(r"bed voidage $\varepsilon$ [-]")
    ax[0].set_ylabel(r"superficial velocity $u_s$ [m/s]")
    ax[0].set_title(r"Richardson--Zaki expansion ($d$=100 $\mu$m sand/water)")
    ax[0].legend(fontsize=7, loc="lower right")
    expansion = (1.0 - RZ_SOLID_FRACTION_PACKED) / (1.0 - eps)
    ax[1].semilogy(eps, expansion, "s-", color="C3", lw=1.4, ms=4)
    ax[1].set_xlabel(r"bed voidage $\varepsilon$ [-]")
    ax[1].set_ylabel(r"bed expansion $H/H_0$ [-]")
    ax[1].set_title(r"mass-conserving expansion $H/H_0=(1-\varepsilon_0)/(1-\varepsilon)$")
    fig.tight_layout()
    p = outdir / "fig1_richardson_zaki.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # --- fig 2: Ergun ----------------------------------------------------------
    eg = results["criteria"]["3_ergun_pressure_drop"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    labels, measured, reference = [], [], []
    for row in eg["rows"]:
        labels.append(f"u={row['u_superficial']:.2f}\n(Re={row['reynolds_superficial']:.0f})")
        measured.append(row["uniform"]["dp_total"])
        reference.append(row["uniform"]["dp_reference"])
    xpos = np.arange(len(labels))
    ax[0].bar(xpos - 0.2, reference, width=0.4, label="reference (hydrostatic + Ergun)")
    ax[0].bar(xpos + 0.2, measured, width=0.4, label="measured (fluid balance)")
    ax[0].set_xticks(xpos, labels)
    ax[0].set_ylabel(r"$\Delta p$ [Pa]")
    ax[0].set_title("fixed-bed pressure drop, two Re ends")
    ax[0].legend(fontsize=7)
    errs_u = [r["uniform"]["relative_error"] for r in eg["rows"]]
    errs_g = [r["graded"]["relative_error"] for r in eg["rows"]]
    ax[1].bar(xpos - 0.2, errs_u, width=0.4, label="uniform bed (exact algebra)")
    ax[1].bar(xpos + 0.2, errs_g, width=0.4, label="graded bed (discretised)")
    ax[1].axhline(THRESHOLDS["ergun_pressure_deviation"], color="k", ls="--", lw=1,
                  label="5% threshold")
    ax[1].set_xticks(xpos, labels)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("relative error [-]")
    ax[1].set_title("deviation from the analytic Ergun relation")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig2_ergun_pressure_drop.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # --- fig 3: momentum conservation + energy ledger --------------------------
    mo = results["criteria"]["1_momentum_conservation"]
    co = results["criteria"]["6_conservation"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    names = [f"$\\varepsilon_s$={r['solid_fraction']:.1f}" for r in mo["rows"]]
    vals = [max(r["reaction_residual"], 1e-20) for r in mo["rows"]]
    ax[0].bar(range(len(vals)), vals, color="C0")
    ax[0].axhline(THRESHOLDS["momentum_residual"], color="k", ls="--",
                  label="1e-12 threshold")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(names)), names)
    ax[0].set_ylabel(r"$|\sum F_{f\to p}+\sum F_{p\to f}|/\sum|F|$")
    ax[0].set_title("action--reaction conservation")
    ax[0].legend(fontsize=7)
    ledger = co["ledger"]
    dts = [r["dt"] for r in ledger]
    ax[1].loglog(dts, [max(r["mechanical_closure_relative"], 1e-18) for r in ledger],
                 "o-", label="mechanical closure (trapezoid, $O(\\Delta t^2)$)")
    ax[1].loglog(dts, [max(r["power_pair_relative_residual"], 1e-18) for r in ledger],
                 "s-", label="inter-phase power pair (exact)")
    ax[1].loglog(dts, [max(r["ledger_close_rk4_relative"], 1e-18) for r in ledger],
                 "^-", label="discrete ledger identity (exact)")
    ax[1].axhline(THRESHOLDS["conservation"], color="k", ls="--", label="1e-10 threshold")
    ax[1].set_xlabel(r"time step $\Delta t$ [s]")
    ax[1].set_ylabel("relative residual [-]")
    ax[1].set_title("energy conservation channels")
    ax[1].legend(fontsize=6.5, loc="center left")
    fig.tight_layout()
    p = outdir / "fig3_conservation.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # --- fig 4: grid convergence ----------------------------------------------
    gc = results["criteria"]["5_grid_convergence"]["study"]
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    dx = np.array(gc["dx"])
    err = np.array(gc["errors"])
    ax.loglog(dx, err, "o-", label="measured error")
    ax.loglog(dx, err[0] * (dx / dx[0]) ** 2, "--",
              label=r"second-order reference $\propto \Delta x^2$")
    for i, ratio in enumerate(gc["error_ratios"]):
        ax.annotate(f"ratio {ratio:.2f}", (dx[i + 1], err[i + 1]),
                    textcoords="offset points", xytext=(6, -12), fontsize=7)
    ax.set_xlabel(r"grid spacing $\Delta x$ [m]")
    ax.set_ylabel("pressure-drop relative error [-]")
    ax.set_title("graded-bed grid convergence (criterion 5: ratio $\\geq$ 2.5)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig4_grid_convergence.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # --- fig 5: negative controls ---------------------------------------------
    nc = results["criteria"]["7_negative_controls"]
    umf = results["criteria"]["4_minimum_fluidization"]["study"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    labels = ["NC1\nreaction off", "NC2\ndrag $\\times 2$", "NC3\n$\\nabla p$ off"]
    vals = [
        nc["nc1_reaction_off"]["deviation_relative"],
        nc["nc2_drag_doubled"]["deviation_relative"],
        nc["nc3_pressure_gradient_off"]["deviation_relative"],
    ]
    thrs = [
        nc["nc1_reaction_off"]["threshold"],
        nc["nc2_drag_doubled"]["threshold"],
        nc["nc3_pressure_gradient_off"]["threshold"],
    ]
    xpos = np.arange(3)
    ax[0].bar(xpos, vals, color="C1")
    ax[0].plot(xpos, thrs, "k_", ms=22, mew=2, label="acceptance threshold")
    for i, v in enumerate(vals):
        ax[0].annotate(f"{v*100:.2f}%", (i, v), textcoords="offset points",
                       xytext=(0, 4), ha="center", fontsize=8)
    ax[0].set_xticks(xpos, labels)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("relative deviation [-]")
    ax[0].set_title("negative controls (must exceed threshold)")
    ax[0].legend(fontsize=7)
    ax[1].bar([0], [umf["u_mf_dynamic"]], width=0.5,
              label="dynamic force balance")
    ax[1].bar([1], [umf["u_mf_ergun_closed_form"]], width=0.5,
              label="Ergun closed form")
    ax[1].set_xticks([0, 1], ["dynamic\n(coupled ODE)", "Ergun\n(closed form)"])
    ax[1].set_ylabel(r"$u_{mf}$ [m/s]")
    ax[1].set_title(
        "minimum fluidisation: deviation "
        f"{umf['u_mf_deviation_relative']*100:.3f}% (threshold 8%)"
    )
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig5_negative_controls.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def summary_markdown(results: dict) -> str:
    """Render the criterion table as Markdown (generated, never transcribed)."""
    c = results["criteria"]
    lines = []
    lines.append("| # | criterion | measured | threshold | verdict |")
    lines.append("|---|---|---|---|---|")
    mom = c["1_momentum_conservation"]
    lines.append(
        f"| 1 | momentum conservation | {mom['worst_reaction_residual']:.3e} "
        f"(reaction off: {mom['reaction_off_worst_residual']:.3f}) | "
        f"$\\leq$ {mom['threshold']:.0e} | {'PASS' if mom['passed'] else 'FAIL'} |"
    )
    rz = c["2_richardson_zaki"]
    lines.append(
        f"| 2 | Richardson--Zaki index | n = {rz['dynamic']['n_fit']:.4f} vs "
        f"{rz['dynamic']['n_correlation']:.4f} -> {100*rz['dynamic']['n_deviation_relative']:.3f}%; "
        f"monotone = {rz['monotonic']} | $\\leq$ 5% | {'PASS' if rz['passed'] else 'FAIL'} |"
    )
    eg = c["3_ergun_pressure_drop"]
    lines.append(
        f"| 3 | Ergun pressure drop | {100*eg['worst_relative_error']:.4f}% "
        f"({', '.join(eg['regimes_exercised'])}) | $\\leq$ 5% | "
        f"{'PASS' if eg['passed'] else 'FAIL'} |"
    )
    umf = c["4_minimum_fluidization"]["study"]
    lines.append(
        f"| 4 | $u_{{mf}}$ force balance | {umf['u_mf_dynamic']:.6e} vs Ergun "
        f"{umf['u_mf_ergun_closed_form']:.6e} -> {100*umf['u_mf_deviation_relative']:.4f}% | "
        f"$\\leq$ 8% | {'PASS' if c['4_minimum_fluidization']['passed'] else 'FAIL'} |"
    )
    gc = c["5_grid_convergence"]["study"]
    lines.append(
        f"| 5 | grid convergence | ratios "
        f"{', '.join(f'{r:.2f}' for r in gc['error_ratios'])} | $\\geq$ 2.5 | "
        f"{'PASS' if c['5_grid_convergence']['passed'] else 'FAIL'} |"
    )
    co = c["6_conservation"]
    lines.append(
        f"| 6 | conservation | mass {co['mass']['relative_drift']:.2e}, power pair "
        f"{co['ledger'][-1]['power_pair_relative_residual']:.2e}, ledger "
        f"{co['ledger'][-1]['ledger_close_rk4_relative']:.2e} | $\\leq$ 1e-10 | "
        f"{'PASS' if co['passed'] else 'FAIL'} |"
    )
    nc = c["7_negative_controls"]
    lines.append(
        f"| 7 | negative controls | NC1 {100*nc['nc1_reaction_off']['deviation_relative']:.2f}%, "
        f"NC2 {100*nc['nc2_drag_doubled']['deviation_relative']:.4f}%, "
        f"NC3 {100*nc['nc3_pressure_gradient_off']['deviation_relative']:.2f}% | "
        f">20% / $\\leq$2% / >20% | {'PASS' if nc['passed'] else 'FAIL'} |"
    )
    lines.append("")
    lines.append(
        f"All criteria passed: **{results['all_criteria_passed']}** "
        f"(wall clock {results['wall_clock_s']:.1f} s, "
        f"{results['environment']['python']}/numpy {results['environment']['numpy']})."
    )
    return "\n".join(lines)


def write_results(results: dict, outdir, *, figures: bool = True) -> dict:
    """Write ``study.json`` plus figures; returns the written paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    if figures:
        paths = make_figures(results, outdir)
        payload["figures"] = [p.name for p in paths]
    json_path = outdir / "study.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    summary_path = outdir / "summary.md"
    summary_path.write_text(summary_markdown(results))
    return {"json": json_path, "summary": summary_path, "figures": payload.get("figures", [])}