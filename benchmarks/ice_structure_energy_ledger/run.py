"""Benchmark / acceptance harness for the ice-structure four-ledger energy audit.

Runs the audit cases, evaluates the machine-checkable criteria (a)-(h) of node
P304W2-N06 against *independent* reference implementations, writes the ledger
histories and the acceptance JSON, and renders the report figures.

Usage:  python benchmarks/ice_structure_energy_ledger/run.py

Artefacts are written to ``docs/artifacts/ice_structure_energy_ledger/`` -- the
tree convention for committed run evidence (``artifacts/`` at the repo root is
not tracked by the repo gate, so evidence placed there disappears in every
fresh checkout and the maturity gate can never verify it).  The per-step ledger
histories of the finer ``dt`` levels of the convergence sweep are bulk data:
they live in the ``sweep/`` subdirectory, which is git-ignored, while their
numbers and sha256 stay in ``acceptance.json``.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_structure_energy_ledger import (  # noqa: E402
    CohesiveBondLaw,
    ContactLaw,
    IceStructureEnergyAudit,
    bond_dissipation_reference,
    build_bonded_lattice,
    contact_strain_energy_reference,
    kinetic_energy_reference,
    run_energy_audit_case,
    viscous_dissipation_reference,
)

OUT = ROOT / "docs" / "artifacts" / "ice_structure_energy_ledger"
OUT.mkdir(parents=True, exist_ok=True)
#: Bulk per-step histories of the sweep levels outside this tuple go to the
#: git-ignored ``sweep/`` subdirectory; only the canonical level is committed.
COMMITTED_DT_LEVELS = (1.0e-5,)
SWEEP_OUT = OUT / "sweep"

RADIUS = 0.05
DENSITY = 900.0
IMPACT = (-2.0, 0.0, 0.0)
GAP = 2.0e-3
DURATION = 8.0e-3

ELASTIC_BOND = CohesiveBondLaw(tensile_strength=4.0e5)     # never fractures at 2 m/s
FRACTURE_BOND = CohesiveBondLaw(tensile_strength=2.0e5)    # partial damage zone
ELASTIC_CONTACT = ContactLaw(k_normal=1.0e6, restitution=1.0)
FRACTURE_CONTACT = ContactLaw(k_normal=1.0e6, restitution=0.9)


def _jsonable(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    keys = [k for k in rows[0].keys()]
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(repr(float(row[k])) for k in keys) + "\n")


def _case(**kwargs) -> IceStructureEnergyAudit:
    return run_energy_audit_case(impact_velocity=IMPACT, gap=GAP,
                                 duration=DURATION, **kwargs)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def run_dt_sweep(dt_levels: tuple[float, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {"elastic": {}, "fracture": {}}
    for dt in dt_levels:
        for name, bond, contact in (
            ("elastic", ELASTIC_BOND, ELASTIC_CONTACT),
            ("fracture", FRACTURE_BOND, FRACTURE_CONTACT),
        ):
            sim = _case(dt=dt, bond_law=bond, contact=contact)
            summary = sim.summary()
            ref_f = bond_dissipation_reference(sim.bond_law, sim.bond_max_extension)
            ev_ref = viscous_dissipation_reference(sim.history)
            csv_dir = OUT if dt in COMMITTED_DT_LEVELS else SWEEP_OUT
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"ledger_{name}_dt{dt:.1e}.csv"
            _write_csv(csv_path, sim.history)
            out[name][f"{dt:.1e}"] = {
                "dt": dt,
                "summary": summary,
                "E_f_reference": ref_f["E_f_reference"],
                "E_f_error_relative": (
                    abs(sim.ledger.E_f - ref_f["E_f_reference"])
                    / ref_f["E_f_reference"]
                    if ref_f["E_f_reference"] > 1.0e-12 * max(summary["E_total_0"], 1.0)
                    else 0.0
                ),
                "n_fully_separated": ref_f["n_fully_separated"],
                "E_v_reference": ev_ref,
                "E_v_error_relative": (
                    abs(sim.ledger.E_v - ev_ref) / ev_ref if ev_ref > 0 else 0.0
                ),
                "ledger_csv": str(csv_path.relative_to(ROOT)),
                "ledger_csv_sha256": _sha256(csv_path),
                "_sim": sim,
            }
    return out


def run_reference_checks(sim: IceStructureEnergyAudit) -> dict[str, float]:
    """Criteria (a)/(b): compare the state ledgers with independent re-derivations."""
    ke = kinetic_energy_reference(sim.velocities, sim.omegas, sim.mass, sim.inertia)
    se = contact_strain_energy_reference(sim)
    ke_err = abs(sim.ledger.E_k - ke["E_k"]) / ke["E_k"] if ke["E_k"] else 0.0
    se_err = abs(sim.ledger.E_s - se["E_s"]) / se["E_s"] if se["E_s"] else 0.0
    return {
        "E_k_ledger": sim.ledger.E_k,
        "E_k_reference": ke["E_k"],
        "E_k_error_relative": ke_err,
        "E_s_ledger": sim.ledger.E_s,
        "E_s_reference": se["E_s"],
        "E_s_error_relative": se_err,
    }


def sphere_restitution_case(restitution: float, dt: float = 1.0e-6) -> dict[str, float]:
    """Single sphere vs wall: analytic viscous loss 0.5 m v0^2 (1 - e^2)."""
    lat = build_bonded_lattice(1, 1, 1, RADIUS, DENSITY)
    pos = lat["positions"].copy()
    pos[:, 0] += RADIUS + 5.0e-3 - pos[:, 0].min()
    v0 = -1.5
    sim = IceStructureEnergyAudit(
        positions=pos,
        velocities=np.array([[v0, 0.0, 0.0]]),
        radius=RADIUS,
        density=DENSITY,
        contact=ContactLaw(k_normal=1.0e6, restitution=restitution),
        bond_law=CohesiveBondLaw(tensile_strength=1.0e30, fracture_energy=1.0e60),
        dt=dt,
        wall_plane_x=0.0,
    )
    sim.run_until(1.2e-2)
    analytic = 0.5 * sim.mass * v0 ** 2 * (1.0 - restitution ** 2)
    out_of_ledger = {
        "clamp_loss_J": sim.clamp_loss_J,
        "spring_release_J": sim.spring_release_J,
        "friction_sliding_J": sim.friction_sliding_J,
    }
    numeric = viscous_dissipation_reference(sim.history)
    return {
        "restitution": restitution,
        "E_v_ledger": sim.ledger.E_v,
        "E_v_analytic": analytic,
        "E_v_error_relative": abs(sim.ledger.E_v - analytic) / analytic,
        "E_v_trapezoid": numeric,
        "E_v_trapezoid_error_relative": abs(sim.ledger.E_v - numeric) / numeric,
        "rebound_velocity": float(sim.velocities[0, 0]),
        "rebound_ratio": float(-sim.velocities[0, 0] / -v0),
        "residual_relative_max": sim.summary()["residual_relative_max"],
        "ledger_channels": {
            "E_k": sim.ledger.E_k, "E_s": sim.ledger.E_s,
            "E_f": sim.ledger.E_f, "E_v": sim.ledger.E_v,
            "E_total_0": sim.reference_total_J,
        },
        "out_of_ledger_channels": out_of_ledger,
    }


def oblique_case() -> dict[str, float]:
    """Oblique impact: exercises tangential spring energy, rotation and c_t."""
    lat = build_bonded_lattice(1, 1, 1, RADIUS, DENSITY)
    pos = lat["positions"].copy()
    pos[:, 0] += RADIUS + 1.0e-3 - pos[:, 0].min()
    sim = IceStructureEnergyAudit(
        positions=pos,
        velocities=np.array([[-1.5, 1.0, 0.0]]),
        radius=RADIUS,
        density=DENSITY,
        contact=ContactLaw(k_normal=1.0e6, restitution=0.95, friction=50.0),
        bond_law=CohesiveBondLaw(tensile_strength=1.0e30, fracture_energy=1.0e60),
        dt=1.0e-6,
        wall_plane_x=0.0,
    )
    # step until the tangential spring is maximally loaded and compare the
    # ledger against the independent spring-potential sum *at that instant*
    n_steps = int(round(8.0e-3 / sim.dt))
    best = None
    for _ in range(n_steps):
        sim.step()
        if best is None or sim.ledger.E_s_contact_tangential > best[0]:
            best = (sim.ledger.E_s_contact_tangential, sim.time, None)
        if best[2] is None and sim.ledger.E_s_contact_tangential > 0.0:
            best = (sim.ledger.E_s_contact_tangential, sim.time,
                    contact_strain_energy_reference(sim)["E_s_contact_tangential"])
    ledger_t, t_peak, ref_t = best
    ref_all = contact_strain_energy_reference(sim)
    peak_rot = max(r["E_k_rot"] for r in sim.history)
    return {
        "peak_tangential_strain_energy": ledger_t,
        "time_of_peak_s": t_peak,
        "peak_rotational_kinetic_energy": peak_rot,
        "E_s_contact_tangential_ledger": ledger_t,
        "E_s_contact_tangential_reference": ref_t,
        "E_s_contact_tangential_error_relative": (
            abs(ledger_t - ref_t) / max(ref_t, 1e-30)),
        "E_s_contact_normal_ledger": sim.ledger.E_s_contact_normal,
        "E_s_contact_normal_reference": ref_all["E_s_contact_normal"],
        "E_v_ledger": sim.ledger.E_v,
        "E_v_tangential_ledger": sim.ledger.E_v_tangential,
        "residual_relative_max": sim.summary()["residual_relative_max"],
        "clamp_loss_J": sim.clamp_loss_J,
        "friction_sliding_J": sim.friction_sliding_J,
        "spring_release_J": sim.spring_release_J,
    }


def negative_control() -> dict[str, float]:
    sim = run_energy_audit_case(impact_velocity=(0.0, 0.0, 0.0), gap=GAP,
                                duration=2.0e-3, dt=1.0e-5,
                                bond_law=ELASTIC_BOND, contact=ELASTIC_CONTACT)
    summary = sim.summary()
    return {
        "E_total_0": summary["E_total_0"],
        "E_k": summary["E_k"],
        "E_s": summary["E_s"],
        "E_f": summary["E_f"],
        "E_v": summary["E_v"],
        "max_abs_total_over_run": max(abs(r["E_total"]) for r in sim.history),
        "max_abs_velocity": float(np.abs(sim.velocities).max()),
        "n_broken": summary["n_broken"],
        "pass": bool(summary["E_total_0"] == 0.0
                     and max(abs(r["E_total"]) for r in sim.history) == 0.0
                     and summary["n_broken"] == 0),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(sweep: dict, sphere: dict, oblique: dict,
                 base_fracture: IceStructureEnergyAudit) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []

    # fig 1: four ledgers over time (fracture case)
    sim = base_fracture
    t = np.array([r["t"] for r in sim.history]) * 1e3
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for key, label in (("E_k", "E_k kinetic"), ("E_s", "E_s contact strain"),
                       ("E_f", "E_f fracture"), ("E_v", "E_v viscous")):
        axes[0].plot(t, [r[key] for r in sim.history], label=label)
    axes[0].plot(t, [r["E_total"] for r in sim.history], "k--", lw=1.2,
                 label="E_total (sum of four)")
    axes[0].axhline(sim.reference_total_J, color="crimson", ls=":", lw=1.0,
                    label="E_total(0)")
    axes[0].set_ylabel("energy [J]")
    axes[0].set_title("Four-ledger energy audit, ice floe vs rigid structure "
                      f"(v0 = 2 m/s, e = 0.9, {sim.n_broken_total} bonds broken)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(t, [max(r["residual_relative"], 1e-12) for r in sim.history])
    axes[1].axhline(1e-2, color="orange", ls="--", lw=1.0, label="1% acceptance")
    axes[1].set_xlabel("time [ms]")
    axes[1].set_ylabel(r"$\varepsilon_E$ [-]")
    axes[1].set_title("per-step closure residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig1_four_ledger_history.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig1_four_ledger_history.png",
                 "caption": "Four-ledger energies and per-step closure residual for the "
                            "fracturing floe impact.",
                 "path": str(p.relative_to(ROOT))})

    # fig 2: residual vs dt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, marker in (("elastic", "o"), ("fracture", "s")):
        dts = [v["dt"] for v in sweep[name].values()]
        res = [v["summary"]["residual_relative_max"] for v in sweep[name].values()]
        order = np.argsort(dts)
        ax.loglog(np.array(dts)[order], np.array(res)[order], marker + "-",
                  label=f"{name} impact")
    ax.axhline(1e-2, color="orange", ls="--", lw=1.0, label="1% acceptance")
    ax.axhline(2e-2, color="red", ls=":", lw=1.0, label="2% acceptance (fracture)")
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"max $\varepsilon_E$ [-]")
    ax.set_title("Timestep sensitivity of the energy residual")
    ax.invert_xaxis()
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "fig2_residual_vs_dt.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig2_residual_vs_dt.png",
                 "caption": "Maximum closure residual versus timestep: halving dt never "
                            "increases the residual (acceptance criterion h).",
                 "path": str(p.relative_to(ROOT))})

    # fig 3: energy budget decomposition at the end of the run
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["E_k", "E_s", "E_f", "E_v"]
    vals = [sim.ledger.E_k, sim.ledger.E_s, sim.ledger.E_f, sim.ledger.E_v]
    bars = ax.bar(labels, vals, color=["#1f77b4", "#2ca02c", "#d62728", "#9467bd"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}", ha="center",
                va="bottom", fontsize=9)
    ax.axhline(sim.reference_total_J, color="k", ls="--", lw=1.0,
               label=f"E_total(0) = {sim.reference_total_J:.4g} J")
    ax.set_ylabel("energy [J]")
    ax.set_title("Final energy budget of the fracturing impact")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig3_energy_budget.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig3_energy_budget.png",
                 "caption": "Final partition of the initial kinetic energy over the four "
                            "ledgers (fracture case).",
                 "path": str(p.relative_to(ROOT))})

    # fig 4: damage/speed field cloud of the floe before and after impact
    lat = build_bonded_lattice(3, 3, 3, RADIUS, DENSITY)
    pos0 = lat["positions"].copy()
    pos0[:, 0] += GAP + RADIUS - pos0[:, 0].min()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    speed0 = np.zeros(pos0.shape[0])
    axes[0].scatter(pos0[:, 0] * 1e3, pos0[:, 1] * 1e3, c=speed0, s=90,
                    cmap="viridis", vmin=0, vmax=4, edgecolor="k")
    axes[0].axvline(0, color="k", lw=3)
    axes[0].set_title("t = 0 (undeformed)")
    speed = np.linalg.norm(sim.velocities, axis=1)
    sc = axes[1].scatter(sim.positions[:, 0] * 1e3, sim.positions[:, 1] * 1e3,
                         c=speed, s=90, cmap="viridis", vmin=0, vmax=4,
                         edgecolor="k")
    axes[1].axvline(0, color="k", lw=3)
    axes[1].set_title(f"t = {sim.time * 1e3:.1f} ms "
                      f"({sim.n_broken_total} bonds broken)")
    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
    cb.set_label("|v| [m/s]")
    fig.suptitle("Ice floe vs rigid structure: speed field before / after impact "
                 "(x-y projection, z collapsed)")
    fig.savefig(OUT / "fig4_floe_speed_field.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    figs.append({"name": "fig4_floe_speed_field.png",
                 "caption": "Particle speed field (x-y projection) of the 3x3x3 ice floe "
                            "before and after the 2 m/s impact with the rigid structure.",
                 "path": str((OUT / "fig4_floe_speed_field.png").relative_to(ROOT))})

    # fig 5: E_v vs analytic restitution loss + trapezoid reference
    fig, ax = plt.subplots(figsize=(7, 5))
    e_vals = [sphere["restitution"]]
    ax.bar(["analytic\n0.5 m v0^2 (1-e^2)", "ledger E_v", "trapezoid integral"],
           [sphere["E_v_analytic"], sphere["E_v_ledger"], sphere["E_v_trapezoid"]],
           color=["#7f7f7f", "#9467bd", "#c5b0d5"])
    ax.set_ylabel("dissipated energy [J]")
    ax.set_title(f"Single ice sphere vs wall (e = {sphere['restitution']}): "
                 f"viscous ledger vs analytic, err = "
                 f"{100 * sphere['E_v_error_relative']:.3f}%")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig5_viscous_analytic.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig5_viscous_analytic.png",
                 "caption": "Viscous dissipation ledger against the analytic restitution "
                            "loss for a single-sphere wall impact.",
                 "path": str(p.relative_to(ROOT))})
    return figs


# ---------------------------------------------------------------------------
def main() -> dict:
    dt_levels = (1.0e-5, 5.0e-6, 2.5e-6)
    sweep = run_dt_sweep(dt_levels)

    base_elastic = sweep["elastic"][f"{dt_levels[1]:.1e}"]["_sim"]
    base_fracture = sweep["fracture"][f"{dt_levels[1]:.1e}"]["_sim"]
    ref_checks_elastic = run_reference_checks(base_elastic)
    ref_checks_fracture = run_reference_checks(base_fracture)
    sphere = sphere_restitution_case(0.8)
    oblique = oblique_case()
    neg = negative_control()

    def strip(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "_sim"}

    criteria = {
        "a_kinetic_energy_error_lt_0.1pct": {
            "value": ref_checks_fracture["E_k_error_relative"],
            "threshold": 1.0e-3,
            "pass": ref_checks_fracture["E_k_error_relative"] < 1.0e-3,
            "detail": ref_checks_fracture,
        },
        "b_contact_strain_energy_error_lt_0.5pct": {
            "value": ref_checks_fracture["E_s_error_relative"],
            "threshold": 5.0e-3,
            "pass": ref_checks_fracture["E_s_error_relative"] < 5.0e-3,
            "detail": ref_checks_fracture,
        },
        "b2_tangential_spring_energy_error_lt_0.5pct": {
            "value": oblique["E_s_contact_tangential_error_relative"],
            "threshold": 5.0e-3,
            "pass": oblique["E_s_contact_tangential_error_relative"] < 5.0e-3,
            "detail": oblique,
        },
        "c_fracture_energy_error_lt_1pct": {
            "value": sweep["fracture"][f"{dt_levels[1]:.1e}"]["E_f_error_relative"],
            "threshold": 1.0e-2,
            "pass": sweep["fracture"][f"{dt_levels[1]:.1e}"]["E_f_error_relative"] < 1e-2,
            "detail": {k: v for k, v in sweep["fracture"][f"{dt_levels[1]:.1e}"].items()
                       if k != "_sim"},
        },
        "d_viscous_dissipation_error_lt_2pct": {
            "value": sweep["fracture"][f"{dt_levels[1]:.1e}"]["E_v_error_relative"],
            "threshold": 2.0e-2,
            "pass": sweep["fracture"][f"{dt_levels[1]:.1e}"]["E_v_error_relative"] < 2e-2,
            "detail": {"trapezoid": sweep["fracture"][f"{dt_levels[1]:.1e}"]["E_v_reference"],
                       "analytic_restitution": sphere},
        },
        "e_elastic_residual_lt_1pct": {
            "value": sweep["elastic"][f"{dt_levels[1]:.1e}"]["summary"]["residual_relative_max"],
            "threshold": 1.0e-2,
            "pass": sweep["elastic"][f"{dt_levels[1]:.1e}"]["summary"]["residual_relative_max"] < 1e-2,
        },
        "f_fracture_residual_lt_2pct": {
            "value": sweep["fracture"][f"{dt_levels[1]:.1e}"]["summary"]["residual_relative_max"],
            "threshold": 2.0e-2,
            "pass": sweep["fracture"][f"{dt_levels[1]:.1e}"]["summary"]["residual_relative_max"] < 2e-2,
        },
        "g_negative_control_zero_velocity": {
            "value": neg["max_abs_total_over_run"],
            "threshold": 0.0,
            "pass": neg["pass"],
            "detail": neg,
        },
        "h_dt_halving_does_not_increase_residual": {
            "pass": True,
            "detail": {},
        },
    }

    for name in ("elastic", "fracture"):
        dts = sorted(dt_levels, reverse=True)
        residuals = [sweep[name][f"{dt:.1e}"]["summary"]["residual_relative_max"]
                     for dt in dts]
        monotone = all(b <= a * (1.0 + 1.0e-3) for a, b in zip(residuals, residuals[1:]))
        criteria["h_dt_halving_does_not_increase_residual"]["value"] = max(
            criteria["h_dt_halving_does_not_increase_residual"].get("value", 0.0),
            max(residuals))
        criteria["h_dt_halving_does_not_increase_residual"]["detail"][name] = {
            "dts": dts,
            "residuals": residuals,
            "monotone": monotone,
        }
        criteria["h_dt_halving_does_not_increase_residual"]["pass"] &= monotone

    figures = make_figures(sweep, sphere, oblique, base_fracture)

    all_pass = all(c["pass"] for c in criteria.values())
    report = {
        "node": "P304W2-N06",
        "capability": "ice-structure contact-collision four-ledger energy conservation audit",
        "model": {
            "particles": int(base_fracture.positions.shape[0]),
            "bonds": int(base_fracture.bond_i.size),
            "radius_m": RADIUS,
            "density_kgm3": DENSITY,
            "mass_kg": base_fracture.mass,
            "contact": {
                "k_normal_N_m": ELASTIC_CONTACT.k_normal,
                "kt_ratio": ELASTIC_CONTACT.kt_ratio,
                "restitution_elastic": ELASTIC_CONTACT.restitution,
                "restitution_fracture": FRACTURE_CONTACT.restitution,
            },
            "elastic_bond": {
                "stiffness_N_m": ELASTIC_BOND.stiffness,
                "area_m2": ELASTIC_BOND.area,
                "tensile_strength_Pa": ELASTIC_BOND.tensile_strength,
                "fracture_energy_J_m2": ELASTIC_BOND.fracture_energy,
                "elastic_limit_m": ELASTIC_BOND.elastic_limit,
                "separation_m": ELASTIC_BOND.separation,
            },
            "fracture_bond": {
                "stiffness_N_m": FRACTURE_BOND.stiffness,
                "area_m2": FRACTURE_BOND.area,
                "tensile_strength_Pa": FRACTURE_BOND.tensile_strength,
                "fracture_energy_J_m2": FRACTURE_BOND.fracture_energy,
                "elastic_limit_m": FRACTURE_BOND.elastic_limit,
                "separation_m": FRACTURE_BOND.separation,
                "surface_energy_J": FRACTURE_BOND.surface_energy,
            },
            "impact_velocity_m_s": abs(IMPACT[0]),
            "initial_gap_m": GAP,
            "duration_s": DURATION,
        },
        "criteria": criteria,
        "all_criteria_pass": all_pass,
        "sweep": {name: {dt: strip(v) for dt, v in cases.items()}
                  for name, cases in sweep.items()},
        "reference_checks": {"elastic": ref_checks_elastic,
                             "fracture": ref_checks_fracture},
        "single_sphere_restitution": sphere,
        "oblique_impact": oblique,
        "negative_control": neg,
        "figures": figures,
    }
    path = OUT / "acceptance.json"
    path.write_text(json.dumps(report, indent=2, default=_jsonable))
    report["acceptance_json"] = str(path.relative_to(ROOT))
    report["acceptance_json_sha256"] = _sha256(path)

    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("value", "threshold", "pass")}
                      for k, v in criteria.items()}, indent=2, default=_jsonable))
    print("\nall_criteria_pass:", all_pass)
    print("acceptance:", report["acceptance_json"])
    return report


if __name__ == "__main__":
    main()