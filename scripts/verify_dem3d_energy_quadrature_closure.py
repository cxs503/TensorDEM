#!/usr/bin/env python
"""Entry point: verify + benchmark the exact-quadrature platen energy ledger.

Runs three independent checks of ``EnergyAuditedIceDEM3D`` and writes a
machine-readable artifact to
``runs/dem3d_energy_quadrature_closure.json``:

1. **One-step identity** — the exact ledger arm must equal the independently
   recomputed boundary potential increment ``U(t+dt) - U(t)`` at fixed particle
   positions (machine-precision reference, tolerance ``1e-9`` relative).
2. **Closed-form defect** — the measured gap ``legacy - exact`` must equal the
   analytic left-endpoint defect ``sum(1/2 k n dz^2)`` of
   :mod:`tensordem.dem3d_energy_quadrature` (tolerance 5 %).
3. **Convergence order** — the per-step defect is second order in ``dt``; a
   least-squares fit of ``log(defect)`` vs ``log(dt)`` over three refinement
   levels must return ``2.0`` within 0.05 (reference value ``2.0`` from the
   closed form).

Plus an integration block that drives the true-triaxial protocol
(:func:`tensordem.dem3d_triaxial.run_triaxial`) and asserts its
``energy_ledger`` carries the predicted/measured defect pair within tolerance.

Exit status is 0 only if every predeclared gate passes; the artifact is written
either way so failures are auditable (honest FAIL, no silent pass).

Reproduce::

    cd <TensorDEM checkout>
    PYTHONPATH=src python scripts/verify_dem3d_energy_quadrature_closure.py

Add ``--emit-doc`` to also regenerate the benchmark section of
``docs/dem3d_energy_quadrature_closure.md`` from the artifact (no hand-copied
numbers).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D
from tensordem.dem3d_energy_quadrature import (
    platen_defect_components,
    quadrature_order,
    relative_error,
)
from tensordem.dem3d_triaxial import TriaxialConfig, run_triaxial

torch.set_num_threads(1)

ONE_STEP_TOL = 1e-9         # exact arm vs independent potential increment
DEFECT_TOL = 5e-2           # closed-form defect vs engine measurement
ORDER_REF = 2.0             # second order per step (closed form)
ORDER_TOL = 5e-2            # |measured order - 2.0| <= 0.05
TRIAXIAL_TOL = 5e-2         # integration: triaxial ledger defect prediction

DT_LEVELS = (8.0e-5, 4.0e-5, 2.0e-5)


def _make_sim(dt: float) -> EnergyAuditedIceDEM3D:
    """Two-layer block between two driven platens, no tool (axial only)."""
    cfg = DEM3DConfig(nx=4, ny=4, nz=3, radius=0.025, top_platen_enabled=True,
                      bottom_platen_enabled=True, top_platen_velocity=-2.0e-3)
    sim = EnergyAuditedIceDEM3D(cfg)
    sim.dt = dt
    return sim


def _boundaries(sim: EnergyAuditedIceDEM3D) -> dict:
    """Kinematic boundary state at the current solver time."""
    cfg = sim.config
    return {
        "top_z": (sim.top_platen_start_z + sim.time * cfg.top_platen_velocity
                  if cfg.top_platen_enabled else None),
        "bottom_z": (sim.bottom_platen_start_z + sim.time * cfg.bottom_platen_velocity
                     if cfg.bottom_platen_enabled else None),
    }


def check_one_step_identity() -> dict:
    """Exact arm vs an independently recomputed potential increment."""
    sim = _make_sim(DT_LEVELS[0])
    for _ in range(30):
        sim.step()
    cfg = sim.config
    b = _boundaries(sim)
    top_z1 = b["top_z"] + sim.dt * cfg.top_platen_velocity
    bottom_z1 = b["bottom_z"] + sim.dt * cfg.bottom_platen_velocity
    u0 = sim._platen_potential(sim.positions, top_z=b["top_z"], bottom_z=b["bottom_z"])
    u1 = sim._platen_potential(sim.positions, top_z=top_z1, bottom_z=bottom_z1)
    before = sim.cumulative_platen_work_exact_J
    sim.step()
    measured = sim.cumulative_platen_work_exact_J - before
    reference = u1 - u0
    err = relative_error(measured, reference)
    return {
        "quantity": "one-step exact platen work vs potential increment",
        "reference_J": reference,
        "measured_J": measured,
        "rel_error": err,
        "tolerance": ONE_STEP_TOL,
        "pass": err <= ONE_STEP_TOL,
    }


def measure_deficit(dt: float, steps: int = 200, warm: int = 50) -> dict:
    """Run ``steps`` after a warm-up and return the ledger bookkeeping."""
    sim = _make_sim(dt)
    for _ in range(warm):
        sim.step()
    l0 = sim.cumulative_platen_work_J
    e0 = sim.cumulative_platen_work_exact_J
    p0 = sim.cumulative_platen_deficit_predicted_J
    for _ in range(steps):
        sim.step()
    measured = sim.cumulative_platen_work_exact_J - e0
    legacy = sim.cumulative_platen_work_J - l0
    predicted = sim.cumulative_platen_deficit_predicted_J - p0
    return {
        "dt_s": dt,
        "steps": steps,
        "exact_work_J": measured,
        "legacy_work_J": legacy,
        "deficit_measured_J": measured - legacy,
        "deficit_predicted_J": predicted,
        "residual_legacy_J": sim.energy_balance_residual_J,
        "residual_exact_J": sim.energy_balance_residual_exact_J,
        "balanced_steps": sim.step_count,
    }


DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / \
    "dem3d_energy_quadrature_closure.md"
DOC_BEGIN = "<!-- BEGIN QUADRATURE-BENCH -->"
DOC_END = "<!-- END QUADRATURE-BENCH -->"


def _doc_section(artifact: dict) -> str:
    """Markdown benchmark section rendered straight from the artifact."""
    lines = [DOC_BEGIN, "",
             "## 7. Closed-form defect + reproducible benchmark (rework m14607)", "",
             "The per-step left-endpoint defect is available in closed form",
             "(``tensordem.dem3d_energy_quadrature``) and is accumulated by the",
             "engine as ``cumulative_platen_deficit_predicted_J``. It is compared",
             "below against the measured ledger gap ``exact - legacy``.", "",
             f"Artifact: `runs/dem3d_energy_quadrature_closure.json` "
             f"(commit `{artifact['git_commit'][:12]}`, "
             f"torch {artifact['environment']['torch']}); "
             f"overall **{'PASS' if artifact['overall_pass'] else 'FAIL'}** in "
             f"{artifact['wall_seconds']} s.", "",
             "| quantity | reference | measured | rel. error | gate | result |",
             "|---|---|---|---|---|---|"]
    for row in artifact["benchmark"]:
        ref = row.get("reference_J", row.get("reference"))
        got = row.get("measured_J", row.get("measured"))
        err = row.get("rel_error", row.get("abs_error"))
        lines.append(f"| {row['quantity']} | {ref:.6e} | {got:.6e} | {err:.3e} "
                     f"| {row['tolerance']:.0e} | {'PASS' if row['pass'] else 'FAIL'} |")
    damped = artifact.get("triaxial_damped_informational", {})
    lines += [
        "",
        "Timestep convergence of the per-step defect (quadratic in ``dz``, i.e.",
        "order 2 in ``dt``; the measured slope is "
        f"{[r for r in artifact['benchmark'] if 'order' in r['quantity']][0]['measured']:.6f}):",
        "",
        "| dt (s) | measured deficit (J) | closed form (J) | rel. error |",
        "|---|---|---|---|",
    ]
    for r in artifact["runs"]:
        lines.append(
            f"| {r['dt_s']:.1e} | {r['deficit_measured_J']:.6e} "
            f"| {r['deficit_predicted_J']:.6e} "
            f"| {relative_error(r['deficit_measured_J'], r['deficit_predicted_J']):.3e} |")
    if damped:
        lines += [
            "",
            "Scope note (informational, not gated): with the library default platen",
            f"dashpot (`platen_damping=0.5`) the triaxial ledger measures "
            f"{damped['measured_J']:.3e} J against a spring-only prediction of "
            f"{damped['reference_J']:.3e} J — the dashpot adds work to the legacy arm",
            "that the penalty potential does not contain, so the closed form is",
            "declared inapplicable (`legacy_deficit_applicable: false`) there.",
        ]
    lines += [
        "",
        "Figure (regenerated by the same command):",
        "`runs/dem3d_energy_quadrature_closure.png` — left: per-step defect vs `dt`",
        "against the closed form; right: relative error against the 5 % gate.",
        "",
        "Reproduce:",
        "",
        "```",
        "PYTHONPATH=src python scripts/verify_dem3d_energy_quadrature_closure.py --emit-doc",
        "```",
        "",
        DOC_END,
    ]
    return "\n".join(lines)


def emit_figure(artifact: dict) -> pathlib.Path:
    """Convergence figure (measured vs closed form) next to the artifact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = artifact["runs"]
    dt = [r["dt_s"] for r in rows]
    meas = [r["deficit_measured_J"] for r in rows]
    pred = [r["deficit_predicted_J"] for r in rows]
    order = [b for b in artifact["benchmark"] if "order" in b["quantity"]][0]["measured"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(dt, meas, "o-", label="measured (exact - legacy)")
    ax[0].loglog(dt, pred, "s--", label="closed form sum(1/2 k n dz^2)")
    ax[0].set_xlabel("dt [s]"); ax[0].set_ylabel("per-step defect [J]")
    ax[0].set_title(f"platen quadrature defect, fitted slope = {order:.4f} (ref 2.0)")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
    rel = [relative_error(m, p) for m, p in zip(meas, pred)]
    ax[1].loglog(dt, rel, "o-", label="closed form vs engine")
    ax[1].axhline(DEFECT_TOL, color="r", ls="--", label="gate 5 %")
    ax[1].set_xlabel("dt [s]"); ax[1].set_ylabel("relative error")
    ax[1].set_title("agreement vs gate")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "runs" / \
        "dem3d_energy_quadrature_closure.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def emit_doc(artifact: dict) -> None:
    text = DOC.read_text() if DOC.exists() else ""
    section = _doc_section(artifact)
    if DOC_BEGIN in text and DOC_END in text:
        head, rest = text.split(DOC_BEGIN, 1)
        _, tail = rest.split(DOC_END, 1)
        text = head + section + tail
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    DOC.write_text(text)


def main() -> int:
    torch.manual_seed(0)
    started = time.time()
    rows = [measure_deficit(dt) for dt in DT_LEVELS]

    defect_rows = []
    for r in rows:
        err = relative_error(r["deficit_measured_J"], r["deficit_predicted_J"])
        defect_rows.append({
            "quantity": f"closed-form left-endpoint defect at dt={r['dt_s']:.1e} s",
            "reference_J": r["deficit_predicted_J"],
            "measured_J": r["deficit_measured_J"],
            "rel_error": err,
            "tolerance": DEFECT_TOL,
            "pass": err <= DEFECT_TOL,
        })

    measured_order = quadrature_order(
        [(r["dt_s"], r["deficit_measured_J"]) for r in rows])
    order_row = {
        "quantity": "per-step defect convergence order (log-log slope)",
        "reference": ORDER_REF,
        "measured": measured_order,
        "abs_error": abs(measured_order - ORDER_REF),
        "tolerance": ORDER_TOL,
        "pass": abs(measured_order - ORDER_REF) <= ORDER_TOL,
    }

    identity_row = check_one_step_identity()

    triax_kw = dict(loading_speed=0.002, confine_steps=40, max_axial_steps=120,
                    peak_min_stress=1e9)
    # Gated case: damping-free platens (the closed form covers the spring part
    # only) with an active lateral pressure servo, so the lateral defect term is
    # exercised as well.
    triax = run_triaxial(TriaxialConfig(platen_damping=0.0, sigma_x=0.5e6,
                                       sigma_y=0.5e6, **triax_kw))
    led = triax["energy_ledger"]
    triax_err = relative_error(led["legacy_deficit_measured_J"],
                               led["legacy_deficit_predicted_J"])
    triax_row = {
        "quantity": "triaxial + lateral servo: measured vs predicted defect",
        "reference_J": led["legacy_deficit_predicted_J"],
        "measured_J": led["legacy_deficit_measured_J"],
        "rel_error": triax_err,
        "tolerance": TRIAXIAL_TOL,
        "pass": bool(led["pass_legacy_deficit"] is True
                     and triax_err <= TRIAXIAL_TOL),
        "closure_residual_relative": led["closure_residual_relative"],
        "closure_residual_relative_exact": led["closure_residual_relative_exact"],
        "pass_exact": led["pass_exact"],
    }
    # Informational: the library default dashpot (platen_damping=0.5) adds work
    # to the legacy arm that the penalty potential does not contain, so the
    # spring-only closed form is expected to disagree; recorded, not gated.
    damped = run_triaxial(TriaxialConfig(**triax_kw))
    dled = damped["energy_ledger"]
    damped_info = {
        "quantity": "triaxial default dashpot (informational, not gated)",
        "reference_J": dled["legacy_deficit_predicted_J"],
        "measured_J": dled["legacy_deficit_measured_J"],
        "rel_error": relative_error(dled["legacy_deficit_measured_J"],
                                    dled["legacy_deficit_predicted_J"]),
        "applicable": dled["legacy_deficit_applicable"],
        "note": "dashpot work inflates the legacy arm; spring-only closed form "
                "is declared inapplicable for platen_damping != 0",
    }

    bench = [identity_row] + defect_rows + [order_row, triax_row]
    ok = all(row["pass"] for row in bench)
    try:
        # Commit that last changed the implementation (stable across artifact
        # refresh commits): git log -1 -- src/tensordem/
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "src/tensordem"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"

    root = pathlib.Path(__file__).resolve().parents[1]
    impl_sha256 = {
        str((root / rel).relative_to(root)): hashlib.sha256(
            (root / rel).read_bytes()).hexdigest()
        for rel in ("src/tensordem/dem3d_energy_quadrature.py",
                    "src/tensordem/dem3d_energy.py",
                    "src/tensordem/dem3d_triaxial.py")
    }
    artifact = {
        "task": "dem3d_energy_quadrature_closure",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Commit that last touched src/tensordem (stable across artifact-only
        # refreshes); impl_sha256 binds the exact file contents.
        "git_commit": commit,
        "impl_sha256": impl_sha256,
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "threads": torch.get_num_threads()},
        "config": {
            "particles": 4 * 4 * 3,
            "radius_m": 0.025,
            "platen_stiffness_N_per_m": DEM3DConfig().platen_stiffness,
            "platen_damping": DEM3DConfig().platen_damping,
            "top_platen_velocity_m_per_s": -2.0e-3,
            "dt_levels_s": list(DT_LEVELS),
            "warmup_steps": 50,
            "measured_steps": 200,
            "triaxial": "loading_speed=0.002 confine_steps=40 max_axial_steps=120, "
                    "gated case platen_damping=0 sigma_x=sigma_y=0.5e6",
        },
        "predeclared_thresholds": {
            "one_step_identity_rel_error_max": ONE_STEP_TOL,
            "closed_form_defect_rel_error_max": DEFECT_TOL,
            "convergence_order_reference": ORDER_REF,
            "convergence_order_abs_error_max": ORDER_TOL,
            "triaxial_defect_rel_error_max": TRIAXIAL_TOL,
        },
        "benchmark": bench,
        "runs": rows,
        "triaxial_damped_informational": damped_info,
        "triaxial_energy_ledger": {
            k: led[k] for k in ("input_work_J", "input_work_exact_J",
                                "closure_residual_relative",
                                "closure_residual_relative_exact",
                                "pass", "pass_exact",
                                "legacy_deficit_predicted_J",
                                "legacy_deficit_measured_J",
                                "legacy_deficit_relative_error",
                                "legacy_deficit_applicable",
                                "pass_legacy_deficit")},
        "overall_pass": ok,
        "wall_seconds": round(time.time() - started, 2),
    }

    out = pathlib.Path(__file__).resolve().parents[1] / "runs" / \
        "dem3d_energy_quadrature_closure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

    print(f"{'quantity':64s} {'ref':>12s} {'got':>12s} {'err':>10s}  pass")
    for row in bench:
        ref = row.get("reference_J", row.get("reference"))
        got = row.get("measured_J", row.get("measured"))
        err = row.get("rel_error", row.get("abs_error"))
        print(f"{row['quantity'][:64]:64s} {ref:12.6e} {got:12.6e} {err:10.3e}"
              f"  {'PASS' if row['pass'] else 'FAIL'}")
    if "--emit-doc" in sys.argv[1:]:
        emit_doc(artifact)
        print(f"doc section refreshed: {DOC}")
        print(f"figure: {emit_figure(artifact)}")
    print(f"\nartifact: {out}")
    print(f"overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())