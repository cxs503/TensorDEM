#!/usr/bin/env python3
"""Contact-impact analytic benchmark for the DEM3D linear spring-dashpot contact.

Produces (under ``artifacts/dem3d_contact_impact/``):
  * ``restitution_sweep.csv``      -- realised vs clamped/unclamped analytic, per e_target
  * ``timestep_convergence.csv``   -- solver error under dt refinement
  * ``mapping_repair.csv``         -- realised restitution with the corrected mapping
  * ``summary.json``               -- machine-readable verdicts
  * ``contact_impact_benchmark.png``

Run from the repository root:

    PYTHONPATH=src python scripts/benchmark_dem3d_contact_impact.py
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.contact_impact import (  # noqa: E402
    clamped_restitution,
    impact_analytic,
    pair_impact,
    restitution_to_damping_clamped,
    rk4_clamped_restitution,
    unclamped_restitution,
)

OUT = pathlib.Path("artifacts/dem3d_contact_impact")
E_TARGETS = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
CONV_TARGETS = (0.7, 0.5, 0.3)
DT_SCALES = (1.0, 0.5, 0.25)
ACCURACY_BAND = 0.03  # 3 % engineering acceptance band


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sweep() -> list[dict]:
    rows = []
    for e_t in E_TARGETS:
        r = pair_impact(e_target=e_t)
        rows.append({
            "e_target": e_t,
            "e_measured": r.e_measured,
            "e_unclamped_analytic": r.e_unclamped,
            "e_clamped_analytic": r.e_clamped,
            "err_vs_clamped_pct": 100.0 * r.e_error_vs_clamped,
            "err_vs_unclamped_pct": 100.0 * r.e_error_vs_unclamped,
            "contact_time_force_ms": 1e3 * r.contact_time_force,
            "contact_time_force_analytic_ms": 1e3 * r.contact_time_force_analytic,
            "contact_time_force_err_pct": 100.0 * r.contact_time_force_error,
            "contact_time_geometric_ms": 1e3 * r.contact_time,
            "contact_time_geometric_analytic_ms": 1e3 * r.contact_time_analytic,
            "contact_time_geometric_err_pct": 100.0 * r.contact_time_error,
            "max_overlap_m": r.max_overlap,
            "momentum_error_kg_m_s": r.momentum_error,
            "force_law_residual": r.force_law_residual,
            "dt_s": r.dt,
            "force_steps": r.force_steps,
        })
        print(f"  e_t={e_t:<5} e_meas={r.e_measured:.5f} clamped={r.e_clamped:.5f} "
              f"err_clamped={100*r.e_error_vs_clamped:+6.3f}%  err_unclamped={100*r.e_error_vs_unclamped:+7.3f}%",
              flush=True)
    return rows


def convergence() -> list[dict]:
    rows = []
    for e_t in CONV_TARGETS:
        for ds in DT_SCALES:
            r = pair_impact(e_target=e_t, dt_scale=ds)
            rows.append({
                "e_target": e_t,
                "dt_scale": ds,
                "dt_s": r.dt,
                "force_steps": r.force_steps,
                "e_measured": r.e_measured,
                "e_clamped_analytic": r.e_clamped,
                "err_vs_clamped_pct": 100.0 * r.e_error_vs_clamped,
                "contact_time_force_err_pct": 100.0 * r.contact_time_force_error,
            })
            print(f"  conv e_t={e_t:<5} ds={ds:<5} dt={r.dt:.3e} err={100*r.e_error_vs_clamped:+6.3f}%",
                  flush=True)
    return rows


def mapping_repair(sweep_rows) -> list[dict]:
    """Realised restitution with the corrected mapping; counterfactual from the sweep."""
    old_err = {r["e_target"]: r["err_vs_unclamped_pct"] for r in sweep_rows}
    rows = []
    for e_t in (0.9, 0.7, 0.5, 0.3):
        r = pair_impact(e_target=e_t, damping="clamped_mapping")
        rows.append({
            "e_target": e_t,
            "e_measured": r.e_measured,
            "rel_err_corrected_pct": 100.0 * (r.e_measured / e_t - 1.0),
            "rel_err_old_mapping_pct": old_err[e_t],
        })
        print(f"  repair e_t={e_t}: measured={r.e_measured:.5f} "
              f"corrected_err={100*(r.e_measured/e_t-1):+.3f}% "
              f"(old mapping realises {old_err[e_t]:+.2f}% off target)", flush=True)
    return rows


def rate_independence() -> list[dict]:
    rows = []
    for v0 in (0.25, 0.5, 1.0):
        r = pair_impact(e_target=0.5, v0=v0)
        rows.append({"v0_m_s": v0, "e_measured": r.e_measured,
                     "e_clamped_analytic": r.e_clamped,
                     "err_vs_clamped_pct": 100.0 * r.e_error_vs_clamped})
        print(f"  rate v0={v0}: e={r.e_measured:.5f} err={100*r.e_error_vs_clamped:+.3f}%", flush=True)
    return rows


def derivation_check() -> list[dict]:
    rows = []
    for zeta in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.69):
        closed = clamped_restitution(zeta)
        rk4 = rk4_clamped_restitution(zeta)
        rows.append({"zeta": zeta, "e_clamped_closed_form": closed,
                     "e_clamped_rk4": rk4, "abs_diff": abs(closed - rk4),
                     "e_unclamped": unclamped_restitution(zeta)})
    return rows


def figure(sweep_rows, conv_rows) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

    ax = axes[0, 0]
    e_t = [r["e_target"] for r in sweep_rows]
    ax.plot(e_t, [r["e_clamped_analytic"] for r in sweep_rows], "k-", lw=2,
            label="clamped analytic $F=\\max(0,k\\delta+c\\dot\\delta)$")
    ax.plot(e_t, [r["e_unclamped_analytic"] for r in sweep_rows], "b--", lw=2,
            label="unclamped analytic (current mapping)")
    ax.plot(e_t, [r["e_measured"] for r in sweep_rows], "ro", ms=6, label="DEM3D solver")
    ax.set_xlabel("target restitution $e_{target}$ of mapping")
    ax.set_ylabel("realised restitution $e$")
    ax.set_title("(a) realised restitution vs analytic references")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.axhspan(-3, 3, color="green", alpha=0.12, label="±3% acceptance band")
    ax.plot(e_t, [r["err_vs_clamped_pct"] for r in sweep_rows], "gs-", label="vs clamped analytic")
    ax.plot(e_t, [r["err_vs_unclamped_pct"] for r in sweep_rows], "b^--",
            label="vs unclamped analytic (mapping basis)")
    ax.set_xlabel("target restitution $e_{target}$")
    ax.set_ylabel("relative error of $e$ [%]")
    ax.set_title("(b) solver error: correct reference matters")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(e_t, [r["contact_time_force_ms"] for r in sweep_rows], "ro-", label="force contact (solver)")
    ax.plot(e_t, [r["contact_time_force_analytic_ms"] for r in sweep_rows], "k-",
            label="force contact (clamped analytic, $\\theta^*/\\omega_d$)")
    ax.plot(e_t, [r["contact_time_geometric_ms"] for r in sweep_rows], "bs--", label="overlap duration (solver)")
    ax.plot(e_t, [r["contact_time_geometric_analytic_ms"] for r in sweep_rows], "k:", lw=1.8,
            label="overlap duration (clamped analytic)")
    ax.set_xlabel("target restitution $e_{target}$")
    ax.set_ylabel("contact duration [ms]")
    ax.set_title("(c) contact durations: two distinct definitions")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for target in CONV_TARGETS:
        rows = [r for r in conv_rows if r["e_target"] == target]
        ax.loglog([r["dt_s"] for r in rows], [abs(r["err_vs_clamped_pct"]) for r in rows],
                  "o-", label=f"$e_{{target}}$={target}")
    ax.set_xlabel("time step $\\Delta t$ [s]")
    ax.set_ylabel("$|e$ error vs clamped analytic$|$ [%]")
    ax.set_title("(d) time-step convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("DEM3D contact-impact analytic benchmark (equal-mass head-on collision)", fontsize=12)
    fig.tight_layout()
    path = OUT / "contact_impact_benchmark.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _md_table(rows, columns) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(r[c] if isinstance(r[c], str) else f"{r[c]:.4g}" for c in columns) + " |")
    return "\n".join([head, sep] + body)


def render_doc(summary: dict, path: pathlib.Path) -> None:
    """Write the benchmark protocol + result tables straight from summary.json."""
    v = summary["verdicts"]
    sweep = summary["restitution_sweep"]
    conv = summary["timestep_convergence"]
    repair = summary["mapping_repair"]
    rate = summary["rate_independence"]
    deriv = summary["derivation_check"]

    sweep_tbl = _md_table(
        [dict(r) for r in sweep],
        ["e_target", "e_measured", "e_clamped_analytic", "e_unclamped_analytic",
         "err_vs_clamped_pct", "err_vs_unclamped_pct"],
    )
    conv_tbl = _md_table([dict(r) for r in conv],
                         ["e_target", "dt_scale", "dt_s", "force_steps", "e_measured",
                          "e_clamped_analytic", "err_vs_clamped_pct"])
    repair_tbl = _md_table([dict(r) for r in repair],
                           ["e_target", "e_measured", "rel_err_corrected_pct", "rel_err_old_mapping_pct"])
    rate_tbl = _md_table([dict(r) for r in rate], ["v0_m_s", "e_measured", "e_clamped_analytic",
                                                   "err_vs_clamped_pct"])
    dur_tbl = _md_table([dict(r) for r in sweep],
                        ["e_target", "contact_time_force_ms", "contact_time_force_analytic_ms",
                         "contact_time_force_err_pct", "contact_time_geometric_ms",
                         "contact_time_geometric_analytic_ms", "contact_time_geometric_err_pct"])
    deriv_tbl = _md_table([dict(r) for r in deriv],
                          ["zeta", "e_clamped_closed_form", "e_clamped_rk4", "abs_diff", "e_unclamped"])

    def yn(x):
        return "YES" if x else "NO"

    text = f"""# DEM3D contact-impact analytic validation

Anchors: `scripts/benchmark_dem3d_contact_impact.py`,
`tests/test_dem3d_contact_impact.py`, `src/tensordem/contact_impact.py`,
artifacts in `artifacts/dem3d_contact_impact/`.
All numbers below are rendered from `summary.json`; do not edit by hand.

## 1. Physics background and case definition

The DEM3D normal contact is the linear spring-dashpot (Hookean) model truncated at
zero force:

    F_n = max(0, k_n * overlap - c_n * v_n)          (pair, tool and platen branches)

`overlap = 2R - |x_j - x_i|`, `v_n` is the relative normal velocity (negative while
closing), `k_n = contact_stiffness`, `c_n = contact_damping`.

Validation case: two equal free particles collide head-on in free flight
(`drag = 0`, all lattice bonds broken before impact, all other lattice sites parked
and fixed). This isolates one contact pair, so the reference is a 1-DOF problem and
the analytic solution is exact.

| parameter | value |
|---|---|
| particle radius R | 0.05 m |
| density | 900 kg/m^3 |
| particle mass m | 0.4712 kg (sphere, 4/3 pi R^3 rho) |
| effective mass m_eff | m/2 = 0.2356 kg |
| normal stiffness k_n | 5.0e4 N/m |
| damping c_n | from the restitution mapping under test |
| impact speed v0 | 0.5 m/s (each particle), relative speed 1.0 m/s |
| time step | solver `recommended_dt` x scale |

## 2. Method

Two independent analytic references are used, because they answer different
questions.

**Unclamped (free oscillator).** For `m_eff*delta'' + c*delta' + k*delta = 0` with
`delta(0)=0`, `delta'(0)=v0`, `zeta = c/(2*sqrt(k*m_eff))`, `omega_n = sqrt(k/m_eff)`:

    delta(t) = (v0/omega_d) exp(-zeta*omega_n*t) sin(omega_d*t),  omega_d = omega_n sqrt(1-zeta^2)
    e = exp(-pi*zeta/sqrt(1-zeta^2)),   t_contact = pi/omega_d

This is the identity inverted by `contact_tangential.restitution_to_damping`.

**Clamped (what the solver actually integrates).** The force vanishes when
`k*delta + c*delta' = 0`, i.e. at `theta* = omega_d*t*` satisfying
`(1-2*zeta^2) sin(theta*) + 2*zeta*sqrt(1-zeta^2) cos(theta*) = 0`:

    theta*   = atan2(2*zeta*sqrt(1-zeta^2), 2*zeta^2 - 1)      (pi/2 < theta* <= pi for zeta < 1/sqrt(2))
    e_clamped = -delta'(t*)/v0
    t_force   = theta*/omega_d
    t_overlap = t_force + delta(t*)/(e_clamped*v0)             (the pair keeps overlapping after the force clamps)

The closed form is cross-checked against an independent RK4 integration of the
clamped ODE over the whole parameter range (maximum absolute difference in
`e`: {summary['analytic_references']['cross_check_max_abs_diff']:.3e}).

Solver protocol: `IceDEM3D.step()` (semi-implicit Euler) until the pair separates;
the realised restitution is `e = -v_after/v0` of the lower particle, momentum drift is
`sum(v_z)` before vs after, and the assembled normal force is compared with the
documented law at a prescribed overlap.

## 3. Results

### 3.1 Restitution versus both analytic references

{sweep_tbl}

### 3.2 Contact duration (two distinct definitions)

{dur_tbl}

### 3.3 Time-step convergence

{conv_tbl}

### 3.4 Corrected clamped mapping

`restitution_to_damping_clamped` inverts the *clamped* model (bisection on zeta)
instead of the free-oscillator identity. Realised restitution against the target:

{repair_tbl}

### 3.5 Rate independence (linear model)

{rate_tbl}

### 3.6 Derivation cross-check

{deriv_tbl}

## 4. Verdict

| quantity | max abs error | within 3 % band |
|---|---|---|
| solver restitution vs clamped analytic | {v['solver_vs_clamped_analytic_max_abs_pct']:.3f} % | {yn(v['solver_vs_clamped_within_3pct'])} |
| solver restitution vs unclamped analytic (mapping basis) | {v['solver_vs_unclamped_analytic_max_abs_pct']:.3f} % | {yn(not v['current_mapping_bias_exceeds_3pct'])} |
| force-contact duration vs analytic | {v['force_contact_time_max_abs_pct']:.3f} % | {yn(v['force_contact_time_max_abs_pct'] < 3)} |
| overlap duration vs analytic | {v['geometric_contact_time_max_abs_pct']:.3f} % | {yn(v['geometric_contact_time_max_abs_pct'] < 3)} |
| corrected mapping, realised vs target | {v['corrected_mapping_max_abs_pct']:.3f} % | {yn(v['corrected_mapping_max_abs_pct'] < 3)} |
| existing (unclamped) mapping, realised vs target | {v['old_mapping_max_abs_pct']:.3f} % | {yn(v['old_mapping_max_abs_pct'] < 3)} |
| momentum drift (total) | {v['momentum_error_max']:.3e} kg m/s | exact zero |
| force-law residual (total) | {v['force_law_residual_max']:.3e} | machine precision |

Reading: the **solver is correct** and reproduces the clamped closed form to well
inside the 3 % band, converging under time-step refinement. The **mapping in
`contact_tangential` is the inaccurate part**: it inverts the free-oscillator
identity while the solver truncates the force at zero, so a requested
`e_target` is systematically overshot once dissipation is significant.

## 5. Notes, limitations and scope

1. The `max(0, ...)` truncation is a deliberate modelling choice (no tensile/adhesive
   contact), not a bug. The consequence is that the classic `e <-> gamma_n` identity
   is only valid in the weak-damping limit; the bias grows with `1-e_target`
   (see 3.1) and is already above the 3 % band at `e_target <= 0.7`.
2. The same truncation exists in the tool and platen branches, so the bias transfers
   to every normal contact in the solver; the pair case is used because it is the
   only configuration with an exact counterpart.
3. The closed-form `e_clamped` assumes `zeta < 1/sqrt(2)`; above that the force
   vanishes during the approach phase and the code falls back to RK4. Adhesive or
   Hertzian contacts are out of scope - this validates the linear law only.
4. The semi-implicit Euler integrator is first order; restitution errors are
   ~O(dt/t_contact). The default `recommended_dt` gives <1 % error for the cases
   tested here, but a soft-damping case with fewer than ~50 steps per contact must
   be re-checked.
5. Fracture, friction (Cundall-Strack) and tangential response are not exercised;
   the pair is bond-free and normal-only.
6. Recommended use: when a measured coefficient of restitution must be imposed,
   use `restitution_to_damping_clamped` (clamped solver) rather than
   `restitution_to_damping`; keep the latter only for the free-oscillator identity
   it documents.

Figure: `artifacts/dem3d_contact_impact/contact_impact_benchmark.png`.
"""
    path.write_text(text)
    print(f"doc        -> {path}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] derivation cross-check (closed form vs RK4)", flush=True)
    deriv = derivation_check()
    max_deriv = max(r["abs_diff"] for r in deriv)
    print(f"  max |closed-form - RK4| = {max_deriv:.3e}", flush=True)

    print("[2/5] restitution sweep (solver vs references)", flush=True)
    sweep_rows = sweep()
    print("[3/5] time-step convergence", flush=True)
    conv_rows = convergence()
    print("[4/5] corrected clamped mapping", flush=True)
    repair_rows = mapping_repair(sweep_rows)
    print("[5/5] rate independence + figure", flush=True)
    rate_rows = rate_independence()

    _write_csv(OUT / "restitution_sweep.csv", sweep_rows)
    _write_csv(OUT / "timestep_convergence.csv", conv_rows)
    _write_csv(OUT / "mapping_repair.csv", repair_rows)
    _write_csv(OUT / "rate_independence.csv", rate_rows)
    _write_csv(OUT / "derivation_check.csv", deriv)
    fig_path = figure(sweep_rows, conv_rows)

    max_force_err = max(abs(r["contact_time_force_err_pct"]) for r in sweep_rows)
    max_geom_err = max(abs(r["contact_time_geometric_err_pct"]) for r in sweep_rows)
    worst_clamped = max(range(len(sweep_rows)),
                        key=lambda i: abs(sweep_rows[i]["err_vs_clamped_pct"]))
    worst_unclamped = max(range(len(sweep_rows)),
                          key=lambda i: abs(sweep_rows[i]["err_vs_unclamped_pct"]))
    summary = {
        "benchmark": "DEM3D linear spring-dashpot contact-impact analytic validation",
        "case": "equal-mass head-on pair collision, free flight, drag=0, bonds broken",
        "analytic_references": {
            "unclamped": "e = exp(-pi*zeta/sqrt(1-zeta^2)); the identity used by "
                         "contact_tangential.restitution_to_damping",
            "clamped": "F = max(0, k*delta + c*delta'); contact ends at "
                       "theta* = atan2(2*zeta*sqrt(1-zeta^2), 2*zeta^2-1)",
            "cross_check": "closed form vs RK4 integration of the clamped ODE",
            "cross_check_max_abs_diff": max_deriv,
        },
        "verdicts": {
            "solver_vs_clamped_analytic_max_abs_pct": abs(sweep_rows[worst_clamped]["err_vs_clamped_pct"]),
            "solver_vs_clamped_within_3pct": abs(sweep_rows[worst_clamped]["err_vs_clamped_pct"]) < 3.0,
            "solver_vs_unclamped_analytic_max_abs_pct": abs(sweep_rows[worst_unclamped]["err_vs_unclamped_pct"]),
            "current_mapping_bias_exceeds_3pct": abs(sweep_rows[worst_unclamped]["err_vs_unclamped_pct"]) > 3.0,
            "force_contact_time_max_abs_pct": max_force_err,
            "geometric_contact_time_max_abs_pct": max_geom_err,
            "corrected_mapping_max_abs_pct": max(abs(r["rel_err_corrected_pct"]) for r in repair_rows),
            "old_mapping_max_abs_pct": max(abs(r["rel_err_old_mapping_pct"]) for r in repair_rows),
            "momentum_error_max": max(abs(r["momentum_error_kg_m_s"]) for r in sweep_rows),
            "force_law_residual_max": max(r["force_law_residual"] for r in sweep_rows),
        },
        "restitution_sweep": sweep_rows,
        "timestep_convergence": conv_rows,
        "mapping_repair": repair_rows,
        "rate_independence": rate_rows,
        "derivation_check": deriv,
        "figure": str(fig_path),
        "accuracy_band_pct": 100.0 * ACCURACY_BAND,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    render_doc(summary, pathlib.Path("docs/dem3d_contact_impact_analytic.md"))

    print("\n=== VERDICT ===")
    for key, value in summary["verdicts"].items():
        print(f"  {key}: {value}")
    print(f"\nartifacts -> {OUT}")
    print(f"figure    -> {fig_path}")
    ok = (summary["verdicts"]["solver_vs_clamped_within_3pct"]
          and summary["verdicts"]["corrected_mapping_max_abs_pct"] < 3.0
          and max_deriv < 1e-8)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())