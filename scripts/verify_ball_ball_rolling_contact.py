#!/usr/bin/env python3
"""Verification driver: ball-ball rolling-friction contact mechanics.

Runs the seven analytic-vs-numerical faces of
:mod:`tensordem.ball_ball_rolling_verification`, writes a machine-readable
results JSON (with a sha256 audit chain) and the report figures.

Acceptance criteria (campaign P40-N04):

(a) tangential stiffness recovered to <= 1 %
(b) Coulomb cone hard assertion, 10 independent cases
(c) rolling-resistance moment vs ``mu_r r_eff F_n``, 9 cases, <= 2 %
(d) rotational spin-down vs ``exp(-t/tau)``, R^2 >= 0.999
(e) rolling-sliding transition, <= 3 %
(f) inclined two-ball stack stability, <= 5 % (plane limit)
(g) collision energy ledger residual <= 1e-4

Usage
-----
    PYTHONPATH=src python scripts/verify_ball_ball_rolling_contact.py \
        --out artifacts/ball_ball_rolling_contact \
        --figdir docs/figures/ball_ball_rolling_contact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import torch

from tensordem.ball_ball_rolling import (
    TangentialState,
    advance_tangential,
    analytic_transition_spin_ratio,
    effective_radius,
    simulate_tangential_spring,
    solid_sphere_inertia,
    solid_sphere_mass,
)
from tensordem.ball_ball_rolling_verification import (
    simulate_roll_slip_transition,
    verify_driven_rolling,
    verify_energy_ledger,
    verify_incline_stability,
    verify_rolling_moment,
    verify_rotational_damping,
    verify_rolling_sliding_transition,
    verify_tangential_stiffness,
    verify_coulomb_limit,
)

CRITERIA = {
    "a_tangential_stiffness_pct": 1.0,
    "c_rolling_moment_pct": 2.0,
    "d_rotational_r2_min": 0.999,
    "e_transition_pct": 3.0,
    "f_stack_plane_limit_pct": 5.0,
    "g_energy_residual": 1.0e-4,
    "h_pytest_min": 18,
}


def _json_default(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def build_results() -> dict:
    torch.set_num_threads(1)
    tangential = verify_tangential_stiffness()
    coulomb = verify_coulomb_limit()
    moment = verify_rolling_moment()
    rotation = verify_rotational_damping()
    transition = verify_rolling_sliding_transition()
    driven = verify_driven_rolling()
    incline = verify_incline_stability()
    energy = verify_energy_ledger()

    checks = {
        "a_tangential_stiffness": {
            "quantity": "k_t recovered / k_t input",
            "worst_rel_err": tangential["worst_k_t_rel_err"],
            "threshold": CRITERIA["a_tangential_stiffness_pct"] / 100.0,
            "passed": tangential["passed"],
        },
        "b_coulomb_limit": {
            "quantity": "|F_t| / (mu_s F_n), 10 cases",
            "worst_rel_err": coulomb["worst_cap_rel_err"],
            "worst_direction_err": coulomb["worst_direction_err"],
            "n_cases": coulomb["n_cases"],
            "threshold": 1.0e-12,
            "passed": coulomb["passed"],
        },
        "c_rolling_moment": {
            "quantity": "M_r / (mu_r r_eff F_n), 9 cases",
            "worst_rel_err": moment["worst_moment_rel_err"],
            "threshold": CRITERIA["c_rolling_moment_pct"] / 100.0,
            "passed": moment["passed"],
        },
        "d_rotational_damping": {
            "quantity": "omega(t) vs exp(-c_r t / I)",
            "r2": rotation["r2"],
            "tau_rel_err": rotation["tau_rel_err"],
            "threshold_r2": CRITERIA["d_rotational_r2_min"],
            "passed": rotation["passed"],
        },
        "e_rolling_sliding_transition": {
            "quantity": "v_t vs omega r (exact) / omega r_eff (plane limit)",
            "worst_exact_rel_err": transition["worst_exact_rolling_err"],
            "plane_limit_r_eff_rel_err": transition["plane_limit_r_eff_err"],
            "spin_ratio_rel_err_plane": transition["spin_ratio_rel_err_plane"],
            "time_rel_err_plane": transition["time_rel_err_plane"],
            "threshold": CRITERIA["e_transition_pct"] / 100.0,
            "passed": transition["passed"],
        },
        "e2_driven_rolling_balance": {
            "quantity": "|F_t| r_j vs M_r (steady rolling)",
            "moment_rel_err": driven["moment_rel_err"],
            "force_balance_rel_err": driven["force_balance_rel_err"],
            "rolling_condition_err": driven["rolling_condition_err"],
            "passed": driven["passed"],
        },
        "f_incline_stack_stability": {
            "quantity": "theta_cr vs atan(mu_r) (plane limit, r_b/r_a = 0.02)",
            "plane_limit_rel_err": incline["plane_limit_rel_err"],
            "worst_rel_err_vs_analytic": incline["worst_rel_err_vs_analytic"],
            "dynamic_below_velocity": incline["dynamic_below"]["v_tail_max"],
            "dynamic_above_velocity": incline["dynamic_above"]["v_tail_max"],
            "threshold": CRITERIA["f_stack_plane_limit_pct"] / 100.0,
            "passed": incline["passed"],
        },
        "g_energy_ledger": {
            "quantity": "collision ledger closure / dissipation",
            "elastic_energy_err": energy["elastic_energy_err"],
            "dissipative_residual": energy["dissipative_residual"],
            "worst_closure_rel": energy["worst_closure_rel"],
            "oblique_restitution_err": energy["oblique_restitution_err"],
            "threshold": CRITERIA["g_energy_residual"],
            "passed": energy["passed"],
        },
    }
    all_passed = all(c["passed"] for c in checks.values())
    return {
        "campaign_node": "P40-N04",
        "repository": "TensorDEM",
        "module": "tensordem.ball_ball_rolling",
        "criteria": CRITERIA,
        "checks": checks,
        "all_passed": all_passed,
        "detail": {
            "tangential_stiffness": tangential,
            "coulomb_limit": coulomb,
            "rolling_moment": moment,
            "rotational_damping": {
                k: v for k, v in rotation.items() if k not in ("times", "measured", "analytic")
            },
            "rotation_series": {
                "times": rotation["times"],
                "measured": rotation["measured"],
                "analytic": rotation["analytic"],
            },
            "rolling_sliding_transition": {
                k: v for k, v in transition.items() if k != "cases"
            },
            "transition_cases": {
                name: {
                    k: v
                    for k, v in case.items()
                    if k not in ("slip_history", "v_history", "w_history")
                }
                for name, case in transition["cases"].items()
            },
            "driven_rolling": driven,
            "incline": incline,
            "energy": energy,
        },
    }


def make_figures(results: dict, figdir: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figdir.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []
    detail = results["detail"]

    # ---- figure 1: Cundall-Strack tangential law (elastic ramp + cone) -----
    f_n, k_t, mu_s = 5.0, 2.0e4, 0.4
    n_steps, slip = 4000, 2.0e-3
    dt = 5.0e-5
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    tangent = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    d_hist, f_hist = [], []
    for _ in range(n_steps):
        f_t = advance_tangential(state, slip * tangent, normal, f_n, k_t, mu_s, dt)
        d_hist.append(float(state.delta_t.norm()))
        f_hist.append(float(f_t.norm()))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(np.array(d_hist) * 1e3, f_hist, lw=1.6, label="Cundall-Strack $|F_t|$")
    ax.plot(
        np.array(d_hist) * 1e3,
        np.minimum(k_t * np.array(d_hist), mu_s * f_n),
        "--", lw=1.2, label=r"analytic $\min(k_t\delta_t,\ \mu_s F_n)$",
    )
    ax.axhline(mu_s * f_n, color="crimson", ls=":", lw=1.1,
               label=r"$\mu_s F_n$ = %.1f N" % (mu_s * f_n))
    ax.set_xlabel(r"tangential displacement $\delta_t$ [mm]")
    ax.set_ylabel(r"$|F_t|$ [N]")
    ax.set_title("(a) Tangential spring + Coulomb cone (ball-ball)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = figdir / "fig1_tangential_law.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": p1.name,
            "caption": "Cundall-Strack tangential force on a ball-ball contact: "
            "elastic ramp, analytic overlay and the Coulomb cap.",
            "path": str(p1),
        }
    )

    # ---- figure 2: rotational spin-down ------------------------------------
    series = detail["rotation_series"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(series["times"], series["measured"], lw=1.6, label="DEM integration")
    ax.semilogy(series["times"], series["analytic"], "--", lw=1.2,
                label=r"$\omega_0 e^{-c_r t/I}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$|\omega|$ [rad/s]")
    rd = detail["rotational_damping"]
    ax.set_title(
        "(b) Rotational damping: $R^2$ = %.7f, $\\tau$ error = %.2e"
        % (rd["r2"], rd["tau_rel_err"])
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p2 = figdir / "fig2_rotational_decay.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": p2.name,
            "caption": "Rotational spin-down of a ball-ball contact against the "
            "closed-form exponential.",
            "path": str(p2),
        }
    )

    # ---- figure 3: rolling-sliding transition ------------------------------
    case = simulate_roll_slip_transition(mu_r=0.0, r_i=1.0e9, r_j=0.05)
    slip = np.array(case["slip_history"])
    w = np.array(case["w_history"])
    t = np.arange(len(slip)) * case["dt"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t * 1e3, slip, lw=1.5, label="contact slip $s$ [m/s]")
    ax.plot(t * 1e3, w / case["omega_0"], lw=1.5, label=r"$\omega/\omega_0$")
    ax.axhline(2.0 / 7.0, color="crimson", ls=":", lw=1.1,
               label=r"analytic $\omega^*/\omega_0 = 2/7$")
    ax.axvline(case["transition_time"] * 1e3, color="k", ls="--", lw=1.0,
               label="transition $s=0$")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("normalised value")
    ax.set_title("(c) Rolling-sliding transition (plane-limit case)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = figdir / "fig3_roll_slip_transition.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": p3.name,
            "caption": "Slip decay and spin decay through the rolling-sliding "
            "transition; the analytic spin ratio 2/7 is recovered.",
            "path": str(p3),
        }
    )

    # ---- figure 4: stack critical angle vs radius ratio --------------------
    rows = detail["incline"]["stack_cases"]
    ratio = [r["radius_ratio"] for r in rows]
    meas = [r["theta_cr_measured_deg"] for r in rows]
    ana = [r["theta_cr_analytic_deg"] for r in rows]
    mu_r = detail["incline"]["mu_r"]
    plane = detail["incline"]["dynamic_below"]["theta_cr_analytic_deg"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ratio, meas, "o", ms=7, label="static equilibrium bisection")
    ax.plot(ratio, ana, "-", lw=1.2, label="closed form (exact geometry factor)")
    ax.axhline(plane, color="crimson", ls=":", lw=1.2,
               label=r"plane limit $\arctan\mu_r$ = %.2f$^\circ$" % plane)
    ax.set_xscale("log")
    ax.set_xlabel(r"radius ratio $r_b/r_a$")
    ax.set_ylabel(r"$\theta_{cr}$ [deg]")
    ax.set_title(r"(d) Two-ball stack stability, $\mu_r$ = %.2f" % mu_r)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p4 = figdir / "fig4_stack_stability.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": p4.name,
            "caption": "Critical incline angle of a two-ball stack vs radius "
            "ratio, compared with the exact geometric factor and the "
            "plane-limit criterion.",
            "path": str(p4),
        }
    )

    # ---- figure 5: energy ledger ------------------------------------------
    energy = detail["energy"]
    labels, shares_dash, shares_roll, shares_tang, shares_kin = [], [], [], [], []
    for key, label in (
        ("elastic_limit", "elastic e=1"),
        ("oblique_frictional", "oblique e=0.6"),
        ("fully_dissipative", "overdamped e~0"),
    ):
        case = energy[key]
        k0 = case["k_initial"]
        labels.append(label)
        shares_dash.append(case["d_dashpot"] / k0)
        shares_roll.append(case["d_rolling"] / k0)
        shares_tang.append(case["d_tangential"] / k0)
        shares_kin.append(case["k_final_rel"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bottom = np.zeros(len(labels))
    for share, name, colour in (
        (shares_dash, "normal dashpot", "tab:blue"),
        (shares_tang, "tangential/friction", "tab:orange"),
        (shares_roll, "rolling resistance", "tab:green"),
        (shares_kin, "kinetic energy left", "tab:grey"),
    ):
        ax.bar(x, share, bottom=bottom, label=name, color=colour)
        bottom += np.array(share)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"fraction of $K_0$")
    ax.set_title("(e) Collision energy ledger (closure residual <= 1e-4)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p5 = figdir / "fig5_energy_ledger.png"
    fig.savefig(p5, dpi=160)
    plt.close(fig)
    figures.append(
        {
            "name": p5.name,
            "caption": "Energy ledger of the ball-ball collision: fractional "
            "share of the initial kinetic energy taken by each dissipative "
            "channel.",
            "path": str(p5),
        }
    )
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/ball_ball_rolling_contact")
    parser.add_argument("--figdir", default="docs/figures/ball_ball_rolling_contact")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    results = build_results()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, default=_json_default)
    (outdir / "results.json").write_text(payload)
    digest = hashlib.sha256((outdir / "results.json").read_bytes()).hexdigest()
    (outdir / "results.sha256").write_text(digest + "  results.json\n")
    results["audit"] = {"results_sha256": digest}

    if not args.skip_figures:
        figures = make_figures(results, pathlib.Path(args.figdir))
        (outdir / "figures.json").write_text(json.dumps(figures, indent=2))
        print(f"figures written: {len(figures)} -> {args.figdir}")

    print(f"results: {outdir / 'results.json'}  sha256={digest}")
    for name, check in results["checks"].items():
        flag = "PASS" if check["passed"] else "FAIL"
        keys = [k for k in check if k.endswith(("_err", "_rel", "r2"))]
        detail_str = ", ".join(f"{k}={check[k]:.3e}" for k in keys)
        print(f"  [{flag}] {name}: {detail_str}")
    print(f"ALL PASSED = {results['all_passed']}")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())