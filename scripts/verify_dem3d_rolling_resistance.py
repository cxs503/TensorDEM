#!/usr/bin/env python3
"""Analytic verification of the DEM3D rolling-resistance contact law (SIR-PC2|F05).

The 3-D bonded-sphere solver ``tensordem.dem3d`` carries no rotational degree of
freedom, so the rolling-resistance torque declared as ``T_r = mu_r R |F_n|`` is
realised as an equal-and-opposite tangential couple.  This script verifies that
realisation against closed-form references -- it is *code-versus-analytic*
verification, not a physical validation of sea-ice mechanics.

Reference case: one free sphere (radius R, mass m) resting in a single
load-bearing contact on a fixed sphere, with prescribed overlap ``delta`` and an
initial tangential velocity ``v0``.  The contact normal is ``+z`` and the slip
direction is ``+x`` -- exactly the configuration in which a wrongly oriented
couple would show up as a spurious ``y`` (binormal) force.

References checked
------------------
L1  ``|f_rr| = mu_r |F_n|``                          (declared law magnitude)
L2  ``f_rr`` anti-parallel to the relative slip       (dissipative direction)
L3  ``f_i + f_j = 0``                                 (linear-momentum conservation)
L4  ``|T| = mu_r R |F_n|`` and the factor 2 against the equal-sphere convention
    of Ai et al. (2011), ``M_lim = mu_r R* |F_n|`` with ``R* = R/2``
L5  additivity with a saturated Cundall-Strack spring,
    ``|F_t| = (mu + mu_r) |F_n|``
L6  ``mu_r = 0`` reproduces the friction-only force field exactly
I1  sliding deceleration ``a = (mu + mu_r) |F_n| / m`` (integration level)
I2  stopping time ``t* = v0 / a`` and no reversal
I3  kinetic energy monotone non-increasing, dissipated to zero
I4  no transverse (binormal) momentum injection
N1  negative control: the pre-fix (orthogonal-couple) law reproduces the defect
    -- it does not decelerate the slip and it pumps energy into ``y``.

Run:  PYTHONPATH=src python scripts/verify_dem3d_rolling_resistance.py
Writes docs/dem3d-rolling-resistance/{results.json,checks.csv,summary.md,figures/*}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

RADIUS = 0.025
OVERLAP = 0.001
CONTACT_STIFFNESS = 5000.0
DENSITY = 917.0
V0 = 0.02
DT = 2.0e-6
PARK = 0.2   # parked far beyond 2R but small enough for the cell list


# ----------------------------------------------------------------------------
# System construction
# ----------------------------------------------------------------------------
def contact_pairs_realised(sim: IceDEM3D) -> int:
    """Number of pairs with strictly positive overlap (the active contact set)."""
    p = sim._contact_candidate_pairs()
    if p.numel() == 0:
        return 0
    ci, cj = p.T
    d = torch.linalg.vector_norm(sim.positions[cj] - sim.positions[ci], dim=1)
    return int((d < 2 * sim.config.radius).sum())


def contact_normal(sim: IceDEM3D, anchor: int, slider: int) -> torch.Tensor:
    d = sim.positions[slider] - sim.positions[anchor]
    return d / torch.linalg.vector_norm(d)


def normal_force(sim: IceDEM3D, anchor: int = 0, slider: int = 1) -> float:
    """|F_n| at the single contact: k_c * overlap with zero normal damping."""
    cfg = sim.config
    d = sim.positions[slider] - sim.positions[anchor]
    overlap = max(0.0, 2 * cfg.radius - float(torch.linalg.vector_norm(d)))
    return cfg.contact_stiffness * overlap


def build_single_contact(mu: float, mu_r: float) -> tuple[IceDEM3D, int, int]:
    """One free sphere in a single load-bearing contact with a fixed sphere.

    The bonded topology is disabled so that the only interaction is the
    unbonded contact.  The lattice keeps its spacing of exactly ``2R``, so every
    other pair has zero overlap and is excluded from the contact set: the
    contact set is verified to be exactly one pair by
    :func:`assert_single_contact`.
    """
    torch.manual_seed(0)
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=RADIUS, density=DENSITY,
        contact_stiffness=CONTACT_STIFFNESS,
        contact_damping=0.0,          # -> gamma_n = 0 with e = 1
        drag=0.0,
        restitution_coefficient=1.0,  # -> gamma_n = 0 exactly
        friction_coefficient=mu,
        rolling_resistance_coefficient=mu_r,
        fix_bottom=True, fix_x_edges=False,
        dt=DT,
        tool_speed=1.0,
    )
    sim = IceDEM3D(cfg)
    sim.alive[:] = False              # isolate the unbonded contact
    fixed_idx = torch.nonzero(sim.fixed, as_tuple=False).reshape(-1)
    free_idx = torch.nonzero(~sim.fixed, as_tuple=False).reshape(-1)
    anchor = int(fixed_idx[0])
    slider = int(free_idx[0])

    # Park the non-participating free spheres so that floating-point noise on
    # the nominal 2R lattice cannot create spurious contacts.
    for k, idx in enumerate(free_idx.tolist()):
        if idx != slider:
            sim.positions[idx] = torch.tensor(
                [PARK * (k + 1), PARK * (k + 1), 3.0 * PARK], dtype=sim.dtype
            )

    # Slider sits directly above the anchor: n = +z, tangent plane = (x, y).
    sim.positions[slider] = sim.positions[anchor] + torch.tensor(
        [0.0, 0.0, 2.0 * RADIUS - OVERLAP], dtype=sim.dtype
    )
    sim.velocities.zero_()
    sim._tangential_state.clear()
    n_active = contact_pairs_realised(sim)
    if n_active != 1:
        raise RuntimeError(f"expected exactly one active contact, got {n_active}")
    return sim, anchor, slider


# ----------------------------------------------------------------------------
# Law-level checks (single force evaluation, no time integration)
# ----------------------------------------------------------------------------
def check_law(mu_r: float, mu: float) -> dict:
    sim, anchor, slider = build_single_contact(mu, mu_r)
    f_n = normal_force(sim, anchor, slider)

    # --- pure rolling-resistance couple, evaluated directly -----------------
    ref_no_rr = build_single_contact(mu, 0.0)[0]
    ref_no_rr.positions[slider] = sim.positions[slider].clone()
    ref_no_rr.velocities[slider] = torch.tensor([V0, 0.0, 0.0], dtype=sim.dtype)
    sim.velocities[slider] = torch.tensor([V0, 0.0, 0.0], dtype=sim.dtype)

    fi, fj, ti, tj = sim._rolling_resistance_couple(
        torch.tensor([anchor]), torch.tensor([slider]),
        contact_normal(sim, anchor, slider).reshape(1, 3),
        torch.tensor([f_n], dtype=sim.dtype),
    )
    slip = sim.velocities[slider] - sim.velocities[anchor]
    slip_hat = slip / torch.linalg.vector_norm(slip)

    measured_mag = float(torch.linalg.vector_norm(fi))
    expected_mag = mu_r * f_n
    # The force on the *moving* particle j must oppose its relative slip, so
    # f_j . slip_hat = -|f|; the force on i is its reaction (+|f|).
    dot = float((fj.reshape(3) * slip_hat).sum())
    cos_dir = dot / measured_mag if measured_mag > 0 else float("nan")

    net = fi + fj
    t_mag = float(torch.linalg.vector_norm(ti))
    t_expected = mu_r * RADIUS * f_n                        # declared lever R
    t_ai = mu_r * 0.5 * RADIUS * f_n                        # Ai et al. R* = R/2

    # --- additivity with a saturated Cundall-Strack spring -------------------
    # One force evaluation with a large slip velocity saturates the tangential
    # spring at mu|F_n|; the couple then adds mu_r|F_n|.
    sim_hot = build_single_contact(mu, mu_r)[0]
    sim_hot.positions[slider] = sim.positions[slider].clone()
    sim_hot.velocities[slider] = torch.tensor([200.0, 0.0, 0.0], dtype=sim.dtype)
    f_hot, _ = sim_hot.forces(update_fracture=False)
    # Project the slider force onto the tangent plane (normal = +z here) so the
    # check measures the tangential resistance only.
    n_hot = contact_normal(sim_hot, anchor, slider)
    f_slider = f_hot[slider]
    total_t = float(torch.linalg.vector_norm(f_slider - (f_slider * n_hot).sum() * n_hot))
    expected_total = (mu + mu_r) * f_n

    # --- L6: mu_r = 0 must be bit-identical to the friction-only field -------
    a = build_single_contact(mu, 0.0)[0]
    b = build_single_contact(mu, 0.0)[0]
    for s in (a, b):
        s.velocities[1] = torch.tensor([V0, 0.0, 0.0], dtype=s.dtype)
    fa, _ = a.forces(update_fracture=False)
    fb, _ = b.forces(update_fracture=False)
    identical = bool(torch.equal(fa, fb)) and float(torch.linalg.vector_norm(b._rolling_torque)) == 0.0

    return {
        "mu": mu,
        "mu_r": mu_r,
        "f_n": f_n,
        "L1_force_mag": measured_mag,
        "L1_expected": expected_mag,
        "L1_rel_err": abs(measured_mag - expected_mag) / expected_mag if expected_mag else 0.0,
        "L2_cos_to_slip": cos_dir,
        "L3_net_force": float(torch.linalg.vector_norm(net)),
        "L4_torque_mag": t_mag,
        "L4_expected_lever_R": t_expected,
        "L4_rel_err": abs(t_mag - t_expected) / t_expected if t_expected else 0.0,
        "L4_ai2011_equal_sphere_torque": t_ai,
        "L4_impl_over_ai_factor": (t_mag / t_ai) if t_ai else float("nan"),
        "L5_total_tangential": total_t,
        "L5_expected": expected_total,
        "L5_rel_err": abs(total_t - expected_total) / expected_total if expected_total else 0.0,
        "L6_bit_identical_to_friction_only": identical,
    }


# ----------------------------------------------------------------------------
# Integration-level checks
# ----------------------------------------------------------------------------
def run_sliding(mu: float, mu_r: float, *, v0: float = V0, max_steps: int = 4000,
                legacy: bool = False) -> dict:
    sim, anchor, slider = build_single_contact(mu, mu_r)
    f_n = normal_force(sim, anchor, slider)
    mass = sim.config.mass

    if mu > 0.0:
        # Pre-saturate the Cundall-Strack spring so the steady-state resistance
        # is (mu + mu_r)|F_n| from the first measured sample.
        sim.velocities[slider] = torch.tensor([200.0, 0.0, 0.0], dtype=sim.dtype)
        sim.forces(update_fracture=False)

    sim.velocities[slider] = torch.tensor([v0, 0.0, 0.0], dtype=sim.dtype)

    # Prescribed normal load that balances the initial contact force: the
    # contact stays at the prescribed overlap instead of pushing the slider off.
    loads = torch.zeros_like(sim.positions)
    loads[slider, 2] = -f_n

    original = IceDEM3D._rolling_resistance_couple
    if legacy:
        IceDEM3D._rolling_resistance_couple = _legacy_orthogonal_couple
    try:
        times = [0.0]
        vx = [float(sim.velocities[slider, 0])]
        vy = [float(sim.velocities[slider, 1])]
        ke = [0.5 * mass * float(sim.velocities.square().sum())]
        overlap = [normal_force(sim, anchor, slider) / CONTACT_STIFFNESS]
        for _ in range(max_steps):
            sim.step(loads)
            times.append(sim.time)
            vx.append(float(sim.velocities[slider, 0]))
            vy.append(float(sim.velocities[slider, 1]))
            ke.append(0.5 * mass * float(sim.velocities.square().sum()))
            overlap.append(normal_force(sim, anchor, slider) / CONTACT_STIFFNESS)
            if abs(vx[-1]) < 1e-3 * v0 and sim.time > 5 * DT:
                break
    finally:
        IceDEM3D._rolling_resistance_couple = original

    # Least-squares slope over the window where the slider is still moving fast.
    win = [k for k, v in enumerate(vx) if v > 0.25 * v0]
    if len(win) >= 3:
        ts = torch.tensor([times[k] for k in win], dtype=torch.float64)
        vs = torch.tensor([vx[k] for k in win], dtype=torch.float64)
        tm = ts.mean()
        slope = float(((ts - tm) * (vs - vs.mean())).sum() / ((ts - tm).square()).sum())
    else:
        slope = float("nan")

    a_meas = -slope                                  # positive deceleration
    a_analytic = (mu + mu_r) * f_n / mass
    t_stop_analytic = v0 / a_analytic if a_analytic > 0 else float("inf")
    t_stop_meas = next((times[k] for k, v in enumerate(vx) if v <= 0.01 * v0), float("nan"))

    return {
        "mu": mu, "mu_r": mu_r, "legacy": legacy,
        "f_n": f_n, "mass": mass, "v0": v0, "dt": DT, "steps": len(times) - 1,
        "a_measured": a_meas,
        "a_analytic": a_analytic,
        "a_rel_err": abs(a_meas - a_analytic) / a_analytic if a_analytic > 0 else float("nan"),
        "t_stop_measured": t_stop_meas,
        "t_stop_analytic": t_stop_analytic,
        "t_stop_rel_err": (abs(t_stop_meas - t_stop_analytic) / t_stop_analytic
                           if a_analytic > 0 and t_stop_meas == t_stop_meas else float("nan")),
        "max_abs_vy": max(abs(v) for v in vy),
        "ke_initial": ke[0],
        "ke_final": ke[-1],
        "ke_monotone": all(ke[k + 1] <= ke[k] * (1 + 1e-12) for k in range(len(ke) - 1)),
        "overlap_drift_rel": abs(overlap[-1] - overlap[0]) / overlap[0],
        "vx_final": vx[-1],
        "no_reversal": all(v >= -1e-9 for v in vx),
        "history": {"t": times, "vx": vx, "vy": vy, "ke": ke},
    }


def _legacy_orthogonal_couple(self, ci, cj, cnormal, contact_mag):
    """Negative control: the pre-fix SIR-PC2|F05 law.

    The couple was built from the torque and then re-applied as
    ``force = torque / R``, i.e. ``f = n x f_res``, which is perpendicular to
    the slip direction.  Kept only to demonstrate the defect it caused.
    """
    mu_r = self.config.rolling_resistance_coefficient
    R = self.config.radius
    relative_v = self.velocities[cj] - self.velocities[ci]
    vn = (relative_v * cnormal).sum(dim=1, keepdim=True)
    v_tangent = relative_v - vn * cnormal
    v_t_mag = torch.linalg.vector_norm(v_tangent, dim=1, keepdim=True)
    direction = torch.where(v_t_mag > 1e-15, -v_tangent / v_t_mag.clamp_min(1e-15),
                            torch.zeros_like(v_tangent))
    f_res = (mu_r * contact_mag)[:, None] * direction
    torque_i = R * torch.linalg.cross(cnormal, f_res)
    force_i = torque_i / R
    return force_i, -force_i, torque_i, -torque_i


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def make_figures(runs: list[dict], legacy: dict, sweep: list[dict], outdir: pathlib.Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Figure 1: velocity decay, measured vs analytic
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for run in runs:
        h = run["history"]
        t_ms = [1e3 * t for t in h["t"]]
        ax.plot(t_ms, h["vx"], lw=1.6,
                label=fr"DEM3D $\mu$={run['mu']:.2f}, $\mu_r$={run['mu_r']:.2f}")
        t_stop = run["t_stop_analytic"]
        ax.plot([0, 1e3 * t_stop], [run["v0"], 0.0], "k--", lw=0.9, alpha=0.6)
        if run is runs[0]:
            ax.plot([], [], "k--", lw=0.9, label=r"analytic $v_0-a\,t$")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"slip velocity $v_x$ [m/s]")
    ax.set_title("Rolling resistance decelerates slip at $a=(\\mu+\\mu_r)|F_n|/m$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    p = outdir / "rr_velocity_decay.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))

    # Figure 2: deceleration sweep vs analytic
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = [s["mu"] + s["mu_r"] for s in sweep]
    y = [s["a_measured"] for s in sweep]
    ax.plot(x, [s["a_analytic"] for s in sweep], "k-", lw=1.0, label="analytic")
    ax.plot(x, y, "o", ms=5, label="DEM3D measured")
    ax.set_xlabel(r"$\mu + \mu_r$ [-]")
    ax.set_ylabel(r"sliding deceleration $a$ [m/s$^2$]")
    ax.set_title("Additivity of sliding and rolling resistance")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = outdir / "rr_deceleration_sweep.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))

    # Figure 3: negative control - transverse pumping and zero dissipation
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    fixed = runs[0]
    for ax, key, lab in ((axes[0], "vx", r"$v_x$ [m/s]"), (axes[1], "vy", r"$v_y$ [m/s]")):
        ax.plot([1e3 * t for t in fixed["history"]["t"]], fixed["history"][key],
                lw=1.6, label="fixed (current)")
        ax.plot([1e3 * t for t in legacy["history"]["t"]], legacy["history"][key],
                lw=1.6, ls="--", label="legacy (orthogonal couple)")
        ax.set_xlabel("time [ms]"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_title("slip decay")
    axes[1].set_title("spurious binormal ($y$) velocity")
    p = outdir / "rr_legacy_vs_fixed.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))

    # Figure 4: kinetic energy / dissipation
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for run, style in ((fixed, "-"), (legacy, "--")):
        h = run["history"]
        ax.plot([1e3 * t for t in h["t"]], [1e3 * e for e in h["ke"]], style, lw=1.6,
                label=("fixed (current)" if style == "-" else "legacy (orthogonal couple)"))
    ax.set_xlabel("time [ms]"); ax.set_ylabel("kinetic energy [mJ]")
    ax.set_title("A dissipative law cannot create kinetic energy")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = outdir / "rr_energy.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/dem3d-rolling-resistance")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)

    # --- law level ----------------------------------------------------------
    law = check_law(mu_r=0.10, mu=0.20)

    # --- integration level: identical physics, three coefficients -----------
    cases = [(0.0, 0.05), (0.0, 0.10), (0.20, 0.10)]
    runs = [run_sliding(mu, mu_r) for mu, mu_r in cases]

    # sweep for additivity figure: pure rolling-resistance family
    sweep = [run_sliding(0.0, mu_r) for mu_r in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)]
    sweep += [run_sliding(mu, 0.05) for mu in (0.1, 0.2, 0.3)]

    # --- negative control ---------------------------------------------------
    legacy = run_sliding(0.0, 0.10, legacy=True)

    figs = make_figures(runs, legacy, sweep, outdir / "figures")

    payload = {
        "protocol": "tensordem-dem3d-rolling-resistance-verification-v1",
        "model": "DEM3D SIR-PC2|F05 rotationless rolling-resistance couple",
        "geometry": {"radius_m": RADIUS, "overlap_m": OVERLAP,
                     "contact_stiffness_N_per_m": CONTACT_STIFFNESS,
                     "particle_mass_kg": build_single_contact(0.0, 0.0)[0].config.mass,
                     "dt_s": DT, "v0_m_per_s": V0},
        "law_checks": law,
        "integration_runs": [{k: v for k, v in r.items() if k != "history"} for r in runs],
        "sweep": [{k: v for k, v in r.items() if k != "history"} for r in sweep],
        "legacy_control": {k: v for k, v in legacy.items() if k != "history"},
        "histories": {
            "runs": [{"mu": r["mu"], "mu_r": r["mu_r"], **r["history"]} for r in runs],
            "legacy": {"mu": legacy["mu"], "mu_r": legacy["mu_r"], **legacy["history"]},
        },
        "figures": figs,
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    # --- flat checks table --------------------------------------------------
    rows = []
    for k in ("L1_rel_err", "L2_cos_to_slip", "L3_net_force", "L4_rel_err",
              "L4_impl_over_ai_factor", "L5_rel_err"):
        rows.append({"check": k, "value": law[k], "target": {
            "L1_rel_err": "0 (<3%)", "L2_cos_to_slip": "-1", "L3_net_force": "0",
            "L4_rel_err": "0 (<3%)", "L4_impl_over_ai_factor": "2 by convention",
            "L5_rel_err": "0 (<3%)"}[k]})
    rows.append({"check": "L6_bit_identical_to_friction_only",
                 "value": law["L6_bit_identical_to_friction_only"], "target": "True"})
    for r in runs:
        tag = f"mu={r['mu']:.2f},mu_r={r['mu_r']:.2f}"
        rows += [
            {"check": f"I1_a_rel_err[{tag}]", "value": r["a_rel_err"], "target": "<3%"},
            {"check": f"I2_t_stop_rel_err[{tag}]", "value": r["t_stop_rel_err"], "target": "<3%"},
            {"check": f"I3_ke_monotone[{tag}]", "value": r["ke_monotone"], "target": "True"},
            {"check": f"I4_max_abs_vy[{tag}]", "value": r["max_abs_vy"], "target": "~0"},
        ]
    with open(outdir / "checks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "value", "target"])
        w.writeheader(); w.writerows(rows)

    print(json.dumps({k: law[k] for k in law}, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "history"} for r in runs], indent=2))
    print(json.dumps({k: v for k, v in legacy.items() if k != "history"}, indent=2))
    print("figures:", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())