"""m14951: produce the evidence table + figure for the dem_vectorized verification.

Writes
------
artifacts/m14951/verification_metrics.json   numbers quoted in the markdown report
docs/figures/dem_vectorized_verification.png  four-panel evidence figure

Every number in ``docs/DEM_VECTORIZED_VERIFICATION.md`` comes from this JSON -- no
hand-typed values.  Run: ``python scripts/m14951_verification_figures.py``.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.dem_vectorized import (  # noqa: E402
    ReferenceDEM,
    VecDEM,
    VecDEMConfig,
    compute_contact_forces,
    find_neighbor_pairs_vectorized,
    realised_restitution_floor,
    realized_restitution_input,
    restitution_response_curve,
)

RADIUS = 0.01
DENSITY = 1000.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "m14951")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


# ---------------------------------------------------------------------------
# measurements (each one mirrors an assertion in the test suite)
# ---------------------------------------------------------------------------
def brute_force_pairs(positions, radius):
    cutoff = 2.0 * radius
    out = set()
    n = len(positions)
    for i in range(n):
        d = torch.linalg.vector_norm(positions[i + 1:] - positions[i], dim=1)
        for j in torch.nonzero(d < cutoff).flatten().tolist():
            out.add((i, j + i + 1))
    return out


def neighbour_audit():
    torch.manual_seed(0)
    side = 4
    g = torch.arange(side, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    dense = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) * (2.0 * RADIUS * 0.99)
    cases = {
        "dense_lattice_N64": dense,
        "random_dilute_N40": torch.rand(40, 3, dtype=torch.float64) * 0.4,
        "clustered_gaussian_N60": torch.randn(60, 3, dtype=torch.float64) * 0.012,
        "collinear_chain_N12": torch.stack(
            [torch.arange(12, dtype=torch.float64) * 1.9 * RADIUS,
             torch.zeros(12, dtype=torch.float64),
             torch.zeros(12, dtype=torch.float64)], dim=1),
    }
    rows = []
    for name, pos in cases.items():
        got = {(int(i), int(j)) for i, j in
               find_neighbor_pairs_vectorized(pos, RADIUS).tolist()}
        ref = brute_force_pairs(pos, RADIUS)
        rows.append({"case": name, "n_particles": int(len(pos)),
                     "pairs_cell_list": len(got), "pairs_bruteforce": len(ref),
                     "missing": len(ref - got), "extra": len(got - ref),
                     "match": got == ref})
    return rows


def contact_law_audit():
    rows = []
    for frac in (1e-3, 1e-2, 1e-1):
        delta = frac * RADIUS
        pos = torch.tensor([[-RADIUS + delta / 2, 0.0, 0.0],
                            [RADIUS - delta / 2, 0.0, 0.0]], dtype=torch.float64)
        zero = torch.zeros_like(pos)
        cfg_lin = VecDEMConfig(radius=RADIUS, density=DENSITY, contact_stiffness=1e5,
                               restitution_coefficient=1.0, friction_coefficient=0.0)
        f_lin = compute_contact_forces(pos, zero, torch.tensor([[0, 1]]), RADIUS, cfg_lin)
        cfg_h = VecDEMConfig(radius=RADIUS, density=DENSITY, contact_model="hertz",
                             young_modulus=1e9, poisson_ratio=0.3,
                             restitution_coefficient=1.0, friction_coefficient=0.0)
        f_h = compute_contact_forces(pos, zero, torch.tensor([[0, 1]]), RADIUS, cfg_h)
        E_star = 1e9 / (2.0 * (1.0 - 0.3 ** 2))
        expected_h = (4.0 / 3.0) * E_star * math.sqrt(RADIUS / 2.0) * delta ** 1.5
        rows.append({
            "delta_over_R": frac,
            "linear_F_sim": abs(float(f_lin[0, 0])),
            "linear_F_analytic": 1e5 * delta,
            "linear_rel_err": abs(abs(float(f_lin[0, 0])) - 1e5 * delta) / (1e5 * delta),
            "hertz_F_sim": abs(float(f_h[0, 0])),
            "hertz_F_analytic": expected_h,
            "hertz_rel_err": abs(abs(float(f_h[0, 0])) - expected_h) / expected_h,
        })
    return rows


def measure_restitution(e_cfg, contact_model="linear", v_impact=1.0, dt_refine=10.0,
                        calibration=True):
    cfg = VecDEMConfig(n_particles=2, radius=RADIUS, density=DENSITY,
                       contact_model=contact_model, restitution_coefficient=e_cfg,
                       friction_coefficient=0.0, gravity=0.0,
                       restitution_calibration=calibration)
    mass = cfg.mass
    dt = cfg.effective_dt() / dt_refine
    pos = torch.tensor([[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]], dtype=torch.float64)
    vel = torch.tensor([[v_impact, 0.0, 0.0], [-v_impact, 0.0, 0.0]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    for _ in range(2_000_000):
        f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
        vel = vel + 0.5 * f / mass * dt
        pos = pos + vel * dt
        f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
        vel = vel + 0.5 * f / mass * dt
        if (float(pos[1, 0] - pos[0, 0]) > 2.0 * RADIUS
                and float(vel[1, 0]) > 0.0 and float(vel[0, 0]) < 0.0):
            return abs(float(vel[1, 0] - vel[0, 0]) / (2.0 * v_impact))
    raise RuntimeError("collision did not separate")


def restitution_audit():
    targets = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    rows = []
    for e in targets:
        legacy = measure_restitution(e, calibration=False)
        calibrated = measure_restitution(e, calibration=True)
        rows.append({"e_configured": e, "e_realised_legacy": legacy,
                     "legacy_rel_err": abs(legacy - e) / e,
                     "e_realised_calibrated": calibrated,
                     "calibrated_rel_err": abs(calibrated - e) / e})
    return rows


def velocity_dependence():
    rows = []
    for model in ("linear", "hertz"):
        for v in (0.05, 0.2, 0.5, 1.0, 2.0):
            rows.append({"contact_model": model, "v_impact": v,
                         "e_realised": measure_restitution(0.5, model, v_impact=v)})
    return rows


def conservation_audit():
    # 1) momentum: 40-particle cluster, contacts only
    cfg = VecDEMConfig(n_particles=40, radius=RADIUS, density=DENSITY, seed=5,
                       friction_coefficient=0.3, restitution_coefficient=0.5,
                       contact_stiffness=1e5, gravity=0.0,
                       domain_size=(0.06, 0.06, 0.06))
    sim = VecDEM(cfg)
    sim.velocities += 0.2 * torch.randn_like(sim.velocities)
    p0 = sim.total_momentum().clone()
    scale = float(cfg.mass * torch.abs(sim.velocities).sum())
    history = []
    for step in range(400):
        sim.step()
        if (step + 1) % 20 == 0:
            history.append(float(torch.abs(sim.total_momentum() - p0).max()) / scale)

    # 2) free fall energy
    n = 16
    cfg2 = VecDEMConfig(n_particles=n, radius=RADIUS, density=DENSITY, gravity=9.81,
                        contact_stiffness=1e5, restitution_coefficient=1.0,
                        friction_coefficient=0.0, seed=2, domain_size=(0.5, 0.5, 0.5))
    ff = VecDEM(cfg2)
    lin = torch.arange(n, dtype=torch.float64)
    ff.positions = torch.stack([lin * 0.02 + 0.05, torch.zeros(n, dtype=torch.float64),
                                torch.full((n,), 0.5, dtype=torch.float64)], dim=1)
    ff.velocities[:] = 0.0
    e0 = ff.total_energy()
    ff_hist = []
    for step in range(1000):
        ff.step()
        if (step + 1) % 100 == 0:
            ff_hist.append(abs(ff.total_energy() - e0) / abs(e0))

    # 3) cell list vs O(N^2) engine, bitwise-equivalent trajectory
    base = dict(n_particles=64, radius=RADIUS, density=DENSITY, gravity=9.81,
                contact_stiffness=1e5, restitution_coefficient=0.5,
                friction_coefficient=0.3, domain_size=(0.06, 0.06, 0.06))
    a = VecDEM(VecDEMConfig(seed=4, **base))
    b = ReferenceDEM(VecDEMConfig(seed=4, **base))
    for _ in range(120):
        a.step()
        b.step()
    return {
        "momentum_drift_history": history,
        "momentum_drift_final": history[-1],
        "free_fall_relative_energy_error": ff_hist,
        "free_fall_final_rel_err": ff_hist[-1],
        "celllist_vs_reference_max_pos_diff": float(
            torch.abs(a.positions - b.positions).max()),
        "celllist_vs_reference_max_vel_diff": float(
            torch.abs(a.velocities - b.velocities).max()),
    }


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    metrics = {
        "neighbour_audit": neighbour_audit(),
        "contact_law_audit": contact_law_audit(),
        "restitution_audit": restitution_audit(),
        "velocity_dependence": velocity_dependence(),
        "conservation_audit": conservation_audit(),
        "restitution_floor": {"linear": realised_restitution_floor("linear"),
                              "hertz": realised_restitution_floor("hertz")},
        "restitution_response_curve": {
            model: restitution_response_curve(model) for model in ("linear", "hertz")},
        "calibration_examples": {
            str(e): realized_restitution_input(e) for e in (0.2, 0.3, 0.5, 0.7, 0.9)},
    }
    with open(os.path.join(OUT_DIR, "verification_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    ax = axes[0, 0]
    curve = metrics["restitution_response_curve"]["linear"]
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="ideal ($e_{realised}=e_{configured}$)")
    ax.plot([c[0] for c in curve], [c[1] for c in curve], "o-", ms=4,
            label="legacy dashpot (no calibration)")
    reb = metrics["restitution_audit"]
    ax.plot([r["e_configured"] for r in reb], [r["e_realised_calibrated"] for r in reb],
            "s-", ms=5, color="tab:green", label="calibrated (this work)")
    ax.axhline(metrics["restitution_floor"]["linear"], color="tab:red", lw=0.8, ls=":")
    ax.text(0.03, metrics["restitution_floor"]["linear"] + 0.02,
            f"clamp floor {metrics['restitution_floor']['linear']:.3f}", color="tab:red",
            fontsize=8)
    ax.set_xlabel("configured restitution coefficient")
    ax.set_ylabel("realised restitution measured from the collision")
    ax.set_title("(a) realised vs configured restitution, linear contact")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for model, marker in (("linear", "o"), ("hertz", "s")):
        rows = [r for r in metrics["velocity_dependence"] if r["contact_model"] == model]
        ax.plot([r["v_impact"] for r in rows], [r["e_realised"] for r in rows],
                marker + "-", ms=5, label=f"{model} (target 0.50)")
    ax.set_xscale("log")
    ax.axhline(0.5, color="k", ls="--", lw=1.0)
    ax.set_ylim(0.4, 0.6)
    ax.set_xlabel("impact velocity [m/s]")
    ax.set_ylabel("realised restitution")
    ax.set_title("(b) velocity dependence of the realised restitution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    law = metrics["contact_law_audit"]
    d = [r["delta_over_R"] for r in law]
    ax.loglog(d, [r["linear_F_sim"] for r in law], "o", ms=6, label="linear sim")
    ax.loglog(d, [r["linear_F_analytic"] for r in law], "k-", lw=1.0,
              label=r"$k\delta$ analytic")
    ax.loglog(d, [r["hertz_F_sim"] for r in law], "s", ms=6, label="Hertz sim")
    ax.loglog(d, [r["hertz_F_analytic"] for r in law], "k--", lw=1.0,
              label=r"$\frac{4}{3}E^*\sqrt{R^*}\delta^{3/2}$ analytic")
    ax.set_xlabel(r"overlap $\delta/R$")
    ax.set_ylabel("normal force [N]")
    ax.set_title("(c) contact laws vs analytic expressions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    cons = metrics["conservation_audit"]
    steps = [20 * (i + 1) for i in range(len(cons["momentum_drift_history"]))]
    ax.semilogy(steps, cons["momentum_drift_history"], "o-", ms=3,
                label="|dP|/|P| (40 contacts)")
    ff_steps = [100 * (i + 1) for i in range(len(cons["free_fall_relative_energy_error"]))]
    ax.semilogy(ff_steps, cons["free_fall_relative_energy_error"], "s-", ms=3,
                label="|dE|/E free fall")
    ax.set_xlabel("integration step")
    ax.set_ylabel("relative drift")
    ax.set_title("(d) integrator invariants")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("TensorDEM dem_vectorized verification (mission m14951)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = os.path.join(FIG_DIR, "dem_vectorized_verification.png")
    fig.savefig(fig_path, dpi=150)
    print("wrote", fig_path)
    print("wrote", os.path.join(OUT_DIR, "verification_metrics.json"))


if __name__ == "__main__":
    main()