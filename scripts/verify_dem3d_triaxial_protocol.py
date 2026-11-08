"""Reproduce the true-triaxial protocol audit (docs/dem3d_triaxial_lode.md).

Cases
-----
A  isotropic confinement, axial platen held: does the pressure servo reach the
   target, and does the energy ledger stay closed while the servo moves?
B  anisotropic confinement with the pressure servo kept running during the
   axial phase: are the *measured* lateral pressures the prescribed ones, so
   that the reported Lode parameter is a measurement and not an input?
C  affine elasticity anchor: the engine's bond+contact force law evaluated on
   a uniformly strained lattice must reproduce the closed-form central-force
   lattice moduli (independent of any dynamics).

Usage
-----
    PYTHONPATH=src python scripts/verify_dem3d_triaxial_protocol.py

Writes ``docs/dem3d_triaxial_audit.json`` and ``docs/dem3d_triaxial_audit.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tensordem.dem3d_triaxial import TriaxialConfig, TriaxialRunner, run_triaxial

torch.set_num_threads(1)

DOCS = Path(__file__).resolve().parents[1] / "docs"
D = 2.0 * TriaxialConfig().radius  # lattice constant of the simple-cubic packing


def case_a(sigma: float = 500.0, budget: int = 1400) -> dict:
    """Confinement only: servo convergence + energy ledger with moving platens."""
    cfg = TriaxialConfig(sigma_x=sigma, sigma_y=sigma, servo_gain=1e-3,
                         servo_v_max=0.02, confine_steps=budget, loading_speed=0.0)
    runner = TriaxialRunner(cfg)
    sim = runner.sim
    trace = []
    for _ in range(budget):
        sim.step()
        fx, fy = runner._lateral_forces(*[float(v) for v in sim.last_lateral_platen_reaction])
        runner._servo(fx, fy)
        trace.append((int(sim.step_count), fx / runner.area_x, fy / runner.area_y))
    led = sim.energy_ledger() if hasattr(sim, "energy_ledger") else None
    e0 = float(sim.reference_mechanical_energy_J)
    e_t = float(sim._mechanical_energy())
    w_in = (float(sim.cumulative_tool_work_J) + float(sim.cumulative_external_work_J)
            + float(sim.cumulative_platen_work_J))
    diss = (float(sim.cumulative_drag_dissipation_J) + float(sim.cumulative_particle_damping_J)
            + float(sim.cumulative_tool_damping_J) + float(sim.cumulative_friction_work_J)
            + float(sim.cumulative_rolling_work_J) + float(sim.cumulative_platen_damping_J)
            + float(sim.cumulative_bond_dissipation_J))
    frac = float(sim.cumulative_fracture_release_J)
    resid = e_t - e0 - w_in + diss + frac
    scale = max(abs(e0) + abs(w_in) + frac, 1e-30)
    return {
        "target_pressure_Pa": sigma,
        "steps": budget,
        "measured_pressure_x_Pa": trace[-1][1],
        "measured_pressure_y_Pa": trace[-1][2],
        "relative_error_x": (trace[-1][1] - sigma) / sigma,
        "relative_error_y": (trace[-1][2] - sigma) / sigma,
        "platen_displacement_m": float(sim.lateral_platen_pos[0]
                                       - (float(sim.initial_positions[:, 0].min())
                                          - cfg.radius)),
        "ledger": {"input_work_J": w_in, "dissipation_J": diss,
                   "fracture_release_J": frac,
                   "closure_residual_J": resid,
                   "closure_residual_relative": resid / scale,
                   "gate": "relative <= 1e-9",
                   "pass": abs(resid / scale) <= 1e-9},
        "trace": trace,
    }


def case_b(budget: int = 700, axial: int = 350) -> dict:
    """Anisotropic confinement held through the axial phase (servo_during_axial)."""
    cfg = TriaxialConfig(sigma_x=500.0, sigma_y=300.0, servo_gain=1e-3,
                         servo_v_max=0.02, confine_steps=budget,
                         max_axial_steps=axial, servo_during_axial=True)
    res = run_triaxial(cfg)
    res.pop("rows")
    res["energy_ledger"].pop("e_mech_0_J", None)
    return res


def case_c(strain: float = 1e-6) -> dict:
    """Affine lattice anchor: engine force law vs closed-form cubic moduli.

    For the simple-cubic bonded lattice (lattice constant d, neighbour shells
    d, sqrt(2)d, sqrt(3)d, bond stiffness k) the virial stress of a uniform
    strain field is

        sigma_ij = (k / V) sum_pairs L0^2 (n.E.n) n_i n_j,

    with V = n_particles * d^3 and every pair counted once.  For the finite
    n_x x n_y x n_z block the pair counts are
        z-axial        nx*ny*(nz-1)                        weight d^2
        x-z diagonals  2*(nx-1)*ny*(nz-1) (both signs)     weight d^2/2
        y-z diagonals  2*nx*(ny-1)*(nz-1) (both signs)     weight d^2/2
        body           4*(nx-1)*(ny-1)*(nz-1)              weight d^2/3
    so C11 = d^2*(N_z + N_xz + N_yz + 4*N_body/3)*k/V with N_xz, N_yz the
    two-sign totals, and C12 = d^2*((N_xz + N_yz)/2 + 4*N_body/3)*k/V.
    """
    cfg = TriaxialConfig()
    runner = TriaxialRunner(cfg)
    sim = runner.sim
    nx, ny, nz, r = cfg.nx, cfg.ny, cfg.nz, cfg.radius
    d = 2.0 * r
    V = nx * ny * nz * d ** 3
    k = cfg.bond_stiffness
    N_z = nx * ny * (nz - 1)
    N_xz = (nx - 1) * ny * (nz - 1)
    N_yz = nx * (ny - 1) * (nz - 1)
    N_body = (nx - 1) * (ny - 1) * (nz - 1)
    # x-z / y-z diagonals exist for both signs (2*N_xz pairs, each 0.5 d^2);
    # the four body-diagonal sign combinations contribute 4*N_body*(d^2/3).
    c11_closed = d ** 2 * (N_z + 1.0 * (N_xz + N_yz) + 4.0 * N_body / 3.0) * k / V
    c12_closed = d ** 2 * (0.5 * (N_xz + N_yz) + 4.0 * N_body / 3.0) * k / V

    X = sim.initial_positions.clone()
    i, j = sim.pairs.T
    rest = sim.rest_lengths
    kc = cfg.contact_stiffness

    def virial(comp: int) -> tuple[float, float]:
        E = torch.zeros((3, 3), dtype=sim.dtype)
        E[comp, comp] = strain
        pos = X + X @ E.T
        L = pos[j] - pos[i]
        dist = torch.linalg.vector_norm(L, dim=1)
        n = L / dist.clamp_min(1e-30)[:, None]
        ext = dist - rest
        f = k * ext + kc * (2.0 * r - dist).clamp_min(0.0)
        fvec = f[:, None] * n
        sig = (fvec[:, :, None] * L[:, None, :]).sum(0) / V
        return float(sig[comp, comp] / strain), float(sig[0, 0] / strain)

    c11_num, c12_num = virial(2)
    return {
        "n_particles": int(nx * ny * nz),
        "pairs": int(sim.pairs.shape[0]),
        "d_m": d,
        "bond_stiffness_N_per_m": k,
        "C11_closed_form_Pa": c11_closed,
        "C11_engine_affine_Pa": c11_num,
        "C12_closed_form_Pa": c12_closed,
        "C12_engine_affine_Pa": c12_num,
        "C11_relative_error": (c11_num - c11_closed) / c11_closed,
        "C12_relative_error": (c12_num - c12_closed) / c12_closed,
        "E_affine_free_lateral_Pa": c11_num - 2 * c12_num ** 2 / (c11_num + c12_num),
        "nu_affine": c12_num / (c11_num + c12_num),
        "gate": "|relative error| <= 1e-5 (exact linear algebra branch)",
        "pass": (abs(c11_num - c11_closed) / c11_closed <= 1e-5
                 and abs(c12_num - c12_closed) / c12_closed <= 1e-5),
    }


def make_figure(a: dict, b: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = [t[0] for t in a["trace"]]
    axes[0].plot(steps, [t[1] for t in a["trace"]], label="measured p_x")
    axes[0].axhline(a["target_pressure_Pa"], color="k", ls="--", label="target")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("lateral pressure (Pa)")
    axes[0].set_title("A: confinement servo convergence")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rows = b.get("rows_axial", [])
    axes[1].set_title("B: stress state at peak (anisotropic)")
    names = ["prescribed\nsigma2", "prescribed\nsigma3", "measured\np_x", "measured\np_y"]
    ss = b["stress_state_at_peak"]
    vals = [ss["prescribed_sigma2_Pa"], ss["prescribed_sigma3_Pa"],
            ss["measured_pressure_x_Pa"], ss["measured_pressure_y_Pa"]]
    axes[1].bar(names, vals, color=["#888", "#888", "#1f77b4", "#1f77b4"])
    axes[1].set_ylabel("pressure (Pa)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    a = case_a()
    b = case_b()
    c = case_c()
    out = {
        "protocol": "tensordem-dem3d-triaxial-audit-v1",
        "units": "SI",
        "case_A_confinement_and_ledger": a,
        "case_B_anisotropic_held": b,
        "case_C_affine_moduli": c,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "dem3d_triaxial_audit.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        make_figure(a, b, DOCS / "dem3d_triaxial_audit.png")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print("figure skipped:", exc)
    print(json.dumps({k: v for k, v in {
        "A_rel_err_x": a["relative_error_x"],
        "A_rel_err_y": a["relative_error_y"],
        "A_ledger_rel": a["ledger"]["closure_residual_relative"],
        "B_b_lode": b["b_lode"],
        "B_b_lode_measured": b["b_lode_measured"],
        "B_verified": b["stress_state_at_peak"]["verified"],
        "C_C11_rel_err": c["C11_relative_error"],
        "C_C12_rel_err": c["C12_relative_error"],
    }.items()}, indent=2))


if __name__ == "__main__":
    main()