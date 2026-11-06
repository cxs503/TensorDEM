#!/usr/bin/env python3
"""Replay the runnable tutorial and record its numbers, log and figures.

The tutorial ``docs/tutorials/first_verified_run.md`` is executed exactly as the
gate ``tests/test_tutorial_first_verified_run.py`` executes it (the same fenced
``python`` blocks, in order, one namespace).  Everything written by this script is
therefore *generated*, never transcribed:

* ``artifacts/tutorial_first_verified_run/run_log.txt``  -- captured stdout
* ``artifacts/tutorial_first_verified_run/summary.json`` -- the headline quantities
* ``artifacts/tutorial_first_verified_run/figures/directional_modulus.png``
* ``artifacts/tutorial_first_verified_run/figures/energy_ledger_convergence.png``

Usage::

    python scripts/collect_tutorial_first_verified_run.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "tutorial_first_verified_run"
FIGURES = OUT / "figures"
sys.path.insert(0, str(ROOT / "src"))


def tutorial_source() -> str:
    """The tutorial's python blocks, via the gate module (single source of truth)."""
    gate = ROOT / "tests" / "test_tutorial_first_verified_run.py"
    spec = importlib.util.spec_from_file_location("_tutorial_gate", gate)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.tutorial_source()


def run() -> tuple[dict[str, object], str]:
    source = tutorial_source()
    namespace: dict[str, object] = {"__name__": "__tutorial__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(source, "docs/tutorials/first_verified_run.md", "exec"), namespace)  # noqa: S102
    return namespace, buffer.getvalue()


def summary(namespace: dict[str, object], log: str) -> dict[str, object]:
    """Headline numbers, read off the executed tutorial namespace."""
    k_bond = float(namespace["K_BOND"])
    return {
        "tutorial": "docs/tutorials/first_verified_run.md",
        "parameters": {
            "bond_stiffness_N_per_m": k_bond,
            "radius_m": float(namespace["RADIUS"]),
            "thickness_m": float(namespace["THICKNESS"]),
            "density_kg_per_m3": float(namespace["DENSITY"]),
            "spacing_m": float(namespace["SPACING"]),
            "particles": int(namespace["n_particles"]),
            "hand_counted_bonds": int(namespace["hand_bonds"]),
        },
        "static_checks": {
            "hand_vs_module_tensor_error": float(namespace["le"].tensor_error(
                namespace["C_hand"], namespace["C_ref"])),
            "engine_vs_closed_form_tensor_error": float(namespace["rel_err"]),
            "free_patch_density_ratio": float(namespace["raw_density"] / namespace["bulk_density"]),
            "deficit_corrected_density_ratio": float(namespace["corrected"] / namespace["bulk_density"]),
            "young_modulus_0deg_N_per_m": float(namespace["young_0"]),
            "young_modulus_45deg_N_per_m": float(namespace["young_45"]),
            "anisotropy_ratio_45_over_0": float(namespace["young_45"] / namespace["young_0"]),
            "isotropic_modulus_N_per_m": float(namespace["iso_moduli"][0]["youngs_modulus_N_per_m"]),
            "isotropic_modulus_directional_spread_N_per_m": float(namespace["spread"]),
            "isotropic_poisson_ratio": float(namespace["iso_moduli"][0]["poisson_ratio"]),
            "engine_isotropic_tensor_error": float(namespace["le"].tensor_error(
                namespace["iso_measured"], namespace["iso_ref"])),
        },
        "dynamic_checks": {
            "ledger_relative_residual_dt": float(namespace["coarse"]),
            "ledger_relative_residual_dt_half": float(namespace["fine"]),
            "residual_ratio": float(namespace["fine"] / namespace["coarse"]),
            "tool_reaction_N": float(namespace["run_impact"].reaction),
            "broken_bonds": int(namespace["run_impact"].broken),
        },
        "calibration": {
            # ``bond_stiffness_from_young_modulus`` returns the *axis* bond stiffness
            # k_axis = 3/4 E t, which is what ``DEMConfig.bond_stiffness`` expects; the
            # isotropic diagonal stiffness is k_diag = 0.5 k_axis, and E = 4/3 k_axis / t
            # only holds with that correction applied.  Recording a single number here
            # used to invite a factor-2 calibration error (k_axis vs 2 k_axis = 19 GPa).
            "young_modulus_Pa": float(namespace["ice_young"]),
            "axis_bond_stiffness_N_per_m": float(namespace["k_axis"]),
            "diagonal_bond_stiffness_N_per_m": float(namespace["k_diag"]),
            "diagonal_to_axis_ratio": float(namespace["le"].ISOTROPIC_DIAGONAL_SCALE),
        },
        "log_lines": log.strip().splitlines(),
    }


def figures(namespace: dict[str, object]) -> list[str]:
    """Two figures: direction dependence of E, and the O(dt) ledger residual."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - viz extra not installed
        print("matplotlib unavailable; skipping figures")
        return []

    le = namespace["le"]
    k_bond = float(namespace["K_BOND"])
    angles = [5.0 * index for index in range(72)]
    default = [row["youngs_modulus_N_per_m"] for row in le.directional_moduli(angles, k_bond)]
    isotropic = [
        row["youngs_modulus_N_per_m"]
        for row in le.directional_moduli(angles, k_bond, le.ISOTROPIC_DIAGONAL_SCALE)
    ]

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []

    polar_path = FIGURES / "directional_modulus.png"
    theta = [math.radians(angle) for angle in angles]
    fig = plt.figure(figsize=(5.6, 5.0))
    axis = fig.add_subplot(111, projection="polar")
    axis.plot(theta, default, color="tab:red", label="engine default $k_d = k_a$")
    axis.plot(theta, isotropic, color="tab:blue", label=r"isotropic $k_d = k_a/2$")
    axis.set_title("2-D bonded lattice: $E(\\theta)$ in N/m", pad=18)
    axis.set_rlabel_position(135)
    axis.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(polar_path, dpi=140)
    plt.close(fig)
    written.append(str(polar_path.relative_to(ROOT)))

    # Energy-ledger residual under dt refinement: rerun the tutorial's scheme at 4 dt levels.
    import torch  # noqa: F401  (engine is torch-based)

    cfg_base = namespace["impact_config"]
    coarse_steps = 400
    dt0 = cfg_base.recommended_dt
    scales = [1, 2, 4, 8]
    residuals, dts = [], []
    for scale in scales:
        from tensordem.dem import DEMConfig, IceDEM

        cfg = DEMConfig(**{**cfg_base.__dict__, "dt": dt0 / scale})
        sim = IceDEM(cfg)
        for _ in range(int(round(coarse_steps * scale))):
            sim.step()
        diag = sim.diagnostics()
        mechanical = sim.mechanical_energy()["mechanical_J"]
        residual = diag["tool_work_J"] - mechanical - diag["damping_dissipation_J"] - diag["fracture_release_J"]
        residuals.append(abs(residual / diag["tool_work_J"]))
        dts.append(cfg.dt)

    convergence = FIGURES / "energy_ledger_convergence.png"
    fig, axis = plt.subplots(figsize=(5.6, 4.0))
    axis.loglog(dts, residuals, "o-", color="tab:purple", label="measured |residual|/W_tool")
    reference = [residuals[0] * (dt / dts[0]) for dt in dts]
    axis.loglog(dts, reference, "--", color="black", label="slope 1 (first order)")
    axis.set_xlabel("time step dt [s]")
    axis.set_ylabel("relative energy-ledger residual [-]")
    axis.set_title("Semi-implicit Euler: ledger residual is O(dt)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(convergence, dpi=140)
    plt.close(fig)
    written.append(str(convergence.relative_to(ROOT)))
    namespace["_convergence"] = {"dt": dts, "residual": residuals}
    return written


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    namespace, log = run()
    (OUT / "run_log.txt").write_text(log + "\n", encoding="utf-8")
    payload = summary(namespace, log)
    written = figures(namespace)
    payload["figures"] = written
    if "_convergence" in namespace:
        payload["dynamic_checks"]["dt_refinement"] = namespace["_convergence"]
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("static_checks", "dynamic_checks", "calibration")},
                     indent=2, ensure_ascii=False))
    print("\nwritten:", OUT.relative_to(ROOT), "->", ", ".join(written) or "no figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())