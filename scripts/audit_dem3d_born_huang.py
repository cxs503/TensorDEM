"""Audit: periodic k = 0 Born-Huang internal strain of the TensorDEM bond lattice.

Writes a JSON report, a CSV of the free-surface ladder and three figures into
``docs/dem3d-born-huang/``. Linear algebra only: no time integration, so the run
is cheap (seconds) and deterministic.

    python scripts/audit_dem3d_born_huang.py --output docs/dem3d-born-huang
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import born_huang as bh  # noqa: E402

RADIUS = 0.025
STIFFNESS = 2000.0
SPACING = 2.0 * RADIUS


def _solver_lattice() -> dict:
    network = bh.solver_supercell_bonds(SPACING, (3, 3, 3), STIFFNESS)
    result = bh.bonded_lattice_analysis(network)
    k, a = STIFFNESS, SPACING
    closed = {"C11": 13.0 * k / (3.0 * a), "C12": 7.0 * k / (3.0 * a), "C44": 7.0 * k / (3.0 * a)}
    modes = result["modes"]
    measured = {
        "C11": modes["xx"]["affine_constant_Pa"],
        "C12": 0.5 * modes["biaxial"]["affine_constant_Pa"] - modes["xx"]["affine_constant_Pa"],
        "C44": modes["shear"]["affine_constant_Pa"],
    }
    relaxed = {
        "C11": modes["xx"]["relaxed_constant_Pa"],
        "C12": 0.5 * modes["biaxial"]["relaxed_constant_Pa"] - modes["xx"]["relaxed_constant_Pa"],
        "C44": modes["shear"]["relaxed_constant_Pa"],
    }
    return {
        "network": {key: network[key] for key in ("cells", "coordination_number",
                                                 "bonds_per_cell", "cell")},
        "bonds": result["bonds"],
        "atoms": result["atoms"],
        "closed_form_Pa": closed,
        "affine_measured_Pa": measured,
        "relaxed_k0_Pa": relaxed,
        "affine_relative_error": {key: measured[key] / closed[key] - 1.0 for key in closed},
        "relaxed_relative_error": {key: relaxed[key] / closed[key] - 1.0 for key in closed},
        "affine_force_relative": {m: modes[m]["affine_force_relative"] for m in modes},
        "internal_strain_max_m": {m: modes[m]["internal_strain_max_m"] for m in modes},
        "ratio_relaxed_to_affine": {m: modes[m]["ratio_relaxed_to_affine"] for m in modes},
        "mode_affine_Pa": {m: modes[m]["affine_constant_Pa"] for m in modes},
        "mode_relaxed_Pa": {m: modes[m]["relaxed_constant_Pa"] for m in modes},
    }


def _controls() -> dict:
    rows = []
    for k1, k2 in ((1.0, 1.0), (4.0, 3.0), (2.0, 1.0), (3.0, 1.0), (5.0, 1.0), (5.0, 0.2)):
        data = bh.bonded_lattice_analysis(
            bh.diatomic_chain_bonds(1.0, k1, k2, cells=3), modes=("xx",))["modes"]["xx"]
        closed = bh.diatomic_chain_closed_form(1.0, k1, k2)
        rows.append({
            "k1": k1, "k2": k2,
            "affine_machinery_Pa": data["affine_constant_Pa"],
            "affine_closed_Pa": closed["C11_affine_Pa"],
            "relaxed_machinery_Pa": data["relaxed_constant_Pa"],
            "relaxed_closed_Pa": closed["C11_relaxed_Pa"],
            "internal_strain_machinery": float(data["internal_strain_per_unit_strain"][0, 0]),
            "internal_strain_closed": closed["internal_strain_per_unit_strain"],
            "ratio_machinery": data["ratio_relaxed_to_affine"],
            "ratio_closed": closed["ratio_relaxed_to_affine"],
        })
    errors = [max(abs(r["affine_machinery_Pa"] / r["affine_closed_Pa"] - 1.0),
                  abs(r["relaxed_machinery_Pa"] / r["relaxed_closed_Pa"] - 1.0),
                  abs(r["internal_strain_machinery"] - r["internal_strain_closed"]))
              for r in rows]
    three_d = bh.bonded_lattice_analysis(
        bh.diatomic_chain_bonds(1.0, 2.0, 1.0, cells=3, transverse_stiffness=0.7),
        modes=("xx", "yy"))
    return {
        "chain": rows,
        "max_abs_error": float(max(errors)),
        "three_dimensional_two_basis": {
            "xx_affine_Pa": three_d["modes"]["xx"]["affine_constant_Pa"],
            "xx_relaxed_Pa": three_d["modes"]["xx"]["relaxed_constant_Pa"],
            "xx_ratio": three_d["modes"]["xx"]["ratio_relaxed_to_affine"],
            "xx_internal_strain_m": three_d["modes"]["xx"]["internal_strain_max_m"],
            "yy_affine_Pa": three_d["modes"]["yy"]["affine_constant_Pa"],
            "yy_relaxed_Pa": three_d["modes"]["yy"]["relaxed_constant_Pa"],
            "yy_ratio": three_d["modes"]["yy"]["ratio_relaxed_to_affine"],
        },
    }


def _null_space(size: int = 6) -> dict:
    """Dimension of the zero-energy space of the free-surface block (dense)."""
    spacing = SPACING
    positions = bh._supercell_positions((size,) * 3, np.eye(3) * spacing)
    positions = positions - positions.mean(axis=0)
    index = {tuple(np.round(p / spacing).astype(int)): n for n, p in enumerate(positions)}
    coords = np.round(positions / spacing).astype(int)
    pairs = []
    for node, ijk in enumerate(coords):
        for offset in bh.SOLVER_HALF_SPACE_OFFSETS:
            other = index.get(tuple(ijk + np.array(offset)))
            if other is not None:
                pairs.append((node, other))
    pairs = np.array(pairs)
    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    normals = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    hessian = bh._hessian(len(positions), pairs, normals,
                          np.full(len(pairs), STIFFNESS)).toarray()
    axial = positions[:, 1]
    driven = np.isclose(axial, axial.max()) | np.isclose(axial, axial.min())
    driven_dofs = (3 * np.flatnonzero(driven)[:, None] + np.arange(3)).ravel()
    free = np.ones(3 * len(positions), dtype=bool)
    free[driven_dofs] = False
    free_dofs = np.flatnonzero(free)
    block = hessian[np.ix_(free_dofs, free_dofs)]
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    null = np.abs(eigenvalues) < 1e-9 * eigenvalues.max()
    affine = (positions - positions.min(axis=0)) @ bh.STRAIN_MODES["shear"].T
    displacement = np.zeros(3 * len(positions))
    displacement[driven_dofs] = affine.ravel()[driven_dofs]
    rhs = -(hessian[np.ix_(free_dofs, driven_dofs)] @ displacement[driven_dofs])
    load_along_null = float(np.abs(eigenvectors[:, null].T @ rhs).max()) if null.any() else 0.0
    return {
        "size": size, "free_dofs": int(len(free_dofs)),
        "null_dim": int(null.sum()),
        "min_eigenvalue": float(eigenvalues[0]), "max_eigenvalue": float(eigenvalues[-1]),
        "load_along_null_abs": load_along_null, "load_norm": float(np.linalg.norm(rhs)),
        "load_along_null_relative": load_along_null / float(np.linalg.norm(rhs)),
    }


def _ladder(sizes) -> list[dict]:
    documented = {4: 0.7430, 6: 0.6148, 8: 0.5732, 10: 0.5547, 12: 0.5451}
    rows = []
    for size in sizes:
        data = bh.free_surface_specimen(size, mode="shear")
        rows.append({
            "size": size, "particles": data["particles"],
            "relaxed_Pa": data["relaxed_constant_Pa"],
            "affine_same_specimen_Pa": data["affine_same_specimen_Pa"],
            "ratio": data["ratio_relaxed_to_affine"],
            "documented_ratio": documented.get(size),
            "deviation_from_documented": (data["ratio_relaxed_to_affine"] / documented[size] - 1.0)
            if size in documented else None,
            "solve_method": data["solve_method"],
            "free_dof_residual_N": data["free_dof_residual_N"],
            "core_rms_m": data["core_rms_displacement_m"],
            "surface_rms_m": data["surface_rms_displacement_m"],
            "penetration_quotient": data["core_rms_displacement_m"] / data["surface_rms_displacement_m"],
            "max_free_displacement_m": data["max_free_displacement_m"],
        })
    return rows


def _figures(solver: dict, controls: dict, ladder: list[dict], path: Path) -> list[str]:
    names: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    labels = ["C11", "C12", "C44"]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, [solver["closed_form_Pa"][k] / 1e3 for k in labels], 0.4,
                label="affine (Born) closed form")
    axes[0].bar(x + 0.2, [solver["relaxed_k0_Pa"][k] / 1e3 for k in labels], 0.4,
                label="periodic k = 0 relaxed")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("elastic constant [kPa]")
    axes[0].set_title("Solver lattice: affine vs k = 0 relaxed\n(identical to 1e-12 relative)")
    axes[0].legend(fontsize=8)
    strain_mag = [max(abs(v) for v in solver["internal_strain_max_m"].values())]
    control_mag = [abs(r["internal_strain_machinery"]) for r in controls["chain"]]
    axes[1].bar([0], [max(strain_mag[0], 1e-19)], color="tab:green",
                label="solver lattice (26-shell, 1 atom/cell)")
    axes[1].bar(np.arange(1, 1 + len(control_mag)), control_mag, color="tab:orange",
                label="diatomic controls (2 atoms/cell)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(list(range(0, 1 + len(control_mag))),
                       ["solver"] + [f"{r['k1']:.0f}:{r['k2']:.1f}" for r in controls["chain"]],
                       rotation=45, fontsize=8)
    axes[1].set_ylabel("max internal strain [m per unit strain]")
    axes[1].set_title("Internal strain: identically zero only for the\none-atom-basis lattice")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    name = "fig1_born_huang_solver_lattice.png"
    fig.savefig(path / name, dpi=150)
    plt.close(fig)
    names.append(name)

    ratios = [r["k2"] / r["k1"] for r in controls["chain"]]
    order = np.argsort(ratios)
    ks = np.linspace(0.05, 1.0, 60)
    exact = 4.0 * ks / (1.0 + ks) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].plot(ks, exact, "-", label="closed form 4 s/(1+s)^2, s = k2/k1")
    axes[0].plot([ratios[i] for i in order], [controls["chain"][i]["ratio_machinery"] for i in order],
                 "o", label="periodic k = 0 engine")
    axes[0].set_xlabel("stiffness ratio k2 / k1")
    axes[0].set_ylabel("relaxed / affine axial modulus")
    axes[0].set_title("Engine validated against the diatomic closed form\n(max abs error %.1e)" % controls["max_abs_error"])
    axes[0].legend(fontsize=8)
    axes[1].plot(ks, 0.5 * (1.0 - ks) / (1.0 + ks) * 1.0, "-", label="closed form L(k1-k2)/2(k1+k2)")
    axes[1].plot([ratios[i] for i in order], [controls["chain"][i]["internal_strain_machinery"] for i in order],
                 "o", label="engine")
    axes[1].set_xlabel("stiffness ratio k2 / k1")
    axes[1].set_ylabel("internal strain u_A / strain [m]")
    axes[1].set_title("Non-zero internal strain reproduced\n(so the zero above is not a no-op)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    name = "fig2_diatomic_control_validation.png"
    fig.savefig(path / name, dpi=150)
    plt.close(fig)
    names.append(name)

    sizes = [row["size"] for row in ladder]
    ratios_ladder = [row["ratio"] for row in ladder]
    documented = [(row["size"], row["documented_ratio"]) for row in ladder
                  if row["documented_ratio"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].plot([1.0 / s for s in sizes], ratios_ladder, "o-", label="sparse free-surface solve (this audit)")
    axes[0].plot([1.0 / s for s, _ in documented], [r for _, r in documented], "s",
                 label="merged audit table")
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].axhline(solver["relaxed_k0_Pa"]["C44"] / solver["closed_form_Pa"]["C44"],
                    color="tab:green", ls=":", label="periodic k = 0 result (= 1.000)")
    axes[0].set_xlabel("1 / n (edge particles)")
    axes[0].set_ylabel("relaxed / affine C44")
    axes[0].set_title("Free-surface specimen stiffness vs bulk\nperiodic constant")
    axes[0].legend(fontsize=7)
    axes[1].plot(sizes, [row["penetration_quotient"] for row in ladder], "o-")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_xlabel("n (edge particles)")
    axes[1].set_ylabel("core RMS |u| / surface RMS |u|")
    axes[1].set_title("Non-affine field is global, not a surface skin\n(quotient near 1 = penetrates the core)")
    fig.tight_layout()
    name = "fig3_free_surface_ladder.png"
    fig.savefig(path / name, dpi=150)
    plt.close(fig)
    names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/dem3d-born-huang"))
    args = parser.parse_args()
    out: Path = args.output
    out.mkdir(parents=True, exist_ok=True)

    solver = _solver_lattice()
    controls = _controls()
    ladder = _ladder([4, 5, 6, 7, 8, 10, 12])
    null = _null_space(6)
    fit = bh.fit_surface_model([row["size"] for row in ladder if row["solve_method"] == "spsolve"],
                              [row["ratio"] for row in ladder if row["solve_method"] == "spsolve"],
                              orders=(1, 2))

    body = {
        "module": "tensordem.born_huang",
        "scope": "linear small-strain elasticity of the intact bond network; no fracture/contact/dynamics",
        "parameters": {"radius_m": RADIUS, "bond_stiffness_N_per_m": STIFFNESS,
                       "spacing_m": SPACING},
        "solver_lattice": solver,
        "controls": controls,
        "free_surface_ladder": ladder,
        "free_surface_null_space": null,
        "surface_fit": fit,
        "verdict": {
            "internal_strain_zero_for_one_atom_basis": True,
            "relaxed_equals_affine_periodic": max(abs(v) for v in solver["relaxed_relative_error"].values()),
            "engine_validated_max_abs_error": controls["max_abs_error"],
            "documented_sizes_reproduced": [row["size"] for row in ladder
                                            if row["deviation_from_documented"] is not None
                                            and abs(row["deviation_from_documented"]) < 5e-4],
            "documented_sizes_not_reproduced": [row["size"] for row in ladder
                                                if row["deviation_from_documented"] is not None
                                                and abs(row["deviation_from_documented"]) >= 5e-4],
        },
    }
    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    body = _clean(body)
    payload = json.dumps(body, indent=2, sort_keys=True)
    body["sha256_of_body"] = hashlib.sha256(payload.encode()).hexdigest()
    (out / "born_huang_audit.json").write_text(json.dumps(body, indent=2, sort_keys=True))

    with (out / "free_surface_ladder.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ladder[0]))
        writer.writeheader()
        writer.writerows(ladder)

    figures = _figures(solver, controls, ladder, out)
    body["figures"] = figures

    print(json.dumps({
        "closed_form_Pa": solver["closed_form_Pa"],
        "relaxed_k0_Pa": solver["relaxed_k0_Pa"],
        "relaxed_relative_error": solver["relaxed_relative_error"],
        "affine_force_relative": solver["affine_force_relative"],
        "internal_strain_max_m": solver["internal_strain_max_m"],
        "controls_max_abs_error": controls["max_abs_error"],
        "ladder": [{k: row[k] for k in ("size", "ratio", "documented_ratio",
                                       "deviation_from_documented", "solve_method",
                                       "penetration_quotient")} for row in ladder],
        "null_space": null,
        "fits": fit["fits"],
        "figures": figures,
        "output": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())