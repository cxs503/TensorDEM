"""Scaling of the fracture-field bond-topology search: pairwise loop vs blocked cdist.

The audit numbers in ``docs/fracture_gen_mass_footprint_audit.md`` were produced
by a generator whose bond search was an O(N^2) Python loop (~6.8 s for the
512-particle reference lattice).  That is why the published audit test module
could not be executed inside a normal CI budget, and part of why the contract
kept being "restored" by successive agents without ever staying green.

This script measures both implementations on the same lattices and writes
``artifacts/fracture_gen_audit/bond_topology_scaling.{json,png}``.  The
vectorised search is verified against the loop *by the test suite*
(``tests/test_fracture_gen_mass_footprint.py::TestTopologyAndCriterionWiring``),
not merely timed here.

Run: python3 scripts/plot_fracture_gen_bond_topology_scaling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.fracture_gen import build_bond_topology, generate_particle_lattice

RADIUS = 0.125
SPACING = 2.0 * RADIUS
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "fracture_gen_audit"
N_VALUES = (3, 4, 5, 6, 7, 8)


def brute_force(positions: torch.Tensor, radius: float) -> list[list[int]]:
    """Reference implementation: the literal O(N^2) Python loop."""
    spacing = 2.0 * radius
    pairs: list[list[int]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if float(torch.linalg.norm(positions[j] - positions[i])) <= spacing * 1.01:
                pairs.append([i, j])
    return pairs


def measure():
    rows = []
    for n in N_VALUES:
        extent = n * SPACING
        domain = (0.0, extent, 0.0, extent, 0.0, extent)
        positions = generate_particle_lattice(domain, RADIUS)
        self_check = len(positions) == n**3
        t0 = time.perf_counter()
        loop_pairs = brute_force(positions, RADIUS)
        t_loop = time.perf_counter() - t0
        t0 = time.perf_counter()
        bonds, rest = build_bond_topology(positions, RADIUS)
        t_vec = time.perf_counter() - t0
        same = torch.equal(bonds, torch.tensor(loop_pairs, dtype=torch.long))
        rows.append(
            {
                "n_layers": n,
                "n_particles": len(positions),
                "n_bonds": int(bonds.shape[0]),
                "n_bonds_analytic": 3 * n**2 * (n - 1),
                "loop_seconds": t_loop,
                "vectorised_seconds": t_vec,
                "speedup": t_loop / t_vec if t_vec > 0 else float("inf"),
                "identical_to_loop": bool(same),
                "lattice_count_as_expected": self_check,
            }
        )
        print(json.dumps(rows[-1]))
    return rows


def plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [row["n_particles"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].loglog(ns, [row["loop_seconds"] for row in rows], "o-", color="tab:red",
                   label="pairwise Python loop (reference)")
    axes[0].loglog(ns, [row["vectorised_seconds"] for row in rows], "s-", color="tab:blue",
                   label="row-blocked cdist (landed)")
    axes[0].set_xlabel("particles N")
    axes[0].set_ylabel("bond-topology build time (s)")
    axes[0].set_title("Bond search: identical output, O(N^2) constant factored out")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    axes[1].bar([str(row["n_layers"]) for row in rows],
                [row["speedup"] for row in rows], color="tab:green", width=0.6)
    for idx, row in enumerate(rows):
        axes[1].annotate("%.0fx" % row["speedup"], (idx, row["speedup"]),
                         ha="center", va="bottom", fontsize=8)
    axes[1].set_xlabel("lattice layers n (N = n^3)")
    axes[1].set_ylabel("speedup vs loop")
    axes[1].set_yscale("log")
    axes[1].set_title("Measured speedup (identical bond order)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Fracture-field generator: why the audit contract could not run in CI", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bond_topology_scaling.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = measure()
    (OUT_DIR / "bond_topology_scaling.json").write_text(json.dumps(rows, indent=2))
    plot(rows)
    print("artifacts ->", OUT_DIR)