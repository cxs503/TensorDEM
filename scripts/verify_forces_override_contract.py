"""Evidence sheet for the `forces()` override / probe-contract repair.

Capability under audit: the 3-D DEM solver's *monitoring* contract, i.e. that
``diagnostics()`` (and every energy/load audit built on it) can be called on any
``IceDEM3D`` subclass without advancing solver history.  On `main` that contract
was broken for two subclasses, which made the moving-wedge bow-resistance
capability and the three-field hull-ice pipeline unable to complete a pass.

Nothing here is hand-copied: every number is measured from the public API and
written to ``artifacts/forces_override_contract/`` as JSON, CSV and one figure.

Run:  PYTHONPATH=src python3 scripts/verify_forces_override_contract.py
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import torch

import tensordem.ridge_dem  # noqa: F401  (subclass registration)
import tensordem.threefield.hull_ice  # noqa: F401
from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D
from tensordem.rudder_ice_load import RudderIceLoadDEM
from tensordem.threefield.hull_ice import HullIceCase, run_hull_ice_case

OUT = Path("artifacts/forces_override_contract")
BASE_KEYWORDS = ("update_fracture", "update_history")


def small_config(**overrides) -> DEM3DConfig:
    base = dict(
        nx=3, ny=2, nz=3, radius=0.02, tool_speed=0.1,
        bond_stiffness=1000.0, contact_stiffness=3000.0,
        breaking_strain=0.5, shear_breaking_strain=0.5,
        drag=0.1, contact_damping=0.5,
        fix_x_edges=False, fix_bottom=False,
    )
    base.update(overrides)
    return DEM3DConfig(**base)


def class_graph(root: type) -> list[type]:
    seen, stack = [], [root]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.append(current)
        stack.extend(current.__subclasses__())
    return seen


def wedge(overlap: bool = True) -> MovingWedgeBowIceDEM3D:
    sim = MovingWedgeBowIceDEM3D(small_config(), bow_speed=0.1, wedge_angle_deg=45.0)
    if overlap:
        sim.positions[:, 0] = sim.bow_tip_x
        sim.positions[:, 2] = sim.bow_tip_z
        sim.velocities.zero_()
    return sim


# ---------------------------------------------------------------------------
# 1. static contract matrix: does each override accept the probe keywords?
# ---------------------------------------------------------------------------
def signature_matrix() -> list[dict]:
    rows = []
    for cls in class_graph(IceDEM3D):
        if "forces" not in cls.__dict__:
            continue
        params = inspect.signature(cls.forces).parameters
        rows.append({
            "class": cls.__name__,
            "overrides_forces": True,
            "has_update_fracture": "update_fracture" in params,
            "has_update_history": "update_history" in params,
            "binds_probe_call": _binds(cls),
        })
    return rows


def _binds(cls) -> bool:
    try:
        inspect.signature(cls.forces).bind(
            object(), None, update_fracture=False, update_history=False)
    except TypeError:
        return False
    return True


# ---------------------------------------------------------------------------
# 2. probe invariance: a probe must not commit the stored reaction
# ---------------------------------------------------------------------------
def probe_invariance_table() -> list[dict]:
    sim = wedge()
    sim.forces()  # owning call commits
    committed = sim.last_reaction.clone()
    rows = []
    for advance in (0.0, 0.25, 0.5, 1.0):
        baseline = sim.last_reaction.clone()
        sim.time += advance if advance else 0.0
        diag = sim.diagnostics()
        probed = torch.tensor(
            [diag["bow_reaction_x_N"], diag["bow_reaction_y_N"], diag["bow_reaction_z_N"]],
            dtype=committed.dtype)
        rows.append({
            "bow_advance_m": advance * sim.bow_speed,
            "probe_reaction_x_N": float(probed[0]),
            "committed_reaction_x_N": float(baseline[0]),
            "stored_after_probe_x_N": float(sim.last_reaction[0]),
            "commit_drift_N": float((sim.last_reaction - baseline).abs().max()),
            "probe_minus_committed_N": float((probed - baseline).abs().max()),
        })
        sim.time -= advance if advance else 0.0  # reposition for the next probe
    return rows


# ---------------------------------------------------------------------------
# 3. analytic anchors (exact identities, not fitted numbers)
# ---------------------------------------------------------------------------
def analytic_anchors() -> list[dict]:
    rows = []
    # (a) bow reaction is the exact negative sum of the nodal bow forces.
    sim = wedge()
    force, reaction = sim.forces()
    nodal, analytic_reaction, overlap = sim._wedge_contact()
    rows.append({
        "anchor": "bow reaction = -sum(nodal bow force)",
        "measured": float(reaction[0]),
        "reference": float(analytic_reaction[0]),
        "abs_error": float((reaction[0] - analytic_reaction[0]).abs()),
        "rel_error": _rel(float(reaction[0]), float(analytic_reaction[0])),
    })
    # (b) elastic bow contact energy = 0.5 k sum(overlap^2), the closed form
    #     the diagnostics field `bow_contact_energy_J` publishes.
    diag = sim.diagnostics()
    closed = 0.5 * sim.config.contact_stiffness * float(overlap.square().sum())
    rows.append({
        "anchor": "bow_contact_energy_J = 0.5 k sum(overlap^2)",
        "measured": float(diag["bow_contact_energy_J"]),
        "reference": closed,
        "abs_error": abs(float(diag["bow_contact_energy_J"]) - closed),
        "rel_error": _rel(float(diag["bow_contact_energy_J"]), closed),
    })
    # (c) a rigid probe must not change state at all (bitwise identical force).
    probe_a, _ = sim.forces(update_fracture=False, update_history=False)
    probe_b, _ = sim.forces(update_fracture=False, update_history=False)
    rows.append({
        "anchor": "repeated probe force is bitwise identical",
        "measured": float(probe_a.abs().sum()),
        "reference": float(probe_b.abs().sum()),
        "abs_error": float((probe_a - probe_b).abs().max()),
        "rel_error": 0.0 if torch.equal(probe_a, probe_b) else float("inf"),
    })
    return rows


def _rel(measured: float, reference: float) -> float:
    scale = max(abs(reference), 1e-30)
    return abs(measured - reference) / scale


# ---------------------------------------------------------------------------
# 4. the capability that was dead: hull-ice resistance history
# ---------------------------------------------------------------------------
def hull_ice_runs() -> dict:
    """Two engaged cases.

    Only thicknesses with the *same* lattice parity are compared: with
    ``layers = round(thickness / 2r)`` an even layer count plus
    ``tip_height_fraction = 0.5`` puts the wedge tip exactly between two layers,
    so a 0.10 m case (2 layers) records zero contact in the window and would be
    read as "zero resistance" rather than "no engagement".  That parity artefact
    is a documented limitation, not a physics result.
    """
    runs = {}
    for thickness in (0.15, 0.25):
        case = HullIceCase(thickness_m=thickness, particle_radius_m=0.025,
                           n_length=5, n_width=2, steps=40)
        result = run_hull_ice_case(case)
        history = result["history"]
        runs[f"{thickness:.2f}"] = {
            "thickness_m": thickness,
            "realised_thickness_m": case.realised_thickness_m,
            "layers": case.layers,
            "steps": len(history),
            "max_ice_resistance_N": max(row["ice_resistance_N"] for row in history),
            "max_contact_particles": max(row["contact_particles"] for row in history),
            "broken_bonds_final": history[-1]["broken_bonds"],
            "history": history,
            "summary": {k: v for k, v in result.items() if k != "history"},
        }
    return runs


# ---------------------------------------------------------------------------
def plot(probe_rows: list[dict], runs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    advance = [row["bow_advance_m"] * 1000.0 for row in probe_rows]
    ax.plot(advance, [row["probe_reaction_x_N"] for row in probe_rows], "o-",
            label="probe value (measured by diagnostics)")
    ax.plot(advance, [row["stored_after_probe_x_N"] for row in probe_rows], "s--",
            label="stored last_reaction after probe")
    ax.set_xlabel("bow advance since commit  [mm]")
    ax.set_ylabel("bow reaction x  [N]")
    ax.set_title("probe measures without committing\n(stored trace flat ⇒ invariant holds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, run in runs.items():
        x0 = run["history"][0]["bow_tip_x_m"]
        advance = [(row["bow_tip_x_m"] - x0) * 1e3 for row in run["history"]]
        ax.plot(advance, [row["ice_resistance_N"] for row in run["history"]],
                label=f"ice thickness {run['realised_thickness_m']:.2f} m "
                      f"({run['layers']} layers)")
    ax.set_xlabel("bow advance  [mm]")
    ax.set_ylabel("ice resistance  [N]")
    ax.set_title("three-field hull-ice pipeline\n(dead on main: TypeError every pass)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "forces_override_contract.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signatures = signature_matrix()
    probes = probe_invariance_table()
    anchors = analytic_anchors()
    runs = hull_ice_runs()

    worst = max(row["rel_error"] for row in anchors)
    payload = {
        "capability": "IceDEM3D forces() override / probe side-effect contract",
        "signatures": signatures,
        "probe_invariance": probes,
        "analytic_anchors": anchors,
        "worst_anchor_rel_error": worst,
        "all_anchors_within_3pct": bool(worst < 0.03),
        "max_commit_drift_N": max(row["commit_drift_N"] for row in probes),
        "hull_ice": runs,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2))

    with (OUT / "probe_invariance.csv").open("w") as handle:
        handle.write("bow_advance_m,probe_reaction_x_N,committed_reaction_x_N,"
                     "stored_after_probe_x_N,commit_drift_N\n")
        for row in probes:
            handle.write(
                f"{row['bow_advance_m']:.6e},{row['probe_reaction_x_N']:.9e},"
                f"{row['committed_reaction_x_N']:.9e},{row['stored_after_probe_x_N']:.9e},"
                f"{row['commit_drift_N']:.3e}\n")

    plot(probes, runs)

    print(f"signatures checked           : {len(signatures)}")
    for row in signatures:
        print(f"  {row['class']:<24} update_history={row['has_update_history']}"
              f"  binds_probe={row['binds_probe_call']}")
    print(f"max commit drift (probes)    : {payload['max_commit_drift_N']:.3e} N")
    print(f"worst analytic anchor error  : {worst:.3e}")
    print("all anchors < 3%             :", payload["all_anchors_within_3pct"])
    for key, run in runs.items():
        print(f"hull-ice t={key} m ({run['layers']} layers): max resistance "
              f"{run['max_ice_resistance_N']:.2f} N, contact particles "
              f"{run['max_contact_particles']:.0f}, "
              f"broken bonds {run['broken_bonds_final']:.0f}")
    assert all(row["binds_probe_call"] and row["has_update_history"] for row in signatures)
    assert payload["max_commit_drift_N"] == 0.0
    assert payload["all_anchors_within_3pct"]
    assert all(run["max_contact_particles"] > 0 for run in runs.values()), \
        "a case never engaged the ice: the curve would be a null result"
    assert math.isfinite(worst)
    print(f"artefacts -> {OUT}/audit.json, probe_invariance.csv, forces_override_contract.png")


if __name__ == "__main__":
    main()