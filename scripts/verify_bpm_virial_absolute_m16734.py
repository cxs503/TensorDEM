"""m16734 audit: is the *absolute* level of the BPM virial stress instrument right?

Question
--------
``bpm_calibration_feasibility.virial_stress_tensor`` feeds the Brazilian-disc
acceptance gate (AC5, Hondros ``sigma_min/sigma_max -> -3``).  That gate only
consumes a *ratio*, so the absolute level of the tensor was never checked.  This
script checks it against an **independent path** that shares nothing with the
virial: the plane-cut traction of :mod:`tensordem.granular3d.stress`
(``sum of the forces crossing a plane / area``).

Two claims are tested, both pre-registered in the code below:

* **C1 (pair counting).** The historical implementation added the full pair
  tensor ``l (x) f`` to *both* ends of every pair, so the volume-weighted sum over
  particles is ``2 * sum_c l (x) f``.  Prediction: the region/global average is a
  factor exactly *2* too large.
* **C2 (normalisation).** The historical normalisation used the lattice cell
  ``8 R^3`` as the volume per particle.  The specific volume of a finite specimen
  is ``V/N``; with the gauge volume ``A * L_gauge`` the volume average must equal
  the plane-cut traction exactly.  Prediction: the historical value is off by the
  analytic factor ``2 * V_p(true) / (8 R^3)``.

State used for the comparison: a *homogeneous affine* strain of the whole gauge,
i.e. exactly the state in which the volume average and the plane cut must agree by
the virial theorem.  It is static (every particle held), so no dynamics enters.

Outputs
-------
``artifacts/m16734_virial_absolute/verification.json`` -- every number of the
report; ``fig_virial_absolute.png`` -- the figure.  Nothing is hand copied.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tensordem.bpm_calibration_feasibility import (
    LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR,
    PARTICLE_RADIUS,
    REFERENCE_VECTOR,
    SpecimenGeometry,
    build_axial,
    pair_stress_state,
    region_stress,
    specific_volume,
    virial_stress_tensor,
)
from tensordem.granular3d.stress import (
    contact_outer,
    plane_stress,
    stress_tensor,
)

OUT = pathlib.Path("artifacts/m16734_virial_absolute")


# ---------------------------------------------------------------------------
# homogeneous affine state
# ---------------------------------------------------------------------------
def affine_state(n_layers: int, cross_n: int, strain: float, *,
                 kill_fraction: float = 0.0, seed: int = 0):
    """Build the simple-cubic specimen and stretch it *affinely* about its base.

    ``strain > 0`` is tension, ``< 0`` compression.  ``kill_fraction`` breaks that
    fraction of bonds *before* the strain is applied, so the state carries the
    two branches the stress instrument must separate (dead bonds -> contact
    branch; alive bonds -> bond branch).
    """
    geo = SpecimenGeometry(n_layers=n_layers, cross_n=cross_n, radius=PARTICLE_RADIUS)
    engine, top, bottom, meta = build_axial(REFERENCE_VECTOR, geo,
                                           mass_scaling=100.0, decay_steps=15)
    base = engine.positions.clone()
    zmin = float(base[:, 2].min())
    lever = (base[:, 2] - zmin)[:, None]
    engine.positions = base + strain * lever * torch.tensor(
        [0.0, 0.0, 1.0], dtype=engine.dtype)
    if kill_fraction > 0.0:
        rng = np.random.default_rng(seed)
        alive = engine.alive.clone()
        n_kill = int(round(kill_fraction * int(alive.numel())))
        dead = rng.choice(int(alive.numel()), size=n_kill, replace=False)
        alive[torch.from_numpy(np.sort(dead))] = False
        engine.alive = alive
    # static state: hold every particle so no dynamics contaminates the identity
    engine.fixed = torch.ones(engine.positions.shape[0], dtype=torch.bool)
    return engine, geo, meta


def measure(engine, geo, *, gauge: float | None = None) -> dict:
    """All stress ordinates of one state, from independent paths."""
    state = pair_stress_state(engine)
    n = state["n_particles"]
    area = float(geo.area)
    gauge = float(geo.thickness if gauge is None else gauge)
    V_total = area * gauge
    V_p_true = specific_volume(V_total, n)
    V_p_legacy = LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR * PARTICLE_RADIUS ** 3

    branch, force, center = state["branch"], state["force_on_i"], \
        engine.positions[torch.from_numpy(state["i"])].detach().cpu().numpy()

    # --- independent path: plane cut through the middle of the gauge ---------
    z_all = engine.positions[:, 2].detach().cpu().numpy()
    plane = 0.5 * (float(z_all.min()) + float(z_all.max()))
    sigma_plane = plane_stress(branch, force, center, axis=2, plane=plane, area=area)

    # --- virial paths --------------------------------------------------------
    sigma_virial = stress_tensor(branch, force, volume=V_total)          # exact virial
    mean_half = np.asarray(virial_stress_tensor(engine).detach().cpu().numpy()).mean(axis=0)
    mean_gauge = np.asarray(virial_stress_tensor(
        engine, volume_per_particle=V_p_true).detach().cpu().numpy()).mean(axis=0)
    mean_legacy = np.asarray(virial_stress_tensor(
        engine, volume_per_particle=V_p_legacy, scatter="legacy"
    ).detach().cpu().numpy()).mean(axis=0)

    # --- algebraic identity: sum_p V_p sigma_p vs sum_c l (x) f --------------
    ref_sum = contact_outer(branch, force).sum(axis=0)
    sum_half = float(np.abs(
        virial_stress_tensor(engine, volume_per_particle=1.0).sum(dim=0).numpy()
        - ref_sum).max())
    sum_legacy = float(np.abs(
        virial_stress_tensor(engine, volume_per_particle=1.0,
                             scatter="legacy").sum(dim=0).numpy()
        - 2.0 * ref_sum).max())

    def _rel(a: float, b: float) -> float:
        return float(abs(a - b) / abs(b)) if b != 0.0 else float("inf")

    return {
        "n_particles": n,
        "n_pairs": int(state["branch"].shape[0]),
        "n_dead_bonds": int(np.count_nonzero(~state["bonded"])),
        "n_contact_pairs": int(np.count_nonzero(state["contact"])),
        "area_m2": area,
        "gauge_m": gauge,
        "V_p_true": V_p_true,
        "V_p_legacy": V_p_legacy,
        "V_p_legacy_over_true": V_p_legacy / V_p_true,
        "sigma_zz_plane_cut": float(sigma_plane[2, 2]),
        "sigma_zz_exact_virial": float(sigma_virial[2, 2]),
        "sigma_zz_per_particle_gauge_volume": float(mean_gauge[2, 2]),
        "sigma_zz_per_particle_cell_volume": float(mean_half[2, 2]),
        "sigma_zz_legacy_double_count": float(mean_legacy[2, 2]),
        "rel_err_exact_virial_vs_plane": _rel(float(sigma_virial[2, 2]),
                                              float(sigma_plane[2, 2])),
        "rel_err_per_particle_gauge_vs_plane": _rel(float(mean_gauge[2, 2]),
                                                    float(sigma_plane[2, 2])),
        "rel_err_per_particle_cell_vs_plane": _rel(float(mean_half[2, 2]),
                                                   float(sigma_plane[2, 2])),
        "rel_err_legacy_vs_plane": _rel(float(mean_legacy[2, 2]),
                                        float(sigma_plane[2, 2])),
        "predicted_legacy_over_plane": 2.0 * V_p_true / V_p_legacy,
        "measured_legacy_over_plane": float(mean_legacy[2, 2])
        / float(sigma_plane[2, 2]),
        "sum_V_sigma_minus_ref_max": sum_half,
        "sum_V_sigma_legacy_minus_2ref_max": sum_legacy,
        "sigma_plane_full": sigma_plane.tolist(),
        "sigma_exact_virial_full": sigma_virial.tolist(),
    }


# ---------------------------------------------------------------------------
# mutation (non-vacuity) set: each classic bug must move the gated number
# ---------------------------------------------------------------------------
def mutated_sigma_zz(engine, geo, *, kind: str) -> float:
    state = pair_stress_state(engine)
    branch = state["branch"].copy()
    force = state["force_on_i"].copy()
    bonded = state["bonded"].copy()
    contact = state["contact"].copy()
    area = float(geo.area)
    V_total = area * float(geo.thickness)
    if kind == "reference":
        pass
    elif kind == "force_sign_flipped":
        force = -force
    elif kind == "branch_reversed":
        branch = -branch
    elif kind == "pair_counted_twice":
        outer = contact_outer(branch, force).sum(axis=0)
        return float(2.0 * outer[2, 2] / V_total)
    elif kind == "dead_bonds_kept":
        # rebuild the bond force without the alive mask
        pos = engine.positions
        i, j = engine.pairs.T
        d = pos[j] - pos[i]
        length = torch.linalg.vector_norm(d, dim=1).clamp_min(
            torch.finfo(pos.dtype).eps)
        unit = (d / length[:, None]).detach().cpu().numpy()
        f_bond = float(engine.config.bond_stiffness) * (
            length - engine.rest_lengths).detach().cpu().numpy()
        overlap = 2.0 * float(engine.config.radius) - length.detach().cpu().numpy()
        f_contact = np.where(contact, float(engine.config.contact_stiffness)
                             * overlap, 0.0)
        force = ((f_bond - f_contact)[:, None] * unit)
    elif kind == "contacts_not_gated":
        pos = engine.positions
        i, j = engine.pairs.T
        d = pos[j] - pos[i]
        length = torch.linalg.vector_norm(d, dim=1).clamp_min(
            torch.finfo(pos.dtype).eps)
        unit = (d / length[:, None]).detach().cpu().numpy()
        f_bond = float(engine.config.bond_stiffness) * (
            length - engine.rest_lengths).detach().cpu().numpy()
        f_bond = np.where(bonded, f_bond, 0.0)
        overlap = 2.0 * float(engine.config.radius) - length.detach().cpu().numpy()
        # include the contact branch for *every* pair, not only the dead ones
        f_contact = float(engine.config.contact_stiffness) * np.clip(overlap, 0.0, None)
        force = ((f_bond - f_contact)[:, None] * unit)
    elif kind == "cell_volume_normalisation":
        outer = contact_outer(branch, force).sum(axis=0)
        V_p_legacy = LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR * PARTICLE_RADIUS ** 3
        return float(outer[2, 2] / (state["n_particles"] * V_p_legacy))
    elif kind == "transpose_convention":
        # f (x) l instead of l (x) f: only visible for non-central forces
        outer = np.einsum("ma,mb->mab", force, branch).sum(axis=0)
        return float(outer[2, 2] / V_total)
    else:
        raise ValueError(f"unknown mutation {kind}")
    return float(contact_outer(branch, force).sum(axis=0)[2, 2] / V_total)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, object] = {"node": "m16734", "PARTICLE_RADIUS": PARTICLE_RADIUS}

    # ---- C1/C2 on homogeneous tension and compression, layer convergence ----
    rows = []
    for n_layers in (3, 5, 9, 17):
        for strain in (+2.0e-3, -2.0e-3):
            engine, geo, _ = affine_state(n_layers, 4, strain)
            rec = measure(engine, geo)
            rec.update(n_layers=n_layers, strain=strain)
            rows.append(rec)
    out["homogeneous_rows"] = rows

    # ---- mutation non-vacuity on a state carrying *both* force branches -----
    engine, geo, _ = affine_state(5, 4, -2.0e-3, kill_fraction=0.25, seed=7)
    ref = measure(engine, geo)
    reference_value = float(ref["sigma_zz_exact_virial"])
    mutations = {}
    for kind in ("reference", "force_sign_flipped", "branch_reversed",
                 "pair_counted_twice", "dead_bonds_kept",
                 "contacts_not_gated", "cell_volume_normalisation",
                 "transpose_convention"):
        value = mutated_sigma_zz(engine, geo, kind=kind)
        mutations[kind] = {
            "sigma_zz_Pa": value,
            "rel_change_vs_reference": abs(value - reference_value)
            / abs(reference_value),
        }
    out["mutation_state"] = {
        "n_layers": 5, "strain": -2.0e-3, "kill_fraction": 0.25,
        "n_dead_bonds": ref["n_dead_bonds"],
        "n_contact_pairs": ref["n_contact_pairs"],
        "sigma_zz_plane_cut": ref["sigma_zz_plane_cut"],
        "sigma_zz_exact_virial": reference_value,
    }
    out["mutations"] = mutations

    # ---- AC5 invariance: the Hondros ratio does not depend on either fix ----
    engine, geo, _ = affine_state(5, 4, -2.0e-3, kill_fraction=0.25, seed=7)
    inv = {}
    for label, kw in (("legacy", dict(volume_per_particle=None, scatter="legacy")),
                      ("fixed", dict(volume_per_particle=None, scatter="half"))):
        rs = region_stress(engine, center=(0.0, 0.0, float(
            engine.positions[:, 2].mean())), radius=0.03, **kw)
        inv[label] = {
            "principal_in_plane": rs["principal_in_plane"],
            "ratio": rs["ratio_compressive_to_tensile"],
            "tensor_zz": rs["tensor_3x3"][2][2],
        }
    out["ac5_invariance"] = inv

    # ---- residual structure: rel err == |strain| (Cauchy vs nominal) --------
    sweep = []
    for eps in (1.0e-4, 5.0e-4, 1.0e-3, 2.0e-3):
        e_t, g_t, _ = affine_state(5, 4, +eps)
        r = measure(e_t, g_t)
        sweep.append({"strain": eps,
                      "rel_err_virial_vs_plane": r["rel_err_exact_virial_vs_plane"],
                      "rel_err_over_strain": r["rel_err_exact_virial_vs_plane"] / eps})
    out["strain_sweep"] = sweep

    # ---- AC5 consumer: cylinder specific volume of the disc lattice ---------
    from tensordem.bpm_calibration_feasibility import DiscGeometry, build_disc
    disc = DiscGeometry(n_lattice=7, n_layers=4)
    e_disc, _, _, m_disc = build_disc(REFERENCE_VECTOR, disc, mass_scaling=100.0,
                                     decay_steps=15)
    V_cyl = math.pi * disc.diameter_m ** 2 / 4.0 * disc.thickness_m
    V_p_disc = specific_volume(V_cyl, m_disc["n_particles"])
    V_p_cell = LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR * PARTICLE_RADIUS ** 3
    out["disc_specific_volume"] = {
        "n_particles": m_disc["n_particles"],
        "V_cylinder_m3": V_cyl,
        "V_p_cylinder_m3": V_p_disc,
        "V_p_cell_m3": V_p_cell,
        "cell_over_cylinder": V_p_cell / V_p_disc,
        "legacy_level_error": abs(V_p_cell / V_p_disc - 1.0),
    }

    # ---- figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
        ax = axes[0]
        names = [k for k in mutations if k != "reference"]
        vals = [mutations[k]["rel_change_vs_reference"] for k in names]
        short = [n.replace("_", "\n") for n in names]
        ax.bar(range(len(vals)), vals, color="#3b6ea5")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel("|delta sigma_zz| / sigma_zz  [-]")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title("(a) mutation non-vacuity (5x4 lattice)")
        ax = axes[1]
        ns = sorted({r["n_layers"] for r in rows})
        for label, key, style in (
                ("legacy (2x count, 8R$^3$)", "rel_err_legacy_vs_plane", "o-"),
                ("8R$^3$ normalisation only", "rel_err_per_particle_cell_vs_plane", "s-"),
                ("fixed (1x count, gauge V/N)", "rel_err_per_particle_gauge_vs_plane", "^-")):
            ys = [next(r[key] for r in rows if r["n_layers"] == n
                       and r["strain"] > 0) for n in ns]
            ax.plot(ns, ys, style, label=label)
        ax.plot(ns, [2.0 * (n - 1) / n - 1.0 for n in ns], "k--", lw=0.9,
                label="analytic 2(n-1)/n - 1")
        ax.set_xlabel("n_layers")
        ax.set_ylabel("|sigma_virial - sigma_plane| / |sigma_plane|")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.set_title("(b) absolute level vs the plane-cut traction")
        ax = axes[2]
        ax.plot([r["strain"] for r in sweep],
                [r["rel_err_virial_vs_plane"] for r in sweep], "o-",
                label="measured rel. error")
        ax.plot([r["strain"] for r in sweep], [r["strain"] for r in sweep], "k--",
                lw=0.9, label="|strain| (Cauchy vs nominal)")
        ax.set_xlabel("affine strain [-]")
        ax.set_ylabel("relative error [-]")
        ax.legend(fontsize=7)
        ax.set_title("(c) residual = strain (physical, not a bug)")
        fig.tight_layout()
        fig.savefig(OUT / "fig_virial_absolute.png", dpi=130)
        print("figure ->", OUT / "fig_virial_absolute.png")
    except Exception as exc:                     # pragma: no cover - plotting only
        print("figure skipped:", exc)
    out["wall_s"] = time.time() - t0

    (OUT / "verification.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("mutations", "mutation_state", "ac5_invariance",
                               "wall_s")}, indent=2))
    for r in rows:
        print(f"n={r['n_layers']:2d} eps={r['strain']:+.1e} "
              f"plane={r['sigma_zz_plane_cut']:.6e} "
              f"virial={r['sigma_zz_exact_virial']:.6e} "
              f"rel={r['rel_err_exact_virial_vs_plane']:.2e} "
              f"gauge_rel={r['rel_err_per_particle_gauge_vs_plane']:.2e} "
              f"cell_rel={r['rel_err_per_particle_cell_vs_plane']:.4f} "
              f"legacy_rel={r['rel_err_legacy_vs_plane']:.4f} "
              f"legacy_meas={r['measured_legacy_over_plane']:.4f} "
              f"legacy_pred={r['predicted_legacy_over_plane']:.4f}")


if __name__ == "__main__":
    main()