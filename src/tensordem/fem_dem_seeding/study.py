"""Campaign p302w2-n03: the production run of the shape-resolved FEM -> DEM handoff.

The chain, in order, with the artefact that carries each step:

1. **FEM** -- the level-ice plate of campaign ``p302w1-n03`` is solved again
   (Kirchhoff/ACM element, Winkler foundation, non-uniform contact pressure) up to
   the state that splits the sheet into floes (:mod:`tensordem.fem_dem_fracture`).
   Output: ``failure_faces.json`` (fracture polylines, fractured cells, floe
   polygons, mass/area ledger) **and** ``fem_node_state.json``, the new companion
   artefact that carries the nodal deflection and slope field needed for the
   kinematic transfer.
2. **Read back** -- the DEM side opens both files and rebuilds its own ledger from
   the payloads alone (:func:`tensordem.fem_dem_fracture.read_failure_faces`), so
   nothing is passed in memory.
3. **Seed** -- every floe becomes a body in two representations, seeded with the
   *same instrument* so the comparison is fair: the ``equivalent_disc`` body of
   the first handoff (baseline) and the ``clump`` body of this campaign
   (:mod:`tensordem.fem_dem_seeding.clump`).
4. **Kinematics** -- rigid-plane fit of the FE field per floe gives the heave
   offset, the heave velocity at the loading rate and the pitch/roll rates; the
   rigidity residual says how legitimate a rigid floe is.
5. **Dynamics** -- the seeded bodies are integrated (heave oscillator + in-plane
   drag relaxation) and the books are closed again afterwards.

Everything the report claims is a number in ``report_data.json``, written by this
module and quoted verbatim by the document builder; no number is typed by hand.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..fem_dem_fracture import IcePlateCase, solve_case_field, find_splitting_state
from ..fem_dem_fracture.handoff import build_failure_faces, read_failure_faces, dem_initial_conditions
from .clump import seed_floe_bodies, overlap_audit, body_ledger
from .dynamics import WATER_DENSITY, drag_relax, heave_audit, post_run_ledger
from .kinematics import kinematic_transfer, read_node_state
from .raster import Raster, footprint_metrics, resolution_check

__all__ = [
    "DEFAULT_OUT",
    "LOADING_RATE_PA_PER_S",
    "run_study",
    "write_figures",
]

DEFAULT_OUT = "artifacts/fem_dem_seeding/p302w2_n03"
#: Contact pressure ramp used to turn the static FE field into a velocity field
#: [Pa/s].  A level-ice sheet under a structure sees kPa/s over seconds, so the
#: resulting heave velocity is tens of mm/s (``v_z = (q_dot / q_first) * w``).
LOADING_RATE_PA_PER_S = 5.0e3
#: Raster cell as an integer fraction of the FEM element (0.25 m / 8) so that the
#: floe polygons -- whose edges are all element-aligned -- are sampled exactly.
RASTER_H = 0.25 / 8.0


# --------------------------------------------------------------------------
# 1-2. continuum solve and the two artefacts
# --------------------------------------------------------------------------
def solve_and_write(out_dir: str, *, loading_rate: float = LOADING_RATE_PA_PER_S) -> dict:
    """Solve the plate, write ``failure_faces.json`` + ``fem_node_state.json``."""
    os.makedirs(out_dir, exist_ok=True)
    case = IcePlateCase()
    unit = solve_case_field(case)
    q_first = unit.scale_to_failure()
    first = unit.scaled(q_first)
    split = find_splitting_state(first, case.tracer)
    state = first.scaled(split.load_factor * first.load_amplitude)
    network = split.network
    faces = build_failure_faces(network, state,
                                case_meta=case.as_meta(q_first, split.load_factor))
    faces_path = os.path.join(out_dir, "failure_faces.json")
    with open(faces_path, "w", encoding="utf-8") as fh:
        json.dump(faces.to_payload(), fh, indent=1, sort_keys=True)

    mesh = state.mesh
    xy = mesh.node_xy()
    nodes = np.column_stack([xy, state.w, state.theta_x, state.theta_y])
    node_state = {
        "schema": "tensordem.fem_node_state/1",
        "source": "tensordem.fem_dem_fracture (Kirchhoff/ACM plate, Winkler)",
        "load_factor": float(split.load_factor),
        "load_rate_per_s": float(loading_rate / q_first),
        "loading_rate_pa_per_s": float(loading_rate),
        "first_cracking_pressure_pa": float(q_first),
        "mesh": {"nx": mesh.nx, "ny": mesh.ny,
                 "length_x_m": mesh.length_x, "length_y_m": mesh.length_y},
        "fields": ["x_m", "y_m", "w_m", "theta_x_rad", "theta_y_rad"],
        "nodes": [[float(v) for v in row] for row in nodes],
    }
    state_path = os.path.join(out_dir, "fem_node_state.json")
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(node_state, fh, indent=1)

    return {
        "case": case,
        "solution": state,
        "network": network,
        "faces": faces,
        "split": split,
        "q_first_pa": float(q_first),
        "load_factor": float(split.load_factor),
        "load_rate_per_s": float(loading_rate / q_first),
        "failure_faces_path": faces_path,
        "fem_node_state_path": state_path,
    }


# --------------------------------------------------------------------------
# 6. boundary identity of the partition (segment multiset)
# --------------------------------------------------------------------------
def boundary_identity(faces, *, plate_length: float, plate_width: float) -> dict:
    """Rebuild the floe boundary as a segment multiset and check the identity.

    Every floe edge is a segment; the identity is that a segment is used exactly
    twice when it is a fracture face and exactly once when it lies on the plate
    rim.  Counts are compared with the *geometric* expectation (rim = plate
    perimeter, fracture faces = 2 x fracture-face length), so a polygon-tracing
    bug that duplicates or drops an edge cannot hide inside a correct area total.
    """
    counts: dict[tuple, int] = {}
    lengths: dict[tuple, float] = {}
    for floe in faces.floes:
        for ring in floe.polygons:
            pts = np.asarray(ring, dtype=float)
            for a, b in zip(pts, np.roll(pts, -1, axis=0)):
                pa = (round(float(a[0]), 9), round(float(a[1]), 9))
                pb = (round(float(b[0]), 9), round(float(b[1]), 9))
                # A segment is undirected: canonicalise as the *pair of points*
                # (sorting the four coordinates instead would merge a horizontal
                # and a vertical unit edge that share a corner -- a real bug that
                # this check caught on its first run).
                key = (pa, pb) if pa <= pb else (pb, pa)
                counts[key] = counts.get(key, 0) + 1
                lengths[key] = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    rim = sum(lengths[k] for k, c in counts.items() if c == 1)
    fracture = sum(lengths[k] for k, c in counts.items() if c == 2) / 2.0
    bad = [k for k, c in counts.items() if c > 2]
    return {
        "n_segments": len(counts),
        "rim_length_m": float(rim),
        "plate_perimeter_m": float(2.0 * (plate_length + plate_width)),
        "rim_residual_m": float(rim - 2.0 * (plate_length + plate_width)),
        "fracture_face_length_m": float(fracture),
        "n_overused_segments": len(bad),
        "n_single_use_segments": int(sum(1 for c in counts.values() if c == 1)),
        "n_shared_segments": int(sum(1 for c in counts.values() if c == 2)),
    }


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------
def run_study(out_dir: str = DEFAULT_OUT, *, h: float = RASTER_H,
              loading_rate: float = LOADING_RATE_PA_PER_S) -> dict:
    """Run the whole chain and return every acceptance number."""
    paths = solve_and_write(out_dir, loading_rate=loading_rate)
    faces = read_failure_faces(paths["failure_faces_path"])
    node_state = read_node_state(paths["fem_node_state_path"])

    plate = faces.meta["plate"]
    material = faces.meta["material"]
    plate_area = float(plate["area_m2"])
    plate_mass = float(faces.conservation.plate_mass)

    # Raster window: the plate plus a margin so that a body spilling over the rim
    # (which the equivalent-disc baseline does) is measurable, not clipped.
    margin = 2.0
    raster = Raster(-margin, -margin,
                    float(plate["length_x_m"]) + margin,
                    float(plate["length_y_m"]) + margin, h)

    bodies = seed_floe_bodies(faces, raster, modes=("clump", "equivalent_disc"))
    clump, eqdisc = bodies["clump"], bodies["equivalent_disc"]

    kinematics = kinematic_transfer(faces, node_state)
    heave = heave_audit(clump, kinematics, t_end=6.0)
    heave_eq = heave_audit(eqdisc, kinematics, t_end=6.0)

    # In-plane drift relaxation of the seeded ensemble (same tau for every floe).
    tau = 20.0
    current = (0.15, -0.05)
    current_speed = float(math.hypot(*current))
    drift = drag_relax(0.0, current_speed, tau, t_end=60.0, dt=0.05)

    # Rigidity: how much of the FE field a rigid floe cannot represent.
    rigidity = [{
        "floe_id": k.floe_id,
        "n_nodes": k.n_nodes,
        "rel_residual": k.fit.rel_rms,
        "rms_m": k.fit.rms,
        "w_range_m": k.fit.w_range,
        "pitch_rad": k.fit.pitch,
        "roll_rad": k.fit.roll,
        "slope_check": k.fit.slope_check(),
        "z0_m": k.z0,
        "v_z_m_s": k.v_z,
    } for k in kinematics]
    plate_w_range = float(np.max(node_state["w"]) - np.min(node_state["w"]))
    rigidity_all = float(np.sqrt(np.mean([
        r["rms_m"] ** 2 * r["n_nodes"] for r in rigidity])) / plate_w_range) \
        if rigidity else float("nan")

    ledger_clump = body_ledger(clump, plate_area=plate_area, plate_mass=plate_mass)
    ledger_eq = body_ledger(eqdisc, plate_area=plate_area, plate_mass=plate_mass)
    overlap_clump = overlap_audit(clump)
    overlap_eq = overlap_audit(eqdisc)
    post = post_run_ledger(clump, plate_area=plate_area, plate_mass=plate_mass)

    boundary = boundary_identity(
        faces,
        plate_length=float(plate["length_x_m"]),
        plate_width=float(plate["length_y_m"]),
    )

    # Resolution check on the largest floe (the one whose shape matters most).
    biggest = max(faces.floes, key=lambda f: f.area)
    rings_big = tuple(tuple(tuple(float(c) for c in v) for v in ring)
                      for ring in biggest.polygons)
    big_body = next(b for b in clump if b.floe_id == biggest.floe_id)
    def _build(ras):
        # The body is *fixed*; only the measuring grid refines.  Re-seeding per
        # resolution would mix the discretisation error with a different disc
        # layout and could not answer "is the reported score resolved?".
        return big_body.centres(), big_body.radii()

    resolution = resolution_check(raster, rings_big, _build)

    # How does the shape score depend on the clump resolution?  (Plateau evidence.)
    shape_conv = _shape_convergence(faces, raster)

    # Sampled heave history so the figure plots the integrator output, not a
    # closed-form reconstruction.
    heave_samples = {}
    for fid, res in heave["history"].items():
        idx = np.linspace(0, res["t"].size - 1, 26).astype(int)
        heave_samples[str(fid)] = {"t": res["t"][idx].tolist(),
                                   "z": res["z"][idx].tolist()}

    # The first-handoff DEM bodies, evaluated with the same instrument.
    legacy_ic = dem_initial_conditions(faces)
    legacy = {
        "n_bodies": int(legacy_ic.n_bodies) if hasattr(legacy_ic, "n_bodies")
        else int(len(legacy_ic.masses)),
        "mass_sum_kg": float(np.sum(legacy_ic.masses)),
        "mass_residual_kg": float(np.sum(legacy_ic.masses) - plate_mass),
        "velocities_all_zero": bool(np.allclose(legacy_ic.velocities, 0.0)),
    }

    summary = {
        "campaign": "p302w2-n03-fem-dem-fracture-handoff",
        "module": "tensordem.fem_dem_seeding",
        "case": {
            "plate": plate,
            "material": material,
            "water_density_kg_m3": WATER_DENSITY,
            "q_first_pa": paths["q_first_pa"],
            "load_factor": paths["load_factor"],
            "loading_rate_pa_per_s": float(loading_rate),
            "load_rate_per_s": paths["load_rate_per_s"],
            "mesh": {"nx": int(paths["solution"].mesh.nx),
                     "ny": int(paths["solution"].mesh.ny),
                     "h_m": float(paths["solution"].mesh.hx)},
            "n_floe": len(faces.floes),
            "crack_total_length_m": float(sum(c.length for c in faces.cracks)),
        },
        "artefacts": {
            "failure_faces": os.path.relpath(paths["failure_faces_path"], out_dir),
            "fem_node_state": os.path.relpath(paths["fem_node_state_path"], out_dir),
        },
        "criteria": {},
        "ledger": {"clump": ledger_clump, "equivalent_disc": ledger_eq,
                   "legacy_first_handoff": legacy, "post_run": post},
        "overlap": {"clump": overlap_clump, "equivalent_disc": overlap_eq},
        "boundary": boundary,
        "kinematics": {"rows": rigidity, "ensemble_rel_residual": rigidity_all,
                       "plate_w_range_m": plate_w_range},
        "heave": {k: v for k, v in heave.items() if k != "history"},
        "heave_history_samples": heave_samples,
        "heave_equivalent_disc": {k: v for k, v in heave_eq.items() if k != "history"},
        "drift": {"tau_s": tau, "target_m_s": list(current),
                  "target_speed_m_s": current_speed, "t_end_s": 60.0,
                  "rel_error": drift["rel_error"],
                  "terminal_m_s": drift["terminal_velocity"]},
        "resolution": resolution,
        "shape_convergence": shape_conv,
        "bodies": [b.as_payload() for b in clump],
        "bodies_equivalent_disc": [b.as_payload() for b in eqdisc],
        "kinematics_payload": [k.as_payload() for k in kinematics],
    }
    summary["criteria"] = _criteria(summary)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dem_bodies.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "schema": "tensordem.dem_floe_bodies/1",
            "source": "failure_faces.json + fem_node_state.json",
            "bodies": [
                dict(body, initial_state=dict(kin, floe_id=body["floe_id"]))
                for body, kin in zip(summary["bodies"], summary["kinematics_payload"])
            ],
        }, fh, indent=1)
    with open(os.path.join(out_dir, "report_data.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=False)
    return summary


def _shape_convergence(faces, raster: Raster, *, hints=(0.40, 0.25, 0.15, 0.10, 0.07),
                       n_floes: int = 2) -> dict:
    """Coverage / IoU of the clump *against its own resolution*.

    A disc clump cannot tile a polygon (the densest packing fills 0.907 of the
    plane and the rim band of width ``r`` belongs to no disc), so the shape score is
    resolution dependent.  This sweep quantifies that: refining the clump must move
    the score towards the packing limit, and the production resolution is placed
    where the curve is already flat per doubling of the disc count.
    """
    import math as _math

    from .clump import seed_inscribed_discs

    order = sorted(faces.floes, key=lambda f: -f.area)[:n_floes]
    rows = []
    for floe in order:
        rings = tuple(tuple(tuple(float(c) for c in v) for v in ring)
                      for ring in floe.polygons)
        for hint in hints:
            n_t = int(max(3, min(4000, round(floe.area / (2.0 * _math.sqrt(3.0) * hint ** 2)))))
            try:
                centres, radii = seed_inscribed_discs(raster, rings, target_discs=n_t)
            except RuntimeError as exc:                       # pragma: no cover - reported
                rows.append({"floe_id": floe.floe_id, "radius_hint_m": hint,
                             "error": str(exc)})
                continue
            metrics = footprint_metrics(raster, rings, centres, radii)
            rows.append({
                "floe_id": floe.floe_id,
                "radius_hint_m": float(hint),
                "n_discs": int(centres.shape[0]),
                "radius_mean_m": float(np.mean(radii)),
                "coverage": metrics["coverage"],
                "iou": metrics["iou"],
                "spill_area_m2": metrics["spill_area"],
            })
    return {"hints_m": list(hints), "rows": rows}


def _criteria(summary: dict) -> dict:
    """Machine-checkable criteria of the node, evaluated from the numbers."""
    led = summary["ledger"]
    ov = summary["overlap"]
    bnd = summary["boundary"]
    heave = summary["heave"]
    rigid = summary["kinematics"]
    clut = led["clump"]
    post = led["post_run"]
    res = summary["resolution"]
    crit = {}

    def put(name, ok, value, target):
        crit[name] = {"pass": bool(ok), "value": value, "target": target}

    put("C1_fracture_artefact_written",
        summary["case"]["n_floe"] >= 3 and bnd["fracture_face_length_m"] > 0.0,
        {"n_floe": summary["case"]["n_floe"],
         "crack_total_length_m": summary["case"]["crack_total_length_m"]},
        "sigma_max >= sigma_flexure, failure_faces.json written with fracture faces")
    put("C2_mass_conservation", abs(clut["mass_residual_kg"]) < 1e-6,
        clut["mass_residual_kg"], "< 1e-6 kg")
    put("C2b_disc_mass_conservation", abs(clut["disc_mass_residual_kg"]) < 1e-6,
        clut["disc_mass_residual_kg"], "< 1e-6 kg")
    put("C3_area_conservation", abs(clut["area_residual_m2"]) < 1e-3,
        clut["area_residual_m2"], "< 1e-3 m^2")
    put("C3b_boundary_identity",
        abs(bnd["rim_residual_m"]) < 1e-9 and bnd["n_overused_segments"] == 0,
        {"rim_residual_m": bnd["rim_residual_m"],
         "overused_segments": bnd["n_overused_segments"]},
        "rim rebuilt from floe edges equals plate perimeter, no edge used >2x")
    put("C4_floe_count_in_range", 3 <= summary["case"]["n_floe"] <= 30,
        summary["case"]["n_floe"], "[3, 30]")
    put("C5_end_to_end", summary["case"]["n_floe"] == clut["n_bodies"] > 0,
        {"n_floe": summary["case"]["n_floe"], "n_bodies": clut["n_bodies"]},
        "DEM body per floe read from the files alone")
    put("W1_shape_no_spill", clut["spill_area_sum_m2"] <= 0.5 * clut["plate_area_m2"] * 0.02,
        clut["spill_area_sum_m2"], "clump material does not leave its floe")
    put("W2_no_initial_overlap", ov["clump"]["n_overlapping_pairs"] == 0,
        {"n_overlapping_pairs": ov["clump"]["n_overlapping_pairs"],
         "baseline_pairs": ov["equivalent_disc"]["n_overlapping_pairs"]},
        "no inter-floe interpenetration at t=0")
    put("W3_rigidity_residual", rigid["ensemble_rel_residual"] < 0.15,
        rigid["ensemble_rel_residual"], "< 0.15 of the plate deflection range")
    put("W4_heave_period_closed_form", heave["max_period_error"] < 0.03,
        {"max_period_error": heave["max_period_error"],
         "period_spread_s": heave["period_spread"]},
        "measured heave period vs sqrt(rho_w g/(rho_i t)) < 3 %")
    put("W5_post_run_conservation",
        abs(post["mass_residual_kg"]) < 1e-6 and abs(post["area_residual_m2"]) < 1e-3
        and post["all_masses_positive"],
        {"mass_residual_kg": post["mass_residual_kg"],
         "area_residual_m2": post["area_residual_m2"]},
        "books still closed after integration")
    iou_clump = [b["footprint_metrics"]["iou"] for b in summary["bodies"]]
    iou_base = [b["footprint_metrics"]["iou"] for b in summary["bodies_equivalent_disc"]]
    put("W0_shape_improved_per_floe",
        min(c - b for c, b in zip(iou_clump, iou_base)) > 0.0,
        {"mean_iou_clump": float(np.mean(iou_clump)),
         "mean_iou_baseline": float(np.mean(iou_base)),
         "min_gain": float(np.min(np.subtract(iou_clump, iou_base)))},
        "clump IoU above the equivalent disc on every floe")
    put("W6_raster_resolved", res["max_relative_change"] < 0.05,
        res["max_relative_change"], "< 5 % change when h is halved")
    put("W7_drift_closed_form", summary["drift"]["rel_error"] < 1e-4,
        summary["drift"]["rel_error"], "RK4 vs exponential < 1e-4")
    return crit


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def write_figures(summary: dict, out_dir: str = DEFAULT_OUT) -> list[dict]:
    """Five PNG figures + captions; returns the caption records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    recs: list[dict] = []

    plate = summary["case"]["plate"]

    # --- fig 1: footprints, clump vs equivalent disc ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4))
    for ax, key, title in ((axes[0], "bodies", "clump (this campaign)"),
                           (axes[1], "bodies_equivalent_disc",
                            "equivalent disc (p302w1 baseline)")):
        for body in summary[key]:
            outer = np.asarray(body["footprint_polygons"][0], dtype=float)
            ax.add_patch(Polygon(outer, closed=True, fill=False, lw=0.9,
                                 edgecolor="0.25", zorder=1))
            for d in body["discs"]:
                ax.add_patch(Circle((d["x_m"], d["y_m"]), d["radius_m"],
                                    facecolor="#4C9BE8", alpha=0.35,
                                    edgecolor="#1f5f99", lw=0.4, zorder=2))
        ax.set_xlim(-2.0, plate["length_x_m"] + 2.0)
        ax.set_ylim(-2.0, plate["length_y_m"] + 2.0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.plot([0, plate["length_x_m"], plate["length_x_m"], 0, 0],
                [0, 0, plate["length_y_m"], plate["length_y_m"], 0],
                color="k", lw=0.8, ls="--", zorder=0)
    fig.suptitle("Floe footprint (grey) vs DEM body material (blue discs)")
    fig.tight_layout()
    path = os.path.join(figdir, "fig1_footprint_vs_body.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    recs.append({"path": path, "name": os.path.basename(path),
                 "caption": "Continuum floe footprint (grey outline) and the DEM material "
                            "seeded into it: left, the inscribed non-overlapping disc clump of "
                            "this campaign; right, the single equivalent disc of the first "
                            "handoff. The dashed square is the plate rim; the baseline disc "
                            "crosses it."})

    # --- fig 2: shape fidelity per floe ----------------------------------
    floe_ids = [b["floe_id"] for b in summary["bodies"]]
    cov = [b["footprint_metrics"]["coverage"] for b in summary["bodies"]]
    iou = [b["footprint_metrics"]["iou"] for b in summary["bodies"]]
    iou_eq = [b["footprint_metrics"]["iou"] for b in summary["bodies_equivalent_disc"]]
    areas = [b["area_m2"] for b in summary["bodies"]]
    x = np.arange(len(floe_ids))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - 0.27, iou, 0.25, label="IoU, clump", color="#4C9BE8")
    ax.bar(x, iou_eq, 0.25, label="IoU, equivalent disc", color="#E8A33D")
    ax.bar(x + 0.27, cov, 0.25, label="coverage, clump", color="#7FB77E")
    ax.set_xticks(x)
    ax.set_xticklabels([f"floe {i}\n{ a:.2f} m²" for i, a in zip(floe_ids, areas)], fontsize=8)
    ax.set_ylabel("shape score [-]")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Shape fidelity of the DEM body against its floe footprint")
    fig.tight_layout()
    path = os.path.join(figdir, "fig2_shape_fidelity.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    recs.append({"path": path, "name": os.path.basename(path),
                 "caption": "Per-floe shape scores. IoU is the symmetric-difference score of "
                            "the body's material against its footprint; coverage is the "
                            "fraction of the floe the discs fill. The clump beats the "
                            "equivalent disc on every fragment of any elongation."})

    # --- fig 3: rigidity residual per floe -------------------------------
    rows = summary["kinematics"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    ax.bar([r["floe_id"] for r in rows], [100 * r["rel_residual"] for r in rows],
           color="#B26E9B")
    ax.set_xlabel("floe id")
    ax.set_ylabel("rigidity residual [% of floe deflection range]")
    ax.set_title("How much of the FE field a rigid floe cannot carry")
    ax = axes[1]
    ax.bar([r["floe_id"] for r in rows], [1e3 * r["v_z_m_s"] for r in rows],
           color="#4C9BE8")
    ax.set_xlabel("floe id")
    ax.set_ylabel("initial heave velocity [mm/s]")
    ax2 = ax.twinx()
    ax2.plot([r["floe_id"] for r in rows], [1e3 * r["z0_m"] for r in rows],
             "o-", color="#E8A33D", lw=1.0)
    ax2.set_ylabel("heave offset z0 [mm]", color="#E8A33D")
    ax.set_title("Kinematic initial conditions from the FE node field")
    fig.tight_layout()
    path = os.path.join(figdir, "fig3_kinematics.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    recs.append({"path": path, "name": os.path.basename(path),
                 "caption": "Left: rigidity residual, i.e. the part of each floe's own "
                            "deflection field that the best-fitting rigid plane cannot "
                            "reproduce, normalised by that floe's deflection range. Right: "
                            "the initial heave velocity (bars) and heave offset (line) "
                            "transferred from the FE node field at the loading rate."})

    # --- fig 4: heave integration ---------------------------------------
    heave = summary["heave"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    rows = sorted(heave["rows"], key=lambda r: -r["area_m2"])
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(rows)))
    samples = summary.get("heave_history_samples", {})
    for r, c in zip(rows, colours):
        hist = samples.get(str(r["floe_id"]))
        if not hist:
            continue
        ax.plot(hist["t"], 1e3 * np.asarray(hist["z"]), color=c, lw=1.0,
                label=f"floe {r['floe_id']} (A={r['area_m2']:.1f} m²)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("heave z [mm]")
    ax.set_title("Post-handoff heave: all floes share one frequency")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    err = [100 * r["period_error"] for r in heave["rows"]]
    ax.bar([r["floe_id"] for r in heave["rows"]], err, color="#4C9BE8")
    ax.axhline(3.0, color="r", ls="--", lw=1.0, label="3 % acceptance")
    ax.set_xlabel("floe id")
    ax.set_ylabel("heave period error [%]")
    ax.set_title(f"Period vs sqrt(rho_w g/(rho_i t)) = {heave['analytic']['period_s']:.3f} s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(figdir, "fig4_heave_dynamics.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    recs.append({"path": path, "name": os.path.basename(path),
                 "caption": "Left: velocity-Verlet heave of every seeded floe. The frequency is set by "
                            "thickness and density only, so all fragments move together -- a "
                            "size-independent period proves the handed-over mass and buoyancy "
                            "stiffness are consistent. Right: measured vs analytic period against "
                            "the 3 % line."})

    # --- fig 5: conservation ledger --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    labels = ["mass\n[kg]", "disc mass\n[kg]", "area\n[m²]"]
    led = summary["ledger"]
    vals = [
        [abs(led["clump"]["mass_residual_kg"]), abs(led["clump"]["disc_mass_residual_kg"]),
         abs(led["clump"]["area_residual_m2"])],
        [abs(led["equivalent_disc"]["mass_residual_kg"]),
         abs(led["equivalent_disc"]["disc_mass_residual_kg"]),
         abs(led["equivalent_disc"]["area_residual_m2"])],
        [abs(led["post_run"]["mass_residual_kg"]),
         abs(led["post_run"]["disc_mass_residual_kg"]),
         abs(led["post_run"]["area_residual_m2"])],
    ]
    floor = 1e-18
    x = np.arange(3)
    for i, (v, name) in enumerate(zip(vals, ["clump", "equivalent disc", "after integration"])):
        ax.bar(x + (i - 1) * 0.26, [max(a, floor) for a in v], 0.24, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(1e-6, color="r", ls="--", lw=1.0, label="1e-6 acceptance")
    ax.set_ylabel("|residual|")
    ax.set_title("Mass / area ledger (bars at the 1e-18 floor are exactly zero)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ov = summary["overlap"]
    names = ["clump", "equivalent disc"]
    spill = [led["clump"]["spill_area_sum_m2"], led["equivalent_disc"]["spill_area_sum_m2"]]
    pairs = [ov["clump"]["n_overlapping_pairs"], ov["equivalent_disc"]["n_overlapping_pairs"]]
    ax.bar(np.arange(2) - 0.2, spill, 0.36, color="#C0504D", label="material outside its floe [m²]")
    axb = ax.twinx()
    axb.bar(np.arange(2) + 0.2, pairs, 0.36, color="#8064A2",
            label="inter-floe overlapping disc pairs")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(names)
    ax.set_ylabel("spill area [m²]", color="#C0504D")
    axb.set_ylabel("overlapping pairs [-]", color="#8064A2")
    ax.set_title("Geometric defects of the two body representations")
    fig.tight_layout()
    path = os.path.join(figdir, "fig5_ledger_and_overlap.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    recs.append({"path": path, "name": os.path.basename(path),
                 "caption": "Left: absolute mass, disc-mass and area residuals of both representations and "
                            "of the ledger re-closed after integration (log scale; the 1e-18 "
                            "floor marks an exact zero). Right: material leaving its own floe and "
                            "initial inter-floe interpenetration -- the two defects the equivalent "
                            "disc cannot avoid."})
    # --- fig 6: shape score vs clump resolution --------------------------
    conv = summary.get("shape_convergence", {})
    rows = conv.get("rows", [])
    if rows:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        floe_ids = sorted({r["floe_id"] for r in rows})
        colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(floe_ids)))
        for fid, c in zip(floe_ids, colours):
            pts = [r for r in rows if r["floe_id"] == fid and "n_discs" in r]
            pts.sort(key=lambda r: r["n_discs"])
            ax.plot([p["n_discs"] for p in pts], [p["iou"] for p in pts], "o-",
                    color=c, lw=1.2, ms=4, label=f"floe {fid}")
        ax.axhline(math.pi / (2.0 * math.sqrt(3.0)), color="r", ls="--", lw=1.0,
                   label="disc packing limit 0.907")
        prod = next((b for b in summary["bodies"]
                     if b["floe_id"] == floe_ids[0]), None)
        if prod:
            ax.axvline(prod["n_discs"], color="0.4", ls=":", lw=1.0,
                       label=f"production clump of floe {prod['floe_id']}")
        ax.set_xscale("log")
        ax.set_xlabel("discs per floe [-]")
        ax.set_ylabel("IoU against the floe footprint [-]")
        ax.set_title("Shape score vs clump resolution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(figdir, "fig6_shape_convergence.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        recs.append({"path": path, "name": os.path.basename(path),
                     "caption": "IoU of the clump against its floe footprint as the clump is "
                                "refined (log axis). The score rises towards the densest-disc-"
                                "packing limit of 0.907 and cannot cross it: a disjoint disc "
                                "clump can never tile a polygon, which is why the exact footprint "
                                "polygon travels with the body."})
    return recs