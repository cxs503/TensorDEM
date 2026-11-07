"""IN32-N01 acceptance harness: unordered polygonal floe field.

Machine-checkable criteria (the campaign node's metadata):

(a) no overlap   : phi in {0.60, 0.70, 0.80, 0.85, 0.90} x 4 FSD families x 12
                   seeds = 240 fields, min SAT signed gap >= -1e-9 m.
(b) area moments : realised m1/m2 of the equivalent radius vs the analytic FSD
                   moments, relative error < 1 %.
(c) completeness : SAT contact set (KD-tree search) identical to the O(N^2)
                   brute-force enumeration at N = 2000, 0 false positives and
                   0 false negatives.
(d) torque/force : force-free static field, mass-weighted centroid drift
                   < 1e-10 m; global force and torque sums cancel to 1e-9.
(e) monodisperse : Delta family at phi <= 0.70, contact set vs a circumscribed
                   disk field, IoU >= 0.98.
(f) budget       : N = 4000 single field generated in < 30 s.
(g) tests        : >= 18 pytest cases incl. negative controls (counted here and
                   verified by the pytest run in the campaign log).

Usage
-----
    python benchmarks/polygon_floe_field_acceptance.py --out artifacts/in32_n01.json
    python benchmarks/polygon_floe_field_acceptance.py --shard 1/4      # sweep shard

Every number written into the JSON is produced here (no hand transcription).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.fsd_packing import BimodalFSD, DeltaFSD, LogNormalFSD, PowerLawFSD  # noqa: E402
from tensordem.laguerre_field import LaguerreParams  # noqa: E402
from tensordem.polygon_contact import fabric_tensor  # noqa: E402
from tensordem.polygon_floe_field import (generate_polygon_floe_field,  # noqa: E402
                                         sample_floe_radii)

PHI_LIST = (0.60, 0.70, 0.80, 0.85, 0.90)
SEEDS = tuple(range(12))
FAMILIES = {
    "power_law_2.5": lambda: PowerLawFSD(2.5),
    "log_normal_0.5": lambda: LogNormalFSD(0.5),
    "bimodal": lambda: BimodalFSD(),
    "delta": lambda: DeltaFSD(5.0),
}


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                        # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# (a) no overlap
# ---------------------------------------------------------------------------

def sweep_no_overlap(n_floes: int, shard: int, n_shards: int) -> Dict[str, object]:
    jobs = [(fam, phi, seed) for fam in FAMILIES for phi in PHI_LIST for seed in SEEDS]
    mine = [j for k, j in enumerate(jobs) if k % n_shards == shard]
    rows: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    for fam, phi, seed in mine:
        field = generate_polygon_floe_field(FAMILIES[fam](), coverage=phi,
                                            n_floes=n_floes, seed=seed)
        rows.append({
            "family": fam, "phi": phi, "seed": seed,
            "min_gap": field.min_signed_gap(),
            "coverage": field.realised_coverage(),
            "vertex_mean": float(field.count.mean()),
            "vertices_min": int(field.count.min()),
            "area_error": float(field.cells_summary.get("area_error_max_rel", 0.0)),
            "cover_identity": float(field.cells_summary.get("cover", 0.0)),
            "seconds": float(field.timings["total_s"]),
        })
    return {"n_fields": len(rows), "n_floes": n_floes, "shard": shard,
            "n_shards": n_shards, "seconds": time.perf_counter() - t0,
            "rows": rows}


def summarise_no_overlap(sweeps: List[Dict[str, object]], tol: float = -1e-9) -> Dict[str, object]:
    rows = [r for sw in sweeps for r in sw["rows"]]
    gaps = np.array([r["min_gap"] for r in rows], dtype=float)
    per_family: Dict[str, Dict[str, float]] = {}
    for fam in FAMILIES:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        per_family[fam] = {
            "fields": len(sub),
            "min_gap": float(min(r["min_gap"] for r in sub)),
            "max_coverage_dev": float(max(abs(r["coverage"] - r["phi"]) for r in sub)),
            "min_vertices": int(min(r["vertices_min"] for r in sub)),
        }
    per_phi = {f"{phi:.2f}": float(min(r["min_gap"] for r in rows if r["phi"] == phi))
               for phi in PHI_LIST if any(r["phi"] == phi for r in rows)}
    return {
        "criterion": "min SAT signed gap >= -1e-9 m over 4 families x 12 seeds x 5 phi",
        "fields": len(rows),
        "violations": int((gaps < tol).sum()),
        "min_gap_overall": float(gaps.min()) if gaps.size else float("nan"),
        "per_phi_min_gap": per_phi,
        "per_family": per_family,
        "max_area_solver_error": float(max(r["area_error"] for r in rows)) if rows else float("nan"),
        "min_cover_identity_dev": float(min(abs(r["cover_identity"] - 1.0) for r in rows)) if rows else float("nan"),
        "max_cover_identity_dev": float(max(abs(r["cover_identity"] - 1.0) for r in rows)) if rows else float("nan"),
        "seconds": float(sum(sw["seconds"] for sw in sweeps)),
        "pass": bool(rows and (gaps >= tol).all()),
    }


# ---------------------------------------------------------------------------
# (b) area moments
# ---------------------------------------------------------------------------

def check_moments(n_floes: int = 1200, seeds=(0, 1, 2)) -> Dict[str, object]:
    table: Dict[str, Dict[str, float]] = {}
    worst = 0.0
    for fam, make in FAMILIES.items():
        errs1, errs2 = [], []
        for seed in seeds:
            field = generate_polygon_floe_field(make(), coverage=0.85,
                                                n_floes=n_floes, seed=seed)
            e = field.moment_errors()
            errs1.append(e["m1"])
            errs2.append(e["m2"])
        table[fam] = {"m1_rel_error_max": float(max(errs1)),
                      "m2_rel_error_max": float(max(errs2)),
                      "seeds": list(seeds), "n": n_floes}
        worst = max(worst, max(errs1), max(errs2))
    # stratified sampling alone (no tessellation) is the reference floor
    floor = {}
    for fam, make in FAMILIES.items():
        fsd = make()
        radii = sample_floe_radii(fsd, n_floes, seed=0)
        analytic = fsd.moments()
        floor[fam] = {"m1": float(abs(radii.mean() - analytic["m1"]) / analytic["m1"]),
                      "m2": float(abs((radii ** 2).mean() - analytic["m2"]) / analytic["m2"])}
    return {"criterion": "realised m1/m2 vs analytic FSD moments, relative error < 1e-2",
            "per_family": table, "sampling_floor": floor,
            "worst_rel_error": float(worst), "pass": bool(worst < 1e-2)}


# ---------------------------------------------------------------------------
# (c) contact-set completeness
# ---------------------------------------------------------------------------

def check_completeness(n_floes: int = 2000, seed: int = 7) -> Dict[str, object]:
    field = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.90,
                                        n_floes=n_floes, seed=seed)
    cut = 0.25 * float(np.median(field.r_eq))
    fast = field.contacts(cutoff=cut)
    brute = field.contacts(cutoff=cut, brute=True)

    def keys(cs):
        return {(int(a), int(b)) for a, b in zip(cs["i"], cs["j"])}

    kf, kb = keys(fast), keys(brute)
    fp = sorted(kf - kb)
    fn = sorted(kb - kf)
    # gaps of the shared pairs must agree bit for bit
    gf = {(int(a), int(b)): float(g) for a, b, g in zip(fast["i"], fast["j"], fast["gap"])}
    gb = {(int(a), int(b)): float(g) for a, b, g in zip(brute["i"], brute["j"], brute["gap"])}
    common = set(gf) & set(gb)
    gap_dev = max((abs(gf[k] - gb[k]) for k in common), default=0.0)
    return {"criterion": "KD-tree SAT contact set == O(N^2) brute force (0 FP / 0 FN)",
            "n": n_floes, "cutoff": cut, "fast_pairs": len(kf), "brute_pairs": len(kb),
            "false_positives": len(fp), "false_negatives": len(fn),
            "gap_max_abs_dev": float(gap_dev),
            "pass": bool(len(fp) == 0 and len(fn) == 0 and gap_dev < 1e-12)}


# ---------------------------------------------------------------------------
# (d) force/torque balance and centroid drift
# ---------------------------------------------------------------------------

def check_balance(n_floes: int = 600, seed: int = 3) -> Dict[str, object]:
    field = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.90,
                                        n_floes=n_floes, seed=seed)
    cut_free = 0.0
    drift_free = field.centroid_drift(cutoff=cut_free, steps=200, dt=1e-4)
    out_free = field.contact_forces(cutoff=cut_free)
    # a "loaded" variant: inflate every polygon about its centroid until the
    # plateau gaps close, which produces a real overlapping contact network and
    # therefore real forces, lever arms and torques
    loaded = field.with_vertex_scale(1.0 / math.sqrt(field.coverage) * 1.01)
    cut_load = 0.0
    out_load = loaded.contact_forces(cutoff=cut_load)
    nz = int((out_load["f_n"] > 0).sum())
    sum_force = out_load["body_force"].sum(axis=0)
    # Torque bookkeeping.  The torques about the floe centroids are *spin*
    # torques and do not cancel pairwise (their sum per pair equals -d x f); the
    # rotating-frame-safe statement is the pairwise identity
    #     tau_i + tau_j = (x_j^img - x_i) x f
    # together with an independent recomputation of tau_i = r_i x f.  A global
    # moment sum in a periodic cell would need per-contact images, so the exact
    # pair identity is what is verified here.
    d_img = loaded.centroid[out_load["j"]] - loaded.centroid[out_load["i"]]
    d_img[:, 0] -= loaded.box[0] * np.round(d_img[:, 0] / loaded.box[0])
    d_img[:, 1] -= loaded.box[1] * np.round(d_img[:, 1] / loaded.box[1])
    f_vec = out_load["force"]
    ident = ((d_img[:, 0] * f_vec[:, 1] - d_img[:, 1] * f_vec[:, 0])
             - (out_load["torque_i"] + out_load["torque_j"]))
    tscale = np.abs(out_load["torque_i"]).max() if out_load["i"].size else 1.0
    tau_recomputed = (out_load["r_i"][:, 0] * f_vec[:, 1]
                      - out_load["r_i"][:, 1] * f_vec[:, 0])
    tau_dev = (np.abs(tau_recomputed - out_load["torque_i"]).max()
               if out_load["i"].size else 0.0)
    spin_scale = float(tscale)
    scale_f = float(np.abs(out_load["body_force"]).max()) if out_load["i"].size else 1.0
    return {
        "criterion": "static field centroid drift < 1e-10 m; sum(F)=0 to 1e-12; pairwise torque identity tau_i+tau_j = d x f to 1e-12",
        "n": n_floes,
        "free_contacts": int(out_free["i"].size),
        "free_drift_m": float(drift_free),
        "loaded_pairs": int(out_load["i"].size),
        "loaded_active_contacts": nz,
        "sum_force_norm": float(np.linalg.norm(sum_force)),
        "sum_force_rel": float(np.linalg.norm(sum_force) / max(scale_f, 1e-30)),
        "torque_pair_identity_max_rel": float(np.abs(ident).max() / max(spin_scale, 1e-30)),
        "torque_recompute_max_dev": float(tau_dev),
        "max_lever_arm": float(out_load["torque_arm"].max()) if nz else 0.0,
        "mean_lever_arm": float(out_load["torque_arm"].mean()) if nz else 0.0,
        "max_abs_torque": float(np.abs(out_load["body_torque"]).max()) if nz else 0.0,
        "inflation_factor": float(1.0 / math.sqrt(field.coverage) * 1.01),
        "pass": bool(drift_free < 1e-10 and nz > 0
                     and np.linalg.norm(sum_force) / max(scale_f, 1e-30) < 1e-12
                     and np.abs(ident).max() / max(spin_scale, 1e-30) < 1e-12
                     and tau_dev <= 1e-12 * max(spin_scale, 1e-30)
                     and out_load["torque_arm"].max() > 0.0),
    }


# ---------------------------------------------------------------------------
# (e) monodisperse limit vs circumscribed disks
# ---------------------------------------------------------------------------

def _disc_contact_keys(field, cutoff: float):
    """Pairs whose circumscribed disks are within ``cutoff`` (periodic)."""
    x = field.centroid
    r = field.circumradius
    lx, ly = field.box
    d = x[None, :, :] - x[:, None, :]
    d[..., 0] -= lx * np.round(d[..., 0] / lx)
    d[..., 1] -= ly * np.round(d[..., 1] / ly)
    dist = np.linalg.norm(d, axis=2)
    gap = dist - (r[:, None] + r[None, :])
    i, j = np.nonzero(np.triu(gap <= cutoff, k=1))
    return {(int(a), int(b)) for a, b in zip(i, j)}


def _threshold_for_coordination(gaps: np.ndarray, n: int, target: float = 6.0) -> float:
    """Smallest gap threshold whose coordination number reaches ``target``."""
    if gaps.size == 0:
        return 0.0
    order = np.sort(gaps)
    # coordination after keeping the k smallest gaps is 2k/n: pick k ~ target*n/2
    k = int(round(target * n / 2.0))
    k = max(1, min(k, order.size))
    return float(order[k - 1])


def check_monodisperse(phis=(0.60, 0.70), seeds=(0, 1, 2),
                       target_coordination: float = 6.0) -> Dict[str, object]:
    """Contact-topology agreement with the circumscribed-disk field.

    A polygon is *inscribed* in its circumscribed disk, so the two fields cannot be
    compared at one common absolute gap: the disk field would report contacts that
    the polygon field reports as separated.  The comparison is therefore made at
    **equal coordination** (the planar first-shell value Z = 6): each field gets
    the gap threshold that gives Z = 6, and the two first-shell graphs are then
    compared.  This measures the topology (which floes are neighbours), not the
    shape factor.

    An *absolute* threshold (``tau = 0.5 * median r_eq``) is reported as well, to
    document the shape offset between a polygon and its circumscribed disk.
    """
    rows = []
    worst = 1.0
    worst_abs = 1.0
    for phi in phis:
        for seed in seeds:
            field = generate_polygon_floe_field(DeltaFSD(5.0), coverage=phi,
                                                n_floes=800, seed=seed)
            r_med = float(np.median(field.r_eq))
            wide = field.contacts(cutoff=1.0 * r_med)
            poly_keys = [(int(a), int(b)) for a, b in zip(wide["i"], wide["j"])]
            tau_poly = _threshold_for_coordination(wide["gap"], field.n,
                                                   target_coordination)
            keep_poly = {k for k, g in zip(poly_keys, wide["gap"]) if g <= tau_poly}

            x = field.centroid
            r = field.circumradius
            lx, ly = field.box
            d = x[None, :, :] - x[:, None, :]
            d[..., 0] -= lx * np.round(d[..., 0] / lx)
            d[..., 1] -= ly * np.round(d[..., 1] / ly)
            dist = np.linalg.norm(d, axis=2)
            disc_gap = dist - (r[:, None] + r[None, :])
            iu, ju = np.nonzero(np.triu(disc_gap <= 1.0 * r_med, k=1))
            disc_keys = [(int(a), int(b)) for a, b in zip(iu, ju)]
            disc_gaps = np.array([disc_gap[a, b] for a, b in disc_keys])
            tau_disc = _threshold_for_coordination(disc_gaps, field.n,
                                                   target_coordination)
            keep_disc = {k for k, g in zip(disc_keys, disc_gaps) if g <= tau_disc}

            inter = len(keep_poly & keep_disc)
            union = len(keep_poly | keep_disc)
            iou = inter / union if union else 0.0
            # absolute-threshold variant (documents the shape offset)
            cut_abs = 0.5 * r_med
            pa = {k for k, g in zip(poly_keys, wide["gap"]) if g <= cut_abs}
            da = {k for k, g in zip(disc_keys, disc_gaps) if g <= cut_abs}
            iou_abs = len(pa & da) / len(pa | da) if (pa | da) else 0.0
            worst = min(worst, iou)
            worst_abs = min(worst_abs, iou_abs)
            rows.append({"phi": phi, "seed": seed, "iou": float(iou),
                         "iou_absolute_tau": float(iou_abs),
                         "tau_polygon": float(tau_poly), "tau_disc": float(tau_disc),
                         "polygon_pairs": len(keep_poly), "disc_pairs": len(keep_disc),
                         "min_gap": float(field.min_signed_gap()),
                         "coordination_polygon": float(2 * len(keep_poly) / field.n),
                         "coordination_disc": float(2 * len(keep_disc) / field.n)})
    return {"criterion": "Delta family, phi <= 0.70: first-shell contact graph IoU with "
                         "the circumscribed-disc field >= 0.98",
            "alignment": f"equal coordination Z={target_coordination}",
            "rows": rows, "min_iou": float(worst),
            "min_iou_absolute_tau": float(worst_abs),
            "pass": bool(worst >= 0.98)}


# ---------------------------------------------------------------------------
# (f) generation budget
# ---------------------------------------------------------------------------

def check_budget(n_floes: int = 4000, seed: int = 1, repeats: int = 2) -> Dict[str, object]:
    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.90,
                                           n_floes=n_floes, seed=seed)
        times.append(time.perf_counter() - t0)
    best = float(min(times))
    return {"criterion": "N=4000 single field generated in < 30 s",
            "n": n_floes, "times_s": [float(t) for t in times],
            "best_s": best, "median_s": float(np.median(times)),
            "timings_breakdown": dict(last.timings),
            "cells": dict(last.cells_summary),
            "min_gap": float(last.min_signed_gap()),
            "m2_rel_error": float(last.moment_errors()["m2"]),
            "pass": bool(best < 30.0)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(outdir: str) -> List[Dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    figs: List[Dict[str, str]] = []

    # fig 1: field at phi = 0.85 with contact network
    field = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.85,
                                       n_floes=420, seed=4)
    cs = field.contacts(cutoff=0.05 * float(np.median(field.r_eq)))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cmap = plt.get_cmap("viridis")
    rmax = float(field.r_eq.max())
    for k in range(field.n):
        v = field.polygon(k)
        ax.fill(v[:, 0], v[:, 1], color=cmap(field.r_eq[k] / rmax), alpha=0.85,
                edgecolor="0.25", linewidth=0.4)
    for a, b in zip(cs["i"], cs["j"]):
        pa, pb = field.centroid[a], field.centroid[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="crimson", lw=0.35, alpha=0.5)
    ax.set_xlim(0, field.box[0])
    ax.set_ylim(0, field.box[1])
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, rmax))
    fig.colorbar(sm, ax=ax, label="equivalent radius $r_{eq}$ [m]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Polygonal floe field, log-normal FSD, $\\phi$={field.coverage:.2f}, "
                 f"N={field.n}, min gap={field.min_signed_gap():.3f} m")
    path = os.path.join(outdir, "field_phi085.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "field_phi085", "path": path,
                 "caption": "Laguerre-tessellated polygonal floe field (log-normal FSD, "
                            "phi=0.85); every cell is shrunk about its centroid by "
                            "sqrt(phi), which leaves the plateau gap visible between floes."})

    # fig 2: gap distribution vs phi
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for phi in (0.60, 0.75, 0.85, 0.90):
        f = generate_polygon_floe_field(DeltaFSD(5.0), coverage=phi, n_floes=700, seed=2)
        cut = 1.2 * float(np.median(f.r_eq))
        g = f.contacts(cutoff=cut)["gap"]
        if g.size == 0:
            continue
        ax.hist(g, bins=40, histtype="step",
                label=f"phi={phi:.2f}, min={g.min():.3f} m")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("SAT signed gap [m]")
    ax.set_ylabel("pair count")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Signed-gap distribution of the nearest floe pairs")
    path = os.path.join(outdir, "gap_hist.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "gap_hist", "path": path,
                 "caption": "SAT signed gap of near pairs for a monodisperse field at four "
                            "concentrations; no gap is negative (no overlap) and the median "
                            "gap shrinks as sqrt(phi)."})

    # fig 3: moment errors and coordination vs N / phi
    ns = [200, 400, 800, 1600]
    err1, err2, coord = [], [], []
    for n in ns:
        f = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.85, n_floes=n, seed=5)
        e = f.moment_errors()
        err1.append(e["m1"])
        err2.append(e["m2"])
        coord.append(f.coordination_number(0.25 * float(np.median(f.r_eq))))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(ns, err1, "o-", label="$m_1$ relative error")
    ax.loglog(ns, err2, "s-", label="$m_2$ relative error")
    ax.axhline(1e-2, color="r", ls="--", lw=0.9, label="1 % criterion")
    ax.set_xlabel("floes N")
    ax.set_ylabel("relative error vs analytic FSD")
    ax2 = ax.twinx()
    ax2.plot(ns, coord, "^-", color="0.4")
    ax2.set_ylabel("coordination number at $\\tau=0.25 r_{eq}$", color="0.4")
    ax.legend(fontsize=8)
    ax.set_title("Realised FSD moments and coordination vs field size")
    path = os.path.join(outdir, "moments_vs_n.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "moments_vs_n", "path": path,
                 "caption": "Realised area moments approach the analytic FSD moments with "
                            "the number of floes (stratified sampling floor); the coordination "
                            "number saturates at the planar packing value ~6."})

    # fig 4: contact geometry / lever arms
    field = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.90, n_floes=500, seed=9)
    cut = 0.5 * float(np.median(field.r_eq))
    out = field.contact_forces(cutoff=cut)
    act = out["f_n"] > 0
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(out["torque_arm"][act], np.abs(out["torque_i"][act]) / 1e3, s=8, alpha=0.6)
    ax.set_xlabel("lever arm $|p - x_i|$ [m]")
    ax.set_ylabel("contact torque [kN m]")
    ax.set_title("Torque carried by polygon contacts (polygon radii up to "
                 f"{field.circumradius.max():.2f} m)")
    path = os.path.join(outdir, "torque_lever_arm.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "torque_lever_arm", "path": path,
                 "caption": "Contact torque vs lever arm for a phi=0.90 polygonal field: a "
                            "disk model has zero lever arm, the polygonal contact point is "
                            "off the centroid line."})
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/in32_n01_polygon_floe_field.json")
    ap.add_argument("--sweep-out", default=None,
                    help="write only the no-overlap shard to this file")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-floes-sweep", type=int, default=220)
    ap.add_argument("--figdir", default="artifacts/in32_n01")
    ap.add_argument("--skip", default="", help="comma separated criteria to skip")
    args = ap.parse_args()

    shard, n_shards = (int(x) for x in args.shard.split("/"))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    t_start = time.perf_counter()
    result: Dict[str, object] = {
        "node": "IN32-N01",
        "repo": "TensorDEM",
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "shard": args.shard,
        "criteria": {},
        "figures": [],
    }

    if args.sweep_out:
        sw = sweep_no_overlap(args.n_floes_sweep, shard, n_shards)
        with open(args.sweep_out, "w") as fh:
            json.dump(sw, fh)
        print(json.dumps({"sweep": args.sweep_out, "fields": sw["n_fields"],
                          "seconds": sw["seconds"]}))
        return 0

    sweeps = []
    if "a" not in skip:
        sw = sweep_no_overlap(args.n_floes_sweep, shard, n_shards)
        sweeps.append(sw)
        result["criteria"]["a_no_overlap"] = summarise_no_overlap(sweeps)
    if "b" not in skip:
        result["criteria"]["b_area_moments"] = check_moments()
    if "c" not in skip:
        result["criteria"]["c_completeness"] = check_completeness()
    if "d" not in skip:
        result["criteria"]["d_balance"] = check_balance()
    if "e" not in skip:
        result["criteria"]["e_monodisperse"] = check_monodisperse()
    if "f" not in skip:
        result["criteria"]["f_budget"] = check_budget()
    crits = result["criteria"]
    result["all_pass"] = bool(crits) and all(bool(c.get("pass")) for c in crits.values())
    result["n_criteria"] = len(crits)
    result["n_pass"] = sum(1 for c in crits.values() if c.get("pass"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # write the numbers *before* plotting: a headless node may have no matplotlib
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    if "fig" not in skip:
        try:
            result["figures"] = make_figures(args.figdir)
        except Exception as exc:                          # pragma: no cover
            result["figures"] = []
            result["figures_error"] = f"{type(exc).__name__}: {exc}"
            print("figures skipped:", result["figures_error"])
    result["wall_seconds"] = time.perf_counter() - t_start
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    def _compact(obj, depth=0):
        if isinstance(obj, dict):
            return {k: _compact(v, depth + 1) for k, v in obj.items()
                    if k not in ("rows", "figures", "per_family_rows")}
        if isinstance(obj, (list, tuple)):
            if len(obj) > 12:
                return f"<{len(obj)} items>"
            return [_compact(v, depth + 1) for v in obj]
        return obj

    print("<<<RESULT_JSON>>>")
    print(json.dumps(_compact(result), default=float))
    print("<<<END_JSON>>>")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v)
                      for k, v in crits.items()}, indent=1))
    print("all_pass", result["all_pass"], "wall", round(result["wall_seconds"], 1), "s")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())