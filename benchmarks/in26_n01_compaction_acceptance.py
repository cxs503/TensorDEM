"""IN26-N01 acceptance harness: compacted polygonal floe fields with contacts.

Machine-checkable criteria (the campaign node metadata):

(a) coverage      : phi in {0.60, 0.70, 0.80, 0.85} x 4 FSD families x seeds,
                    |realised - target| / target < 0.5 %   (nominal *and* rasterised
                    union coverage are reported and both are checked);
(b) no overlap    : max SAT penetration depth <= 0.01 m;
(c) monodisperse  : the monodisperse limit reproduces the IN32-N01 polygonal field
                    bit-for-bit (array equality + frozen SHA-256) and the H1
                    single-dispersion lattice field is regenerated deterministically;
(d) tests         : >= 12 pytest cases incl. negative controls (the suite is run
                    here and the collected/passed counts are recorded);
plus the *capability* the node exists for: a **contact network** at phi >= 0.85
that percolates (the buffered IN32-N01 field has zero contacts at every phi).

Every number in the JSON is produced by this script; nothing is transcribed.

Usage
-----
    python benchmarks/in26_n01_compaction_acceptance.py --out artifacts/in26_n01/acceptance.json
    python benchmarks/in26_n01_compaction_acceptance.py --quick --no-figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.floe_compaction import (  # noqa: E402
    CompactionParams, ConsolidationParams, compact_floe_field, contact_percolation,
    generate_compacted_floe_field, generate_consolidated_floe_field,
    h1_reference_digest, union_coverage, zone_order,
)
from tensordem.fsd_packing import (  # noqa: E402
    BimodalFSD, DeltaFSD, LogNormalFSD, PowerLawFSD,
)
from tensordem.laguerre_field import LaguerreParams  # noqa: E402
from tensordem.polygon_floe_field import (  # noqa: E402
    generate_polygon_floe_field, sample_floe_radii,
)

PHI_LIST = (0.60, 0.70, 0.80, 0.85)
FAMILIES = {
    "log_normal_0.5": lambda: LogNormalFSD(0.5),
    "power_law_2.5": lambda: PowerLawFSD(2.5),
    "bimodal": lambda: BimodalFSD(),
    "delta_5m": lambda: DeltaFSD(5.0),
}
COVERAGE_TOL = 5.0e-3      # criterion (a): 0.5 %
OVERLAP_TOL = 1.0e-2       # criterion (b): 0.01 m
CONTACT_GAP = 1.0e-2       # a pair within 1 cm counts as a contact


# ---------------------------------------------------------------------------
# bookkeeping helpers
# ---------------------------------------------------------------------------

def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def _field_digest(field) -> str:
    hasher = hashlib.sha256()
    for arr in (field.centroid, field.verts, field.count, field.area, field.r_eq):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def _solver() -> LaguerreParams:
    return LaguerreParams(area_tol=5.0e-4, max_iter=80)


# ---------------------------------------------------------------------------
# one (family, phi, seed) sweep point
# ---------------------------------------------------------------------------

def _measure(field, *, resolution: int) -> Dict[str, object]:
    cs = field.contacts(cutoff=CONTACT_GAP)
    perco = contact_percolation(field, contact_gap=CONTACT_GAP)
    uc = union_coverage(field.centroid, field.verts, field.count, field.box,
                        resolution=resolution)
    gmin = field.min_signed_gap()
    return {
        "nominal_coverage": float(field.area.sum() / (field.box[0] * field.box[1])),
        "union_coverage": float(uc["union_coverage"]),
        "double_covered": float(uc["double_covered"]),
        "min_gap": float(gmin),
        "max_overlap": float(max(0.0, -gmin)) if np.isfinite(gmin) else 0.0,
        "contacts": int(cs["i"].size),
        "coordination": float(perco["coordination"]),
        "clusters": int(perco["clusters"]),
        "largest_cluster_fraction": float(perco["largest_cluster_fraction"]),
        "spanning_clusters": int(perco["spanning_clusters"]),
        "percolating": bool(perco["percolating"]),
        "largest_span": float(perco["largest_span"]),
        "contact_gap": CONTACT_GAP,
    }


def sweep_point(family: str, phi: float, seed: int, n_floes: int,
                resolution: int) -> Dict[str, object]:
    fsd = FAMILIES[family]()
    t0 = time.perf_counter()
    buffer = generate_polygon_floe_field(fsd, coverage=phi, n_floes=n_floes,
                                        seed=seed)
    t_buffer = time.perf_counter() - t0
    buf = _measure(buffer, resolution=resolution)

    t0 = time.perf_counter()
    field = generate_consolidated_floe_field(fsd, coverage=phi, n_floes=n_floes,
                                             seed=seed, diagnose=True,
                                             params=ConsolidationParams(
                                                 laguerre=_solver()))
    t_field = time.perf_counter() - t0
    con = _measure(field, resolution=resolution)
    comp = dict(field.compaction)

    cov_dev_nom = abs(con["nominal_coverage"] - phi) / phi
    cov_dev_union = abs(con["union_coverage"] - phi) / phi
    return {
        "family": family,
        "phi": phi,
        "seed": seed,
        "n_floes": int(field.n),
        "box": [float(field.box[0]), float(field.box[1])],
        "buffer": buf,
        "consolidated": con,
        "construction": {k: comp.get(k) for k in (
            "zone", "lead_coverage_requested", "lead_coverage_realised",
            "lead_zone_factor", "consolidated_floes", "consolidated_area_fraction",
            "consolidated_area_target", "consolidated_area",
            "consolidated_area_rel_error", "budget_residual_rel",
            "cell_area_error", "solver_iterations", "tessellation_cover",
            "coordination_consolidated", "contacts_inside_consolidated")},
        "coverage_deviation_nominal": float(cov_dev_nom),
        "coverage_deviation_union": float(cov_dev_union),
        "criteria": {
            "a_coverage_nominal": bool(cov_dev_nom < COVERAGE_TOL),
            "a_coverage_union": bool(cov_dev_union < COVERAGE_TOL),
            "b_overlap": bool(con["max_overlap"] <= OVERLAP_TOL),
            "contacts_present": bool(con["contacts"] > 0),
            "percolating": bool(con["percolating"]),
        },
        "wall_s": {"buffer": t_buffer, "consolidated": t_field},
    }


# ---------------------------------------------------------------------------
# criterion (c): monodisperse regression locks
# ---------------------------------------------------------------------------

def monodisperse_locks(n_floes: int, seed: int = 7) -> Dict[str, object]:
    fsd = DeltaFSD(5.0)
    t0 = time.perf_counter()
    plain = generate_polygon_floe_field(fsd, coverage=0.85, n_floes=n_floes, seed=seed)
    locked = generate_compacted_floe_field(fsd, coverage=0.85, n_floes=n_floes,
                                          seed=seed, compaction=None)
    arrays_equal = all(
        np.array_equal(np.asarray(getattr(plain, k)), np.asarray(getattr(locked, k)))
        for k in ("centroid", "verts", "count", "area", "perimeter", "r_eq", "mass",
                  "inertia"))
    digests_equal = _field_digest(plain) == _field_digest(locked)

    h1 = h1_reference_digest(coverage=0.80, domain_length=120.0, domain_width=80.0,
                             seed=0)

    mono = generate_consolidated_floe_field(fsd, coverage=0.85, n_floes=n_floes,
                                            seed=seed, diagnose=False,
                                            params=ConsolidationParams(laguerre=_solver()))
    perco = contact_percolation(mono, contact_gap=CONTACT_GAP)
    cs = mono.contacts(cutoff=CONTACT_GAP)
    r_eq_spread = float(mono.r_eq.max() - mono.r_eq.min())
    return {
        "bit_for_bit_vs_in32_generator": bool(arrays_equal and digests_equal),
        "monodisperse_field_digest": _field_digest(plain),
        "h1_lattice": h1,
        "monodisperse_consolidated": {
            "r_eq_mean": float(mono.r_eq.mean()),
            "r_eq_spread": r_eq_spread,
            "r_eq_rel_spread": float(r_eq_spread / mono.r_eq.mean()),
            "coverage": float(mono.realised_coverage()),
            "contacts": int(cs["i"].size),
            "coordination": float(perco["coordination"]),
            "percolating": bool(perco["percolating"]),
            "max_overlap": float(max(0.0, -mono.min_signed_gap())),
        },
        "criteria": {
            "c_bit_for_bit": bool(arrays_equal and digests_equal),
            "c_h1_deterministic": bool(h1["deterministic"]),
            "c_h1_method": bool(h1["method"] == "h1_lattice"),
            "c_monodisperse_size": bool(r_eq_spread / mono.r_eq.mean() < 5e-3),
            "c_monodisperse_contacts": bool(cs["i"].size > 0 and perco["percolating"]),
        },
        "wall_s": float(time.perf_counter() - t0),
    }


# ---------------------------------------------------------------------------
# the negative experiment: homogeneous attraction/projection compaction
# ---------------------------------------------------------------------------

def homogeneous_compaction(phi: float, n_floes: int, seed: int) -> Dict[str, object]:
    fsd = LogNormalFSD(0.5)
    buffer = generate_polygon_floe_field(fsd, coverage=phi, n_floes=n_floes, seed=seed)
    t0 = time.perf_counter()
    comp = compact_floe_field(buffer, CompactionParams(stages=2, steps_per_stage=20,
                                                      projection_sweeps=6))
    wall = time.perf_counter() - t0
    perco = contact_percolation(comp, contact_gap=CONTACT_GAP)
    cs = comp.contacts(cutoff=CONTACT_GAP)
    return {
        "phi": float(phi),
        "n_floes": int(comp.n),
        "steps": int(comp.compaction["steps"]),
        "energy_initial": float(comp.compaction["energy_initial"]),
        "energy_final": float(comp.compaction["energy_final"]),
        "energy_history": [float(x) for x in comp.compaction["energy_history"]],
        "coordination_history": [float(x) for x in comp.compaction["coordination_history"]],
        "residual_overlap": float(comp.compaction["residual_overlap"]),
        "contacts": int(cs["i"].size),
        "coordination": float(perco["coordination"]),
        "clusters": int(perco["clusters"]),
        "percolating": bool(perco["percolating"]),
        "largest_cluster_fraction": float(perco["largest_cluster_fraction"]),
        "wall_s": float(wall),
    }


# ---------------------------------------------------------------------------
# criterion (d): pytest
# ---------------------------------------------------------------------------

def run_pytest(test_file: str = "tests/test_floe_compaction.py"
               ) -> Dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
    root = os.path.join(os.path.dirname(__file__), "..")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest", test_file, "-q"],
                          cwd=root, env=env, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    passed = 0
    import re
    m_pass = re.search(r"(\d+) passed", proc.stdout or "")
    m_total = re.search(r"(\d+) (passed|failed|error)", proc.stdout or "")
    if m_pass:
        passed = int(m_pass.group(1))
    return {"command": f"python -m pytest {test_file} -q",
            "returncode": int(proc.returncode),
            "passed": passed,
            "summary": tail[0],
            "wall_s": float(time.perf_counter() - t0)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figures(outdir: str, sweep: List[Dict[str, object]],
             negative: List[Dict[str, object]]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    os.makedirs(outdir, exist_ok=True)
    made: List[str] = []

    # 1: field snapshots, buffered vs consolidated at phi = 0.85
    row = [r for r in sweep if r["phi"] == 0.85 and r["family"] == "log_normal_0.5"][0]
    fsd = FAMILIES[row["family"]]()
    buffer = generate_polygon_floe_field(fsd, coverage=0.85, n_floes=row["n_floes"],
                                        seed=row["seed"])
    field = generate_consolidated_floe_field(fsd, coverage=0.85, n_floes=row["n_floes"],
                                             seed=row["seed"], diagnose=False,
                                             params=ConsolidationParams(laguerre=_solver()))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, fld, title in ((axes[0], buffer, "IN32-N01 buffered field (no contacts)"),
                           (axes[1], field, "IN26-N01 consolidated field (contacts)")):
        cs = fld.contacts(cutoff=CONTACT_GAP)
        polys = [fld.polygon(k) for k in range(fld.n)]
        pcol = PolyCollection(polys, array=np.asarray(fld.r_eq),
                              cmap="viridis", edgecolors="k", linewidths=0.3)
        ax.add_collection(pcol)
        if cs["i"].size:
            p0 = cs["point"]
            ax.plot(p0[:, 0], p0[:, 1], ".", ms=1.2, color="crimson")
        ax.set_xlim(0.0, fld.box[0])
        ax.set_ylim(0.0, fld.box[1])
        ax.set_aspect("equal")
        ax.set_title(f"{title}\ncontacts={cs['i'].size}, "
                     f"Z={2 * cs['i'].size / fld.n:.2f}, phi={fld.realised_coverage():.4f}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(pcol, ax=ax, label="r_eq [m]", fraction=0.046)
    fig.tight_layout()
    path = os.path.join(outdir, "field_phi085.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    # 2: coordination vs coverage
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for fam in FAMILIES:
        rows = [r for r in sweep if r["family"] == fam]
        rows.sort(key=lambda r: r["phi"])
        ax.plot([r["phi"] for r in rows],
                [r["consolidated"]["coordination"] for r in rows], "o-",
                label=f"{fam} (consolidated)")
        ax.plot([r["phi"] for r in rows],
                [r["buffer"]["coordination"] for r in rows], "x--", color="0.6",
                label=f"{fam} (buffered, IN32)")
    ax.set_xlabel("target coverage phi")
    ax.set_ylabel("mean coordination number Z")
    ax.set_title("Contact network: buffered field is contact-free, consolidated field is jammed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    path = os.path.join(outdir, "coordination_vs_phi.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    # 3: coverage error and max overlap with the acceptance thresholds
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    labels = [f"{r['family']}\nphi={r['phi']:.2f}" for r in sweep]
    x = np.arange(len(sweep))
    axes[0].bar(x - 0.2, [r["coverage_deviation_nominal"] for r in sweep], width=0.4,
                label="nominal (area bookkeeping)")
    axes[0].bar(x + 0.2, [r["coverage_deviation_union"] for r in sweep], width=0.4,
                label="rasterised union")
    axes[0].axhline(COVERAGE_TOL, color="r", ls="--", label="0.5 % limit")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative coverage deviation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=5, rotation=90)
    axes[0].legend(fontsize=7)
    axes[0].set_title("criterion (a): coverage")

    axes[1].bar(x, [max(r["consolidated"]["max_overlap"], 1e-16) for r in sweep])
    axes[1].axhline(OVERLAP_TOL, color="r", ls="--", label="0.01 m limit")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max SAT overlap [m]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=5, rotation=90)
    axes[1].legend(fontsize=7)
    axes[1].set_title("criterion (b): overlap")
    fig.tight_layout()
    path = os.path.join(outdir, "acceptance_criteria.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    # 4: the negative experiment (homogeneous attraction stalls)
    if negative:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        for rec in negative:
            axes[0].plot(rec["energy_history"],
                         label=f"phi={rec['phi']:.2f} (E {rec['energy_initial']:.3f}"
                               f" -> {rec['energy_final']:.3f})")
            axes[1].plot(rec["coordination_history"], label=f"phi={rec['phi']:.2f}")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("interface energy E (normalised)")
        axes[0].set_title("homogeneous attraction/projection: energy barely decreases")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("coordination number during compaction")
        axes[1].set_title("... and the contact network does not form")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, "negative_homogeneous_compaction.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(path)

    # 5: zone gallery
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for ax, zone in zip(axes, ("band", "disc", "random")):
        fld = generate_consolidated_floe_field(
            DeltaFSD(5.0), coverage=0.85, n_floes=180, seed=4, diagnose=True,
            params=ConsolidationParams(zone=zone, laguerre=_solver()))
        polys = [fld.polygon(k) for k in range(fld.n)]
        perco = contact_percolation(fld, contact_gap=CONTACT_GAP)
        ax.add_collection(PolyCollection(polys, facecolors="#cfe8f5",
                                         edgecolors="k", linewidths=0.4))
        ax.set_xlim(0.0, fld.box[0])
        ax.set_ylim(0.0, fld.box[1])
        ax.set_aspect("equal")
        ax.set_title(f"zone={zone}\nZ={perco['coordination']:.2f}, "
                     f"percolating={perco['percolating']}")
        ax.set_xlabel("x [m]")
    fig.tight_layout()
    path = os.path.join(outdir, "zone_gallery.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)
    return made


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/in26_n01/acceptance.json")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--n-floes", type=int, default=250)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--phis", default=",".join(str(x) for x in PHI_LIST))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-pytest", action="store_true")
    ap.add_argument("--no-negative", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="one family, one seed (smoke run)")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    phis = [float(s) for s in args.phis.split(",") if s.strip()]
    families = list(FAMILIES)
    if args.quick:
        families = ["log_normal_0.5"]
        seeds = seeds[:1]
        phis = [0.85]
        args.resolution = min(args.resolution, 256)

    t_start = time.perf_counter()
    sweep: List[Dict[str, object]] = []
    for fam in families:
        for phi in phis:
            for seed in seeds:
                rec = sweep_point(fam, phi, seed, args.n_floes, args.resolution)
                sweep.append(rec)
                print(f"[sweep] {fam:15s} phi={phi:.2f} seed={seed} "
                      f"cov_dev={rec['coverage_deviation_nominal']:.2e}/"
                      f"{rec['coverage_deviation_union']:.2e} "
                      f"maxov={rec['consolidated']['max_overlap']:.1e} "
                      f"Z={rec['consolidated']['coordination']:.2f} "
                      f"perco={rec['consolidated']['percolating']}", flush=True)

    locks = monodisperse_locks(args.n_floes)
    print(f"[lock] bit_for_bit={locks['bit_for_bit_vs_in32_generator']} "
          f"h1={locks['h1_lattice']['method']}", flush=True)

    negative: List[Dict[str, object]] = []
    if not args.no_negative:
        for phi in (0.60, 0.85):
            neg = homogeneous_compaction(phi, args.n_floes, 0)
            negative.append(neg)
            print(f"[negative] homogeneous compaction phi={phi:.2f} "
                  f"Z={neg['coordination']:.2f} perco={neg['percolating']} "
                  f"clusters={neg['clusters']}", flush=True)

    tests = {"skipped": True}
    if not args.no_pytest:
        tests = run_pytest()
        print(f"[pytest] {tests['summary']}", flush=True)

    figures: List[str] = []
    if not args.no_figures:
        figdir = args.figdir or os.path.join(os.path.dirname(args.out), "figures")
        figures = _figures(figdir, sweep, negative)
        print(f"[figures] {len(figures)} written to {figdir}", flush=True)

    crit_a_nom = all(r["criteria"]["a_coverage_nominal"] for r in sweep)
    crit_a_union = all(r["criteria"]["a_coverage_union"] for r in sweep)
    crit_b = all(r["criteria"]["b_overlap"] for r in sweep)
    crit_contacts = all(r["criteria"]["contacts_present"] for r in sweep)
    crit_perco = all(r["criteria"]["percolating"] for r in sweep)
    crit_c = all(locks["criteria"].values())
    crit_d = bool(tests.get("returncode") == 0 and tests.get("passed", 0) >= 12)

    payload = {
        "node": "IN26-N01",
        "title": "Compacted polygonal floe fields with a percolating contact network",
        "git_rev": _git_rev(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "numpy": np.__version__},
        "settings": {"n_floes": args.n_floes, "seeds": seeds, "phis": phis,
                     "families": families, "resolution": args.resolution,
                     "contact_gap": CONTACT_GAP, "coverage_tol": COVERAGE_TOL,
                     "overlap_tol": OVERLAP_TOL},
        "sweep": sweep,
        "monodisperse_locks": locks,
        "negative_result": negative,
        "tests": tests,
        "figures": figures,
        "thresholds": {"coverage_rel": COVERAGE_TOL, "overlap_m": OVERLAP_TOL,
                       "min_pytest_cases": 12},
        "verdict": {
            "a_coverage_nominal": {"pass": bool(crit_a_nom), "threshold": COVERAGE_TOL},
            "a_coverage_union": {"pass": bool(crit_a_union), "threshold": COVERAGE_TOL},
            "b_overlap": {"pass": bool(crit_b), "threshold": OVERLAP_TOL},
            "c_monodisperse_lock": {"pass": bool(crit_c)},
            "d_pytest": {"pass": bool(crit_d), "cases": tests.get("passed")},
            "contacts_present": {"pass": bool(crit_contacts)},
            "contact_network_percolates": {"pass": bool(crit_perco)},
            "all_pass": bool(crit_a_nom and crit_a_union and crit_b and crit_c
                             and crit_d and crit_contacts and crit_perco),
        },
        "wall_s": float(time.perf_counter() - t_start),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload["verdict"], indent=2))
    print(f"[out] {args.out}  ({payload['wall_s']:.1f} s)")
    return 0 if payload["verdict"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())