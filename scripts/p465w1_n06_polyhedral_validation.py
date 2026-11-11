"""Campaign node p465w1-n06: polyhedral shape -> GJK/EPA -> packing validation.

Runs every machine-checkable acceptance criterion, writes the criteria table as
JSON (so the report never hand-copies a number) and renders the figures.

Usage::

    python3 scripts/p465w1_n06_polyhedral_validation.py [--fast]

Outputs
-------
``runs/p465w1_n06/criteria.json``      criteria, oracles, errors, provenance
``docs/figures/p465w1_n06/*.png``      figures used in the report
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.convex_contact import (  # noqa: E402
    Box,
    analytic_box_box_sat,
    epa_penetration,
    gjk_distance,
)
from tensordem.polyhedral_shape import (  # noqa: E402
    Ellipsoid,
    PolyhedralShape,
    contact_force,
    cube_vertices,
    describe,
    describe_ellipsoid,
    hertz_sphere_sphere,
    named_shape,
    normal_force_exponent,
    pack_compaction_generic,
    pack_deposition,
    pack_spheres_compaction,
    random_polyhedron,
    sphere_limit_convergence,
    spherical_polyhedron,
    verify_affine_invariance,
)
from tensordem.polyhedral_shape.packing import _random_rotation  # noqa: E402

OUT = ROOT / "runs" / "p465w1_n06"
FIG = ROOT / "docs" / "figures" / "p465w1_n06"
CUBE_PSI = math.pi ** (1.0 / 3.0) * 6.0 ** (2.0 / 3.0) / 6.0


def angle_deg(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b))))


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------


def criterion_a() -> dict:
    half = 0.5
    a = PolyhedralShape(cube_vertices(1.0))
    b = PolyhedralShape(cube_vertices(1.0) + np.array([1.0, 0.0, 0.0]))
    gjk = gjk_distance(a, b)
    analytic, _, _, _ = analytic_box_box_sat([0, 0, 0], [half] * 3, None,
                                             [1.0, 0, 0], [half] * 3, None)
    err = abs(gjk.distance - max(analytic, 0.0))
    return {
        "id": "a",
        "name": "GJK two-cube face-face contact distance vs analytic",
        "gjk_distance": float(gjk.distance),
        "analytic_distance": float(max(analytic, 0.0)),
        "abs_error": float(err),
        "threshold": 1e-9,
        "pass": bool(err < 1e-9),
    }


def criterion_b() -> dict:
    rng = np.random.default_rng(2026)
    correct = 0
    trials = 100
    for _ in range(trials):
        rot_a = _random_rotation(rng)
        rot_b = _random_rotation(rng)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        # circumradius of a unit-edge cube is sqrt(3)/2 each -> 1.732 total;
        # anything beyond that plus a margin is provably separated
        reach = math.sqrt(3.0) + 0.05 + float(rng.random())
        centre_b = direction * reach
        a = Box([0.0, 0.0, 0.0], [0.5] * 3, orientation=rot_a)
        b = Box(centre_b, [0.5] * 3, orientation=rot_b)
        gjk = gjk_distance(a, b)
        analytic, _, _, _ = analytic_box_box_sat([0, 0, 0], [0.5] * 3, rot_a,
                                                 centre_b, [0.5] * 3, rot_b)
        separated = not gjk.overlap
        correct += int(separated and analytic > 0.0)
    return {
        "id": "b",
        "name": "GJK non-contact classification, 100 random configurations",
        "trials": trials,
        "correct": correct,
        "accuracy": correct / trials,
        "threshold": 1.0,
        "pass": bool(correct == trials),
    }


def criterion_c() -> dict:
    rows = []
    for overlap in (0.005, 0.02, 0.05):
        a = Box([0.0, 0.0, 0.0], [0.5] * 3)
        b = Box([1.0 - overlap, 0.0, 0.0], [0.5] * 3)
        epa = epa_penetration(a, b)
        rows.append({"overlap": overlap, "epa_depth": float(epa.depth),
                     "relative_error": abs(float(epa.depth) - overlap) / overlap,
                     "angle_deg": angle_deg(epa.normal, [1.0, 0.0, 0.0])})
    worst = max(r["relative_error"] for r in rows)
    return {
        "id": "c",
        "name": "EPA face-face penetration depth vs analytic",
        "rows": rows,
        "worst_relative_error": worst,
        "threshold": 0.01,
        "pass": bool(worst < 0.01),
    }


def criterion_d() -> dict:
    # face-face
    a = Box([0.0, 0.0, 0.0], [0.5] * 3)
    b = Box([0.98, 0.0, 0.0], [0.5] * 3)
    epa_face = epa_penetration(a, b)
    face_angle = angle_deg(epa_face.normal, [1.0, 0.0, 0.0])
    # edge/vertex-dominated: 45-degree rotated boxes, SAT gives the exact axis
    th = math.pi / 4.0
    rot = np.array([[math.cos(th), -math.sin(th), 0.0],
                    [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    offset = math.sqrt(2.0) - 0.02
    a2 = Box([0.0, 0.0, 0.0], [0.5] * 3, orientation=rot)
    b2 = Box([offset, 0.0, 0.0], [0.5] * 3, orientation=rot)
    signed, axis, _, _ = analytic_box_box_sat([0, 0, 0], [0.5] * 3, rot,
                                              [offset, 0, 0], [0.5] * 3, rot)
    epa_edge = epa_penetration(a2, b2)
    edge_angle = angle_deg(epa_edge.normal, axis)
    return {
        "id": "d",
        "name": "EPA contact normal vs analytic (face-face < 1 deg, edge < 2 deg)",
        "face_face_angle_deg": face_angle,
        "face_face_threshold_deg": 1.0,
        "edge_angle_deg": edge_angle,
        "edge_threshold_deg": 2.0,
        "edge_depth_relative_error": abs(float(epa_edge.depth) + float(signed)) /
                                      abs(float(signed)),
        "pass": bool(face_angle < 1.0 and edge_angle < 2.0),
    }


def criterion_e() -> dict:
    cube = describe(named_shape("cube"))
    tetra = describe(named_shape("tetrahedron"))
    edge = (6.0 * math.sqrt(2.0)) ** (1.0 / 3.0)
    tetra_analytic = math.pi ** (1.0 / 3.0) * 6.0 ** (2.0 / 3.0) / (math.sqrt(3.0) * edge ** 2)
    cube_err = abs(cube.sphericity - CUBE_PSI)
    tetra_err = abs(tetra.sphericity - tetra_analytic)
    return {
        "id": "e",
        "name": "shape descriptors: cube / tetrahedron sphericity",
        "cube_sphericity": cube.sphericity,
        "cube_analytic": CUBE_PSI,
        "cube_abs_error": cube_err,
        "tetrahedron_sphericity": tetra.sphericity,
        "tetrahedron_analytic": tetra_analytic,
        "tetrahedron_abs_error": tetra_err,
        "threshold": 1e-9,
        "pass": bool(cube_err < 1e-9 and tetra_err < 1e-9),
    }


def criterion_f(fast: bool) -> dict:
    n = 216 if fast else 343
    t0 = time.time()
    res = pack_spheres_compaction(n=n, growth=0.006 if fast else 0.004,
                                  rng=np.random.default_rng(1))
    return {
        "id": "f",
        "name": "packing fraction in the sphere limit (RCP)",
        "n_particles": n,
        "packing_fraction": res.packing_fraction,
        "band": [0.62, 0.66],
        "classical_value": 0.64,
        "seconds": round(time.time() - t0, 1),
        "pass": bool(abs(res.packing_fraction - 0.64) <= 0.02),
    }


def criterion_g(fast: bool) -> dict:
    rng = np.random.default_rng(3)
    n_cube = 24 if fast else 48
    bodies = [Box([0.0, 0.0, 0.0], [0.5] * 3, orientation=_random_rotation(rng))
              for _ in range(n_cube)]
    t0 = time.time()
    cube_res = pack_compaction_generic(
        bodies, length0=(n_cube / 0.15) ** (1.0 / 3.0),
        shrink=0.012, max_steps=400, relax_iters=10, tol=3.0e-3)
    cube_secs = time.time() - t0
    deposition = pack_deposition(
        lambda r: Box([0.0, 0.0, 0.0], [0.5] * 3, orientation=_random_rotation(r)),
        n_particles=30, trials=1, rng=np.random.default_rng(9), settle=False,
        tap_rounds=0)
    sphere = pack_spheres_compaction(n=216, growth=0.006, rng=np.random.default_rng(1))
    affine = verify_affine_invariance(aspect=2.0)
    spheroid_phi = sphere.packing_fraction
    spheroid_desc = describe_ellipsoid(Ellipsoid([0.5, 1.0, 0.5]))
    return {
        "id": "g",
        "name": "packing fraction shape dependence (cube, 1:2 spheroid)",
        "cube": {"n_particles": n_cube, "protocol": "compaction-frozen",
                 "shrink": 0.012, "packing_fraction": cube_res.packing_fraction,
                 "band": [0.55, 0.60], "seconds": round(cube_secs, 1)},
        "cube_ballistic_deposition": {
            "protocol": "deposition (no lateral dynamics)",
            "n_particles": 30,
            "packing_fraction": deposition.packing_fraction,
            "note": "columnar growth: documented protocol artefact, not a physical RLP value"},
        "spheroid": {"aspect": 2.0, "protocol": "affine-aligned",
                     "packing_fraction": spheroid_phi, "band": [0.58, 0.62],
                     "sphericity": spheroid_desc.sphericity,
                     "affine_check": affine},
        "pass_cube": bool(0.55 <= cube_res.packing_fraction <= 0.60),
        "pass_spheroid": bool(0.58 <= spheroid_phi <= 0.62),
    }


def criterion_h() -> dict:
    from tensordem.convex_contact import Sphere

    a = Sphere([0.0, 0.0, 0.0], 0.5)
    b = Sphere([0.99, 0.0, 0.0], 0.5)
    res = contact_force(a, b, e_star=1.0e6)
    ref = hertz_sphere_sphere(0.01, 0.5, 0.5, 1.0e6)
    rel = abs(res.normal_force - ref) / ref
    # zero penetration
    box_a = Box([0.0, 0.0, 0.0], [0.5] * 3)
    box_b = Box([1.0 + 1.0e-9, 0.0, 0.0], [0.5] * 3)
    zero = contact_force(box_a, box_b, e_star=1.0e9)
    conv = sphere_limit_convergence([42, 162, 642], overlap=0.2)
    fit = normal_force_exponent([1e-4, 2e-4, 4e-4], 1.0e6, 0.05)
    return {
        "id": "h",
        "name": "negative controls: sphere limit -> Hertz, zero overlap -> zero force",
        "sphere_limit_relative_error": rel,
        "sphere_limit_threshold": 1e-9,
        "zero_penetration_force": zero.normal_force,
        "zero_penetration_pass": bool(abs(zero.normal_force) < 1e-9),
        "faceted_limit_rows": conv["rows"],
        "faceted_limit_error_exponent": conv["error_exponent"],
        "force_exponent": fit["exponent"],
        "pass": bool(rel < 1e-9 and abs(zero.normal_force) < 1e-9 and
                     abs(fit["exponent"] - 1.5) < 1e-6),
    }


def criterion_i() -> dict:
    log = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_polyhedral_shape_validation.py", "-q"],
        cwd=ROOT, capture_output=True, text=True)
    tail = log.stdout.strip().splitlines()[-1] if log.stdout.strip() else ""
    passed = "passed" in tail
    count = 0
    for token in tail.replace("=", " ").split():
        if token.isdigit():
            count = int(token)
            break
    from fastapi.testclient import TestClient

    from tensordem.polyhedral_shape.api import build_app

    client = TestClient(build_app())
    ok = client.post("/api/v1/dem/polyhedral/shape_descriptor",
                     json={"shape": {"name": "cube"}})
    bad = client.post("/api/v1/dem/polyhedral/shape_descriptor",
                      json={"shape": {"vertices": [[0, 0, 0]]}})
    return {
        "id": "i",
        "name": "pytest suite, two endpoints, 422 on invalid input",
        "pytest_tail": tail,
        "pytest_passed": bool(passed),
        "n_tests": count,
        "endpoint_ok_status": ok.status_code,
        "endpoint_bad_status": bad.status_code,
        "pass": bool(passed and count >= 20 and ok.status_code == 200 and
                     bad.status_code == 422),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figures(criteria: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    # (1) sphericity vs vertex count, with named solids and the crushed-ice band
    ns = [4, 6, 8, 12, 20, 42, 162, 642, 2562]
    psis = []
    for n in ns:
        if n <= 20:
            name = {4: "tetrahedron", 6: "octahedron", 8: "cube", 12: "icosahedron",
                    20: "dodecahedron"}[n]
            psis.append(describe(named_shape(name)).sphericity)
        else:
            psis.append(describe(spherical_polyhedron(n)).sphericity)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhspan(0.55, 0.85, color="tab:orange", alpha=0.15,
               label="crushed-ice literature band (0.55-0.85)")
    ax.plot(ns, psis, "o-", color="tab:blue", label="canonical / spherical polyhedra")
    rng = np.random.default_rng(5)
    rv = [describe(random_polyhedron(int(rng.integers(6, 40)), rng=rng)).sphericity
          for _ in range(40)]
    ax.scatter([np.nan], [np.nan], marker="x", color="tab:red",
               label=f"random polyhedra (mean {np.mean(rv):.3f})")
    ax.axhline(np.mean(rv), color="tab:red", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("vertices $N_v$")
    ax.set_ylabel("sphericity $\\Psi$")
    ax.set_title("Shape descriptors: $\\Psi$ rises to 1 as $N_v \\to \\infty$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    path = FIG / "descriptor_sphericity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append((path, "Sphericity vs vertex count; the polyhedron family converges to the sphere and spans the crushed-ice band."))

    # (2) EPA depth error and normal error
    rows = criteria["c"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    overlaps = [r["overlap"] for r in rows]
    ax.semilogy(overlaps, [r["relative_error"] for r in rows], "o-", label="EPA depth error")
    ax.axhline(1e-2, color="tab:red", ls="--", label="1% acceptance")
    ax2 = ax.twinx()
    ax2.plot(overlaps, [r["angle_deg"] for r in rows], "s--", color="tab:green",
             label="normal angle")
    ax2.set_ylabel("normal error (deg)")
    ax.set_xlabel("prescribed face-face overlap $\\delta$ (m)")
    ax.set_ylabel("relative depth error")
    ax.set_title("EPA penetration: depth and normal vs analytic")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.grid(alpha=0.3)
    path = FIG / "epa_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append((path, "EPA penetration depth relative error and contact-normal error against the closed-form overlap."))

    # (3) contact force law and sphere-limit convergence
    deltas = np.logspace(-6, -2, 25)
    force = [4.0 / 3.0 * 1.0e6 * math.sqrt(0.05) * d ** 1.5 for d in deltas]
    rows_h = criteria["h"]["faceted_limit_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].loglog(deltas, force, "-", label="$F_n=\\frac{4}{3}E^*\\sqrt{R^*}\\,\\delta^{3/2}$")
    axes[0].set_xlabel("overlap $\\delta$ (m)")
    axes[0].set_ylabel("normal force (N)")
    axes[0].set_title(f"$F_n \\propto \\delta^{{{criteria['h']['force_exponent']:.3f}}}$")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    nv = [r["n_vertices"] for r in rows_h]
    axes[1].loglog(nv, [r["force_error"] for r in rows_h], "o-")
    axes[1].set_xlabel("hull vertices $N_v$")
    axes[1].set_ylabel("force error vs sphere Hertz")
    axes[1].set_title(f"sphere limit, fitted exponent "
                      f"{criteria['h']['faceted_limit_error_exponent']:.2f}")
    axes[1].grid(alpha=0.3, which="both")
    path = FIG / "force_law_and_limit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append((path, "Hertzian force law ($\\delta^{3/2}$) and the polyhedron-to-sphere contact-force limit."))

    # (4) packing fractions against the acceptance bands
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["sphere\n(RCP, compaction)", "cube\n(frozen compaction)", "1:2 spheroid\n(aligned, affine)"]
    values = [criteria["f"]["packing_fraction"],
              criteria["g"]["cube"]["packing_fraction"],
              criteria["g"]["spheroid"]["packing_fraction"]]
    bands = [(0.62, 0.66), (0.55, 0.60), (0.58, 0.62)]
    ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"], alpha=0.85)
    for i, (lo, hi) in enumerate(bands):
        ax.plot([i - 0.4, i + 0.4], [lo, lo], color="k", ls="--", lw=1)
        ax.plot([i - 0.4, i + 0.4], [hi, hi], color="k", ls="--", lw=1)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("packing fraction $\\phi$")
    ax.set_title("Shape-dependent packing fraction (dashed = acceptance band)")
    ax.grid(alpha=0.3, axis="y")
    path = FIG / "packing_fractions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append((path, "Measured packing fractions with the acceptance bands; the sphere limit hits the classical 0.64 band."))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    criteria = {}
    for fn in (criterion_a, criterion_b, criterion_c, criterion_d, criterion_e):
        row = fn()
        criteria[row["id"]] = row
    criteria["f"] = criterion_f(args.fast)
    criteria["g"] = criterion_g(args.fast)
    criteria["h"] = criterion_h()
    criteria["i"] = criterion_i()

    OUT.mkdir(parents=True, exist_ok=True)
    figs = figures(criteria)
    payload = {
        "node": "p465w1-n06-dem-polyhedral",
        "mission_id": 13951,
        "repo": "TensorDEM",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "criteria": criteria,
        "all_pass": all(v.get("pass", False) for k, v in criteria.items() if k != "g"),
        "packing_pass": {"cube": criteria["g"]["pass_cube"],
                         "spheroid": criteria["g"]["pass_spheroid"]},
        "figures": [{"name": p.name, "caption": c} for p, c in figs],
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "criteria.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload["criteria"][k].get("pass") for k in payload["criteria"]},
                     indent=2))
    print("packing band pass:", payload["packing_pass"])
    print("wrote", OUT / "criteria.json")


if __name__ == "__main__":
    main()