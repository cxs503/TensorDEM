"""Campaign P27W2-N07: non-spherical representation verdict (TensorDEM).

Modes
-----
``hertz``      C1 anchors: closed-form ellipsoid/plane Hertz stiffness vs the
               half-space integral equation, the free-boundary elliptic-relation
               scan, the superellipsoid pole gap/power law and the axisymmetric
               power-law indenter exponents (Love's equation).
``geom``       C2 (geometry): clump surface-deviation convergence in N for two
               shape exponents and two independent fitters.
``dissipation`` C3: spurious energy loss rate of a static clump bed.
``case NAME``  one repose/heap case (one process per case so the sweep can be
               sharded): ``sphere``, ``true_m4``, ``true_m8``, ``clump_m4_N32``,
               ``clump_m8_N8``, ``clump_m8_N32``, ``clump_m8_N128``.
``verdict``    C5: merge every fragment into ``shape_representation_verdict.json``.
``figures``    matplotlib figures for the report.

All numbers in the report are read from these JSON fragments (no hand copying).

Usage::

    PYTHONPATH=src python scripts/shape_representation_campaign.py hertz
    PYTHONPATH=src python scripts/shape_representation_campaign.py case sphere
    PYTHONPATH=src python scripts/shape_representation_campaign.py verdict
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from tensordem.shape_representation.clump_fit import clump_convergence
from tensordem.shape_representation.hertz_ellipsoid import (
    axisymmetric_indenter_law,
    ellipsoid_plane_contact,
    pole_contact_exponent,
    superellipsoid_pole_gap,
    verify_closed_form,
    verify_elliptic_relation,
)
from tensordem.shape_representation.repose_channels import (
    ChannelSpec,
    run_repose,
    static_pile_dissipation,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "shape_representation")
os.makedirs(ART, exist_ok=True)

# ---------------------------------------------------------------- case table
# All repose/heap cases share the same material, particle count and geometry.
BASE = dict(mu=0.6, mu_rolling=0.5, n_particles=320, box_half=0.045,
            gate_half=0.02, hold_steps=200, settle_steps=6000, dt=None,
            e_star=1.1e6, restitution=0.1, damping=50.0, seed=7)
RADIUS = 4.0e-3

CASES: Dict[str, Dict[str, object]] = {
    "sphere": {"spec": ChannelSpec("sphere", radius=RADIUS)},
    "true_m4": {"spec": ChannelSpec("true", shape_exponent=4.0, radius=RADIUS)},
    "true_m8": {"spec": ChannelSpec("true", shape_exponent=8.0, radius=RADIUS)},
    "clump_m4_N32": {"spec": ChannelSpec("clump", shape_exponent=4.0, n_spheres=32,
                                         radius=RADIUS)},
    "clump_m8_N8": {"spec": ChannelSpec("clump", shape_exponent=8.0, n_spheres=8,
                                        radius=RADIUS)},
    "clump_m8_N32": {"spec": ChannelSpec("clump", shape_exponent=8.0, n_spheres=32,
                                         radius=RADIUS)},
    "clump_m8_N128": {"spec": ChannelSpec("clump", shape_exponent=8.0, n_spheres=128,
                                          radius=RADIUS)},
}


def _write(name: str, payload: Dict[str, object]) -> str:
    path = os.path.join(ART, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(f"[write] {path}", flush=True)
    return path


def _read(name: str) -> Dict[str, object]:
    with open(os.path.join(ART, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# C1: Hertz anchors
# ---------------------------------------------------------------------------

def run_hertz(n_theta: int = 2048, n_s: int = 24) -> Dict[str, object]:
    rows = []
    for ratio in (1.0, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.16, 0.1):
        contact = ellipsoid_plane_contact(1.0 / ratio, 1.0, 1.0e9, load=1.0)
        ver = verify_closed_form(contact, n_theta=n_theta, n_s=n_s,
                                 n_ring=3, n_theta_pts=6)
        row = contact.to_dict()
        row.update(ver.to_dict())
        rows.append(row)
        print(f"  ratio {ratio:.2f}: k_rel_err={ver.stiffness_rel_error:.2e} "
              f"spread={ver.flatness_spread:.2e}", flush=True)
    # sphere limit against the textbook Hertz solution (radius R, load P)
    R, e_star, P = 10.0e-3, 1.0e9, 100.0
    sphere = ellipsoid_plane_contact(R, R, e_star, load=P)
    a_h = (3.0 * P * R / (4.0 * e_star)) ** (1.0 / 3.0)
    sphere_check = {
        "a_rel_error": abs(sphere.a_major - a_h) / a_h,
        "approach_rel_error": abs(sphere.approach - a_h * a_h / R) / (a_h * a_h / R),
        "stiffness_rel_error": abs(sphere.stiffness - 2.0 * e_star * a_h) / (2.0 * e_star * a_h),
        "closed_form": sphere.to_dict(),
        "textbook_a_m": a_h,
        "textbook_stiffness_n_per_m": 2.0 * e_star * a_h,
    }
    # free-boundary scan (contact shape from the integral equation only)
    scans = [verify_elliptic_relation(r, n_theta=1024, n_s=16, ratios=9)
             for r in (0.8, 0.6, 0.4, 0.25)]
    # superellipsoid pole: radii / power law
    poles = [superellipsoid_pole_gap(4e-3, 4e-3, 4e-3, m) for m in (2.0, 3.0, 4.0, 8.0)]
    laws = []
    for m in (1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
        out = axisymmetric_indenter_law(m, n_cells=100, n_gauss=32, kappa=1.0, e_star=1.0)
        laws.append({k: v for k, v in out.items() if k != "rows"})
        laws[-1]["rows_head"] = out["rows"][:3]
    payload = {
        "criterion": "C1: ellipsoid-plane normal stiffness vs Hertz elliptic-integral closed form <= 2%",
        "worst_stiffness_rel_error": float(max(r["stiffness_rel_error"] for r in rows)),
        "worst_flatness_spread": float(max(r["flatness_spread_rel"] for r in rows)),
        "rows": rows,
        "sphere_limit": sphere_check,
        "free_boundary_scan": scans,
        "superellipsoid_pole": poles,
        "power_law_indenter": laws,
        "verification_method": ("closed-form Hertz pressure substituted into the elastic "
                                "half-space integral equation and integrated with a "
                                "singularity-free polar quadrature; the stiffness is "
                                "recomputed from that integral"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return _write("hertz_anchors.json", payload)


# ---------------------------------------------------------------------------
# C2 (geometry): clump convergence
# ---------------------------------------------------------------------------

def run_geom() -> Dict[str, object]:
    out = {}
    for name, m in (("m4", 4.0), ("m8", 8.0)):
        res = clump_convergence(RADIUS, RADIUS, RADIUS, m, m, (8, 32, 128, 256),
                                n_samples=300)
        out[name] = res
        print(f"  {name}: covering rel {res['relative_error_covering']} "
              f"exp {res['convergence_exponent_covering']:.3f}", flush=True)
    payload = {"criterion": "C2 (geometry): clump surface deviation vs N", "shapes": out,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return _write("geom_convergence.json", payload)


# ---------------------------------------------------------------------------
# C2/C4: repose channel cases
# ---------------------------------------------------------------------------

def run_case(name: str) -> Dict[str, object]:
    case = CASES[name]
    spec: ChannelSpec = case["spec"]
    t0 = time.perf_counter()
    out = run_repose(spec, **BASE)
    out["case"] = name
    out["settle_probe_steps"] = 0
    out["wall_clock_total_s"] = time.perf_counter() - t0
    return _write(f"heap_{name}.json", out)


# ---------------------------------------------------------------------------
# C3: static-bed dissipation
# ---------------------------------------------------------------------------

def run_dissipation(n_particles: int = 40, steps: int = 1500) -> Dict[str, object]:
    """Static bed of clumps / spheres at rest: spurious energy loss per step."""
    out: Dict[str, object] = {}
    for name, spec in (("sphere", ChannelSpec("sphere", radius=RADIUS)),
                       ("clump_m8_N8", ChannelSpec("clump", shape_exponent=8.0,
                                                   n_spheres=8, radius=RADIUS)),
                       ("clump_m8_N32", ChannelSpec("clump", shape_exponent=8.0,
                                                    n_spheres=32, radius=RADIUS))):
        # simple cubic bed, spacing chosen so particles start just touching
        per = int(round(n_particles ** (1.0 / 3.0))) + 1
        sp = 2.02 * RADIUS
        pts = np.array([[ (i - per / 2) * sp, (j - per / 2) * sp, RADIUS * 1.02 + k * sp]
                        for k in range(3) for i in range(per) for j in range(per)])
        pts = pts[:n_particles]
        res = static_pile_dissipation(pts, spec, mu=BASE["mu"],
                                      mu_rolling=BASE["mu_rolling"],
                                      steps=steps, dt=None, density=2500.0,
                                      e_star=BASE["e_star"], restitution=0.05,
                                      box_half=max(3.0 * sp * per, 4.0 * RADIUS),
                                      pre_settle_steps=800)
        out[name] = res
        print(f"  {name}: drift/step {res['drift_per_step']:.3e} "
              f"(pass={res['passes']})", flush=True)
    payload = {
        "criterion": "C3: static-pile spurious dissipation <= 1e-6 / step",
        "note": ("a resting bed is rebuilt at zero velocity and run on; any energy "
                 "loss is numerical/representation induced (no work is done on it)"),
        "cases": out,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return _write("dissipation.json", payload)


# ---------------------------------------------------------------------------
# C5: verdict
# ---------------------------------------------------------------------------

def build_verdict() -> Dict[str, object]:
    from tensordem.shape_representation.verdict import assemble_verdict
    payload = assemble_verdict(ART)
    return _write("shape_representation_verdict.json", payload)


def main(argv: List[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "all"
    if mode == "hertz":
        run_hertz()
    elif mode == "geom":
        run_geom()
    elif mode == "dissipation":
        run_dissipation()
    elif mode == "case":
        run_case(argv[2])
    elif mode == "verdict":
        build_verdict()
    elif mode == "figures":
        from tensordem.shape_representation.figures import make_figures
        make_figures(ART, os.path.join(ROOT, "docs", "figures"))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))