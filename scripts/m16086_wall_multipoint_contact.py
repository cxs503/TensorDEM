#!/usr/bin/env python3
"""m16086: vertex-distributed (multi-point) wall contact vs the single-point law.

Closes the A5 limitation recorded in
``docs/p273w3_n02_friction_polyhedron_verification.md``:

    "Single-point face contact cannot shift its center of pressure.  A flat-face
     cube on an incline has its contact collapsed to the face centroid ... a
     freely rotating block creeps at angles below atan(mu) (measured 0.19 mm creep
     at 4 deg with mu = 0.3).  Remedy: multi-point (vertex-face) wall contact ...
     not yet implemented."

The remedy is implemented as the opt-in law
``FrictionPolyhedronConfig.wall_multi_point`` (``src/tensordem/friction_polyhedron.py``).
This script measures, in the *same* harness and with the same parameters, what
each law gives:

* ``creep``      -- down-slope drift of a freely rotating cube at ``theta < atan(mu)``;
* ``critical``   -- critical incline angle by bisection against the measured slide;
* ``translation``-- above the limit the block must slide with ``a = g(sin-mu cos)``
  (guards against the multi-point law adding spurious resistance);
* ``tipping``    -- a tall block whose centre of mass leaves the support polygon
  must still tip (guards against spurious over-stabilisation).

All numbers are written to ``artifacts/m16086/`` as JSON/CSV so that no value in
the report is hand-copied.

Usage
-----
    PYTHONPATH=src python3 scripts/m16086_wall_multipoint_contact.py creep
    PYTHONPATH=src python3 scripts/m16086_wall_multipoint_contact.py critical --mu 0.30 --law both
    PYTHONPATH=src python3 scripts/m16086_wall_multipoint_contact.py translation
    PYTHONPATH=src python3 scripts/m16086_wall_multipoint_contact.py tipping
    PYTHONPATH=src python3 scripts/m16086_wall_multipoint_contact.py collect
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tensordem.friction_polyhedron import (
    FrictionPolyhedronConfig,
    PolyhedronDEMSystem,
    PolyhedronParticle,
)
from tensordem.friction_polyhedron_real_verification import (
    build_incline_case,
    incline_basis,
)

OUTDIR = Path("artifacts/m16086")
LAW_FLAG = {"single": False, "multi": True}


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------
def _apply_law(system: PolyhedronDEMSystem, law: str) -> None:
    system.config = dataclasses.replace(system.config,
                                        wall_multi_point=LAW_FLAG[law])


def _run(system: PolyhedronDEMSystem, particle: PolyhedronParticle,
         dt: float, t_end: float, samples: int = 0,
         downslope: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Integrate the system and return displacement diagnostics."""
    x0 = particle.x.copy()
    n_steps = int(round(t_end / dt))
    series: List[Dict[str, float]] = []
    every = max(1, n_steps // samples) if samples else 0
    for k in range(n_steps):
        system.step(dt)
        if every and (k + 1) % every == 0:
            d = particle.x - x0
            series.append({
                "t": (k + 1) * dt,
                "slide_m": float(np.dot(d, downslope)) if downslope is not None else float(np.linalg.norm(d)),
                "speed_m_s": float(np.linalg.norm(particle.v)),
                "omega_rad_s": float(np.linalg.norm(particle.omega)),
            })
    disp = particle.x - x0
    return {
        "slide_m": float(np.dot(disp, downslope)) if downslope is not None else 0.0,
        "displacement_m": float(np.linalg.norm(disp)),
        "speed_m_s": float(np.linalg.norm(particle.v)),
        "omega_rad_s": float(np.linalg.norm(particle.omega)),
        "n_steps": n_steps,
        "dt": dt,
        "series": series,
    }


def _incline(theta_deg: float, mu: float, law: str, mu_r: float = 0.0,
             t_end: float = 0.08, dt: float = 1e-5,
             samples: int = 0) -> Dict[str, Any]:
    system, particle = build_incline_case(theta_deg, mu, mu_r)
    _apply_law(system, law)
    _, downslope = incline_basis(theta_deg)
    out = _run(system, particle, dt, t_end, samples=samples, downslope=downslope)
    out.update({"theta_deg": theta_deg, "mu": mu, "mu_r": mu_r, "law": law})
    return out


# ---------------------------------------------------------------------------
# Case 1: creep below the friction limit
# ---------------------------------------------------------------------------
def case_creep(theta_deg: float = 4.0, mu: float = 0.30,
               t_end: float = 0.16, dt: float = 1e-5) -> Dict[str, Any]:
    rows = []
    for law in ("single", "multi"):
        r = _incline(theta_deg, mu, law, t_end=t_end, dt=dt, samples=16)
        # linear creep rate from the last two samples
        tail = r["series"][-2:]
        rate = (tail[1]["slide_m"] - tail[0]["slide_m"]) / (tail[1]["t"] - tail[0]["t"])
        r["creep_rate_m_s"] = rate
        rows.append(r)
    single, multi = rows
    return {
        "case": "creep",
        "theta_deg": theta_deg,
        "mu": mu,
        "atan_mu_deg": math.degrees(math.atan(mu)),
        "t_end": t_end,
        "dt": dt,
        "rows": rows,
        "slide_ratio_single_over_multi": (
            single["slide_m"] / multi["slide_m"] if multi["slide_m"] > 0 else float("inf")),
        "omega_ratio_single_over_multi": (
            single["omega_rad_s"] / multi["omega_rad_s"] if multi["omega_rad_s"] > 0 else float("inf")),
    }


# ---------------------------------------------------------------------------
# Case 2: critical angle by bisection
# ---------------------------------------------------------------------------
def case_critical(mu: float, law: str, lo: float = 2.0, hi: float = 60.0,
                  n_bisect: int = 9, slide_tol_m: float = 1.0e-4,
                  t_end: float = 0.08, dt: float = 1e-5) -> Dict[str, Any]:
    probes = []
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        r = _incline(mid, mu, law, t_end=t_end, dt=dt)
        probes.append({"theta_deg": mid, "slide_m": r["slide_m"]})
        if r["slide_m"] > slide_tol_m:
            hi = mid
        else:
            lo = mid
    theta_c = 0.5 * (lo + hi)
    theta_at = math.degrees(math.atan(mu))
    return {
        "case": "critical",
        "mu": mu,
        "law": law,
        "theta_c_deg": theta_c,
        "atan_mu_deg": theta_at,
        "rel_error_pct": 100.0 * (theta_c - theta_at) / theta_at,
        "slide_tol_m": slide_tol_m,
        "t_end": t_end,
        "dt": dt,
        "probes": probes,
    }


# ---------------------------------------------------------------------------
# Case 3: translation above the limit (no spurious resistance)
# ---------------------------------------------------------------------------
def case_translation(theta_deg: float = 30.0, mu: float = 0.30,
                     t_end: float = 0.02, dt: float = 1e-5) -> Dict[str, Any]:
    th = math.radians(theta_deg)
    a_analytic = 9.81 * (math.sin(th) - mu * math.cos(th))
    rows = []
    for law in ("single", "multi"):
        r = _incline(theta_deg, mu, law, t_end=t_end, dt=dt)
        # a = 2 s / t^2 from the measured down-slope displacement
        r["accel_measured_m_s2"] = 2.0 * r["slide_m"] / t_end**2
        r["accel_analytic_m_s2"] = a_analytic
        r["accel_rel_error_pct"] = 100.0 * (r["accel_measured_m_s2"] - a_analytic) / a_analytic
        rows.append(r)
    return {"case": "translation", "theta_deg": theta_deg, "mu": mu,
            "t_end": t_end, "dt": dt, "rows": rows}


# ---------------------------------------------------------------------------
# Case 4: tipping control (over-stabilisation guard)
# ---------------------------------------------------------------------------
def _tall_block_case(theta_deg: float, mu: float, half_width: float = 0.002,
                     half_height: float = 0.006) -> Any:
    """Rectangular block whose side faces are not square, resting on the incline."""
    from tensordem.friction_polyhedron_real_verification import _rotation_y

    normal, _ = incline_basis(theta_deg)
    rot = _rotation_y(theta_deg)
    a, h = half_width, half_height
    local = np.array([[sx * a, sy * a, sz * h]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    center = normal * (h + 1e-7 + 1e-9)
    verts = local @ rot.T + center
    cfg = FrictionPolyhedronConfig(k_n=1.0e5, gamma_n=20.0, k_t=1.0e5,
                                   mu=mu, mu_r=0.0, k_r=1.0e3, density=1500.0)
    particle = PolyhedronParticle.from_polytope(
        verts, density=1500.0, position=center, label="tall_block")
    system = PolyhedronDEMSystem([particle], cfg, gravity=np.array([0.0, 0.0, -9.81]))
    system.add_wall(normal, 0.0)
    return system, particle, normal


def case_tipping(theta_deg: float = 30.0, mu: float = 0.90,
                 t_end: float = 0.06, dt: float = 1e-5) -> Dict[str, Any]:
    rows = []
    tan_ratio = 0.002 / 0.006  # half_width / half_height -> tips above atan(1/3)
    for law in ("single", "multi"):
        system, particle, normal = _tall_block_case(theta_deg, mu)
        _apply_law(system, law)
        r = _run(system, particle, dt, t_end)
        # tilt of the block's long (local +z) axis away from the wall normal
        verts = np.asarray(particle.shape.vertices, dtype=np.float64)
        axis = verts[4:].mean(axis=0) - verts[:4].mean(axis=0)
        axis = axis / np.linalg.norm(axis)
        r["tilt_deg"] = float(math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, normal)))))))
        r["law"] = law
        rows.append(r)
    return {"case": "tipping", "theta_deg": theta_deg, "mu": mu,
            "tip_threshold_deg": math.degrees(math.atan(tan_ratio)),
            "t_end": t_end, "dt": dt, "rows": rows}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _merge(path: Path, payload: Dict[str, Any]) -> None:
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[payload["case"] + "_" + str(payload.get("mu", "")) + "_" +
         str(payload.get("law", ""))] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=["creep", "critical", "translation", "tipping", "collect"])
    ap.add_argument("--mu", type=float, default=0.30)
    ap.add_argument("--law", default="both", choices=["single", "multi", "both"])
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    jpath = args.outdir / "wall_multipoint_contact.json"

    if args.case == "creep":
        payload = case_creep()
        payload["t_end"] = payload["t_end"]
        _merge(jpath, payload)
        print(json.dumps({k: payload[k] for k in
                          ("theta_deg", "mu", "slide_ratio_single_over_multi",
                           "omega_ratio_single_over_multi")}, indent=2))
        for r in payload["rows"]:
            print(f"  {r['law']:6s} slide={r['slide_m']*1e6:9.3f} um  "
                  f"v={r['speed_m_s']:.3e} m/s  omega={r['omega_rad_s']:.3e} rad/s  "
                  f"rate={r['creep_rate_m_s']:.3e} m/s")
    elif args.case == "critical":
        laws = ["single", "multi"] if args.law == "both" else [args.law]
        for law in laws:
            payload = case_critical(args.mu, law, t_end=args.t_end or 0.08)
            _merge(jpath, payload)
            print(f"mu={payload['mu']:.2f} law={law:6s} theta_c={payload['theta_c_deg']:.2f} deg "
                  f"atan(mu)={payload['atan_mu_deg']:.2f} deg err={payload['rel_error_pct']:+.2f} %")
    elif args.case == "translation":
        payload = case_translation()
        _merge(jpath, payload)
        for r in payload["rows"]:
            print(f"  {r['law']:6s} a={r['accel_measured_m_s2']:.4f} m/s2 "
                  f"(analytic {r['accel_analytic_m_s2']:.4f}, err {r['accel_rel_error_pct']:+.2f} %)")
    elif args.case == "tipping":
        payload = case_tipping()
        _merge(jpath, payload)
        for r in payload["rows"]:
            print(f"  {r['law']:6s} tilt={r['tilt_deg']:6.2f} deg "
                  f"displacement={r['displacement_m']*1e3:.3f} mm "
                  f"omega={r['omega_rad_s']:.3f} rad/s")
    else:  # collect
        data = json.loads(jpath.read_text())
        csv_path = args.outdir / "wall_multipoint_contact.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "law", "mu", "theta_deg", "value", "unit"])
            for key, payload in sorted(data.items()):
                if payload["case"] == "creep":
                    for r in payload["rows"]:
                        w.writerow(["creep", r["law"], payload["mu"], payload["theta_deg"],
                                    f"{r['slide_m']:.6e}", "m"])
                        w.writerow(["creep_rate", r["law"], payload["mu"], payload["theta_deg"],
                                    f"{r['creep_rate_m_s']:.6e}", "m/s"])
                elif payload["case"] == "critical":
                    w.writerow(["critical_angle", payload["law"], payload["mu"],
                                f"{payload['theta_c_deg']:.4f}", "deg", ""])
                elif payload["case"] == "translation":
                    for r in payload["rows"]:
                        w.writerow(["accel", r["law"], payload["mu"], payload["theta_deg"],
                                    f"{r['accel_measured_m_s2']:.6f}", "m/s2"])
        print(f"wrote {csv_path} and {jpath}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())