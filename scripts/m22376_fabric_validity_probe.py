#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION · micro-mechanical validity probe (m22376).

Why this probe exists
---------------------
Nine rounds of this case (`docs/CASE-DEM-BIAXIAL-FRICTION_R*.md`) have measured a
critical-state friction angle of only ~14 deg against the case-pool reference band
(Rowe 1962 ``arctan(mu) = 28.81 deg`` for ``mu = 0.55``; Bolton 1986 ``33 deg``).
Every round so far changed the *contact operator* (regularisation, rolling
surrogate) or the *protocol* (servo -> isochoric).  None of them checked whether
the specimen itself sits in a numerically valid contact regime, i.e. whether the
contact overlap is small enough for the linear-spring contact law and the
"contact point" geometry to mean anything.

The anchor geometry runs r = 0.75 m discs with ``contact_stiffness = 1e7 N/m``
under ``sigma_3 = 100 kPa``.  The mean normal contact force in a 2D packing is
``F ~ sigma * d * t`` = 1.5e5 N, so the linear law implies ``delta = F / kn`` =
1.5e-2 m, i.e. ``delta / d ~ 1 %`` -- right at the upper edge of what is
acceptable, and this scales linearly with kn.  The probe therefore measures, in
one go and for a pre-declared ladder of ``kn``:

1. **Geometric validity**: overlap ratio ``delta / d`` (mean, 95th pct, max),
   coordination number ``Z``, contact count;
2. **Fabric kinematics**: contact-normal anisotropy ``a_c`` and its orientation,
   normal-force anisotropy ``a_f`` (linear-law surrogate ``F_n = kn delta``),
   plus the 2D stress-force-fabric closure ``sin(phi_SFF) = (a_c + a_f) / 2``
   (Rothenburg & Bathurst 1989) to check whether the measured stress ratio is
   consistent with the measured *micro-structure* rather than an accounting
   artefact;
3. **Macro response**: critical state through the pre-existing four-gate locator
   (`tensordem.biaxial_critical_state`, log strain measure, isochoric closure).

The point of the closure check is decisive: if ``sin(phi_measured)`` matches
``(a_c + a_f) / 2`` then the stress measurement is right and the low friction
angle is a genuine micro-structural outcome; if it does not, the stress
measurement (not the contact law) is the suspect.

Usage
-----
    TENSORDEM_THREADS=8 PYTHONPATH=src python scripts/m22376_fabric_validity_probe.py \
        --tag kn1e7 --kn 1e7 --rate 0.05 --strain-target 0.30

Every arm writes one JSON under
``artifacts/CASE-DEM-BIAXIAL-FRICTION/fabric_validity/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
    phi_deg_from_stresses,
    rowe_stress_dilatancy,
)
from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_SUBDIR = os.path.join("artifacts", "CASE-DEM-BIAXIAL-FRICTION",
                          "fabric_validity")

# Stiffness ladder: dimensionless stiffness kn / (p * d) at sigma ~ 100 kPa,
# d = 1.5 m is 6.7 / 67 / 667 / 6667.  Real DEM practice starts around 1e3.
STIFFNESS_LADDER = (1.0e8, 1.0e9)

# Rowe (1962) D = 0 limit, the case-pool reference band used since round 2.
ROWE_MU = 0.55


def pairwise_census(pos: np.ndarray, radius: float):
    """Return per-contact geometry (indices, overlap, unit normal)."""
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    two_r = 2.0 * radius
    iu, ju = np.triu_indices(n, k=1)
    dij = d[iu, ju]
    mask = dij < two_r
    i, j, dij = iu[mask], ju[mask], dij[mask]
    vec = pos[j] - pos[i]
    dist = np.maximum(dij, 1e-12)
    nrm = vec / dist[:, None]
    overlap = two_r - dij
    return {"i": i, "j": j, "dist": dij, "normal": nrm, "overlap": overlap,
            "n_particles": n, "n_contacts": int(i.size)}


def tensor_anisotropy(normals: np.ndarray, weights: np.ndarray | None = None):
    """2D fabric/force tensor anisotropy magnitude and major-axis orientation.

    ``F_ij = <w n_i n_j> / <w>`` with ``F = (1/2)(I + a [[cos2t, sin2t],
    [sin2t, -cos2t]])``, so ``a = sqrt((F_xx - F_yy)^2 + (2 F_xy)^2)`` and the
    major principal axis sits at ``t``.  For ``weights = None`` this is the
    contact-normal (fabric) tensor; with ``weights = F_n`` it is the normal-force
    tensor.
    """
    if normals.size == 0:
        return float("nan"), float("nan"), np.zeros((2, 2))
    w = np.ones(normals.shape[0]) if weights is None else np.asarray(weights, float)
    tot = float(np.sum(w))
    if tot <= 0.0:
        return float("nan"), float("nan"), np.zeros((2, 2))
    f = np.einsum("k,ki,kj->ij", w / tot, normals, normals)
    a = math.hypot(f[0, 0] - f[1, 1], 2.0 * f[0, 1])
    theta = 0.5 * math.atan2(2.0 * f[0, 1], f[0, 0] - f[1, 1])
    return float(a), float(math.degrees(theta)), f


def micro_diagnostics(snapshot: dict, radius: float, kn: float,
                      sigma_v: float, sigma_h: float) -> dict:
    pos = np.asarray(snapshot["positions"], dtype=float)
    census = pairwise_census(pos, radius)
    overlap = census["overlap"]
    n_p = census["n_particles"]
    n_c = census["n_contacts"]
    diameter = 2.0 * radius
    # coordination number: each contact shared by two particles
    z = 2.0 * n_c / n_p if n_p else float("nan")
    ov = overlap / diameter
    a_c, theta_c, f_c = tensor_anisotropy(census["normal"])
    fn = kn * overlap                       # linear-law normal force surrogate
    a_f, theta_f, f_f = tensor_anisotropy(census["normal"], fn)
    # virial stress from normal forces only (branch vector = 2 r n), 2D thickness 1
    area = float(np.ptp(pos[:, 0])) * float(np.ptp(pos[:, 1]))
    sigma_n = np.zeros((2, 2))
    if area > 0 and n_c:
        for k in range(n_c):
            nk = census["normal"][k]
            sigma_n += (-fn[k] * diameter) * np.outer(nk, nk)
        sigma_n /= area
    sv_n = -sigma_n[1, 1]
    sh_n = -sigma_n[0, 0]
    phi_sff = math.degrees(math.asin(max(-1.0, min(1.0, 0.5 * (a_c + a_f)))))
    phi_meas = phi_deg_from_stresses(sigma_v, sigma_h)
    return {
        "strain": float(snapshot["strain"]),
        "n_particles": int(n_p),
        "n_contacts": int(n_c),
        "coordination_number": z,
        "overlap_ratio_mean": float(np.mean(ov)) if ov.size else float("nan"),
        "overlap_ratio_p95": float(np.percentile(ov, 95)) if ov.size else float("nan"),
        "overlap_ratio_max": float(np.max(ov)) if ov.size else float("nan"),
        "overlap_abs_mean_m": float(np.mean(overlap)) if ov.size else float("nan"),
        "dimensionless_stiffness_kn_over_pd": (
            float(kn / (sigma_h * diameter)) if sigma_h else float("nan")),
        "fabric_anisotropy_a_c": a_c,
        "fabric_major_axis_deg": theta_c,
        "normal_force_anisotropy_a_f": a_f,
        "force_major_axis_deg": theta_f,
        "sin_phi_stress_force_fabric": math.sin(math.radians(phi_sff)),
        "phi_stress_force_fabric_deg": phi_sff,
        "phi_from_measured_stress_deg": phi_meas,
        "sff_closure_relative_error": (
            abs(phi_sff - phi_meas) / phi_meas if phi_meas else float("nan")),
        "normal_force_virial_sigma_v_Pa": float(sv_n),
        "normal_force_virial_sigma_h_Pa": float(sh_n),
        "normal_force_virial_phi_deg": phi_deg_from_stresses(sv_n, sh_n),
        "contacts_per_particle_vs_isostatic": z - 3.0,
    }


def _safe(fn):
    """Return fn() or None -- a diagnostic must never invalidate a run record."""
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}


def _overrides(args) -> dict:
    ov = {"freeze_bond_fraction": 0.0, "friction_coefficient": args.mu,
          "contact_stiffness": args.kn}
    if args.vreg is not None:
        ov["velocity_regularisation"] = float(args.vreg)
    return ov


def mobilisation_summary(mob: list, strain_lo: float) -> dict:
    """Mean absolute mobilisation tanh(v_t/v_reg) over the critical window."""
    if not mob:
        return {}
    rows = [m for m in mob if float(m.get("axial_strain", 0.0)) >= strain_lo]
    if not rows:
        rows = mob
    keys = sorted({k for r in rows for k in r
                   if isinstance(r.get(k), (int, float))
                   and k not in ("axial_strain", "step")})
    out = {k: float(np.mean([float(r[k]) for r in rows if k in r]))
           for k in keys}
    out["n_samples"] = len(rows)
    out["strain_lo"] = strain_lo
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kn", type=float, default=1.0e7,
                    help="contact_stiffness [N/m]")
    ap.add_argument("--mu", type=float, default=0.55)
    ap.add_argument("--rate", type=float, default=0.05, help="axial strain rate [1/s]")
    ap.add_argument("--strain-target", type=float, default=0.30, help="log strain")
    ap.add_argument("--columns", type=int, default=10)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--consolidation-steps", type=int, default=600)
    ap.add_argument("--snapshot-strain", type=float, default=0.05)
    ap.add_argument("--confining", type=float, default=100.0e3)
    ap.add_argument("--vreg", type=float, default=None,
                    help="velocity_regularisation [m/s]; None = RidgeMaterial default")
    ap.add_argument("--ladder-index", type=int, default=None,
                    help="pick stiffness arm from STIFFNESS_LADDER (1-based); "
                         "overrides --kn and --tag")
    ap.add_argument("--mobilisation-every", type=int, default=0,
                    help="sample the friction mobilisation meter every N steps")
    ap.add_argument("--min-strain", type=float, default=0.2,
                    help="critical-state window floor (default 0.2, as declared)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.ladder_index is not None:
        kn = STIFFNESS_LADDER[args.ladder_index - 1]
        args.kn = kn
        args.tag = f"kn{kn:.0e}"

    out_dir = args.out_dir or os.path.join(REPO, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    radius = 0.75
    started = time.time()
    res = run_biaxial_shear_band(
        confining_stress=args.confining, columns=args.columns, rows=args.rows,
        radius=radius, strain_rate=args.rate, strain_target=args.strain_target,
        consolidation_steps=args.consolidation_steps,
        snapshot_strain=args.snapshot_strain, wall_friction=0.0,
        lateral_control="isochoric", mobilisation_every=args.mobilisation_every,
        record_snapshots=True,
        material_overrides=_overrides(args))
    elapsed = time.time() - started

    cs = critical_state_from_stress_path(
        res["stress_path"], sigma3_target=args.confining,
        strain_measure="log", boundary_control="isochoric",
        min_strain=args.min_strain)

    snaps = res.get("snapshots") or []
    micro = []
    for snap in snaps:
        # stress at that snapshot: use the path record nearest the strain
        micro.append(micro_diagnostics(snap, radius, args.kn, math.nan, math.nan))
    # recompute the final-snapshot micro block with the critical-state stresses
    if snaps:
        final = micro_diagnostics(snaps[-1], radius, args.kn,
                                  cs.q_cv_Pa + cs.sigma_h_cv_Pa, cs.sigma_h_cv_Pa)
        micro[-1] = final

    rowe = math.degrees(math.atan(ROWE_MU))
    record = {
        "tag": args.tag,
        "config": {
            "kn": args.kn, "mu": args.mu, "rate": args.rate,
            "vreg": (args.vreg if args.vreg is not None else 0.01),
            "strain_target": args.strain_target, "columns": args.columns,
            "rows": args.rows, "radius": radius, "confining": args.confining,
            "consolidation_steps": args.consolidation_steps,
            "lateral_control": "isochoric", "freeze_bond_fraction": 0.0,
            "wall_friction": 0.0,
        },
        "loading": res["loading"],
        "specimen": res["specimen"],
        "critical_state": cs.as_dict(),
        "phi_cv_deg": cs.phi_cv_deg,
        "rowe_reference_deg": rowe,
        "rowe_relative_error": (abs(cs.phi_cv_deg - rowe) / rowe
                                if rowe else float("nan")),
        "rowe_stress_dilatancy": _safe(lambda: rowe_stress_dilatancy(
            res["stress_path"], mu=args.mu, min_strain=args.min_strain)),
        "micro": micro,
        "mobilisation_critical_window": mobilisation_summary(
            res.get("mobilisation") or [], cs.window[0]),
        "mobilisation_min_strain": cs.window[0],
        "elapsed_s": elapsed,
    }
    path = os.path.join(out_dir, f"arm_{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else str(o))

    last = micro[-1] if micro else {}
    print(f"== {args.tag} kn={args.kn:.1e} elapsed={elapsed:.0f}s "
          f"steps={res['loading']['steps']} I={res['loading']['inertial_number']:.2e}")
    print(f"   phi_cv={cs.phi_cv_deg:.2f} deg (Rowe {rowe:.2f}) "
          f"err={100.0 * record['rowe_relative_error']:.2f}%")
    print(f"   overlap/d mean={last.get('overlap_ratio_mean', float('nan')):.4f} "
          f"max={last.get('overlap_ratio_max', float('nan')):.4f} "
          f"Z={last.get('coordination_number', float('nan')):.2f} "
          f"a_c={last.get('fabric_anisotropy_a_c', float('nan')):.3f} "
          f"a_f={last.get('normal_force_anisotropy_a_f', float('nan')):.3f}")
    print(f"   phi_SFF={last.get('phi_stress_force_fabric_deg', float('nan')):.2f} deg "
          f"vs phi_stress={last.get('phi_from_measured_stress_deg', float('nan')):.2f} deg")
    print(f"   wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())