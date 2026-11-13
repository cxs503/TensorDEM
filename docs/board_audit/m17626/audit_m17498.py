"""Board audit of m17498 (CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE, TensorDEM).

Independent re-derivation of the three deliverables the mission claims
(run artefacts / benchmark ref-vs-got / repro command) against the only
authority the board accepts: the benchmark case row and the raw artefacts.

What this script does (no writes outside its own output paths):

  1. provenance  -- sha256 of every delivered artefact, plus a scan for the
     raw per-specimen fields (npz) that would let a third party re-apply the
     measurement operators;
  2. arithmetic  -- re-parse the primary log (logs/r3_ensemble.log) and the
     per-specimen table in the artefact JSON, recompute the ensemble mean,
     sample std, 95 % CI and relative error from scratch, and compare with the
     published numbers;
  3. referee     -- compare ref/got against the registered benchmark case
     (tensor_benchmark_cases.anchor_slug = CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE)
     under the owner 15 % rule, and report the 3 % bar separately;
  4. operator    -- synthetic known-angle recovery of the production Radon
     focus on the exact 24 x 48 case lattice (the crux: is 53.8 deg a
     physical measurement or an estimator artefact?).

Usage:
    PYTHONPATH=src python docs/board_audit/m17626/audit_m17498.py \
        [--m17498 /root/tp-agent/work/m17498] [--skip-operator] [--out ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

THETA_REF_DEG = 60.0          # tensor_benchmark_cases id=59, D&V 2004 midpoint
TOLERANCE_REL = 0.15          # "<15% (owner standard)"
LIT_BAND = (55.0, 60.0)       # Desrues & Viggiani 2004 dense Hostun sand

LOG_RE = re.compile(
    r"^\[done\]\s+(?P<name>\S+)\s+radon=(?P<radon>[-+0-9.eEnan]+)\s+"
    r"tensor=(?P<tensor>[-+0-9.eEnan]+)\s+resolved=(?P<resolved>True|False)"
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(values):
    """Mean / sample std / 95 % CI (Student t, df = n-1) -- no library call."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}[n - 1]
    half = t975 * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95_half": half}


def provenance(root):
    art = os.path.join(root, "artifacts", "biaxial_shear_band")
    log = os.path.join(root, "logs", "r3_ensemble.log")
    files = {}
    for path in (os.path.join(art, "r3_ensemble.json"),
                 os.path.join(art, "r3_ensemble.png"),
                 os.path.join(art, "r3_estimator_audit.json"),
                 os.path.join(art, "r3_estimator_audit.png"),
                 log):
        files[os.path.relpath(path, root)] = (
            {"exists": True, "sha256": _sha256(path), "bytes": os.path.getsize(path)}
            if os.path.exists(path) else {"exists": False})
    npz = []
    for dirpath, _dirnames, filenames in os.walk(art):
        npz.extend(os.path.relpath(os.path.join(dirpath, f), root)
                   for f in filenames if f.endswith(".npz"))
    cache = os.path.join(art, "r3_cache")
    return {
        "artifacts": files,
        "raw_field_snapshots_npz": npz,
        "per_specimen_cache_dir": {
            "path": os.path.relpath(cache, root),
            "exists": os.path.isdir(cache),
            "entries": len(os.listdir(cache)) if os.path.isdir(cache) else 0,
        },
        "re_derivable_from_raw_fields": bool(npz),
    }


def arithmetic(root):
    log_path = os.path.join(root, "logs", "r3_ensemble.log")
    ens = json.load(open(os.path.join(
        root, "artifacts", "biaxial_shear_band", "r3_ensemble.json")))
    parsed = []
    with open(log_path) as fh:
        for line in fh:
            m = LOG_RE.match(line.strip())
            if m:
                parsed.append({k: (float(v) if k in ("radon", "tensor") else v)
                               for k, v in m.groupdict().items()
                               if k in ("name", "radon", "tensor", "resolved")})
    anchor = [row for row in parsed if row["name"].startswith("anchor_")]
    radon = [row["radon"] for row in anchor]
    tensor = [row["tensor"] for row in anchor]
    log_stats = _stats(radon)
    log_stats_t = _stats(tensor)

    table = {row["name"]: row for row in ens.get("specimens", [])
             if row.get("ensemble_member")}
    mismatches = []
    for row in anchor:
        pub = table.get(row["name"])
        if pub is None:
            mismatches.append({"specimen": row["name"], "issue": "missing in JSON"})
            continue
        for key, field in (("radon", "theta_radon_mean_deg"), ("tensor", "theta_tensor_mean_deg")):
            got = pub.get(field)
            if got is None or abs(float(got) - row[key]) > 5e-3:
                mismatches.append({"specimen": row["name"], "field": field,
                                   "log": row[key], "json": got})

    published = ens["verdict"]["ensemble_radon"]
    recomputed_rel = abs(log_stats["mean"] - THETA_REF_DEG) / THETA_REF_DEG
    return {
        "log_lines_parsed": len(parsed),
        "anchor_specimens_in_log": len(anchor),
        "recomputed_radon": log_stats,
        "published_radon": {"n": published["n"], "mean": published["mean_deg"],
                            "std": published["std_deg"],
                            "ci95_half": published["ci95_half_width_deg"],
                            "rel_error": published["relative_error"]},
        "delta_mean_deg": log_stats["mean"] - published["mean_deg"],
        "delta_std_deg": log_stats["std"] - published["std_deg"],
        "delta_ci95_half_deg": log_stats["ci95_half"] - published["ci95_half_width_deg"],
        "recomputed_rel_error": recomputed_rel,
        "per_specimen_log_vs_json_mismatches": mismatches,
        "recomputed_tensor": log_stats_t,
        "specimens_completed_vs_planned": {
            "log_rows": len(parsed),
            "declared_completed": ens["verdict"]["provenance"]["completed_specimens"],
            "declared_planned": ens["verdict"]["provenance"]["planned_specimens"],
            "anchor_members": len(anchor),
        },
    }


def referee(arith):
    got = arith["recomputed_radon"]["mean"]
    err = abs(got - THETA_REF_DEG) / THETA_REF_DEG
    return {
        "case": "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE",
        "registered_row": {"tensor_benchmark_cases.id": 59},
        "metric": "shear band inclination vs anchor (deg)",
        "ref_deg": THETA_REF_DEG,
        "ref_source": "Desrues & Viggiani 2004 IJNAMG 28:279-321 (55-60 deg, midpoint 60)",
        "got_deg": got,
        "abs_error_deg": abs(got - THETA_REF_DEG),
        "rel_error": err,
        "tolerance_rel": TOLERANCE_REL,
        "within_owner_15pct": bool(err <= TOLERANCE_REL),
        "within_3pct": bool(err <= 0.03),
        "vs_literature_lower_bound_55": abs(got - LIT_BAND[0]) / LIT_BAND[0],
        "ci95_deg": [arith["recomputed_radon"]["mean"]
                     - arith["recomputed_radon"]["ci95_half"],
                     arith["recomputed_radon"]["mean"]
                     + arith["recomputed_radon"]["ci95_half"]],
        "ci95_inside_tolerance_band": bool(
            arith["recomputed_radon"]["mean"] - arith["recomputed_radon"]["ci95_half"]
            > THETA_REF_DEG * (1 - TOLERANCE_REL)),
    }


def operator_check(angles=(45.0, 60.0, 70.0), slip=0.02):
    """Synthetic known-angle recovery of the production Radon focus."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tensordem.biaxial_band_diagnostics import synthetic_band_configuration
    from tensordem.biaxial_shear_band import _make_specimen, measure_band
    from tensordem.ridge_dem import RidgeMaterial

    material = RidgeMaterial(particle_radius=0.75, freeze_bond_fraction=0.35,
                             seed=20261031)
    assembly, geometry = _make_specimen(
        material, columns=24, rows=48, confining_stress=100.0e3, jitter=0.03,
        servo_gain=3.0e-4, servo_velocity_limit=2.0, wall_friction=0.55)
    pos = assembly.pos.numpy()
    aspect =(float(np.ptp(pos[:, 0])) / float(np.ptp(pos[:, 1])))
    rows = []
    for angle in angles:
        clean = synthetic_band_configuration(pos, angle_deg=angle, slip=slip,
                                             band_width=1.5)
        measured = float(measure_band(pos, clean)["folded_angle_deg"])
        rows.append({"angle_true_deg": angle, "radon_deg": measured,
                     "error_deg": measured - angle, "slip_m": slip})
    return {
        "lattice": {"columns": 24, "rows": 48, "particles": int(pos.shape[0]),
                    "window_aspect_ptp": aspect},
        "note": "noise-free synthetic band injected into the case lattice, "
                "production Radon focus (binning='equal_count')",
        "rows": rows,
        "max_abs_error_deg": max(abs(r["error_deg"]) for r in rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m17498", default="/root/tp-agent/work/m17498")
    parser.add_argument("--skip-operator", action="store_true")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "audit_m17498.json"))
    args = parser.parse_args(argv)

    payload = {
        "mission_audited": 17498,
        "anchor": "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE",
        "repo": "TensorDEM",
        "auditor_mission": 17626,
        "worktree": args.m17498,
    }
    payload["provenance"] = provenance(args.m17498)
    payload["arithmetic"] = arithmetic(args.m17498)
    payload["referee"] = referee(payload["arithmetic"])
    if not args.skip_operator:
        payload["operator"] = operator_check()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(json.dumps({k: payload[k] for k in ("provenance", "referee")},
                     indent=2, default=float)[:3000])
    print("arithmetic:", json.dumps(payload["arithmetic"], indent=2, default=float)[:1500])
    if "operator" in payload:
        print("operator:", json.dumps(payload["operator"], indent=2, default=float))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())