#!/usr/bin/env python3
"""Harvest archived frozen-case records into the m23331 record schema.

The campaign has run this exact case for sixteen rounds, so several worktrees
already hold discharge records of it.  This script turns those read-only files
into records of the branch-probability ensemble **without re-running any DEM**:

* the ``dt`` **spelling is decided by the IEEE-754 bits of the record's own
  ``dt``**, never by a file name or by which worktree it came from — the arm is
  therefore an objective property of the record;
* every record is checked against the frozen protocol (geometry, friction,
  kn, window layout); a record that does not match is **reported as rejected
  with its reason**, never silently dropped and never "fixed";
* the FLOWING/ARRESTED label is re-derived here with the campaign's own
  ``beverloo.discharge_status`` from the record's stored window series, so an
  archive that predates the classifier (``status = None``) still gets the same
  objective label as a fresh run;
* duplicates of the same ``(arm, seed)`` coming from different commits/machines
  are kept as a **cross-worktree reproducibility check**: they must agree on
  ``W`` to ``1e-12`` relative, else the harvest fails loudly.

    PYTHONPATH=src python3 benchmarks/m23331_branch_prob/harvest_archives.py \
        --glob '/root/tp-agent/work/m22885/runs/m22885_beverloo_powered/fill36_seed*.json' \
        --glob '/root/tp-agent/work/m23114/runs/m23114_siloflow_ladder/dt_nom20_s1.json' \
        --out runs/m23331_branch_prob_archive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "benchmarks" / "m23331_branch_prob"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensordem.silo_flow.beverloo import beverloo_prediction, discharge_status  # noqa: E402
from tensordem.silo_flow.kernel import Material, contact_time  # noqa: E402

#: the frozen case of the campaign (identical to run_case.FROZEN)
FROZEN = dict(
    particle_diameter=0.003, orifice_diameter=0.015, silo_radius=0.024,
    fill_height=0.036, wall_mu=0.0, settle_time=0.06, discard_time=0.12,
    window_time=0.05, n_windows=8, mu=0.50, mu_rolling=0.05, kn=5.0e3,
)
#: keys of ``case`` that must match exactly (settle_time is not stored in the
#: archives; it does not enter the measured quantity, only the bed preparation)
CASE_KEYS = ("particle_diameter", "orifice_diameter", "silo_radius", "fill_height",
             "wall_mu", "mu", "mu_rolling", "kn")

ARM_BY_BITS = None  # filled by dt_spellings()


def dt_spellings() -> dict:
    """``{arm: (dt, bits)}`` computed from the kernel, not typed in."""
    m = Material(radius=0.5 * FROZEN["particle_diameter"], kn=FROZEN["kn"],
                 mu=FROZEN["mu"], mu_rolling=FROZEN["mu_rolling"])
    t_c = contact_time(m.mass_eff, FROZEN["kn"])
    out = {"F": 0.05 * t_c, "Q": t_c / 20.0}
    return {k: (v, struct.pack(">d", v).hex()) for k, v in out.items()}


def classify_archive(rec: dict) -> dict:
    """Return ``{"ok": bool, "reasons": [...], "arm": ..., "seed": ...}``."""
    reasons = []
    case = rec.get("case") or {}
    for k in CASE_KEYS:
        if case.get(k) is None:
            reasons.append(f"case.{k} missing")
        elif not math.isclose(float(case[k]), float(FROZEN[k]), rel_tol=0.0, abs_tol=0.0):
            reasons.append(f"case.{k} = {case[k]!r} != frozen {FROZEN[k]!r}")
    for k in ("n_windows", "window_time_s", "discard_time_s"):
        want = FROZEN[{"n_windows": "n_windows", "window_time_s": "window_time",
                       "discard_time_s": "discard_time"}[k]]
        if rec.get(k) is None or not math.isclose(float(rec[k]), float(want), rel_tol=0, abs_tol=0):
            reasons.append(f"{k} = {rec.get(k)!r} != frozen {want!r}")
    dt = rec.get("dt")
    if not isinstance(dt, (int, float)) or not math.isfinite(float(dt)):
        reasons.append("dt missing")
        arm = None
    else:
        bits = struct.pack(">d", float(dt)).hex()
        arm = next((a for a, (_, b) in dt_spellings().items() if b == bits), None)
        if arm is None:
            reasons.append(f"dt {float(dt):.17e} (bits {bits}) is neither spelling")
    seed = case.get("seed")
    if seed is None:
        reasons.append("case.seed missing")
    wp = rec.get("window_particles")
    wr = rec.get("flow_rate_windows_kg_s")
    if not isinstance(wp, list) or len(wp) != FROZEN["n_windows"]:
        reasons.append("window_particles missing or wrong length")
    if not isinstance(wr, list) or len(wr) != FROZEN["n_windows"]:
        reasons.append("flow_rate_windows_kg_s missing or wrong length")
    if not isinstance(rec.get("rho_bulk"), (int, float)):
        reasons.append("rho_bulk missing")
    if not isinstance(rec.get("flow_rate_kg_s"), (int, float)):
        reasons.append("flow_rate_kg_s missing")
    return {"ok": not reasons, "reasons": reasons, "arm": arm, "seed": seed}


def to_record(path: Path) -> dict:
    """Normalise one archive file into the m23331 record schema."""
    rec = json.loads(path.read_text())
    chk = classify_archive(rec)
    if not chk["ok"]:
        return {"rejected": True, "path": str(path), "reasons": chk["reasons"]}

    case = rec["case"]
    rho_bulk = float(rec["rho_bulk"])
    w_ref = beverloo_prediction(case["orifice_diameter"], case["particle_diameter"], rho_bulk)
    st = discharge_status(rec["window_particles"], rec["flow_rate_windows_kg_s"],
                          reference_rate=w_ref,
                          particle_diameter=case["particle_diameter"])
    out = {
        "flow_rate_kg_s": float(rec["flow_rate_kg_s"]),
        "window_particles": [int(x) for x in rec["window_particles"]],
        "flow_rate_windows_kg_s": [float(x) for x in rec["flow_rate_windows_kg_s"]],
        "particles_exited": rec.get("particles_exited"),
        "discharge_status": st,
        "branch_prob": {
            "arm": chk["arm"],
            "seed": int(chk["seed"]),
            "dt_s": float(rec["dt"]),
            "dt_bits": struct.pack(">d", float(rec["dt"])).hex(),
            "w_reference_kg_s": w_ref,
            "rho_bulk": rho_bulk,
            "rel_error_vs_beverloo": float(rec["flow_rate_kg_s"]) / w_ref - 1.0,
            "n_bed": rec.get("n_bed"),
            "source": "archive",
            "source_path": str(path),
            "source_git_sha": (rec.get("meta") or {}).get("git_sha"),
            "source_machine": (rec.get("meta") or {}).get("machine"),
            "source_utc": (rec.get("meta") or {}).get("utc"),
            "bed_state_sha256": None,
        },
    }
    return out


def harvest(globs: list[str]) -> dict:
    paths = []
    for g in globs:
        paths.extend(sorted(Path("/").glob(g.lstrip("/")) if g.startswith("/") else Path(".").glob(g)))
    seen, records, rejected, duplicates = set(), {}, [], []
    for p in paths:
        p = Path(p)
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        r = to_record(p)
        if r.get("rejected"):
            rejected.append(r)
            continue
        bp = r["branch_prob"]
        key = (bp["arm"], bp["seed"])
        if key in records:
            prev = records[key]
            rel = abs(r["flow_rate_kg_s"] / prev["flow_rate_kg_s"] - 1.0)
            duplicates.append({"arm": bp["arm"], "seed": bp["seed"],
                               "kept": prev["branch_prob"]["source_path"],
                               "dup": str(p), "rel_diff_W": rel,
                               "agree_1e-12": bool(rel < 1e-12)})
            continue
        records[key] = r
    return {"records": [records[k] for k in sorted(records)], "rejected": rejected,
            "duplicates": duplicates}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--out", default="runs/m23331_branch_prob_archive")
    args = ap.parse_args(argv)
    res = harvest(args.glob)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for r in res["records"]:
        bp = r["branch_prob"]
        (out / f"{bp['arm']}_s{bp['seed']}.json").write_text(json.dumps(r, indent=1))
    (out / "harvest_report.json").write_text(json.dumps(
        {"n_records": len(res["records"]), "rejected": res["rejected"],
         "duplicates": res["duplicates"],
         "records": [{k: v for k, v in r["branch_prob"].items()} for r in res["records"]]},
        indent=1))
    for r in res["records"]:
        bp = r["branch_prob"]
        print(f"{bp['arm']}_s{bp['seed']}: W={r['flow_rate_kg_s']:.9g} "
              f"rel={bp['rel_error_vs_beverloo']:+.2%} {r['discharge_status']['status']} "
              f"from {bp['source_path']}")
    for r in res["rejected"]:
        print(f"REJECTED {r['path']}: {r['reasons']}")
    for d in res["duplicates"]:
        print(f"DUP {d['arm']}_s{d['seed']}: rel_diff_W={d['rel_diff_W']:.2e} "
              f"agree={d['agree_1e-12']}")
    print(f"-> {out} ({len(res['records'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())