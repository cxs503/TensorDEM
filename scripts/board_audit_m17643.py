#!/usr/bin/env python3
"""Board audit of m17643 (XVAL-DEM-ROUND2-METHOD-SELECTION): recompute, do not trust.

Everything is read from the mission commit through ``git cat-file`` (no reliance on the
mission worktree being alive) and every reported number is recomputed here from
``results.json``.  Referee rows come from ``tensor_benchmark_cases`` -- the only judge.

Output: JSON on stdout (also written by the caller), so zero hand-copy reaches the verdict.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys

REPO = "/root/tp-agent/work/m17671"
COMMIT = "e1cdcab1"
ANCHOR = "XVAL-DEM-ROUND2-METHOD-SELECTION"
MISSION = 17643
ENV = "/root/tensor-ecosystem/TensorPlatform/.env"

FILES = [
    "benchmarks/xval_dem_round2_method_selection.py",
    "benchmarks/xval_dem_round2_column_arms.py",
    "docs/XVAL_DEM_ROUND2_METHOD_SELECTION.md",
    "prereg/XVAL_DEM_ROUND2_METHOD_SELECTION.json",
    "src/tensordem/xval/__init__.py",
    "src/tensordem/xval/oss_rolling.py",
    "src/tensordem/xval/scenes.py",
    "src/tensordem/granular3d_core.py",
    "src/tensordem/granular_column/kernel.py",
    "src/tensordem/granular_column/axisym.py",
    "tests/test_xval_oss_rolling.py",
    "artifacts/xval_dem_round2_method_selection/results.json",
    "artifacts/xval_dem_round2_method_selection/table_main.md",
    "artifacts/xval_dem_round2_method_selection/traces.json",
    "docs/xval/oss_snapshot/PROVENANCE.json",
]


def blob(path: str) -> bytes:
    return subprocess.run(["git", "-C", REPO, "cat-file", "blob", f"{COMMIT}:{path}"],
                          check=True, capture_output=True).stdout


def dsn() -> str:
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                         "postgresql://")
    raise SystemExit("no DATABASE_URL")


def main() -> None:
    out: dict = {"anchor": ANCHOR, "mission": MISSION, "commit": COMMIT}

    # --- 1. inventory: do the claimed files exist in the commit, and are they non-trivial?
    inv = {}
    for p in FILES:
        try:
            b = blob(p)
            inv[p] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()[:16]}
        except subprocess.CalledProcessError:
            inv[p] = None
    out["inventory"] = inv

    figs = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--long", COMMIT,
                           "artifacts/xval_dem_round2_method_selection/figures"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    out["figures"] = [{"path": ln.split()[-1], "bytes": int(ln.split()[3])} for ln in figs]

    # --- 2. OSS snapshot provenance: sha256 must match the recorded upstream hashes
    prov = json.loads(blob("docs/xval/oss_snapshot/PROVENANCE.json"))
    snap = {}
    for name, meta in prov["files"].items():
        b = blob(meta["path"])
        snap[name] = {"bytes_ok": len(b) == meta["bytes"],
                      "sha256_ok": hashlib.sha256(b).hexdigest() == meta["sha256"]}
    out["oss_provenance"] = {"upstream": prov["repo"], "head": prov["head_commit"],
                             "files": snap,
                             "all_ok": all(v["bytes_ok"] and v["sha256_ok"]
                                           for v in snap.values())}

    # --- 3. recompute every headline number from results.json
    res = json.loads(blob("artifacts/xval_dem_round2_method_selection/results.json"))
    tol = res["tolerance"]
    out["tolerance_declared"] = tol

    e2 = []
    for row in res["spin_down"]["rows"]:
        t, th = row["t_stop"], row["t_stop_theory_s"]
        rec = {"mu_r": row["mu_r"]}
        for arm in ("kernel", "oss_epsd2", "oss_cdt"):
            rec[f"t_{arm}_ms"] = 1e3 * t[arm]
            rec[f"err_{arm}_vs_theory_pct"] = 1e2 * (t[arm] / th - 1.0)
        rec["err_kernel_vs_oss_epsd2_pct"] = 1e2 * (t["kernel"] / t["oss_epsd2"] - 1.0)
        rec["kernel_within_tol_vs_theory"] = abs(t["kernel"] / th - 1.0) <= tol
        rec["kernel_within_tol_vs_oss"] = abs(t["kernel"] / t["oss_epsd2"] - 1.0) <= tol
        e2.append(rec)
    out["E2_spin_down"] = {"rows": e2,
                           "n_points": len(e2),
                           "n_fail_vs_theory": sum(not r["kernel_within_tol_vs_theory"]
                                                   for r in e2),
                           "all_pass_vs_oss": all(r["kernel_within_tol_vs_oss"] for r in e2)}

    pair = res["pair_lever"]
    m, r_, fn = pair["mu_r"], 0.03, pair["fn_N"]
    I = 0.4 * (2500.0 * 4.0 / 3.0 * math.pi * r_ ** 3) * r_ ** 2          # (2/5) m r^2
    # independent closed form: t_half = 0.25 * w0 * I / (mu_r * reff * Fn)  (two-body)
    # closed form t_half = 0.25 * w0 * I / (mu_r * reff * Fn): invert it per arm and check
    # that one single w0 explains every arm (a fitted reference would not).
    w0 = {}
    for arm, (mu, reff) in {
            "kernel_reff_r": (m, r_),
            "kernel_mapped_mu_half": (m / 2.0, r_),
            "oss_epsd2_reff_r2": (m, r_ / 2.0),
            "oss_cdt_reff_r2": (m, r_ / 2.0)}.items():
        w0[arm] = pair["t_half_theory"][arm] / (0.25 * I / (mu * reff * fn))
    out["E3_pair_lever"] = {
        "t_half_ms": {k: 1e3 * v for k, v in pair["t_half"].items()},
        "implied_w0_rad_s": w0,
        "implied_w0_spread_pct": 1e2 * (max(w0.values()) / min(w0.values()) - 1.0),
        "halving_mu_in_the_kernel_reproduces_oss_pct":
            1e2 * (pair["t_half"]["kernel_mapped_mu_half"] / pair["t_half"]["oss_epsd2_reff_r2"] - 1.0),
        "lever_factor_analytic": pair["lever_factor_analytic"],
        "implied_mapping_mu_oss_over_mu_td":
            pair["t_half"]["oss_epsd2_reff_r2"] / pair["t_half"]["kernel_reff_r"],
        "implied_mapping_note": "t_half proportional to 1/(mu_r*reff) => mu_r must scale "
                                "as reff^-1; OSS lever is half, so mu_r,OSS = 2*mu_r,TD",
    }

    ident = res["kernel_identity"]
    out["E1_identity"] = {"max_abs_diff_omega": ident["max_abs_diff_omega"],
                          "below_1e-12": ident["max_abs_diff_omega"] < 1e-12}

    tor = res["torsion"]
    out["E5_torsion"] = {"w_end_kernel": tor["w_end_kernel_torsion_on"],
                         "w_end_production": tor["w_end_production_kernel"],
                         "w_end_oss_default": tor["w_end_oss_default"],
                         "production_matches_kernel":
                             abs(tor["w_end_production_kernel"]
                                 - tor["w_end_kernel_torsion_on"]) < 1e-9,
                         "spin_lost_pct_kernel": 1e2 * tor["spin_lost_fraction_kernel"]}

    dt = res["dt_sensitivity"]
    spread = dt["spread_fraction_over_dt_sweep"]
    out["E7_dt"] = {"spread_pct": {k: 1e2 * v for k, v in spread.items()},
                    "max_spread_pct": 1e2 * max(spread.values()),
                    "below_0p5pct": max(spread.values()) < 5e-3}

    ts = res["timestep"]
    out["E6_timestep"] = {"dt_over_rayleigh": ts["fraction_of_rayleigh"],
                          "oss_practice_band": ts["oss_recommended_band"],
                          "inside_oss_band": ts["oss_recommended_band"][0]
                          <= ts["fraction_of_rayleigh"] <= ts["oss_recommended_band"][1]}

    # --- 4. referee rows (the only judge) + merge state
    import psycopg
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT id, case_name, metric, tolerance, status FROM tensor_benchmark_cases"
                    " WHERE anchor_slug=%s OR case_name ILIKE %s ORDER BY id",
                    (ANCHOR, "%ROUND2%"))
        rows = cur.fetchall()
        out["referee_rows"] = [{"id": r[0], "case_name": r[1], "metric": r[2],
                                "tolerance": r[3], "status": r[4]} for r in rows]
        out["referee_row_count"] = len(rows)
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE 'XVAL-%'")
        out["referee_rows_other_xval_anchors"] = cur.fetchone()[0]

    anc = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", COMMIT, "origin/main"])
    out["merged_into_main"] = anc.returncode == 0
    out["branch"] = subprocess.run(["git", "-C", REPO, "branch", "--contains", COMMIT,
                                    "--format=%(refname:short)"],
                                   capture_output=True, text=True).stdout.split()

    if "--assert" in sys.argv:
        checks = {
            "all_15_declared_files_present": all(v is not None for v in inv.values()),
            "oss_snapshot_sha256_verified": out["oss_provenance"]["all_ok"],
            "E1_identity_below_1e-12": out["E1_identity"]["below_1e-12"],
            "E3_mapping_is_2x_mu_td": abs(out["E3_pair_lever"]
                                          ["implied_mapping_mu_oss_over_mu_td"] - 2.0) < 1e-6,
            "E3_theory_is_one_closed_form": out["E3_pair_lever"]["implied_w0_spread_pct"] < 1e-6,
            "E2_all_points_within_15pct_of_oss": out["E2_spin_down"]["all_pass_vs_oss"],
            "E7_dt_spread_below_0p5pct": out["E7_dt"]["below_0p5pct"],
            "E5_production_hits_the_defect": out["E5_torsion"]["production_matches_kernel"],
        }
        out["assertions"] = checks
        json.dump(out, sys.stdout, indent=1, default=str)
        print()
        for k, ok in checks.items():
            print(("PASS " if ok else "FAIL ") + k)
        sys.exit(0 if all(checks.values()) else 1)

    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()