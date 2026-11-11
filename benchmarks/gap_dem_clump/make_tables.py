#!/usr/bin/env python3
"""Render docs/GAP_DEM_CLUMP_TABLES.md from artifacts/GAP-DEM-CLUMP/results.json.

Every number in the generated markdown is read from the results JSON, so no
value in the documentation is hand-transcribed.

    python benchmarks/gap_dem_clump/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts" / "GAP-DEM-CLUMP" / "results.json"
OUT = ROOT / "docs" / "GAP_DEM_CLUMP_TABLES.md"
TARGET = ROOT / "docs" / "GAP_DEM_CLUMP.md"
MARK = "<!-- AUTO-TABLES -->"


def fmt(x: float, nd: int = 4) -> str:
    if isinstance(x, (int,)):
        return str(x)
    if x == 0.0:
        return "0"
    a = abs(x)
    if a < 1e-3 or a >= 1e5:
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


def main() -> int:
    d = json.loads(RES.read_text())
    c1 = d["C1_analytic"]
    c2 = d["C2_inertia_scan"]
    c3 = d["C3_repose_scan"]
    sweep = d.get("C3_rolling_sweep", {})
    crit = d["criteria"]
    lines = []
    lines.append("# GAP-DEM-CLUMP — program-generated tables\n")
    lines.append(f"Generated from `{RES.relative_to(ROOT)}` "
                 f"(campaign run {d['generated_utc']}, wall {d['wall_seconds']} s, "
                 f"{d['environment']['machine']}, numpy {d['environment']['numpy']}).\n")
    lines.append("## C1 — analytic inertia kernel vs independent references\n")
    lines.append("| check | value | acceptance | verdict |")
    lines.append("|---|---|---|---|")
    rows = [
        ("single sphere (exact $2/5 m r^2$)", c1["sphere_rel_err"], "<1e-12"),
        ("tangent dumbbell (textbook tensor)", c1["dumbbell_rel_err"], "<1e-12"),
        ("overlapping pair vs 384³ voxel (inertia)", c1["overlapping_pair_rel_err_vs_voxel384"], "<1e-3"),
        ("overlapping pair vs 384³ voxel (volume)", c1["overlapping_pair_volume_rel_err"], "<1e-3"),
        ("three-sphere cluster, 3rd order vs 384³ voxel", c1["triple_rel_err_order3_vs_voxel384"], "<1e-3"),
        ("three-sphere cluster, 2nd order vs 384³ voxel", c1["triple_rel_err_order2_vs_voxel384"], "reference"),
    ]
    for name, val, acc in rows:
        ok = "—" if acc == "reference" else ("PASS" if float(val) < float(acc.replace("<", "")) else "FAIL")
        lines.append(f"| {name} | {fmt(float(val))} | {acc} | {ok} |")
    lines.append(f"| triple-intersection volume fraction | {fmt(c1['triple_volume_fraction'])} | "
                 f">0 (non-trivial) | {'PASS' if c1['triple_volume_fraction'] > 1e-3 else 'FAIL'} |")
    lines.append(f"\nThird order improves the triple case: "
                 f"**{c1['rollover'] if 'rollover' in c1 else c1['third_order_improves']}**; "
                 f"kernel criterion C1 = **{'PASS' if c1['pass'] else 'FAIL'}**.\n")

    lines.append("## C2 — production clump inertia vs analytic (acceptance 1 %)\n")
    lines.append("| a/b | n members | achieved AR | sphericity | volume [m³] | mass [kg] | "
                 "I_analytic (ax, t1, t2) [kg·m²] | I_production | rel.err production | rel.err fine voxel (192³) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in c2["rows"]:
        ia = r["analytic_inertia_diag"]
        ip = r["production_inertia_diag"]
        lines.append(
            f"| {fmt(r['aspect_ratio_param'],2)} | {r['n_members']} | {fmt(r['achieved_aspect_ratio'],3)} | "
            f"{fmt(r['sphericity'],4)} | {fmt(r['volume_m3'])} | {fmt(r['mass_kg'])} | "
            f"{fmt(ia[0])}, {fmt(ia[1])}, {fmt(ia[2])} | "
            f"{fmt(ip[0])}, {fmt(ip[1])}, {fmt(ip[2])} | "
            f"**{fmt(r['rel_err_production'])}** | {fmt(r['rel_err_fine_voxel'])} |")
    lines.append(f"\nWorst production-vs-analytic component error over the scan: "
                 f"**{fmt(c2['worst_rel_err_production'])}** (acceptance 1 %) → "
                 f"C2 = **{'PASS' if c2['pass'] else 'FAIL'}**.\n")

    if c3.get("skipped"):
        lines.append("## C3 — skipped in this run\n")
    else:
        lines.append("## C3 — repose angle vs aspect ratio (2 seeds, "
                     f"mu_roll = {c3['rows'][0]['mu_roll']})\n")
        lines.append("| a/b | seed | repose angle [deg] | peak height [m] | flank radius [m] | "
                     "n particles | steps | wall [s] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in c3["rows"]:
            lines.append(f"| {fmt(r['aspect_ratio_param'],2)} | {r['seed']} | "
                         f"{fmt(r['angle_profile_deg'],3)} | {fmt(r['peak_height'])} | "
                         f"{fmt(r['max_radius'])} | {r['n_particles']} | {r['steps']} | "
                         f"{r['wall_seconds']} |")
        lines.append("\n| a/b | mean [deg] | std over seeds | Δ vs sphere [deg] |")
        lines.append("|---|---|---|---|")
        for k, v in c3["by_ar"].items():
            dl = c3["deltas_vs_sphere_deg"][k]
            lines.append(f"| {k} | {fmt(v['mean'],3)} | {fmt(v['std'],3)} | {fmt(dl,3)} |")
        lines.append(f"\nMonotone non-decreasing: **{c3['monotone_increasing']}**; "
                     f"max Δ vs sphere: **{fmt(c3['max_delta_deg'],3)}°** (need ≥ 3°); "
                     f"sphere baseline inside the 22–25° literature band: "
                     f"**{c3['sphere_in_literature_band']}** → "
                     f"C3 = **{'PASS' if c3['pass'] else 'FAIL'}**.\n")
        if c3.get("failure_reason"):
            lines.append(f"Failure reason (round-1, for the round-2 research order): "
                         f"{c3['failure_reason']}\n")
    if sweep:
        lines.append("## C3 — rolling-resistance mechanism sweep (sphere vs a/b = 3, seed 0)\n")
        lines.append("| mu_roll | sphere [deg] | a/b = 3 [deg] | Δ [deg] |")
        lines.append("|---|---|---|---|")
        for k, v in sweep.items():
            vals = list(v.values())
            lines.append(f"| {k.split('=')[1]} | {fmt(vals[0],3)} | {fmt(vals[1],3)} | "
                         f"{fmt(vals[1]-vals[0],3)} |")
        lines.append("")
    lines.append("## Criteria summary\n")
    lines.append("| id | criterion | verdict |")
    lines.append("|---|---|---|")
    names = {
        "C1_analytic_kernel": "analytic inertia kernel vs closed forms / fine voxel",
        "C2_inertia_vs_analytic_under_1pct": "production clump inertia vs analytic < 1 %",
        "C3_repose_trend_matches_literature": "repose-angle trend vs aspect ratio",
    }
    for k, v in crit.items():
        lines.append(f"| {k} | {names.get(k, k)} | **{'PASS' if v else 'FAIL'}** |")
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text)} bytes)")
    if TARGET.exists():
        src = TARGET.read_text()
        if MARK in src:
            print(f"docs/GAP_DEM_CLUMP.md keeps placeholder {MARK}; "
                  f"tables are in the generated file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())