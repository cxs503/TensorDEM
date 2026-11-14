#!/usr/bin/env python3
"""Render the m20934 result tables into docs/CASE-DEM-SILO-BEVERLOO_m20934.md.

Reads ``runs/m20934_beverloo_kn/summary.json`` and rewrites the
``<!-- TABLES:BEGIN --> ... <!-- TABLES:END -->`` block of the doc so every
number in the report is programmatic (zero hand-copy).

Usage
-----
    PYTHONPATH=src python benchmarks/m20934_beverloo_kn/render_doc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m20934.md"
SUMMARY = ROOT / "runs" / "m20934_beverloo_kn" / "summary.json"


def render(arms: list[dict]) -> str:
    lines = [
        "| arm | D/d | kn [N/m] | dt [s] | n_bed | ρ_b [kg/m³] | W_got [kg/s] | W_ref [kg/s] | rel err | C_rec | CV | <10% |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in arms:
        lines.append(
            f"| {r['tag']} | {r['D_over_d']:.1f} | {r['kn_N_m']:.1e} | {r['dt_s']:.2e} "
            f"| {r['n_bed']} | {r['rho_b_kg_m3']:.0f} | {r['W_got_kg_s']:.4f} "
            f"| {r['W_ref_kg_s']:.4f} | {r['rel_err_pct']:+.1f}% | {r['C_recovered']:.3f} "
            f"| {r['steady_rate_cv']:.3f} | {'PASS' if r['pass_10pct'] else 'FAIL'} |")
    return "\n".join(lines)


def verdict_block(arms: list[dict]) -> tuple[str, str]:
    lines = [
        "| 关键量 | 基准 | 实测 | 相对误差 | <3% | <10%（总账验收） |",
        "|---|---|---|---|---|---|",
    ]
    for r in arms:
        if "dthalf" in r["tag"]:
            continue
        lines.append(
            f"| W ({r['tag']}) | {r['W_ref_kg_s']:.4f} kg/s | {r['W_got_kg_s']:.4f} kg/s "
            f"| {r['rel_err_pct']:+.1f}% | {'是' if abs(r['rel_err_pct']) < 3 else '否'} "
            f"| {'是' if r['pass_10pct'] else '否'} |")
    for r in arms:
        if "dthalf" not in r["tag"]:
            continue
        base = next(a for a in arms if a["tag"] == r["tag"].replace("_dthalf", ""))
        dW = (r["W_got_kg_s"] - base["W_got_kg_s"]) / base["W_got_kg_s"] * 100
        lines.append(
            f"| W (dt/2 守卫, {r['tag']}) | {base['W_got_kg_s']:.4f} kg/s (同臂 dt) "
            f"| {r['W_got_kg_s']:.4f} kg/s | {dW:+.1f}% | — | — |")
    n10 = sum(r["pass_10pct"] for r in arms if "dthalf" not in r["tag"])
    n = len([r for r in arms if "dthalf" not in r["tag"]])
    summary = (f"{n10}/{n} 个 kn 臂落在 ±10% 验收带内")
    return "\n".join(lines), summary


def _fill(text: str, begin: str, end: str, block: str) -> str:
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    return pre + begin + "\n" + block + "\n" + end + post


def main() -> int:
    if not SUMMARY.exists():
        print(f"missing {SUMMARY}; run the sweep + analyse.py first", file=sys.stderr)
        return 1
    summary = json.loads(SUMMARY.read_text())
    text = DOC.read_text()
    text = _fill(text, "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->",
                 render(summary["arms"]))
    vblock, vsummary = verdict_block(summary["arms"])
    text = _fill(text, "<!-- VERDICT:BEGIN -->", "<!-- VERDICT:END -->", vblock)
    text = text.replace("<!-- VERDICT:SUMMARY -->", vsummary)
    DOC.write_text(text)
    print(f"rendered {len(summary['arms'])} arms into {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
