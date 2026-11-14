#!/usr/bin/env python3
"""Generate paper-style report JSON for CASE-DEM-RCP-PACKING-FRACTION (m20441)."""
from __future__ import annotations
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
FIG_DIR = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "figures")
DOC = os.path.join(ROOT, "docs", "CASE_RCP_DEPOSITION.md")

def b64_png(name):
    path = os.path.join(FIG_DIR, name)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def read_doc():
    with open(DOC) as f:
        return f.read()

def main():
    figures = []
    for name, caption in [
        ("fig_profiles.png", "Settled solid-fraction profiles φ(z) for both friction tiers, with the ledger band [0.62, 0.66] and the reference value φ_ref = 0.6366."),
        ("fig_phi_vs_mu.png", "Bulk solid fraction vs inter-particle friction coefficient. One marker per seed; error bars show ±1σ across seeds."),
        ("fig_phi_vs_N.png", "Finite-size scan of φ_bulk showing convergence from N=1000 to N=10000."),
        ("fig_settling.png", "Kinetic-energy history during settling. The stopping criterion KE/(Nmgd) < 5×10⁻⁴ sustained for 4 gravity times is marked."),
        ("fig_phi_vs_stiffness.png", "Bulk solid fraction vs 1/k_n for both friction tiers. Squares are the Richardson (hard-sphere) limits; annotation gives the observed order p."),
    ]:
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": b64_png(name),
        })

    doc = read_doc()

    report = {
        "title": "CASE-DEM-RCP-PACKING-FRACTION: Random Close Packing Volume Fraction — Gravity-Deposition DEM Benchmark",
        "summary": (
            "TensorDEM gravity-deposition RCP benchmark: N=10000, μ=0/0.1 tiers, 3 seeds each, 4 stiffness levels. "
            "Frictionless φ=0.6424±0.0002 (+0.91% vs Bernal&Mason 1960 ref 0.6366); low-friction φ=0.6263±0.0004 (−1.62%). "
            "Both within ledger band [0.62,0.66] and <3% of reference. Hard-sphere limits: 0.6354 (μ=0), 0.6174 (μ=0.1)."
        ),
        "scenario_key": "CASE-DEM-RCP-PACKING-FRACTION",
        "mission_id": 20441,
        "tags": ["DEM", "RCP", "packing", "validation", "gravity-deposition", "monodisperse", "friction-scan", "stiffness-convergence"],
        "markdown": doc,
        "figures": figures,
    }

    out = "/tmp/report_m20441.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False)
    print(f"Wrote {out} ({os.path.getsize(out)} bytes, {len(figures)} figures)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
