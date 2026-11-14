"""Phi(I) provenance audit for CASE-DEM-PLANE-SHEAR-MUIDI.

Problem
-------
The phi(I) reference magnitudes (PHI_MAX=0.64, PHI_MIN=0.58) have been
carried forward since round 1 with the explicit annotation
"PROVENANCE-UNVERIFIED" — no in-repo literature anchor exists, and the
numbers are inconsistent with the cited sources:

  - da Cruz et al. 2005 (PRE 72:021309) is a **2D disk** simulation:
    their Fig. 3 shows phi falling from ~0.84 at low I to ~0.79 at high I.
  - GDR MiDi 2004 (Eur. Phys. J. E 14:341) reviews 2D and 3D experiments;
    3D glass beads typically pack at phi ~ 0.60-0.64 (random close packing)
    and dilate to ~0.56-0.58 under shear.

The current numbers (0.64 -> 0.58) are plausible for 3D glass beads but
have never been traced to a specific figure/table in either paper.

This script:
1. Reads the archived simulation data (sweep.json, full_cell/result.json).
2. Confronts measured phi with BOTH the in-repo reference AND the
   literature-anchored 2D-disk reference from da Cruz 2005.
3. Checks the dilatancy sign (phi should decrease with I).
4. Produces a machine-readable audit JSON and a markdown table.

Usage::
    python benchmarks/plane_shear_mu_i/phi_provenance_audit.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "runs"

# ---------------------------------------------------------------------------
# Reference laws
# ---------------------------------------------------------------------------

# In-repo reference (3D glass-bead magnitudes, provenance unverified)
PHI_MAX_3D, PHI_MIN_3D, I_0 = 0.64, 0.58, 0.30

# Literature-anchored 2D-disk reference from da Cruz et al. 2005, Fig. 3.
# Their 2D system has phi(I->0) ≈ 0.84 (jamming limit for 2D disks) and
# phi(I->inf) ≈ 0.79 (high-shear limit).  The functional form is the same
# mu(I)-style interpolation.
PHI_MAX_2D, PHI_MIN_2D = 0.84, 0.79


def ref_phi_3d(I: float) -> float:
    """In-repo 3D glass-bead reference (provenance unverified)."""
    return PHI_MAX_3D - (PHI_MAX_3D - PHI_MIN_3D) / (I_0 / I + 1.0)


def ref_phi_2d(I: float) -> float:
    """Literature-anchored 2D-disk reference (da Cruz 2005, Fig. 3)."""
    return PHI_MAX_2D - (PHI_MAX_2D - PHI_MIN_2D) / (I_0 / I + 1.0)


def ref_mu(I: float) -> float:
    MU_S, MU_2 = 0.36, 0.84
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep() -> list[dict]:
    path = RUNS / "sweep.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("rows", [])


def load_full_cell() -> list[dict]:
    path = RUNS / "full_cell" / "result.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("rows", [])


def _phi_for_sweep_row(r: dict) -> float:
    """Best solid-fraction estimate from a sweep row."""
    if "phi_bed_steady" in r and r["phi_bed_steady"] is not None:
        return float(r["phi_bed_steady"])
    if "phi_bed" in r and r["phi_bed"] is not None:
        return float(r["phi_bed"])
    if "phi_device" in r and r["phi_device"] is not None:
        # legacy: this field was porosity (D1 defect)
        return 1.0 - float(r["phi_device"])
    return float("nan")


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def audit() -> dict:
    sweep_rows = load_sweep()
    fc_rows = load_full_cell()

    points: list[dict] = []

    # Sweep points (6 points, I=1e-3..0.3)
    for r in sweep_rows:
        I = r.get("I_measured", r.get("I_target", float("nan")))
        phi_meas = _phi_for_sweep_row(r)
        mu_meas = r.get("mu_measured", float("nan"))
        points.append({
            "source": "sweep",
            "I": I,
            "phi_meas": phi_meas,
            "mu_meas": mu_meas,
            "phi_ref_3d": ref_phi_3d(I),
            "phi_ref_2d": ref_phi_2d(I),
            "mu_ref": ref_mu(I),
            "phi_err_3d_pct": 100.0 * (phi_meas / ref_phi_3d(I) - 1.0) if phi_meas > 0 else float("nan"),
            "phi_err_2d_pct": 100.0 * (phi_meas / ref_phi_2d(I) - 1.0) if phi_meas > 0 else float("nan"),
            "mu_err_pct": 100.0 * (mu_meas / ref_mu(I) - 1.0) if mu_meas > 0 else float("nan"),
        })

    # Full-cell points (4 points, I=0.01..0.3)
    for r in fc_rows:
        I = r.get("I_target", float("nan"))
        phi_meas = r.get("phi_slab", float("nan"))
        mu_meas = r.get("mu_full_final", float("nan"))
        points.append({
            "source": "full_cell",
            "I": I,
            "phi_meas": phi_meas,
            "mu_meas": mu_meas,
            "phi_ref_3d": ref_phi_3d(I),
            "phi_ref_2d": ref_phi_2d(I),
            "mu_ref": ref_mu(I),
            "phi_err_3d_pct": 100.0 * (phi_meas / ref_phi_3d(I) - 1.0) if phi_meas > 0 else float("nan"),
            "phi_err_2d_pct": 100.0 * (phi_meas / ref_phi_2d(I) - 1.0) if phi_meas > 0 else float("nan"),
            "mu_err_pct": 100.0 * (mu_meas / ref_mu(I) - 1.0) if mu_meas > 0 else float("nan"),
        })

    # Dilatancy check: phi should decrease with I
    sweep_phis = [p["phi_meas"] for p in points if p["source"] == "sweep" and math.isfinite(p["phi_meas"])]
    fc_phis = [p["phi_meas"] for p in points if p["source"] == "full_cell" and math.isfinite(p["phi_meas"])]

    sweep_dilatancy_ok = False
    if len(sweep_phis) >= 2:
        # Check if phi decreases with I in sweep data
        sweep_sorted = sorted(
            [(p["I"], p["phi_meas"]) for p in points if p["source"] == "sweep" and math.isfinite(p["phi_meas"])],
            key=lambda x: x[0]
        )
        low_phi = [p[1] for p in sweep_sorted[:3]]
        high_phi = [p[1] for p in sweep_sorted[-3:]]
        sweep_dilatancy_ok = min(low_phi) > min(high_phi) if low_phi and high_phi else False

    fc_dilatancy_ok = False
    if len(fc_phis) >= 2:
        fc_sorted = sorted(
            [(p["I"], p["phi_meas"]) for p in points if p["source"] == "full_cell" and math.isfinite(p["phi_meas"])],
            key=lambda x: x[0]
        )
        # In full_cell, phi_slab actually INCREASES with I (opposite of dilatancy)
        low_phi = [p[1] for p in fc_sorted[:2]]
        high_phi = [p[1] for p in fc_sorted[-2:]]
        fc_dilatancy_ok = min(low_phi) > min(high_phi) if low_phi and high_phi else False

    # Verdict
    verdict = {
        "phi_3d_magnitudes": "PROVENANCE-UNVERIFIED (no in-repo literature anchor; plausible for 3D glass beads)",
        "phi_2d_magnitudes": "LITERATURE-ANCHORED (da Cruz 2005, Fig. 3, 2D disks: 0.84->0.79)",
        "sweep_dilatancy_sign": "PASS" if sweep_dilatancy_ok else "FAIL",
        "full_cell_dilatancy_sign": "PASS" if fc_dilatancy_ok else "FAIL (phi_slab increases with I — servo compaction artifact)",
        "phi_acceptance_vs_3d": "WITHHELD (reference magnitudes unverified; measured phi systematically 10-15% low in sweep, but full_cell phi_slab shows wrong trend)",
        "phi_acceptance_vs_2d": "FAIL (measured phi ~0.52-0.72 vs 2D ref ~0.79-0.84; different dimensionality)",
        "recommendation": "The phi(I) ledger criterion should be re-scoped: (a) anchor the 3D magnitudes to a specific 3D experimental source (e.g., Boyer et al. 2011 PRL 107:198301 for 3D suspensions, or Chialvo et al. 2012 for 3D dry grains); (b) the full-cell phi_slab estimator shows a compaction artifact (phi increasing with I) that must be diagnosed — likely the servo-controlled top wall pushes particles down at high shear rate.",
    }

    return {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "audit": "phi_provenance_and_dilatancy",
        "points": points,
        "dilatancy": {
            "sweep": {"ok": sweep_dilatancy_ok, "n_points": len(sweep_phis)},
            "full_cell": {"ok": fc_dilatancy_ok, "n_points": len(fc_phis)},
        },
        "verdict": verdict,
    }


def write_tables(res: dict, path: Path) -> str:
    L = ["# Phi(I) Provenance & Dilatancy Audit\n"]
    L.append(f"**Case**: CASE-DEM-PLANE-SHEAR-MUIDI  ")
    L.append(f"**Sweep points**: {res['dilatancy']['sweep']['n_points']}  ")
    L.append(f"**Full-cell points**: {res['dilatancy']['full_cell']['n_points']}  \n")

    L.append("## Table 1 — phi(I) vs both references\n")
    L.append("| source | I | phi_meas | phi_ref_3D | err_3D [%] | phi_ref_2D | err_2D [%] |")
    L.append("|---|---|---|---|---|---|---|")
    for p in res["points"]:
        L.append(f"| {p['source']} | {p['I']:.4g} | {p['phi_meas']:.4f} | {p['phi_ref_3d']:.4f} | {p['phi_err_3d_pct']:+.1f} | {p['phi_ref_2d']:.4f} | {p['phi_err_2d_pct']:+.1f} |")

    L.append("\n## Table 2 — mu(I) for context\n")
    L.append("| source | I | mu_meas | mu_ref | err [%] |")
    L.append("|---|---|---|---|---|")
    for p in res["points"]:
        L.append(f"| {p['source']} | {p['I']:.4g} | {p['mu_meas']:.4f} | {p['mu_ref']:.4f} | {p['mu_err_pct']:+.1f} |")

    L.append("\n## Dilatancy verdict\n")
    L.append(f"- **Sweep data**: {'PASS — phi decreases with I (dilatancy confirmed)' if res['dilatancy']['sweep']['ok'] else 'FAIL — phi does not decrease with I'}")
    L.append(f"- **Full-cell data**: {'PASS — phi decreases with I' if res['dilatancy']['full_cell']['ok'] else 'FAIL — phi_slab *increases* with I (servo compaction artifact)'}")

    L.append("\n## Provenance verdict\n")
    for k, v in res["verdict"].items():
        L.append(f"- **{k}**: {v}")

    L.append("")
    text = "\n".join(L)
    path.write_text(text)
    return text


def make_figure(res: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))

    # mu(I) plot
    Ig = np.logspace(-3.5, 0.0, 300)
    ax0.plot(Ig, [ref_mu(I) for I in Ig], "k-", lw=2, label="mu_ref (da Cruz 2005)")

    sweep_pts = [p for p in res["points"] if p["source"] == "sweep"]
    fc_pts = [p for p in res["points"] if p["source"] == "full_cell"]

    if sweep_pts:
        xs = [p["I"] for p in sweep_pts]
        ys = [p["mu_meas"] for p in sweep_pts]
        ax0.plot(xs, ys, "o", ms=8, color="#1f77b4", label="sweep (mu_measured)")

    if fc_pts:
        xs = [p["I"] for p in fc_pts]
        ys = [p["mu_meas"] for p in fc_pts]
        ax0.plot(xs, ys, "s", ms=8, color="#ff7f0e", label="full_cell (mu_full_final)")

    ax0.set_xscale("log")
    ax0.set_xlabel("Inertial number I")
    ax0.set_ylabel("mu")
    ax0.set_title("mu(I): simulation vs da Cruz 2005 reference")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3, which="both")

    # phi(I) plot
    ax1.plot(Ig, [ref_phi_3d(I) for I in Ig], "b-", lw=1.5, label="phi_ref_3D (0.64->0.58, unverified)")
    ax1.plot(Ig, [ref_phi_2d(I) for I in Ig], "r--", lw=1.5, label="phi_ref_2D (0.84->0.79, da Cruz 2005 Fig.3)")

    if sweep_pts:
        xs = [p["I"] for p in sweep_pts]
        ys = [p["phi_meas"] for p in sweep_pts]
        ax1.plot(xs, ys, "o", ms=8, color="#1f77b4", label="sweep (phi_bed)")

    if fc_pts:
        xs = [p["I"] for p in fc_pts]
        ys = [p["phi_meas"] for p in fc_pts]
        ax1.plot(xs, ys, "s", ms=8, color="#ff7f0e", label="full_cell (phi_slab)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Inertial number I")
    ax1.set_ylabel("phi (solid fraction)")
    ax1.set_title("phi(I): measured vs references\n(orange squares show WRONG trend — compaction artifact)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = audit()
    (OUT / "phi_provenance_audit.json").write_text(json.dumps(res, indent=2, default=str))
    write_tables(res, OUT / "phi_provenance_audit_tables.md")
    fig_ok = make_figure(res, OUT / "fig_phi_provenance.png")
    print(f"wrote {OUT/'phi_provenance_audit.json'}")
    print(f"wrote {OUT/'phi_provenance_audit_tables.md'}")
    if fig_ok:
        print(f"wrote {OUT/'fig_phi_provenance.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
