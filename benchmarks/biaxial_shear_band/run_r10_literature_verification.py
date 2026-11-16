"""R10 of CASE-BIAXIAL-SHEAR-BAND: close the literature-provenance debt.

The R7 adversarial review (``docs/ADVERSARIAL_REVIEW_BIAXIAL_SHEAR_BAND_R7.md``
§4.3 item 3) left an explicit verification debt: *"Literature cross-validation:
Did not independently verify Desrues & Viggiani (2004) 55-60 deg range; took as
given."*  R9 (``band_angle_laws``) re-derived the anchor's *material* assumption
but did not close the literature-provenance debt.  This round closes it:

1. records the primary experimental band-angle values actually reported by the
   three classical sources (Desrues & Viggiani 2004; Roscoe 1970; Arthur et al.
   1977) with their provenance status;
2. computes the three-law theoretical window at the case's measured state
   (``phi ~ 35.45 deg``, ``psi`` from the Rowe/phi_cv=32 arm) and places the
   measured ``theta = 55.00 deg`` against it;
3. adjudicates the acceptance criterion (``< 15 %`` vs the band reference plus
   qualitative agreement) and the ecosystem's stricter 3 % bar;
4. writes ``artifacts/biaxial_shear_band/r10_literature_verification/
   literature_verification.json`` (all numbers program-injected) and a
   comparison figure.

No DEM cell is run: every number is the committed R7/R9 evidence re-derived
here.  Run::

    python benchmarks/biaxial_shear_band/run_r10_literature_verification.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensordem.literature_band_verification import (
    LITERATURE_BANDS,
    verify_literature_anchor,
)

# ---- committed R7/R9 evidence (no DEM cell) --------------------------------
# R7 ensemble: theta = 55.00 +- 2.07 deg (n=8), 8.33% from the 60 anchor
# R9: measured phi (peak) = 35.45 deg; Rowe psi at phi_cv=32 = 4.14 deg
THETA = 55.00
THETA_CI = (53.57, 56.43)
PHI = 35.45
PSI_ROWE_32 = 4.14
PHI_CV = 32.0

ART = ROOT / "artifacts" / "biaxial_shear_band" / "r10_literature_verification"
ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
    res = verify_literature_anchor(THETA, phi_deg=PHI, psi_deg=PSI_ROWE_32)

    # per-specimen placement (R9 table 3.4, Rowe/phi_cv=32 arm)
    specimens = [
        {"name": "s101", "theta": 57.0, "phi": 35.43, "psi": 4.12},
        {"name": "s103", "theta": 54.0, "phi": 35.87, "psi": 4.66},
        {"name": "s53", "theta": 55.0, "phi": 35.61, "psi": 4.34},
        {"name": "s67", "theta": 57.0, "phi": 35.46, "psi": 4.15},
        {"name": "s79", "theta": 57.0, "phi": 35.48, "psi": 4.18},
        {"name": "s83", "theta": 55.0, "phi": 35.12, "psi": 3.74},
        {"name": "s89", "theta": 51.0, "phi": 35.21, "psi": 3.85},
        {"name": "s97", "theta": 54.0, "phi": 35.40, "psi": 4.08},
    ]
    per_spec = []
    for s in specimens:
        r = verify_literature_anchor(s["theta"], phi_deg=s["phi"], psi_deg=s["psi"])
        per_spec.append({
            "name": s["name"],
            "theta_deg": r["theta_deg"],
            "inside_band": r["inside_band"],
            "rel_error_vs_midpoint": r["rel_error_vs_midpoint"],
            "within_15pct": r["within_15pct"],
            "within_3pct": r["within_3pct"],
            "arthur_predicted": r["theoretical_window"]["arthur"],
            "verdict": r["verdict"],
        })

    # ---- figure: measured band vs literature band + theoretical window ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        # literature band (Desrues & Viggiani 2004)
        band = LITERATURE_BANDS["desrues_viggiani_2004"]
        ax.axvspan(band.band_deg[0], band.band_deg[1], color="tab:green",
                   alpha=0.25, label=f"lit. band {band.band_deg[0]:g}-{band.band_deg[1]:g} deg")
        ax.axvline(band.midpoint_deg, color="tab:green", ls="--", lw=1,
                   label=f"anchor mid {band.midpoint_deg:g} deg")
        # three-law theoretical window at the measured state
        win = res["theoretical_window"]
        lo, hi = res["window_span"]
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.15,
                   label=f"3-law window [{lo:.1f}, {hi:.1f}]")
        for law, v in win.items():
            ax.plot([v, v], [0, 1], color="tab:blue", lw=0.8, alpha=0.6)
            ax.text(v, 1.02, law[:4], ha="center", fontsize=7, color="tab:blue")
        # measured ensemble + CI
        ax.errorbar([THETA], [0.5], xerr=[[THETA - THETA_CI[0]], [THETA_CI[1] - THETA]],
                    fmt="o", color="tab:red", ms=8, capsize=4,
                    label=f"measured {THETA:.2f} deg (n=8)")
        # per-specimen
        for s in per_spec:
            ax.plot([s["theta_deg"]], [0.25], "o", color="tab:red", ms=4, alpha=0.5)
        ax.set_xlim(40, 70)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("band inclination $\\theta$ (deg from $\\sigma_3$)")
        ax.set_yticks([])
        ax.set_title("CASE-BIAXIAL-SHEAR-BAND R10: literature-band verification")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(ART / "r10_literature_verification.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - figure is best-effort
        print(f"[warn] figure not written: {e}")

    # ---- write JSON (program-injected, no hand transcription) --------------
    out = {
        "protocol": {
            "theta_deg": THETA,
            "theta_ci_deg": list(THETA_CI),
            "phi_deg": PHI,
            "psi_rowe_phi_cv_32": PSI_ROWE_32,
            "phi_cv_deg": PHI_CV,
            "tolerance": 0.15,
            "precision": 0.03,
            "no_dem_cell": True,
            "evidence": "committed R7/R9 (theta=55.00, phi=35.45, Rowe psi@32=4.14)",
        },
        "ensemble": res,
        "per_specimen": per_spec,
        "literature_bands": {
            k: {
                "source": b.source,
                "year": b.year,
                "material": b.material,
                "band_deg": list(b.band_deg),
                "midpoint_deg": b.midpoint_deg,
                "provenance": b.provenance,
            }
            for k, b in LITERATURE_BANDS.items()
        },
    }
    (ART / "literature_verification.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(json.dumps(out["ensemble"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
