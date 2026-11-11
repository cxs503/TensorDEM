#!/usr/bin/env python
"""Reproduce the columnar-ice anisotropy study (node p474w1-n08).

Outputs
-------
``artifacts/ice_anisotropy/study_report.json``
    criterion table (value / threshold / pass), fabric tensors, both model
    curves, per-angle comparison, isotropic control, DEM table, limitations.
``artifacts/ice_anisotropy/fabric_ledger.json``
    the ODF ledger the report's parameters were derived from.
``docs/figures/ice_anisotropy_*.png``
    pole figures, strength-vs-angle, yield loci, DEM saturation curve.

Usage
-----
    PYTHONPATH=src python scripts/run_ice_anisotropy_study.py [--skip-dem]

The DEM block steps the bonded-particle engine (6 directional tests at 1200
axial steps each); ``--skip-dem`` reproduces everything else in a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_anisotropy import (  # noqa: E402
    DirectionalTestConfig,
    FabricLedger,
    FabricODF,
    micro_contrast_sweep,
    run_verification,
    strength_surface_from_fabric,
)
from tensordem.ice_anisotropy.visualization import (  # noqa: E402
    plot_dem_saturation,
    plot_pole_figures,
    plot_strength_vs_angle,
    plot_yield_loci,
)

#: Transverse uniaxial strength of columnar sea ice used as the absolute scale.
#: The model's claim is the *ratio*, not this number; the value sits inside the
#: 1-5 MPa columnar range reviewed by Timco & Weeks (2010) at -10 C.
SIGMA_PERP_PA = 3.0e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dem", action="store_true",
                    help="skip the discrete DEM sweep (slow: ~2 min)")
    ap.add_argument("--steps", type=int, default=1200,
                    help="axial steps per DEM directional test")
    ap.add_argument("--dem-contrast", type=float, default=None,
                    help="only run the DEM sweep for this micro contrast and write "
                         "artifacts/ice_anisotropy/dem_part_<beta>.json (lets the "
                         "slow engine block be split into short jobs)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "ice_anisotropy"))
    ap.add_argument("--figdir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    outdir = Path(args.out)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    outdir.mkdir(parents=True, exist_ok=True)

    # -- partial DEM contribution mode -----------------------------------
    if args.dem_contrast is not None:
        part = micro_contrast_sweep(contrasts=(args.dem_contrast,), angles=(0.0, 90.0),
                                    base=DirectionalTestConfig(max_axial_steps=args.steps))
        path = outdir / f"dem_part_{args.dem_contrast:+.2f}.json"
        path.write_text(json.dumps(part, indent=2), encoding="utf-8")
        print(f"dem part -> {path}")
        for row in part["table"]:
            print(json.dumps(row))
        return 0

    ledger = FabricLedger()
    payload = run_verification(sigma_perp=SIGMA_PERP_PA, ratio_columnar=3.5,
                               ledger=ledger)
    ledger_path = ledger.save(outdir / "fabric_ledger.json")
    payload["artifacts"] = {"fabric_ledger": str(ledger_path.relative_to(ROOT))}

    # -- figures ---------------------------------------------------------
    odf_col = ledger.odf("odf-columnar-S2")
    odf_uni = ledger.odf("odf-uniform")
    figures = {}
    figures["pole_figures"] = str(plot_pole_figures(
        {"S2 columnar (reference)": odf_col, "uniform (control)": odf_uni},
        figdir / "ice_anisotropy_pole_figures.png").relative_to(ROOT))

    surf = strength_surface_from_fabric(odf_col, sigma_perp=SIGMA_PERP_PA)
    surf_iso = strength_surface_from_fabric(odf_uni, sigma_perp=SIGMA_PERP_PA)
    curves = {
        "continuum (Hill TI)": payload["continuum_curve"],
        "slip reference, mean-field": payload["slip_reference"]["curve_mean"],
        "slip reference, Sachs bound": payload["slip_reference"]["curve_max_sachs"],
    }
    figures["strength_vs_angle"] = str(plot_strength_vs_angle(
        curves, figdir / "ice_anisotropy_strength_vs_angle.png").relative_to(ROOT))
    figures["yield_loci"] = str(plot_yield_loci(
        {"columnar (ratio 3.5)": surf, "isotropic control": surf_iso},
        figdir / "ice_anisotropy_yield_loci.png").relative_to(ROOT))

    # -- discrete DEM block ----------------------------------------------
    parts = sorted(outdir.glob("dem_part_*.json"))
    if args.skip_dem:
        payload["dem"] = {"skipped": True}
    else:
        if parts:
            merged = {"runs": [], "table": [], "source_parts": []}
            for part_path in parts:
                part = json.loads(part_path.read_text(encoding="utf-8"))
                merged["runs"].extend(part["runs"])
                merged["table"].extend(part["table"])
                merged["source_parts"].append(str(part_path.relative_to(ROOT)))
            dem = merged
        else:
            dem = micro_contrast_sweep(contrasts=(-0.7, 0.0, 3.0),
                                       angles=(0.0, 90.0),
                                       base=DirectionalTestConfig(
                                           max_axial_steps=args.steps))
        dem["table"] = sorted(dem["table"], key=lambda r: r["micro_contrast"])
        payload["dem"] = dem
        figures["dem_saturation"] = str(plot_dem_saturation(
            dem["table"], figdir / "ice_anisotropy_dem_saturation.png").relative_to(ROOT))
        payload["criteria"].append({
            "id": "dem-negative-control",
            "statement": "isotropic micro-strength law must leave the specimen "
                         "strength direction independent (discrete engine control)",
            "value_macro_anisotropy": next(row["macro_anisotropy"] for row in dem["table"]
                                           if row["micro_contrast"] == 0.0),
            "threshold": 1e-9,
            "pass": bool(all(abs(row["macro_anisotropy"] - 1.0) <= 1e-9
                             for row in dem["table"] if row["micro_contrast"] == 0.0)),
        })
    payload["figures"] = figures

    report_path = outdir / "study_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({c["id"]: c["pass"] for c in payload["criteria"]}, indent=1))
    print(f"report  -> {report_path}")
    print(f"ledger  -> {ledger_path}")
    for name, path in figures.items():
        print(f"figure  -> {path}")
    n_pass = sum(1 for c in payload["criteria"] if c["pass"])
    print(f"criteria: {n_pass}/{len(payload['criteria'])} pass")
    return 0 if n_pass == len(payload["criteria"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())