"""Publish the round-16 progress report for CASE-DEM-PLANE-SHEAR-MUIDI (m20855).

Reproduces the posted report (id 18899) from the archived artifacts; every table
and figure is regenerated, nothing is hand-copied.  Run:

    python post_report_m20855.py            # build + post
    python post_report_m20855.py --dry-run  # build only, write /tmp/report_m20855.json
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE / "benchmarks" / "plane_shear_mu_i" / "runs" / "round16"
AUDIT = HERE / "benchmarks" / "plane_shear_mu_i" / "runs" / "phi_audit" / \
    "strain_convergence_audit.json"
DOC = HERE / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R16.md"

CAPTIONS = {
    "fig_strain_budget.png":
        "(a) planned shear-step floor vs steps executed; (b) strain achieved / "
        "requested vs I with the 1.05 tolerance line. The budget is honoured "
        "below I=0.01 and fails by up to 49x at I=0.3.",
    "fig_phi_mu_vs_reference.png":
        "Measured phi(I) (container definition) and mu(I) (full-cell virial) vs "
        "reference_law.py on the I_bulk abscissa. Filled = strain budget "
        "respected and therefore quotable; open squares = budget violated.",
    "fig_phi_convergence.png":
        "phi_extent along the accumulated strain for all six runs; dotted = "
        "budget violated. The three solid curves flatten (terminal drift "
        "<= 0.5 %).",
}


def build() -> dict:
    figs = [{"name": os.path.basename(p), "caption": CAPTIONS.get(os.path.basename(p), ""),
             "png_base64": base64.b64encode(Path(p).read_bytes()).decode()}
            for p in sorted(glob.glob(str(ROUND / "*.png")))]
    audit = json.loads(AUDIT.read_text())
    return {
        "title": ("CASE-DEM-PLANE-SHEAR-MUIDI R16: strain-convergence measured - "
                  "phi converges, mu does not, and half the sweep is a void budget"),
        "summary": (
            "Round-15's six strain-convergence runs (cj2950-cj2955) had succeeded "
            "but were never collected. Audited: strain 2.0 was delivered only at "
            "I<=0.01; the pressure-recovery gate overran it 1.43x/3.13x/49.46x at "
            "I=0.03/0.1/0.3 (the last spending exactly its 20000-step budget). At "
            "the honoured points phi is stationary (0.550-0.561) but 12-14 % below "
            "reference, falsifying 'phi deficit is a transient'. mu is not "
            "stationary anywhere in this sweep."),
        "scenario_key": "dem_plane_shear_mu_i",
        "mission_id": 20855,
        "tags": ["DEM", "mu(I)", "plane shear", "rheology", "verification",
                 "granular", "strain convergence", "honest-FAIL", "TensorDEM",
                 "CASE-DEM-PLANE-SHEAR-MUIDI"],
        "markdown": DOC.read_text(),
        "figures": figs,
        "_verdict": audit["verdict"],
    }


def post(payload: dict) -> dict:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")

    def _post(url, body, tok=None):
        h = {"Content-Type": "application/json"}
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    d = _post(f"{base}/auth/login",
              {"username": os.environ["TENSOR_AGENT_USER"],
               "password": os.environ["TENSOR_AGENT_PASS"]})
    tok = d.get("access_token") or d.get("token")
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    return _post(f"{base}/tensor/reports", body, tok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rep = build()
    out = Path("/tmp") / "report_m20855.json"
    out.write_text(json.dumps(rep))
    print(f"built {out} ({out.stat().st_size} bytes, {len(rep['figures'])} figures)")
    print(f"verdict: {rep['_verdict']}")
    if args.dry_run:
        return 0
    print("posted:", post(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())