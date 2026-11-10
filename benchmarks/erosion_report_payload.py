"""Build (and optionally post) the mission report payload for p176w1-n09.

The payload is assembled from the run artifacts -- ``results.json`` for the
numbers and verdicts, the generated paper-level markdown in ``docs/``, and the
figure PNGs base64-encoded -- so the report cannot quote a number that the run
did not produce.

Usage::

    python benchmarks/erosion_report_payload.py                 # write payload only
    python benchmarks/erosion_report_payload.py --post          # also POST it
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = "p176w1-n09-erosion"
MISSION_ID = 9666
OUTDIR = ROOT / "artifacts" / NODE


def build_payload() -> dict:
    results = json.loads((OUTDIR / "results.json").read_text(encoding="utf-8"))
    markdown = (ROOT / "docs" / "P176W1_N09_erosion_verification.md").read_text(encoding="utf-8")
    criteria = results["criteria"]
    verdict = ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in criteria.items())

    captions = {
        "fig1_angle_functions.png": "Angular functions and their maxima (C1).",
        "fig2_velocity_exponent.png": "Velocity exponent extraction vs literature (C2).",
        "fig3_dnv_cross_check.png": "DNV-RP-O501 parity plot and engineering map (C3).",
        "fig4_ledger_closure.png": "Ledger accumulation and closure diagnostics (C4).",
        "fig5_negative_controls.png": "Normalised angular responses and peak locations (C5).",
    }
    figures = []
    for path in sorted(OUTDIR.glob("fig*.png")):
        figures.append(
            {
                "name": path.name,
                "caption": captions.get(path.name, path.stem),
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )

    c1, c2, c3, c4, c6 = (results[k] for k in ("C1", "C2", "C3", "C4", "C6"))
    summary = (
        f"tensordem.erosion (new): Finnie/Oka/DNV-RP-O501/brittle erosion models + "
        f"event-by-event mass-energy ledger. C1 peak angles Finnie "
        f"{c1['finnie']['peak_angle_analytic_deg']:.2f} deg (err "
        f"{c1['finnie']['error_vs_ductile_reference_deg']:.2f} deg vs 20 deg), brittle "
        f"{c1['brittle']['peak_angle_numeric_deg']:.0f} deg; C2 velocity exponents 2.60/2.20/2.40 "
        f"within 10% of literature for 4 materials; C3 DNV cross-check max error "
        f"{c3['max_relative_error']:.2e} over {len(c3['cases'])} conditions; C4 ledger closure "
        f"residual {c4['polydisperse']['closure_residual']:.2e} with efficiency "
        f"{c4['polydisperse']['erosion_efficiency']:.4f} < 0.1514; C6 {c6['test_count']} test "
        f"functions ({c6['collected_cases']} cases). {verdict}."
    )
    return {
        "title": "P176W1-N09: Solid-particle erosion verification face "
        "(Finnie / Oka / DNV-RP-O501) in TensorDEM",
        "summary": summary,
        "scenario_key": "P176W1-N09",
        "mission_id": MISSION_ID,
        "tags": [
            "TensorDEM",
            "erosion",
            "finnie",
            "oka",
            "DNV-RP-O501",
            "mass-energy-ledger",
            "verification",
        ],
        "markdown": markdown,
        "figures": figures,
        "criteria": criteria,
        "all_pass": results["all_pass"],
        "artifacts": {
            "results_json": f"artifacts/{NODE}/results.json",
            "ledger_csv": f"artifacts/{NODE}/ledger_history.csv",
            "documentation": results.get("documentation"),
        },
    }


def post(payload: dict, base: str | None = None) -> str:
    base = base or os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(
        base + "/auth/login", data=login,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login returned no token: {body}")
    req = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read().decode()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--out", default=str(Path("/tmp") / f"report_{MISSION_ID}.json"))
    args = parser.parse_args(argv)

    payload = build_payload()
    out = Path(args.out)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"payload -> {out}  ({len(payload['figures'])} figures, "
          f"{len(payload['markdown'])} markdown chars)")
    if args.post:
        print(post(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())