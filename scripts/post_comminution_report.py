"""Publish the comminution study as a platform report (paper-level, six elements).

Idempotency: the platform rejects a second report with the same
``(mission_id, md5(markdown))`` ("uq_reports_mission_md5"), so posting is a
one-shot action per document revision -- re-running it after the document has been
published returns an error by design, and a *changed* document is the way to
republish. Keep probe/test posts out of the mission id for that reason: they
consume the slot for their own markdown but also clutter the mission ledger.

The markdown body is the *generated* document (``docs/COMMINUTION_ENERGY_SURFACE.md``,
produced by ``scripts/comminution_report.py`` from the study JSON), so every number
in the report is injected programmatically -- nothing is hand-typed.

Usage:
    PYTHONPATH=src python3 scripts/post_comminution_report.py \
        --json runs/comminution_energy_surface.json \
        --doc docs/COMMINUTION_ENERGY_SURFACE.md \
        --figdir artifacts/comminution_energy_surface \
        --mission 12432
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="runs/comminution_energy_surface.json")
    ap.add_argument("--doc", default="docs/COMMINUTION_ENERGY_SURFACE.md")
    ap.add_argument("--figdir", default="artifacts/comminution_energy_surface")
    ap.add_argument("--mission", type=int, default=12432)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    study = json.loads(Path(args.json).read_text())
    body = Path(args.doc).read_text()
    v = study["verdict"]
    cs, pw, bo = study["critical_speed"], study["power"], study["bond"]
    br = study["breakage"]
    d1 = br["operator_verification"]

    summary = (
        f"TensorDEM comminution first verification: critical speed DEM-detected within "
        f"{100 * cs['max_abs_error_vs_nominal']:.2f}% of 42.3/sqrt(D); mill power mean "
        f"deviation {100 * pw['mean_abs_rel_error']:.1f}% vs Hogg-Fuerstenau with "
        f"momentum closure {100 * pw['max_moment_closure']:.1f}%; Bond third theory exact "
        f"and closed against the grindability equation; sectional breakage kinetics exact "
        f"to {d1['max_abs_cdf_difference']:.0e} in CDF, but the two PSD model structures "
        f"diverge by {100 * br['max_p80_rel_error']:.0f}% in P80 (honest FAIL; the "
        f"experimental PSD criterion is WITHHELD - no dataset offline)."
    )
    figures = []
    figdir = Path(args.figdir)
    spec = json.loads((figdir / "figures.json").read_text()) if (figdir / "figures.json").exists() else []
    for entry in spec:
        p = figdir / entry["name"]
        if p.exists():
            figures.append({"name": entry["name"], "caption": entry["caption"],
                            "png_base64": base64.b64encode(p.read_bytes()).decode()})

    payload = {
        "title": "TensorDEM comminution: ball-mill power, critical speed, Bond energy and breakage kinetics",
        "summary": summary,
        "scenario_key": "comminution_energy_surface",
        "mission_id": args.mission,
        "tags": ["TensorDEM", "comminution", "ball-mill", "DEM", "Bond"],
        "markdown": body,
        "figures": figures,
    }
    if args.dry_run:
        Path("/tmp/comminution_report_payload.json").write_text(json.dumps(payload)[:2000])
        print(f"dry run: {len(figures)} figures, {len(body)} chars of markdown")
        return 0
    token = _post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                  "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
    out = _post("/tensor/reports", payload, token)
    print(json.dumps({"report_id": out.get("id") or out.get("report_id"),
                      "figures": len(figures)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())