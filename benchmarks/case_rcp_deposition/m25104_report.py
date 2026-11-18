#!/usr/bin/env python3
"""m25104 — build the paper-grade report JSON for the RCP fresh verification.

All numbers are injected from ``runs/m25104_rcp_verify/verify_summary.json``
(zero hand-copy).  Figures are embedded as base64.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EV = ROOT / "runs" / "m25104_rcp_verify"
FIG = EV / "figures"

PHI_REF = 0.6366


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main() -> int:
    s = json.load(open(EV / "verify_summary.json"))
    lv0 = s["levels"]["mu=0"]
    lv1 = s["levels"]["mu=0.1"]

    def row(lv: dict) -> str:
        p = lv["phi_pooled"]
        return (f"| {lv['n_seeds']} (3 archived + 3 fresh) | {p['mean']:.5f} "
                f"± {p['std']:.5f} | {lv['Z_mean']:.2f} | "
                f"{lv['abs_err_vs_ref']:.5f} | {lv['rel_err_vs_ref']*100:+.2f}% | "
                f"{'是' if lv['in_band'] else '否'} |")

    per_seed_rows = []
    for mukey, lv in s_levels(s := s):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
