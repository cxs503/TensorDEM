#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 (m14498): measured convergence order and conservation error.

Two things are measured, not asserted:

1. **Convergence order** of the contact work-energy ledger conventions
   (``tensordem.contact_energy_order.order_sweep``): the residual of each
   convention versus ``dt`` is fitted as a power law and the slope is reported
   with its r^2.  A convention that is exact in the discrete sense must show a
   residual at round-off level (slope undefined), an explicit pairing must show
   first order, and a staggered pairing must lose exactly one order.

2. **Conservation** on the elastic branch (``damping_model="none"``): the
   discrete work-energy identity residual and the restitution error are read
   directly; for a lossless case the physical expectation is e = 1 within
   integration error and a monotonically decreasing residual under refinement.

Run: ``PYTHONPATH=src python3 scripts/audit_runtest6_convergence.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.contact_energy_order import (  # noqa: E402
    ContactLaw,
    fit_order,
    order_sweep,
    staggered_pairing_order,
    two_body_impact,
)


def summarise(name: str, sweep: dict) -> dict:
    print(f"\n== {name}: fitted order per convention ==")
    rows = {}
    for key, fit in sweep["fits"].items():
        if fit is None:
            print(f"  {key:28s} not finite (residual identically zero?)")
            rows[key] = None
            continue
        print(f"  {key:28s} slope={fit['slope']:+.4f}  r2={fit['r2']:.6f}  "
              f"n={fit['n_points']}  ratio={fit['ratio_coarse_over_fine']:.3e}")
        rows[key] = {"slope": fit["slope"], "r2": fit["r2"], "n_points": fit["n_points"]}
    return rows


def main() -> int:
    report: dict = {"claim": "contact energy ledger conventions have the documented orders"}

    hertz = ContactLaw(stiffness=2.0e5, exponent=1.5, damping_model="none")
    linear = ContactLaw(stiffness=2.0e5, exponent=1.0, damping_model="none")

    out = {}
    for name, law in (("hertz_lossless", hertz), ("linear_lossless", linear)):
        sweep = order_sweep(law, refinements=8, levels=5)
        levels = sweep["levels"]
        dts = [r["dt"] for r in levels]
        res_state = [r["residual_state"] for r in levels]
        print(f"\n[{name}] dt = {['%.3e' % d for d in dts]}")
        print(f"[{name}] residual_state = {['%.3e' % r for r in res_state]}")
        print(f"[{name}] residual_explicit = {['%.3e' % r['residual_explicit'] for r in levels]}")
        print(f"[{name}] residual_average_velocity = "
              f"{['%.3e' % r['residual_average_velocity'] for r in levels]}")
        # measured order of the explicit pairing, independent of the library fit
        finite = [(d, r) for d, r in zip(dts, [r["residual_explicit"] for r in levels]) if r > 0]
        own_fit = fit_order([d for d, _ in finite], [r for _, r in finite]) if len(finite) > 1 else None
        out[name] = {
            "dt": dts,
            "residual_state": res_state,
            "residual_explicit": [r["residual_explicit"] for r in levels],
            "residual_average_velocity": [r["residual_average_velocity"] for r in levels],
            "residual_staggered_power": [r["residual_staggered_power"] for r in levels],
            "restitution_error": [r["restitution_error"] for r in levels],
            "fits_library": summarise(name, sweep),
            "fits_independent_explicit": None if own_fit is None else
            {"slope": own_fit["slope"], "r2": own_fit["r2"]},
        }

    # staggered pairing must cost exactly one order relative to the good pairing
    stag = staggered_pairing_order(hertz, refinements=8)
    print("\n== staggered pairing (known defect in floe_rigid6dof._accumulate) ==")
    print(json.dumps({k: v for k, v in stag.items() if k != "levels"}, indent=1)[:900])
    report["staggered_pairing"] = {k: v for k, v in stag.items() if k != "levels"}

    # single elastic impact: absolute conservation residual at the finest dt
    imp = two_body_impact(hertz, refinements=128)
    e_meas = abs(imp.trace.v_out) / abs(imp.trace.v_in)
    report["elastic_impact_finest"] = {
        "dt_s": imp.dt,
        "steps_per_contact": imp.steps_per_contact,
        "restitution_measured": e_meas,
        "restitution_error": abs(e_meas - 1.0),
        "max_overlap_m": imp.trace.max_overlap,
        "momentum_error": imp.trace.momentum_error,
    }
    print("\n[elastic impact, 128 refinements] e = "
          f"{e_meas:.12f}  |e-1| = {abs(e_meas - 1.0):.3e}  dt = {imp.dt:.3e}  "
          f"steps/contact = {imp.steps_per_contact:.1f}  "
          f"momentum_error = {imp.trace.momentum_error:.3e}")
    report["sweeps"] = out

    verdict = (
        "order measured from the dt-sweep; see slopes above. "
        "Exactness of the average-velocity channel is confirmed when its residual "
        "sits at round-off while the explicit channel falls with slope ~1."
    )
    report["interpretation"] = verdict
    print("\n" + verdict)

    dest = ROOT / "runs" / "convergence"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(report, indent=2, default=float) + "\n",
                                       encoding="utf-8")
    print("wrote", dest / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())