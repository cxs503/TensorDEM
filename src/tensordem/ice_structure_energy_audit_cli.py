"""Command-line entry point for the ice-structure four-ledger energy audit.

Installed as ``tensordem-energy-audit`` (see ``[project.scripts]``) and runnable
as ``python -m tensordem.ice_structure_energy_audit_cli``.  It runs the audit
case of node P304W2-N03 for one or both load cases, prints the four ledgers and
the closure residual, and optionally writes the per-step ledger history and a
verdict JSON so a run can be archived as evidence.

Exit code is 0 only when every requested residual is inside its bar
(1 % elastic / 2 % with fracture), so the command is usable as a machine gate.

Example::

    python -m tensordem.ice_structure_energy_audit_cli --case both \
        --out docs/artifacts/ice_structure_energy_ledger/cli
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ice_structure_energy_ledger import (
    CohesiveBondLaw,
    ContactLaw,
    contact_strain_energy_reference,
    kinetic_energy_reference,
    run_energy_audit_case,
)

__all__ = ["main", "run_cli_case", "THRESHOLDS"]

#: Closure residual bar per load case (task criteria (e) / (f)).
THRESHOLDS = {"elastic": 1.0e-2, "fracture": 2.0e-2}

#: Bond strength that keeps the floe elastic / that fractures it, in Pa.
BOND_STRENGTH = {"elastic": 4.0e5, "fracture": 2.0e5}
RESTITUTION = {"elastic": 1.0, "fracture": 0.9}


def run_cli_case(name: str, *, dt: float, duration: float) -> dict:
    """Run one audited impact case and return the verdict record."""
    if name not in THRESHOLDS:
        raise ValueError(f"unknown case {name!r}; choose from {sorted(THRESHOLDS)}")
    sim = run_energy_audit_case(
        impact_velocity=(-2.0, 0.0, 0.0),
        contact=ContactLaw(k_normal=1.0e6, restitution=RESTITUTION[name]),
        bond_law=CohesiveBondLaw(tensile_strength=BOND_STRENGTH[name]),
        dt=dt, duration=duration, gap=2.0e-3,
    )
    summary = sim.summary()
    kinetic = kinetic_energy_reference(sim.velocities, sim.omegas,
                                       sim.mass, sim.inertia)
    strain = contact_strain_energy_reference(sim)
    e_k_error = (abs(sim.ledger.E_k - kinetic["E_k"]) / kinetic["E_k"]
                 if kinetic["E_k"] else 0.0)
    e_s_error = (abs(sim.ledger.E_s - strain["E_s"]) / strain["E_s"]
                 if strain["E_s"] else 0.0)
    residual = summary["residual_relative_max"]
    return {
        "case": name,
        "dt_s": dt,
        "duration_s": duration,
        "n_steps": summary["n_steps"],
        "ledger_final": {k: summary[k] for k in ("E_k", "E_s", "E_f", "E_v")},
        "E_total_0_J": summary["E_total_0"],
        "residual_relative_max": residual,
        "threshold": THRESHOLDS[name],
        "pass": bool(residual < THRESHOLDS[name]),
        "E_k_reference_error": e_k_error,
        "E_s_reference_error": e_s_error,
        "n_broken": summary["n_broken"],
        "history": sim.history,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ice-structure four-ledger energy-conservation audit")
    parser.add_argument("--case", default="both",
                        choices=["elastic", "fracture", "both"])
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--duration", type=float, default=8.0e-3)
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for verdict.json / ledger_<case>.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.dt <= 0.0 or args.duration <= 0.0:
        parser.error("--dt and --duration must be positive")

    cases = ["elastic", "fracture"] if args.case == "both" else [args.case]
    records = [run_cli_case(name, dt=args.dt, duration=args.duration)
               for name in cases]

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for rec in records:
            history = rec.pop("history")
            path = args.out / f"ledger_{rec['case']}.csv"
            keys = [k for k in history[0] if k != "out_of_ledger"]
            with path.open("w", encoding="utf-8") as stream:
                stream.write(",".join(keys) + "\n")
                for row in history:
                    stream.write(",".join(repr(row[k]) for k in keys) + "\n")
            rec["ledger_csv"] = str(path)
        verdict = {
            "node": "P304W2-N03",
            "command": "tensordem-energy-audit",
            "cases": records,
            "all_pass": all(r["pass"] for r in records),
        }
        (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        if not args.quiet:
            print(f"verdict: {args.out / 'verdict.json'}")

    for rec in records:
        if not args.quiet:
            print(f"[{rec['case']}] eps_E = {rec['residual_relative_max']:.6g} "
                  f"(bar {rec['threshold']:.0e}) -> "
                  f"{'PASS' if rec['pass'] else 'FAIL'}; "
                  f"E_k={rec['ledger_final']['E_k']:.6g} J "
                  f"E_s={rec['ledger_final']['E_s']:.6g} J "
                  f"E_f={rec['ledger_final']['E_f']:.6g} J "
                  f"E_v={rec['ledger_final']['E_v']:.6g} J")
    return 0 if all(r["pass"] for r in records) else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI test
    raise SystemExit(main())