"""Single-bond engine driver: isolate the constitutive rate response.

Purpose
-------
The specimen-level campaign (``verify_ice_bond_specimen_rate.py``) was an
honest FAIL: at 0.25-4 s^-1 a 4x4x3 opposed-platen specimen is
inertia-dominated, so the emergent strength exponent could not be compared to
the analytic ductile envelope ``1/(m+1)``.  The gap ledger's next node is to
*isolate the constitutive response* so the inertial term no longer dominates.

This driver does exactly that at the single-bond level:

* A 2-particle / 1-bond assembly is built from the real ``IceDEM3D`` engine
  (a 2x2x2 grid reduced to one bond), so the *engine's* call sites
  (``_bond_force`` with history advance, ``_commit_fracture``,
  ``_commit_creep_damage`` with the engine's ``F_ref``) are the ones under
  test -- not the raw law in isolation.
* The bond extension is prescribed kinematically at a constant strain rate
  ``edot``: ``d(t) = edot * t * L0``.  Because the extension is prescribed
  (not integrated), there is no inertia in the constitutive response; the
  measured failure force ``F_f`` is the pure constitutive failure force.
* The bond fails by whichever mechanism triggers first: brittle cleavage
  (``strain > eps_b``, failure mode 1) or Kachanov creep damage
  (``D >= 1``, failure mode 4).

Analytic reference (linear bond + creep, exact)
-----------------------------------------------
With a linear ramp ``F(t) = k * edot * L0 * t`` and ``F_ref = k * eps_b * L0``
(``fraction = 1``), the Kachanov clock gives

    D(t) = (1/t_ref) * (k*edot*L0/F_ref)**m * t**(m+1)/(m+1),

so ``D = 1`` at

    t_f = [(m+1) * t_ref * (eps_b/edot)**m]**(1/(m+1)),
    F_f(edot) = k * L0 * eps_b**(m/(m+1)) * ((m+1)*t_ref)**(1/(m+1))
               * edot**(1/(m+1)),

i.e. the emergent envelope is ``F_f ~ edot**(1/(m+1))``.  The brittle
criterion fails the bond at ``F_b = k * eps_b * L0`` (rate independent).  The
measured strength is ``min(F_b, F_f(edot))``; the crossover rate is
``edot_c = eps_b / ((m+1) * t_ref)``.

Arms
----
A ``linear``                 negative control: rate-independent strength.
B ``linear+creep``           the closed-form anchor: exponent 1/(m+1).
C ``maxwell_wiechert``       rate *stiffening* only (relaxation spectrum).
D ``maxwell_wiechert+creep`` the physical model; exponent compared to 1/(m+1).

Every number in the JSON is produced by the run; the analytic references are
computed in the same script from the same constants, so nothing is
transcribed.

Usage
-----
    PYTHONPATH=src python scripts/verify_ice_bond_single_bond_engine.py
    PYTHONPATH=src python scripts/verify_ice_bond_single_bond_engine.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

# ---------------------------------------------------------------------------
# Single-bond constants (single source of truth for the campaign)
# ---------------------------------------------------------------------------
RADIUS = 0.025
L0 = 2.0 * RADIUS                 # bond rest length (m)
BOND_STIFFNESS = 2_000.0          # N/m, unrelaxed
BREAKING_STRAIN = 0.015
CREEP_EXPONENT = 2                # m -> ductile envelope exponent 1/(m+1) = 1/3
CREEP_TIME_S = 5.0e-3             # t_ref
RELAXATION_TIME_S = (5.0e-3, 5.0e-2)
EQUILIBRIUM_FRACTION = 0.35
DELAYED_FRACTION = (0.45, 0.20)
DT = 1.0e-4                       # s; damage-clock increment, O(dt) accurate
# Creep-dominated window (edot < edot_c) plus a brittle plateau check.
CREEP_RATES = (0.03, 0.1, 0.3)    # s^-1
BRITTLE_RATES = (1.0, 3.0)        # s^-1
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_bond_single_bond"


def analytic_exponent() -> float:
    return 1.0 / (CREEP_EXPONENT + 1.0)


def analytic_crossover_rate() -> float:
    """edot where creep and brittle failure coincide: eps_b/((m+1) t_ref)."""
    return BREAKING_STRAIN / ((CREEP_EXPONENT + 1.0) * CREEP_TIME_S)


def analytic_brittle_force() -> float:
    return BOND_STIFFNESS * BREAKING_STRAIN * L0


def analytic_creep_force(edot: float) -> float:
    m = CREEP_EXPONENT
    return (BOND_STIFFNESS * L0 * BREAKING_STRAIN ** (m / (m + 1.0))
            * (((m + 1.0) * CREEP_TIME_S) ** (1.0 / (m + 1.0)))
            * (edot ** (1.0 / (m + 1.0))))


def build_single_bond(arm: str) -> tuple[IceDEM3D, list[int]]:
    """Build a 2x2x2 engine and reduce it to a single bond (2 active particles)."""
    mw = arm.startswith("maxwell")
    creep = arm.endswith("creep")
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=917.0,
        bond_stiffness=BOND_STIFFNESS, breaking_strain=BREAKING_STRAIN,
        drag=0.0, tool_speed=1e-6, tool_gap=1.0, dt=DT,
        fix_x_edges=False, contact_damping=0.0,
        bond_model="maxwell_wiechert" if mw else "linear",
        bond_equilibrium_fraction=EQUILIBRIUM_FRACTION,
        bond_delayed_elastic_fraction=DELAYED_FRACTION,
        bond_relaxation_time_s=RELAXATION_TIME_S,
        bond_creep_exponent=CREEP_EXPONENT if creep else 0,
        bond_creep_time_s=CREEP_TIME_S,
    )
    sim = IceDEM3D(cfg)
    p0 = sim.pairs[0].tolist()
    keep = [p0[0], p0[1]]
    others = [i for i in range(len(sim.positions)) if i not in keep]
    # Park the six unused particles far away so no contact pair survives.
    sim.initial_positions[others, :] = torch.tensor(
        [[10.0, 10.0, 10.0]] * len(others), dtype=sim.dtype)
    sim.pairs = torch.tensor([p0], dtype=torch.long, device=sim.device)
    ref = torch.tensor([0.0, 0.0, L0], dtype=sim.dtype, device=sim.device)
    sim.rest_lengths = torch.tensor([L0], dtype=sim.dtype, device=sim.device)
    sim.reference_bond_vectors = ref.clone().unsqueeze(0)
    sim.initial_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=sim.dtype, device=sim.device)
    sim.bonded = torch.ones(1, dtype=torch.bool, device=sim.device)
    sim.alive = sim.bonded.clone()
    sim.bond_indices = sim.pairs
    sim.failure_mode = torch.zeros(1, dtype=torch.uint8, device=sim.device)
    sim.failure_time_s = torch.full((1,), -1.0, dtype=sim.dtype, device=sim.device)
    sim.failure_extension_m = torch.zeros(1, dtype=sim.dtype, device=sim.device)
    sim.failure_energy_J = torch.zeros(1, dtype=sim.dtype, device=sim.device)
    sim.bond_damage = torch.zeros(1, dtype=sim.dtype, device=sim.device)
    sim._bond_elongation_prev = torch.zeros(1, dtype=sim.dtype, device=sim.device)
    nb = sim.bond_law.n_branches if sim.bond_law is not None else 0
    sim.branch_offsets = torch.zeros((1, nb), dtype=sim.dtype, device=sim.device)
    sim._bond_keys = torch.tensor([p0[0] * 8 + p0[1]], dtype=torch.long, device=sim.device)
    sim.initial_positions[keep[0]] = torch.tensor([0, 0, 0], dtype=sim.dtype, device=sim.device)
    sim.initial_positions[keep[1]] = torch.tensor([0, 0, L0], dtype=sim.dtype, device=sim.device)
    sim.positions = sim.initial_positions.clone()
    sim.velocities.zero_()
    return sim, keep


def run_one(arm: str, edot: float) -> dict:
    """Drive one bond at strain rate ``edot``; return the failure record."""
    sim, keep = build_single_bond(arm)
    t = 0.0
    i, j = sim.pairs.T
    for step in range(200_000):
        d = edot * t * L0
        sim.positions[keep[1], 2] = L0 + d
        sim.time = t
        ext = (torch.linalg.vector_norm(sim.positions[j] - sim.positions[i], dim=1)
               - sim.rest_lengths)
        # The engine's constitutive call sites, in the engine's own order.
        bond_mag = sim._bond_force(ext, update_history=True)
        sim._commit_fracture(ext)
        if sim.config.bond_creep_exponent > 0:
            sim._commit_creep_damage(bond_mag, ext)
        if not bool(sim.alive.any()):
            return {
                "arm": arm, "strain_rate_s-1": float(edot),
                "F_f_N": float(bond_mag[0]),
                "failure_mode": int(sim.failure_mode[0]),
                "t_f_s": float(t + DT),
                "strain_at_failure": float(ext[0] / L0),
                "damage_at_failure": float(sim.bond_damage[0]),
                "n_steps": int(step + 1),
                "dt_s": float(DT),
            }
        t += DT
    return {"arm": arm, "strain_rate_s-1": float(edot), "F_f_N": None,
            "failure_mode": None, "t_f_s": None,
            "strain_at_failure": float(edot * t / 1.0),
            "damage_at_failure": float(sim.bond_damage[0]),
            "n_steps": 200_000, "dt_s": float(DT)}


def power_law_slope(rates: list[float], strengths: list[float]) -> float:
    xs = [math.log10(r) for r in rates]
    ys = [math.log10(s) for s in strengths]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="single creep rate, smoke test for CI")
    args = parser.parse_args()
    torch.set_num_threads(1)

    arms = ("linear", "linear+creep", "maxwell_wiechert", "maxwell_wiechert+creep")
    creep_rates = CREEP_RATES[:1] if args.quick else CREEP_RATES
    brittle_rates = BRITTLE_RATES[:1] if args.quick else BRITTLE_RATES

    runs: list[dict] = []
    for arm in arms:
        rates = creep_rates + brittle_rates
        for rate in rates:
            row = run_one(arm, rate)
            runs.append(row)
            ff = row["F_f_N"]
            print(f"{arm:24s} edot={rate:6.3f}  F_f={ff if ff is not None else float('nan'):9.4f} N  "
                  f"mode={row['failure_mode']}  t_f={row['t_f_s']}  steps={row['n_steps']}")

    summary: dict[str, dict] = {}
    for arm in arms:
        rows = [r for r in runs if r["arm"] == arm and r["F_f_N"] is not None]
        creep_rows = [r for r in rows if r["failure_mode"] == 4]
        brittle_rows = [r for r in rows if r["failure_mode"] == 1]
        cc = [r["strain_rate_s-1"] for r in creep_rows]
        cs = [r["F_f_N"] for r in creep_rows]
        summary[arm] = {
            "creep_rates_s-1": cc,
            "creep_F_f_N": cs,
            "creep_fitted_exponent": power_law_slope(cc, cs) if len(cc) >= 2 else None,
            "brittle_rates_s-1": [r["strain_rate_s-1"] for r in brittle_rows],
            "brittle_F_f_N": [r["F_f_N"] for r in brittle_rows],
            "creep_failed_bonds": len(creep_rows),
            "brittle_failed_bonds": len(brittle_rows),
        }

    payload = {
        "campaign": "ice_bond_single_bond_engine",
        "specimen": {"n_particles": 2, "n_bonds": 1, "radius_m": RADIUS,
                     "rest_length_m": L0,
                     "unrelaxed_bond_stiffness_N_m": BOND_STIFFNESS,
                     "breaking_strain": BREAKING_STRAIN, "dt_s": DT},
        "bond_law": {"equilibrium_fraction": EQUILIBRIUM_FRACTION,
                     "delayed_elastic_fraction": list(DELAYED_FRACTION),
                     "relaxation_time_s": list(RELAXATION_TIME_S),
                     "creep_exponent": CREEP_EXPONENT,
                     "creep_time_s": CREEP_TIME_S},
        "analytic_references": {
            "ductile_exponent_1_over_m_plus_1": analytic_exponent(),
            "creep_brittle_crossover_rate_s-1": analytic_crossover_rate(),
            "brittle_force_N": analytic_brittle_force(),
            "creep_force_N_by_rate": {f"{r:g}": analytic_creep_force(r)
                                     for r in CREEP_RATES},
        },
        "runs": runs,
        "summary": summary,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("quick.json" if args.quick else "verification.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\narm                          creep exponent   brittle F_f (N)")
    for arm, row in summary.items():
        exp = row["creep_fitted_exponent"]
        exp_s = f"{exp:+.4f}" if exp is not None else "   n/a"
        bf = ", ".join(f"{v:.4f}" for v in row["brittle_F_f_N"]) or "n/a"
        print(f"{arm:24s} {exp_s}            {bf}")
    print(f"\nanalytic: exponent={analytic_exponent():.6f}  "
          f"crossover={analytic_crossover_rate():.4f} 1/s  "
          f"brittle F_b={analytic_brittle_force():.4f} N")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
