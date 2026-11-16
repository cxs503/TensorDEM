#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO: Beverloo law validation campaign (m20707).

Uses the ``tensordem.silo_flow`` module (Cundall-Strack wall friction,
disordered packing, energy ledger) to measure the steady mass flow rate
through a flat-bottom silo orifice and compare with the Beverloo (1961)
correlation:

    W = C · rho_b · sqrt(g) · (D - k·d)^(5/2)

Reference coefficients (from ref_beverloo_1961.json):
    C = 0.58  (range 0.55-0.60 for spherical particles)
    k = 1.4   (range 1.2-1.6)

Acceptance (predeclared, do not tune after seeing results):
  * primary case D5  : |W - W_ref| / W_ref < 10 %
  * law check D4, D6 : same 10 % band, reported but not decisive
    (D/d = 4 is at the known lower validity limit of Beverloo)
  * exponent check   : |p - 5/2| <= 0.20 from the fixed-k log-log fit
  * gravity scaling  : |W(g/4)/W(g) - 0.5| < 0.10
  * steadiness       : window-rate CV < 0.05

Presets
-------
``full``  (d = 3 mm, kn = 5e3) is the campaign that produces the verdict.
``pilot`` (d = 2 mm, kn = 2e3) reproduces the cheaper scoping study that was
originally quoted in ``docs/m20707_beverloo_verification.md``; it is kept in
the repository so those numbers stay reproducible instead of being an
unreachable, undocumented side experiment.

Case matrix (preset ``full``)
-----------------------------
| id       | D/d | mu_p | purpose                    |
|----------|-----|------|----------------------------|
| D4       | 4   | 0.50 | law check point            |
| D5       | 5   | 0.50 | primary case (verdict)     |
| D6       | 6   | 0.50 | law check point            |
| D5_mu40  | 5   | 0.40 | friction sensitivity       |
| D5_mu55  | 5   | 0.55 | friction sensitivity       |
| D5_g4    | 5   | 0.50 | gravity scaling (g/4)      |

Shared (full): d = 3 mm, rho_p = 2500 kg/m³, mu_r = 0.05, e = 0.30,
kn = 5e3 N/m, kt/kn = 0.5, R_silo = 0.025 m, H_fill = 0.06 m, wall_mu = 0,
seed = 20707.

Usage
-----
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --list
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D5
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --preset pilot --all
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --preset both --shard 2/6
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tensordem.silo_flow.beverloo import BEVERLOO_C, BEVERLOO_K, DischargeCase
from tensordem.silo_flow.kernel import Material, SiloGeometry
from tensordem.silo_flow.packing import build_bed

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"

# Results live inside the benchmark package (not under runs/) so that the
# machine-readable evidence travels with the code that produced it.
RESULTS_ROOT = ROOT / "benchmarks" / "m20707_beverloo" / "results"

# ---------------------------------------------------------------------------
# Preset ``full``: d = 3 mm — the campaign that carries the verdict
# ---------------------------------------------------------------------------
BASE = dict(
    particle_diameter=0.003,     # d = 3 mm
    silo_radius=0.025,           # R = 25 mm
    fill_height=0.060,           # H = 60 mm
    wall_mu=0.0,                 # smooth walls (Beverloo glass-bead experiments)
    seed=20707,
    settle_time=0.08,
    discard_time=0.15,
    window_time=0.20,
    n_windows=6,
    kn=5.0e3,                    # moderate stiffness
    mu=0.50,                     # inter-particle friction (primary)
    mu_rolling=0.05,
)

CASES: Dict[str, dict] = {
    "D4":      dict(orifice_diameter=0.012),    # D/d = 4
    "D5":      dict(orifice_diameter=0.015),    # D/d = 5, primary
    "D6":      dict(orifice_diameter=0.018),    # D/d = 6
    "D5_mu40": dict(orifice_diameter=0.015, mu=0.40),
    "D5_mu55": dict(orifice_diameter=0.015, mu=0.55),
    "D5_g4":   dict(orifice_diameter=0.015),    # gravity handled in build_case
}

# ---------------------------------------------------------------------------
# Preset ``pilot``: d = 2 mm — scoping study, documented since day one but
# previously unreachable from this script (no preset existed, so the numbers
# in the verification note could not be reproduced from the repository).
# ---------------------------------------------------------------------------
BASE_PILOT = dict(BASE)
BASE_PILOT.update(
    particle_diameter=0.002,     # d = 2 mm
    silo_radius=0.012,           # R = 12 mm (silo D = 24 mm)
    fill_height=0.020,           # H = 20 mm
    settle_time=0.03,
    discard_time=0.05,
    window_time=0.08,
    n_windows=4,
    kn=2.0e3,                    # softer contacts, cheaper pilot
)

CASES_PILOT: Dict[str, dict] = {
    "D4":      dict(orifice_diameter=0.008),    # D/d = 4
    "D5":      dict(orifice_diameter=0.010),    # D/d = 5
    "D6":      dict(orifice_diameter=0.012),    # D/d = 6
    "D5_mu40": dict(orifice_diameter=0.010, mu=0.40),
    "D5_mu55": dict(orifice_diameter=0.010, mu=0.55),
    "D5_g4":   dict(orifice_diameter=0.010),    # gravity handled in build_case
}

PRESETS: Dict[str, Tuple[dict, Dict[str, dict]]] = {
    "full": (BASE, CASES),
    "pilot": (BASE_PILOT, CASES_PILOT),
}

DEFAULT_OUT = {name: RESULTS_ROOT / name for name in PRESETS}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_case(case_id: str, preset: str = "full") -> Tuple[DischargeCase, float]:
    """Return the (validated) case plus the gravity it must actually run at."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    base, cases = PRESETS[preset]
    if case_id not in cases:
        raise ValueError(
            f"case {case_id!r} is not defined for preset {preset!r}; "
            f"choose from {sorted(cases)}"
        )
    # DischargeCase carries no gravity field (the Material inside the kernel
    # does), so g is returned alongside and injected by run_one().
    params = dict(base)
    params.update(cases[case_id])
    gravity = 9.81 / 4.0 if case_id.endswith("g4") else 9.81
    return DischargeCase(**params), gravity


def analyse(case_id: str, preset: str, case: DischargeCase, gravity: float,
            res: dict, wall: float) -> dict:
    ref = json.loads(REF_PATH.read_text())
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])
    d = case.particle_diameter
    D = case.orifice_diameter
    rho_b = float(res["rho_bulk"])
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(gravity) * Deff ** 2.5
    W_got = float(res["flow_rate_kg_s"])
    rel = (W_got - W_ref) / W_ref if W_ref > 0 else float("nan")

    C_got = W_got / (rho_b * math.sqrt(gravity) * Deff ** 2.5) if Deff > 0 else float("nan")

    return {
        "case_id": case_id,
        "preset": preset,
        "wall_clock_s": wall,
        "git_sha": git_sha(),
        "host": platform.node(),
        "gravity": gravity,
        "case": {
            "orifice_diameter": D,
            "particle_diameter": d,
            "D_over_d": D / d,
            "silo_radius": case.silo_radius,
            "fill_height": case.fill_height,
            "mu": case.mu,
            "mu_rolling": case.mu_rolling,
            "wall_mu": case.wall_mu,
            "kn": case.kn,
            "seed": case.seed,
        },
        "result": {
            "n_bed": res["n_bed"],
            "rho_bulk": rho_b,
            "rho_bulk_measured": res.get("rho_bulk_measured", rho_b),
            "surface_height_m": res.get("surface_height_m"),
            "flow_rate_kg_s": W_got,
            "flow_rate_std_kg_s": res["flow_rate_std_kg_s"],
            "flow_rate_windows_kg_s": res["flow_rate_windows_kg_s"],
            "particles_exited": res["particles_exited"],
            "dt": res["dt"],
            "steps": res["steps"],
        },
        "beverloo": {
            "C_ref": C,
            "k_ref": k,
            "W_ref_kg_s": W_ref,
            "C_recovered": C_got,
            "Deff_m": Deff,
            "rel_err_pct": rel * 100.0,
            "abs_rel_err_pct": abs(rel) * 100.0,
            "pass_10pct": abs(rel) < 0.10,
            "pass_15pct": abs(rel) < 0.15,
        },
    }


def run_one(case_id: str, out_dir: Path, preset: str = "full") -> dict:
    case, gravity = build_case(case_id, preset)

    mat = Material(
        radius=0.5 * case.particle_diameter,
        density=2500.0,
        mu=case.mu,
        mu_rolling=case.mu_rolling,
        restitution=0.3,
        kn=case.kn,
        kt_ratio=0.5,
        gravity=gravity,
    )
    geom = SiloGeometry(
        cylinder_radius=case.silo_radius,
        cylinder_height=case.fill_height + 0.015,
        outlet_radius=0.5 * case.orifice_diameter,
    )

    t0 = time.time()
    flow = build_bed(
        geom, mat, case.fill_height,
        wall_mu=case.wall_mu, seed=case.seed,
        settle_time=case.settle_time, plug_outlet=True,
    )
    surface_height = flow.surface_height()
    rho_b = flow.bulk_density(case.fill_height)
    rho_b_meas = flow.bulk_density(surface_height)
    bed_mass = flow.bed_mass
    n_bed = flow.n

    # Open the orifice
    flow.plugged = False

    # Discard start-up
    n_discard = int(round(case.discard_time / flow.dt))
    for _ in range(n_discard):
        flow.step()

    # Measurement windows
    n_win = int(round(case.window_time / flow.dt))
    windows = []
    for _ in range(case.n_windows):
        m0 = flow.mass_exited
        n0 = flow.n_exited
        for _ in range(n_win):
            flow.step()
        windows.append({
            "mass_kg": flow.mass_exited - m0,
            "particles": flow.n_exited - n0,
            "time_s": n_win * flow.dt,
        })

    rates = np.array([w["mass_kg"] / w["time_s"] for w in windows])
    n_steady = max(1, len(rates) // 2)
    steady = rates[-n_steady:]

    wall = time.time() - t0

    res = {
        "n_bed": n_bed,
        "bed_mass_kg": bed_mass,
        "rho_bulk": rho_b,
        "rho_bulk_measured": rho_b_meas,
        "surface_height_m": surface_height,
        "dt": flow.dt,
        "steps": flow.steps,
        "n_windows": case.n_windows,
        "window_time_s": case.window_time,
        "discard_time_s": case.discard_time,
        "flow_rate_windows_kg_s": [float(r) for r in rates],
        "flow_rate_kg_s": float(np.mean(steady)),
        "flow_rate_std_kg_s": float(np.std(steady, ddof=1)) if len(steady) > 1 else 0.0,
        "flow_rate_steady_windows": n_steady,
        "particles_exited": int(sum(w["particles"] for w in windows)),
    }

    out = analyse(case_id, preset, case, gravity, res, wall)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case_id}.json"
    path.write_text(json.dumps(out, indent=2, default=float))

    bev = out["beverloo"]
    print(
        f"[{preset}/{case_id}] D/d={out['case']['D_over_d']:.1f} "
        f"mu={out['case']['mu']:.2f} g={gravity:.2f} "
        f"n_bed={res['n_bed']} rho_b={rho_b:.0f} "
        f"W={bev['C_recovered']:.3f}*rho*sqrt(g)*Deff^2.5 "
        f"= {res['flow_rate_kg_s']:.6f} kg/s "
        f"W_ref={bev['W_ref_kg_s']:.6f} kg/s "
        f"rel={bev['rel_err_pct']:+.1f}% "
        f"{'PASS' if bev['pass_10pct'] else 'FAIL'} "
        f"wall={wall:.0f}s",
        flush=True,
    )
    return out


def job_list(presets: List[str]) -> List[Tuple[str, str]]:
    """Ordered (preset, case_id) work list used by --all and --shard."""
    jobs: List[Tuple[str, str]] = []
    for preset in presets:
        for cid in sorted(PRESETS[preset][1]):
            jobs.append((preset, cid))
    return jobs


def shard_jobs(jobs: List[Tuple[str, str]], idx: int, nsh: int) -> List[Tuple[str, str]]:
    """Deterministic disjoint partition of the work list for cluster shards."""
    if nsh < 1 or not (0 <= idx < nsh):
        raise ValueError("require NSHARDS >= 1 and 0 <= INDEX < NSHARDS")
    return [job for k, job in enumerate(jobs) if k % nsh == idx]


def parse_shard(text: str) -> Tuple[int, int]:
    try:
        i_s, n_s = text.split("/")
        i, n = int(i_s), int(n_s)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError("shard must look like INDEX/NSHARDS") from exc
    if n < 1 or not (0 <= i < n):
        raise argparse.ArgumentTypeError("require 0 <= INDEX < NSHARDS")
    return i, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="D5")
    ap.add_argument("--preset", default="full", choices=sorted(PRESETS) + ["both"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", type=parse_shard, default=None,
                    help="process only jobs with index %% NSHARDS == INDEX")
    ap.add_argument("--out-dir", default=None,
                    help="override output directory (only valid with a single preset)")
    args = ap.parse_args(argv)

    presets = ["full", "pilot"] if args.preset == "both" else [args.preset]

    if args.list:
        for preset in presets:
            print(preset, " ".join(sorted(PRESETS[preset][1])))
        return 0

    def out_for(preset: str) -> Path:
        if args.out_dir is None:
            return DEFAULT_OUT[preset]
        if len(presets) > 1:
            return Path(args.out_dir) / preset
        return Path(args.out_dir)

    if args.shard is not None:
        idx, nsh = args.shard
        jobs = shard_jobs(job_list(presets), idx, nsh)
        if not jobs:
            print(f"shard {idx}/{nsh}: no jobs", flush=True)
            return 0
        print(f"shard {idx}/{nsh}: {jobs}", flush=True)
        for preset, cid in jobs:
            run_one(cid, out_for(preset), preset)
        return 0

    if args.all:
        for preset, cid in job_list(presets):
            run_one(cid, out_for(preset), preset)
    else:
        if len(presets) > 1:
            ap.error("--case requires a single preset")
        run_one(args.case, out_for(presets[0]), presets[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())