"""Ridge-keel granular-flow benchmark driver (protocol v1).

Runs a plane-strain DEM of a bow wedge penetrating a poly-disperse, Hertz-Mindlin
ice-rubble keel (optionally freeze-bonded), samples the force-chain network,
stress state, porosity and shear band in measurement windows, compares the wedge
resistance with the Lindqvist-family keel anchor, and writes
``results/ridge_keel_benchmark.json`` plus two figures into the same folder.

Usage
-----
    PYTHONPATH=src python benchmarks/pending/ridge_keel/run_benchmark.py

Every number in the JSON is produced by the run (no hand copying).  The status
flags follow ``prereg/p257w2_n05_ridge_keel.json``; a criterion that is not met
is written as ``false`` with its measured value, it is never relaxed.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.ridge_keel import (  # noqa: E402
    KeelSection,
    PowerLawPSD,
    RidgeKeelDEM,
    RubbleMaterial,
    WedgeIndenter,
    chain_lifetime,
    lindqvist_keel_resistance,
    stable_timestep,
)

RESULTS = os.path.join(HERE, "results")


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two undirected axes in degrees."""
    d = abs((a - b) % 180.0)
    return float(min(d, 180.0 - d))


def run_case(*, half_angle_deg: float, speed: float, bonded: bool,
             steps: int = 1500, window: int = 250, seed: int = 1) -> dict:
    section = KeelSection(keel_depth=4.0, keel_width=14.0, porosity=0.40)
    psd = PowerLawPSD(d_min=0.25, d_max=1.6, exponent=2.0)
    material = RubbleMaterial(young_modulus=2.0e8)
    dem = RidgeKeelDEM(section, psd, material, solid_fraction=0.62, seed=seed,
                       max_particles=220)
    dt = stable_timestep(material, float(np.min(dem.radius)))
    bonds = dem.install_bonds() if bonded else 0
    entry_depth = 2.0
    half = float(np.atleast_1d(section.half_width(entry_depth))[0])
    dem.wedge = WedgeIndenter(half_angle_deg, speed, apex_x=-half - 0.05,
                              apex_y=-entry_depth, friction=0.30)
    dem.ledger.initial_kinetic = dem.kinetic_energy()
    dem.ledger.initial_contact_elastic, dem.ledger.initial_bond_elastic = \
        dem.elastic_energy()
    snapshot = dem.snapshot()

    records = []
    strong_windows = []
    t0 = time.time()
    for k in range(steps):
        state = dem.step(dt)
        if (k + 1) % window == 0:
            chains = dem.force_chain_report()
            stress = dem.stress_report()
            record = {
                "step": k + 1,
                "time_s": state["time"],
                "penetration_m": dem.wedge.travelled,
                "wedge_force_N_per_m": state["wedge_force"],
                "kinetic_energy_J": state["kinetic"],
                "n_contacts": chains.get("n_contacts", 0),
                "chain_direction_deg": chains.get("chain_direction_deg", float("nan")),
                "chain_load_fraction": chains.get("load_fraction", float("nan")),
                "fabric_anisotropy": chains.get("fabric_anisotropy", float("nan")),
                "chain_sizes": chains.get("chain_sizes_top", []),
                "major_principal_deg": stress.get("major_direction_deg", float("nan")),
                "minor_principal_deg": stress.get("minor_direction_deg", float("nan")),
                "mobilised_friction_sin": stress.get("mobilised_friction_sin",
                                                     float("nan")),
                "bonds_alive": int(dem.bond_alive.sum()),
            }
            records.append(record)
            strong_windows.append(chains.get("strong_contact_keys", []))
    wall = time.time() - t0

    dem.ledger.final_kinetic = dem.kinetic_energy()
    dem.ledger.final_contact_elastic, dem.ledger.final_bond_elastic = \
        dem.elastic_energy()
    ledger = dem.ledger.as_dict()

    # steady window: second half of the penetration
    steady = [r for r in records if r["penetration_m"] >= 0.5 * dem.wedge.travelled]
    forces = np.array([r["wedge_force_N_per_m"] for r in steady]) if steady else np.array([])
    steady_force = float(np.mean(forces)) if forces.size else float("nan")
    force_std = float(np.std(forces)) if forces.size else float("nan")

    lifetime = chain_lifetime(strong_windows) if strong_windows else {}
    porosity = dem.porosity_report(snapshot)
    band = dem.shear_band_report(snapshot)
    stress = dem.stress_report()
    phi_deg = math.degrees(math.asin(min(abs(stress.get("mobilised_friction_sin", 0.0)),
                                         0.95)))
    expected_band = 45.0 + 0.5 * phi_deg
    theta_shear = float("nan")
    if band and "minor_principal_deg" in stress:
        theta_shear = _angle_diff(band["band_direction_deg"],
                                  stress["minor_principal_deg"])

    anchor = lindqvist_keel_resistance(section, half_angle_deg, hull_friction=0.60,
                                       solid_fraction=dem.measure_solid_fraction())
    anchor_force = anchor["resistance_N_per_m"]
    deviation = (abs(steady_force - anchor_force) / anchor_force
                 if anchor_force > 0 and np.isfinite(steady_force) else float("nan"))

    criteria = {
        "C1_particle_conservation": bool(dem.n_particles == snapshot["alive"].size),
        "C2_energy_closure": bool(abs(ledger["residual_fraction"]) < 0.05
                                  if np.isfinite(ledger["residual_fraction"]) else False),
        "C3_force_chain_angle": bool(
            np.isfinite(stress.get("major_direction_deg", float("nan")))
            and np.isfinite(records[-1]["chain_direction_deg"])
            and _angle_diff(records[-1]["chain_direction_deg"],
                            stress["major_direction_deg"]) < 15.0),
        "C4_shear_band_angle": bool(np.isfinite(theta_shear)
                                    and abs(theta_shear - expected_band) <= 10.0),
        "C5_chain_load_fraction": bool(records and records[-1]["chain_load_fraction"] >= 0.60),
        "C6_porosity_evolution": bool(porosity
                                      and -0.10 <= porosity["delta_porosity"] <= 0.20),
        "C7_lindqvist": bool(np.isfinite(deviation) and deviation < 0.30),
    }
    return {
        "case": {"half_angle_deg": half_angle_deg, "speed_m_s": speed,
                 "bonded": bonded, "seed": seed, "steps": steps},
        "assembly": {"n_grains": dem.n_particles, "bonds_installed": bonds,
                     "bonds_alive": int(dem.bond_alive.sum()),
                     "solid_fraction": dem.measure_solid_fraction(),
                     "dt_s": dt, "envelope_scale": dem.envelope_scale,
                     "mean_diameter_m": float(2.0 * np.mean(dem.radius))},
        "penetration_m": dem.wedge.travelled,
        "steady_force_N_per_m": steady_force,
        "steady_force_std_N_per_m": force_std,
        "lindqvist": anchor,
        "lindqvist_deviation": deviation,
        "energy_ledger": ledger,
        "porosity": porosity,
        "shear_band": band,
        "stress": stress,
        "mobilised_phi_deg": phi_deg,
        "expected_band_deg": expected_band,
        "theta_shear_deg": theta_shear,
        "chain_lifetime": lifetime,
        "records": records,
        "criteria": criteria,
        "wall_clock_s": wall,
    }


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    cases = []
    for bonded in (False, True):
        cases.append(run_case(half_angle_deg=30.0, speed=1.0, bonded=bonded))
    payload = {
        "protocol_id": "tensordem-ridge-keel-granular-flow-v1",
        "node": "p257w2-n05-ridge-keel-granular-flow",
        "cases": cases,
        "summary": {
            "criteria_primary": cases[0]["criteria"],
            "criteria_bonded": cases[1]["criteria"],
            "criteria_all_pass": bool(all(cases[0]["criteria"].values())),
        },
    }
    with open(os.path.join(RESULTS, "ridge_keel_benchmark.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    _figures(cases)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _figures(cases) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                    # pragma: no cover
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for case in cases:
        label = "bonded" if case["case"]["bonded"] else "unbonded"
        pen = [r["penetration_m"] for r in case["records"]]
        force = [r["wedge_force_N_per_m"] for r in case["records"]]
        axes[0].plot(pen, force, marker="o", ms=3, label=f"{label} DEM")
        axes[0].axhline(case["lindqvist"]["resistance_N_per_m"], ls="--", lw=1,
                        label=f"{label} Lindqvist")
        axes[1].plot(pen, [r["chain_load_fraction"] for r in case["records"]],
                     marker="s", ms=3, label=f"{label} chain load share")
    axes[0].set_xlabel("wedge penetration s [m]")
    axes[0].set_ylabel("resistance [N/m]")
    axes[0].set_title("ridge-keel resistance vs Lindqvist anchor")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.60, ls=":", color="k", lw=1)
    axes[1].set_xlabel("wedge penetration s [m]")
    axes[1].set_ylabel("share of load in strong contacts")
    axes[1].set_title("force-chain load fraction (C5 threshold 0.60)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "ridge_keel_forces.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())