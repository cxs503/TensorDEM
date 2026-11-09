"""Runnable example of the integrated DEM contact-physics module.

Runs the three matured modules (``P121W5-N02`` bond, ``P121W5-N05`` ridge
statistics, ``P121W5-N08`` lubrication-contact coupling) on one operating point
each in three physical regimes, and prints the numbers a user is expected to
read off.  No files are written, no plotting is required -- it is the smoke test
quoted by the module documentation.

Usage
-----
``PYTHONPATH=src python3 examples/contact_physics_demo.py``
"""
from __future__ import annotations

import json

from tensordem.contact_physics import (
    ContactPhysicsMarch,
    OperatingPoint,
    RidgeContactStatistics,
    RidgeMorphologyCase,
    TemperatureDependentBond,
)
from tensordem.interface_thermal.coupling import IceSteelContact
from tensordem.interface_thermal.lubrication import MeltLubricationModel
from tensordem.ridge_dem import RidgeKeel, RidgeMaterial


def cold_sliding_contact() -> dict:
    """Cold ice-steel sliding contact (no melt film): mu(T) and bond creep."""
    point = OperatingPoint(
        contact=IceSteelContact(cells=61),
        strain_rate_1_s=1.0e-5, dt_s=2.0e-4, statistics_stride=10)
    summary = ContactPhysicsMarch(point).run(200)
    return {
        "regime": "cold_sliding_contact",
        "interface_temperature_c": summary["final"]["interface_temperature_c"],
        "mu": summary["final"]["mu"],
        "film_thickness_m": summary["final"]["film_thickness_m"],
        "bond_force_mean_n": summary["final"]["bond_force_mean_n"],
        "damage_max": summary["final"]["damage_max"],
        "ledger_error": summary["energy"]["ledger_error"],
        "realised_porosity": summary["statistics"]["morphology"]["realised_porosity"],
        "mean_coordination": summary["statistics"]["morphology"]["mean_coordination"],
        "strong_load_fraction":
            summary["statistics"]["force_network"]["strong_load_fraction"],
    }


def melting_contact() -> dict:
    """Interface pinned at the melting point: Stefan film grows, mu -> mu_wet."""
    contact = IceSteelContact(
        cells=61,
        lubrication=MeltLubricationModel(mu_base=0.10, t_onset=-3.0, mu_wet=0.02))
    point = OperatingPoint(contact=contact, strain_rate_1_s=1.0e-5, dt_s=2.0e-4)
    summary = ContactPhysicsMarch(point).run(300)
    return {
        "regime": "melting_contact",
        "interface_temperature_c": summary["final"]["interface_temperature_c"],
        "mu": summary["final"]["mu"],
        "film_thickness_m": summary["final"]["film_thickness_m"],
        "film_normal_stiffness_n_m": summary["final"]["film_normal_stiffness_n_m"],
        "contact_stiffness_n_m": contact.contact_stiffness_n_m,
        "ledger_error": summary["energy"]["ledger_error"],
    }


def bond_at_temperature() -> dict:
    """The N02 temperature factor on its own (no marching needed)."""
    bond = TemperatureDependentBond.from_unrelaxed_stiffness(5.0e6)
    return {
        "regime": "bond_strength_vs_temperature",
        "strength_factor": {f"{t:+.0f}C": round(bond.strength_factor(t), 6)
                            for t in (-2.0, -5.0, -10.0, -20.0, -30.0)},
        "reference_force_n": {f"{t:+.0f}C": round(bond.reference_force(t), 3)
                              for t in (-2.0, -5.0, -10.0, -20.0, -30.0)},
        "relaxation_modulus_n_m": {f"{t:g}s": round(bond.relaxation_modulus(t), 3)
                                  for t in (0.0, 0.5, 5.0, 60.0)},
    }


def ridge_morphology() -> dict:
    """The N05 morphology / contact-network statistics on their own."""
    case = RidgeMorphologyCase(
        keel=RidgeKeel(keel_depth=8.0, keel_width=6.0, porosity=0.30),
        material=RidgeMaterial(particle_radius=0.30))
    statistics = RidgeContactStatistics(case)
    morphology = statistics.morphology()
    return {
        "regime": "ridge_morphology",
        "realised_porosity": morphology["realised_porosity"],
        "n_particles": int(morphology["n_particles"]),
        "n_bonds": int(morphology["n_bonds"]),
        "mean_coordination": morphology["mean_coordination"],
        "keel_angle_deg": morphology["keel_angle_deg"],
        "coordination_sweep": statistics.coordination_sweep()[:4],
    }


def main() -> None:
    results = [cold_sliding_contact(), melting_contact(), bond_at_temperature(),
               ridge_morphology()]
    for record in results:
        print(json.dumps(record, indent=2, sort_keys=True))
    print("\nAll four regimes ran; see docs/contact_physics_module.md for the "
          "acceptance table.")


if __name__ == "__main__":
    main()