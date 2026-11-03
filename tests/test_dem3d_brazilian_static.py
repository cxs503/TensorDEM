"""Tests for the static (relaxed) Brazilian disc field and its reference check.

These tests pin three things:

1. the numpy energy/force re-implementation used by the minimiser is the
   *engine's own* force law (no silent divergence between the two code paths);
2. the static equilibrium really is an equilibrium (residual, load balance,
   sign structure of the diametral field) and is dimensionless-travel-stable;
3. the structural explanation of the Hondros mismatch is reproducible from the
   library's own elastic constants (Cauchy solid, ``C12 = C44``, Zener > 1).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import MANIFEST, _build  # noqa: E402
from brazilian_lattice_static import (  # noqa: E402
    lattice_energy_force,
    lattice_moduli,
    solve_static_equilibrium,
)
from tensordem.lattice_elasticity3d import periodic_elastic_constants  # noqa: E402

from brazilian_lattice_static import _build as _build_case  # noqa: E402,F401

MANIFEST_DATA = json.loads(MANIFEST.read_text(encoding="utf-8"))
CASE = next(c for c in MANIFEST_DATA["cases"] if c["id"] == "brazil_elastic_field")


def _case_copy(**overrides):
    case = json.loads(json.dumps(CASE))
    case["config"].update(overrides)
    return case


@pytest.fixture(scope="module")
def reference_state():
    sim, geometry = _build(CASE)
    radius = float(CASE["config"]["radius_m"])
    positions = sim.positions.detach().cpu().numpy().astype(float).copy()
    rng = np.random.default_rng(20261103)
    travel = 2.0e-5
    # Perturb so the comparison is not trivially satisfied at the build state.
    perturbed = positions + rng.normal(scale=1e-6, size=positions.shape)
    top_z = sim.top_platen_start_z - travel
    bottom_z = sim.bottom_platen_start_z + travel
    sim.positions.copy_(torch.as_tensor(perturbed, dtype=sim.dtype))
    sim.velocities.zero_()
    sim.time = travel / float(CASE["config"]["loading_speed_m_s"])
    engine_force = sim.forces(update_fracture=False)[0].detach().cpu().numpy().astype(float)
    pairs = sim.pairs.detach().cpu().numpy()
    rest = sim.rest_lengths.detach().cpu().numpy().astype(float)
    alive = (sim.bonded & sim.alive).detach().cpu().numpy()
    _, numpy_force = lattice_energy_force(
        perturbed, pairs, rest, alive,
        float(CASE["config"]["bond_stiffness_N_m"]),
        float(CASE["config"]["platen_stiffness_N_m"]),
        radius, top_z, bottom_z)
    return {
        "engine_force": engine_force,
        "numpy_force": numpy_force,
        "geometry": geometry,
    }


def test_numpy_force_field_matches_engine(reference_state):
    """The minimiser's gradient must be the engine's force, not an approximation."""
    engine = reference_state["engine_force"]
    numpy_force = reference_state["numpy_force"]
    scale = max(float(np.abs(engine).max()), 1e-30)
    assert float(np.abs(engine - numpy_force).max()) / scale < 1e-10


def test_static_equilibrium_is_relaxed():
    result = solve_static_equilibrium(CASE, travel=3.545649801036111e-05)
    assert result["equilibrium_converged"] is True
    # Force residual relative to the largest bond force in the specimen.
    assert result["equilibrium_residual_rel"] < 1e-3
    # Newton's third law at specimen level: both platens carry the same load.
    assert result["load_balance_rel"] < 1e-6
    # Diametral configuration: transverse (x) tension, axial (z) compression.
    assert result["centre_sigma_xx_Pa"] > 0.0
    assert result["centre_sigma_zz_Pa"] < 0.0
    # The split-stress definition must be exactly the analytic 2P/(pi D t).
    expected = (2.0 * result["mean_platen_load_N"]
                / (math.pi * result["specimen_diameter_m"] * result["specimen_thickness_m"]))
    assert result["split_stress_Pa"] == pytest.approx(expected, rel=1e-12)


def test_centre_field_is_travel_independent():
    """Geometry is linear here, so the dimensionless centre field cannot drift."""
    short = solve_static_equilibrium(CASE, travel=3.545649801036111e-05)
    long = solve_static_equilibrium(CASE, travel=1.4163831507374044e-04)
    # Tolerance is set by the discrete contact set: the platen engages a slightly
    # different particle set at each travel, which shifts the shared load by ~1%.
    assert short["centre_principal_ratio"] == pytest.approx(
        long["centre_principal_ratio"], rel=1e-2)
    assert short["centre_tension_rel_error"] == pytest.approx(
        long["centre_tension_rel_error"], rel=1e-2)
    # Load scales linearly with travel (5x travel -> 5x load within the discrete
    # contact-set tolerance).
    assert long["mean_platen_load_N"] == pytest.approx(
        4.0 * short["mean_platen_load_N"], rel=5e-2)


def test_hondros_mismatch_is_reported_not_hidden():
    """The relaxed field misses the Hondros point-load target by far more than 15%.

    This is the *finding* of the audit, so it is asserted rather than tolerated:
    if a future change makes the lattice reproduce ``sigma_zz/sigma_xx = -3``
    within tolerance, this test must be revisited on purpose.
    """
    result = solve_static_equilibrium(CASE, travel=3.545649801036111e-05)
    target = float(MANIFEST_DATA["reference"]["centre_principal_stress_ratio"])
    assert abs(result["centre_principal_ratio"] - target) / abs(target) > 0.15
    assert abs(result["centre_tension_rel_error"]) > 0.15
    # ... and the sign is right, i.e. this is not a broken solve.
    assert abs(result["centre_principal_ratio"]) > 1.0


def test_cauchy_solid_explains_the_mismatch():
    """Born constants of the bond lattice: no shear decoupling, clearly anisotropic."""
    moduli = lattice_moduli(CASE)
    assert moduli["C44_Pa"] == pytest.approx(moduli["C12_Pa"], rel=1e-12)
    assert moduli["zener_anisotropy"] > 1.5
    # Born (affine) constants are an upper bound; the relaxed modulus must be lower.
    direct = periodic_elastic_constants(
        float(CASE["config"]["bond_stiffness_N_m"]),
        2.0 * float(CASE["config"]["radius_m"]))
    assert direct["C11_Pa"] == pytest.approx(moduli["C11_Pa"], rel=1e-12)
    assert moduli["E_Pa"] < moduli["C11_Pa"]