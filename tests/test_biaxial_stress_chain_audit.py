"""Verification of the biaxial cell stress chain (CASE-DEM-BIAXIAL-FRICTION).

The plane-strain biaxial cell measures its stress state two independent ways:

* **boundary traction** -- wall contact force / current wall area, i.e. what an
  instrumented testing machine records;
* **Love-Weber virial** -- the internal (volume averaged) particle stress.

This module pins the three properties the case's numbers rest on:

1. the *legacy* consolidation (lateral servo walls only, platens position
   fixed) does **not** produce the isotropic ``sigma_v = sigma_h = 100 kPa``
   state the biaxial test definition assumes -- the measured ratio is
   ``K = sigma_v / sigma_h < 1`` (a K0 state);
2. ``axial_servo=True`` (platens pressure controlled during consolidation, new
   in this mission) *does* deliver ``K = 1`` within the 10 % band and is
   released before the displacement-controlled shear phase;
3. the two stress measures agree (``|dphi| <= 2 deg``) once the specimen is in
   a controlled (isotropic-servo) state, so the measured friction angle is not
   an artefact of the stress *definition*.

The numbers asserted here are invariants of the boundary conditions, not
fitted values; the anchor-scale record lives in
``artifacts/CASE-DEM-BIAXIAL-FRICTION/stress_chain_audit.json``.
"""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from tensordem.biaxial_shear_band import (
    _make_specimen,
    run_biaxial_shear_band,
)

torch.set_num_threads(1)

CONF = 100.0e3
SMALL = dict(confining_stress=CONF, columns=8, rows=16, radius=0.75,
             strain_rate=0.02, strain_target=0.03, snapshot_strain=0.01,
             servo_gain=3.0e-4, servo_velocity_limit=0.5, wall_friction=0.0,
             material_overrides={"freeze_bond_fraction": 0.35},
             record_snapshots=False)


@pytest.fixture(scope="module")
def runs():
    """Legacy vs axial-servo consolidation on the same small specimen."""
    legacy = run_biaxial_shear_band(consolidation_steps=400, axial_servo=False, **SMALL)
    servo = run_biaxial_shear_band(consolidation_steps=400, axial_servo=True, **SMALL)
    return {"legacy": legacy, "servo": servo}


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------

def test_axial_servo_defaults_to_off_for_backwards_compatibility():
    """The legacy boundary condition stays the default (published records)."""
    sig = inspect.signature(run_biaxial_shear_band)
    assert sig.parameters["axial_servo"].default is False
    sig_spec = inspect.signature(_make_specimen)
    assert sig_spec.parameters["axial_servo"].default is False


def test_consolidation_state_schema(runs):
    required = {
        "sigma_v", "sigma_h", "sigma_v_over_sigma_h",
        "sigma_v_internal", "sigma_h_internal",
        "sigma_v_relative_difference", "sigma_h_relative_difference",
        "platen_asymmetry", "target", "axial_servo",
        "isotropic_within_10pct",
    }
    for tag, res in runs.items():
        cs = res["consolidation_state"]
        assert required <= set(cs), f"{tag}: missing {required - set(cs)}"
        assert cs["target"] == pytest.approx(CONF)
        assert isinstance(cs["isotropic_within_10pct"], bool)
        assert cs["axial_servo"] is (tag == "servo")
        for key in ("sigma_v", "sigma_h", "sigma_v_over_sigma_h"):
            assert math.isfinite(cs[key]), f"{tag}:{key}"


def test_stress_path_records_both_platens(runs):
    """Both platen stresses are recorded so the equilibrium check is auditable."""
    for tag, res in runs.items():
        hist = res["stress_path"]
        for key in ("sigma_v_top", "sigma_v_mean", "platen_asymmetry",
                    "sigma_v_internal", "sigma_h_internal"):
            assert key in hist, f"{tag}: missing history key {key}"


# ---------------------------------------------------------------------------
# Q1: consolidation isotropy
# ---------------------------------------------------------------------------

def test_legacy_consolidation_is_a_k0_state_not_isotropic(runs):
    """Only the lateral walls are pressure controlled in the legacy cell."""
    cs = runs["legacy"]["consolidation_state"]
    assert cs["sigma_h"] == pytest.approx(CONF, rel=0.25)
    assert cs["sigma_v_over_sigma_h"] < 0.80
    assert cs["isotropic_within_10pct"] is False


def test_axial_servo_consolidation_is_isotropic(runs):
    """With pressure-controlled platens the end-of-consolidation state is K ~ 1."""
    cs = runs["servo"]["consolidation_state"]
    assert cs["sigma_v"] == pytest.approx(CONF, rel=0.15)
    assert cs["sigma_h"] == pytest.approx(CONF, rel=0.15)
    assert abs(cs["sigma_v_over_sigma_h"] - 1.0) <= 0.15
    assert cs["isotropic_within_10pct"] is True


def test_axial_servo_improves_isotropy_over_legacy(runs):
    k_legacy = runs["legacy"]["consolidation_state"]["sigma_v_over_sigma_h"]
    k_servo = runs["servo"]["consolidation_state"]["sigma_v_over_sigma_h"]
    assert abs(k_servo - 1.0) < 0.5 * abs(k_legacy - 1.0)


def test_platen_asymmetry_is_zero_by_construction_under_the_axial_servo(runs):
    """Both platens are regulated to the same pressure, so they must agree."""
    cs = runs["servo"]["consolidation_state"]
    assert abs(cs["platen_asymmetry"]) < 0.05


def test_axial_servo_is_released_before_the_shear_phase(runs):
    """A live axial servo would fight the displacement-controlled platen."""
    peak = runs["servo"]["peak"]
    assert peak["sigma_v"] > 1.5 * CONF, "platen still pressure controlled during shear"
    # and the deviator really is built up by the platen motion
    assert peak["deviator"] > 0.5 * CONF


# ---------------------------------------------------------------------------
# Q2: boundary traction vs internal (Love-Weber) stress
# ---------------------------------------------------------------------------

def test_stress_chain_consistent_at_the_consolidated_state(runs):
    """In a controlled state the wall traction and the virial must agree."""
    cs = runs["servo"]["consolidation_state"]
    assert abs(cs["sigma_v_relative_difference"]) <= 0.15
    assert abs(cs["sigma_h_relative_difference"]) <= 0.20


def test_friction_angle_is_stress_definition_independent(runs):
    """The two independent stress measures give the same friction angle."""
    for tag, res in runs.items():
        sd = res["stress_definitions"]
        gap = abs(sd["friction_angle_internal_deg"]
                  - res["peak"]["friction_angle_deg"])
        assert gap <= 2.0, f"{tag}: |dphi| = {gap:.2f} deg"


def test_legacy_unbalanced_state_shows_up_as_a_platen_asymmetry(runs):
    """The diagnostic has to be sensitive: the legacy transient is unbalanced."""
    hist = runs["legacy"]["stress_path"]["platen_asymmetry"]
    assert max(abs(v) for v in hist[:5]) > 0.05