"""Gate tests for the round-24 2-D disk plane-shear device.

These tests do not measure ``mu(I)`` (that is the benchmark's job and it needs
cluster runs).  They pin the things the round-24 verdict *depends* on, so that a
future round cannot silently break the device:

1. **geometry**: the two pinned rows are blocking (a free disk cannot squeeze
   between wall grains) and the free packing sits strictly inside the gap;
2. **kinematics**: ``I = gammadot d / sqrt(P/rho)`` is invertible and the wall
   speed that realises it is consistent with the measured shear rate;
3. **wall bookkeeping**: the two rows carry equal and opposite traction, which is
   the invariant the primary ordinate (``mu_wall``) is built on;
4. **servo**: the constant-stress servo closes the gap when ``P`` is raised;
5. **determinism**: same seed -> bit-identical state digest, different seed ->
   different packing;
6. **contract with the pre-registration**: the judged inertial numbers are inside
   the archived law's validity, the tolerance is the ledger's 10 %, and the reference
   constants are read from the archive instead of being retyped here.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sys

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
for p in (REPO / "src", BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plane_shear_2d import (  # noqa: E402
    PlaneShear2DCase,
    RoughWallShearCell,
    _wall_row,
    simulate,
)
import reference_archive  # noqa: E402

SMALL = dict(width=0.12, gap=0.08, n_particles=80)


def small_case(**kw) -> PlaneShear2DCase:
    base = PlaneShear2DCase(**SMALL)
    return replace(base, **kw) if kw else base


# --------------------------------------------------------------------------- #
# 1. geometry
# --------------------------------------------------------------------------- #
def test_wall_row_is_blocking():
    """Wall grains must be touching: a free disk of the same size cannot pass."""
    case = small_case()
    pos, rad = _wall_row(case, 1.0 * case.radius, 0.0)
    gaps = np.diff(np.sort(pos[:, 0]))
    spacing = gaps.min()
    assert spacing > 2.0 * case.radius, "wall grains overlap each other"
    assert spacing < 2.1 * case.radius, "wall grains are too far apart to block a free disk"
    assert np.allclose(rad, case.radius)
    assert pos[:, 0].max() <= case.width


def test_free_disks_sit_inside_the_gap():
    case = small_case()
    cell = RoughWallShearCell(case)
    free_y = cell.system.pos[cell.free_idx, 1]
    assert np.allclose(cell.gap_eff, case.gap)
    assert free_y.min() > cell.y_bottom + case.radius
    assert free_y.max() < cell.y_top - case.radius


def test_lattice_capacity_is_checked():
    with pytest.raises(ValueError):
        RoughWallShearCell(small_case(n_particles=400))


def test_geometry_validation():
    with pytest.raises(ValueError):
        small_case(gap=2.0 * 0.01).validate()
    with pytest.raises(ValueError):
        small_case(I_target=0.0).validate()


# --------------------------------------------------------------------------- #
# 2. kinematics
# --------------------------------------------------------------------------- #
def test_inertial_number_roundtrip():
    case = small_case()
    for I in (0.001, 0.01, 0.1, 0.3):
        gd = case.shear_rate_for(I)
        back = gd * case.diameter / case.velocity_scale
        assert back == pytest.approx(I, rel=1e-12)


def test_wall_speed_realises_the_shear_rate():
    case = small_case(I_target=0.05)
    cell = RoughWallShearCell(case)
    assert cell.shear_rate == pytest.approx(case.gammadot, rel=1e-9)
    assert cell.v_wall == pytest.approx(0.5 * case.gammadot * case.gap, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. wall bookkeeping (the primary ordinate's invariant)
# --------------------------------------------------------------------------- #
def test_pinned_rows_do_not_move_and_free_disks_stay_periodic():
    case = small_case(I_target=0.05)
    cell = RoughWallShearCell(case)
    y_bottom = cell.y_bottom
    cell.advance(200)
    assert np.allclose(cell.system.pos[cell.bottom_idx, 1], y_bottom)
    assert np.allclose(cell.system.omega[cell.pinned], 0.0)
    x_free = cell.system.pos[cell.free_idx, 0]
    assert x_free.min() >= 0.0 and x_free.max() < case.width


def test_wall_rows_carry_opposite_traction():
    """Newton's third law (exact) plus the sign structure the ordinate needs.

    ``mu_wall`` is built as ``|F_t| / |F_n|`` summed **per row**, because the row
    sums of a confined packing are opposite in sign.  What is asserted here is
    (a) the total force over all disks vanishes (the pair loop is equal-and-
    opposite) and (b) the top row is loaded upwards while the bottom row is
    loaded downwards, i.e. the material transmits compression.  The magnitudes
    are *not* asserted to balance to a tight tolerance: while the servo is still
    compacting the packing accelerates vertically, and the residual is the
    vertical momentum rate, not an error of the device (the production runs
    require quiescence separately, see ``ke_settle``).
    """
    case = small_case(I_target=0.05)
    cell = RoughWallShearCell(case)
    cell.v_wall = 0.0
    cell._set_wall_kinematics(0.0)
    cell.advance(4000)
    force = cell.system._contact_force
    scale = max(float(np.abs(force).max()), 1e-12)
    assert abs(float(np.sum(force))) < 1e-9 * scale * len(force)
    tops, bots = [], []
    for _ in range(40):
        cell.advance(5)
        force = cell.system._contact_force
        tops.append(float(np.sum(force[cell.top_idx, 1])))
        bots.append(float(np.sum(force[cell.bottom_idx, 1])))
    fy_top = float(np.mean(tops))
    fy_bot = float(np.mean(bots))
    assert fy_top > 0.0, "the top row must be pushed up by the confined packing"
    assert fy_bot < 0.0, "the bottom row must be pushed down by the confined packing"
    # same order of magnitude (a factor 3 is far below the factor needed to make
    # the two-row average meaningless)
    assert 1.0 / 3.0 < abs(fy_top) / abs(fy_bot) < 3.0


def test_phi_definitions_are_ordered():
    case = small_case()
    cell = RoughWallShearCell(case)
    cell.advance(10)
    m = cell.measure()
    assert 0.0 < m["phi_cell"] < m["phi_gap"] < 1.0


def test_contacts_are_admissible_after_compaction():
    case = small_case()
    cell = RoughWallShearCell(case)
    cell.v_wall = 0.0
    cell._set_wall_kinematics(0.0)
    cell.advance(3000)
    contacts = cell.system.contacts
    overlap = np.asarray(contacts["overlap"])
    assert overlap.size > 0, "no contacts: the packing is not compressed"
    assert overlap.max() < 0.02 * case.diameter, "overlap exceeds the rigid-grain bound"


# --------------------------------------------------------------------------- #
# 4. servo
# --------------------------------------------------------------------------- #
def test_servo_closes_the_gap_when_pressure_is_raised(case_pairs=None):
    def settle(normal_stress: float) -> float:
        case = small_case(I_target=0.05, normal_stress=normal_stress, settle_time_units=8.0,
                          shear_settle_strain=0.05, strain_measure=0.05, sample_every=200)
        out = simulate(case)
        return out["gap_eff_final"]

    soft = settle(2.0e3)
    stiff = settle(2.0e4)
    assert stiff < soft, "a larger confining stress must produce a thinner gap"


# --------------------------------------------------------------------------- #
# 5. determinism
# --------------------------------------------------------------------------- #
def _digest(seed: int) -> str:
    case = small_case(seed=seed, settle_time_units=2.0, shear_settle_strain=0.05,
                      strain_measure=0.05, sample_every=100)
    out = simulate(case)
    return out["config_digest"]


def test_same_seed_is_bit_identical():
    assert _digest(0) == _digest(0)


def test_different_seed_gives_a_different_packing():
    assert _digest(0) != _digest(7)


# --------------------------------------------------------------------------- #
# 6. contract with the pre-registration and the archive
# --------------------------------------------------------------------------- #
def test_prereg_matches_the_code():
    prereg = json.loads((BENCH / "prereg_r24.json").read_text())
    case = PlaneShear2DCase()
    assert prereg["device"]["diameter_m"] == pytest.approx(case.diameter)
    assert prereg["device"]["normal_stress_Pa"] == pytest.approx(case.normal_stress)
    assert prereg["tolerance"]["mu"] == pytest.approx(0.10)
    assert prereg["tolerance"]["phi"] == pytest.approx(0.10)
    assert prereg["device"]["mu_particle_particle"] == pytest.approx(case.mu_p)
    assert set(prereg["predictions"]) == {"P1", "P2", "P3", "P4", "P5"}
    assert prereg["decision_rule"]["acceptance"]


def test_judged_inertial_numbers_are_inside_the_archived_validity():
    prereg = json.loads((BENCH / "prereg_r24.json").read_text())
    law = reference_archive.archived_law()
    judged = prereg["device"]["inertial_numbers_judged"]
    assert judged, "no judged point"
    for I in judged:
        assert I <= law.i_saturation
        assert law.mu(I) is not None and law.phi(I) is not None
    for I in prereg["device"]["inertial_numbers_reported_not_judged"]:
        assert law.mu(I) is None, "an extrapolation must never be judgeable"


def test_reference_constants_come_from_the_archive():
    law = reference_archive.archived_law()
    assert law.i_saturation == pytest.approx(0.2)
    assert law.mu(0.0) == pytest.approx(law.mu_min)
    assert law.mu(0.1) == pytest.approx(law.mu_min + 0.1 * law.b)
    assert law.phi(0.0) == pytest.approx(law.nu_max)
    # the 2-D disk values this round is judged against
    assert law.mu_min == pytest.approx(0.22, abs=5e-3)
    assert law.b == pytest.approx(1.0, abs=5e-3)
    assert law.nu_max == pytest.approx(0.81, abs=5e-3)
    assert law.a == pytest.approx(0.30, abs=5e-3)