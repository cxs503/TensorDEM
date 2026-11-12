"""Steady-state bed bookkeeping of the servo shear cell (m16427).

Round-1 of CASE-DEM-PLANE-SHEAR-MUIDI judged ``phi(I)`` with a bed solid
fraction built from ``plate_height``, the platen height **frozen at the end of
the settle phase**.  Because the settle phase is identical for every shear
rate (same seed, no shear), that quantity is I-independent by construction: an
I-independent "measured" phi was therefore not evidence about the material at
all.  These tests pin down both halves of the story:

* ``phi_bed_settled`` really is frozen (identical for two shear rates),
* ``phi_bed_steady`` / ``dilatancy`` are tail averages of the shear-phase
  platen height and do respond to the shear rate,
* the bookkeeping identities hold exactly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402


def _case(inertial_number: float) -> ServoShearCase:
    """Small, fast cell (10 particles) that still exercises both phases."""
    return ServoShearCase(
        inertial_number=inertial_number,
        mu=0.5,
        mu_rolling=0.05,
        restitution=0.3,
        kn=2.0e4,
        normal_stress=2.0e4,
        shear_strain=0.1,
        settle_steps=400,
        min_shear_steps=150,
        seed=5279,
        radius=0.005,
        density=2500.0,
        box_width=0.04,
        box_depth=0.02,
        fill_height=0.03,
    )


@pytest.fixture(scope="module")
def two_rates():
    slow = simulate_servo_shear(_case(1e-2))
    fast = simulate_servo_shear(_case(2e-1))
    return slow, fast


def test_legacy_plate_height_is_the_settled_height(two_rates):
    slow, _ = two_rates
    assert slow["plate_height_settled"] == pytest.approx(slow["plate_height"])
    assert slow["plate_height_steady"] > 0.0
    assert slow["plate_height_steady"] != pytest.approx(slow["plate_height"])
    # a few percent at most -- a sanity band on the servo drift
    assert abs(slow["dilatancy"]) < 0.2


def test_solid_fraction_identities(two_rates):
    slow, _ = two_rates
    v_part = 4.0 / 3.0 * math.pi * slow["radius"] ** 3
    n = slow["n_particles"]
    area = slow["box_volume"] / slow["plate_height"]  # W*D
    phi_settled = n * v_part / (area * slow["plate_height_settled"])
    phi_steady = n * v_part / (area * slow["plate_height_steady"])
    assert slow["phi_bed_settled"] == pytest.approx(phi_settled, rel=1e-12)
    assert slow["phi_bed_steady"] == pytest.approx(phi_steady, rel=1e-12)
    assert slow["dilatancy"] == pytest.approx(
        slow["plate_height_steady"] / slow["plate_height_settled"] - 1.0, rel=1e-12
    )


def test_settled_bed_fraction_is_rate_independent_by_construction(two_rates):
    """The round-1 metric: identical for both rates -- documents the defect."""
    slow, fast = two_rates
    assert slow["phi_bed_settled"] == pytest.approx(fast["phi_bed_settled"], rel=1e-12)
    assert slow["plate_height_settled"] == pytest.approx(
        fast["plate_height_settled"], rel=1e-12
    )


def test_steady_bed_fraction_responds_to_shear_rate(two_rates):
    """The round-2 metric: the sheared platen height is rate dependent."""
    slow, fast = two_rates
    assert slow["plate_height_steady"] != pytest.approx(
        fast["plate_height_steady"], rel=1e-9
    )
    assert slow["phi_bed_steady"] != pytest.approx(fast["phi_bed_steady"], rel=1e-9)
    # the rheology itself (not just the bookkeeping) still changes with I
    assert slow["mu_effective"] > fast["mu_effective"]