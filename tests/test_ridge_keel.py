"""Verification of the ridge-keel granular-flow module.

Protocol id: ``tensordem-ridge-keel-granular-flow-v1``

Scope
-----
Two groups of checks:

1. **Closed-form / algebraic** (fast, deterministic): keel geometry, power-law
   particle-size distribution moments and inverse-CDF sampling, the wedge
   contact geometry, the Hertz timestep criterion, the energy-ledger algebra,
   the strong-contact network statistics on a synthetic contact set, chain
   lifetimes, and the Lindqvist keel-resistance anchor.
2. **Integration smoke** (short): a small keel is assembled, drivers are
   integrated, and the acceptance criteria that can be checked at this level
   are asserted -- grain-number conservation (C1) and finiteness of every
   exported number.  The full penetration campaign and its physics criteria
   (C2-C7) live in ``benchmarks/pending/ridge_keel/``; their current status is
   documented in ``docs/p257w2_n05_ridge_keel_granular_flow.md`` and is *not*
   passing yet (honest FAIL, see the report).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ridge_keel import (
    EnergyLedger,
    KeelSection,
    PowerLawPSD,
    RidgeKeelDEM,
    RubbleMaterial,
    WedgeIndenter,
    biaxial_friction_angle,
    chain_lifetime,
    lindqvist_keel_resistance,
    stable_timestep,
)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_keel_half_width_closes_to_the_tip():
    keel = KeelSection(keel_depth=8.0, keel_width=24.0)
    assert float(keel.half_width(0.0)) == pytest.approx(12.0)
    assert float(keel.half_width(8.0)) == pytest.approx(0.0)
    assert float(keel.half_width(4.0)) == pytest.approx(6.0)


def test_keel_cross_section_area_is_a_triangle():
    keel = KeelSection(keel_depth=5.0, keel_width=20.0)
    assert keel.cross_section_area == pytest.approx(50.0)


def test_keel_contains_respects_the_envelope():
    keel = KeelSection(keel_depth=6.0, keel_width=20.0)
    x = np.array([0.0, 7.0, 9.0, 5.0])
    y = np.array([-1.0, -1.0, -1.0, -7.0])
    inside = keel.contains(x, y)
    assert inside.tolist() == [True, True, False, False]


def test_keel_rejects_bad_geometry():
    with pytest.raises(ValueError):
        KeelSection(keel_depth=-1.0, keel_width=20.0)
    with pytest.raises(ValueError):
        KeelSection(keel_depth=5.0, keel_width=20.0, porosity=0.95)


# --------------------------------------------------------------------------- #
# particle size distribution
# --------------------------------------------------------------------------- #
def test_power_law_sample_stays_in_range_and_matches_moments():
    psd = PowerLawPSD(d_min=0.15, d_max=2.0, exponent=2.5)
    rng = np.random.default_rng(7)
    d = psd.sample(40000, rng)
    assert d.min() >= psd.d_min - 1e-12
    assert d.max() <= psd.d_max + 1e-12
    # E[d^2] from the closed form vs the sample (2 % tolerance on 4e4 draws)
    closed = psd.mean_square_diameter()
    assert np.mean(d ** 2) == pytest.approx(closed, rel=0.05)
    assert psd.mean_area() == pytest.approx(0.25 * math.pi * closed)


def test_power_law_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        PowerLawPSD(d_min=1.0, d_max=1.0)
    with pytest.raises(ValueError):
        PowerLawPSD(exponent=1.0)


def test_material_derived_quantities():
    m = RubbleMaterial(young_modulus=1.0e9, poisson_ratio=0.3)
    assert m.e_star == pytest.approx(1.0e9 / (2.0 * (1.0 - 0.09)))
    assert m.e_star_rigid() == pytest.approx(1.0e9 / (1.0 - 0.09))
    # k_t / k_n textbook ratio for equal spheres
    assert m.mindlin_ratio == pytest.approx(2.0 * 0.7 / 1.7)
    assert 0.0 < m.damping_ratio < 1.0


def test_stable_timestep_scales_with_grain_size():
    m = RubbleMaterial()
    small = stable_timestep(m, 0.075)
    large = stable_timestep(m, 0.30)
    # omega ~ sqrt(k/m) with k ~ sqrt(r) and m ~ r^2  =>  dt ~ r^{3/4}
    assert large > small
    assert large / small == pytest.approx((0.30 / 0.075) ** 0.75, rel=0.05)


# --------------------------------------------------------------------------- #
# wedge geometry
# --------------------------------------------------------------------------- #
def test_wedge_apex_contact_is_radial():
    wedge = WedgeIndenter(30.0, 1.0, apex_x=0.0, apex_y=0.0)
    pos = np.array([[0.3, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    radius = np.array([0.35, 0.35, 0.35])
    overlap, normal, active = wedge.contact(pos, radius)
    # a grain just ahead of the apex is in contact, one further away is not
    assert active[0] and not active[1]
    assert overlap[0] == pytest.approx(0.05, rel=1e-6)
    assert normal[0] == pytest.approx(np.array([1.0, 0.0]), abs=1e-6)
    # a grain behind the apex (inside the body) is pushed out radially
    assert active[2]


def test_wedge_face_normal_follows_the_half_angle():
    for angle in (15.0, 30.0, 45.0):
        wedge = WedgeIndenter(angle, 1.0)
        a = math.radians(angle)
        # a point placed perpendicular to the upper face at distance d < r
        direction = np.array([math.sin(a), math.cos(a)])
        pos = (0.9 * direction)[None, :]
        overlap, normal, active = wedge.contact(pos, np.array([1.0]))
        assert active[0]
        assert abs(abs(float(normal[0] @ direction)) - 1.0) < 1e-6
        assert overlap[0] == pytest.approx(0.1, rel=1e-6)


def test_wedge_advance_is_kinematic():
    wedge = WedgeIndenter(30.0, 2.0, apex_x=-1.0, apex_y=-3.0)
    wedge.advance(0.25)
    assert wedge.travelled == pytest.approx(0.5)
    assert wedge.apex[0] == pytest.approx(-0.5)
    assert wedge.apex[1] == pytest.approx(-3.0)


# --------------------------------------------------------------------------- #
# energy ledger algebra
# --------------------------------------------------------------------------- #
def test_energy_ledger_closes_on_consistent_numbers():
    ledger = EnergyLedger(wedge_work=1000.0, body_work=-100.0,
                          normal_dissipation=600.0, slip_dissipation=150.0,
                          final_kinetic=90.0, final_contact_elastic=60.0,
                          initial_kinetic=0.0, initial_contact_elastic=0.0)
    assert ledger.external_work == pytest.approx(900.0)
    assert ledger.dissipated == pytest.approx(750.0)
    assert ledger.residual == pytest.approx(0.0, abs=1e-9)
    assert ledger.residual_fraction == pytest.approx(0.0, abs=1e-12)


def test_energy_ledger_flags_a_missing_term():
    ledger = EnergyLedger(wedge_work=1000.0, final_kinetic=900.0)
    assert ledger.residual_fraction == pytest.approx(0.1)
    empty = EnergyLedger()
    assert not np.isfinite(empty.residual_fraction)


# --------------------------------------------------------------------------- #
# strong-contact network statistics on synthetic data
# --------------------------------------------------------------------------- #
def _synthetic_two_chain_dem():
    """Four grains, two collinear strong contacts (50 N, 45 N) plus weak ones."""
    dem = RidgeKeelDEM.__new__(RidgeKeelDEM)
    dem.n_particles = 5
    dem.radius = np.full(5, 0.5)
    dem.pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    dem._contact_cache = {
        "active": np.array([True, True, True, True, True]),
        "i": np.array([0, 1, 0, 1, 0]),
        "j": np.array([1, 2, 3, 4, 4]),
        "normal": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.7, 0.7]]),
        "f_n": np.array([50.0, 45.0, 2.0, 2.0, 1.0]),
        "f_t": np.zeros((5, 2)),
        "overlap": np.full(5, 1e-4),
        "r_eff": np.full(5, 0.25),
        "k_3_2": np.full(5, 1.0),
        "k_t": np.full(5, 1.0),
    }
    return dem


def test_force_chain_report_finds_the_horizontal_chain():
    dem = _synthetic_two_chain_dem()
    report = dem.force_chain_report(threshold=2.0)
    assert report["n_contacts"] == 5
    assert report["mean_normal_force"] == pytest.approx(100.0 / 5.0)
    # only the two horizontal contacts (50 N and 45 N) exceed 2 x mean
    assert report["strong_fraction"] == pytest.approx(2.0 / 5.0)
    assert report["load_fraction"] == pytest.approx(95.0 / 100.0)
    assert report["chain_direction_deg"] == pytest.approx(0.0, abs=1.0)
    assert report["n_chains"] == 1
    assert report["chain_sizes_top"][0] == 3


def test_force_chain_report_is_empty_without_contacts():
    dem = _synthetic_two_chain_dem()
    dem._contact_cache = {"active": np.zeros(0, dtype=bool)}
    assert dem.force_chain_report() == {}


def test_chain_lifetime_counts_consecutive_windows():
    windows = [{1, 2, 3}, {1, 2}, {3}, {1, 3}]
    report = chain_lifetime(windows)
    assert report["n_contacts"] == 3
    # contact 1: windows 0-1 (run 2) then window 3 (run 1)
    # contact 2: windows 0-1 (run 2)
    # contact 3: window 0 (run 1) then windows 2-3 (run 2)
    assert report["max_lifetime_windows"] == 2
    assert report["n_runs"] == 5
    assert report["mean_lifetime_windows"] == pytest.approx(8.0 / 5.0)
    assert report["persistent_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Lindqvist-family anchor
# --------------------------------------------------------------------------- #
def test_lindqvist_anchor_scales_with_the_face_length():
    keel = KeelSection(keel_depth=8.0, keel_width=32.0, porosity=0.30)
    thirty = lindqvist_keel_resistance(keel, 30.0, hull_friction=0.6)
    forty_five = lindqvist_keel_resistance(keel, 45.0, hull_friction=0.6)
    # face length h_k / sin(alpha)
    assert thirty["face_length_m"] == pytest.approx(8.0 / math.sin(math.radians(30.0)))
    assert forty_five["face_length_m"] == pytest.approx(8.0 / math.sin(math.radians(45.0)))
    # weight per area uses the solid fraction
    assert thirty["weight_per_area_Pa"] == pytest.approx(917.0 * 9.81 * 0.7)
    assert thirty["resistance_N_per_m"] == pytest.approx(
        0.6 * 917.0 * 9.81 * 0.7 * 16.0, rel=1e-12)
    assert thirty["coefficient_grade"] == "WITHHELD"
    assert thirty["structure_grade"] == "REAL"


def test_lindqvist_band_brackets_the_central_value():
    keel = KeelSection(keel_depth=6.0, keel_width=20.0)
    anchor = lindqvist_keel_resistance(keel, 30.0, hull_friction=0.6,
                                       friction_band=(0.4, 1.0))
    low, high = anchor["resistance_band_N_per_m"]
    assert low < anchor["resistance_N_per_m"] < high
    assert high / low == pytest.approx(1.0 / 0.4, rel=1e-12)


def test_lindqvist_anchor_validates_inputs():
    keel = KeelSection(keel_depth=6.0, keel_width=20.0)
    with pytest.raises(ValueError):
        lindqvist_keel_resistance(keel, 0.0)
    with pytest.raises(ValueError):
        lindqvist_keel_resistance(keel, 30.0, hull_friction=0.0)


# --------------------------------------------------------------------------- #
# integration smoke: assembly, integrator, C1 (grain conservation)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_keel():
    keel = KeelSection(keel_depth=4.0, keel_width=14.0)
    psd = PowerLawPSD(d_min=0.25, d_max=1.6, exponent=2.0)
    material = RubbleMaterial(young_modulus=2.0e8)
    dem = RidgeKeelDEM(keel, psd, material, solid_fraction=0.60, seed=3,
                       max_particles=200)
    return dem


def test_assembly_fills_the_envelope(small_keel):
    dem = small_keel
    assert dem.n_particles > 40
    assert dem.section.contains(dem.pos[:, 0] + 0.0, dem.pos[:, 1]).mean() > 0.95
    assert 0.35 < dem.measure_solid_fraction() < 0.85
    assert dem._wall_overlap(dem.pos, dem.radius) < 0.05


def test_grain_number_is_conserved_while_integrating(small_keel):
    """C1: no grain is created or destroyed by the integrator."""
    dem = small_keel
    dem.install_bonds()
    dt = stable_timestep(dem.material, float(np.min(dem.radius)))
    half = float(np.atleast_1d(dem.section.half_width(2.0))[0])
    dem.wedge = WedgeIndenter(30.0, 1.0, apex_x=-half - 0.2, apex_y=-2.0,
                              friction=0.3)
    count = dem.n_particles
    for _ in range(120):
        dem.step(dt)
        assert dem.n_particles == count
        assert np.isfinite(dem.pos).all()
        assert np.isfinite(dem.vel).all()
    assert dem.n_particles == count


def test_energy_ledger_terms_are_finite_after_a_short_run(small_keel):
    dem = small_keel
    ledger = dem.ledger.as_dict()
    for key in ("wedge_work", "normal_dissipation", "slip_dissipation",
                "rolling_dissipation", "drag_dissipation", "external_work"):
        assert np.isfinite(ledger[key])
    assert ledger["normal_dissipation"] >= 0.0
    assert ledger["slip_dissipation"] >= 0.0


def test_porosity_report_is_bounded(small_keel):
    dem = small_keel
    snapshot = dem.snapshot()
    report = dem.porosity_report(snapshot)
    assert report["n_grains"] > 20
    assert 0.0 <= report["solid_fraction_initial"] <= 1.5
    assert abs(report["delta_porosity"]) < 0.35


def test_biaxial_driver_fails_loudly_instead_of_returning_a_silent_zero():
    """Known-open item: the biaxial calibration cell is not yet stable.

    The driver must raise instead of returning ``phi = 0`` from a diverged
    state; the friction angle used for C4 therefore comes from the granular
    friction coefficient with its own uncertainty, and this is flagged in the
    report (honest FAIL, do not paper over it).
    """
    with pytest.raises(RuntimeError):
        biaxial_friction_angle(
            RubbleMaterial(young_modulus=2.0e8),
            PowerLawPSD(d_min=0.25, d_max=1.2, exponent=2.0),
            width=3.0, height=3.0, solid_fraction=0.60, seed=5,
            confining_pressure=2.0e4, axial_strain=0.03, max_particles=90)
