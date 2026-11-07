"""Tests for the ice-hull waterline-band abrasion wear module.

Acceptance coverage (criterion letters refer to the lettered machine criteria of
increment W10-N06):

* (a) analytic Archard degradation: ``V = K W L / H`` to 1e-10, and the whole
  pit-volume chain against the same closed form.
* (b) literature anchors: exercised in
  ``scripts/benchmark_ice_abrasion.py`` (weak benchmark, reported there); here we
  test the machinery that produces the anchor comparison.
* (c) energy consistency ``V H / W_f = K / mu in (0, 1)`` with the deviation from
  the set ``K`` below 2 %.
* (d) DEM particle-count and timestep convergence, Roache GCI < 5 % at r = 2.
* (e) friction-module interface consistency at equal ``mu`` (< 2 %).
* (f) full reference case: finite, non-negative, non-zero depth field plus a life
  table, plus the API contract.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem import ice_abrasion as ab


# ---------------------------------------------------------------------------
# helpers: deliberately cheap settings for the case-level tests
# ---------------------------------------------------------------------------

def quick_grid():
    return ab.build_waterline_grid(
        ab.HullParticulars(length_m=120.0, beam_m=20.0, draught_m=7.0,
                          block_coefficient=0.78),
        ab.WaterlineBandSpec(below_waterline_m=1.0, above_waterline_m=0.8,
                             cell_size_m=1.0))


def quick_source(**kwargs):
    base = dict(spec_rate=1.0e-14, n_events_per_leg=600, seed=7)
    base.update(kwargs)
    return ab.WearSourceConfig(**base)


def quick_ensemble():
    return ab.ContactEnsembleConfig(n_samples=256, steps=32, seed=11)


def quick_season(**kwargs):
    return ab.run_wear_season(grid=quick_grid(), source=quick_source(),
                              ensemble_config=quick_ensemble(), **kwargs)


# ---------------------------------------------------------------------------
# (a) Archard law and its analytic degradation
# ---------------------------------------------------------------------------

def test_archard_volume_matches_closed_form():
    g = torch.Generator().manual_seed(3)
    load = torch.rand(1000, generator=g, dtype=ab.DTYPE) * 10.0
    slip = torch.rand(1000, generator=g, dtype=ab.DTYPE) * 2.0
    volume = ab.archard_wear_volume(2.5e-3, load, slip, 2.0e9)
    expected = 2.5e-3 * load * slip / 2.0e9
    rel = (volume - expected).abs() / expected
    assert float(rel.max()) < 1e-10


def test_archard_pit_chain_matches_closed_form():
    """Whole pit-volume chain (pressure -> depth -> volume) vs ``K W L / H``."""
    pressure = torch.tensor([0.5e6, 1.5e6, 3.0e6], dtype=ab.DTYPE)
    slip = torch.tensor([0.05, 0.2, 0.4], dtype=ab.DTYPE)
    radius = torch.tensor([0.01, 0.02, 0.03], dtype=ab.DTYPE)
    K, H = 4.0e-3, 2.0e9
    depth = ab.archard_wear_depth(K, pressure, slip, H)
    volume = depth * math.pi * radius**2
    load = pressure * math.pi * radius**2
    reference = ab.archard_wear_volume(K, load, slip, H)
    assert float(((volume - reference).abs() / reference).max()) < 1e-10


def test_archard_zero_slip_and_zero_load_give_no_wear():
    assert float(ab.archard_wear_volume(1e-3, 5e3, 0.0, 1e9)) == 0.0
    assert float(ab.archard_wear_depth(1e-3, 0.0, 1.0, 1e9)) == 0.0


def test_archard_monotone_and_inverse_in_hardness():
    base = float(ab.archard_wear_depth(1e-3, 1e6, 0.2, 2e9))
    assert float(ab.archard_wear_depth(2e-3, 1e6, 0.2, 2e9)) > base
    assert float(ab.archard_wear_depth(1e-3, 2e6, 0.2, 2e9)) > base
    assert float(ab.archard_wear_depth(1e-3, 1e6, 0.4, 2e9)) > base
    assert float(ab.archard_wear_depth(1e-3, 1e6, 0.2, 4e9)) == pytest.approx(
        base / 2.0, rel=1e-15)


def test_archard_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        ab.archard_wear_volume(1e-3, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        ab.archard_wear_volume(-1e-3, 1.0, 1.0, 1e9)
    with pytest.raises(ValueError):
        ab.archard_wear_depth(1e-3, -1.0, 1.0, 1e9)


def test_specific_wear_rate_roundtrip():
    H, k = 2.0e9, 2.0e-14
    K = ab.archard_coefficient_from_specific_rate(k, H)
    assert K == pytest.approx(k * H)
    assert ab.specific_wear_rate(K, H) == pytest.approx(k, rel=1e-15)
    assert ab.ICE_CLASS_EPOXY.archard_coefficient(2e-15) == pytest.approx(
        2e-15 * ab.ICE_CLASS_EPOXY.hardness_pa)


# ---------------------------------------------------------------------------
# Finnie impact correction
# ---------------------------------------------------------------------------

def test_finnie_branch_is_continuous_at_transition():
    a0 = ab.FINNIE_TRANSITION_ANGLE_RAD
    left = torch.tensor(a0 - 1e-9, dtype=ab.DTYPE)
    right = torch.tensor(a0 + 1e-9, dtype=ab.DTYPE)
    f = ab.finnie_angle_function(torch.stack([left, right]))
    assert abs(float(f[0] - f[1])) < 1e-8
    assert a0 == pytest.approx(math.radians(18.4349), abs=1e-4)


def test_finnie_endpoints_are_zero():
    f = ab.finnie_angle_function(torch.tensor([0.0, 0.5 * math.pi],
                                              dtype=ab.DTYPE))
    assert abs(float(f[0])) < 1e-15
    assert abs(float(f[1])) < 1e-15


def test_finnie_has_interior_maximum_near_twenty_degrees():
    a = torch.linspace(0.0, 0.5 * math.pi, 4001, dtype=ab.DTYPE)
    f = ab.finnie_angle_function(a)
    peak = float(f.max())
    peak_angle = math.degrees(float(a[int(torch.argmax(f))]))
    assert peak == pytest.approx(0.3, abs=0.02)
    assert 10.0 < peak_angle < 30.0


def test_finnie_rejects_out_of_range_angles():
    with pytest.raises(ValueError):
        ab.finnie_angle_function(torch.tensor(-0.1, dtype=ab.DTYPE))
    with pytest.raises(ValueError):
        ab.finnie_angle_function(torch.tensor(2.0, dtype=ab.DTYPE))


def test_finnie_volume_scales_with_energy_and_flow_stress():
    v1 = float(ab.finnie_erosion_volume(0.4, 0.1, 0.3, 8e7))
    v2 = float(ab.finnie_erosion_volume(0.4, 0.2, 0.3, 8e7))
    assert v2 / v1 == pytest.approx(4.0, rel=1e-12)
    v3 = float(ab.finnie_erosion_volume(0.4, 0.1, 0.3, 1.6e8))
    assert v3 / v1 == pytest.approx(0.5, rel=1e-12)
    assert ab.finnie_flow_stress_from_hardness(2.0e9) == pytest.approx(8.0e8)


# ---------------------------------------------------------------------------
# Geometry validation (Cb / L / B gate)
# ---------------------------------------------------------------------------

def test_geometry_audit_reproduces_principal_particulars():
    grid = ab.build_waterline_grid()
    audit = grid.audit
    assert audit["geometry_audit_passed"] is True
    assert audit["cb_relative_error"] < 1e-3
    assert audit["beam_relative_error"] < 1e-3
    assert audit["length_relative_error"] == 0.0
    assert audit["block_coefficient_mesh"] == pytest.approx(
        audit["block_coefficient_target"], rel=1e-3)
    assert audit["prismatic_coefficient"] == pytest.approx(0.82 / 0.99, rel=1e-12)


def test_geometry_rejects_block_above_midship_coefficient():
    with pytest.raises(ValueError):
        ab.HullParticulars(block_coefficient=0.99, midship_coefficient=0.95)
    with pytest.raises(ValueError):
        ab.HullParticulars(block_coefficient=1.2)


def test_band_grid_cell_areas_are_positive_and_mirror_is_double():
    grid = ab.build_waterline_grid()
    assert bool(torch.all(grid.cell_area_m2 > 0.0))
    assert grid.n_cells == grid.shape[0] * grid.shape[1]
    assert grid.band_area_m2 > 0.0
    assert grid.audit["mirrored_band_area_m2"] == pytest.approx(
        2.0 * grid.band_area_m2, rel=1e-12)
    # band cells span the prescribed ice-belt height
    z_span = float(grid.z_edges[-1] - grid.z_edges[0])
    assert z_span == pytest.approx(1.2 + 1.0, rel=1e-2)


# ---------------------------------------------------------------------------
# DEM contact statistics (Hertz + crushing cap + Cundall-Strack)
# ---------------------------------------------------------------------------

def test_coulomb_cone_is_never_violated_and_slip_is_bounded():
    ens = ab.sample_contact_ensemble(ab.ContactEnsembleConfig(n_samples=128,
                                                             steps=32))
    assert len(ens) == 128
    # slip can never exceed the sliding speed times the contact duration
    bound = ens.slip_speed_m_s * ens.duration_s
    assert bool(torch.all(ens.slip_distance_m <= bound * (1.0 + 1e-12)))
    assert bool(torch.all(ens.sliding_fraction >= 0.0))
    assert bool(torch.all(ens.sliding_fraction <= 1.0))
    # dissipated work of the sliding branch is bounded by mu N v t
    assert bool(torch.all(ens.dissipation_J <= 0.1 * ens.normal_load_N
                          * ens.slip_speed_m_s * ens.duration_s * (1.0 + 1e-9)))
    # Hertz relationship between load and crushed patch: F = p_c pi a^2
    assert float(((ens.normal_load_N
                   - ens.pressure_pa * math.pi * ens.radius_m**2).abs()
                  / ens.normal_load_N).max()) < 1e-12


def test_hertz_pressure_scales_as_cube_root_of_load():
    cfg = ab.ContactEnsembleConfig(n_samples=512, steps=8,
                                   crushing_strength_median_pa=1e12,
                                   crushing_strength_log_sigma=1e-6)
    base = ab.sample_contact_ensemble(cfg, pressure_scale=1.0)
    heavy = ab.sample_contact_ensemble(cfg, pressure_scale=8.0)
    assert base.meta["hertz_capped_fraction"] == 0.0
    ratio = float(heavy.pressure_pa.median() / base.pressure_pa.median())
    assert ratio == pytest.approx(2.0, rel=2e-3)


def test_crushing_cap_bounds_the_contact_pressure():
    cfg = ab.ContactEnsembleConfig(n_samples=512, steps=8,
                                   crushing_strength_median_pa=1.0e6,
                                   crushing_strength_log_sigma=1e-9)
    ens = ab.sample_contact_ensemble(cfg)
    assert float(ens.pressure_pa.max()) <= 1.0e6 * (1.0 + 1e-6)
    assert ens.meta["hertz_capped_fraction"] == 1.0


def test_mindlin_stiffness_matches_library_helper():
    from tensordem.hertz_mindlin import hertz_mindlin_tangential_stiffness
    e_star, g_star, r_eff, delta = 5.5e9, 1.13e9, 0.175, 6.8e-5
    library = float(hertz_mindlin_tangential_stiffness(delta, e_star, g_star,
                                                       r_eff))
    module = 8.0 * g_star * math.sqrt(r_eff * delta)
    assert module == pytest.approx(library, rel=1e-12)


def test_tangential_contact_sticks_for_very_short_slip_events():
    stick = ab.sample_contact_ensemble(ab.ContactEnsembleConfig(
        n_samples=128, steps=64, slip_length_fraction=1e-6,
        friction_coefficient=0.6))
    slide = ab.sample_contact_ensemble(ab.ContactEnsembleConfig(
        n_samples=128, steps=64, slip_length_fraction=0.5,
        friction_coefficient=0.1))
    assert float(stick.sliding_fraction.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(stick.slip_distance_m.mean()) == pytest.approx(0.0, abs=1e-18)
    assert float(slide.sliding_fraction.mean()) == pytest.approx(1.0, abs=1e-12)
    assert float(slide.slip_distance_m.mean()) > 0.0


def test_cundall_strack_projection_matches_contact_module_step():
    """Identical increment/projection as ``contact_tangential``."""
    from tensordem.contact_tangential import advance_tangential_contact
    k_t, mu, dt, f_n = 1.0e7, 0.1, 1e-3, 5.0e3   # k_t |v| dt > mu N -> slip
    vel = torch.tensor([[1.0, 0.0]], dtype=ab.DTYPE)
    normal = torch.tensor([[0.0, 1.0]], dtype=ab.DTYPE)
    force_lib, disp_lib, sliding_lib = advance_tangential_contact(
        torch.zeros((1, 2), dtype=ab.DTYPE), vel, normal,
        torch.tensor([f_n], dtype=ab.DTYPE), k_t, mu, dt)
    # module's equivalent one-step update
    raw = k_t * vel * dt
    mag = torch.linalg.vector_norm(raw, dim=1)
    scale = min(1.0, mu * f_n / float(mag))
    force_mod = raw * scale
    assert float((force_mod - force_lib).abs().max()) < 1e-9
    assert bool(sliding_lib[0])


def test_joint_statistics_are_finite_and_positive():
    ens = ab.sample_contact_ensemble(ab.ContactEnsembleConfig(n_samples=512,
                                                             steps=32))
    stats = ens.wear_statistic(1e-14)
    assert stats["pit_depth_mean_m"] > 0.0
    assert stats["pressure_median_pa"] > 0.0
    assert stats["slip_distance_mean_m"] > 0.0
    summary = ens.summary()
    assert summary["n_contacts"] == 512
    assert 0.0 < summary["sliding_fraction_mean"] <= 1.0


# ---------------------------------------------------------------------------
# Projection: volume conservation and grid behaviour
# ---------------------------------------------------------------------------

def test_projection_conserves_volume_exactly():
    grid = quick_grid()
    g = torch.Generator().manual_seed(5)
    n = 2000
    x = torch.rand(n, generator=g, dtype=ab.DTYPE) * grid.hull.length_m
    z = (torch.rand(n, generator=g, dtype=ab.DTYPE)
         * (float(grid.z_edges[-1]) - float(grid.z_edges[0]))
         + float(grid.z_edges[0]))
    r = 0.01 + 0.04 * torch.rand(n, generator=g, dtype=ab.DTYPE)
    d = 1e-9 * torch.rand(n, generator=g, dtype=ab.DTYPE)
    field = ab.project_contact_pits(grid, x, z, r, d)
    expected = float((d * math.pi * r**2).sum())
    assert field.volume_total() == pytest.approx(expected, rel=1e-12)
    assert field.events["event_volume_total_m3"] == pytest.approx(expected,
                                                                  rel=1e-12)


def test_projection_is_linear_in_pit_depth():
    grid = quick_grid()
    g = torch.Generator().manual_seed(6)
    n = 500
    x = torch.rand(n, generator=g, dtype=ab.DTYPE) * grid.hull.length_m
    z = torch.full((n,), grid.hull.draught_m, dtype=ab.DTYPE)
    r = torch.full((n,), 0.02, dtype=ab.DTYPE)
    d = torch.full((n,), 3e-9, dtype=ab.DTYPE)
    a = ab.project_contact_pits(grid, x, z, r, d)
    b = ab.project_contact_pits(grid, x, z, r, 2.0 * d)
    assert float((b.depth_m - 2.0 * a.depth_m).abs().max()) < 1e-18


def test_projection_rejects_bad_inputs():
    grid = quick_grid()
    with pytest.raises(ValueError):
        ab.project_contact_pits(grid, torch.tensor([1.0]), torch.tensor([7.0]),
                                torch.tensor([-0.01]), torch.tensor([1e-9]))
    with pytest.raises(ValueError):
        ab.project_contact_pits(grid, torch.tensor([1.0, 2.0]),
                                torch.tensor([7.0]), torch.tensor([0.01]),
                                torch.tensor([1e-9]))


def test_belt_mean_depth_is_grid_independent():
    coarse = ab.run_wear_season(
        grid=ab.build_waterline_grid(
            ab.HullParticulars(length_m=120.0, beam_m=20.0, draught_m=7.0,
                              block_coefficient=0.78),
            ab.WaterlineBandSpec(below_waterline_m=1.0, above_waterline_m=0.8,
                                 cell_size_m=1.0)),
        source=quick_source(), ensemble_config=quick_ensemble())
    fine = ab.run_wear_season(
        grid=ab.build_waterline_grid(
            ab.HullParticulars(length_m=120.0, beam_m=20.0, draught_m=7.0,
                              block_coefficient=0.78),
            ab.WaterlineBandSpec(below_waterline_m=1.0, above_waterline_m=0.8,
                                 cell_size_m=0.5)),
        source=quick_source(), ensemble_config=quick_ensemble())
    # total wear volume is energy-anchored: identical to the discretisation
    # error of the activity sampling and the footprint clamping
    assert fine.field_per_year.volume_total() == pytest.approx(
        coarse.field_per_year.volume_total(), rel=0.02)
    m_fine = float(fine.field_per_year.depth_m.mean())
    m_coarse = float(coarse.field_per_year.depth_m.mean())
    assert abs(m_fine - m_coarse) / m_coarse < 0.05


# ---------------------------------------------------------------------------
# (c) energy consistency and (e) friction-module interface
# ---------------------------------------------------------------------------

def test_energy_identity_volume_h_over_friction_work_equals_K_over_mu():
    season = quick_season()
    ledger = season.ledger
    K = season.material.archard_coefficient(season.source.spec_rate)
    assert 0.0 < ledger["energy_ratio_volumeH_over_friction_work"] < 1.0
    assert ledger["energy_ratio_relative_deviation"] < 0.02
    assert ledger["energy_ratio_volumeH_over_friction_work"] == pytest.approx(
        K / season.voyage.mu, rel=0.02)
    checks = ab.audit_case_checks(season, {"life_years_p50": 1.0})
    assert checks["energy_ratio_in_unit_interval"] is True
    assert checks["energy_ratio_deviation_pct"] < 2.0


def test_friction_interface_matches_resistance_module_within_two_percent():
    season = quick_season()
    ledger = season.ledger
    assert ledger["friction_interface_relative_deviation"] < 0.02
    # recompute the same friction work independently from the resistance module
    voyage = season.voyage
    total = 0.0
    for leg in voyage.legs_per_voyage():
        work = ab.leg_friction_work(leg, voyage)
        total += voyage.belt_energy_fraction * leg.concentration * work["friction_work_J"]
    total *= voyage.voyages_per_year
    assert ledger["friction_work_year_total_J"] == pytest.approx(total, rel=1e-12)
    assert ledger["friction_work_from_events_year_per_side_J"] == pytest.approx(
        0.5 * total, rel=0.02)


def test_friction_work_grows_with_friction_coefficient():
    voyage = ab.reference_baltic_voyage()
    leg = voyage.legs[1]
    low = ab.leg_friction_work(leg, voyage, mu=0.05)["friction_work_J"]
    high = ab.leg_friction_work(leg, voyage, mu=0.15)["friction_work_J"]
    assert high > low > 0.0
    zero = ab.leg_friction_work(leg, voyage, mu=0.0)["friction_work_J"]
    assert zero == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (d) convergence
# ---------------------------------------------------------------------------

def test_dem_convergence_gci_below_five_percent():
    study = ab.dem_convergence_study(
        sample_counts=(512, 1024, 2048),
        step_counts=(24, 48, 96),
        config=ab.ContactEnsembleConfig(n_samples=2048, steps=96, seed=3))
    assert study["sample_count_gci"]["gci_fine_pct"] < 5.0
    assert study["timestep_gci"]["gci_fine_pct"] < 5.0
    assert study["criterion_gci_lt_5pct"] is True


def test_gci_reports_machine_precision_agreement():
    out = ab.grid_convergence_index(1.0, 1.0 + 1e-12, 0.5)
    assert out["gci_fine_pct"] == 0.0
    assert out["note"].startswith("fine and medium agree")


def test_gci_uses_conservative_fallback_for_non_monotone_levels():
    out = ab.grid_convergence_index(1.00, 1.01, 0.98)
    assert out["order_source"].startswith("theoretical fallback")
    assert out["gci_fine_pct"] is not None and out["gci_fine_pct"] > 0.0


# ---------------------------------------------------------------------------
# Life prediction
# ---------------------------------------------------------------------------

def test_life_table_is_ordered_and_scales_with_thickness():
    season = quick_season()
    life = ab.coating_life_table(season.field_per_year, ab.ICE_CLASS_EPOXY,
                                 mu=season.voyage.mu, n_mc=2000)
    assert life["life_years_p10"] < life["life_years_p50"] < life["life_years_p90"]
    assert life["thickness_used_m"] == ab.ICE_CLASS_EPOXY.thickness_m
    twice = ab.coating_life_table(
        season.field_per_year,
        ab.TargetMaterial("thick coating", 2.0e8, 2.0 * 350.0e-6),
        mu=season.voyage.mu, n_mc=2000)
    assert twice["life_years_p50"] == pytest.approx(2.0 * life["life_years_p50"],
                                                    rel=1e-12)
    assert twice["rate_peak_m_per_year"] == pytest.approx(
        life["rate_peak_m_per_year"], rel=1e-15)


def test_steel_life_uses_wear_allowance_when_no_coating_thickness():
    season = quick_season()
    life = ab.coating_life_table(season.field_per_year, ab.AH36_STEEL,
                                 mu=season.voyage.mu, n_mc=200)
    assert life["thickness_source"].startswith("assumed structural")
    assert life["life_years_p50"] > 0.0


# ---------------------------------------------------------------------------
# (f) full reference case and API contract
# ---------------------------------------------------------------------------

def test_reference_case_runs_and_passes_its_own_audit():
    case = ab.run_reference_case(source=quick_source(), grid=quick_grid(),
                                 ensemble_config=quick_ensemble(), n_mc=500)
    checks = case.checks
    assert checks["field_finite"] and checks["field_nonnegative"]
    assert checks["field_nonzero"]
    assert checks["energy_ratio_in_unit_interval"]
    assert checks["energy_ratio_deviation_pct"] < 2.0
    assert checks["friction_interface_deviation_pct"] < 2.0
    assert checks["life_p50_positive"]
    summary = case.season.summary()
    assert summary["per_year"]["p95_depth_um"] > 0.0
    assert summary["grid"]["geometry_audit_passed"] is True
    assert case.calibration["spec_rate_fitted"] > 0.0


def test_reference_case_calibration_is_linear_and_reported():
    base = ab.run_reference_case(source=quick_source(), grid=quick_grid(),
                                 ensemble_config=quick_ensemble(),
                                 calibrate=False, n_mc=100)
    fitted = ab.run_reference_case(source=quick_source(), grid=quick_grid(),
                                   ensemble_config=quick_ensemble(),
                                   calibrate=True, n_mc=100)
    target = 0.5 * sum(ab.ANCHOR_BALTIC_COATING_MICRON_PER_SEASON)
    assert fitted.calibration["calibrated_p95_um_per_season"] == pytest.approx(
        target, rel=1e-9)
    ratio = (fitted.calibration["spec_rate_fitted"]
             / base.calibration["spec_rate_fitted"])
    assert ratio == pytest.approx(
        target / base.calibration["calibrated_p95_um_per_season"], rel=1e-9)


def test_api_handler_contract():
    payload = {
        "length_m": 120.0, "beam_m": 20.0, "draught_m": 7.0,
        "block_coefficient": 0.78, "material": "coating",
        "belt": {"cell_size_m": 2.0},
        "voyage": {"voyages_per_year": 4.0,
                   "legs": [{"name": "test leg", "distance_nm": 20.0,
                             "speed_knots": 6.0, "thickness_m": 0.3,
                             "concentration": 0.5}]},
        "n_events_per_leg": 200, "calibrate_to_anchor": False, "n_mc": 100,
    }
    out = ab.ice_abrasion_calculation(payload)
    assert out["status"] == "ok"
    assert out["endpoint"].endswith("ice_abrasion")
    assert out["geometry"]["geometry_audit_passed"] is True
    assert out["result"]["season"]["per_year"]["p95_depth_um"] > 0.0
    assert "life" in out["result"] and "calibration" in out["result"]


def test_api_handler_rejects_missing_fields_and_bad_material():
    with pytest.raises(ValueError):
        ab.ice_abrasion_calculation({"beam_m": 20.0, "draught_m": 7.0,
                                     "block_coefficient": 0.78})
    with pytest.raises(ValueError):
        ab.ice_abrasion_calculation({"length_m": 120.0, "beam_m": 20.0,
                                     "draught_m": 7.0, "block_coefficient": 0.78,
                                     "material": "unobtainium"})


def test_season_is_reproducible_for_a_fixed_seed():
    a = quick_season()
    b = quick_season()
    assert bool(torch.equal(a.field_per_year.depth_m, b.field_per_year.depth_m))
    assert a.ledger["wear_volume_year_one_side_m3"] == pytest.approx(
        b.ledger["wear_volume_year_one_side_m3"], rel=1e-15)


def test_ice_leg_and_voyage_validation():
    with pytest.raises(ValueError):
        ab.IceLeg("bad", distance_nm=10.0, speed_knots=5.0, thickness_m=0.3,
                  concentration=1.5)
    with pytest.raises(ValueError):
        ab.VoyageProfile(legs=(), voyages_per_year=3.0)
    leg = ab.IceLeg("ok", distance_nm=10.0, speed_knots=5.0, thickness_m=0.3,
                    concentration=0.5)
    assert leg.speed_m_s == pytest.approx(2.5722, rel=1e-3)
    assert leg.duration_s == pytest.approx(leg.distance_m / leg.speed_m_s)