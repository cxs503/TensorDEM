"""Acceptance tests for the DEM-side erosion verification face (p176w2-n11).

Every test name carries the criterion tag it protects:

* ``c1`` -- angle functions: Finnie peak window, analytic vs numeric, Oka peak
  window for the case material, DEM stream peak window, self-similar shape
  mutual adjudication and its band sensitivity;
* ``c2`` -- velocity exponent: window, literature agreement, anchor linearity;
* ``c3`` -- DNV-RP-O501 per-point comparison with one fitted anchor;
* ``c4`` -- mass / volume / energy ledger closure, crater geometry round-trips;
* ``c5`` -- double monotonicity plus a negative control of the checker itself;
* ``c6`` -- negative controls (zero impacts, zero speed);
* ``c7`` -- breakage decoupling: bitwise identity, static import audit, mass
  bookkeeping, and a negative control for the audit.

Stream runs use small particle counts so the file stays fast (the full
freeze-out is ``benchmarks/erosion_dem_verification.py``).
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from tensordem.erosion import (
    LITERATURE_VELOCITY_EXPONENTS,
    finnie_angle_function,
    finnie_peak_angle_deg,
    finnie_peak_angle_deg_numeric,
    oka_angle_function,
    oka_peak_angle_deg,
)
from tensordem.erosion.models import dnv_ductile_erosion_ratio
from tensordem.erosion_dem import (
    CASE_LAW,
    CASE_MATERIAL,
    JointImpactKinematics,
    OkaAnchoredLaw,
    StreamConfig,
    WallSurface,
    aggregate_exponent,
    angle_sweep,
    erosion_core_imports_comminution,
    fragmentation_hook,
    indentation_depth_m,
    monotonicity,
    oblique_restitution,
    peak_angle_from_samples,
    plastic_normal_restitution,
    run_impact_stream,
    single_impact_crater,
    sphere_cap_depth_from_volume,
    sphere_cap_radius,
    sphere_cap_volume,
    velocity_sweep,
    yield_velocity_m_s,
)

FAST = dict(n_particles=160)


# ---------------------------------------------------------------- C1: angle
def test_c1_finnie_peak_inside_ductile_window():
    peak = finnie_peak_angle_deg()
    assert 15.0 <= peak <= 25.0
    assert abs(finnie_peak_angle_deg_numeric() - peak) < 1e-3


def test_c1_finnie_peak_closed_form_root():
    """Analytic peak ``0.5 atan(2/3)`` vs the derivative's numerical zero."""
    peak = finnie_peak_angle_deg()
    a = np.linspace(1.0, 60.0, 200001)
    f = np.asarray(finnie_angle_function(a), float)
    assert abs(a[int(np.argmax(f))] - peak) < 2e-3
    assert abs(peak - 0.5 * math.degrees(math.atan(2.0 / 3.0))) < 1e-9


def test_c1_oka_peak_inside_window_for_case_material():
    peak = oka_peak_angle_deg(CASE_MATERIAL.hardness_hv_gpa)
    assert 15.0 <= peak <= 25.0


def test_c1_oka_peak_moves_towards_normal_incidence_with_hardness():
    """Oka's peak slides to 90 deg as the target hardens (brittle limit)."""
    hvs = (0.5, 0.87, 1.07, 1.8, 2.0, 5.5, 15.0)
    peaks = [oka_peak_angle_deg(hv) for hv in hvs]
    assert all(p2 > p1 for p1, p2 in zip(peaks, peaks[1:]))
    assert peaks[0] < 25.0 < peaks[-1] < 90.0


def test_c1_stream_peak_inside_window():
    sweep = angle_sweep(
        [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 90.0],
        cfg=StreamConfig(**FAST),
        speed_mean_m_s=30.0,
    )
    peak = peak_angle_from_samples(
        sweep["launch_angles_deg"], sweep["erosion_ratios_kg_kg"]
    )
    assert 15.0 <= peak["peak_angle_deg"] <= 25.0


def test_c1_stream_peak_is_interior():
    """The aggregate curve must not be monotone: an interior maximum exists."""
    sweep = angle_sweep(
        [5.0, 20.0, 45.0, 90.0], cfg=StreamConfig(**FAST), speed_mean_m_s=30.0
    )
    r = sweep["erosion_ratios_kg_kg"]
    assert r[1] > r[0] and r[1] > r[-1]


def test_c1_self_similar_shape_mutual_agreement():
    """Finnie vs Oka on the peak-centred +-40 % band agree within 15 %.

    Both curves are normalised to their own maximum and evaluated at
    ``alpha / alpha_peak`` in ``[0.6, 1.4]``: the neighbourhood of the peak,
    where both published models describe cutting-dominated erosion.
    """
    a = np.linspace(0.0, 90.0, 9001)
    f = np.asarray(finnie_angle_function(a), float)
    g = np.asarray(oka_angle_function(a, CASE_MATERIAL.hardness_hv_gpa), float)
    f, g = f / f.max(), g / g.max()
    t = np.linspace(0.6, 1.4, 401)
    fa = np.interp(t * finnie_peak_angle_deg(), a, f)
    ga = np.interp(t * oka_peak_angle_deg(CASE_MATERIAL.hardness_hv_gpa), a, g)
    dev = np.abs(fa - ga) / np.maximum(np.maximum(fa, ga), 1e-12)
    assert dev.max() < 0.15


def test_c1_shape_divergence_grows_with_band():
    """Sensitivity: the two model classes diverge once the band leaves the peak."""
    a = np.linspace(0.0, 90.0, 9001)
    f = np.asarray(finnie_angle_function(a), float)
    g = np.asarray(oka_angle_function(a, CASE_MATERIAL.hardness_hv_gpa), float)
    f, g = f / f.max(), g / g.max()
    fp, op = finnie_peak_angle_deg(), oka_peak_angle_deg(CASE_MATERIAL.hardness_hv_gpa)
    devs = []
    for band in (0.2, 0.4, 0.6, 0.8):
        t = np.linspace(1 - band, 1 + band, 401)
        fa = np.interp(t * fp, a, f)
        ga = np.interp(t * op, a, g)
        devs.append(float((np.abs(fa - ga) / np.maximum(np.maximum(fa, ga), 1e-12)).max()))
    assert all(b >= a_ for a_, b in zip(devs, devs[1:]))


def test_c1_model_class_divergence_documented():
    """Finnie vanishes at normal incidence, Oka does not: a model-class fact."""
    assert float(finnie_angle_function(90.0)) == pytest.approx(0.0, abs=1e-12)
    g90 = float(oka_angle_function(90.0, CASE_MATERIAL.hardness_hv_gpa))
    assert g90 > 0.5 * float(oka_angle_function(oka_peak_angle_deg(CASE_MATERIAL.hardness_hv_gpa),
                                                CASE_MATERIAL.hardness_hv_gpa))


# ---------------------------------------------------------------- C2: exponent
def test_c2_stream_velocity_exponent_in_window():
    sweep = velocity_sweep(
        [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
        cfg=StreamConfig(**FAST),
        angle_mean_deg=90.0,
    )
    fit = aggregate_exponent(sweep["launch_speeds_m_s"], sweep["erosion_ratios_kg_kg"])
    assert 2.1 <= fit["exponent"] <= 2.5
    assert fit["log_residual_rms"] < 0.01
    assert len(sweep["launch_speeds_m_s"]) >= 5


def test_c2_exponent_matches_literature_within_10pct():
    sweep = velocity_sweep(
        [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
        cfg=StreamConfig(**FAST),
        angle_mean_deg=90.0,
    )
    fit = aggregate_exponent(sweep["launch_speeds_m_s"], sweep["erosion_ratios_kg_kg"])
    lit = LITERATURE_VELOCITY_EXPONENTS["aluminium_6061"][0]
    assert abs(fit["exponent"] - lit) / lit < 0.10


def test_c2_exponent_matches_law_exponent():
    sweep = velocity_sweep(
        [20.0, 30.0, 40.0], cfg=StreamConfig(**FAST), angle_mean_deg=90.0
    )
    fit = aggregate_exponent(sweep["launch_speeds_m_s"], sweep["erosion_ratios_kg_kg"])
    assert abs(fit["exponent"] - CASE_LAW.velocity_exponent) < 0.02


def test_c2_stream_linear_in_anchor():
    """The stream is exactly linear in the single calibrated anchor."""
    cfg = StreamConfig(**FAST)
    law1 = OkaAnchoredLaw.for_material(CASE_MATERIAL, 1.0e-6)
    law2 = OkaAnchoredLaw.for_material(CASE_MATERIAL, 3.0e-6)
    r1 = run_impact_stream(cfg, law=law1)["erosion_ratio_kg_kg"]
    r2 = run_impact_stream(cfg, law=law2)["erosion_ratio_kg_kg"]
    assert r2 / r1 == pytest.approx(3.0, rel=1e-12)


def test_c2_exponent_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        aggregate_exponent([10.0], [1e-6])
    with pytest.raises(ValueError):
        aggregate_exponent([10.0, 10.0], [1e-6, 2e-6])


# ---------------------------------------------------------------- C3: DNV
def test_c3_dnv_per_point_within_15pct():
    speeds = [20.0, 25.0, 30.0, 35.0, 40.0]
    pair_k, pair_n = 1.3e-9, 2.4
    sweep = velocity_sweep(speeds, cfg=StreamConfig(**FAST), angle_mean_deg=90.0)
    dnv = [dnv_ductile_erosion_ratio(v, 90.0, "carbon_steel", k=pair_k, n=pair_n)
           for v in speeds]
    ln_scale = np.mean(np.log(dnv) - np.log(sweep["erosion_ratios_kg_kg"]))
    scale = math.exp(float(ln_scale))
    for v, d, r in zip(speeds, dnv, sweep["erosion_ratios_kg_kg"]):
        assert abs(scale * r - d) / d < 0.15, f"deviation at {v} m/s"


def test_c3_dnv_monotone_in_speed():
    rates = [dnv_ductile_erosion_ratio(v, 90.0, "carbon_steel", k=1.3e-9, n=2.4)
             for v in (10.0, 20.0, 40.0)]
    assert rates[0] < rates[1] < rates[2]


# ---------------------------------------------------------------- C4: ledger
def test_c4_sphere_cap_volume_matches_numeric_integral():
    r = 7.5e-5
    for h in (0.1 * r, 0.5 * r, 0.9 * r):
        z = np.linspace(0.0, h, 400001)
        area = math.pi * (2.0 * r * z - z * z)
        numeric = float(np.trapezoid(area, z))
        assert sphere_cap_volume(h, r) == pytest.approx(numeric, rel=2e-5)


def test_c4_depth_inversion_roundtrip():
    r = 7.5e-5
    for frac in (1e-3, 1e-2, 0.1, 0.5, 0.99):
        v = frac * sphere_cap_volume(r, r)
        h = sphere_cap_depth_from_volume(v, r)
        assert sphere_cap_volume(h, r) == pytest.approx(v, rel=1e-9)
        assert sphere_cap_radius(h, r) > 0.0


def test_c4_depth_inversion_monotone_and_guarded():
    r = 7.5e-5
    hs = [sphere_cap_depth_from_volume(f * sphere_cap_volume(r, r), r)
          for f in (0.01, 0.1, 0.5)]
    assert hs[0] < hs[1] < hs[2]
    with pytest.raises(ValueError):
        sphere_cap_depth_from_volume(2.0 * sphere_cap_volume(r, r), r)
    with pytest.raises(ValueError):
        sphere_cap_depth_from_volume(-1.0, r)


def test_c4_crater_geometry_consistent():
    c = single_impact_crater(30.0, 20.0, 150e-6, 2650.0, CASE_MATERIAL, law=CASE_LAW)
    assert c.physically_admissible
    assert 0.0 < c.depth_fraction_of_particle_radius < 1.0
    assert 0.0 < c.volume_fraction_of_particle < 1.0
    assert sphere_cap_volume(c.depth_m, c.particle_radius_m) == pytest.approx(
        c.removed_volume_m3, rel=1e-9
    )
    assert math.pi * c.radius_m**2 == pytest.approx(c.footprint_area_m2, rel=1e-12)


def test_c4_stream_ledger_closes():
    out = run_impact_stream(StreamConfig(n_particles=400))
    assert out["closure_residual"] < 0.05
    assert out["delivered_mass_kg"] > 0.0
    assert out["removed_mass_kg"] >= 0.0
    assert out["removed_mass_kg"] == pytest.approx(
        out["removed_volume_m3"] * CASE_MATERIAL.density_kg_m3, rel=1e-12
    )
    assert out["inadmissible_craters"] == 0


def test_c4_wall_surface_deposit_conserves_volume():
    wall = WallSurface(16, 16, 1e-8)
    rng = np.random.default_rng(7)
    for _ in range(200):
        wall.deposit(rng.uniform(-6e-3, 6e-3), rng.uniform(-6e-3, 6e-3),
                     1e-15 * rng.uniform(0.5, 2.0), 1.2e-2)
    assert wall.closure_residual < 1e-12
    assert wall.max_depth_m > 0.0


def test_c4_wall_surface_rejects_negative_volume():
    wall = WallSurface(8, 8, 1e-8)
    with pytest.raises(ValueError):
        wall.deposit(0.0, 0.0, -1e-18, 1e-2)


def test_c4_removal_efficiency_below_one():
    out = run_impact_stream(StreamConfig(n_particles=300))
    efficiency = (out["removed_volume_m3"] * CASE_MATERIAL.flow_stress_pa) / max(
        out["kinetic_energy_in_j"], 1e-300
    )
    assert 0.0 <= efficiency < 1.0


# ---------------------------------------------------------------- C5: monotone
def test_c5_erosion_increases_with_speed():
    sweep = velocity_sweep([15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
                           cfg=StreamConfig(**FAST), angle_mean_deg=90.0)
    chk = monotonicity(sweep["launch_speeds_m_s"], sweep["erosion_ratios_kg_kg"],
                       increasing=True)
    assert chk["monotone"] and chk["samples"] >= 5


def test_c5_erosion_decays_after_angle_peak():
    sweep = angle_sweep([20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0],
                        cfg=StreamConfig(**FAST), speed_mean_m_s=30.0)
    chk = monotonicity(sweep["launch_angles_deg"], sweep["erosion_ratios_kg_kg"],
                       increasing=False)
    assert chk["monotone"] and chk["samples"] >= 5


def test_c5_monotonicity_checker_detects_violation():
    """Negative control of the checker itself."""
    bad = monotonicity([1, 2, 3, 4, 5], [1.0, 2.0, 1.5, 3.0, 4.0], increasing=True)
    assert not bad["monotone"] and bad["violations"] == 1
    bad_dec = monotonicity([1, 2, 3], [3.0, 2.0, 2.5], increasing=False)
    assert not bad_dec["monotone"]


# ---------------------------------------------------------------- C6: negatives
def test_c6_zero_particles_gives_zero_erosion():
    out = run_impact_stream(StreamConfig(n_particles=0))
    assert out["impacts"] == 0
    assert abs(out["erosion_ratio_kg_kg"]) < 1e-12
    assert abs(out["removed_mass_kg"]) < 1e-12


def test_c6_zero_speed_gives_zero_erosion():
    out = run_impact_stream(
        StreamConfig(n_particles=40, speed_mean_m_s=0.0, speed_cv=0.0, gravity_m_s2=0.0)
    )
    assert out["impacts"] == 0
    assert abs(out["erosion_ratio_kg_kg"]) < 1e-12


def test_c6_zero_speed_in_law_and_crater():
    assert abs(float(CASE_LAW.ratio(0.0, 30.0, 150.0))) < 1e-12
    c = single_impact_crater(0.0, 30.0, 150e-6, 2650.0, CASE_MATERIAL, law=CASE_LAW)
    assert c.removed_volume_m3 == 0.0 and c.depth_m == 0.0


def test_c6_zero_normal_speed_indents_nothing():
    assert indentation_depth_m(0.0, 1e-9, 7.5e-5, 2.5e8) == 0.0


def test_c6_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        StreamConfig(n_particles=-1)
    with pytest.raises(ValueError):
        StreamConfig(angle_mean_deg=120.0)
    with pytest.raises(ValueError):
        StreamConfig(d_p_m=0.0)
    with pytest.raises(ValueError):
        sphere_cap_volume(-1e-6, 7.5e-5)


# ---------------------------------------------------------------- C7: breakage
def test_c7_identity_fragmentation_is_bitwise_transparent():
    cfg = StreamConfig(n_particles=120)
    plain = run_impact_stream(cfg)
    same = run_impact_stream(cfg, on_impact=fragmentation_hook(1))
    for key in ("erosion_ratio_kg_kg", "removed_mass_kg", "closure_residual", "impacts"):
        assert same[key] == plain[key], key
    assert abs(same["closure_residual"] - plain["closure_residual"]) < 1e-6


def test_c7_erosion_core_has_no_comminution_import():
    report = erosion_core_imports_comminution()
    assert report["decoupled"] is True
    assert report["coupled_modules"] == []


def test_c7_static_audit_detects_a_coupling(tmp_path):
    """Negative control: the audit must fire on an actually coupled module."""
    (tmp_path / "crater.py").write_text("import tensordem.erosion_pbm.erosion\n")
    (tmp_path / "contact.py").write_text("import math\n")
    (tmp_path / "stream.py").write_text("from tensordem.granule_crush import x\n")
    report = erosion_core_imports_comminution(tmp_path)
    assert report["decoupled"] is False
    assert {h["module"] for h in report["coupled_modules"]} == {"crater.py", "stream.py"}


def test_c7_fragmentation_conserves_erodent_mass():
    out = run_impact_stream(
        StreamConfig(n_particles=120, angle_mean_deg=20.0, angle_std_deg=0.0),
        on_impact=fragmentation_hook(4),
    )
    residual = abs(
        out["carried_mass_end_kg"] + out["sibling_mass_kg"] - out["delivered_mass_kg"]
    ) / out["delivered_mass_kg"]
    assert residual < 1e-12
    assert out["sibling_mass_kg"] > 0.0


def test_c7_fragmentation_changes_erosion_one_way():
    """Comminution drives erosion (finer erodent), never the other way round."""
    cfg = StreamConfig(n_particles=150, angle_mean_deg=20.0, angle_std_deg=0.0)
    plain = run_impact_stream(cfg)["erosion_ratio_kg_kg"]
    frag = run_impact_stream(cfg, on_impact=fragmentation_hook(4))["erosion_ratio_kg_kg"]
    assert frag != plain


# ---------------------------------------------------------------- contact
def test_contact_plastic_restitution_quarter_power():
    v_y = 0.05
    assert plastic_normal_restitution(0.01, v_y) == 1.0
    for v in (0.5, 1.0, 10.0):
        assert plastic_normal_restitution(v, v_y) == pytest.approx((v_y / v) ** 0.25)


def test_contact_yield_velocity_scaling():
    """``v_y ~ sqrt(R**3 p_y**5 / (E*star**4 m))`` -- verified by a ratio test."""
    m = 2650 * math.pi / 6 * (150e-6) ** 3
    v1 = yield_velocity_m_s(m, 7.5e-5, 0.25e9, 5e10)
    v2 = yield_velocity_m_s(m, 2 * 7.5e-5, 0.25e9, 5e10)
    assert v2 / v1 == pytest.approx(math.sqrt(2.0**3), rel=1e-12)
    v3 = yield_velocity_m_s(m, 7.5e-5, 2 * 0.25e9, 5e10)
    assert v3 / v1 == pytest.approx(2.0**2.5, rel=1e-12)


def test_contact_oblique_restitution_never_adds_energy():
    rng = np.random.default_rng(11)
    for _ in range(200):
        v = rng.uniform(1.0, 80.0)
        a = rng.uniform(0.0, 90.0)
        e = plastic_normal_restitution(v * math.sin(math.radians(a)), 0.04)
        reb = oblique_restitution(v, a, e, 0.5)
        assert reb["kinetic_energy_out_per_kg"] <= reb["kinetic_energy_in_per_kg"]
        assert 0.0 <= reb["tangential_restitution"] <= 1.0
        assert reb["dissipated_fraction"] >= 0.0


def test_contact_sticking_at_normal_incidence():
    reb = oblique_restitution(30.0, 90.0, 0.2, 0.35)
    assert reb["sticking"] is True
    assert reb["tangential_speed_out_m_s"] == 0.0


def test_contact_grazing_impact_keeps_tangential_speed():
    reb = oblique_restitution(30.0, 1.0, 0.2, 0.35)
    assert reb["tangential_restitution"] > 0.98


def test_contact_flight_time_matches_free_fall_and_angle():
    kin = JointImpactKinematics(0.05)
    assert kin.flight_time_s(0.0, 45.0) == pytest.approx(math.sqrt(2 * 0.05 / 9.81), rel=1e-12)
    out = kin.impact(30.0, 15.0)
    assert out["impact_angle_deg"] == pytest.approx(15.0, abs=0.3)
    assert out["impact_speed_m_s"] == pytest.approx(math.sqrt(30.0**2 + 2 * 9.81 * 0.05), rel=1e-9)


def test_law_angle_factor_normalised_at_anchor():
    assert float(CASE_LAW.angle_factor(90.0)) == pytest.approx(1.0, rel=1e-12)


def test_package_smoke():
    """Import surface stays coherent (no cyclic imports, exports present)."""
    import tensordem.erosion_dem as pkg

    for name in ("run_impact_stream", "single_impact_crater", "CASE_LAW", "StreamConfig"):
        assert name in pkg.__all__