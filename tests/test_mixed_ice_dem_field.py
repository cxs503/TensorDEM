"""Mixed FY/MY ice field DEM: layout algebra, material plumbing, criteria.

The DEM assertions here are *structural* (degeneration, bounds, sign of the
deviation) on deliberately short runs; the quantitative mixture-rule evidence is
produced by ``scripts/run_mixed_ice_dem_field.py`` and archived in
``artifacts/mixed_ice_dem_field/campaign.json`` so that the unit tests stay fast.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.level_ice.campaign import CaseSpec
from tensordem.level_ice.sheet import LevelIceSheet
from tensordem.mixed_ice import aging
from tensordem.mixed_ice import dem_field as D

#: short-but-real DEM settings for the unit tests (thin field, small advance)
FAST_BASE = dict(D.DEM_FIELD_BASE, advance_m=3.0, length_m=24.0, layers=2)
FAST_CONFIG = CaseSpec(**{k: v for k, v in FAST_BASE.items()
                          if k != "window_fraction"}).sheet_config()


# ---------------------------------------------------------------------------
# 1. layout algebra
# ---------------------------------------------------------------------------
def test_traversed_mask_matches_sheet_geometry():
    sheet = LevelIceSheet(FAST_CONFIG)
    advance = FAST_BASE["advance_m"]
    x0 = sheet.hull_position()
    centres = (np.arange(FAST_CONFIG.n_x) + 0.5) * FAST_CONFIG.spacing_m
    expected = (centres >= x0) & (centres <= x0 + advance)
    mask = D.traversed_mask(FAST_CONFIG.n_x, FAST_CONFIG.spacing_m,
                            FAST_CONFIG.thickness_m, FAST_BASE["bow_angle_deg"],
                            advance)
    assert np.array_equal(mask, expected)
    assert mask.sum() >= 2


def test_periodic_labels_expose_exactly_the_nominal_fraction():
    n_traversed = int(D.traversed_mask(FAST_CONFIG.n_x, FAST_CONFIG.spacing_m,
                                       FAST_CONFIG.thickness_m,
                                       FAST_BASE["bow_angle_deg"],
                                       FAST_BASE["advance_m"]).sum())
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        for phase in range(n_traversed):
            labels = D.periodic_labels(FAST_CONFIG.n_x, p, patch_columns=2,
                                       period_columns=n_traversed, phase=phase)
            mask = D.traversed_mask(FAST_CONFIG.n_x, FAST_CONFIG.spacing_m,
                                    FAST_CONFIG.thickness_m,
                                    FAST_BASE["bow_angle_deg"],
                                    FAST_BASE["advance_m"])
            assert labels[mask].mean() == pytest.approx(p, abs=1e-12)


def test_random_patch_labels_hit_the_target_fraction_and_form_patches():
    labels = D.patch_labels(200, 0.3, patch_columns=8, seed=7)
    assert labels.mean() == pytest.approx(0.3, abs=1.0 / 200)
    runs = D._patch_runs(labels)
    assert runs and max(runs) >= 8          # grouped into patches, not scattered
    assert D._patch_runs(np.zeros(10, dtype=bool)) == []


def test_patch_labels_rejects_bad_fractions():
    with pytest.raises(ValueError):
        D.patch_labels(20, 1.5, 4)
    with pytest.raises(ValueError):
        D.periodic_labels(20, -0.1, 4, 10)


# ---------------------------------------------------------------------------
# 2. material plumbing
# ---------------------------------------------------------------------------
def test_material_field_validation_and_summary():
    n = FAST_CONFIG.n_x * FAST_CONFIG.n_z
    with pytest.raises(ValueError):
        LevelIceSheet(FAST_CONFIG, material_field={"strength_scale": np.ones(n + 1)})
    with pytest.raises(ValueError):
        LevelIceSheet(FAST_CONFIG,
                      material_field={"strength_scale": -np.ones(n)})
    with pytest.raises(ValueError):
        LevelIceSheet(FAST_CONFIG, material_field={"nope": np.ones(n)})
    material = {"strength_scale": np.full(n, 2.0),
                "modulus_scale": np.full(n, 1.5),
                "crush_scale": np.full(n, 3.0)}
    sheet = LevelIceSheet(FAST_CONFIG, material_field=material)
    summary = sheet.material_summary()
    assert summary["heterogeneous"] is True
    assert summary["strength_scale"]["mean"] == pytest.approx(2.0)
    assert sheet.strength_t_b[0] == pytest.approx(2.0 * sheet.strength_t)
    # the *bond* thresholds are scaled by the strength multiplier (the crushing
    # cap uses the crush multiplier per contacting particle, see hull_forces)
    assert sheet.strength_c_b[0] == pytest.approx(2.0 * sheet.strength_c)
    assert sheet.node_crush_scale.mean() == pytest.approx(3.0)


def test_fy_class_scales_are_unity_and_my_exceeds_fy():
    scales = D.class_scales()
    fy, my = scales["FY"], scales["MY"]
    assert fy.strength_scale == pytest.approx(1.0, rel=0, abs=0)
    assert fy.modulus_scale == pytest.approx(1.0, rel=0, abs=0)
    assert fy.crush_scale == pytest.approx(1.0, rel=0, abs=0)
    assert my.strength_scale > 1.0 and my.modulus_scale > 1.0
    assert my.crush_scale == pytest.approx(
        my.compressive_strength_pa / fy.compressive_strength_pa)


def test_uniform_p0_field_is_bit_for_bit_the_existing_model():
    """Criterion (d): p = 0 must reproduce the homogeneous model exactly."""
    spec = D.FieldSpec(my_fraction=0.0, seed=3, base=FAST_BASE)
    field = D.run_field_case(spec)
    assert field["material_summary"]["heterogeneous"] is False
    config = CaseSpec(**{k: v for k, v in FAST_BASE.items()
                         if k != "window_fraction"}).sheet_config()
    sheet = LevelIceSheet(config)
    result = sheet.run(distance_m=FAST_BASE["advance_m"], record_stride=25,
                       window_fraction=tuple(FAST_BASE["window_fraction"]))
    assert field["mean_total_n_per_m"] == result.steady_mean("total_n_per_m")
    assert field["realised_my_fraction_area"] == 0.0


# ---------------------------------------------------------------------------
# 3. mixture-rule algebra
# ---------------------------------------------------------------------------
def test_mixture_rules_endpoints_and_ordering():
    r_fy, r_my = 5.0, 11.0
    for p in (0.0, 0.3, 0.6, 1.0):
        lin = D.linear_mixture(r_fy, r_my, p)
        ser = D.series_mixture(r_fy, r_my, p)
        assert ser <= lin + 1e-12
        assert min(r_fy, r_my) - 1e-12 <= ser <= max(r_fy, r_my) + 1e-12
    assert D.linear_mixture(r_fy, r_my, 0.0) == r_fy
    assert D.linear_mixture(r_fy, r_my, 1.0) == r_my
    assert D.series_mixture(r_fy, r_my, 0.0) == pytest.approx(r_fy)
    assert D.series_mixture(r_fy, r_my, 1.0) == pytest.approx(r_my)


def test_mixture_fit_recovers_a_line():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = 3.0 + 4.0 * p
    fit = D.mixture_fit(p, values)
    assert fit["intercept_n_per_m"] == pytest.approx(3.0)
    assert fit["slope_n_per_m"] == pytest.approx(4.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["max_residual_pct"] == pytest.approx(0.0, abs=1e-9)


def test_heterogeneous_field_is_bounded_by_the_pure_classes():
    """A mixed field must sit between the two pure classes (structural check).

    The mixture rule's *accuracy* is a statistical statement made by the
    campaign; here only the deterministic ordering is asserted, on a short run.
    """
    base = dict(D.DEM_FIELD_BASE, advance_m=3.0, layers=2)
    fy = D.run_field_case(D.FieldSpec(my_fraction=0.0, base=base))
    my = D.run_field_case(D.FieldSpec(my_fraction=1.0, base=base))
    mix = D.run_field_case(D.FieldSpec(my_fraction=0.5, seed=1,
                                       patch_length_m=2.0, layout="periodic",
                                       base=base))
    assert fy["mean_total_n_per_m"] < my["mean_total_n_per_m"]
    assert fy["mean_total_n_per_m"] < mix["mean_total_n_per_m"] < my["mean_total_n_per_m"]
    assert mix["traversed_my_fraction"] == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. criteria (a) and (c)
# ---------------------------------------------------------------------------
def test_criterion_a_strength_ratio_inside_the_literature_band():
    audit = D.strength_ratio_audit()
    assert audit["band"] == [1.5, 3.0]
    assert audit["criterion_a_pass"] is True
    assert 1.5 <= audit["ratio_my_over_fy"] <= 3.0
    # the pre-extension (porosity-only) law is below the band: this is the gap
    assert audit["ratio_porosity_only_my_over_fy"] < 1.5
    # every swept exponent keeps the ratio inside the band
    for row in audit["exponent_sensitivity"]:
        assert row["in_band"] is True


def test_criterion_a_fy_degeneration_is_exact_for_every_exponent():
    fy = aging.material_state(1.0)
    assert fy.compressive_strength_pa == pytest.approx(1.5e6, rel=1e-12)
    v_b = aging.brine_volume_fraction(aging.FY_SALINITY_PERMILLE,
                                      aging.FY_TEMPERATURE_C)
    for m in (0.0, 0.2, 0.35, 0.5, 1.0):
        value = aging.compressive_strength_pa(v_b, aging.FY_TEMPERATURE_C,
                                              temperature_exponent=m)
        assert value == pytest.approx(1.5e6, rel=1e-12)


def test_criterion_c_age_curve_is_monotone_and_saturating():
    curve = D.age_strength_curve()
    assert curve["criterion_c_pass"] is True
    assert curve["strictly_increasing"] is True
    sigma = curve["sigma_c_pa"]
    assert all(b > a for a, b in zip(sigma, sigma[1:]))
    assert curve["saturation_ratio_to_fy"] > 1.5
    # the asymptote is approached from below (saturating, not diverging)
    assert curve["sigma_c_pa"][-1] < curve["saturation_sigma_c_pa"] + 1e-9


def test_anchor_class_ratio_is_ordered_and_reflects_the_strength_law():
    new = D.anchor_class_ratio()
    old = D.anchor_class_ratio(
        sigma_c_my_pa=float(aging.compressive_strength_pa(
            aging.brine_volume_fraction(aging.salinity_permille(D.MY_CLASS_AGE_YEARS),
                                        aging.temperature_c(D.MY_CLASS_AGE_YEARS)))))
    assert new["class_ratio_anchor"] > old["class_ratio_anchor"] > 1.0
    assert new["crushing_ratio_anchor"] == pytest.approx(
        D.class_scales()["MY"].crush_scale)