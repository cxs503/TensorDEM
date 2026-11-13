"""Machine gate for the CASE-COLUMN-COLLAPSE deposit/archive audit (m18385).

Two layers, both required:

* **Mechanical layer** — every diagnostic in
  :mod:`tensordem.granular_column.deposit_audit` is checked against a synthetic
  configuration whose value is known analytically (rings, pure dipole /
  quadrupole, a solid cone whose ``h = 3 h_eq`` relation is exact, a radially
  degenerate deposit where any subsample must give the identical metric), plus
  the input-validation paths.
* **Data layer** — the numbers quoted in
  ``docs/AUDIT_COLUMN_COLLAPSE_DEPOSIT_ARCHIVE_m18385.md`` are asserted against
  the archived reduction ``docs/artifacts/column_collapse_deposit_audit_m18385/``
  (``summary.json`` + ``cases_deposit_audit.csv``), so the report cannot drift
  from the data.  A manipulation guard checks that the archived scalar runout
  of a case equals the value recomputed from its raw coordinates.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column.deposit_audit import (  # noqa: E402
    azimuthal_moments, azimuthal_shuffle_floor, azimuthal_uniform_floor,
    decimated, decimation_bias,
    edge_effective_count, floor_clearance, horizontal_com_drift, metric_bundle,
    shape_factor, subsample_noise)

ART = os.path.join(ROOT, "docs", "artifacts", "column_collapse_deposit_audit_m18385")


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
def ring(n=360, r=1.0, z=0.0):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, z)])


def antipodal(r=1.0):
    """Two opposite grains: A1 = 0, A2 = A4 = A6 = 1 exactly."""
    th = np.array([0.0, math.pi])
    r = np.array([1.0, r])
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(2)])


def random_ring(n=500, r=1.0, seed=11):
    th = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(n)])


def solid_cone(n=40000, r_base=1.0, height=1.5, seed=7):
    """Grains uniform in the volume of a cone (apex up) -> ``h_max = 3 h_eq``."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(o) for o in out) < n:
        z = rng.uniform(0.0, height, 4 * n)
        u = rng.uniform(0.0, 1.0, 4 * n)
        keep = u <= (1.0 - z / height) ** 2          # area of the horizontal slice
        z = z[keep]
        rad = r_base * (1.0 - z / height) * np.sqrt(rng.uniform(0.0, 1.0, z.size))
        th = rng.uniform(0.0, 2.0 * math.pi, z.size)
        out.append(np.column_stack([rad * np.cos(th), rad * np.sin(th), z]))
    return np.vstack(out)[:n]


def monolayers(n=2000, r_disc=1.0, z=0.0, seed=3):
    rng = np.random.default_rng(seed)
    rad = r_disc * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    return np.column_stack([rad * np.cos(th), rad * np.sin(th), np.full(n, z)])


# --------------------------------------------------------------------------- #
# mechanical layer
# --------------------------------------------------------------------------- #
def test_com_drift_of_a_coaxial_ring_is_zero():
    d = horizontal_com_drift(ring(), radius=0.05)
    assert abs(d["com_x"]) < 1e-15 and abs(d["com_y"]) < 1e-15
    assert d["drift"] < 1e-15
    assert d["drift_over_outer"] < 1e-15


def test_com_drift_recovers_a_known_translation():
    pos = ring() + np.array([0.01, -0.02, 0.0])
    d = horizontal_com_drift(pos, radius=0.05)
    assert d["drift"] == pytest.approx(math.hypot(0.01, 0.02), rel=1e-12)
    outer = metal_outer(pos, 0.05)
    assert d["drift_over_outer"] == pytest.approx(d["drift"] / outer, rel=1e-12)


def metal_outer(pos, radius):
    rr = np.hypot(pos[:, 0], pos[:, 1])
    return float(np.quantile(rr, 0.995)) + radius


def test_azimuthal_moments_are_exact_for_pure_multipoles():
    m = azimuthal_moments(antipodal(), orders=(1, 2, 4, 6))
    assert m["a1"] == pytest.approx(0.0, abs=1e-12)     # dipole cancels
    for key in ("a2", "a4", "a6"):                      # m even -> both grains in phase
        assert m[key] == pytest.approx(1.0, rel=1e-12)
    # equally spaced sets give zero for every order that does not wrap a full period
    m360 = azimuthal_moments(ring(360), orders=(1, 2, 6))
    for key in ("a1", "a2", "a6"):
        assert m360[key] < 1e-12
    # ... and a 90 deg 4-grain cross has a2 = a6 = 0 but a4 = 1
    cross = azimuthal_moments(ring(4), orders=(2, 4))
    assert cross["a2"] < 1e-15 and cross["a4"] == pytest.approx(1.0, rel=1e-12)


def test_azimuthal_moment_of_a_half_mass_dipole():
    # half the grains at theta=0, half at theta=pi with r weights summing equally:
    # |sum r e^{i theta}| / sum r must be exactly the imbalance 0.
    half = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert azimuthal_moments(half, orders=(1,))["a1"] < 1e-15
    lopsided = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert azimuthal_moments(lopsided, orders=(1,))["a1"] == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_uniform_angle_floor_detects_a_real_anisotropy():
    # 40 % of the grains piled up at theta = 0 on a disc with varying radii:
    # A1 = 0.46 measured, uniform-angle null floor q95 = 0.088 -> a 5x signal.
    rng = np.random.default_rng(4)
    th = rng.uniform(0.0, 2.0 * math.pi, 400)
    th[:160] = 0.0
    rad = np.sqrt(rng.uniform(0.05, 1.0, th.size))
    pos = np.column_stack([rad * np.cos(th), rad * np.sin(th), np.zeros(th.size)])
    measured = azimuthal_moments(pos, orders=(1,))["a1"]
    floor = azimuthal_uniform_floor(pos, orders=(1,), n_trials=64, seed=2)["a1"]
    assert measured > 0.4
    assert measured > 4.0 * floor["q95"]                 # signal far above the null
    assert 0.0 < floor["q95"] < floor["max"]
    again = azimuthal_uniform_floor(pos, orders=(1,), n_trials=64, seed=2)["a1"]
    assert floor["mean"] == again["mean"]                # seeded -> reproducible


def test_permutation_floor_cannot_see_anisotropy_when_the_radii_are_equal():
    """Documented limitation: permuting angles keeps the angle multiset (hence the anisotropy)."""
    th = np.concatenate([np.linspace(0.0, 2.0 * math.pi, 300, endpoint=False),
                         np.zeros(100)])                  # 25 % pile-up at theta = 0
    ring_pos = np.column_stack([np.cos(th), np.sin(th), np.zeros(th.size)])
    measured = azimuthal_moments(ring_pos, orders=(1,))["a1"]
    perm = azimuthal_shuffle_floor(ring_pos, orders=(1,), n_trials=16, seed=2)["a1"]
    unif = azimuthal_uniform_floor(ring_pos, orders=(1,), n_trials=64, seed=2)["a1"]
    assert measured == pytest.approx(0.25, rel=1e-12)
    assert perm["mean"] == pytest.approx(measured, rel=1e-12)   # blind by construction
    assert unif["mean"] < 0.3 * measured                        # the correct null
    assert 0.005 < unif["mean"] < 0.1


def test_shuffle_floor_scales_like_one_over_sqrt_n():
    small = azimuthal_shuffle_floor(ring(200), orders=(1,), n_trials=128, seed=5)["a1"]["mean"]
    large = azimuthal_shuffle_floor(ring(2000), orders=(1,), n_trials=128, seed=5)["a1"]["mean"]
    # mean of |sum e^{i theta}| ~ C/sqrt(n) for the azimuth-shuffled null
    assert large < small
    assert small / large > 1.5


def test_floor_clearance_reads_the_floor_contact_overlap():
    pos = monolayers(n=100, z=0.0)
    pos[:, 2] = 1.0            # resting on the floor (z = radius)
    pos[0, 2] = 0.99           # one grain pressed 1 % of the radius into the floor
    pos[1, 2] = 0.70           # one grain pushed through
    d = floor_clearance(pos, radius=1.0)
    assert d["min_clearance_over_radius"] == pytest.approx(-0.30, rel=1e-12)
    assert d["frac_below_minus_20pct"] == pytest.approx(1.0 / 100.0, rel=1e-12)
    assert d["frac_below_minus_5pct"] == pytest.approx(1.0 / 100.0, rel=1e-12)


def test_shape_factor_is_three_for_a_solid_cone():
    n, r_base, height = 40000, 1.0, 1.5
    cone = solid_cone(n=n, r_base=r_base, height=height)
    volume = math.pi * r_base ** 2 * height / 3.0
    radius = (3.0 * volume / (4.0 * math.pi * n)) ** (1.0 / 3.0)   # fills the cone
    s = shape_factor(cone, radius=radius)
    assert s["cone_factor"] == pytest.approx(3.0, rel=0.05)


def test_shape_factor_uses_the_true_grain_count_for_a_truncated_archive():
    n, r_base, height = 20000, 1.0, 1.5
    cone = solid_cone(n=n, r_base=r_base, height=height)
    radius = (3.0 * (math.pi * r_base ** 2 * height / 3.0) / (4.0 * math.pi * n)) ** (1 / 3)
    full = shape_factor(cone, radius=radius, n_total=cone.shape[0])
    sub = cone[::4]
    truncated = shape_factor(sub, radius=radius, n_total=n)
    assert truncated["cone_factor"] == pytest.approx(full["cone_factor"], rel=0.05)
    wrong = shape_factor(sub, radius=radius)          # n_total omitted -> 4x too tall
    assert wrong["cone_factor"] > 2.5 * full["cone_factor"]


def test_shape_factor_grows_with_aspect_ratio():
    def cf(height, n=20000, r_base=1.0):
        cone = solid_cone(n=n, r_base=r_base, height=height)
        radius = (3.0 * (math.pi * r_base ** 2 * height / 3.0) / (4.0 * math.pi * n)) ** (1 / 3)
        return shape_factor(cone, radius=radius)["cone_factor"]
    assert cf(2.5) > cf(0.5)


def test_radially_degenerate_deposit_has_no_subsample_bias():
    ring_pos = ring(400, r=1.0)
    dec = decimation_bias(ring_pos, radius=0.02, strides=(2, 4), orders=(1,))
    for stride in (2, 4):
        assert dec[stride]["delta_p995_over_outer"] == pytest.approx(0.0, abs=1e-12)
        assert dec[stride]["max_delta_p995_over_outer"] == pytest.approx(0.0, abs=1e-12)


def test_subsample_noise_vanishes_on_a_ring_and_is_positive_on_a_disc():
    ring_noise = subsample_noise(ring(400, r=1.0), radius=0.01, n_trials=32)[ "0.25"]
    assert ring_noise["std_over_outer"] == pytest.approx(0.0, abs=1e-12)
    disc = monolayers(n=4000, r_disc=1.0)
    disc[0, :2] = (2.5, 0.0)                        # one escaping grain
    noise = subsample_noise(disc, radius=0.02, fractions=(1.0, 0.25), n_trials=64)
    assert noise["1.0"]["std_over_outer"] == pytest.approx(0.0, abs=1e-12)
    assert noise["0.25"]["std_over_outer"] > 0.0


def test_edge_effective_count_matches_construction():
    # 990 grains at r = 0.5 and 10 scattered grains at r = 1.0: the 99.5 % quantile
    # lands on the outer shell, so exactly those 10 grains define the metric.
    th = np.linspace(0.0, 2.0 * math.pi, 990, endpoint=False)
    inner = np.column_stack([0.5 * np.cos(th), 0.5 * np.sin(th), np.zeros(990)])
    th2 = np.linspace(0.0, 2.0 * math.pi, 10, endpoint=False)
    outer = np.column_stack([np.cos(th2), np.sin(th2), np.zeros(10)])
    eff = edge_effective_count(np.vstack([inner, outer]), radius=0.01)
    assert eff["n_eff"] == 10
    assert eff["n_eff_fraction"] == pytest.approx(0.01, rel=1e-12)


def test_decimated_is_deterministic_and_rejects_bad_strides():
    pos = ring(10)
    assert np.array_equal(decimated(pos, 2, 1), pos[1::2])
    assert decimated(pos, 1).shape == pos.shape
    with pytest.raises(ValueError):
        decimated(pos, 0)
    with pytest.raises(ValueError):
        decimated(pos[:2], 4)


@pytest.mark.parametrize("bad", [
    np.zeros((0, 3)),                       # empty
    np.zeros((4, 2)),                       # wrong shape
    np.array([[0.0, 0.0, np.nan]]),         # non-finite
])
def test_invalid_positions_are_rejected(bad):
    with pytest.raises(ValueError):
        azimuthal_moments(bad)


def test_invalid_scalars_are_rejected():
    with pytest.raises(ValueError):
        horizontal_com_drift(ring(), radius=0.0)
    with pytest.raises(ValueError):
        floor_clearance(ring(), radius=-1.0)
    with pytest.raises(ValueError):
        azimuthal_moments(ring(), orders=(0,))
    with pytest.raises(ValueError):
        azimuthal_shuffle_floor(ring(), n_trials=1)
    with pytest.raises(ValueError):
        edge_effective_count(ring(), radius=1.0, quantile=0.2)
    with pytest.raises(ValueError):
        subsample_noise(ring(), radius=0.1, fractions=(0.0,))
    with pytest.raises(ValueError):
        shape_factor(ring(10), radius=0.1, n_total=5)


def test_metric_bundle_round_trip():
    pos = monolayers(n=500)
    b = metric_bundle(pos, radius=0.02, n_total=2000)
    assert b["shape"]["n_used_for_volume"] == 2000
    assert set(b) == {"com", "moments", "floor", "shape"}


# --------------------------------------------------------------------------- #
# data layer: the report numbers must equal the archived reduction
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def summary():
    with open(os.path.join(ART, "summary.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def rows():
    with open(os.path.join(ART, "cases_deposit_audit.csv")) as fh:
        return list(csv.DictReader(fh))


def test_corpus_size(summary):
    assert summary["verdicts"]["n_cases"] == 25
    assert summary["verdicts"]["n_with_positions"] == 25


def test_report_integrity_is_exact_on_full_archives(summary):
    v = summary["verdicts"]["H_INT"]
    assert v["verdict"] == "supported"
    assert v["n_full_archives"] == 23
    assert v["max_p995_rel_diff"] == 0.0
    assert v["max_max_rel_diff"] == 0.0
    # the two truncated archives are off by <1 % and are named
    assert 0.0 < v["max_p995_rel_diff_short"] < 1e-2
    assert set(v["short_cases"]) == {"w5_law_gpr12_a3", "w5c_gpr12_a3"}


def test_archive_truncation_is_detected(summary):
    v = summary["verdicts"]["H_ARCH"]
    assert v["verdict"] == "falsified"
    offenders = {o["case"]: o for o in v["offenders"]}
    assert set(offenders) == {"w5_law_gpr12_a3", "w5c_gpr12_a3"}
    for o in offenders.values():
        assert o["n_particles"] == 19687 and o["n_archived"] == 4922
        assert 3.9 < o["archived_stride"] < 4.1


def test_symmetry_verdict_is_driven_by_the_truncated_archives(summary):
    v = summary["verdicts"]["H_SYM"]
    assert v["verdict"] == "falsified"                 # pre-registered, all cases
    assert v["verdict_full_archives"] == "supported"    # 23 valid archives
    assert v["max_drift_full_archives"] < 5e-3
    assert v["max_drift_subsampled_archives"] > 1e-2   # COM of a sub-lattice is biased


def test_footprint_is_statistically_axisymmetric(summary):
    v = summary["verdicts"]["H_ANISO"]
    assert v["verdict"] == "supported"
    assert v["a1_median_ratio"] < 0.05 and v["a2_median_ratio"] < 0.05
    assert v["a1_median_ratio"] < v["a1_max_ratio"]


def test_floor_contact_is_bounded(summary):
    v = summary["verdicts"]["H_FLOOR"]
    assert v["verdict"] == "supported"
    assert -0.2 < v["min_clearance_over_radius"] < 0.0
    assert v["max_frac_below_minus_20pct"] == 0.0


def test_metric_resolution_exceeds_three_percent(summary):
    v = summary["verdicts"]["H_METRIC"]
    assert v["verdict"] == "supported"
    assert v["median_noise25_over_outer"] > 0.03
    assert v["median_noise10_over_outer"] > v["median_noise25_over_outer"] > \
        v["median_noise50_over_outer"]
    assert 5 <= v["min_n_eff"] <= v["median_n_eff"] <= 40


def test_stride_subsampling_is_worse_than_random_subsampling(summary):
    v = summary["verdicts"]["H_SUB"]
    assert v["verdict"] == "falsified"
    metric = summary["verdicts"]["H_METRIC"]
    # structured (HCP sublattice) truncation costs an order of magnitude more
    assert v["max_dec4_delta"] > 3.0 * metric["max_noise25_over_outer"]
    assert v["max_dec4_delta"] > 0.2                    # 24.8 % of R_out, measured
    assert v["case_of_max_dec4"].startswith("w5b_confirm_wall0.8_a1.5")


def test_rows_are_consistent_with_the_recomputed_lube_law(rows):
    for r in rows:
        a = float(r["a_measured"])
        ref = 1.0 + 1.24 * a if a < 1.7 else 1.0 + 1.6 * math.sqrt(a)
        assert float(r["reference_runout_recomputed"]) == pytest.approx(ref, rel=1e-12)
        err = 100.0 * (float(r["stored_runout_ratio"]) / ref - 1.0)
        assert float(r["runout_error_pct"]) == pytest.approx(err, rel=1e-9, abs=1e-9)


def test_closure_arm_row_matches_the_published_round5_value(rows):
    row = next(r for r in rows if r["case"] == "w5b_confirm_wall0.8_a0.5")
    assert float(row["stored_runout_ratio"]) == pytest.approx(1.5729, abs=5e-4)
    assert float(row["runout_error_pct"]) == pytest.approx(-6.4, abs=0.1)
    assert float(row["com_drift_over_outer"]) < 1e-4
    assert int(row["n_eff"]) >= 5


def test_manifest_covers_every_case(rows):
    with open(os.path.join(ART, "manifest_sha256.json")) as fh:
        man = json.load(fh)
    assert set(man["files"]) == {r["file"] for r in rows}
    assert all(len(h) == 64 for h in man["files"].values())
    with open(os.path.join(ART, "summary.json")) as fh:
        figs = json.load(fh)["figures_sha256"]
    for name in ("fig_azimuthal_anisotropy.png", "fig_decimation_bias.png",
                 "fig_com_drift.png", "fig_footprint_profiles.png",
                 "fig_runout_error_vs_metric_noise.png"):
        assert name in figs and len(figs[name]) == 64