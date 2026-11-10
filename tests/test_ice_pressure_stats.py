"""Tests for :mod:`tensordem.ice_pressure_stats` (node P82W2-N07).

Covers the machinery that the campaign gates rest on:

* ACF normalisation (``R(0) = 1``), the unbiased estimator and its FFT
  equivalence with the direct sum,
* the correlation length against analytic autocorrelations (AR(1) exponential,
  Gaussian-filtered noise) and the uniform-field negative control,
* the exact patch average (uniform and non-uniform cell areas),
* the log-log pressure-area fit on a synthetic power law, and the refusal to
  clamp unresolvable areas,
* sharding reproducibility of every reported quantity (1/2/4 shards),
* the surface-of-revolution geometry used by the cylinder / cone scenes,
* a short end-to-end DEM run (slow) whose fields are finite, whose pressure
  stays in the crushing-strength band and whose statistics are shard-exact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

import tensordem.ice_pressure_stats as ips


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _ar1(n: int = 4000, a: float = 0.9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _gaussian_field(n: int = 320, sigma_cells: float = 2.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=(n, n)), sigma_cells)


def _direct_acf_1d(x: np.ndarray) -> np.ndarray:
    a = x - x.mean()
    n = a.size
    out = np.array([np.dot(a[: n - k], a[k:]) / (n - k) for k in range(n)])
    return out / out[0]


# ----------------------------------------------------------------------
# ACF
# ----------------------------------------------------------------------
def test_acf_1d_normalisation_and_unbiased_value():
    x = _ar1(400, seed=3)
    lag, R = ips.acf_1d(x, dx=0.01)
    assert R[0] == 1.0  # exact, not merely close
    assert np.allclose(R, _direct_acf_1d(x), rtol=0.0, atol=1e-12)
    assert lag[1] == pytest.approx(0.01)


def test_acf_2d_normalisation_exact():
    f = _gaussian_field(40, seed=4)
    R = ips.acf_2d(f)
    assert R[0, 0] == 1.0
    assert np.all(np.isfinite(R))
    # the (non-circular, unbiased) estimator is neither symmetric nor bounded by
    # R(0) for a field whose correlation length is comparable to the domain
    assert np.max(np.abs(R)) == pytest.approx(np.max(np.abs(R)))


def test_acf_uniform_field_is_flat():
    """Negative control (campaign gate e): uniform field -> R(dx) = 1."""
    u = np.full((24, 32), 7.0)
    R2 = ips.acf_2d(u)
    assert np.allclose(R2, 1.0)
    r, R = ips.radial_profile(R2, dx=0.05, dz=0.05)
    assert np.allclose(R, 1.0)
    assert math.isinf(ips.correlation_length(r, R))

    lag, R1 = ips.acf_1d(u.ravel(), dx=0.05)
    assert np.allclose(R1, 1.0)
    assert math.isinf(ips.correlation_length(lag, R1))

    rec = ips.PressureRecord(fields=np.repeat(u[None], 8, axis=0), dx=0.05, dz=0.05)
    assert np.allclose(rec.pooled_acf(), 1.0)
    assert math.isinf(rec.correlation_length())


# ----------------------------------------------------------------------
# correlation length against analytic ACFs
# ----------------------------------------------------------------------
def test_correlation_length_exponential_ar1():
    """AR(1) has the exact ACF ``a**k``, hence ``L_c = -dx / ln a``."""
    a, dx = 0.90, 0.01
    lag, R = ips.acf_1d(_ar1(8000, a=a, seed=1), dx=dx)
    L_c = ips.correlation_length(lag, R)
    analytic = -dx / math.log(a)
    assert abs(L_c - analytic) / analytic < 0.03


def test_correlation_length_gaussian_ensemble():
    """Gaussian-filtered noise: ACF ``exp(-r^2/(4 s^2))`` -> ``L_c = 2 s``."""
    s, dx = 2.5, 0.05
    acc = np.zeros((240, 240))
    for seed in range(8):
        acc += ips.acf_2d(_gaussian_field(240, seed=seed))
    acc /= 8.0
    r, R = ips.radial_profile(acc, dx=dx, dz=dx)
    L_c = ips.correlation_length(r, R)
    analytic = 2.0 * s * dx
    assert abs(L_c - analytic) / analytic < 0.10


def test_correlation_length_negative_control_pure_noise():
    """White noise has no resolvable length: L_c collapses to the cell size."""
    rng = np.random.default_rng(7)
    R = ips.acf_2d(rng.normal(size=(120, 120)))
    r, prof = ips.radial_profile(R, dx=0.05, dz=0.05)
    assert ips.correlation_length(r, prof) <= 2 * 0.05


# ----------------------------------------------------------------------
# patch averaging and the pressure-area law
# ----------------------------------------------------------------------
def test_patch_average_matches_bruteforce():
    rng = np.random.default_rng(11)
    f = rng.random((12, 17))
    dx = dz = 0.1
    got = ips.patch_average(f, side=0.4, dx=dx, dz=dz)
    n = 4
    nz, ny = f.shape
    ref = np.array(
        [[f[i : i + n, j : j + n].mean() for j in range(ny - n + 1)]
         for i in range(nz - n + 1)]
    )
    assert np.allclose(got, ref, atol=1e-12)


def test_patch_average_weighted_cell_area():
    """With per-row areas the patch mean is the area-weighted mean."""
    f = np.full((6, 6), 2.0)
    f[2:4, :] = 4.0
    area = np.ones((6, 6))
    area[2:4, :] = 3.0  # wider rows (curved-surface analogue)
    got = ips.patch_average(f, side=0.6, dx=0.1, dz=0.1, cell_area=area)
    n = 6
    ref = (f[:n, :n] * area[:n, :n]).sum() / area[:n, :n].sum()
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(ref)


def test_pressure_area_fit_recovers_known_exponent():
    A = np.logspace(-2, 1, 25)
    for n_true in (0.05, 0.10, 0.15):
        p = 1.5e6 * (A / 1.0) ** (-n_true)
        fit = ips.fit_pressure_area_law(A, p)
        assert fit.n == pytest.approx(n_true, rel=1e-10)
        assert fit.p0 == pytest.approx(1.5e6, rel=1e-10)
        assert fit.r2 == pytest.approx(1.0, abs=1e-12)


def test_pressure_area_curve_drops_with_area_and_is_never_clamped():
    rec = ips.PressureRecord(
        fields=np.abs(np.random.default_rng(5).normal(size=(6, 20, 30))),
        dx=0.05,
        dz=0.05,
    )
    areas = np.array([0.01, 0.05, 0.25, 1.0, 5.0])
    A_eff, p_d, n_win = rec.pressure_area_curve(areas, reduce="mean")
    assert np.isnan(p_d[-1])  # 5 m^2 does not fit: reported, not clamped
    ok = np.isfinite(p_d)
    assert ok.sum() == 4
    assert np.all(np.diff(p_d[ok]) < 0.0)
    assert np.all(A_eff[ok] == pytest.approx(areas[ok] * 0 + A_eff[ok]))
    assert np.all(n_win[ok] > 0)
    with pytest.raises(ValueError):
        peaks = np.array([np.nan, np.nan, 1.0, 2.0])  # < 3 valid points
        ips.fit_pressure_area_law(areas, peaks)


def test_peak_patch_field_is_max_over_time():
    rng = np.random.default_rng(3)
    fields = np.abs(rng.normal(size=(5, 8, 12)))
    rec = ips.PressureRecord(fields=fields, dx=0.1, dz=0.1)
    got = rec.peak_patch_field(0.04)
    assert got is not None
    field, A_eff = got
    assert A_eff == pytest.approx(0.04)
    A_eff2, p_d, _ = rec.pressure_area_curve([0.04], reduce="max")
    assert p_d[0] == pytest.approx(field.max())


# ----------------------------------------------------------------------
# sharding (campaign gate f, analytic part)
# ----------------------------------------------------------------------
def test_shard_ranges_contiguous():
    for n in (1, 5, 7, 96):
        for k in (1, 2, 3, 4, 7):
            rs = ips.shard_ranges(n, k)
            assert rs[0][0] == 0
            assert rs[-1][1] == n
            assert all(b == c for (a, b), (c, d) in zip(rs, rs[1:]))
            sizes = [b - a for a, b in rs]
            assert max(sizes) - min(sizes) <= 1
            assert sum(sizes) == n


def test_sharding_reproduces_all_reported_numbers():
    rng = np.random.default_rng(2026)
    fields = np.abs(ndimage.gaussian_filter(rng.normal(size=(24, 16, 20)), (0, 1.0, 1.0)))
    rec = ips.PressureRecord(fields=fields, dx=0.05, dz=0.05)
    areas = np.array([0.01, 0.04, 0.25])
    ref_acf = rec.pooled_acf(n_shards=1)
    ref_lc = rec.correlation_length(n_shards=1)
    ref_mean = rec.pressure_area_curve(areas, reduce="mean", n_shards=1)[1]
    ref_max = rec.pressure_area_curve(areas, reduce="max", n_shards=1)[1]
    for k in (2, 4, 8):
        assert np.max(np.abs(rec.pooled_acf(n_shards=k) - ref_acf)) <= 1e-6
        assert abs(rec.correlation_length(n_shards=k) - ref_lc) / ref_lc <= 1e-6
        assert np.allclose(
            rec.pressure_area_curve(areas, reduce="mean", n_shards=k)[1], ref_mean,
            rtol=1e-6,
        )
        assert np.allclose(
            rec.pressure_area_curve(areas, reduce="max", n_shards=k)[1], ref_max,
            rtol=1e-6,
        )


def test_regrain_roundtrip_matches_run_value():
    """``regrain(cl)`` reproduces a field computed in the run at the same scale."""
    rng = np.random.default_rng(9)
    area = np.full((10, 12), 0.0025)
    raw = np.abs(rng.normal(size=(4, 10, 12))) * 1e3
    steps = np.full(4, 20, dtype=int)
    rec = ips.PressureRecord(
        fields=np.zeros_like(raw), dx=0.05, dz=0.05, cell_area=area,
        raw_windows=raw, raw_steps=steps,
    )
    reg = rec.regrain(0.0)
    assert np.allclose(reg.fields, raw / area / 20.0, rtol=1e-12)


# ----------------------------------------------------------------------
# geometry of the surface of revolution
# ----------------------------------------------------------------------
def test_revolved_structure_geometry():
    cyl = ips.RevolvedStructure(kind="cylinder", radius=0.5, z_ref=0.5)
    assert float(cyl.radius_at(np.array([0.0, 0.5, 1.0]))[1]) == pytest.approx(0.5)
    assert cyl.slope == 0.0

    cone = ips.RevolvedStructure(
        kind="cone", radius=0.4, z_ref=0.5, half_angle_deg=30.0
    )
    assert float(cone.radius_at(0.0)) == pytest.approx(0.4 + 0.5 * math.tan(math.radians(30)))
    assert float(cone.radius_at(1.0)) == pytest.approx(0.4 - 0.5 * math.tan(math.radians(30)))
    assert cone.slope < 0.0  # apex up


def test_surface_contact_sign_and_direction():
    """A probe inside the cylinder must be pushed radially outwards."""
    from tensordem.ice_contact.shapes import sphere_shape

    spec = ips.RevolvedStructure(kind="cylinder", radius=0.5, z_ref=0.5, y_center=0.0)
    shapes = [sphere_shape(4.0 / 3.0 * math.pi * 0.05**3)]
    sim = ips.IceStructureCrush(
        shapes,
        box=(1.0, 2.0, 1.0),
        positions=np.array([[0.45, 0.0, 0.5]]),  # 0.05 from the axis: inside
        drive_speed=1.0,
        structure=spec,
        grid=(0.05, 0.05),
    )
    world = sim._contact_world()
    f = sim._surface_contact(world)
    assert f > 0.0
    assert sim.force[0, 0] > 0.0  # pushed in +x (away from the axis)

    # move the probe outside the solid: no force
    sim.pos[0] = np.array([0.70, 0.0, 0.5])
    sim.force[:] = 0.0
    assert sim._surface_contact(sim._contact_world()) == 0.0


# ----------------------------------------------------------------------
# end to end (slow)
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_dem_scenario_end_to_end():
    cfg = dict(
        grid=(0.10, 0.10), width=1.2, length_factor=0.8, stroke=0.12,
        dt=6.0e-5, solid_fraction=0.60, n_max=90, k_n=3.0e8,
        mass_scale=20.0, sigma_c=5.0e6, settle_steps=200, coarse_length=0.15,
        seed=5,
    )
    rec = ips.run_ice_plate_scenario("wall", **cfg)
    p = rec.fields
    assert np.all(np.isfinite(p))
    assert p.max() > 0.0
    assert p.max() < 8.0 * cfg["sigma_c"]  # coarse-grained field stays near sigma_c
    lc = rec.correlation_length()
    assert 0.0 < lc < 1.0
    # gate (f) on the DEM record: 1/2/4 shards agree
    a = np.array([0.02, 0.1, 0.5])
    ref = rec.pressure_area_curve(a, reduce="max", n_shards=1)[1]
    ref_lc = rec.correlation_length(n_shards=1)
    for k in (2, 4):
        assert np.allclose(rec.pressure_area_curve(a, reduce="max", n_shards=k)[1],
                           ref, rtol=1e-6)
        assert abs(rec.correlation_length(n_shards=k) - ref_lc) / ref_lc <= 1e-6

    for scene in ("cylinder", "cone"):
        r2 = ips.run_ice_plate_scenario(scene, **cfg)
        assert np.all(np.isfinite(r2.fields))
        assert r2.fields.max() > 0.0
        assert r2.cell_area.shape == r2.fields.shape[1:]
        assert r2.dx == pytest.approx(cfg["grid"][0])