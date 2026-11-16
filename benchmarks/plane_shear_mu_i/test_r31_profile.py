"""Round 31 tests: the depth-profile / stress-split instrument of the 2-D cell.

The instrument decides two ledger questions (R30 items 1 and 2), so it must be
tested against profiles whose answer is known analytically rather than merely
"executed once":

* :func:`couette_fit` recovers an exact Couette slope, is blind to a constant
  slip offset, and *detects* a localised (shear-band) profile;
* :func:`kinetic_stress_terms` returns zero shear traction for isotropic
  fluctuations and the exact ``-rho phi cov`` otherwise;
* :func:`ProfileAccumulator` conserves the particle count, reproduces a
  hand-computed contact virial, and returns ``nan`` (not 0) for empty bins;
* the whole-cell consistency identity ``p_bins == p_bulk`` holds when the bin
  virial is summed.

Run with ``python -m pytest benchmarks/plane_shear_mu_i/test_r31_profile.py -q``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[1]
for p in (str(REPO / "src"), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import profile_r31 as pr  # noqa: E402


# --------------------------------------------------------------------------- #
# binning
# --------------------------------------------------------------------------- #
def test_bin_index_maps_and_clips_the_edges():
    y0, y1, nb = 0.0, 1.0, 10
    y = np.array([-0.5, 0.0, 0.05, 0.95, 1.0, 1.5])
    k = pr.bin_index(y, y0, y1, nb)
    assert k.tolist() == [0, 0, 0, 9, 9, 9]


def test_bin_range_uses_the_centre_band_and_rejects_an_empty_cell():
    y0, y1 = pr.bin_range(0.20, 0.35, 0.005)
    assert y0 == pytest.approx(0.205) and y1 == pytest.approx(0.345)
    with pytest.raises(ValueError):
        pr.bin_range(0.0, 0.008, 0.005)


def test_clipping_keeps_disk_count_conserved():
    rng = np.random.default_rng(0)
    y = rng.uniform(-0.2, 1.2, 1000)
    k = pr.bin_index(y, 0.0, 1.0, 10)
    assert (k >= 0).all() and (k <= 9).all()
    assert np.bincount(k, minlength=10).sum() == len(y)


# --------------------------------------------------------------------------- #
# Couette fit: known answers
# --------------------------------------------------------------------------- #
def _couette(nb=10, gamma=10.0, y0=0.0, h=0.15, v=0.75, slip=0.0):
    y = y0 + (np.arange(nb) + 0.5) * h / nb
    ux = v + slip - gamma * (y - y0)
    return y, ux


def test_couette_fit_recovers_an_exact_linear_profile():
    y, ux = _couette()
    fit = pr.couette_fit(y, ux)
    assert fit["slope"] == pytest.approx(-10.0, rel=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_used"] >= 3


def test_couette_fit_is_blind_to_a_constant_slip_offset():
    y, ux = _couette(slip=0.3)
    fit = pr.couette_fit(y, ux)
    assert fit["slope"] == pytest.approx(-10.0, rel=1e-10)
    # the intercept now carries the slip; the slope must not absorb it
    assert fit["intercept"] == pytest.approx(0.75 + 0.3 - 0.0, rel=1e-9)


def test_couette_fit_flags_localisation():
    """Only the central band shears: the slope is too large and r2 collapses."""
    y, ux = _couette()
    ux = np.where(np.abs(y - np.median(y)) < 0.03, ux, ux[5] * 0.0 + 0.0)
    fit = pr.couette_fit(y, ux)
    assert fit["r2"] < pr.R2_LINEAR
    assert not (fit["r2"] >= pr.R2_LINEAR)


def test_couette_fit_ignores_empty_bins_and_needs_three():
    y, ux = _couette()
    ux = ux.copy()
    ux[[0, 4, 9]] = np.nan
    fit = pr.couette_fit(y, ux)
    # the fitted band keeps only its own central bins -> at least three, and the
    # slope must still be the exact one (the profile is collinear)
    assert fit["n_used"] >= 3
    assert fit["slope"] == pytest.approx(-10.0, rel=1e-10)
    assert math.isnan(pr.couette_fit(y[:2], ux[:2])["slope"])


# --------------------------------------------------------------------------- #
# kinetic stress
# --------------------------------------------------------------------------- #
def test_kinetic_stress_isotropic_fluctuations_carry_no_shear():
    out = pr.kinetic_stress_terms(0.8, 1e-4, 1e-4, 0.0, 2500.0)
    assert float(out["tau_kin"]) == pytest.approx(0.0)
    assert float(out["p_kin"]) == pytest.approx(2500.0 * 0.8 * 1e-4)


def test_kinetic_stress_uses_the_exact_shear_covariance():
    out = pr.kinetic_stress_terms(0.5, 2e-4, 8e-4, 3e-4, 1000.0)
    assert float(out["tau_kin"]) == pytest.approx(-1000.0 * 0.5 * 3e-4)
    assert float(out["p_kin"]) == pytest.approx(1000.0 * 0.5 * 0.5 * 1e-3)


# --------------------------------------------------------------------------- #
# the abscissa
# --------------------------------------------------------------------------- #
def test_inertial_number_scales_linearly_with_rate_and_inversely_with_pressure():
    I1 = pr.i_from_rate(10.0, 0.01, 1e4, 2500.0)
    I2 = pr.i_from_rate(20.0, 0.01, 1e4, 2500.0)
    I3 = pr.i_from_rate(10.0, 0.01, 4e4, 2500.0)
    assert I2 == pytest.approx(2.0 * I1)
    assert I3 == pytest.approx(0.5 * I1)
    assert I1 == pytest.approx(10.0 * 0.01 / math.sqrt(4.0))


# --------------------------------------------------------------------------- #
# accumulator on a synthetic cell with a known answer
# --------------------------------------------------------------------------- #
def _fake_cell(n_bins=10, n_per_bin=7, gamma=10.0, h=0.15, v=0.75,
               radius=5e-3, width=0.30, y_bottom=0.20, thickness=0.01,
               contacts=None, t_shear=0.0):
    y_top = y_bottom + h + 2.0 * radius
    ys = []
    for b in range(n_bins):
        yc = y_bottom + radius + (b + 0.5) * h / n_bins
        ys += [yc] * n_per_bin
    ys = np.asarray(ys)
    pos = np.zeros((len(ys), 2))
    pos[:, 1] = ys
    pos[:, 0] = np.linspace(0.0, width, len(ys), endpoint=False)
    vel = np.zeros_like(pos)
    vel[:, 0] = v - gamma * (pos[:, 1] - (y_bottom + radius))
    radii = np.full(len(ys), radius)
    if contacts is None:
        contacts = {"i": np.array([], dtype=np.int64),
                    "j": np.array([], dtype=np.int64),
                    "f_on_i": np.zeros((0, 2))}
    system = SimpleNamespace(pos=pos, vel=vel, radii=radii, contacts=contacts)
    case = SimpleNamespace(width=width, radius=radius, thickness=thickness)
    cell = SimpleNamespace(system=system, free_idx=np.arange(len(ys)),
                           y_bottom=y_bottom, y_top=y_top, case=case,
                           time=t_shear)
    acc = pr.ProfileAccumulator(n_bins=n_bins)
    acc.add(cell)
    return acc, cell


def test_accumulator_conserves_particles_and_measures_the_imposed_rate():
    acc, cell = _fake_cell()
    prof = acc.finalize(width=0.30, thickness=0.01, y_bottom=cell.y_bottom,
                        y_top=cell.y_top, radius=5e-3, density=2500.0,
                        v_wall=0.75)
    assert sum(prof["count_mean"]) == pytest.approx(70.0)
    assert prof["gamma_dot_measured"] == pytest.approx(10.0, rel=1e-9)
    assert prof["fit"]["r2"] == pytest.approx(1.0, abs=1e-12)
    d = pr.compare_to_nominal(prof, gap_eff=0.15, diameter=0.01,
                              pressure=1e4, density=2500.0)
    assert d["gamma_dot_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert d["I_measured"] == pytest.approx(d["I_nominal"], rel=1e-9)
    # no fluctuation in the synthetic field -> no kinetic stress at all
    assert prof["p_kinetic_total"] == pytest.approx(0.0)
    # per-bin stress ratio is well defined where contacts live
    assert np.isfinite(np.asarray(prof["mu_local"], dtype=float)).sum() == 0


def test_accumulator_reproduces_a_hand_computed_contact_virial():
    """Two contacts straddling y_bins, force chosen so the answer is by hand."""
    radius, width, thickness = 5e-3, 0.30, 0.01
    y_bottom, h = 0.20, 0.15
    contacts = {
        "i": np.array([0, 1], dtype=np.int64),
        "j": np.array([2, 3], dtype=np.int64),
        # repulsive: the force on i from j points *against* the branch vector
        "f_on_i": np.array([[-1.0, -0.5], [0.0, 2.0]]),
    }
    acc, cell = _fake_cell(contacts=contacts)
    # place the contact pairs inside known bins: (i,j) share a bin
    acc.samples = 0
    acc.n[:] = 0; acc.s_vx[:] = 0; acc.s_vy[:] = 0; acc.s_vx2[:] = 0
    acc.s_vy2[:] = 0; acc.s_vxvy[:] = 0; acc.s_area[:] = 0
    acc.c_count[:] = 0; acc.virial[:] = 0
    y0 = y_bottom + radius
    bin_h = h / acc.n_bins
    pos = cell.system.pos
    pos[0] = [0.05, y0 + 0.5 * bin_h]
    pos[2] = [0.05 + 1e-3, y0 + 0.5 * bin_h + 1e-4]
    pos[1] = [0.10, y0 + 2.5 * bin_h]
    pos[3] = [0.10 + 1e-3, y0 + 2.5 * bin_h + 1e-4]
    acc.add(cell)
    prof = acc.finalize(width=width, thickness=thickness, y_bottom=y_bottom,
                        y_top=cell.y_top, radius=radius, density=2500.0,
                        v_wall=0.75)
    vol_bin = width * bin_h * thickness
    # bin 0: r = (1e-3, 1e-4), f = (-1, -0.5) -> sigma_xx = -1e-3/vol,
    # sigma_yy = -0.5e-4/vol, so p = +0.5(1e-3 + 0.5e-4)/vol (compression > 0);
    # sigma_xy = -5e-4/vol, sigma_yx = -1e-4/vol -> tau = 0.5(5e-4 + 1e-4)/vol
    p0 = 0.5 * (1e-3 + 0.5e-4) / vol_bin
    tau0 = 0.5 * (5e-4 + 1e-4) / vol_bin
    assert prof["p_contact"][0] == pytest.approx(p0, rel=1e-9)
    assert prof["tau_contact"][0] == pytest.approx(tau0, rel=1e-9)
    assert prof["mu_local"][0] == pytest.approx(tau0 / p0, rel=1e-9)
    # bin 2: telegraphed through the periodic image in x must not change y
    assert prof["n_contacts_mean"][2] == pytest.approx(1.0)


def test_area_fraction_excludes_the_out_of_plane_thickness():
    """Regression: phi is a 2-D *area* fraction (``measure().phi_gap`` divides by
    ``W x gap`` with no thickness); multiplying by the thickness inflated it 100x."""
    acc, cell = _fake_cell(n_per_bin=7)
    prof = acc.finalize(width=0.30, thickness=0.01, y_bottom=cell.y_bottom,
                        y_top=cell.y_top, radius=5e-3, density=2500.0, v_wall=0.75)
    bin_h = prof["bin_height"]
    expected = 7 * math.pi * 5e-3 ** 2 / (0.30 * bin_h)
    assert prof["phi"][0] == pytest.approx(expected, rel=1e-12)
    assert prof["phi"][0] < 0.84          # below 2-D close packing, by construction


def test_empty_bins_are_nan_not_zero():
    acc, cell = _fake_cell()
    y0, y1 = pr.bin_range(cell.y_bottom, cell.y_top, 5e-3)
    k = pr.bin_index(cell.system.pos[:, 1], y0, y1, acc.n_bins)
    keep = ~np.isin(k, [4, 5])                        # blank out two mid bins
    cell.system.pos = cell.system.pos[keep]
    cell.system.vel = cell.system.vel[keep]
    cell.system.radii = cell.system.radii[keep]
    cell.free_idx = np.arange(keep.sum())
    acc2 = pr.ProfileAccumulator(n_bins=acc.n_bins)
    acc2.add(cell)
    prof = acc2.finalize(width=0.30, thickness=0.01, y_bottom=cell.y_bottom,
                         y_top=cell.y_top, radius=5e-3, density=2500.0,
                         v_wall=0.0)
    assert math.isnan(prof["ux"][4]) and math.isnan(prof["phi"][5])
    assert math.isnan(prof["p_kinetic"][4])
    assert sum(prof["count_mean"]) == pytest.approx(keep.sum())


# --------------------------------------------------------------------------- #
# end-to-end on the real 2-D device (small cell, short window)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strain", [0.5])
def test_simulate_attaches_a_consistent_profile(strain):
    import plane_shear_2d as ps

    case = ps.PlaneShear2DCase(
        I_target=0.05, n_particles=200, width=0.24, gap=0.12,
        settle_time_units=20.0, shear_settle_strain=0.3,
        strain_measure=strain, init_mode="random", seed=0, polydispersity=0.5,
        sample_every=20)
    case.validate()
    out = ps.simulate(case)
    prof = out["profile"]
    assert prof["samples"] > 0
    assert sum(prof["count_mean"]) == pytest.approx(out["n_free"], rel=1e-9)
    # the binned virial must reproduce the whole-cell virial (no fit involved).
    # 10 % is what the two bookkeeping differences allow: the bins use the *final*
    # gap while ``measure()`` divides by the instantaneous one, and ``mu_bulk`` is
    # a mean of per-sample ratios while the profile is a ratio of pooled sums.
    assert out["profile_consistency"]["p_bin_over_p_bulk"] == pytest.approx(1.0, rel=0.10)
    assert out["profile_consistency"]["mu_bin_over_mu_bulk"] == pytest.approx(1.0, rel=0.10)
    assert math.isfinite(out["profile_derived"]["gamma_dot_measured"])
    assert out["profile_derived"]["gamma_dot_ratio"] > 0.0