"""Acceptance tests for the 2D granular column collapse verification face.

Criteria covered (campaign node p40w2-n06, granular half):

* (c) runout scaling law: the fit helper recovers a prescribed exponent and the
  DEM runs give a runout that grows with the aspect ratio;
* (d) inertial number / packing fraction sampling and the mu(I) fit helpers;
* (e) energy ledger: the sliding-disk dissipation matches the analytic value and
  the collapse potential energy is closed by the integrated dissipation to < 5 %.

The DEM runs in this file are deliberately small (few grains, short time) so the
module stays inside a normal test-suite budget; the production scan with the
published parameters is ``benchmarks/column_collapse_runout.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular2d import DiskConfig, DiskSystem, box_walls
from tensordem.granular2d.column_collapse import (
    LUBE_TWO_THIRDS,
    ColumnCollapseError,
    ColumnSpec,
    build_column,
    fit_mu_inertial,
    fit_phi_inertial,
    fit_runout_scaling,
    flow_samples,
    run_collapse,
)

SMALL = dict(width=0.060, radius=0.006, kn=1.0e4, thickness=0.02,
             mu=0.5, mu_rolling=0.5, max_time=0.3, snapshot_interval=0.05)


def test_column_geometry_is_measured_from_the_packing():
    spec = ColumnSpec(aspect_ratio=2.0, **SMALL)
    system, geo = build_column(spec)
    assert system.n >= 18
    assert geo["length_initial"] == pytest.approx(spec.width, rel=0.05)
    assert geo["height_initial"] == pytest.approx(spec.height, rel=0.05)
    assert geo["aspect_ratio_measured"] == pytest.approx(2.0, rel=0.06)
    # the floor must be long enough that no grain can run off its end
    assert system.walls[:, :, 0].max() > spec.width + 4.0 * spec.height
    # no initial overlaps
    contacts = system.contact_forces()
    overlap = contacts["overlap"]
    disk = np.asarray(contacts["kind"]) == 0
    assert float(np.max(overlap[disk], initial=0.0)) < 0.02 * spec.radius


def test_sliding_disk_dissipation_matches_analytic_friction_work():
    """Criterion (e) anchor: the ledger must reproduce the analytic friction work.

    A disk launched at ``v0`` on a frictional floor spins up and leaves rolling
    without slipping at ``2 v0 / 3``; the friction work is ``m v0^2 / 6`` while
    the slip distance is finite.  This is an exact check of the ledger integrator.
    """
    radius = 0.01
    config = DiskConfig(kn=3.0e4, mu=0.3, mu_rolling=0.0, zeta=0.0,
                        gravity=(0.0, -9.81), skin=0.004)
    system = DiskSystem([[0.1, radius - 1.0e-6]], [radius], config,
                        walls=box_walls(1.0, 0.2, right=False, top=False))
    system.vel[0, 0] = 0.5
    system.enable_energy_ledger()
    start = system.energies()
    system.step(3000)
    end = system.energies()
    mass = system.mass[0]
    analytic = mass * 0.5 ** 2 / 6.0
    assert system.dissipated_work() == pytest.approx(analytic, rel=0.02)
    assert system.vel[0, 0] == pytest.approx(2.0 * 0.5 / 3.0, rel=0.02)
    residual = ((start["kinetic"] + start["potential"] + start["elastic_normal"])
                - (end["kinetic"] + end["potential"] + end["elastic_normal"])
                - system.dissipated_work())
    assert abs(residual) < 0.01 * analytic


def test_collapse_energy_ledger_closes_within_five_percent():
    """Criterion (e): E_p0 is accounted for by the integrated dissipation."""
    result = run_collapse(ColumnSpec(aspect_ratio=1.0, **SMALL), sample_flow=False)
    assert result.n_particles >= 18
    assert result.energy_potential_initial > 0.0
    assert result.energy_dissipated > 0.0
    assert abs(result.energy_residual_relative) < 0.05, result.energy_residual_relative
    assert 0.0 < result.energy_dissipated_relative <= 1.2


def test_runout_grows_with_aspect_ratio():
    small = run_collapse(ColumnSpec(aspect_ratio=1.0, **SMALL), sample_flow=False)
    tall = run_collapse(ColumnSpec(aspect_ratio=2.5, **SMALL), sample_flow=False)
    assert tall.runout_dimensionless > small.runout_dimensionless
    assert tall.height_initial > small.height_initial


def test_fit_runout_scaling_recovers_the_lube_exponent():
    class _Proxy:
        def __init__(self, aspect, runout):
            self.aspect_ratio = aspect
            self.runout_dimensionless = runout

    aspects = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    prefactor = 1.9
    proxies = [_Proxy(a, prefactor * a ** LUBE_TWO_THIRDS) for a in aspects]
    fit = fit_runout_scaling(proxies)
    assert fit["exponent"] == pytest.approx(LUBE_TWO_THIRDS, rel=1e-6)
    assert fit["prefactor"] == pytest.approx(prefactor, rel=1e-6)
    assert fit["r_squared"] > 0.999
    with pytest.raises(ColumnCollapseError):
        fit_runout_scaling(proxies[:2])


def test_mu_i_and_phi_i_fits_recover_prescribed_laws():
    mu_s, mu_2, i_0 = 0.35, 0.70, 0.30
    points = []
    for inertial in (0.01, 0.03, 0.1, 0.3, 0.6):
        mu = mu_s + (mu_2 - mu_s) / (i_0 / inertial + 1.0)
        points.append({"inertial_number": inertial, "mu": mu,
                       "packing_fraction": 0.60 - 0.05 * inertial ** 0.4})
    mu_fit = fit_mu_inertial(points)
    assert mu_fit["I_0"] == pytest.approx(i_0, rel=0.05)
    assert mu_fit["mu_s"] == pytest.approx(mu_s, abs=0.02)
    assert mu_fit["r_squared"] > 0.999
    phi_fit = fit_phi_inertial(points)
    assert phi_fit["spearman_logI_phi"] < -0.9
    assert phi_fit["alpha"] > 0.0


def test_flow_samples_are_physically_bounded():
    """During the flow the sampled I, mu and phi must stay in admissible ranges."""
    result = run_collapse(ColumnSpec(aspect_ratio=2.0, **SMALL), sample_flow=True)
    sample = flow_samples(result)
    if len(sample["inertial_number"]) == 0:
        pytest.skip("no flow samples were captured for this small run")
    assert np.all(sample["inertial_number"] > 0.0)
    assert np.all(sample["mu"] >= 0.0)
    assert np.all(sample["packing_fraction"] > 0.05)
    assert np.all(sample["packing_fraction"] < 0.92)


def test_spec_validation_and_shard_helper():
    with pytest.raises(ColumnCollapseError):
        ColumnSpec(aspect_ratio=-1.0)
    with pytest.raises(ColumnCollapseError):
        ColumnSpec(width=0.001, radius=0.006)
    with pytest.raises(ColumnCollapseError):
        ColumnSpec(floor_extension=1.0)
    spec = ColumnSpec(aspect_ratio=3.0)
    assert spec.height == pytest.approx(3.0 * spec.width)
    assert spec.particle_mass > 0.0