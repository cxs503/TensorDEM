"""Conduction solver vs three closed-form solutions plus a cross-discretisation.

The thermal field is the backbone of the whole chain, so it is graded against
independent mathematics rather than against itself:

* constant-flux semi-infinite body (Carslaw & Jaeger, eq. 2-46);
* uniformly heated slab held at its ends (sine series, derived in the module);
* instantaneous plane source (Gaussian impulse response);
* an independent nodal finite-element solve of the same problem;
* the discrete energy ledger (nothing may be created by the scheme).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_thermal_stress.conduction import (
    SlabHeat1D,
    exact_plane_source,
    exact_point_source_sphere,
    exact_semi_infinite_flux,
    exact_slab_uniform_source,
    fv_fem_disagreement,
    observed_order,
    slab_fem_nodes,
)
from tensordem.ice_thermal_stress.properties import ice_properties


@pytest.fixture(scope="module")
def props():
    return ice_properties(-10.0)


def test_grid_grading_refines_the_contact_face(props):
    uniform = SlabHeat1D(cells=100, thickness_m=0.01, properties=props)
    graded = SlabHeat1D(cells=100, thickness_m=0.01, properties=props, grading=50.0)
    assert uniform.dx.sum() == pytest.approx(0.01)
    assert graded.dx.sum() == pytest.approx(0.01)
    assert graded.dx[0] < uniform.dx[0] < graded.dx[-1]
    assert graded.dx[-1] / graded.dx[0] == pytest.approx(50.0, rel=1e-6)
    with pytest.raises(ValueError):
        SlabHeat1D(cells=10, thickness_m=0.01, properties=props, grading=0.5)


def test_constant_flux_profile_matches_the_semi_infinite_solution(props):
    fv = SlabHeat1D(cells=4000, thickness_m=0.01, properties=props)
    t_end, flux = 0.02, 1.0e4
    hist = fv.run(t_end, 2e-5, q_surface=flux, record_times=[t_end])
    exact = exact_semi_infinite_flux(fv.x, t_end, flux, props)
    peak = np.max(exact)
    assert np.max(np.abs(hist["profiles"][-1] - exact)) / peak < 0.01
    # the surface value must equal the repository's own C&J surface formula
    from tensordem.interface_thermal.conduction import semi_infinite_surface_temperature

    assert exact_semi_infinite_flux(0.0, t_end, flux, props) == pytest.approx(
        float(semi_infinite_surface_temperature(
            t_end, props.diffusivity_m2_s, props.k_w_m_k, flux, 0.0)), rel=1e-12)


def test_volumetric_source_matches_the_heated_slab_series(props):
    fv = SlabHeat1D(cells=4000, thickness_m=0.05, properties=props, grading=400.0)
    q_vol = 1.0e6
    hist = fv.run(2.0e-3, 1.0e-6, q_vol=np.full(fv.cells, q_vol), t_surface=0.0,
                  record_times=[2.0e-3])
    exact = exact_slab_uniform_source(fv.x, 2.0e-3, q_vol, 0.05, props)
    mask = fv.x < 0.02
    rel = np.max(np.abs(hist["profiles"][-1][mask] - exact[mask])) / np.max(exact)
    assert rel < 0.01
    # the series must relax to the steady parabola q L^2/(2k) xi (1-xi)
    steady = exact_slab_uniform_source(0.025, 1e9, q_vol, 0.05, props)
    assert steady == pytest.approx(q_vol * 0.05 ** 2 / (8.0 * props.k_w_m_k))


def test_plane_source_impulse_contains_the_injected_energy(props):
    x = np.linspace(-0.2, 0.2, 400001)
    q_areal = 1.0e3
    energy = props.volumetric_heat_capacity_j_m3_k * np.trapezoid(
        exact_plane_source(x, 0.5, q_areal, props), x)
    assert energy == pytest.approx(q_areal, rel=1e-3)


def test_point_source_kernel_is_normalised(props):
    # radial integral of the 3-D kernel: 4 pi int r^2 T dr = Q/(rho cp)
    r = np.linspace(0.0, 0.02, 400001)
    kernel = exact_point_source_sphere(r, 1.0e-3, 1.0, props)
    integral = 4.0 * np.pi * np.trapezoid(r ** 2 * kernel, r)
    assert integral * props.volumetric_heat_capacity_j_m3_k == pytest.approx(1.0, rel=1e-3)


def test_energy_ledger_closes_for_flux_and_source(props):
    fv = SlabHeat1D(cells=500, thickness_m=0.02, properties=props, grading=20.0)
    q_vol = fv.source_profile(total_power_w_m2=5000.0, depth_m=1.0e-3)
    fv.run(1.0e-3, 1.0e-6, q_surface=2000.0, q_vol=q_vol, record_times=[1.0e-3])
    balance = fv.energy_balance()
    assert balance["relative_error"] < 1e-10
    assert balance["source_heat_j_m2"] == pytest.approx(5.0, rel=1e-9)
    assert balance["surface_heat_j_m2"] == pytest.approx(2.0, rel=1e-9)


def test_finite_element_reference_agrees_with_the_finite_volume_field(props):
    fv = SlabHeat1D(cells=2000, thickness_m=0.02, properties=props, grading=50.0)
    q_vol = fv.source_profile(3000.0, 2.0e-3)
    fv.run(2.0e-3, 1e-6, q_vol=q_vol, record_times=[2.0e-3])
    n_el = 2000
    h = 0.02 / n_el
    q_el = np.where(np.arange(n_el) * h < 2.0e-3, 3000.0 / 2.0e-3, 0.0)
    fem = slab_fem_nodes(n_el, 0.02, props, t_end=2.0e-3, dt=1e-6, q_vol=q_el,
                         record_times=[2.0e-3])
    assert fv_fem_disagreement(fv, fem) / np.max(np.abs(fv.t)) < 0.02


def test_space_convergence_order_is_second(props):
    t_end, flux = 0.02, 1.0e4
    errors, cells = [], []
    for n in (250, 500, 1000):
        fv = SlabHeat1D(cells=n, thickness_m=0.01, properties=props)
        hist = fv.run(t_end, 2e-5, q_surface=flux, record_times=[t_end])
        exact = exact_semi_infinite_flux(fv.x, t_end, flux, props)
        errors.append(float(np.max(np.abs(hist["profiles"][-1] - exact))))
        cells.append(n)
    ratios = [cells[i + 1] / cells[i] for i in range(len(errors) - 1)]
    orders = [observed_order(errors[i:i + 2], ratios[i:i + 1])
              for i in range(len(errors) - 1)]
    assert min(orders) > 1.5
    assert max(orders) < 2.5


def test_consistent_mass_fails_at_a_flux_boundary_and_lumped_mass_does_not(props):
    """Documented numerical finding, kept as a regression guard.

    With a concentrated surface flux the consistent-mass FE solution does not
    converge (the surface error saturates and grows with refinement), while the
    lumped variant converges.  This is why the package defaults to lumped mass.
    """
    t_end, flux = 0.02, 1.0e4
    exact = exact_semi_infinite_flux(np.array([0.0]), t_end, flux, props)[0]
    errors = {}
    for mass in ("consistent", "lumped"):
        fem = slab_fem_nodes(2000, 0.01, props, t_end=t_end, dt=2e-5,
                             q_surface=flux, mass=mass, record_times=[t_end])
        errors[mass] = abs(fem["profiles"][-1][0] - exact) / exact
    assert errors["lumped"] < 0.01
    assert errors["consistent"] > 0.10          # non-convergent, not a typo
    assert errors["consistent"] > 5.0 * errors["lumped"]


def test_rejects_bad_inputs(props):
    fv = SlabHeat1D(cells=10, thickness_m=0.01, properties=props)
    with pytest.raises(ValueError):
        fv.step(0.0)
    with pytest.raises(ValueError):
        fv.step(1e-3, q_vol=np.zeros(3))
    with pytest.raises(ValueError):
        SlabHeat1D(cells=1, thickness_m=0.01, properties=props)
    with pytest.raises(ValueError):
        slab_fem_nodes(10, 0.01, props, t_end=1e-3, dt=1e-5, q_vol=np.zeros(3))