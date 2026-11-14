"""W28 · unit tests for the dual-convergence machinery.

The tests pin the two things the study depends on: (a) the keel-to-beam coupling
solves the *right* problem (verified against the classical Hetenyi infinite-beam
solution and against global equilibrium), and (b) the convergence analysis
recovers a known order from synthetic data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import keel_foundation_dem as kd
from tensordem import keel_grounding_coupled as kc

DEM_SETTLE = [1.3333e-3, 2.6667e-3, 4.0e-3, 5.3333e-3, 6.6667e-3, 8.0e-3]
DEM_PRESS = [4.0817e6, 6.9547e6, 9.2988e6, 11.3277e6, 13.1406e6, 14.7937e6]


@pytest.fixture(scope="module")
def law():
    return kc.KeelPressureLaw(DEM_SETTLE, DEM_PRESS)


def test_pressure_law_monotone_and_origin(law):
    assert law.pressure([0.0])[0] == 0.0
    w = np.linspace(0.0, 8e-3, 50)
    p = law.pressure(w)
    assert np.all(np.diff(p) > 0.0)
    assert law.pressure([-5e-3])[0] == 0.0            # no tension
    # beyond the measured range the last secant is continued
    w_hi = 2.0 * law.w_max
    assert law.pressure([w_hi])[0] == pytest.approx(
        law.p_max + law.secant_max * (w_hi - law.w_max))


def test_keel_pressure_law_rejects_non_monotone():
    with pytest.raises(ValueError):
        kc.KeelPressureLaw([1e-3, 2e-3], [5e6, 4e6])


def test_bond_law_length_scaling():
    """Legacy kn is particle-size independent; the spacing-consistent kn scales with a."""
    a, b = 5000, 20000
    s_a = kd.build_keel_sample(a, bond_reference_length=kd.REFERENCE_LENGTH)
    s_b = kd.build_keel_sample(b, bond_reference_length=kd.REFERENCE_LENGTH)
    l_a = kd.build_keel_sample(a, bond_reference_length=None)
    l_b = kd.build_keel_sample(b, bond_reference_length=None)
    # spacing-consistent: kn / a invariant (k ~ E / Lambda)
    ratio_k = (np.mean(s_b.kn) / s_b.spacing) / (np.mean(s_a.kn) / s_a.spacing)
    assert ratio_k == pytest.approx(1.0, rel=0.05)
    # legacy: mean kn independent of the particle size, so kn / a grows like 1 / a
    assert np.mean(l_b.kn) == pytest.approx(np.mean(l_a.kn), rel=0.05)
    growth = (np.mean(l_b.kn) / l_b.spacing) / (np.mean(l_a.kn) / l_a.spacing)
    assert growth == pytest.approx(s_a.spacing / s_b.spacing, rel=0.05)


def test_beam_solver_matches_hetenyi():
    """Linear Winkler layer, patch load: match the classical infinite-beam solution."""
    ei, k, q, a, length = 1.545e9, 1.6e9, 1.0e5, 1.0, 40.0
    out = kc.beam_on_unilateral_keel(length=length, ei=ei, n_elements=160,
                                     platen_centre=0.5 * length, k_uniform=k,
                                     load_intensity=q, load_half_width=a,
                                     samples_per_element=8)
    assert out["converged"]
    xs = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    ref = kc.winkler_infinite_beam_patch(ei=ei, k=k, load_intensity=q,
                                        load_half_width=a, x=xs, n_gauss=300)
    fem = np.interp(xs, np.array(out["x_m"]) - 0.5 * length, np.array(out["w_m"]))
    assert np.max(np.abs(fem - ref)) / np.max(np.abs(ref)) < 0.01


def test_platen_equilibrium_and_no_tension(law):
    ei = 3.66e9
    out = kc.beam_on_unilateral_keel(length=16.0, ei=ei, n_elements=64,
                                     platen_centre=8.0, platen_width=2.0, w0=8e-3,
                                     law=law, ramp_width_m=8e-4, samples_per_element=16)
    assert out["converged"]
    # global equilibrium: keel reaction equals the platen reaction
    assert out["equilibrium_rel_error"] < 1e-3
    # no tension: the pressure vanishes wherever the beam has lifted off
    prof = out["contact_pressure_profile"]
    w = np.interp(np.array(prof["x_m"]), np.array(out["x_m"]), np.array(out["w_m"]))
    assert np.all(np.array(prof["p_Pa"])[w < -1e-9] == 0.0)
    # the platen settlement is prescribed exactly
    assert np.max(np.array(out["w_m"])) == pytest.approx(8e-3, rel=1e-9)
    # contact area cannot exceed the beam length
    assert 0.0 < out["contact_area_m2_per_m"] <= 16.0


def test_mesh_convergence_of_the_coupled_solve(law):
    """Refining the beam mesh must move the resistance monotonically toward a limit."""
    vals = []
    for ne in (32, 64, 128):
        out = kc.beam_on_unilateral_keel(length=16.0, ei=3.66e9, n_elements=ne,
                                         platen_centre=8.0, platen_width=2.0, w0=8e-3,
                                         law=law, ramp_width_m=8e-4, samples_per_element=16)
        vals.append(out["resistance_N_per_m"])
    assert vals[0] < vals[1] < vals[2]
    assert abs(vals[2] - vals[1]) < abs(vals[1] - vals[0])


def test_graded_nodes_refine_at_the_platen_edges():
    xn = kc.graded_nodes(16.0, 64, centres=[7.0, 9.0], sigma=0.4, amplitude=16.0)
    assert xn[0] == pytest.approx(0.0) and xn[-1] == pytest.approx(16.0)
    assert np.all(np.diff(xn) > 0.0)
    h = np.diff(xn)
    near = np.min(h[np.abs(0.5 * (xn[1:] + xn[:-1]) - 7.0) < 0.4])
    far = np.max(h)
    assert near < 0.2 * far


def test_observed_order_recovers_a_known_power_law():
    f_inf, c, p = 7.5e7, 3.0e6, 2.0
    h = np.array([1.0, 0.5, 0.25])
    vals = f_inf + c * h ** p                      # fine -> coarse with r = 2
    info = kc.observed_order(vals[2], vals[1], vals[0], r=2.0)
    assert info["observed_order_p"] == pytest.approx(p, rel=1e-6)
    lim = kc.richardson_limit(vals[2], vals[1], vals[0], r=2.0)
    assert lim["richardson_limit"] == pytest.approx(f_inf, rel=1e-6)
    assert lim["uncertainty_band_rel"] is not None


def test_divergent_triad_has_no_order():
    info = kc.observed_order(30.0, 20.0, 10.0, r=2.0)     # wrong-direction sequence
    assert info["observed_order_p"] is None


def test_spearman_monotone_and_noise():
    w = np.linspace(1e-3, 8e-3, 6)
    assert kc.spearman(w, 2.0 * w ** 1.3)["rho"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert kc.spearman(w, rng.standard_normal(6))["rho"] < 0.9
