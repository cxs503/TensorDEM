"""Tests for the independent DEM-FEM coupling anchors (``tensordem.dem_fem_gates``).

The point of these tests is not to re-run the example end to end (the driver in
``examples/dem_fem_oneway/independent_gates.py`` does that and writes
``artifacts/dem_fem_coupling/independent_gates.json``), but to lock the anchors
themselves: a closed form that is wrong would "verify" the wrong thing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.dem_fem_gates import (
    INDEPENDENCE,
    beam_mode_shape,
    beam_natural_frequencies,
    evaluate_gates,
    hertz_impact,
    hookean_impact,
    modal_response,
    modal_static_deflection,
    summarise,
)

# simply supported steel beam used by examples/dem_fem_oneway
EI = 2.1e11 * (0.05 * 0.08 ** 3 / 12.0)
RHO_A = 7850.0 * 0.05 * 0.08
L = 0.3


# ------------------------------------------------------------- closed forms --
def test_hookean_impact_closed_form_is_self_consistent():
    """Half-sine contact: t_c = π/ω, F_max = v√(mk), δ = v/ω, J = (1+e) m v."""
    m, k, v = 0.06001750965, 2.0e4, 1.0
    res = hookean_impact(m, k, v, restitution=1.0)
    omega = math.sqrt(k / m)
    assert res["contact_time_s"] == pytest.approx(math.pi / omega, rel=1e-14)
    assert res["peak_force_N"] == pytest.approx(v * math.sqrt(m * k), rel=1e-14)
    assert res["max_overlap_m"] == pytest.approx(v / omega, rel=1e-14)
    assert res["impulse_Ns"] == pytest.approx(2.0 * m * v, rel=1e-14)
    # energy identity of an elastic Hookean impact: 0.5 k δ² = 0.5 m v²
    assert 0.5 * k * res["max_overlap_m"] ** 2 == pytest.approx(0.5 * m * v ** 2, rel=1e-14)


def test_hookean_impact_reduced_mass_and_restitution():
    a = hookean_impact(1.0, 100.0, 2.0, restitution=0.5, target_mass=1.0)
    assert a["m_eff"] == pytest.approx(0.5)
    assert a["impulse_Ns"] == pytest.approx(1.5 * 0.5 * 2.0)
    with pytest.raises(ValueError):
        hookean_impact(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        hookean_impact(1.0, 1.0, 1.0, restitution=1.5)


def test_hertz_impact_matches_published_coefficients():
    """Hertz: δ=(5mv²/4k)^{2/5}, t_c=2.943275 (5m/4k)^{2/5} v^{-1/5}, J=2mv."""
    m, k, v = 0.06, 1.0e9, 0.5
    res = hertz_impact(m, k, v)
    delta = (5.0 * m * v ** 2 / (4.0 * k)) ** 0.4
    assert res["max_overlap_m"] == pytest.approx(delta, rel=1e-13)
    assert res["peak_force_N"] == pytest.approx(k * delta ** 1.5, rel=1e-13)
    assert res["contact_time_s"] == pytest.approx(
        2.943275 * (5.0 * m / (4.0 * k)) ** 0.4 * v ** -0.2, rel=1e-13)
    assert res["impulse_Ns"] == pytest.approx(2.0 * m * v, rel=1e-13)


# -------------------------------------------------------------------- beam --
def test_beam_natural_frequencies_match_closed_form():
    omega = beam_natural_frequencies(EI, RHO_A, L, 5)
    base = (math.pi / L) ** 2 * math.sqrt(EI / RHO_A)
    expected = base * np.arange(1, 6) ** 2
    assert np.allclose(omega, expected, rtol=1e-14)
    # T1 of the verified case must reproduce the shipped claim t_c/T1 ≈ 11
    t1 = 2.0 * math.pi / omega[0]
    tc = hookean_impact(0.06001750965, 2.0e4, 1.0)["contact_time_s"]
    assert 10.0 < tc / t1 < 12.0


def test_mode_shape_is_orthogonal_and_normalised():
    x = np.linspace(0.0, L, 20001)
    phi1 = np.array([beam_mode_shape(1, xi, L) for xi in x])
    phi2 = np.array([beam_mode_shape(2, xi, L) for xi in x])
    assert np.trapezoid(phi1 * phi1, x) == pytest.approx(L / 2.0, rel=1e-6)
    assert np.trapezoid(phi1 * phi2, x) == pytest.approx(0.0, abs=1e-6)


def test_modal_response_reproduces_static_deflection():
    """A slow ramp to a plateau must converge to P L³/48EI (mode truncation only)."""
    load = 34.646
    t = np.linspace(0.0, 0.2, 20001)
    f = np.where(t < 0.1, load * t / 0.1, load)
    res = modal_response(f, t, ei=EI, mass_per_length=RHO_A, length=L,
                         x_load=L / 2, x_probe=L / 2, n_modes=9)
    expected = modal_static_deflection(load, L, EI)
    assert res["deflection"][-1] == pytest.approx(expected, rel=1e-3)


def test_modal_response_peak_amplification_beats_static():
    """Half-sine pulse with t_c/T1 ≈ 1 must overshoot the static deflection."""
    F, tc = 34.646, 0.5e-3
    t = np.linspace(0.0, 4.0 * tc, 4001)
    f = F * np.sin(np.pi * np.clip(t / tc, 0.0, 1.0))
    res = modal_response(f, t, ei=EI, mass_per_length=RHO_A, length=L,
                         x_load=L / 2, x_probe=L / 2, n_modes=9)
    static = modal_static_deflection(F, L, EI)
    assert res["peak_deflection"] > static
    assert res["peak_deflection"] < 2.2 * static


def test_modal_response_rejects_bad_input():
    with pytest.raises(ValueError):
        modal_response([0.0, 0.0], [0.0, 0.0], ei=EI, mass_per_length=RHO_A,
                       length=L, x_load=L / 2, x_probe=L / 2)
    with pytest.raises(ValueError):
        modal_response([0.0, 1.0], [0.0, -1.0], ei=EI, mass_per_length=RHO_A,
                       length=L, x_load=L / 2, x_probe=L / 2)
    with pytest.raises(NotImplementedError):
        beam_natural_frequencies(EI, RHO_A, L, 3, bc="clamped")


# ------------------------------------------------------------------- gates --
def test_gate_arithmetic_and_negative_control():
    good = evaluate_gates([
        ("peak force", "N", 34.646, 34.646 * (1.0 + 1e-5), 0.03, "analytic"),
        ("deflection", "m", 4.5e-8, 4.5e-8 * 1.07, 0.05, "independent_implementation"),
    ])
    assert good[0].passed and not good[1].passed
    assert good[1].rel_error == pytest.approx(0.07, rel=1e-12)
    summary = summarise(good)
    assert summary["n_external"] == 2 and summary["n_self_comparison"] == 0
    assert summary["external_pass"] == 1 and not summary["all_external_pass"]
    assert summary["worst_external_rel_error"] == pytest.approx(0.07, rel=1e-12)


def test_self_comparison_is_counted_but_not_evidence():
    """A tautological gate may pass, but it must never count as an external anchor."""
    gates = evaluate_gates([
        ("same-array impulse", "N s", 0.1200342, 0.1200342, 1e-10, "self_comparison"),
    ])
    summary = summarise(gates)
    assert summary["n_external"] == 0
    assert summary["n_self_comparison"] == 1
    assert summary["all_external_pass"] is True   # nothing external to fail
    assert summary["external_pass"] == 0          # ... and nothing external passing


def test_unknown_independence_class_is_rejected():
    with pytest.raises(ValueError):
        evaluate_gates([("g", "J", 1.0, 1.0, 0.1, "looks_fine_to_me")])
    assert set(INDEPENDENCE) == {
        "analytic", "independent_implementation", "cross_side", "self_comparison"}