"""Analytic verification of the Euler-Bernoulli beam FEM used by DEM-FEM coupling.

Reference solutions (Timoshenko & Gere, *Mechanics of Materials*; Clough & Penzien,
*Dynamics of Structures*):

* simply supported, central point load ``P``:   w(L/2) = P L^3 / (48 EI)
* clamped-clamped, central point load ``P``:    w(L/2) = P L^3 / (192 EI)
* cantilever, tip point load ``P``:             w(L)   = P L^3 / (3 EI)
* simply supported, uniform load ``q``:         w(L/2) = 5 q L^4 / (384 EI)
* clamped-clamped, uniform load ``q``:          w(L/2) =   q L^4 / (384 EI)
* simply supported, uniform load ``q``, slope:  dw/dx(0) = q L^3 / (24 EI)
* cantilever, tip point load ``P``, slope:      dw/dx(L) = P L^2 / (2 EI)
* simply supported eigenfrequencies:            omega_n = (n pi)^2 sqrt(EI / (rho A L^4))
* cantilever eigenfrequency (n=1):              omega_1 = 3.5160 sqrt(EI / (rho A L^4))

Nodal degrees of freedom are (w, dw/dx) with w positive downward, so a
downward positive load produces a positive deflection and a positive
left-end slope for the simply supported case.

The two engineering defects found during this verification are pinned by
dedicated regression tests:
``test_eigen_reduction_is_symmetric_positive_definite`` (a spurious transpose
made the Rayleigh-damping eigenvalue problem non-symmetric with negative
eigenvalues, so every damped run silently produced NaN) and
``test_odd_element_count_midspan_interpolation`` (mid-span output used the
node index ``n_node // 2``, which is not at L/2 for odd element counts).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.fem_euler_bernoulli import (
    BeamConfig,
    EulerBernoulliBeam,
    beam_center_deflection_analytic,
    beam_tip_deflection_analytic,
)

P_LOAD = 1000.0
Q_UDL = 500.0


def _midspan_node_load(beam: EulerBernoulliBeam, load: float) -> torch.Tensor:
    """Point load exactly at the node nearest mid-span (exact for even meshes)."""
    f = torch.zeros(beam.n_dof, dtype=torch.float64)
    f[2 * (beam.n_node // 2)] = load
    return f


def _beam(n_elements: int, bc: str = "simply_supported") -> EulerBernoulliBeam:
    return EulerBernoulliBeam(BeamConfig(n_elements=n_elements, bc=bc))


def _uniform_load_vector(beam: EulerBernoulliBeam, q: float) -> torch.Tensor:
    """Variationally consistent nodal loads for a uniform transverse load q."""
    f = torch.zeros(beam.n_dof, dtype=torch.float64)
    h = beam.h_elem
    for e in range(beam.config.n_elements):
        f[2 * e] += q * h / 2.0
        f[2 * e + 1] += q * h * h / 12.0
        f[2 * e + 2] += q * h / 2.0
        f[2 * e + 3] -= q * h * h / 12.0
    return f


def _analytic_w1(cfg: BeamConfig) -> float:
    return math.pi ** 2 * math.sqrt(cfg.young_modulus * cfg.inertia
                                   / (cfg.mass_per_length * cfg.length ** 4))


@pytest.mark.parametrize("n_elements", [2, 10, 20])
def test_simply_supported_central_point_load_is_nodally_exact(n_elements):
    """Cubic Hermite elements reproduce the exact nodal deflection for point loads."""
    cfg = BeamConfig(n_elements=n_elements)
    beam = EulerBernoulliBeam(cfg)
    result = beam.static_solve(_midspan_node_load(beam, P_LOAD))
    exact = beam_center_deflection_analytic(
        P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia, bc="simply_supported")
    assert result["center_deflection"] == pytest.approx(exact, rel=1e-10)

    # slope at the left support: dw/dx(0) = P L^2 / (16 EI), positive downward convention
    exact_slope = P_LOAD * cfg.length ** 2 / (16.0 * cfg.young_modulus * cfg.inertia)
    assert result["rotation"][0] == pytest.approx(exact_slope, rel=1e-8)
    # symmetry of the deflection field about mid-span
    left = beam.deflection_at(0.5 * cfg.length - 0.1 * cfg.length)
    right = beam.deflection_at(0.5 * cfg.length + 0.1 * cfg.length)
    assert left == pytest.approx(right, rel=1e-12)
    # reactions: R = P/2 at each support, no moment at a simple support
    assert result["reaction"][0] == pytest.approx(-P_LOAD / 2.0, rel=1e-9)
    assert result["reaction"][1] == pytest.approx(-P_LOAD / 2.0, rel=1e-9)


@pytest.mark.parametrize("n_elements", [2, 10, 20])
def test_clamped_clamped_central_point_load_is_nodally_exact(n_elements):
    cfg = BeamConfig(n_elements=n_elements, bc="clamped")
    beam = EulerBernoulliBeam(cfg)
    result = beam.static_solve(_midspan_node_load(beam, P_LOAD))
    exact = beam_center_deflection_analytic(
        P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia, bc="clamped")
    assert result["center_deflection"] == pytest.approx(exact, rel=1e-10)
    # zero slope at both clamped ends
    assert result["rotation"][0] == pytest.approx(0.0, abs=1e-14)
    assert result["rotation"][-1] == pytest.approx(0.0, abs=1e-14)
    assert result["reaction"][0] == pytest.approx(-P_LOAD / 2.0, rel=1e-9)
    assert result["reaction"][2] == pytest.approx(-P_LOAD / 2.0, rel=1e-9)


@pytest.mark.parametrize("n_elements", [2, 10, 20])
def test_cantilever_tip_point_load_is_nodally_exact(n_elements):
    cfg = BeamConfig(n_elements=n_elements, bc="cantilever")
    beam = EulerBernoulliBeam(cfg)
    load = torch.zeros(beam.n_dof, dtype=torch.float64)
    load[-2] = P_LOAD
    result = beam.static_solve(load)
    exact = beam_tip_deflection_analytic(
        P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia)
    assert result["deflection"][-1] == pytest.approx(exact, rel=1e-10)
    # tip slope dw/dx(L) = P L^2 / (2 EI)
    exact_slope = P_LOAD * cfg.length ** 2 / (2.0 * cfg.young_modulus * cfg.inertia)
    assert result["rotation"][-1] == pytest.approx(exact_slope, rel=1e-10)
    # reactions at the clamped root: vertical -P and moment -P L (equilibrium)
    assert result["reaction"][0] == pytest.approx(-P_LOAD, rel=1e-10)
    assert result["reaction"][1] == pytest.approx(-P_LOAD * cfg.length, rel=1e-10)


def test_cantilever_config_accepted_and_analytic_helpers_agree():
    # regression: BeamConfig used to reject "cantilever" although the analytic
    # helper advertised the cantilever formula, i.e. an unreachable API path.
    cfg = BeamConfig(n_elements=8, bc="cantilever")
    assert cfg.bc == "cantilever"
    with pytest.raises(ValueError):
        beam_tip_deflection_analytic(P_LOAD, 1.0, 2.1e11, 1e-8, bc="simply_supported")
    tip = beam_tip_deflection_analytic(P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia)
    # the legacy centre helper keeps the cantilever formula for backward compatibility
    legacy = beam_center_deflection_analytic(
        P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia, bc="cantilever")
    assert legacy == pytest.approx(tip, rel=1e-15)


@pytest.mark.parametrize("bc,coefficient", [("simply_supported", 5.0 / 384.0),
                                            ("clamped", 1.0 / 384.0)])
def test_uniform_load_midspan_deflection_is_nodally_exact(bc, coefficient):
    """Consistent nodal loads give exact *nodal* deflections for a uniform load.

    This is a property of the cubic Hermite element (the exact quartic solution
    is sampled exactly at the nodes); it does not make the intra-element field
    exact, which is why ``deflection_at`` is verified separately below.
    """
    cfg = BeamConfig(n_elements=4, bc=bc)
    beam = EulerBernoulliBeam(cfg)
    result = beam.static_solve(_uniform_load_vector(beam, Q_UDL))
    exact = coefficient * Q_UDL * cfg.length ** 4 / (cfg.young_modulus * cfg.inertia)
    assert result["center_deflection"] == pytest.approx(exact, rel=1e-12)


def test_simply_supported_uniform_load_rotation():
    cfg = BeamConfig(n_elements=10)
    beam = EulerBernoulliBeam(cfg)
    result = beam.static_solve(_uniform_load_vector(beam, Q_UDL))
    exact = Q_UDL * cfg.length ** 3 / (24.0 * cfg.young_modulus * cfg.inertia)
    assert result["rotation"][0] == pytest.approx(exact, rel=1e-12)
    assert result["rotation"][-1] == pytest.approx(-exact, rel=1e-12)


def test_consistent_point_load_at_node_reduces_to_nodal_force():
    beam = _beam(7)  # odd count: element boundaries at multiples of L/7
    x_node = 3 * beam.h_elem
    f = beam.consistent_point_load(x_node, P_LOAD)
    assert float(f.abs().sum()) == pytest.approx(P_LOAD, rel=1e-12)
    expected = torch.zeros(beam.n_dof, dtype=torch.float64)
    expected[2 * 3] = P_LOAD
    assert torch.allclose(f, expected, atol=1e-9)


def test_odd_element_count_midspan_interpolation():
    """Mid-span output must interpolate, not pick node ``n_node // 2``.

    Regression: for ``n_elements=9`` the old code returned the deflection at
    x=0.556 L (-1.8 % vs the analytic mid-span value) because the reported node
    was not at L/2.
    """
    cfg = BeamConfig(n_elements=9)
    beam = EulerBernoulliBeam(cfg)
    assert beam.x_nodes[beam.n_node // 2].item() != pytest.approx(cfg.length / 2)
    result = beam.static_solve(beam.consistent_point_load(0.5 * cfg.length, P_LOAD))
    exact = beam_center_deflection_analytic(
        P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia)
    assert result["center_deflection"] == pytest.approx(exact, rel=5e-4)
    # the old off-mid-span node value is several percent away from the truth
    old_node = beam.x_nodes[beam.n_node // 2].item()
    assert beam._interpolate_deflection(
        torch.tensor(_as_dofs(result), dtype=torch.float64), old_node) == pytest.approx(
            result["deflection"][beam.n_node // 2], rel=1e-12)


def _as_dofs(result: dict) -> list[float]:
    dofs: list[float] = []
    for w, theta in zip(result["deflection"], result["rotation"]):
        dofs.extend([w, theta])
    return dofs


def test_odd_mesh_midspan_error_decreases_with_refinement():
    """Consistent mid-element point load: discretization error must shrink with h."""
    errors = []
    for n_elements in (9, 19, 39):
        cfg = BeamConfig(n_elements=n_elements)
        beam = EulerBernoulliBeam(cfg)
        result = beam.static_solve(beam.consistent_point_load(0.5 * cfg.length, P_LOAD))
        exact = beam_center_deflection_analytic(
            P_LOAD, cfg.length, cfg.young_modulus, cfg.inertia)
        errors.append(abs(result["center_deflection"] - exact) / exact)
    assert errors[1] < errors[0] / 5.0
    assert errors[2] < errors[1] / 5.0
    assert errors[-1] < 1e-5


def test_eigen_reduction_is_symmetric_positive_definite():
    """Regression: the generalized eigenproblem reduction must be L^-1 K L^-T."""
    beam = _beam(10)
    A = beam._A_eig
    scale = float(A.abs().max())
    assert float((A - A.T).abs().max()) / scale < 1e-10
    assert float(beam._eigvals[0]) > 0.0
    assert torch.all(beam._omegas[:5] > 0.0)


def test_modal_frequencies_converge_to_analytic():
    """Simply supported eigenfrequencies omega_n = (n pi)^2 sqrt(EI/(rho A L^4)).

    Consistent-mass cubic Hermite elements converge from above at O(h^4); the
    measured error ratio for every refinement step below is >= 10 (observed
    ~15-16), which is the quantitative statement this test pins down.
    """
    errors = {1: [], 2: [], 3: []}
    for n_elements in (10, 20, 40):
        cfg = BeamConfig(n_elements=n_elements)
        beam = EulerBernoulliBeam(cfg)
        base = _analytic_w1(cfg)
        for mode in (1, 2, 3):
            numeric = float(beam._omegas[mode - 1])
            exact = base * mode ** 2
            errors[mode].append(abs(numeric - exact) / exact)
        # consistent mass gives an upper bound on the discrete frequencies
        assert float(beam._omegas[0]) >= base * (1.0 - 1e-12)
    for mode, series in errors.items():
        assert series[0] < 1e-2, f"mode {mode}: coarse-mesh error {series[0]:.3e}"
        assert series[1] < series[0] / 10.0, f"mode {mode}: {series}"
        assert series[2] < series[1] / 10.0, f"mode {mode}: {series}"
        assert series[2] < 1e-4, f"mode {mode}: fine-mesh error {series[2]:.3e}"


def test_cantilever_fundamental_frequency_matches_textbook():
    cfg = BeamConfig(n_elements=40, bc="cantilever")
    beam = EulerBernoulliBeam(cfg)
    exact = 3.516015 * math.sqrt(cfg.young_modulus * cfg.inertia
                                 / (cfg.mass_per_length * cfg.length ** 4))
    rel = abs(float(beam._omegas[0]) - exact) / exact
    assert rel < 1e-4


def test_static_solve_is_side_effect_free_by_default_and_can_commit():
    cfg = BeamConfig(n_elements=10)
    beam = EulerBernoulliBeam(cfg)
    load = beam.consistent_point_load(0.5 * cfg.length, P_LOAD)
    result = beam.static_solve(load)
    assert beam.disp.abs().max() == 0.0  # documented: pure probe
    committed = beam.static_solve(load, commit=True)
    assert committed["center_deflection"] == pytest.approx(result["center_deflection"],
                                                           rel=1e-15)
    assert beam.center_deflection() == pytest.approx(result["center_deflection"], rel=1e-14)
    assert beam.vel.abs().max() == 0.0 and beam.acc.abs().max() == 0.0


def _free_vibration_history(beam: EulerBernoulliBeam, amplitude_load: float,
                            dt: float, n_steps: int) -> list[tuple[float, float]]:
    beam.static_solve(beam.consistent_point_load(0.5 * beam.config.length, amplitude_load),
                      commit=True)
    zero = torch.zeros(beam.n_dof, dtype=torch.float64)
    history = []
    for _ in range(n_steps):
        beam.apply_load(zero, dt)
        history.append((beam.time, beam.center_deflection()))
    return history


def test_undamped_free_vibration_period_and_energy():
    cfg = BeamConfig(n_elements=20)
    beam = EulerBernoulliBeam(cfg)
    w1 = _analytic_w1(cfg)
    period = 2.0 * math.pi / w1
    dt = period / 500.0
    beam.static_solve(beam.consistent_point_load(0.5 * cfg.length, P_LOAD), commit=True)
    energy0 = beam.total_energy()
    zero = torch.zeros(beam.n_dof, dtype=torch.float64)
    history = []
    for _ in range(int(2.5 * period / dt)):
        beam.apply_load(zero, dt)
        history.append((beam.time, beam.center_deflection()))

    crossings = []
    for (t0, d0), (t1, d1) in zip(history, history[1:]):
        if d0 > 0.0 >= d1 or d0 < 0.0 <= d1:
            crossings.append(t0 + (t1 - t0) * abs(d0) / (abs(d0) + abs(d1)))
    periods = [2.0 * (crossings[i + 1] - crossings[i]) for i in range(len(crossings) - 1)]
    measured = sum(periods) / len(periods)
    assert measured == pytest.approx(period, rel=2e-3)
    assert abs(beam.total_energy() - energy0) / abs(energy0) < 5e-3


def test_rayleigh_damping_produces_finite_state_and_target_decay():
    """Regression + physics: damping_ratio>0 used to yield NaN (bad eigenproblem).

    Log decrement per *cycle* (two successive |peak| values are half a period
    apart) must recover the requested damping ratio.
    """
    cfg = BeamConfig(n_elements=20, damping_ratio=0.02)
    beam = EulerBernoulliBeam(cfg)
    assert not torch.isnan(beam.C_free).any()
    period = 2.0 * math.pi / _analytic_w1(cfg)
    dt = period / 500.0
    history = _free_vibration_history(beam, P_LOAD, dt, int(2.5 * period / dt))
    assert all(math.isfinite(d) for _, d in history)

    peaks = []
    for i in range(1, len(history) - 1):
        a_prev, a_i, a_next = (abs(history[i - 1][1]), abs(history[i][1]),
                               abs(history[i + 1][1]))
        if a_i >= a_prev and a_i >= a_next and a_i > 1e-9:
            peaks.append(a_i)
    assert len(peaks) >= 4
    half_cycle_delta = math.log(peaks[0] / peaks[-1]) / (len(peaks) - 1)
    zeta = 2.0 * half_cycle_delta / math.sqrt(4.0 * math.pi ** 2 + (2.0 * half_cycle_delta) ** 2)
    assert zeta == pytest.approx(cfg.damping_ratio, rel=0.05)


def test_rayleigh_matrix_is_stiffness_proportional_and_finite():
    cfg = BeamConfig(n_elements=6, damping_ratio=0.03)
    beam = EulerBernoulliBeam(cfg)
    omega1 = float(beam._omegas[0])
    expected = (2.0 * cfg.damping_ratio / omega1) * beam.K_free
    assert torch.allclose(beam.C_free, expected, rtol=1e-12, atol=0.0)
    assert torch.isfinite(beam.C_free).all()


def test_deflection_at_validates_domain():
    beam = _beam(4)
    with pytest.raises(ValueError):
        beam.deflection_at(-0.01)
    with pytest.raises(ValueError):
        beam.consistent_point_load(1.01, P_LOAD)
    # endpoints are clamped into the domain and equal the nodal values
    assert beam.deflection_at(0.0) == pytest.approx(float(beam.disp[0]), abs=1e-15)