"""Verification of the explicit time-step stability criteria (CFL-class).

Reference values used here are obtained **independently of the module under
test**:

* hand-derived amplification matrix of semi-implicit Euler (the scheme used by
  ``IceDEM3D.step``),
* a minimal, self-contained scalar / chain integrator written inside this test
  file,
* closed-form dispersion relations of one-dimensional and simple-cubic
  spring lattices,
* dense eigenvalue solutions (``torch.linalg.eigvalsh``) of lattices assembled
  independently in the test,
* brute-force O(N^2) pair enumeration for the per-particle stiffness sums.

No test generates its reference with ``tensordem.stability`` itself.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem import stability


# --------------------------------------------------------------------------- #
# Independent reference implementations (deliberately not importing the module)
# --------------------------------------------------------------------------- #
def _step_scalar(x: float, v: float, dt: float, omega: float, damping_rate: float):
    """One semi-implicit Euler step of ``x'' = -omega^2 x - damping_rate x'``."""
    v_new = v + dt * (-omega * omega * x - damping_rate * v)
    x_new = x + dt * v_new
    return x_new, v_new


def _amplification_matrix(dt: float, omega: float, damping_rate: float) -> list[list[float]]:
    """Measure the 2x2 step matrix by probing the integrator with two unit states."""
    e1 = _step_scalar(1.0, 0.0, dt, omega, damping_rate)
    e2 = _step_scalar(0.0, 1.0, dt, omega, damping_rate)
    return [[e1[0], e2[0]], [e1[1], e2[1]]]


def _spectral_radius_numeric(matrix: list[list[float]]) -> float:
    """Spectral radius of a real 2x2 matrix from its characteristic polynomial."""
    trace = matrix[0][0] + matrix[1][1]
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    disc = trace * trace - 4.0 * det
    if disc >= 0.0:
        root = math.sqrt(disc)
        return max(abs(0.5 * (trace + root)), abs(0.5 * (trace - root)))
    return math.sqrt(abs(det))


def _run_scalar(x: float, v: float, dt: float, omega: float, damping_rate: float,
                steps: int) -> list[float]:
    """Trajectory of the undamped/damped oscillator energy (0.5*(w^2 x^2 + v^2))."""
    energy = []
    for _ in range(steps):
        x, v = _step_scalar(x, v, dt, omega, damping_rate)
        energy.append(0.5 * (omega * omega * x * x + v * v))
    return energy


def _run_chain(stiffness: float, mass: float, n: int, dt: float, steps: int,
               initial_velocity: list[float]) -> list[float]:
    """Total energy of a free-free 1-D spring-mass chain under semi-implicit Euler."""
    x = [0.0] * n
    v = list(initial_velocity)
    energy = []
    for _ in range(steps):
        force = [0.0] * n
        for i in range(n):
            if i > 0:
                force[i] += stiffness * (x[i - 1] - x[i])
            if i < n - 1:
                force[i] += stiffness * (x[i + 1] - x[i])
        for i in range(n):
            v[i] += dt * force[i] / mass
        for i in range(n):
            x[i] += dt * v[i]
        total = 0.5 * mass * sum(component * component for component in v)
        total += 0.5 * stiffness * sum((x[i + 1] - x[i]) ** 2 for i in range(n - 1))
        energy.append(total)
    return energy


def _highest_mode_shape(n: int) -> list[float]:
    """Exact highest eigenvector of a free-free chain: sin((n-1) * pi * i / n)."""
    return [math.sin((n - 1) * math.pi * (index + 1) / n) for index in range(n)]


# --------------------------------------------------------------------------- #
# 1. The amplification matrix itself (verifies the hand derivation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [0.05, 0.4, 1.0, 1.9, 2.0, 2.3])
@pytest.mark.parametrize("gamma", [0.0, 0.25, 1.0, 1.99, 2.0, 2.4])
def test_step_matrix_matches_hand_derivation(d, gamma):
    omega = 137.0
    dt = d / omega
    damping_rate = gamma / dt
    measured = _amplification_matrix(dt, omega, damping_rate)
    expected = [[1.0 - d * d, dt * (1.0 - gamma)],
                [-dt * omega * omega, 1.0 - gamma]]
    for row in range(2):
        for column in range(2):
            scale = 1.0 + abs(expected[row][column])
            assert abs(measured[row][column] - expected[row][column]) <= 1e-12 * scale


@pytest.mark.parametrize("d", [0.05, 0.4, 1.0, 1.9, 2.0, 2.3, 2.9])
@pytest.mark.parametrize("gamma", [0.0, 0.25, 1.0, 1.99, 2.0, 2.4])
def test_growth_factor_matches_independent_eigenvalues(d, gamma):
    omega = 50.0
    dt = d / omega
    damping_rate = gamma / dt
    measured = _amplification_matrix(dt, omega, damping_rate)
    reference = _spectral_radius_numeric(measured)
    assert stability.symplectic_euler_growth_factor(dt, omega, damping_rate) == pytest.approx(
        reference, rel=1e-12)


def test_stability_region_is_exactly_the_closed_form_boundary():
    """The predicate flips exactly at d = 2 and at gamma = 2."""
    omega = 200.0
    # Pure stiffness: dt_crit = 2 / omega
    assert stability.critical_timestep(omega) == pytest.approx(2.0 / omega, rel=1e-15)
    assert stability.symplectic_euler_stable(0.999 * 2.0 / omega, omega)
    assert not stability.symplectic_euler_stable(1.001 * 2.0 / omega, omega)
    # Pure damping: dt_crit = 2 m / c = 2 / damping_rate
    damping_rate = 30.0
    assert stability.critical_timestep(0.0, damping_rate) == pytest.approx(2.0 / damping_rate, rel=1e-15)
    assert stability.symplectic_euler_stable(0.999 * 2.0 / damping_rate, 0.0, damping_rate)
    assert not stability.symplectic_euler_stable(1.001 * 2.0 / damping_rate, 0.0, damping_rate)


# --------------------------------------------------------------------------- #
# 2. Physical threshold behaviour of the integrator (independent simulation)
# --------------------------------------------------------------------------- #
def test_undamped_energy_bounded_below_threshold_and_explosive_above():
    omega = 100.0
    stable_energy = _run_scalar(1.0, 0.0, 1.95 / omega, omega, 0.0, 3000)
    # rho = 1 for every d < 2, so the energy band stays finite near the threshold.
    assert max(stable_energy) / stable_energy[0] < 25.0
    unstable_energy = _run_scalar(1.0, 0.0, 2.05 / omega, omega, 0.0, 200)
    assert unstable_energy[-1] / unstable_energy[0] > 1e10


def test_marginal_step_grows_only_polynomially():
    """At dt = 2/omega the step matrix is a Jordan block: growth is algebraic (n^2)."""
    omega = 100.0
    dt = 2.0 / omega

    def energy_after(steps: int) -> float:
        return _run_scalar(1.0, 0.0, dt, omega, 0.0, steps)[-1]

    e1, e2, e3 = energy_after(500), energy_after(1000), energy_after(2000)
    assert e2 / e1 == pytest.approx(4.0, rel=5e-3)
    assert e3 / e2 == pytest.approx(4.0, rel=5e-3)
    # rho = 1 exactly, i.e. the step is stable but secular growth is real.
    assert stability.symplectic_euler_growth_factor(dt, omega, 0.0) == pytest.approx(1.0, rel=1e-12)


def test_damping_threshold_matches_two_over_gamma_rate():
    dt = 0.01
    for gamma in (0.5, 1.5, 1.99):
        damping_rate = gamma / dt
        x, v = 0.0, 1.0
        for _ in range(100):
            x, v = _step_scalar(x, v, dt, 0.0, damping_rate)
        assert v == pytest.approx((1.0 - gamma) ** 100, rel=1e-12)
        assert v > 0.0
    grow_rate = 2.01 / dt
    x, v = 0.0, 1.0
    for _ in range(100):
        x, v = _step_scalar(x, v, dt, 0.0, grow_rate)
    assert abs(v) == pytest.approx(1.01 ** 100, rel=1e-12)
    assert abs(v) > 1.0
    # Exactly at the boundary the velocity keeps its modulus (marginal).
    x, v = 0.0, 1.0
    for _ in range(101):
        x, v = _step_scalar(x, v, dt, 0.0, 2.0 / dt)
    assert abs(v) == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# 3. Many-body bound: Gershgorin row sum vs dense eigenvalues
# --------------------------------------------------------------------------- #
def _assemble_chain_stiffness(n: int, stiffness: float = 1.0) -> torch.Tensor:
    matrix = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n - 1):
        matrix[i, i] += stiffness
        matrix[i + 1, i + 1] += stiffness
        matrix[i, i + 1] -= stiffness
        matrix[i + 1, i] -= stiffness
    return matrix


def _assemble_simple_cubic_stiffness(side: int, stiffness: float = 1.0) -> torch.Tensor:
    index = lambda x, y, z: (z * side + y) * side + x
    n = side ** 3
    matrix = torch.zeros((n, n), dtype=torch.float64)
    for z in range(side):
        for y in range(side):
            for x in range(side):
                a = index(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    x2, y2, z2 = x + dx, y + dy, z + dz
                    if x2 < side and y2 < side and z2 < side:
                        b = index(x2, y2, z2)
                        matrix[a, a] += stiffness
                        matrix[b, b] += stiffness
                        matrix[a, b] -= stiffness
                        matrix[b, a] -= stiffness
    return matrix


def test_gershgorin_bound_converges_to_the_zone_corner_for_simple_cubic():
    """Finite free-free cubic lattice: closed-form top mode, bound -> exact as N grows."""
    stiffness, mass = 1.0, 1.0
    ratios = []
    for side in (2, 3, 4, 6):
        matrix = _assemble_simple_cubic_stiffness(side, stiffness)
        dense = float(torch.linalg.eigvalsh(matrix).max())
        # 2k(3 - cos_t1 - cos_t2 - cos_t3) with cos_t = -cos(pi/N) for the top mode
        cos_top = math.cos(math.pi * (side - 1) / side)
        analytic = 2.0 * stiffness * 3.0 * (1.0 - cos_top) / mass
        assert dense == pytest.approx(analytic, rel=1e-10)
        # interior particle: 6 bonds -> Gershgorin row sum of |.| = 2 * 6k = 12k
        sums = torch.full((side ** 3,), 6.0 * stiffness, dtype=torch.float64)
        bound = stability.gershgorin_omega_bound(sums, mass)
        assert bound == pytest.approx(math.sqrt(12.0 * stiffness / mass), rel=1e-12)
        assert bound >= math.sqrt(analytic) * (1.0 - 1e-12)
        ratios.append(bound / math.sqrt(analytic))
    assert ratios[0] > ratios[1] > ratios[2] > ratios[3]      # monotone tightening
    exact_top = 6.0 * (1.0 + math.cos(math.pi / 6))
    assert ratios[-1] == pytest.approx(math.sqrt(12.0 / exact_top), rel=1e-12)
    assert ratios[-1] < 1.08                                   # 7% conservative at N=6


def test_gershgorin_bound_is_exact_for_one_dimensional_chain():
    n, stiffness, mass = 16, 1.0, 1.0
    matrix = _assemble_chain_stiffness(n, stiffness)
    dense = float(torch.linalg.eigvalsh(matrix).max())
    analytic = (2.0 * stiffness / mass) * (1.0 + math.cos(math.pi / n))
    assert dense == pytest.approx(analytic, rel=1e-12)
    sums = torch.full((n,), 2.0 * stiffness, dtype=torch.float64)
    bound = stability.gershgorin_omega_bound(sums, mass)
    assert bound == pytest.approx(2.0 * math.sqrt(stiffness / mass), rel=1e-12)
    assert bound >= math.sqrt(analytic)  # conservative, ratio -> 1 for large n


def test_gershgorin_bound_never_underestimates_random_lattices():
    generator = torch.Generator().manual_seed(20261101)
    for _ in range(20):
        n = 6
        mass = float(torch.rand((), generator=generator, dtype=torch.float64) + 0.5)
        matrix = torch.zeros((n, n), dtype=torch.float64)
        sums = torch.zeros(n, dtype=torch.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand((), generator=generator, dtype=torch.float64) < 0.6:
                    k = float(torch.rand((), generator=generator, dtype=torch.float64) * 9.0 + 1.0)
                    matrix[i, i] += k
                    matrix[j, j] += k
                    matrix[i, j] -= k
                    matrix[j, i] -= k
                    sums[i] += k
                    sums[j] += k
        omega_exact = math.sqrt(float(torch.linalg.eigvalsh(matrix).max()) / mass)
        bound = stability.gershgorin_omega_bound(sums, mass)
        assert bound >= omega_exact * (1.0 - 1e-12)


def test_chain_threshold_follows_exact_dispersion_limit():
    n, stiffness, mass, steps = 32, 1.0, 1.0, 1500
    omega_max = math.sqrt((2.0 * stiffness / mass) * (1.0 + math.cos(math.pi / n)))
    mode = _highest_mode_shape(n)
    stable = _run_chain(stiffness, mass, n, 0.5 * 2.0 / omega_max, steps, mode)
    assert max(stable) / stable[0] < 2.5
    with pytest.raises(OverflowError):
        _run_chain(stiffness, mass, n, 1.05 * 2.0 / omega_max, steps, mode)


# --------------------------------------------------------------------------- #
# 4. Engine-level audit: the existing dt guard is inside the stability limit
# --------------------------------------------------------------------------- #
ENGINE_CASES = [
    {},
    dict(nx=3, ny=3, nz=3),
    dict(nx=6, ny=4, nz=3),
    dict(nx=12, ny=6, nz=3),
    dict(top_platen_enabled=True, bottom_platen_enabled=True),
    dict(top_platen_enabled=True, platen_damping=50.0, top_platen_velocity=-0.01),
    dict(geometry="cylinder_xz", nx=5, ny=3, nz=5),
    dict(bond_stiffness=2.0e5),
    dict(contact_stiffness=2.0e5),
    dict(contact_damping=500.0),
    dict(drag=100.0),
    dict(radius=0.005),
    dict(fix_bottom=True),
]


@pytest.mark.parametrize("overrides", ENGINE_CASES)
def test_engine_step_is_inside_the_cfl_limit(overrides):
    model = IceDEM3D(DEM3DConfig(**overrides))
    report = stability.stability_report(model)
    assert report["is_stable"]
    assert report["critical_dt_s"] >= report["dt_s"]
    assert report["engine_recommended_dt_s"] <= report["critical_dt_s"]
    assert report["margin_factor"] >= 1.0
    assert report["growth_factor_at_dt"] <= 1.0 + 1e-12
    json.dumps(report)  # report must stay JSON serialisable


def test_engine_limit_scales_like_inverse_sqrt_of_stiffness():
    reference = stability.stability_report(IceDEM3D(DEM3DConfig(bond_stiffness=1.0e3)))
    for factor in (4.0, 16.0):
        boosted = stability.stability_report(
            IceDEM3D(DEM3DConfig(bond_stiffness=1.0e3 * factor)))
        assert boosted["critical_dt_s"] / reference["critical_dt_s"] == pytest.approx(
            1.0 / math.sqrt(factor), rel=1e-12)


def test_engine_run_is_bounded_at_the_audited_step():
    """Closed system (indenter disengaged): perturbation must decay, not grow."""
    torch.manual_seed(7)
    model = IceDEM3D(DEM3DConfig(nx=6, ny=4, nz=3))
    model.tool_start = torch.tensor([0.0, 0.0, 1.0e3], dtype=torch.float64)
    model.velocities += 1.0e-3 * torch.randn_like(model.velocities)
    initial = float(0.5 * model.config.mass * model.velocities.square().sum())
    peak = initial
    for _ in range(2000):
        model.step()
        kinetic = float(0.5 * model.config.mass * model.velocities.square().sum())
        peak = max(peak, kinetic)
    final = float(0.5 * model.config.mass * model.velocities.square().sum())
    assert torch.isfinite(model.positions).all()
    assert peak < 1.5 * initial       # no growth of the perturbation
    assert final < 0.9 * initial      # drag must remove energy
    assert model.broken_bonds == 0


def test_engine_closed_system_energy_does_not_drift():
    """Closed, unforced block: kinetic + bond energy must stay bounded (symplectic)."""
    torch.manual_seed(7)
    model = IceDEM3D(DEM3DConfig(nx=6, ny=4, nz=3))
    model.tool_start = torch.tensor([0.0, 0.0, 1.0e3], dtype=torch.float64)
    model.velocities += 1.0e-3 * torch.randn_like(model.velocities)

    def mechanical_energy() -> float:
        kinetic = 0.5 * model.config.mass * float(model.velocities.square().sum())
        i, j = model.pairs.T
        distance = torch.linalg.vector_norm(model.positions[j] - model.positions[i], dim=1)
        extension = distance - model.rest_lengths
        bond = 0.5 * model.config.bond_stiffness * float((extension.square() * model.alive).sum())
        return kinetic + bond

    initial = mechanical_energy()
    peak = initial
    for _ in range(2000):
        model.step()
        peak = max(peak, mechanical_energy())
    assert model.broken_bonds == 0
    assert peak / initial == pytest.approx(1.0, abs=5e-3)


def test_per_particle_stiffness_sums_match_brute_force_enumeration():
    """Cross-check the bound inputs against O(N^2) pair enumeration."""
    cfg = DEM3DConfig(nx=3, ny=3, nz=2, fix_x_edges=True)
    model = IceDEM3D(cfg)
    model.positions = 0.6 * model.initial_positions  # compress -> nanobond contacts
    stiffness_sums = stability.particle_stiffness_sums(model)

    n = len(model.positions)
    alive = model.alive
    pairs = model.pairs
    bond_key = {(int(a), int(b)) for a, b in pairs[alive]}
    expected = torch.zeros(n, dtype=torch.float64)
    bond_counts = torch.zeros(n, dtype=torch.float64)
    for a, b in bond_key:
        bond_counts[a] += 1
        bond_counts[b] += 1
    expected += bond_counts * cfg.bond_stiffness
    positions = model.positions
    tool_position = model.tool_position
    for i in range(n):
        for j in range(i + 1, n):
            distance = float(torch.linalg.vector_norm(positions[j] - positions[i]))
            if distance < 2.0 * cfg.radius and (i, j) not in bond_key:
                expected[i] += cfg.contact_stiffness
                expected[j] += cfg.contact_stiffness
    for i in range(n):
        distance = float(torch.linalg.vector_norm(positions[i] - tool_position))
        if distance < cfg.radius + cfg.tool_radius:
            expected[i] += cfg.contact_stiffness
    assert torch.allclose(stiffness_sums, expected, rtol=1e-12, atol=1e-12)


def test_contact_stiffness_enters_the_limit():
    """With every bond broken and every particle coincident, S_i = (n-1) k_c."""
    cfg = DEM3DConfig(nx=3, ny=3, nz=2, fix_x_edges=False)
    model = IceDEM3D(cfg)
    model.alive[:] = False
    model.positions[:] = 0.0
    model.tool_start = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)  # disengage indenter
    report = stability.stability_report(model)
    n = len(model.positions)
    expected_sum = (n - 1) * cfg.contact_stiffness
    assert report["max_stiffness_sum_N_per_m"] == pytest.approx(expected_sum, rel=1e-12)
    assert report["critical_dt_s"] == pytest.approx(
        math.sqrt(2.0 * cfg.mass / expected_sum), rel=1e-12)


def test_fixed_particles_are_excluded_from_the_governing_bound():
    cfg = DEM3DConfig(nx=4, ny=3, nz=2, fix_x_edges=True)
    model = IceDEM3D(cfg)
    model.positions[:] = 0.0
    model.alive[:] = False
    model.tool_start = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)
    free = ~model.fixed
    n = len(model.positions)
    # A free particle still couples to fixed neighbours, so S is unchanged, but
    # the reported governing sum must be taken over free rows only.
    sums = stability.particle_stiffness_sums(model)
    report = stability.stability_report(model)
    assert report["n_free_particles"] == int(free.sum().item())
    assert report["max_stiffness_sum_N_per_m"] == pytest.approx(
        float(sums[free].max().item()), rel=1e-15)
    fully_fixed = IceDEM3D(DEM3DConfig(nx=2, ny=2, nz=2))
    assert bool(fully_fixed.fixed.all())
    fixed_report = stability.stability_report(fully_fixed)
    assert fixed_report["n_free_particles"] == 0
    assert fixed_report["is_stable"]  # no free degree of freedom, nothing to destabilise
    assert n > 0


# --------------------------------------------------------------------------- #
# 5. Input validation and the CLI surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    dict(dt=0.0), dict(dt=-1.0), dict(dt=float("nan")), dict(dt=float("inf")),
])
def test_growth_factor_rejects_invalid_dt(kwargs):
    with pytest.raises(ValueError):
        stability.symplectic_euler_growth_factor(kwargs["dt"], 1.0)


@pytest.mark.parametrize("omega", [-1.0, float("nan"), float("inf")])
def test_growth_factor_rejects_invalid_omega(omega):
    with pytest.raises(ValueError):
        stability.symplectic_euler_growth_factor(1e-3, omega)


def test_negative_stiffness_sum_rejected():
    with pytest.raises(ValueError):
        stability.gershgorin_omega_bound(torch.tensor([1.0, -1.0], dtype=torch.float64), 1.0)


def test_critical_timestep_reports_infinity_without_coupling():
    assert stability.critical_timestep(0.0, 0.0) == math.inf
    assert stability.critical_timestep(4.0, 1.0) == pytest.approx(0.5)


def test_cli_stability_report_writes_json(tmp_path, capsys):
    from tensordem.dem3d_cli import main

    main(["--nx", "4", "--ny", "3", "--nz", "2", "--steps", "1",
          "--output", str(tmp_path), "--stability-report"])
    payload = json.loads((tmp_path / "stability_report.json").read_text(encoding="utf-8"))
    assert payload["is_stable"] is True
    assert payload["critical_dt_s"] > payload["dt_s"]
    assert payload["engine_recommended_dt_s"] <= payload["critical_dt_s"]
    assert "growth_factor_at_dt" in payload
    printed = json.loads(capsys.readouterr().out)
    assert printed["critical_dt_s"] == payload["critical_dt_s"]
    # Read-only: no solver history is written by the audit.
    assert not (tmp_path / "history.csv").exists()