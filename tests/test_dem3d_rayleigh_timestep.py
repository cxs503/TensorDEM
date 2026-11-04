"""Verification of the Rayleigh time-step estimate against exact lattice dynamics.

Every reference value below is generated **independently of the module**: the
textbook Rayleigh table, an explicit 2x2 map built with NumPy, brute-force bond
enumeration inside the test, closed-form dispersion relations and a dense
eigenvalue solve of a lattice assembled here. Nothing imports
``tensordem.rayleigh_timestep`` to build a reference.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import born_huang
from tensordem import rayleigh_timestep as rt

K = 2_000.0
R = 0.025
RHO = 917.0
A = 2.0 * R
M = RHO * (4.0 / 3.0) * math.pi * R**3


# --------------------------------------------------------------------------- #
# Independent references
# --------------------------------------------------------------------------- #
def reference_offsets() -> np.ndarray:
    """The 26 neighbour offsets, enumerated here (not imported)."""
    return np.array(
        [(dx, dy, dz)
         for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
         if not (dx == dy == dz == 0)],
        dtype=float,
    )


def reference_constants(stiffness: float, spacing: float) -> tuple[float, float]:
    """Brute-force C11 and C44 of the affine central-force lattice."""
    offsets = reference_offsets()
    lengths = np.linalg.norm(offsets, axis=1) * spacing
    normals = offsets / np.linalg.norm(offsets, axis=1)[:, None]
    weights = 0.5 * stiffness * lengths**2  # 26 offsets = 2 copies of the 13 shells
    scale = 1.0 / spacing**3
    c11 = scale * math.fsum(weights * normals[:, 0] ** 4)
    c44 = scale * math.fsum(weights * normals[:, 0] ** 2 * normals[:, 1] ** 2)
    return c11, c44


def reference_supercell_hessian(supercell: int, stiffness: float, spacing: float) -> np.ndarray:
    """O(N^2) Hessian assembly: every atom against every periodic image."""
    offsets = reference_offsets()
    n = supercell**3
    positions = np.array(
        [(i, j, k) for k in range(supercell) for j in range(supercell)
         for i in range(supercell)], dtype=float
    ) * spacing
    hessian = np.zeros((3 * n, 3 * n))
    box = supercell * spacing
    for i in range(n):
        for j in range(i + 1, n):
            delta = positions[j] - positions[i]
            delta -= box * np.round(delta / box)  # minimum image
            distance = np.linalg.norm(delta)
            # bond exists if the minimum image matches one of the 26 shell vectors
            if abs(distance - spacing) > 1e-12 and True:
                shells = {spacing, spacing * math.sqrt(2.0), spacing * math.sqrt(3.0)}
                if not any(abs(distance - s) < 1e-12 for s in shells):
                    continue
            elif distance == 0.0:  # pragma: no cover
                continue
            normal = delta / distance
            block = stiffness * np.outer(normal, normal)
            hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] += block
            hessian[3 * j:3 * j + 3, 3 * j:3 * j + 3] += block
            hessian[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= block
            hessian[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= block
    return hessian


def reference_mode_growth(omega: float, timestep: float) -> float:
    """Spectral radius of the 2x2 semi-implicit Euler map, via NumPy eigenvalues."""
    transition = np.array(
        [[1.0 - (timestep * omega) ** 2, timestep], [-timestep * omega**2, 1.0]]
    )
    return float(np.max(np.abs(np.linalg.eigvals(transition))))


# --------------------------------------------------------------------------- #
# Rayleigh speed ratio
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("poisson", "published", "tolerance"),
    [
        (0.0, 0.874032, 1e-6),
        (0.25, 0.919402, 1e-6),
        # six-decimal tables disagree in the fifth digit at nu = 1/3; both exact
        # formulations below agree to 1e-12, so the tolerance is widened rather
        # than the reference value bent toward the implementation.
        (1.0 / 3.0, 0.932472, 1e-4),
        (0.5, 0.955313, 1e-6),
    ],
)
def test_rayleigh_ratio_matches_published_table(poisson, published, tolerance):
    exact = rt.rayleigh_speed_ratio(poisson)
    cubic = rt.rayleigh_speed_ratio_lamb_cubic(poisson)
    assert abs(exact - published) / published < tolerance
    assert abs(cubic - exact) / exact < 1e-12


def test_rayleigh_ratio_two_paths_agree_over_the_whole_range():
    for poisson in np.linspace(0.0, 0.5, 21):
        exact = rt.rayleigh_speed_ratio(poisson)
        cubic = rt.rayleigh_speed_ratio_lamb_cubic(poisson)
        assert abs(cubic - exact) / exact < 1e-12
        assert 0.87 < exact < 0.96


def test_rayleigh_ratio_is_monotone_and_bounded_by_the_shear_speed():
    values = [rt.rayleigh_speed_ratio(nu) for nu in np.linspace(0.0, 0.5, 11)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert all(0.0 < v < 1.0 for v in values)


def test_rayleigh_ratio_rejects_out_of_range_poisson():
    for bad in (-0.1, 0.6, float("nan")):
        with pytest.raises(ValueError):
            rt.rayleigh_speed_ratio(bad)


def test_bergmann_fit_is_only_an_approximation():
    worst = max(
        abs(rt.rayleigh_speed_ratio_approx(nu) - rt.rayleigh_speed_ratio(nu))
        / rt.rayleigh_speed_ratio(nu)
        for nu in np.linspace(0.0, 0.5, 21)
    )
    assert 1e-4 < worst < 1e-2  # documented: good to a few tenths of a percent


# --------------------------------------------------------------------------- #
# Dispersion and the exact maximum frequency
# --------------------------------------------------------------------------- #
def test_acoustic_slopes_recover_the_macroscopic_elastic_constants():
    """c_L^2 = C11/rho and c_T^2 = C44/rho, with rho = m/a^3 (independent C11, C44)."""
    c11, c44 = reference_constants(K, A)
    rho_lattice = M / A**3
    q = 1e-4
    wave_vectors = np.array([[q, 0.0, 0.0]])
    omega = rt.dispersion_frequencies(wave_vectors, stiffness=K, mass=M, spacing=A)[0]
    c_long = omega[2] / q
    c_trans = omega[0] / q
    assert abs(c_long**2 - c11 / rho_lattice) / (c11 / rho_lattice) < 1e-6
    assert abs(c_trans**2 - c44 / rho_lattice) / (c44 / rho_lattice) < 1e-6


def test_dispersion_along_100_matches_closed_forms():
    qs = np.linspace(0.0, math.pi / A, 17)
    wave_vectors = np.column_stack([qs, np.zeros_like(qs), np.zeros_like(qs)])
    omega = rt.dispersion_frequencies(wave_vectors, stiffness=K, mass=M, spacing=A)
    closed_l = np.sqrt(26.0 * K / (3.0 * M) * (1.0 - np.cos(qs * A)))
    closed_t = np.sqrt(14.0 * K / (3.0 * M) * (1.0 - np.cos(qs * A)))
    assert np.allclose(omega[:, 2], closed_l, rtol=1e-12, atol=0.0)
    assert np.allclose(omega[:, 0], closed_t, rtol=1e-12, atol=0.0)
    assert np.allclose(omega[:, 1], closed_t, rtol=1e-12, atol=0.0)  # transverse degeneracy


def test_max_frequency_closed_form_scan_and_supercell_agree():
    closed = rt.exact_lattice_omega_max(stiffness=K, mass=M)
    assert abs(closed - math.sqrt(52.0 * K / (3.0 * M))) / closed < 1e-15
    scanned, argmax = rt.lattice_max_frequency(stiffness=K, mass=M, spacing=A)
    assert abs(scanned - closed) / closed < 1e-12
    # argmax is the zone boundary X (or a symmetry image)
    assert sorted(np.abs(argmax * A / math.pi).round(6)) == [0.0, 0.0, 1.0]

    hessian = reference_supercell_hessian(4, K, A)
    omega = np.sqrt(np.clip(np.linalg.eigvalsh(hessian) / M, 0.0, None))
    assert abs(omega.max() - closed) / closed < 1e-10


def test_periodic_supercell_builder_matches_the_established_one():
    """The new builder agrees with ``born_huang.solver_supercell_bonds`` at n = 3."""
    ours = rt.periodic_supercell_bonds((3, 3, 3), spacing=A)
    theirs = born_huang.solver_supercell_bonds(A, supercell=(3, 3, 3), stiffness=K)
    assert ours["pairs"].shape == theirs["pairs"].shape
    assert abs(ours["bonds_per_cell"] - theirs["bonds_per_cell"]) < 1e-12
    assert np.allclose(np.sort(ours["positions"], axis=0),
                       np.sort(theirs["positions"], axis=0))

    ours_h = rt.supercell_hessian((3, 3, 3), stiffness=K, spacing=A)
    theirs_h = reference_supercell_hessian(3, K, A)
    assert np.allclose(np.linalg.eigvalsh(ours_h), np.linalg.eigvalsh(theirs_h),
                       rtol=1e-10, atol=1e-8)


def test_supercell_needs_three_cells():
    with pytest.raises(ValueError):
        rt.periodic_supercell_bonds((2, 2, 2), spacing=A)


# --------------------------------------------------------------------------- #
# Step limits and the threshold
# --------------------------------------------------------------------------- #
def test_exact_timestep_closed_form_and_gershgorin_ratio():
    exact = rt.exact_lattice_timestep(stiffness=K, mass=M)
    assert abs(exact - math.sqrt(3.0 * M / (13.0 * K))) / exact < 1e-15
    gershgorin = rt.gershgorin_timestep(stiffness=K, mass=M)
    assert abs(gershgorin / exact - 1.0 / math.sqrt(3.0)) < 1e-15


def test_mode_growth_factor_matches_the_two_by_two_map():
    for omega in (1.0, 50.0, 760.0):
        for timestep in (1e-5, 1e-3, 2.0 / omega * 0.5, 2.0 / omega * 0.99,
                         2.0 / omega * 1.02):
            ours = rt.mode_growth_factor(omega, timestep)
            reference = reference_mode_growth(omega, timestep)
            assert abs(ours - reference) < 1e-12


def test_growth_rate_is_one_at_the_exact_limit():
    hessian = rt.supercell_hessian((4, 4, 4), stiffness=K, spacing=A)
    exact = rt.exact_lattice_timestep(stiffness=K, mass=M)
    # at the limit the pair sits on the double root -1 (marginally stable); the
    # measured 1.00000006 is the round-off of the supercell eigenvalue, not growth
    assert rt.growth_rate(hessian, mass=M, timestep=exact) <= 1.0 + 1e-6
    assert rt.growth_rate(hessian, mass=M, timestep=exact * 0.999) <= 1.0 + 1e-6
    assert rt.growth_rate(hessian, mass=M, timestep=exact * 1.001) > 1.0


def test_simulated_growth_matches_the_prediction():
    exact = rt.exact_lattice_timestep(stiffness=K, mass=M)
    unstable = rt.simulate_periodic_lattice(
        stiffness=K, mass=M, spacing=A, timestep=1.02 * exact, steps=200
    )
    assert unstable["measured_growth_per_step"] > 1.05
    assert abs(unstable["measured_growth_per_step"]
               - unstable["theoretical_growth_per_step"]) / unstable[
        "theoretical_growth_per_step"] < 1e-3

    stable = rt.simulate_periodic_lattice(
        stiffness=K, mass=M, spacing=A, timestep=0.99 * exact, steps=200
    )
    assert stable["measured_growth_per_step"] <= 1.0 + 1e-3
    assert np.all(np.isfinite(stable["max_displacement"]))


def test_rayleigh_rule_is_density_convention_dependent():
    report = rt.timestep_limit_report(bond_stiffness=K, radius=R, density=RHO, mass=M,
                                      points_per_axis=21)
    ratios = report["ratios"]
    assert ratios["gershgorin_over_exact"] < 1.0
    assert ratios["rayleigh_voigt_lattice_density_over_exact"] < 1.0
    assert ratios["rayleigh_voigt_material_density_over_exact"] > 1.0
    assert ratios["engine_elastic_over_exact"] < 0.5
    assert abs(report["omega_max_bz"] - report["omega_max_closed_form"]) / report[
        "omega_max_closed_form"] < 1e-12


def test_timestep_report_rejects_non_physical_input():
    with pytest.raises(ValueError):
        rt.timestep_limit_report(bond_stiffness=-1.0)
    with pytest.raises(ValueError):
        rt.rayleigh_timestep(radius=0.0, shear_speed=10.0, poisson_ratio=0.3)