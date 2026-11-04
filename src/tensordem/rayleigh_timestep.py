"""Rayleigh-surface-wave time-step estimate for the 3-D bond lattice, and its audit.

Why this module exists
----------------------
``docs/dem3d_timestep_stability.md`` closes with an explicit, recorded gap:

    **No Rayleigh-wave estimate adopted.** Classical DEM practice uses a Rayleigh
    surface-wave limit ``dt ~ r_min / c_R``. It needs a calibrated shear modulus
    ... the Gershgorin bound is rigorous and needs no such calibration, so it was
    preferred and the Rayleigh estimate was not implemented.

This module implements that estimate *and* the reference it has to be judged
against, so the folklore rule stops being an unquantified assumption:

1. :func:`rayleigh_speed_ratio` solves the Rayleigh secular equation for
   ``c_R / c_S`` exactly (bisection on the traction-free surface condition),
   with a second, algebraically independent solution via the Lamb/Rayleigh
   cubic (:func:`rayleigh_speed_ratio_lamb_cubic`) and the classical
   Bergmann fit (:func:`rayleigh_speed_ratio_approx`) kept separate.
2. :func:`dynamical_matrix` / :func:`dispersion_frequencies` give the exact
   Bloch spectrum of the solver's 26-neighbour central-force lattice, and
   :func:`lattice_max_frequency` the exact
   ``omega_max = max_q omega(q)``.
3. :func:`exact_lattice_timestep` turns that into the *exact* explicit-step
   limit ``dt = 2 / omega_max``. For symplectic (semi-implicit) Euler applied to
   a linear, undamped, equal-mass system this is not a heuristic: every normal
   mode is an independent 2x2 map that is non-expanding iff ``dt * omega <= 2``.
4. :func:`gershgorin_timestep` (the bound the repository already trusts),
   :func:`rayleigh_timestep` (the classical rule) and the engine's own
   ``DEM3DConfig.recommended_dt`` elastic factor are reported side by side by
   :func:`timestep_limit_report`, in ratios, so the conservativeness or
   unsafety of each is a number rather than an opinion.

Analytic reference (closed forms used below, derived by hand)
-------------------------------------------------------------
With the 13 half-space offsets of ``IceDEM3D`` split into 3 axis bonds of
length ``a``, 6 face-diagonal bonds of length ``a*sqrt(2)`` and 4 body-diagonal
bonds of length ``a*sqrt(3)`` (``a = 2r``), the Bloch dynamical matrix is

    D(q) = (2 k / m) * sum_{13 half bonds} n n^T (1 - cos(q . d)),

so the acoustic slopes recover the macroscopic constants of
``tensordem.lattice_elasticity3d.periodic_elastic_constants``:

    c_L^2 = C11 / rho_lattice,   c_T^2 = C44 / rho_lattice,
    rho_lattice = m / a^3.

Along ``q = (q, 0, 0)`` the branches are exactly

    omega_L^2 = (26 k / 3 m) (1 - cos(q a)),
    omega_T^2 = (14 k / 3 m) (1 - cos(q a)),

both maximised at the zone-boundary point X = (pi/a, 0, 0):

    omega_max^2 = 52 k / (3 m)   =>   dt_exact = 2 / omega_max = sqrt(3 m / (13 k)).

``rho_lattice`` (not the material density) is the consistent density for the
discrete wave equation, and the difference matters: ``m = rho (4/3) pi r^3``
gives ``rho_lattice = (pi/6) rho``, i.e. a factor ``sqrt(6/pi) = 1.38`` in
``c_R`` and therefore in the Rayleigh step. Both conventions are reported.

Scope: linear, undamped, intact-bond lattice with one translational DOF per
particle. No contact, fracture, damping, rotation or material calibration claim.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "POISSON_RATIO_TABLE",
    "RAYLEIGH_RATIO_TABLE",
    "solver_half_space_offsets",
    "rayleigh_speed_ratio",
    "rayleigh_speed_ratio_lamb_cubic",
    "rayleigh_speed_ratio_approx",
    "rayleigh_ratio_table",
    "shear_wave_speed",
    "longitudinal_wave_speed",
    "rayleigh_wave_speed",
    "dynamical_matrix",
    "dispersion_frequencies",
    "lattice_max_frequency",
    "exact_lattice_omega_max",
    "exact_lattice_timestep",
    "gershgorin_timestep",
    "rayleigh_timestep",
    "periodic_supercell_bonds",
    "supercell_hessian",
    "mode_growth_factor",
    "growth_rate",
    "simulate_periodic_lattice",
    "timestep_limit_report",
]

#: Published ``c_R / c_S`` anchors for an isotropic solid, the four entries that
#: appear in every textbook table (Achenbach 1973; Graff, *Wave Motion in Elastic
#: Solids*).  Values are quoted to six decimals, so agreement can only be
#: asserted at the ~1e-6 level; the sweep in :func:`rayleigh_ratio_sweep` carries
#: the remaining Poisson ratios, where the only available reference is the two
#: independent exact formulations.
POISSON_RATIO_TABLE: tuple[float, ...] = (0.0, 0.25, 1.0 / 3.0, 0.5)
RAYLEIGH_RATIO_TABLE: tuple[float, ...] = (0.874032, 0.919402, 0.932472, 0.955313)


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _check_poisson(poisson_ratio: float) -> float:
    nu = float(poisson_ratio)
    # nu = 0.5 is the incompressible limit (c_P -> infinity, kappa = 0) and is
    # admissible here: the secular equation stays regular for kappa = 0.
    if not math.isfinite(nu) or nu < 0.0 or nu > 0.5:
        raise ValueError("poisson_ratio must satisfy 0 <= nu <= 0.5")
    return nu


# --------------------------------------------------------------------------- #
# Rayleigh surface-wave speed
# --------------------------------------------------------------------------- #
def rayleigh_speed_ratio(poisson_ratio: float, *, iterations: int = 200) -> float:
    """Exact ``c_R / c_S`` from the traction-free surface secular equation.

    With ``x = (c_R/c_S)^2`` and ``kappa = (c_S/c_P)^2 = (1-2nu)/(2(1-nu))`` the
    stress-free condition is

        f(x) = (2 - x)^2 - 4 * sqrt(1 - x) * sqrt(1 - kappa x) = 0.

    ``f(0) = 0`` is the spurious shear root; just above 0 the function is
    negative (``f ~ 2x(kappa-1)``) and ``f(1) = 1``, so the physical root is
    unique in ``(0, 1)`` and plain bisection is globally convergent. Returned as
    ``sqrt(x)``.
    """
    nu = _check_poisson(poisson_ratio)
    kappa = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))

    def secular(x: float) -> float:
        return (2.0 - x) ** 2 - 4.0 * math.sqrt(max(0.0, 1.0 - x)) * math.sqrt(
            1.0 - kappa * x
        )

    lo, hi = 1e-15, 1.0
    if secular(lo) > 0.0:  # pragma: no cover - guarded by kappa < 1
        raise ArithmeticError("Rayleigh bracket is not sign-separated")
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if secular(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    root = 0.5 * (lo + hi)
    return math.sqrt(root)


def rayleigh_speed_ratio_lamb_cubic(poisson_ratio: float) -> float:
    """Second, independent solution of the same condition via the Lamb cubic.

    Squaring the secular equation gives, for ``x = (c_R/c_S)^2``,

        x^3 - 8 x^2 + 8 x (3 - 2 kappa) - 16 (1 - kappa) = 0,

    whose unique real root in ``(0, 1)`` is the Rayleigh root. Solving the cubic
    instead of the half-integer powers avoids the ``sqrt`` entirely, so agreement
    between the two paths is evidence about the arithmetic, not about a shared
    expression.
    """
    nu = _check_poisson(poisson_ratio)
    kappa = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    coeffs = [1.0, -8.0, 8.0 * (3.0 - 2.0 * kappa), -16.0 * (1.0 - kappa)]
    roots = np.roots(coeffs)
    real = [r.real for r in roots if abs(r.imag) < 1e-8 and 0.0 < r.real < 1.0]
    if len(real) != 1:  # pragma: no cover - the cubic has exactly one such root
        raise ArithmeticError(f"expected one physical Rayleigh root, got {real}")
    return math.sqrt(real[0])


def rayleigh_speed_ratio_approx(poisson_ratio: float) -> float:
    """Bergmann's engineering fit ``(0.87 + 1.12 nu) / (1 + nu)`` (not exact)."""
    nu = _check_poisson(poisson_ratio)
    return (0.87 + 1.12 * nu) / (1.0 + nu)


def rayleigh_ratio_table(
    poisson_ratios: Sequence[float] = POISSON_RATIO_TABLE,
    references: Sequence[float] = RAYLEIGH_RATIO_TABLE,
) -> list[dict[str, float]]:
    """Compare both exact paths and the Bergmann fit with the published anchors."""
    if len(poisson_ratios) != len(references):
        raise ValueError("poisson_ratios and references must have equal length")
    rows: list[dict[str, float]] = []
    for nu, ref in zip(poisson_ratios, references):
        bisect = rayleigh_speed_ratio(nu)
        cubic = rayleigh_speed_ratio_lamb_cubic(nu)
        approx = rayleigh_speed_ratio_approx(nu)
        rows.append(
            {
                "poisson_ratio": float(nu),
                "ratio_bisection": bisect,
                "ratio_lamb_cubic": cubic,
                "ratio_bergmann": approx,
                "ratio_reference": float(ref),
                "err_bisection_vs_reference": (bisect - ref) / ref,
                "err_cubic_vs_bisection": (cubic - bisect) / bisect,
                "err_bergmann_vs_reference": (approx - ref) / ref,
            }
        )
    return rows


def rayleigh_ratio_sweep(
    poisson_ratios: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Both exact paths and the Bergmann fit over the whole admissible range."""
    if poisson_ratios is None:
        poisson_ratios = [0.5 * i / 10.0 for i in range(11)]
    rows: list[dict[str, float]] = []
    for nu in poisson_ratios:
        bisect = rayleigh_speed_ratio(nu)
        cubic = rayleigh_speed_ratio_lamb_cubic(nu)
        approx = rayleigh_speed_ratio_approx(nu)
        rows.append(
            {
                "poisson_ratio": float(nu),
                "ratio_bisection": bisect,
                "ratio_lamb_cubic": cubic,
                "ratio_bergmann": approx,
                "err_cubic_vs_bisection": (cubic - bisect) / bisect,
                "err_bergmann_vs_exact": (approx - bisect) / bisect,
            }
        )
    return rows


def shear_wave_speed(shear_modulus: float, density: float) -> float:
    return math.sqrt(_finite_positive(shear_modulus, "shear_modulus")
                     / _finite_positive(density, "density"))


def longitudinal_wave_speed(bulk_p_modulus: float, density: float) -> float:
    return math.sqrt(_finite_positive(bulk_p_modulus, "modulus")
                     / _finite_positive(density, "density"))


def rayleigh_wave_speed(shear_modulus: float, density: float,
                        poisson_ratio: float) -> float:
    """Rayleigh speed from the isotropic ``c_S`` and the exact ratio."""
    return rayleigh_speed_ratio(poisson_ratio) * shear_wave_speed(shear_modulus, density)


# --------------------------------------------------------------------------- #
# Exact Bloch spectrum of the solver lattice
# --------------------------------------------------------------------------- #
def solver_half_space_offsets() -> np.ndarray:
    """The solver's 13 half-space neighbour offsets ``(13, 3)`` (single source)."""
    from .born_huang import solver_half_space_offsets as _offsets

    return _offsets()


def _bonds(stiffness: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    offsets = solver_half_space_offsets().astype(float)
    lengths = np.linalg.norm(offsets, axis=1)
    return offsets * float(spacing), offsets / lengths[:, None]


def dynamical_matrix(
    wave_vector: Sequence[float],
    *,
    stiffness: float,
    mass: float,
    spacing: float,
) -> np.ndarray:
    """Bloch dynamical matrix ``D(q)`` with units rad^2/s^2, shape ``(3, 3)``.

    ``D(q) = (2 k / m) sum_{13 half bonds} n n^T (1 - cos(q . d))``; eigenvalues
    are ``omega^2`` of the three polarisations.
    """
    k = _finite_positive(stiffness, "stiffness")
    m = _finite_positive(mass, "mass")
    a = _finite_positive(spacing, "spacing")
    q = np.asarray(wave_vector, dtype=float)
    if q.shape != (3,):
        raise ValueError("wave_vector must have shape (3,)")
    vectors, normals = _bonds(k, a)
    phase = 1.0 - np.cos(vectors @ q)
    return (2.0 * k / m) * (normals * phase[:, None]).T @ normals


def dispersion_frequencies(
    wave_vectors: np.ndarray,
    *,
    stiffness: float,
    mass: float,
    spacing: float,
) -> np.ndarray:
    """Angular frequencies ``omega`` for each ``q``; shape ``(n_q, 3)``, ascending."""
    qs = np.atleast_2d(np.asarray(wave_vectors, dtype=float))
    if qs.shape[1] != 3:
        raise ValueError("wave_vectors must have shape (n, 3)")
    k = _finite_positive(stiffness, "stiffness")
    m = _finite_positive(mass, "mass")
    a = _finite_positive(spacing, "spacing")
    vectors, normals = _bonds(k, a)
    # (n_q, 13) phases, then the 3x3 matrix per q without materialising identities
    phase = 1.0 - np.cos(qs @ vectors.T)                       # (n_q, 13)
    # nn^T blocks are q-independent: build the 13 (3,3) projectors once
    projectors = normals[:, :, None] * normals[:, None, :]     # (13, 3, 3)
    matrices = (2.0 * k / m) * np.einsum("qb,bij->qij", phase, projectors)
    return np.sqrt(np.clip(np.linalg.eigvalsh(matrices), 0.0, None))


def lattice_max_frequency(
    *,
    stiffness: float,
    mass: float,
    spacing: float,
    points_per_axis: int = 33,
) -> tuple[float, np.ndarray]:
    """``(omega_max, q_argmax)`` by sampling the first Brillouin zone.

    The grid spans ``[-pi/a, pi/a]`` in each axis and is centred so that the
    zone-boundary points ``X = (pi/a, 0, 0)`` and the corners are included
    exactly, which is where the maximum of this lattice sits.
    """
    n = int(points_per_axis)
    if n < 3 or n % 2 == 0:
        raise ValueError("points_per_axis must be odd and >= 3 to include pi/a")
    a = _finite_positive(spacing, "spacing")
    axis = np.linspace(-1.0, 1.0, n) * (math.pi / a)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    omega = dispersion_frequencies(grid, stiffness=stiffness, mass=mass, spacing=spacing)
    flat = omega[:, -1]
    index = int(np.argmax(flat))
    return float(flat[index]), grid[index]


def exact_lattice_omega_max(*, stiffness: float, mass: float) -> float:
    """Closed form ``omega_max = sqrt(52 k / (3 m))`` (zone-boundary X, longitudinal)."""
    k = _finite_positive(stiffness, "stiffness")
    m = _finite_positive(mass, "mass")
    return math.sqrt(52.0 * k / (3.0 * m))


def exact_lattice_timestep(*, stiffness: float, mass: float) -> float:
    """Exact semi-implicit Euler limit ``2 / omega_max = sqrt(3 m / (13 k))``."""
    return 2.0 / exact_lattice_omega_max(stiffness=stiffness, mass=mass)


def gershgorin_timestep(
    *, stiffness: float, mass: float, coordination_number: int = 26
) -> float:
    """Row-sum bound already used by the repository: ``2 / sqrt(2 Z k / m)``."""
    k = _finite_positive(stiffness, "stiffness")
    m = _finite_positive(mass, "mass")
    z = int(coordination_number)
    if z < 1:
        raise ValueError("coordination_number must be >= 1")
    return 2.0 / math.sqrt(2.0 * z * k / m)


def rayleigh_timestep(*, radius: float, shear_speed: float,
                      poisson_ratio: float) -> float:
    """Classical rule ``dt = r_min / c_R`` with ``c_R = (c_R/c_S) c_S``."""
    r = _finite_positive(radius, "radius")
    c_s = _finite_positive(shear_speed, "shear_speed")
    return r / (rayleigh_speed_ratio(poisson_ratio) * c_s)


# --------------------------------------------------------------------------- #
# Atomistic control lattice (2x2x2 torus is the smallest that samples X)
# --------------------------------------------------------------------------- #
def periodic_supercell_bonds(
    supercell: Sequence[int] = (4, 4, 4), *, spacing: float
) -> dict[str, Any]:
    """Bond list of the 26-neighbour lattice on a periodic torus, size >= 3.

    Size 3 is the minimum that keeps the ``+n`` and ``-n`` images distinct (on a
    2-cell torus ``(1,1,0)`` and ``(-1,-1,0)`` collapse onto the same node pair
    and the bond set is silently halved). Size 4 is the default because the
    admissible Bloch wavevectors of an ``n = 4`` torus are multiples of
    ``pi/(2a)``, which *include* the zone-boundary point
    ``X = (pi/a, 0, 0)`` where ``omega_max`` lives -- so a 4x4x4 cell reproduces
    the infinite lattice's ``omega_max`` exactly and gives an atomistic reference
    for it. The builder is cross-checked against
    ``born_huang.solver_supercell_bonds`` in the test module.
    """
    a = _finite_positive(spacing, "spacing")
    cell = tuple(int(v) for v in supercell)
    if len(cell) != 3 or any(v < 3 for v in cell):
        raise ValueError("supercell must hold three integers >= 3")
    nx, ny, nz = cell
    offsets = solver_half_space_offsets()

    def node(ijk: Sequence[int]) -> int:
        i, j, k = (int(v) % d for v, d in zip(ijk, cell))
        return (k * ny + j) * nx + i

    positions = np.array(
        [(i * a, j * a, k * a) for k in range(nz) for j in range(ny) for i in range(nx)],
        dtype=float,
    )
    seen: set[tuple[int, int]] = set()
    pairs, vectors = [], []
    for idx in range(nx * ny * nz):
        ijk = (idx % nx, (idx // nx) % ny, idx // (nx * ny))
        for offset in offsets:
            other = node(np.array(ijk) + offset)
            key = (min(idx, other), max(idx, other))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            vectors.append(offset * a)  # minimum image on the torus
    return {
        "positions": positions,
        "pairs": np.array(pairs, dtype=np.int64),
        "vectors": np.array(vectors, dtype=float),
        "cells": nx * ny * nz,
        "bonds_per_cell": len(pairs) / float(nx * ny * nz),
    }


def supercell_hessian(
    supercell: Sequence[int] = (4, 4, 4), *, stiffness: float, spacing: float
) -> np.ndarray:
    """Translational stiffness matrix ``K`` (``3N x 3N``) of the periodic lattice."""
    k = _finite_positive(stiffness, "stiffness")
    bonds = periodic_supercell_bonds(supercell, spacing=spacing)
    n_atoms = bonds["positions"].shape[0]
    hessian = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=float)
    for (i, j), vector in zip(bonds["pairs"], bonds["vectors"]):
        if i == j:  # pragma: no cover - excluded by supercell >= 3
            raise ValueError("self bond on the torus")
        normal = vector / np.linalg.norm(vector)
        block = k * np.outer(normal, normal)
        hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] += block
        hessian[3 * j:3 * j + 3, 3 * j:3 * j + 3] += block
        hessian[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= block
        hessian[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= block
    return hessian


def mode_growth_factor(omega: float, timestep: float) -> float:
    """Per-step spectral radius of semi-implicit Euler for one undamped mode.

    The 2x2 map is ``[[1 - d^2, dt], [-dt omega^2, 1]]`` with ``d = dt*omega``;
    ``trace = 2 - d^2`` and ``det = 1``, so the pair is non-expanding iff
    ``d <= 2`` (repeated root at ``-1`` at ``d = 2``).
    """
    d = float(timestep) * float(omega)
    trace = 2.0 - d * d
    disc = trace * trace - 4.0
    if disc < 0.0:
        return 1.0
    root = math.sqrt(disc)
    return max(abs(0.5 * (trace + root)), abs(0.5 * (trace - root)))


def growth_rate(hessian: np.ndarray, *, mass: float, timestep: float) -> float:
    """Worst per-step growth factor over all modes of ``K`` (exact, no dynamics)."""
    m = _finite_positive(mass, "mass")
    eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    omega = np.sqrt(np.clip(eigenvalues / m, 0.0, None))
    return max(mode_growth_factor(float(w), timestep) for w in omega)


def simulate_periodic_lattice(
    *,
    stiffness: float,
    mass: float,
    spacing: float,
    timestep: float,
    steps: int = 400,
    supercell: Sequence[int] = (4, 4, 4),
    seed: int = 0,
) -> dict[str, Any]:
    """Time-step the *un-damped* linear lattice with the engine's integrator.

    Same semi-implicit (symplectic) Euler as ``IceDEM3D.step`` --
    ``v += dt f / m; x += dt v`` -- applied to a purely linear bond network, so
    the guard of ``DEM3DConfig.recommended_dt`` is deliberately bypassed and the
    threshold itself becomes observable. Returns the maximum displacement per
    step, the fitted per-step growth over the last half of the run, and the
    mechanical energy.
    """
    m = _finite_positive(mass, "mass")
    hessian = supercell_hessian(supercell, stiffness=stiffness, spacing=spacing)
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    n_atoms = hessian.shape[0] // 3
    rng = np.random.default_rng(seed)
    displacement = rng.normal(size=3 * n_atoms) * 1e-9
    velocity = np.zeros(3 * n_atoms)
    history = np.empty(int(steps) + 1)
    energy = np.empty(int(steps) + 1)
    omega = np.sqrt(eigenvalues / m)

    def total_energy() -> float:
        modal = eigenvectors.T @ displacement
        modal_v = eigenvectors.T @ velocity
        return float(0.5 * m * np.sum(modal_v**2) + 0.5 * m * np.sum(omega**2 * modal**2))

    history[0] = float(np.max(np.abs(displacement)))
    energy[0] = total_energy()
    for step in range(1, int(steps) + 1):
        acceleration = -(symmetric @ displacement) / m
        velocity = velocity + float(timestep) * acceleration
        displacement = displacement + float(timestep) * velocity
        history[step] = float(np.max(np.abs(displacement)))
        energy[step] = total_energy()
    half = len(history) // 2
    tail = history[half:]
    if np.all(tail > 0.0) and len(tail) > 2:
        slope = float(np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0])
    else:  # pragma: no cover - only for zero fields
        slope = float("nan")
    return {
        "max_displacement": history,
        "energy": energy,
        "measured_growth_per_step": math.exp(slope),
        "theoretical_growth_per_step": growth_rate(hessian, mass=m, timestep=timestep),
        "omega_max": float(omega.max()),
        "eigenvalues": eigenvalues,
    }


# --------------------------------------------------------------------------- #
# Consolidated audit
# --------------------------------------------------------------------------- #
def timestep_limit_report(
    *,
    bond_stiffness: float = 2_000.0,
    radius: float = 0.025,
    density: float = 917.0,
    mass: float | None = None,
    points_per_axis: int = 33,
) -> dict[str, Any]:
    """Compare the exact lattice limit with every estimate in the repository."""
    from .lattice_elasticity3d import derived_moduli, periodic_elastic_constants

    k = _finite_positive(bond_stiffness, "bond_stiffness")
    r = _finite_positive(radius, "radius")
    rho = _finite_positive(density, "density")
    a = 2.0 * r
    m = float(mass) if mass is not None else rho * (4.0 / 3.0) * math.pi * r**3
    m = _finite_positive(m, "mass")
    rho_lattice = m / a**3

    constants = periodic_elastic_constants(k, a)
    c11, c12, c44 = (constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"])
    moduli = derived_moduli(c11, c12, c44)
    nu = float(moduli["nu"])

    omega_exact = exact_lattice_omega_max(stiffness=k, mass=m)
    omega_bz, q_argmax = lattice_max_frequency(
        stiffness=k, mass=m, spacing=a, points_per_axis=points_per_axis
    )
    hessian = supercell_hessian((4, 4, 4), stiffness=k, spacing=a)
    spectrum = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    if spectrum.min() < -1e-9 * max(1.0, spectrum.max()):  # pragma: no cover
        raise ArithmeticError("periodic Hessian is not positive semi-definite")
    omega_supercell = float(np.sqrt(np.clip(spectrum, 0.0, None) / m).max())

    dt_exact = 2.0 / omega_exact
    dt_gershgorin = gershgorin_timestep(stiffness=k, mass=m, coordination_number=26)
    ratio_r = rayleigh_speed_ratio(nu)

    rayleigh: dict[str, float] = {}
    for label, shear, rho_used in (
        ("voigt_lattice_density", c44, rho_lattice),
        ("voigt_material_density", c44, rho),
        ("reuss_lattice_density", float(moduli["G_reuss_Pa"]), rho_lattice),
    ):
        c_s = shear_wave_speed(shear, rho_used)
        rayleigh[label] = r / (ratio_r * c_s)

    engine_elastic = 0.15 * math.sqrt(m / k)  # DEM3DConfig.recommended_dt, bond-only
    return {
        "bond_stiffness_N_per_m": k,
        "radius_m": r,
        "density_kg_per_m3": rho,
        "mass_kg": m,
        "spacing_m": a,
        "density_lattice_kg_per_m3": rho_lattice,
        "density_lattice_over_material": rho_lattice / rho,
        "C11_Pa": c11,
        "C12_Pa": c12,
        "C44_Pa": c44,
        "poisson_ratio_lattice": nu,
        "zener_anisotropy": float(moduli["zener_anisotropy"]),
        "rayleigh_ratio_cR_over_cS": ratio_r,
        "omega_max_closed_form": omega_exact,
        "omega_max_bz": omega_bz,
        "omega_max_supercell_4x4x4": omega_supercell,
        "q_argmax_over_pi_over_a": [float(v) for v in q_argmax * a / math.pi],
        "dt_exact_s": dt_exact,
        "dt_from_bz_s": 2.0 / omega_bz,
        "dt_from_supercell_s": 2.0 / omega_supercell,
        "dt_gershgorin_s": dt_gershgorin,
        "dt_rayleigh_s": rayleigh,
        "dt_engine_elastic_s": engine_elastic,
        "errors": {
            "omega_closed_vs_bz": (omega_bz - omega_exact) / omega_exact,
            "omega_closed_vs_supercell": (omega_supercell - omega_exact) / omega_exact,
        },
        "ratios": {
            "gershgorin_over_exact": dt_gershgorin / dt_exact,
            **{f"rayleigh_{k2}_over_exact": v / dt_exact for k2, v in rayleigh.items()},
            "engine_elastic_over_exact": engine_elastic / dt_exact,
        },
    }