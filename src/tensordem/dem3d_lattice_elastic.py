"""Analytic and numeric elastic constants of the TensorDEM 3-D bond lattice.

Motivation
----------
``tensordem.dem3d`` documents that its lattice is a simple-cubic network with
face and body diagonals inside the initial cutoff and that its *directional
stiffness must be calibrated before engineering use*. This module replaces that
advice with numbers:

1. :func:`analytic_constants` gives the closed-form (Born / affine) cubic elastic
   tensor of the *exact* bond set used by the solver, derived by a lattice sum
   over the same half-space offset set that ``IceDEM3D`` builds. For bond
   stiffness ``k`` and particle spacing ``a = 2 * radius`` the result is
   ``C11 = 13 k / (3 a)`` and ``C12 = C44 = 7 k / (3 a)``.
2. :func:`affine_patch` re-derives those constants from the solver's own bond
   list and positions on a finite specimen. The estimator restricts the lattice
   sum to particles with a complete 13-bond neighbourhood, so it must reproduce
   the analytic values to machine precision; any mismatch would mean the
   analytic derivation and the solver disagree about bond set or force law.
3. :func:`relaxed_patch` frees the interior degrees of freedom (harmonic solve on
   the same bond network) so the non-affine part of the response is separated
   from the affine one. Central-force lattices without tangential stiffness are
   known to relax strongly in shear; the reported size trend quantifies how much
   of ``C44`` is an affine artifact.
4. :func:`stiffness_calibration` inverts the formula for a target Young's modulus
   and states what a single scalar ``bond_stiffness`` cannot deliver (a free
   Poisson ratio and an isotropic shear modulus).

Scope
-----
Small-strain linear elasticity of the intact bond network: no fracture, no
contacts, no dynamics, no gravity and no free surfaces in the analytic results.
These are verification and calibration numbers, not ice material data and not a
ship-ice validation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "bond_offset_set",
    "spacing",
    "analytic_constants",
    "affine_patch",
    "relaxed_patch",
    "stiffness_calibration",
    "hessian_cross_check",
]


def bond_offset_set() -> list[tuple[int, int, int]]:
    """Return the 13 undirected bond offsets built by ``IceDEM3D``.

    Mirrors ``IceDEM3D.__init__``: all 26 nearest-image offsets with one
    half-space representative kept per undirected bond.
    """
    offsets: list[tuple[int, int, int]] = []
    for dz in range(0, 2):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                if not (dz > 0 or (dz == 0 and dy > 0) or
                        (dz == 0 and dy == 0 and dx > 0)):
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def spacing(radius: float) -> float:
    """Lattice spacing ``a`` of the solver's cubic particle arrangement."""
    return 2.0 * float(radius)


def _offset_tensors(radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.tensor(bond_offset_set(), dtype=torch.float64)
    vectors = offsets * spacing(radius)
    lengths = torch.linalg.vector_norm(vectors, dim=1)
    return vectors, vectors / lengths[:, None]


def analytic_constants(bond_stiffness: float, radius: float) -> dict[str, Any]:
    """Closed-form cubic elastic constants of the intact 3-D bond lattice.

    For a central-force bond network under affine strain the strain energy per
    unit volume is ``U/V = 0.5 * (k/a^3) * sum_S |r|^2 (n.E.n)^2`` with the sum
    taken over the 13 half-space offsets followed by ``IceDEM3D``. Each
    representative stands for two bonds of a particle (``+m`` and ``-m``), so the
    quadratic forms are even in ``n`` and the factor works out exactly; the
    interior coordination number is ``Z = 26``. Therefore

        C11 = (k/a^3) * sum_S |r|^2 n1^4
        C12 = (k/a^3) * sum_S |r|^2 n1^2 n2^2
        C44 = (k/a^3) * sum_S |r|^2 n1^2 n2^2

    The last identity is the Cauchy relation ``C12 = C44`` obeyed by any
    central-force lattice at zero prestress.
    """
    k = float(bond_stiffness)
    a = spacing(radius)
    if not math.isfinite(k) or k <= 0:
        raise ValueError("bond_stiffness must be finite and positive")
    if not math.isfinite(a) or a <= 0:
        raise ValueError("radius must be finite and positive")
    vectors, normals = _offset_tensors(radius)
    weights = vectors.square().sum(dim=1)
    n = normals

    def lattice_sum(alpha: int, beta: int, gamma: int, delta: int) -> float:
        return float((weights * n[:, alpha] * n[:, beta] * n[:, gamma] * n[:, delta]).sum())

    unit = {
        "C11": lattice_sum(0, 0, 0, 0),
        "C12": lattice_sum(0, 0, 1, 1),
        "C44": lattice_sum(0, 1, 0, 1),
        "C22": lattice_sum(1, 1, 1, 1),
        "C33": lattice_sum(2, 2, 2, 2),
        "C13": lattice_sum(0, 0, 2, 2),
        "C23": lattice_sum(1, 1, 2, 2),
        "C66": lattice_sum(0, 2, 0, 2),
    }
    scale = k / a**3
    values = {key: scale * value for key, value in unit.items()}
    c11, c12, c44 = values["C11"], values["C12"], values["C44"]
    young_100 = c11 - 2.0 * c12**2 / (c11 + c12)
    poisson_100 = c12 / (c11 + c12)
    shear_isotropic = young_100 / (2.0 * (1.0 + poisson_100))
    offsets = bond_offset_set()
    return {
        "bond_stiffness_N_per_m": k,
        "radius_m": float(radius),
        "spacing_m": a,
        "coordination_number": 2 * len(offsets),
        "half_space_offset_count": len(offsets),
        "bonds_per_particle": 2 * len(offsets),
        "offsets": [list(offset) for offset in offsets],
        "unit_coefficients": unit,
        "coefficient_over_k_per_a": {key: value / a**2 for key, value in unit.items()},
        "C11_Pa": c11,
        "C22_Pa": values["C22"],
        "C33_Pa": values["C33"],
        "C12_Pa": c12,
        "C13_Pa": values["C13"],
        "C23_Pa": values["C23"],
        "C44_Pa": c44,
        "C66_Pa": values["C66"],
        "cubic_symmetry_relative_deviation": {
            "C22_vs_C11": abs(values["C22"] / c11 - 1.0),
            "C33_vs_C11": abs(values["C33"] / c11 - 1.0),
            "C13_vs_C12": abs(values["C13"] / c12 - 1.0),
            "C23_vs_C12": abs(values["C23"] / c12 - 1.0),
            "C66_vs_C44": abs(values["C66"] / c44 - 1.0),
        },
        "cauchy_relation_relative_deviation": abs(c44 / c12 - 1.0),
        "young_100_Pa": young_100,
        "poisson_100": poisson_100,
        "E_over_k_over_a": young_100 / (k / a),
        "shear_modulus_isotropic_equivalent_Pa": shear_isotropic,
        "zener_anisotropy_C44_over_half_C11_minus_C12": 2.0 * c44 / (c11 - c12),
        "shear_to_isotropic_ratio": c44 / shear_isotropic,
        "model": "affine (Born) central-force lattice sum; intact interior network",
    }


def _unit_axis(index: int) -> torch.Tensor:
    vector = torch.zeros(3, dtype=torch.float64)
    vector[index] = 1.0
    return vector


def _affine_displacement(solver: IceDEM3D, mode: str, strain: float) -> torch.Tensor:
    base = solver.initial_positions
    rel = base - base.amin(dim=0)
    if mode == "xx":
        return strain * rel[:, 0:1] * _unit_axis(0)[None, :]
    if mode == "biaxial":
        return strain * (rel[:, 0:1] * _unit_axis(0)[None, :]
                         + rel[:, 1:2] * _unit_axis(1)[None, :])
    if mode == "shear":
        return strain * rel[:, 1:2] * _unit_axis(0)[None, :]
    raise ValueError(f"unknown affine mode {mode!r}")


def _apply_affine(solver: IceDEM3D, mode: str, strain: float) -> torch.Tensor:
    """Set ``solver.positions`` to the affine image of the rest lattice."""
    displacement = _affine_displacement(solver, mode, strain)
    with torch.no_grad():
        solver.positions.copy_(solver.initial_positions + displacement)
    return displacement


def _mode_constant(mode: str, energy_density: float, strain: float) -> float:
    """Invert ``U/V = 0.5 * C * strain^2`` style relations."""
    if mode in ("xx", "shear"):
        return 2.0 * energy_density / strain**2
    if mode == "biaxial":
        return energy_density / strain**2
    raise ValueError(f"unknown mode {mode!r}")


def _core_mask(positions: torch.Tensor, spacing_m: float) -> torch.Tensor:
    """Particles whose full 26-bond neighbourhood lies inside the specimen."""
    low = positions.amin(dim=0)
    high = positions.amax(dim=0)
    tolerance = 1e-9 * spacing_m
    mask = torch.ones(len(positions), dtype=torch.bool)
    for axis in range(3):
        mask &= positions[:, axis] > low[axis] + spacing_m - tolerance
        mask &= positions[:, axis] < high[axis] - spacing_m + tolerance
    return mask


def _core_energy_density(solver: IceDEM3D, displacement: torch.Tensor,
                         spacing_m: float) -> tuple[float, int]:
    """Interior energy per unit volume, using the solver's own bond bookkeeping.

    Restricted to particles with a complete neighbour shell, so the
    bonds-per-volume ratio is exactly the infinite-lattice value; each bond is
    weighted by one half per endpoint to avoid double counting.
    """
    base = solver.initial_positions
    core = _core_mask(base, spacing_m)
    count = int(core.sum().item())
    if count == 0:
        raise ValueError("specimen too small for a core estimator; increase nx/ny/nz")
    i, j = solver.pairs.T
    extension = ((displacement[j] - displacement[i]) * solver.initial_normals).sum(dim=1)
    energy = 0.5 * solver.config.bond_stiffness * extension.square()
    share = 0.5 * torch.where(core[i], energy, torch.zeros_like(energy))
    share = share + 0.5 * torch.where(core[j], energy, torch.zeros_like(energy))
    return float(share.sum()) / (count * spacing_m**3), count


def affine_patch(config: DEM3DConfig, *, strain: float = 1e-6,
                 modes: Sequence[str] = ("xx", "biaxial", "shear")) -> dict[str, Any]:
    """Measure affine (Voigt) elastic constants from the solver's bond network."""
    solver = IceDEM3D(config)
    a = spacing(config.radius)
    measured: dict[str, float] = {}
    for mode in modes:
        displacement = _apply_affine(solver, mode, strain)
        density, count = _core_energy_density(solver, displacement, a)
        measured[mode] = _mode_constant(mode, density, strain)
    with torch.no_grad():
        solver.positions.copy_(solver.initial_positions)
    analytic = analytic_constants(config.bond_stiffness, config.radius)
    results: dict[str, Any] = {
        "strain": strain,
        "core_particles": count,
        "particle_count": int(len(solver.positions)),
        "bond_count": int(len(solver.pairs)),
        "measured": {},
        "analytic": {},
        "relative_error": {},
    }
    if "xx" in measured:
        results["measured"]["C11_Pa"] = measured["xx"]
        results["analytic"]["C11_Pa"] = analytic["C11_Pa"]
        results["relative_error"]["C11"] = abs(measured["xx"] / analytic["C11_Pa"] - 1.0)
    if "biaxial" in measured and "xx" in measured:
        c12 = measured["biaxial"] - measured["xx"]
        results["measured"]["C12_Pa"] = c12
        results["analytic"]["C12_Pa"] = analytic["C12_Pa"]
        results["relative_error"]["C12"] = abs(c12 / analytic["C12_Pa"] - 1.0)
    if "shear" in measured:
        results["measured"]["C44_Pa"] = measured["shear"]
        results["analytic"]["C44_Pa"] = analytic["C44_Pa"]
        results["relative_error"]["C44"] = abs(measured["shear"] / analytic["C44_Pa"] - 1.0)
    results["max_relative_error"] = (max(results["relative_error"].values())
                                     if results["relative_error"] else math.nan)
    return results


def _assemble_stiffness(solver: IceDEM3D) -> torch.Tensor:
    """Linear-elastic harmonic stiffness matrix of the intact bond network."""
    n = len(solver.positions)
    k = solver.config.bond_stiffness
    i, j = solver.pairs.T
    normal = solver.initial_normals
    block = k * normal[:, :, None] * normal[:, None, :]
    rows_i = 3 * i[:, None] + torch.arange(3)[None, :]
    rows_j = 3 * j[:, None] + torch.arange(3)[None, :]
    matrix = torch.zeros((3 * n, 3 * n), dtype=solver.dtype)
    for row, col, sign in ((rows_i, rows_i, 1.0), (rows_j, rows_j, 1.0),
                           (rows_i, rows_j, -1.0), (rows_j, rows_i, -1.0)):
        flat_rows = row[:, :, None].expand(-1, 3, 3).reshape(-1)
        flat_cols = col[:, None, :].expand(-1, 3, 3).reshape(-1)
        matrix.index_put_((flat_rows, flat_cols), sign * block.reshape(-1), accumulate=True)
    return matrix


def _boundary_mask(positions: torch.Tensor, spacing_m: float, boundary: str) -> torch.Tensor:
    low = positions.amin(dim=0)
    high = positions.amax(dim=0)
    tolerance = 1e-9 * spacing_m
    mask = torch.zeros(len(positions), dtype=torch.bool)
    axes = {"x_faces": (0,), "yz_faces": (1,), "all_faces": (0, 1, 2)}[boundary]
    for axis in axes:
        mask |= (positions[:, axis] - low[axis]).abs() < tolerance
        mask |= (positions[:, axis] - high[axis]).abs() < tolerance
    return mask


def relaxed_patch(config: DEM3DConfig, *, strain: float = 1e-6, mode: str = "xx",
                  boundary: str = "x_faces") -> dict[str, Any]:
    """Relaxed small-strain constant with the interior degrees of freedom free.

    Solves ``K_ff u_f = -K_fb u_b`` with the requested faces kinematically driven
    by the affine field. The constant uses the same core energy-density estimator
    as :func:`affine_patch`, so affine and relaxed values are directly comparable.
    """
    solver = IceDEM3D(config)
    a = spacing(config.radius)
    n = len(solver.positions)
    if n > 4000:
        raise ValueError("dense relaxation limited to 4000 particles")
    matrix = _assemble_stiffness(solver)
    prescribed = _apply_affine(solver, mode, strain).reshape(-1).clone()
    mask = _boundary_mask(solver.initial_positions, a, boundary)
    nodes = torch.nonzero(mask, as_tuple=False).flatten()
    boundary_dofs = (3 * nodes[:, None] + torch.arange(3)[None, :]).reshape(-1)
    free = torch.ones(3 * n, dtype=torch.bool)
    free[boundary_dofs] = False
    if boundary == "x_faces":
        # Remove the remaining transverse rigid-body translation.
        left = torch.nonzero(mask & (solver.initial_positions[:, 0]
                                     - solver.initial_positions[:, 0].min()).abs().lt(1e-9 * a),
                             as_tuple=False).flatten()[0]
        for axis in (1, 2):
            extra = int(3 * left + axis)
            free[extra] = False
            boundary_dofs = torch.cat((boundary_dofs, torch.tensor([extra])))
    free_dofs = torch.nonzero(free, as_tuple=False).flatten()
    displacement = torch.zeros(3 * n, dtype=solver.dtype)
    displacement[boundary_dofs] = prescribed[boundary_dofs]
    kff = matrix[free_dofs[:, None], free_dofs[None, :]]
    kfb = matrix[free_dofs[:, None], boundary_dofs[None, :]]
    displacement[free_dofs] = torch.linalg.solve(kff, -(kfb @ displacement[boundary_dofs]))
    residual = float((matrix @ displacement)[free_dofs].abs().max().item())
    planar = displacement.reshape(-1, 3)
    density, count = _core_energy_density(solver, planar, a)
    constant = _mode_constant(mode, density, strain)
    # Affine reference on the *same* specimen and estimator: the variational
    # bound (relaxed energy <= affine energy at equal prescribed strain) is only
    # meaningful against this number, not against the long-wavelength limit.
    affine_displacement = _affine_displacement(solver, mode, strain)
    affine_density, _ = _core_energy_density(solver, affine_displacement, a)
    affine_same = _mode_constant(mode, affine_density, strain)
    analytic = analytic_constants(config.bond_stiffness, config.radius)
    result: dict[str, Any] = {
        "mode": mode,
        "boundary": boundary,
        "strain": strain,
        "particle_count": n,
        "core_particles": count,
        "boundary_nodes": int(len(nodes)),
        "relaxed_constant_Pa": constant,
        "affine_same_specimen_Pa": affine_same,
        "ratio_to_affine_same_specimen": constant / affine_same,
        "analytic_laterally_constrained_Pa": (analytic["C11_Pa"] if mode == "xx" else analytic["C44_Pa"]),
        "free_dof_residual_N": residual,
        "free_dof_count": int(len(free_dofs)),
        "max_free_displacement_m": float(planar[free_dofs // 3].abs().max().item()),
    }
    if mode == "xx" and boundary == "x_faces":
        low = solver.initial_positions.amin(dim=0)
        high = solver.initial_positions.amax(dim=0)
        tol = 1e-9 * a
        top = (solver.initial_positions[:, 1] - high[1]).abs() < tol
        bottom = (solver.initial_positions[:, 1] - low[1]).abs() < tol
        height = float(high[1] - low[1])
        transverse = float(planar[top, 1].mean() - planar[bottom, 1].mean()) / height
        result["poisson_ratio_relaxed"] = -transverse / strain
    if mode == "xx":
        result["young_modulus_relaxed_Pa"] = constant
        # Long-wavelength limit of a traction-free lateral-surface tension test.
        result["analytic_reference_young_100_Pa"] = analytic["young_100_Pa"]
        result["young_ratio_to_analytic_E100"] = constant / analytic["young_100_Pa"]
    if mode == "shear":
        result["shear_stiffness_relaxed_Pa"] = constant
        result["shear_ratio_to_analytic_C44"] = constant / analytic["C44_Pa"]
    with torch.no_grad():
        solver.positions.copy_(solver.initial_positions)
    return result


def stiffness_calibration(target_young_modulus: float, radius: float,
                          target_poisson: float = 0.3) -> dict[str, Any]:
    """Invert the analytic formula and state the unreachable targets.

    A single scalar ``bond_stiffness`` fixes both the modulus and the Poisson
    ratio, because the lattice elastic tensor is proportional to ``k``. This
    returns the stiffness that matches the requested modulus and the resulting
    mismatch on Poisson ratio and isotropic shear modulus.
    """
    unit = analytic_constants(1.0, radius)
    a = unit["spacing_m"]
    coefficient = unit["E_over_k_over_a"]  # E100 / (k/a)
    k = float(target_young_modulus) * a / coefficient
    solved = analytic_constants(k, radius)
    shear_required = float(target_young_modulus) / (2.0 * (1.0 + target_poisson))
    return {
        "target_young_modulus_Pa": float(target_young_modulus),
        "target_poisson": float(target_poisson),
        "radius_m": float(radius),
        "spacing_m": a,
        "required_bond_stiffness_N_per_m": k,
        "achieved_young_100_Pa": solved["young_100_Pa"],
        "young_relative_error": abs(solved["young_100_Pa"] / float(target_young_modulus) - 1.0),
        "achieved_poisson_100": solved["poisson_100"],
        "poisson_absolute_error": abs(solved["poisson_100"] - float(target_poisson)),
        "poisson_is_stiffness_independent": True,
        "achieved_shear_modulus_Pa": solved["C44_Pa"],
        "isotropic_shear_modulus_for_target_Pa": shear_required,
        "shear_relative_error": abs(solved["C44_Pa"] / shear_required - 1.0),
        "zener_anisotropy": solved["zener_anisotropy_C44_over_half_C11_minus_C12"],
        "note": ("Scalar bond_stiffness sets E and leaves nu locked at the lattice "
                 "value 0.35; the Cauchy-constrained C44=7k/(3a) cannot match an "
                 "isotropic shear modulus. Both need tangential/angular stiffness."),
    }

def hessian_cross_check(config: DEM3DConfig, *, amplitude_ratio: float = 1e-5,
                        seed: int = 20261101) -> dict[str, Any]:
    """Compare the assembled harmonic stiffness with the solver's force law.

    Applies a deterministic pseudo-random displacement field of amplitude
    ``amplitude_ratio * spacing`` and checks both the bond energy and the force
    vector against ``0.5 * u^T K u`` and ``-K u``. This ties the linear operator
    used by :func:`relaxed_patch` to the solver's own bond evaluation, so a change
    in one without the other is detected.
    """
    solver = IceDEM3D(config)
    a = spacing(config.radius)
    displacement = torch.randn(solver.positions.shape, generator=torch.Generator().manual_seed(seed),
                               dtype=solver.dtype) * (amplitude_ratio * a)
    matrix = _assemble_stiffness(solver)
    flat = displacement.reshape(-1)
    quadratic = 0.5 * float(flat @ (matrix @ flat))
    with torch.no_grad():
        solver.positions.copy_(solver.initial_positions + displacement)
    diagnostics = solver.diagnostics()
    force, _ = solver.forces(update_fracture=False)
    solver_energy = float(diagnostics["bond_elastic_energy_J"])
    solver_force = force.reshape(-1)
    linear_force = -(matrix @ flat)
    with torch.no_grad():
        solver.positions.copy_(solver.initial_positions)
    energy_scale = max(abs(quadratic), 1e-30)
    force_scale = max(float(solver_force.abs().max()), 1e-30)
    return {
        "amplitude_ratio_of_spacing": amplitude_ratio,
        "quadratic_form_energy_J": quadratic,
        "solver_bond_energy_J": solver_energy,
        "energy_relative_difference": abs(solver_energy / energy_scale - 1.0),
        "force_relative_difference": float((solver_force - linear_force).abs().max()) / force_scale,
        "criterion": "relative differences below 1e-4 are consistent with linearisation error",
        "linearisation_error_estimate": float(amplitude_ratio) ** 2,
        "pass": (abs(solver_energy / energy_scale - 1.0) < 1e-4
                 and float((solver_force - linear_force).abs().max()) / force_scale < 1e-3),
    }
