"""Affine (Cauchy-Born) elasticity of the DEM3D bonded-sphere lattice.

Why this module exists
----------------------
``IceDEM3D`` builds a simple-cubic lattice with face and body diagonal bonds
inside the initial cutoff (a 13-neighbour network). Its docstring states:

    "calibrate its directional stiffness before engineering use"

so far without a reference value to calibrate against.  This module supplies
that reference: the *affine* (uniform-strain, Cauchy-Born) elastic stiffness
tensor of the bond network, which is the exact small-strain response of the
lattice when every bond deforms with the macroscopic strain field.

Conventions
-----------
Small-strain linear elasticity, SI units, isotropic (Voigt) notation for shear:
``strain_voigt = [e11, e22, e33, g12, g13, g23]`` with engineering shear
``g_ij = 2 e_ij``.  Then ``stress_voigt = C @ strain_voigt`` with
``stress_voigt = [s11, s22, s33, s12, s13, s23]`` and the strain energy density
is ``u = 0.5 * strain_voigt @ C @ strain_voigt``.  With this convention ``C`` is
symmetric.

The affine stiffness of a central-force bond network (Born & Huang, *Dynamical
Theory of Crystal Lattices*, 1954; standard Cauchy-Born result) is

    C_ijkl = (1/V) * sum_b k_b * l_b**2 * n_i n_j n_k n_l

where the sum runs over the bonds of *one* unit cell of volume ``V``, ``k_b`` is
the bond axial stiffness, ``l_b`` the bond rest length and ``n_b`` its unit
direction.  For a central-force network with no internal pre-stress this
immediately implies the Cauchy relation ``C12 = C44`` (in reduced cubic
notation), which this module exposes as an exact manufactured invariant.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

__all__ = [
    "VOIGT_PAIRS",
    "voigt_strain_basis",
    "affine_stiffness_tensor",
    "stress_under_strain",
    "compliance_tensor",
    "directional_young_modulus",
    "cubic_constants",
    "zener_ratio",
    "voigt_reuss_hill",
    "isotropic_moduli",
    "ideal_box_bonds",
    "lattice_bond_table",
    "analytic_box_constants",
    "finite_lattice_bonds",
    "affine_stiffness_from_bonds",
    "affine_stiffness_from_solver",
    "solver_virial_stress",
    "max_phonon_frequency",
]

DTYPE = torch.float64

#: Voigt index -> (i, j) tensor component (0-based).
VOIGT_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def voigt_strain_basis() -> torch.Tensor:
    """Return the six symmetric basis strains ``E_J`` of shape ``(6, 3, 3)``.

    Normals (``J < 3``) are unit tensor components; shears (``J >= 3``) carry
    ``e_ij = e_ji = 0.5`` so that the corresponding engineering shear is 1.
    Because ``stress(E_J) = C[:, J]``, applying these modes recovers the Voigt
    matrix column by column.
    """
    basis = torch.zeros((6, 3, 3), dtype=DTYPE)
    for index, (i, j) in enumerate(VOIGT_PAIRS):
        if i == j:
            basis[index, i, i] = 1.0
        else:
            basis[index, i, j] = 0.5
            basis[index, j, i] = 0.5
    return basis


def stress_under_strain(
    normals: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: float | torch.Tensor,
    strains: torch.Tensor,
    volume: float,
) -> torch.Tensor:
    """Return the affine Cauchy stress for each strain mode.

    Args:
        normals: ``(B, 3)`` unit bond directions.
        rest_lengths: ``(B,)`` bond rest lengths, m.
        stiffness: scalar or ``(B,)`` axial bond stiffness, N/m.
        strains: ``(6, 3, 3)`` (or ``(M, 6, 3, 3)``) symmetric strain modes.
        volume: reference volume associated with ``normals``, m^3.

    Returns:
        ``(6, 6)`` (or ``(M, 6, 6)``) stress components in Voigt order, Pa.
    """
    if volume <= 0:
        raise ValueError("volume must be positive")
    normals = normals.to(DTYPE)
    rest_lengths = rest_lengths.to(DTYPE)
    k = torch.as_tensor(stiffness, dtype=DTYPE)
    if k.ndim == 0:
        k = k.expand(rest_lengths.shape[0])
    if normals.shape[0] != rest_lengths.shape[0] or k.shape[0] != normals.shape[0]:
        raise ValueError("normals, rest_lengths and stiffness must agree in bond count")
    strains = strains.to(DTYPE)
    if strains.shape[-3:] != (6, 3, 3):
        raise ValueError("strains must have trailing shape (6, 3, 3)")

    # elongation of bond b: dl_b = l_b * (n . e . n)
    ne = torch.einsum("bi,...ij,bj->...b", normals, strains, normals)
    # k * dl * l with dl = l * (n . e . n), i.e. the k l^2 prefactor of C.
    weight = k * rest_lengths.square()
    axial = weight[None, :].expand_as(ne) * ne  # k * l^2 * (n . e . n)
    outer = normals[:, :, None] * normals[:, None, :]  # (B, 3, 3)
    sigma = torch.einsum("...b,bij->...ij", axial, outer) / volume

    out = torch.empty(strains.shape[:-2] + (6,), dtype=DTYPE)
    for index, (i, j) in enumerate(VOIGT_PAIRS):
        out[..., index] = sigma[..., i, j]
    return out


def affine_stiffness_tensor(
    normals: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: float | torch.Tensor,
    volume: float,
) -> torch.Tensor:
    """Cauchy-Born stiffness matrix ``C`` of shape ``(6, 6)`` in Pa."""
    return stress_under_strain(normals, rest_lengths, stiffness, voigt_strain_basis(), volume)


def compliance_tensor(voigt_compliance: torch.Tensor) -> torch.Tensor:
    """Expand a Voigt compliance matrix to the full ``(3, 3, 3, 3)`` tensor.

    Inverse of the Voigt contraction used for ``C``: a component with ``m``
    shear indices among its two Voigt slots is divided by ``2**m`` so that
    ``e_ij = S_ijkl * s_kl`` reproduces ``strain_voigt = S @ stress_voigt``.
    """
    if tuple(voigt_compliance.shape) != (6, 6):
        raise ValueError("voigt_compliance must have shape (6, 6)")
    out = torch.zeros((3, 3, 3, 3), dtype=DTYPE)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, l) in enumerate(VOIGT_PAIRS):
            shear_slots = int(i != j) + int(k != l)
            value = voigt_compliance[a, b] / (2.0 ** shear_slots)
            # distribute over the symmetric permutations of (ij) and (kl)
            perms = {(i, j), (j, i)} if i != j else {(i, j)}
            perms_k = {(k, l), (l, k)} if k != l else {(k, l)}
            for p in perms:
                for q in perms_k:
                    out[p[0], p[1], q[0], q[1]] += value
    return out


def directional_young_modulus(stiffness: torch.Tensor, direction: Sequence[float]) -> float:
    """Young's modulus along ``direction`` from the compliance tensor, Pa.

    ``1/E(d) = S_ijkl d_i d_j d_k d_l`` with ``S = C^-1``.
    """
    d = torch.as_tensor(list(direction), dtype=DTYPE)
    norm = torch.linalg.vector_norm(d)
    if float(norm) == 0.0:
        raise ValueError("direction must be non-zero")
    d = d / norm
    compliance = torch.linalg.inv(stiffness)
    tensor = compliance_tensor(compliance)
    inverse_e = torch.einsum("i,j,k,l,ijkl->", d, d, d, d, tensor)
    return float(1.0 / inverse_e)


def cubic_constants(stiffness: torch.Tensor) -> dict[str, float]:
    """Return the reduced cubic constants and the off-pattern residual.

    ``residual`` is the largest magnitude among entries that must vanish for
    cubic (m3m) symmetry, normalised by ``max|C|``.  ``symmetry_residual``
    reports ``max|C - C^T| / max|C|``.
    """
    if tuple(stiffness.shape) != (6, 6):
        raise ValueError("stiffness must have shape (6, 6)")
    scale = float(stiffness.abs().max())
    pattern = {
        (0, 0), (1, 1), (2, 2),  # C11
        (0, 1), (0, 2), (1, 2),  # C12
        (3, 3), (4, 4), (5, 5),  # C44
    }
    residual = 0.0
    for a in range(6):
        for b in range(6):
            if (a, b) not in pattern and (b, a) not in pattern:
                residual = max(residual, abs(float(stiffness[a, b])))
    c11 = float((stiffness[0, 0] + stiffness[1, 1] + stiffness[2, 2]) / 3.0)
    c12 = float((stiffness[0, 1] + stiffness[0, 2] + stiffness[1, 2]) / 3.0)
    c44 = float((stiffness[3, 3] + stiffness[4, 4] + stiffness[5, 5]) / 3.0)
    return {
        "C11": c11,
        "C12": c12,
        "C44": c44,
        "off_pattern_residual": residual / scale if scale else 0.0,
        "symmetry_residual": float((stiffness - stiffness.T).abs().max()) / scale if scale else 0.0,
    }


def zener_ratio(c11: float, c12: float, c44: float) -> float:
    """Zener anisotropy ratio ``A = 2 C44 / (C11 - C12)`` (1 for isotropy)."""
    denominator = c11 - c12
    if denominator == 0.0:
        raise ValueError("C11 - C12 must be non-zero")
    return 2.0 * c44 / denominator


def voigt_reuss_hill(c11: float, c12: float, c44: float) -> dict[str, float]:
    """Voigt (uniform strain) and Reuss (uniform stress) shear-modulus bounds."""
    bulk = (c11 + 2.0 * c12) / 3.0
    g_voigt = (c11 - c12 + 3.0 * c44) / 5.0
    g_reuss = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    return {
        "bulk": bulk,
        "shear_voigt": g_voigt,
        "shear_reuss": g_reuss,
        "shear_hill": 0.5 * (g_voigt + g_reuss),
    }


def isotropic_moduli(c11: float, c12: float, c44: float) -> dict[str, float]:
    """Hill-averaged isotropic moduli and the [100]/[111] Young moduli.

    ``E_100`` and ``nu_100`` follow from the exact cubic compliance along a
    cube axis; ``E_111`` is the body-diagonal value, so ``E_111 / E_100``
    quantifies the lattice anisotropy that biases specimen-direction results.
    """
    s11 = (c11 + c12) / ((c11 - c12) * (c11 + 2.0 * c12))
    s12 = -c12 / ((c11 - c12) * (c11 + 2.0 * c12))
    s44 = 1.0 / c44
    inverse_e111 = s11 - 2.0 * (s11 - s12 - 0.5 * s44) * (1.0 / 3.0)
    bounds = voigt_reuss_hill(c11, c12, c44)
    shear = bounds["shear_hill"]
    bulk = bounds["bulk"]
    return {
        "E_100": 1.0 / s11,
        "nu_100": -s12 / s11,
        "E_111": 1.0 / inverse_e111,
        "E_111_over_E_100": (1.0 / inverse_e111) / (1.0 / s11),
        "G_hill": shear,
        "K_hill": bulk,
        "E_hill_isotropic": 9.0 * shear * bulk / (3.0 * bulk + shear),
        "nu_hill_isotropic": (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear)),
    }


# ---------------------------------------------------------------------------
# Lattice bond tables
# ---------------------------------------------------------------------------

#: The 13 neighbour classes of the DEM3D box lattice, as integer offsets.
#: The solver keeps one half-space per pair, which is exactly this list.
IDEAL_OFFSETS: tuple[tuple[int, int, int], ...] = (
    # 3 axis neighbours
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    # 6 face diagonals
    (1, 1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
    # 4 body diagonals
    (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
)


def ideal_box_bonds(spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normals and rest lengths of the 13 ideal neighbour classes.

    ``spacing`` is the grid pitch (the solver uses ``2 * radius``).
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    vectors = torch.tensor(IDEAL_OFFSETS, dtype=DTYPE) * spacing
    lengths = torch.linalg.vector_norm(vectors, dim=1)
    return vectors / lengths[:, None], lengths


def lattice_bond_table(spacing: float) -> list[dict[str, Any]]:
    """Human-readable table of the ideal neighbour classes (counts and lengths)."""
    rows: list[dict[str, Any]] = []
    for offset in IDEAL_OFFSETS:
        length = spacing * math.sqrt(sum(component * component for component in offset))
        kind = "axis" if sum(abs(c) for c in offset) == 1 else (
            "face-diagonal" if sum(abs(c) for c in offset) == 2 else "body-diagonal")
        rows.append({
            "offset": list(offset),
            "kind": kind,
            "rest_length_m": length,
            "rest_length_over_spacing": length / spacing,
        })
    return rows


def analytic_box_constants(bond_stiffness: float, spacing: float) -> dict[str, float]:
    """Closed-form cubic constants of the ideal 13-neighbour box lattice.

    Summing ``C_ijkl = (1/V) sum_b k l_b^2 n_i n_j n_k n_l`` over the 13
    classes of one cell (``V = spacing**3``) and using ``l = a, a*sqrt(2),
    a*sqrt(3)`` for axis / face-diagonal / body-diagonal bonds gives

        C11 = k/a * [ 1 + 2*4*(1/4) + 3*4*(1/9) ]    = (1 + 2 + 4/3) k/a = 13/3 k/a
        C12 = k/a * [ 0 + 2*2*(1/4) + 3*4*(1/9) ]    = (1 + 4/3) k/a     =  7/3 k/a
        C44 = C12                                    =  7/3 k/a

    accounting for the 3 axis, 6 face-diagonal and 4 body-diagonal classes
    (``n_i^4`` and ``n_i^2 n_j^2`` weights respectively).  ``C44 == C12`` is the
    Cauchy relation, exact for any central-force network without pre-stress.
    Dropping the diagonal bonds would give ``C44 = 0``: the simple-cubic
    nearest-neighbour lattice is shear-unstable, which is why the diagonals exist.
    """
    if bond_stiffness <= 0 or spacing <= 0:
        raise ValueError("bond_stiffness and spacing must be positive")
    unit = bond_stiffness / spacing
    return {
        "C11": 13.0 / 3.0 * unit,
        "C12": 7.0 / 3.0 * unit,
        "C44": 7.0 / 3.0 * unit,
        "unit": unit,
        "spacing": spacing,
        "bond_stiffness": bond_stiffness,
    }


def finite_lattice_bonds(nx: int, ny: int, nz: int, spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Bonds of a finite ``nx x ny x nz`` box lattice, same rule as ``IceDEM3D``.

    Returns bond vectors ``r_b`` of shape ``(B, 3)`` and their lengths.  The
    half-space selection ``dz > 0 or (dz == 0 and dy > 0) or (dz == dy == 0 and
    dx > 0)`` mirrors the solver so the bond count can be cross-checked against
    ``IceDEM3D.pairs``.
    """
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    vectors: list[tuple[float, float, float]] = []
    for dz in range(0, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == dy == dz == 0:
                    continue
                if not (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0)):
                    continue
                for z in range(nz):
                    if not 0 <= z + dz < nz:
                        continue
                    for y in range(ny):
                        if not 0 <= y + dy < ny:
                            continue
                        for x in range(nx):
                            if not 0 <= x + dx < nx:
                                continue
                            vectors.append((dx * spacing, dy * spacing, dz * spacing))
    bonds = torch.tensor(vectors, dtype=DTYPE).reshape(-1, 3)
    return bonds, torch.linalg.vector_norm(bonds, dim=1)


def affine_stiffness_from_bonds(
    bonds: torch.Tensor,
    bond_stiffness: float,
    volume: float,
) -> torch.Tensor:
    """Affine stiffness from raw bond vectors."""
    lengths = torch.linalg.vector_norm(bonds, dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("bond vectors must be non-zero")
    return affine_stiffness_tensor(bonds / lengths[:, None], lengths, bond_stiffness, volume)


def affine_stiffness_from_solver(sim: Any) -> torch.Tensor:
    """Affine stiffness of the bond network actually built by ``IceDEM3D``.

    Uses the solver's own bond pairs, reference vectors and reference volume, so
    the result reflects the finite specimen (including missing surface bonds)
    rather than the ideal infinite lattice.
    """
    cfg = sim.config
    spacing = 2.0 * cfg.radius
    volume = float(cfg.nx * cfg.ny * cfg.nz) * spacing ** 3
    bonds = sim.reference_bond_vectors
    return affine_stiffness_from_bonds(bonds, cfg.bond_stiffness, volume)


@torch.no_grad()
def solver_virial_stress(sim: Any, strain: torch.Tensor) -> torch.Tensor:
    """Cauchy stress (Voigt, Pa) of the solver bond network under affine strain.

    The positions are mapped affinely as ``x -> (I + e) x`` about the specimen
    centroid, all velocities are zero, and the stress is taken from the solver's
    own force evaluation through the virial sum ``sigma_ij = (1/V) sum_p f_i x_j``.
    This is an independent route to ``C``: it exercises ``IceDEM3D.forces`` and
    the finite-difference-free bond law, not the ``k l^2 n^4`` sum.
    """
    cfg = sim.config
    if cfg.top_platen_enabled or cfg.bottom_platen_enabled:
        raise ValueError("solver_virial_stress requires platens disabled")
    spacing = 2.0 * cfg.radius
    volume = float(cfg.nx * cfg.ny * cfg.nz) * spacing ** 3
    reference = sim.initial_positions.clone()
    centroid = reference.mean(dim=0, keepdim=True)
    state = sim.state_dict()
    tool_start = sim.tool_start.clone()
    try:
        # Park the indenter far away: an indenter that touches the specimen would
        # add its contact forces to the virial and bias the measured stiffness.
        sim.tool_start.fill_(1.0e6)
        sim.velocities.zero_()
        sim.alive.copy_(sim.bonded)
        sim.positions.copy_((reference - centroid) @ (torch.eye(3, dtype=sim.dtype) + strain) + centroid)
        forces, _ = sim.forces(update_fracture=False)
        offsets = sim.positions - sim.positions.mean(dim=0, keepdim=True)
        # The solver applies +k*dl*n on particle i and -k*dl*n on particle j,
        # so the tensile virial carries an overall minus sign.
        sigma = -(offsets[:, :, None] * forces[:, None, :]).sum(dim=0) / volume
    finally:
        sim.load_state_dict(state)
        sim.tool_start.copy_(tool_start)
    out = torch.empty(6, dtype=DTYPE)
    for index, (i, j) in enumerate(VOIGT_PAIRS):
        out[index] = sigma[i, j]
    return out


def max_phonon_frequency(
    normals: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: float | torch.Tensor,
    mass: float,
    q_samples: int = 21,
    spacing: float = 1.0,
) -> float:
    """Largest phonon angular frequency of the ideal lattice over a q grid.

    ``D(q) = (2/m) sum_b k_b (1 - cos(q . r_b)) n_b (x) n_b`` over the 13
    half-space bond classes ``b``; the factor 2 accounts for the ``-r_b`` partner
    of every class, which carries the same ``1 - cos`` weight but appears only
    once in the half-space list.  (Omitting it halves ``D`` and under-reports
    ``omega_max`` by ``sqrt(2)``, which in turn over-states the stability bound
    ``dt_crit = 2 / omega_max``.)  The largest eigenvalue over the sampled
    Brillouin zone gives ``omega_max``, hence ``dt_crit``.
    """
    if mass <= 0:
        raise ValueError("mass must be positive")
    if q_samples < 3:
        raise ValueError("q_samples must be >= 3")
    k = torch.as_tensor(stiffness, dtype=DTYPE)
    if k.ndim == 0:
        k = k.expand(rest_lengths.shape[0])
    vectors = normals * rest_lengths[:, None]
    outer = normals[:, :, None] * normals[:, None, :]
    weighted = outer * k[:, None, None]
    grid = torch.linspace(-math.pi / spacing, math.pi / spacing, q_samples, dtype=DTYPE)
    qz, qy, qx = torch.meshgrid(grid, grid, grid, indexing="ij")
    qs = torch.stack((qx, qy, qz), dim=-1).reshape(-1, 3)
    phase = qs @ vectors.T  # (Q, B)
    # Each half-space class stands for a +/- pair of bonds with identical weight.
    weights = 2.0 * (1.0 - torch.cos(phase))  # (Q, B)
    dyn = torch.einsum("qb,bij->qij", weights, weighted) / mass
    eigenvalues = torch.linalg.eigvalsh(dyn)
    if bool((eigenvalues < -1e-10 * eigenvalues.abs().max().clamp_min(1e-30)).any()):
        raise ValueError("dynamical matrix has negative eigenvalues (unstable lattice)")
    return float(torch.sqrt(eigenvalues.clamp_min(0.0).max()))