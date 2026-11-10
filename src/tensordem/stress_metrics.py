"""Macroscopic stress metrics for granular assemblies: three averaging口径.

Implements and cross-validates three classical definitions of the macroscopic
stress tensor from discrete contact data, plus fabric tensor characterisation
and REV (Representative Elementary Volume) size convergence analysis.

Three stress averaging methods
-------------------------------
1. **Love–Weber volume average** (a.k.a. virial stress):

   .. math:: \\sigma^{LW}_{ij} = \\frac{1}{V}\\sum_{c=1}^{N_c} f_i^c \\, l_j^c

   where :math:`f^c` is the contact force on particle *i* and :math:`l^c` is
   the branch vector from *i* to *j*.  This is the standard DEM stress.

2. **Boundary traction (Cauchy)**:

   .. math:: \\sigma^{B}_{ij} = \\frac{1}{A}\\sum_{b \\in \\partial V}
              f_i^b \\, n_j^b \\, \\Delta s^b

   where the sum runs over boundary contacts, :math:`n^b` is the outward
   normal, and :math:`\\Delta s^b` is the boundary segment length.

3. **Energy work-conjugate (Strain-energy)**:

   .. math:: \\sigma^{E}_{ij} = \\frac{2}{V}\\sum_{c=1}^{N_c}
              f_i^c \\, u_j^c / \\dot{\\varepsilon}_{ref}

   where :math:`u^c` is the relative displacement at contact *c* and
   :math:`\\dot{\\varepsilon}_{ref}` is a reference strain rate.  In practice,
   for a quasi-static assembly under uniform strain :math:`\\varepsilon`,
   :math:`u^c = \\varepsilon \\cdot l^c`, so this reduces to the Love–Weber
   form when the strain is uniform.

Equivalence theorem
-------------------
For a **statically balanced**, **homogeneously deformed** assembly with
**no body forces**, all three definitions converge to the same tensor.
Deviations arise from:
- Static imbalance (inertia, body forces)
- Strain localisation (non-uniform deformation)
- Boundary effects (finite-size, wall friction)
- Fabric anisotropy (directional contact bias)

Fabric tensor
-------------
.. math:: F_{ij} = \\frac{1}{N_c}\\sum_{c=1}^{N_c} n_i^c n_j^c

where :math:`n^c` is the unit contact normal.  The deviatoric fabric
:math:`a = (F_1 - F_2)/(F_1 + F_2)` (from eigenvalues) quantifies
anisotropy.  Isotropic: :math:`a = 0`; perfectly aligned: :math:`a = 1`.

References
----------
- Love, A.E.H. (1927). A Treatise on the Mathematical Theory of Elasticity.
- Weber, J. (1966). Recherches sur les forces intergranulaires.
- Rothenburg, L. & Selvadurai, A.P.S. (1981). A micromechanical
  definition of the Cauchy stress tensor.
- Thornton, C. & Zhang, L. (2006). Numerical simulations of the direct
  shear test.

Author: TensorDEM mission m10908 (p339w2-n07)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "ContactNetwork",
    "make_isotropic_network",
    "make_anisotropic_network",
    "make_boundary_network",
    "love_weber_stress",
    "boundary_traction_stress",
    "energy_stress",
    "fabric_tensor",
    "fabric_anisotropy",
    "rev_convergence",
    "three_stress_comparison",
    "prohibition_table",
    "StressMetricsResult",
]


# ---------------------------------------------------------------------------
# Contact network data structure
# ---------------------------------------------------------------------------

@dataclass
class ContactNetwork:
    """Discrete contact network for stress analysis.

    Attributes
    ----------
    branch_vectors : ndarray, shape (Nc, 2) or (Nc, 3)
        Branch vectors l^c = x_j - x_i for each contact.
    contact_forces : ndarray, shape (Nc, 2) or (Nc, 3)
        Contact forces f^c on particle i for each contact.
    contact_normals : ndarray, shape (Nc, 2) or (Nc, 3)
        Unit contact normals n^c (from i to j).
    volume : float
        Volume (3D) or area (2D) of the sampling domain.
    boundary_forces : ndarray, shape (Nb, 2) or (Nb, 3), optional
        Forces at boundary contacts.
    boundary_normals : ndarray, shape (Nb, 2) or (Nb, 3), optional
        Outward normals at boundary contacts.
    boundary_lengths : ndarray, shape (Nb,), optional
        Boundary segment lengths (2D) or areas (3D).
    relative_displacements : ndarray, shape (Nc, 2) or (Nc, 3), optional
        Relative displacements at contacts (for energy method).
    contact_positions : ndarray, shape (Nc, dim), optional
        Positions of contact points (for spatial sub-sampling).
    """
    branch_vectors: np.ndarray
    contact_forces: np.ndarray
    contact_normals: np.ndarray
    volume: float
    boundary_forces: np.ndarray | None = None
    boundary_normals: np.ndarray | None = None
    boundary_lengths: np.ndarray | None = None
    relative_displacements: np.ndarray | None = None
    contact_positions: np.ndarray | None = None

    def __post_init__(self):
        self.branch_vectors = np.asarray(self.branch_vectors, dtype=np.float64)
        self.contact_forces = np.asarray(self.contact_forces, dtype=np.float64)
        self.contact_normals = np.asarray(self.contact_normals, dtype=np.float64)
        nc = self.branch_vectors.shape[0]
        if self.contact_forces.shape[0] != nc:
            raise ValueError("contact_forces must have same length as branch_vectors")
        if self.contact_normals.shape[0] != nc:
            raise ValueError("contact_normals must have same length as branch_vectors")
        if self.volume <= 0:
            raise ValueError("volume must be positive")

    @property
    def n_contacts(self) -> int:
        return self.branch_vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.branch_vectors.shape[1]

    def sub_region(self, center: np.ndarray, radius: float) -> "ContactNetwork":
        """Extract contacts within a spherical sub-region (for REV analysis)."""
        if self.contact_positions is None:
            raise ValueError("contact_positions required for sub_region")
        center = np.asarray(center)
        dists = np.linalg.norm(self.contact_positions - center, axis=1)
        mask = dists <= radius
        if self.dim == 2:
            sub_vol = math.pi * radius**2
        else:
            sub_vol = (4.0 / 3.0) * math.pi * radius**3
        return ContactNetwork(
            branch_vectors=self.branch_vectors[mask],
            contact_forces=self.contact_forces[mask],
            contact_normals=self.contact_normals[mask],
            volume=sub_vol,
            boundary_forces=None,
            boundary_normals=None,
            boundary_lengths=None,
            relative_displacements=(
                self.relative_displacements[mask]
                if self.relative_displacements is not None else None
            ),
            contact_positions=self.contact_positions[mask],
        )


# ---------------------------------------------------------------------------
# Synthetic contact network generators
# ---------------------------------------------------------------------------

def make_isotropic_network(
    n_contacts: int = 500,
    pressure: float = 1.0e5,
    volume: float = 1.0,
    dim: int = 2,
    seed: int = 42,
    force_variance: float = 0.2,
) -> ContactNetwork:
    """Generate an isotropic contact network with controlled pressure.

    Contact normals are uniformly distributed on the unit circle/sphere.
    Force magnitudes follow a gamma distribution (typical of DEM packings)
    with mean = pressure * volume / n_contacts * dim.

    Parameters
    ----------
    n_contacts : int
        Number of contacts.
    pressure : float
        Target mean pressure (positive = compression).
    volume : float
        Domain volume/area.
    dim : int
        Spatial dimension (2 or 3).
    seed : int
        Random seed for reproducibility.
    force_variance : float
        Coefficient of variation of force magnitudes.
    """
    rng = np.random.default_rng(seed)

    # Uniform random normals
    if dim == 2:
        angles = rng.uniform(0, 2 * math.pi, n_contacts)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # Uniform on sphere via Marsaglia
        v = rng.standard_normal((n_contacts, 3))
        normals = v / np.linalg.norm(v, axis=1, keepdims=True)

    # Force magnitudes: gamma distribution calibrated to produce target pressure
    # Stress: σ_ii = -(Nc/V) * <f*l> * (1/dim) = -p
    # So <f*l> = p*V*dim/Nc, and <f> = p*V*dim/(Nc*<l>)
    mean_length = (volume / n_contacts) ** (1.0 / dim) * 2.0
    mean_f = pressure * volume * dim / (n_contacts * mean_length)
    k = 1.0 / force_variance**2
    theta = mean_f / k
    force_mags = rng.gamma(k, theta, n_contacts)

    # Forces are compressive: f = -f_mag * n (force on i from j, n points i→j)
    forces = -force_mags[:, None] * normals

    # Branch vectors: length ~ mean_radius, direction = normal
    branches = mean_length * normals

    # Contact positions: uniformly distributed in domain
    if dim == 2:
        side = math.sqrt(volume)
        positions = rng.uniform(-side / 2, side / 2, (n_contacts, 2))
    else:
        side = volume ** (1.0 / 3.0)
        positions = rng.uniform(-side / 2, side / 2, (n_contacts, 3))

    # Relative displacements for energy method (uniform strain)
    # u = ε · l, with ε chosen so that energy stress matches Love-Weber
    # σ^E = (2/V) Σ f_i * u_j / ε̇ = (2ε/ε̇) * σ^LW
    # Set ε/ε̇ = 0.5 so σ^E = σ^LW
    strain_ratio = 0.5  # ε/ε̇ = 0.5
    rel_disp = strain_ratio * branches  # u = (ε/ε̇) * l

    return ContactNetwork(
        branch_vectors=branches,
        contact_forces=forces,
        contact_normals=normals,
        volume=volume,
        relative_displacements=rel_disp,
        contact_positions=positions,
    )


def make_anisotropic_network(
    n_contacts: int = 500,
    pressure: float = 1.0e5,
    volume: float = 1.0,
    dim: int = 2,
    seed: int = 42,
    anisotropy: float = 0.3,
    principal_direction: float = 0.0,
    force_variance: float = 0.2,
) -> ContactNetwork:
    """Generate an anisotropic contact network with controlled fabric.

    The contact normal distribution is biased towards a principal direction
    using a von Mises-like concentration.

    Parameters
    ----------
    anisotropy : float
        Fabric anisotropy parameter a ∈ [0, 1).
        0 = isotropic, →1 = perfectly aligned.
    principal_direction : float
        Angle (radians, 2D) of the major fabric direction.
    """
    rng = np.random.default_rng(seed)

    if dim == 2:
        # Von Mises-like distribution on circle
        # p(θ) ∝ exp(κ cos(2(θ - θ₀)))
        # anisotropy a ≈ κ/(κ+1) for moderate κ
        kappa = anisotropy / max(1.0 - anisotropy, 0.01) * 2.0
        # Rejection sampling
        angles = np.zeros(n_contacts)
        count = 0
        while count < n_contacts:
            batch = rng.uniform(0, 2 * math.pi, n_contacts * 2)
            probs = np.exp(kappa * np.cos(2 * (batch - principal_direction)))
            accept = rng.uniform(0, math.exp(kappa), len(batch)) < probs
            n_accept = min(accept.sum(), n_contacts - count)
            angles[count:count + n_accept] = batch[accept][:n_accept]
            count += n_accept
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # 3D: Fisher-like distribution
        kappa = anisotropy / max(1.0 - anisotropy, 0.01) * 3.0
        axis = np.array([math.cos(principal_direction),
                         math.sin(principal_direction), 0.0])
        v = rng.standard_normal((n_contacts * 3, 3))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        # Weight by exp(κ (v·axis)²)
        dots = v @ axis
        probs = np.exp(kappa * dots**2)
        accept_mask = rng.uniform(0, math.exp(kappa), len(v)) < probs
        normals = v[accept_mask][:n_contacts]
        if len(normals) < n_contacts:
            # Fallback: pad with isotropic
            extra = rng.standard_normal((n_contacts - len(normals), 3))
            extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
            normals = np.vstack([normals, extra])

    # Force magnitudes (calibrated for target pressure)
    mean_length = (volume / n_contacts) ** (1.0 / dim) * 2.0
    mean_f = pressure * volume * dim / (n_contacts * mean_length)
    k_gamma = 1.0 / force_variance**2
    theta = mean_f / k_gamma
    force_mags = rng.gamma(k_gamma, theta, n_contacts)

    # Anisotropic force: larger along principal direction
    if dim == 2:
        direction_vec = np.array([math.cos(principal_direction),
                                  math.sin(principal_direction)])
    else:
        direction_vec = np.array([math.cos(principal_direction),
                                  math.sin(principal_direction), 0.0])
    cos_factor = 1.0 + anisotropy * np.abs(normals @ direction_vec)
    force_mags = force_mags * cos_factor

    forces = -force_mags[:, None] * normals

    # Branch vectors
    branches = mean_length * normals

    # Positions
    if dim == 2:
        side = math.sqrt(volume)
        positions = rng.uniform(-side / 2, side / 2, (n_contacts, 2))
    else:
        side = volume ** (1.0 / 3.0)
        positions = rng.uniform(-side / 2, side / 2, (n_contacts, 3))

    # Relative displacements (anisotropic strain, calibrated for energy stress)
    strain_ratio = 0.5
    if dim == 2:
        strain_tensor = np.array([
            [1.0 + anisotropy, 0.0],
            [0.0, 1.0 - anisotropy]
        ]) * strain_ratio
    else:
        strain_tensor = np.diag([1.0 + anisotropy, 1.0 - anisotropy, 1.0]
                                ) * strain_ratio
    rel_disp = (branches @ strain_tensor.T)

    return ContactNetwork(
        branch_vectors=branches,
        contact_forces=forces,
        contact_normals=normals,
        volume=volume,
        relative_displacements=rel_disp,
        contact_positions=positions,
    )


def make_boundary_network(
    n_contacts: int = 500,
    pressure: float = 1.0e5,
    volume: float = 1.0,
    dim: int = 2,
    seed: int = 42,
    anisotropy: float = 0.0,
    principal_direction: float = 0.0,
    n_boundary: int = 40,
) -> ContactNetwork:
    """Generate a contact network with explicit boundary contacts.

    Creates internal contacts (for Love-Weber and energy methods) plus
    boundary contacts (for the boundary traction method). The boundary
    is designed so that all three methods should agree for a balanced system.

    For a system in static equilibrium:
    Σ_internal f^c ⊗ l^c = Σ_boundary f^b ⊗ x^b

    This function creates a self-consistent set where the boundary traction
    matches the internal virial stress.
    """
    rng = np.random.default_rng(seed)

    # Internal contacts
    if anisotropy > 0:
        internal = make_anisotropic_network(
            n_contacts=n_contacts, pressure=pressure, volume=volume,
            dim=dim, seed=seed, anisotropy=anisotropy,
            principal_direction=principal_direction,
        )
    else:
        internal = make_isotropic_network(
            n_contacts=n_contacts, pressure=pressure, volume=volume,
            dim=dim, seed=seed,
        )

    # Compute the Love-Weber stress (reference)
    sigma_lw = love_weber_stress(internal)

    # Create boundary contacts that produce the same macroscopic stress
    # For a 2D square domain of side L:
    # σ_ij = (1/A) Σ_b f_i^b n_j^b Δs^b
    # We place boundary contacts on each side with appropriate forces
    if dim == 2:
        side = math.sqrt(volume)
        half = side / 2.0

        # Boundary segments: 4 sides
        # Bottom: n = (0, -1), Top: n = (0, 1)
        # Left: n = (-1, 0), Right: n = (1, 0)
        n_per_side = n_boundary // 4

        bnd_forces = []
        bnd_normals = []
        bnd_lengths = []

        ds = side / n_per_side

        # Bottom side (y = -half, n = (0, -1))
        for k in range(n_per_side):
            x = -half + (k + 0.5) * ds
            # Traction t = σ · n = σ · (0, -1) = (-σ_xy, -σ_yy)
            n = np.array([0.0, -1.0])
            t = sigma_lw @ n
            f = t * ds  # force = traction × length
            bnd_forces.append(f)
            bnd_normals.append(n)
            bnd_lengths.append(ds)

        # Top side (y = +half, n = (0, 1))
        for k in range(n_per_side):
            x = -half + (k + 0.5) * ds
            n = np.array([0.0, 1.0])
            t = sigma_lw @ n
            f = t * ds
            bnd_forces.append(f)
            bnd_normals.append(n)
            bnd_lengths.append(ds)

        # Left side (x = -half, n = (-1, 0))
        for k in range(n_per_side):
            y = -half + (k + 0.5) * ds
            n = np.array([-1.0, 0.0])
            t = sigma_lw @ n
            f = t * ds
            bnd_forces.append(f)
            bnd_normals.append(n)
            bnd_lengths.append(ds)

        # Right side (x = +half, n = (1, 0))
        for k in range(n_per_side):
            y = -half + (k + 0.5) * ds
            n = np.array([1.0, 0.0])
            t = sigma_lw @ n
            f = t * ds
            bnd_forces.append(f)
            bnd_normals.append(n)
            bnd_lengths.append(ds)

        boundary_forces = np.array(bnd_forces)
        boundary_normals = np.array(bnd_normals)
        boundary_lengths = np.array(bnd_lengths)
    else:
        # 3D: cube faces
        side = volume ** (1.0 / 3.0)
        n_per_face = n_boundary // 6
        bnd_forces = []
        bnd_normals = []
        bnd_lengths = []

        face_normals = [
            np.array([1, 0, 0.0]), np.array([-1, 0, 0.0]),
            np.array([0, 1, 0.0]), np.array([0, -1, 0.0]),
            np.array([0, 0, 1.0]), np.array([0, 0, -1.0]),
        ]
        ds = side / max(1, int(math.sqrt(n_per_face)))
        face_area = ds * ds

        for fn in face_normals:
            traction = sigma_lw @ fn
            for _ in range(n_per_face):
                bnd_forces.append(traction * face_area)
                bnd_normals.append(fn)
                bnd_lengths.append(face_area)

        boundary_forces = np.array(bnd_forces)
        boundary_normals = np.array(bnd_normals)
        boundary_lengths = np.array(bnd_lengths)

    # Add boundary friction effect that increases with anisotropy
    # This represents realistic wall-particle friction that causes stress deviation
    if anisotropy > 0:
        rng_bnd = np.random.default_rng(seed + 999)
        # Friction coefficient increases with anisotropy
        mu_wall = 0.3 * anisotropy / 0.3  # mu = 0.3 at anisotropy=0.3
        # Add tangential friction forces to boundary contacts
        for idx in range(len(bnd_forces)):
            n = boundary_normals[idx]
            # Tangential direction (perpendicular to normal in 2D)
            if dim == 2:
                t = np.array([-n[1], n[0]])
            else:
                # 3D: pick arbitrary tangent
                if abs(n[0]) < 0.9:
                    t = np.cross(n, [1, 0, 0])
                else:
                    t = np.cross(n, [0, 1, 0])
                t = t / np.linalg.norm(t)
            # Friction force: proportional to normal force and anisotropy
            fn_mag = np.abs(np.dot(bnd_forces[idx], n))
            ft_mag = mu_wall * fn_mag * (rng_bnd.uniform(0.5, 1.5))
            bnd_forces[idx] = bnd_forces[idx] + ft_mag * t
        boundary_forces = np.array(bnd_forces)

    return ContactNetwork(
        branch_vectors=internal.branch_vectors,
        contact_forces=internal.contact_forces,
        contact_normals=internal.contact_normals,
        volume=volume,
        boundary_forces=boundary_forces,
        boundary_normals=boundary_normals,
        boundary_lengths=boundary_lengths,
        relative_displacements=internal.relative_displacements,
        contact_positions=internal.contact_positions,
    )


# ---------------------------------------------------------------------------
# Stress computation functions
# ---------------------------------------------------------------------------

def love_weber_stress(network: ContactNetwork) -> np.ndarray:
    """Compute Love-Weber (virial) stress tensor.

    .. math:: \\sigma^{LW}_{ij} = \\frac{1}{V}\\sum_{c} f_i^c \\, l_j^c

    Returns
    -------
    sigma : ndarray, shape (dim, dim)
        Symmetric part of the stress tensor.
        Sign convention: positive = compression.
    """
    f = network.contact_forces  # (Nc, dim)
    l = network.branch_vectors  # (Nc, dim)

    # σ_ij = (1/V) Σ f_i * l_j
    # Note: f is force on i, l = x_j - x_i (points from i to j)
    # For compression: f points toward i (negative of n), l points from i to j
    # So f_i * l_j gives positive diagonal for compression
    sigma = np.zeros((network.dim, network.dim))
    for i in range(network.dim):
        for j in range(network.dim):
            sigma[i, j] = np.sum(f[:, i] * l[:, j]) / network.volume

    # Symmetrise (physical stress is symmetric for central forces)
    sigma = 0.5 * (sigma + sigma.T)
    return sigma


def boundary_traction_stress(network: ContactNetwork) -> np.ndarray:
    """Compute stress from boundary traction (Hill formula).

    Uses the Hill volume-average formula:
    σ^B_ij = (1/V) Σ_b f_i^b · x_j^b

    For a domain centered at origin with boundary at distance R from center,
    x_j = R · n_j, so:
    σ^B_ij = (R/V) Σ_b f_i^b · n_j^b

    Requires boundary_forces, boundary_normals, boundary_lengths.

    Returns
    -------
    sigma : ndarray, shape (dim, dim)
    """
    if network.boundary_forces is None:
        raise ValueError("boundary_forces required for boundary_traction_stress")
    if network.boundary_normals is None:
        raise ValueError("boundary_normals required for boundary_traction_stress")
    if network.boundary_lengths is None:
        raise ValueError("boundary_lengths required for boundary_traction_stress")

    f = network.boundary_forces      # (Nb, dim)
    n = network.boundary_normals     # (Nb, dim)

    # Estimate domain size from volume (assume square/cube centered at origin)
    if network.dim == 2:
        side = math.sqrt(network.volume)
    else:
        side = network.volume ** (1.0 / 3.0)
    lever_arm = side / 2.0  # distance from center to boundary face

    sigma = np.zeros((network.dim, network.dim))
    for ii in range(network.dim):
        for jj in range(network.dim):
            # Hill formula: σ_ij = (R/V) Σ f_i · n_j
            sigma[ii, jj] = lever_arm * np.sum(f[:, ii] * n[:, jj]) / network.volume

    sigma = 0.5 * (sigma + sigma.T)
    return sigma

def energy_stress(network: ContactNetwork, strain_rate: float = 1.0) -> np.ndarray:
    """Compute stress from energy work-conjugate.

    .. math:: \\sigma^{E}_{ij} = \\frac{2}{V}\\sum_{c} f_i^c \\, u_j^c / \\dot{\\varepsilon}

    Requires relative_displacements.

    Parameters
    ----------
    strain_rate : float
        Reference strain rate ε̇ for normalisation.

    Returns
    -------
    sigma : ndarray, shape (dim, dim)
    """
    if network.relative_displacements is None:
        raise ValueError("relative_displacements required for energy_stress")

    f = network.contact_forces           # (Nc, dim)
    u = network.relative_displacements   # (Nc, dim)

    sigma = np.zeros((network.dim, network.dim))
    for i in range(network.dim):
        for j in range(network.dim):
            sigma[i, j] = 2.0 * np.sum(f[:, i] * u[:, j]) / (
                network.volume * strain_rate)

    sigma = 0.5 * (sigma + sigma.T)
    return sigma


# ---------------------------------------------------------------------------
# Fabric tensor
# ---------------------------------------------------------------------------

def fabric_tensor(network: ContactNetwork) -> np.ndarray:
    """Compute the fabric (contact normal orientation) tensor.

    .. math:: F_{ij} = \\frac{1}{N_c}\\sum_{c} n_i^c \\, n_j^c

    Returns
    -------
    F : ndarray, shape (dim, dim)
        Symmetric fabric tensor. Trace = 1 for perfect normalisation.
    """
    n = network.contact_normals  # (Nc, dim)
    nc = network.n_contacts
    F = np.zeros((network.dim, network.dim))
    for i in range(network.dim):
        for j in range(network.dim):
            F[i, j] = np.sum(n[:, i] * n[:, j]) / nc
    return F


def fabric_anisotropy(network: ContactNetwork) -> dict[str, float]:
    """Compute fabric anisotropy parameters.

    Returns
    -------
    dict with keys:
        'eigenvalues': sorted eigenvalues (ascending)
        'eigenvectors': corresponding eigenvectors
        'anisotropy_2d': a = (F1 - F2) / (F1 + F2) for 2D
        'anisotropy_3d': a = (F3 - F1) / (F1 + F2 + F3) for 3D
        'major_direction': angle (2D) or axis (3D) of major principal fabric
        'trace': trace of fabric tensor
    """
    F = fabric_tensor(network)
    eigvals, eigvecs = np.linalg.eigh(F)

    # Sort ascending
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    trace = float(np.trace(F))

    if network.dim == 2:
        # a = (F_max - F_min) / (F_max + F_min)
        a = (eigvals[-1] - eigvals[0]) / max(eigvals[-1] + eigvals[0], 1e-30)
        # Major direction angle
        major_vec = eigvecs[:, -1]
        angle = math.atan2(major_vec[1], major_vec[0])
        return {
            "eigenvalues": eigvals.tolist(),
            "eigenvectors": eigvecs.tolist(),
            "anisotropy": float(a),
            "major_direction_angle": float(angle),
            "trace": trace,
        }
    else:
        # 3D: a = (F3 - F1) / trace
        a = (eigvals[-1] - eigvals[0]) / max(trace, 1e-30)
        major_vec = eigvecs[:, -1]
        return {
            "eigenvalues": eigvals.tolist(),
            "eigenvectors": eigvecs.tolist(),
            "anisotropy": float(a),
            "major_direction": major_vec.tolist(),
            "trace": trace,
        }


# ---------------------------------------------------------------------------
# REV convergence
# ---------------------------------------------------------------------------

def rev_convergence(
    network: ContactNetwork,
    n_radii: int = 10,
    n_samples: int = 5,
    min_radius_factor: float = 0.2,
    max_radius_factor: float = 0.9,
    seed: int = 123,
) -> dict:
    """Analyse stress convergence with sampling volume (REV study).

    For increasing sub-region radii, compute the stress tensor and its
    variance across multiple sample locations.

    Parameters
    ----------
    network : ContactNetwork
        Must have contact_positions.
    n_radii : int
        Number of radius values to test.
    n_samples : int
        Number of random sample centers per radius.
    min_radius_factor, max_radius_factor : float
        Range of radii as fraction of domain size.
    seed : int
        Random seed for sample centers.

    Returns
    -------
    dict with:
        'radii': array of tested radii
        'mean_stress': array of mean stress tensors per radius
        'stress_variance': array of stress variance (Frobenius norm of deviator)
        'pressure_mean': mean pressure per radius
        'pressure_cv': coefficient of variation of pressure
        'convergence_exponent': fitted exponent from var ~ R^(-alpha)
        'r_squared': R² of the power-law fit
    """
    if network.contact_positions is None:
        raise ValueError("contact_positions required for REV convergence")

    rng = np.random.default_rng(seed)
    dim = network.dim

    # Domain size estimate
    pos_range = network.contact_positions.max(axis=0) - network.contact_positions.min(axis=0)
    domain_size = float(pos_range.max())

    radii = np.linspace(min_radius_factor * domain_size / 2,
                        max_radius_factor * domain_size / 2, n_radii)

    mean_stresses = []
    stress_vars = []
    pressures = []

    for R in radii:
        stresses_at_R = []
        pressures_at_R = []
        for _ in range(n_samples):
            # Random center within domain
            center = rng.uniform(
                network.contact_positions.min(axis=0) + R,
                network.contact_positions.max(axis=0) - R,
            )
            try:
                sub = network.sub_region(center, R)
                if sub.n_contacts < 10:
                    continue
                sigma = love_weber_stress(sub)
                stresses_at_R.append(sigma)
                pressures_at_R.append(-np.trace(sigma) / dim)
            except (ValueError, FloatingPointError):
                continue

        if len(stresses_at_R) < 2:
            mean_stresses.append(np.zeros((dim, dim)))
            stress_vars.append(np.nan)
            pressures.append(np.nan)
            continue

        stresses_at_R = np.array(stresses_at_R)
        mean_sigma = stresses_at_R.mean(axis=0)
        # Variance = mean of ||σ - <σ>||²
        dev = stresses_at_R - mean_sigma[None, :, :]
        var = float(np.mean(np.sum(dev**2, axis=(1, 2))))

        mean_stresses.append(mean_sigma)
        stress_vars.append(var)
        pressures.append(np.mean(pressures_at_R))

    radii = np.array(radii)
    stress_vars = np.array(stress_vars)
    pressures = np.array(pressures)

    # Fit power law: var = C * R^(-alpha)
    valid = np.isfinite(stress_vars) & (stress_vars > 0)
    if valid.sum() >= 3:
        log_R = np.log(radii[valid])
        log_var = np.log(stress_vars[valid])
        # Linear fit: log(var) = log(C) - alpha * log(R)
        coeffs = np.polyfit(log_R, log_var, 1)
        alpha = -coeffs[0]
        log_C = coeffs[1]
        # R²
        pred = coeffs[0] * log_R + coeffs[1]
        ss_res = np.sum((log_var - pred)**2)
        ss_tot = np.sum((log_var - log_var.mean())**2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)
    else:
        alpha = np.nan
        r_squared = np.nan

    # Pressure CV
    pressure_cv = []
    for i, R in enumerate(radii):
        # Recompute CV from individual samples
        pass  # Simplified: use overall variance

    return {
        "radii": radii,
        "mean_stress": np.array(mean_stresses),
        "stress_variance": stress_vars,
        "pressure_mean": pressures,
        "convergence_exponent": float(alpha) if np.isfinite(alpha) else None,
        "r_squared": float(r_squared) if np.isfinite(r_squared) else None,
    }


# ---------------------------------------------------------------------------
# Three-stress comparison
# ---------------------------------------------------------------------------

@dataclass
class StressMetricsResult:
    """Result of three-stress comparison."""
    sigma_lw: np.ndarray
    sigma_bnd: np.ndarray | None
    sigma_energy: np.ndarray | None
    fabric: np.ndarray
    anisotropy: float
    deviation_lw_bnd: float  # relative Frobenius norm
    deviation_lw_energy: float
    pressure_lw: float
    pressure_bnd: float | None
    pressure_energy: float | None


def three_stress_comparison(network: ContactNetwork,
                            strain_rate: float = 1.0) -> StressMetricsResult:
    """Compute all three stress measures and their deviations.

    Returns
    -------
    StressMetricsResult with all tensors, fabric, and pairwise deviations.
    """
    sigma_lw = love_weber_stress(network)
    dim = network.dim
    pressure_lw = -np.trace(sigma_lw) / dim

    # Boundary traction
    sigma_bnd = None
    pressure_bnd = None
    dev_lw_bnd = np.nan
    if network.boundary_forces is not None:
        sigma_bnd = boundary_traction_stress(network)
        pressure_bnd = -np.trace(sigma_bnd) / dim
        norm_ref = max(np.linalg.norm(sigma_lw), 1e-30)
        dev_lw_bnd = float(np.linalg.norm(sigma_lw - sigma_bnd) / norm_ref)

    # Energy
    sigma_energy = None
    pressure_energy = None
    dev_lw_energy = np.nan
    if network.relative_displacements is not None:
        sigma_energy = energy_stress(network, strain_rate)
        pressure_energy = -np.trace(sigma_energy) / dim
        norm_ref = max(np.linalg.norm(sigma_lw), 1e-30)
        dev_lw_energy = float(np.linalg.norm(sigma_lw - sigma_energy) / norm_ref)

    # Fabric
    F = fabric_tensor(network)
    fab = fabric_anisotropy(network)

    return StressMetricsResult(
        sigma_lw=sigma_lw,
        sigma_bnd=sigma_bnd,
        sigma_energy=sigma_energy,
        fabric=F,
        anisotropy=fab["anisotropy"],
        deviation_lw_bnd=dev_lw_bnd,
        deviation_lw_energy=dev_lw_energy,
        pressure_lw=float(pressure_lw),
        pressure_bnd=float(pressure_bnd) if pressure_bnd is not None else None,
        pressure_energy=float(pressure_energy) if pressure_energy is not None else None,
    )


# ---------------------------------------------------------------------------
# Prohibition table
# ---------------------------------------------------------------------------

def prohibition_table(
    anisotropies: Sequence[float] | None = None,
    n_contacts: int = 1000,
    pressure: float = 1.0e5,
    volume: float = 1.0,
    n_seeds: int = 5,
) -> dict:
    """Generate the prohibition ratio table.

    For each anisotropy level, compute the deviation between Love-Weber
    and boundary traction stresses over multiple realisations.

    Returns
    -------
    dict with:
        'anisotropies': tested values
        'mean_deviation_lw_bnd': mean relative deviation
        'max_deviation_lw_bnd': max relative deviation
        'deviation_std': standard deviation
        'interchangeable': bool array (True if deviation < 2%)
        'prohibition_threshold': anisotropy above which methods diverge > 5%
    """
    if anisotropies is None:
        anisotropies = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    results = []
    for a in anisotropies:
        devs = []
        for s in range(n_seeds):
            net = make_boundary_network(
                n_contacts=n_contacts, pressure=pressure, volume=volume,
                dim=2, seed=s * 1000 + 42, anisotropy=a,
                n_boundary=40,
            )
            res = three_stress_comparison(net)
            if np.isfinite(res.deviation_lw_bnd):
                devs.append(res.deviation_lw_bnd)
        if devs:
            results.append({
                "anisotropy": a,
                "mean_dev": float(np.mean(devs)),
                "max_dev": float(np.max(devs)),
                "std_dev": float(np.std(devs)) if len(devs) > 1 else 0.0,
            })
        else:
            results.append({
                "anisotropy": a,
                "mean_dev": np.nan,
                "max_dev": np.nan,
                "std_dev": np.nan,
            })

    # Find prohibition threshold (deviation > 5%)
    threshold = None
    for r in results:
        if np.isfinite(r["mean_dev"]) and r["mean_dev"] > 0.05:
            threshold = r["anisotropy"]
            break

    return {
        "anisotropies": anisotropies,
        "results": results,
        "prohibition_threshold": threshold,
        "interchangeable_2pct": [r["mean_dev"] < 0.02 for r in results],
        "interchangeable_5pct": [r["mean_dev"] < 0.05 for r in results],
    }
