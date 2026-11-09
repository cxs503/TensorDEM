"""Static granular-bed effective thermal conductivity (conduction only).

Physics
-------
A static (non-convecting) packed bed of spheres saturated by a fluid conducts
heat through two parallel channels: the solid skeleton (spheres linked by
elastically flattened contacts) and the interstitial fluid (tortuous gaps).
The module provides

* the exact Wiener (series / parallel layer) bounds
  ``k_series <= k_eff <= k_parallel`` which any admissible model must respect;
* two literature correlations: Krupiczka (1967) and the Zehner-Schluender
  (1970) unit-cell model, the latter evaluated as a radial quadrature;
* **two independent numerical operators on the same static packing**:
  - :func:`conductivity_flux_average` - finite-volume field solve of
    ``div(k grad T) = 0`` on the periodic voxelisation of the DEM packing with
    the lumped sphere-sphere contact conductance inserted as an interface
    resistance; ``k_eff`` is the volume-averaged heat flux (operator B);
  - :func:`conductivity_contact_network` - Kirchhoff resistance-network
    summation on the sphere contact graph for the solid skeleton, added to a
    fluid-only field solve for the interstitial channel (operator A).

Both operators share the *same* contact law, so the difference between them is
a discretisation error (network vs. field averaging) - the acceptance gate is
5 %.

Contact law
-----------
A static bed carries its own overburden, so neighbouring spheres are flattened
elastically.  The normal force per contact is estimated from the confining
stress ``sigma`` and the coordination number ``Z``,

    F = sigma (2R)^2 / Z,

and Hertz theory for equal spheres gives the contact-patch radius

    a_c = (3 F R_eff / (4 E*))^(1/3),   R_eff = R/2,
    E*  = E / (2 (1 - nu^2)).

Maxwell constriction across the pair is ``R_contact = 1/(2 k_s a_c)``.  The law
is *lumped*: the whole pair conductance is distributed over the voxel faces
shared by the two spheres, so refining the voxel grid does not change the
contact physics.  The confining stress is therefore the single mechanical input
of the thermal model; its sensitivity is quantified in the benchmark.

Homogenisation convention
-------------------------
All solves use the periodic + linear-background-field formulation
``T = T' + G x`` with ``|G| = 1 K/m`` along ``z``; no wall effects enter, and

    k_eff = sum_faces q_f (x_j - x_i)_z / (V |G|),   q_f = G_f (T_i - T_j).

Units: SI (W m^-1 K^-1, m, Pa, -).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

__all__ = [
    "POROSITY_SCAN",
    "CONDUCTIVITY_RATIO_SCAN",
    "ContactModel",
    "StaticPackedBed",
    "ZehnerSchluenderConstants",
    "contact_radius",
    "conductivity_contact_network",
    "conductivity_flux_average",
    "effective_conductivity",
    "rescale_porosity",
    "generate_static_bed",
    "hertz_contact_radius",
    "krupiczka_conductivity",
    "maxwell_eucken_1",
    "maxwell_eucken_2",
    "wiener_bounds",
    "zehner_schluender_conductivity",
]

POROSITY_SCAN: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60)
CONDUCTIVITY_RATIO_SCAN: tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)

#: Zehner & Schluender (1970) flattening / shape factor for monodisperse spheres
ZS_SHAPE_PREFACTOR = 1.25
ZS_SHAPE_EXPONENT = 10.0 / 9.0


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _porosity(porosity: float) -> float:
    value = _finite("porosity", porosity)
    if not 0.0 < value < 1.0:
        raise ValueError("porosity must lie in (0, 1)")
    return value


# ---------------------------------------------------------------------------
# Bounds and literature correlations
# ---------------------------------------------------------------------------
def wiener_bounds(porosity: float, k_solid: float, k_fluid: float) -> dict[str, float]:
    """Series and parallel Wiener bounds of a two-phase mixture.

    ``k_series = 1 / ((1 - phi)/k_solid + phi/k_fluid)`` is the harmonic
    (layer-in-series) bound and ``k_parallel = (1 - phi) k_solid + phi k_fluid``
    the arithmetic (layer-in-parallel) bound; any correct effective
    conductivity lies between them.
    """
    phi = _porosity(porosity)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    series = 1.0 / ((1.0 - phi) / ks + phi / kf)
    parallel = (1.0 - phi) * ks + phi * kf
    return {"k_series": series, "k_parallel": parallel}


def krupiczka_conductivity(porosity: float, k_solid: float, k_fluid: float) -> float:
    """Krupiczka (1967) packed-bed correlation.

    ``k_eff / k_fluid = lambda**n`` with
    ``n = 0.280 - 0.757 log10(phi) - 0.057 log10(lambda)`` and
    ``lambda = k_solid / k_fluid``.
    """
    phi = _porosity(porosity)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    lam = ks / kf
    exponent = 0.280 - 0.757 * math.log10(phi) - 0.057 * math.log10(lam)
    return kf * lam**exponent



def maxwell_eucken_1(porosity: float, k_solid: float, k_fluid: float) -> float:
    """Maxwell-Eucken ME1: solid spheres dispersed in continuous fluid matrix.

    Valid when solid fraction (1-phi) is moderate.  Reduces to k_solid at phi=0
    and k_fluid at phi=1.

    ``k_eff = k_f * (k_s + 2*k_f + 2*(1-phi)*(k_s - k_f))
                   / (k_s + 2*k_f - (1-phi)*(k_s - k_f))``
    """
    phi = _porosity(porosity)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    f_s = 1.0 - phi  # solid fraction
    num = ks + 2.0 * kf + 2.0 * f_s * (ks - kf)
    den = ks + 2.0 * kf - f_s * (ks - kf)
    return kf * num / den


def maxwell_eucken_2(porosity: float, k_solid: float, k_fluid: float) -> float:
    """Maxwell-Eucken ME2: fluid bubbles dispersed in continuous solid matrix.

    Valid when porosity phi is moderate.  Reduces to k_solid at phi=0
    and k_fluid at phi=1.

    ``k_eff = k_s * (k_f + 2*k_s + 2*phi*(k_f - k_s))
                   / (k_f + 2*k_s - phi*(k_f - k_s))``
    """
    phi = _porosity(porosity)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    num = kf + 2.0 * ks + 2.0 * phi * (kf - ks)
    den = kf + 2.0 * ks - phi * (kf - ks)
    return ks * num / den


@dataclass(frozen=True)
class ZehnerSchluenderConstants:
    """Constants of the Zehner-Schluender (1970) unit-cell model."""

    shape_prefactor: float = ZS_SHAPE_PREFACTOR
    shape_exponent: float = ZS_SHAPE_EXPONENT
    quadrature_points: int = 4001

    def shape_factor(self, porosity: float) -> float:
        return self.shape_prefactor * ((1.0 - porosity) / porosity) ** self.shape_exponent


ZS_CONSTANTS = ZehnerSchluenderConstants()


def zehner_schluender_conductivity(
    porosity: float,
    k_solid: float,
    k_fluid: float,
    *,
    constants: ZehnerSchluenderConstants = ZS_CONSTANTS,
) -> float:
    """Zehner-Schluender unit-cell effective conductivity (radial quadrature).

    The unit cell is a fluid cylinder with one sphere on the axis.  Heat takes
    (i) a fluid-only path of area fraction ``1 - sqrt(1-phi)`` and (ii) a
    particle path of area fraction ``sqrt(1-phi)`` where solid and fluid are
    locally in series along the sphere chord.  The Zehner-Schluender flattening
    factor ``B = 1.25 ((1-phi)/phi)^(10/9)`` limits the near-axis path
    (``u < B/lambda``) by the finite contact patch, which is what makes the
    correlation differ from the naive no-contact superposition.

    The integral is evaluated by quadrature rather than transcribed from the
    1970 paper, so it stays verifiable against its analytic limits.
    """
    phi = _porosity(porosity)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    lam = ks / kf
    n = int(constants.quadrature_points)
    solid_area = math.sqrt(1.0 - phi)
    fluid_area = 1.0 - solid_area
    u = (np.arange(n) + 0.5) / n
    chord = np.sqrt(np.maximum(1.0 - u * u, 0.0))
    local = 1.0 / (chord / lam + (1.0 - chord))
    b = constants.shape_factor(phi) / lam
    if 0.0 < b < 1.0:
        contact = 1.0 / (u / b)  # constriction-limited near-axis path
        local = np.where(u < b, np.minimum(local, contact), local)
    kappa_particle = float(np.sum(local * 2.0 * u) / n)
    return kf * (fluid_area + solid_area * kappa_particle)


# ---------------------------------------------------------------------------
# Contact mechanics
# ---------------------------------------------------------------------------
def contact_radius(radius: float, overlap: float) -> float:
    """Hertzian contact radius from a flattening ``a_c = sqrt(R_eff delta)``."""
    r = _finite("radius", radius, positive=True)
    d = _finite("overlap", overlap, positive=True)
    return math.sqrt(0.5 * r * d)


def hertz_contact_radius(
    radius: float, normal_force: float, youngs_modulus: float, poisson_ratio: float
) -> float:
    """Hertz contact-patch radius of two equal elastic spheres."""
    r = _finite("radius", radius, positive=True)
    force = _finite("normal_force", normal_force, positive=True)
    e = _finite("youngs_modulus", youngs_modulus, positive=True)
    nu = _finite("poisson_ratio", poisson_ratio)
    if not -1.0 < nu < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    r_eff = 0.5 * r
    e_star = e / (2.0 * (1.0 - nu * nu))
    return (3.0 * force * r_eff / (4.0 * e_star)) ** (1.0 / 3.0)


@dataclass(frozen=True)
class ContactModel:
    """Mechanical state that fixes the contact-patch radius of a static bed."""

    confining_stress_pa: float = 1.0e4
    youngs_modulus_pa: float = 9.0e9
    poisson_ratio: float = 0.33

    def normal_force(self, bed: "StaticPackedBed") -> float:
        coordination = max(bed.coordination_number, 1.0)
        return self.confining_stress_pa * (2.0 * bed.radius) ** 2 / coordination

    def contact_radius(self, bed: "StaticPackedBed") -> float:
        return hertz_contact_radius(
            bed.radius,
            self.normal_force(bed),
            self.youngs_modulus_pa,
            self.poisson_ratio,
        )


DEFAULT_CONTACT_MODEL = ContactModel()


# ---------------------------------------------------------------------------
# Static packing generation (soft-sphere relaxation, periodic box)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StaticPackedBed:
    """A relaxed, periodic static sphere packing plus contact bookkeeping."""

    positions: np.ndarray
    radius: float
    box_length: float
    porosity: float
    contact_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    contact_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    contact_overlaps: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    @property
    def n_spheres(self) -> int:
        return int(self.positions.shape[0])

    @property
    def volume(self) -> float:
        return float(self.box_length**3)

    @property
    def coordination_number(self) -> float:
        return 2.0 * len(self.contact_pairs) / self.n_spheres

    @property
    def mean_overlap(self) -> float:
        if len(self.contact_overlaps) == 0:
            return 0.0
        return float(np.mean(self.contact_overlaps))

    def geometric_contact_radius(self) -> float:
        """Contact radius implied by the numerical overlap of the relaxed state."""
        if len(self.contact_overlaps) == 0:
            return 0.0
        return contact_radius(self.radius, max(self.mean_overlap, 1e-6 * self.radius))


def _minimum_image(vectors: np.ndarray, box: float) -> np.ndarray:
    return vectors - box * np.round(vectors / box)


def generate_static_bed(
    porosity: float,
    *,
    n_spheres: int = 150,
    seed: int = 20261107,
    relax_iterations: int = 400,
    damping: float = 0.35,
    max_overlap_fraction: float = 0.05,
) -> StaticPackedBed:
    """Relax a monodisperse packing to the requested porosity (periodic box).

    Spheres start on a jittered lattice and are pushed apart by position-based
    relaxation with periodic boundaries; the box is compressed and relaxed
    again until the target porosity is reached.  The numerical overlaps left in
    the jammed state are *not* used as contact physics (see ``ContactModel``).
    """
    phi_target = _porosity(porosity)
    if n_spheres < 8:
        raise ValueError("n_spheres must be at least 8")
    rng = np.random.default_rng(seed)

    n_side = int(math.ceil(n_spheres ** (1.0 / 3.0)))
    radius = 1.0
    spacing = radius * (4.0 * math.pi / 3.0 / max(1.0 - phi_target, 1e-3)) ** (1.0 / 3.0)
    box = spacing * n_side
    grid = np.stack(
        np.meshgrid(*[np.arange(n_side) * spacing for _ in range(3)], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    positions = grid[:n_spheres] + rng.normal(scale=0.25 * spacing, size=(n_spheres, 3))
    positions = np.mod(positions, box)

    def overlap_stats(pos: np.ndarray, box_length: float, rad: float):
        d = _minimum_image(pos[:, None, :] - pos[None, :, :], box_length)
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        return 2.0 * rad - dist, d, dist

    def relax(pos, box_length, rad, iterations):
        for _ in range(iterations):
            overlap, d, dist = overlap_stats(pos, box_length, rad)
            push = np.maximum(overlap, 0.0)
            if push.max() <= 0.5 * max_overlap_fraction * rad:
                break
            scale = (0.5 * push / np.maximum(dist, 1e-12))[:, :, None]
            pos = pos + damping * np.sum(scale * d, axis=1)
            pos = np.mod(pos, box_length)
        return pos

    positions = relax(positions, box, radius, relax_iterations)

    def solid_fraction(box_length: float) -> float:
        return n_spheres * 4.0 / 3.0 * math.pi * radius**3 / box_length**3

    for _ in range(60):
        phi_now = 1.0 - solid_fraction(box)
        if abs(phi_now - phi_target) < 2e-3:
            break
        shrink = (solid_fraction(box) / (1.0 - phi_target)) ** (1.0 / 3.0)
        shrink = min(max(shrink, 0.95), 1.02)
        box *= shrink
        positions = np.mod(positions * shrink, box)
        positions = relax(positions, box, radius, max(20, relax_iterations // 6))

    overlap, d, dist = overlap_stats(positions, box, radius)
    pairs = np.argwhere(np.triu(overlap > 0.0))
    return StaticPackedBed(
        positions=positions,
        radius=radius,
        box_length=float(box),
        porosity=float(1.0 - solid_fraction(box)),
        contact_pairs=pairs.astype(int),
        contact_vectors=d[pairs[:, 0], pairs[:, 1]],
        contact_overlaps=overlap[pairs[:, 0], pairs[:, 1]],
    )


def rescale_porosity(bed: StaticPackedBed, porosity: float) -> StaticPackedBed:
    """Return the same packing with the sphere radius rescaled to ``porosity``.

    Used for the ``phi -> 0`` / ``phi -> 1`` limit tests: the sphere centres are
    kept, the radius is scaled so that the solid fraction matches the target,
    and the contact list is rebuilt from the new overlaps.
    """
    target = _porosity(porosity)
    solid_fraction_target = 1.0 - target
    current = 1.0 - bed.porosity
    radius = bed.radius * (solid_fraction_target / current) ** (1.0 / 3.0)
    d = _minimum_image(bed.positions[:, None, :] - bed.positions[None, :, :], bed.box_length)
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    overlap = 2.0 * radius - dist
    pairs = np.argwhere(np.triu(overlap > 0.0))
    return StaticPackedBed(
        positions=bed.positions,
        radius=float(radius),
        box_length=bed.box_length,
        porosity=float(target),
        contact_pairs=pairs.astype(int),
        contact_vectors=d[pairs[:, 0], pairs[:, 1]],
        contact_overlaps=overlap[pairs[:, 0], pairs[:, 1]],
    )


def _fluid_only_channel(
    bed: StaticPackedBed,
    k_fluid: float,
    grid: int,
    contact_radius_m: float,
) -> float:
    """Interstitial channel: field solve with the solid phase almost insulating."""
    return conductivity_flux_average(
        bed, k_solid=1e-9 * k_fluid, k_fluid=k_fluid, grid=grid,
        contact_radius_m=contact_radius_m,
    )["k_eff"]


# ---------------------------------------------------------------------------
# Voxelisation and the periodic finite-volume solver
# ---------------------------------------------------------------------------
def _voxel_labels(bed: StaticPackedBed, grid: int) -> np.ndarray:
    """Label each voxel with the id of the sphere containing its centre."""
    n = int(grid)
    h = bed.box_length / n
    centres = (np.arange(n) + 0.5) * h
    pts = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).reshape(-1, 3)
    labels = np.full(pts.shape[0], -1, dtype=np.int32)
    best = np.full(pts.shape[0], np.inf)
    for index, centre in enumerate(bed.positions):
        delta = _minimum_image(pts - centre, bed.box_length)
        dist = np.linalg.norm(delta, axis=1)
        inside = (dist <= bed.radius) & (dist < best)
        labels[inside] = index
        best[inside] = dist[inside]
    return labels.reshape(n, n, n)


def _periodic_solve(
    conductances: list[np.ndarray], h: float, n: int, gradient: float
) -> tuple[float, np.ndarray]:
    """Solve the periodic conduction problem; return ``(k_eff, T)``.

    ``conductances[d]`` holds the face conductance (W/K) between a cell and its
    ``+d`` neighbour, periodic wrap allowed.  The background field is
    ``T0 = -gradient z`` (``<grad T> = -gradient e_z``); the periodic image of
    the far cell carries ``+gradient L`` and is unwrapped in the flux sum.
    """
    from scipy.sparse import coo_matrix, diags, identity
    from scipy.sparse.linalg import cg

    size = n**3
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(size)
    idx = np.arange(size).reshape(n, n, n)
    face_terms = []

    for direction, gfaces in enumerate(conductances):
        g = gfaces.reshape(-1)
        i0 = idx.reshape(-1)
        i1 = np.roll(idx, -1, axis=direction).reshape(-1)
        rows.extend((i0, i1))
        cols.extend((i1, i0))
        vals.extend((-g, -g))
        diag += (gfaces + np.roll(gfaces, 1, axis=direction)).reshape(-1)
        face_terms.append((direction, i0, i1, g))

    rows.append(np.arange(size))
    cols.append(np.arange(size))
    vals.append(diag)
    matrix = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    ).tocsr()
    matrix = matrix + (1e-12 * float(np.mean(diag))) * identity(size, format="csr")

    # edge-based background-field drive: rhs_i = sum_j g_ij |G| d_z,ij with the
    # minimum-image edge vector (d_z = -h for every z face, including the
    # periodic wrap, because the wrap face joins geometric neighbours)
    rhs = np.zeros(size)
    for direction, gfaces in enumerate(conductances):
        if direction != 2:
            continue
        g = gfaces.reshape(-1)
        i1 = np.roll(idx, -1, axis=direction).reshape(-1)
        contribution = g * gradient * (-h)
        rhs += contribution
        rhs[i1] -= contribution
    rhs -= rhs.mean()
    preconditioner = diags(1.0 / matrix.diagonal()).tocsr()
    tprime, info = cg(matrix, rhs, rtol=1e-11, atol=0.0, maxiter=20000, M=preconditioner)
    if info != 0:
        raise RuntimeError(f"periodic conduction solve did not converge (info={info})")
    tprime -= tprime.mean()
    temperature = tprime.reshape(n, n, n)

    flat = temperature.reshape(-1)
    flux_sum = 0.0
    for direction, i0, i1, g in face_terms:
        if direction != 2:
            continue
        # q_{i->j} = g (T'_i - T'_j - |G| d_z), d_z = -h, and (x_j - x_i)_z = h
        q = g * (flat[i0] - flat[i1] + gradient * h)
        flux_sum += float(np.sum(q)) * h
    return flux_sum / (size * h**3 * gradient), temperature


# ---------------------------------------------------------------------------
# Operator B: volume-averaged heat flux
# ---------------------------------------------------------------------------
def conductivity_flux_average(
    bed: StaticPackedBed,
    k_solid: float,
    k_fluid: float,
    *,
    grid: int = 32,
    contact_model: ContactModel = DEFAULT_CONTACT_MODEL,
    contact_radius_m: float | None = None,
) -> dict[str, float]:
    """Operator B: volume-averaged flux of a finite-volume field solve.

    Solid voxels carry ``k_solid``, fluid voxels ``k_fluid``; faces shared by
    two *different* spheres carry the lumped pair conductance ``2 k_s a_c`` in
    series with the bulk bridge, so the contact resistance is mesh independent.
    """
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    n = int(grid)
    if n < 8:
        raise ValueError("grid must be at least 8")
    h = bed.box_length / n
    labels = _voxel_labels(bed, n)
    solid = labels >= 0
    if solid.all() or not solid.any():
        raise ValueError("packing must contain both phases on the voxel grid")

    a_c = float(contact_radius_m) if contact_radius_m is not None else contact_model.contact_radius(bed)
    pair_conductance = 2.0 * ks * a_c
    contact_keys = {
        int(min(a, b)) * 100000 + int(max(a, b)) for a, b in bed.contact_pairs.tolist()
    }

    # the lumped pair conductance must be shared by *all* voxel faces that
    # bridge the pair, in every direction, otherwise the pair is counted once
    # per direction
    pair_keys: list[np.ndarray] = []
    masks = []
    for direction in (0, 1, 2):
        left = labels
        right = np.roll(labels, -1, axis=direction)
        bridging = (left >= 0) & (right >= 0) & (left != right)
        masks.append(bridging)
        keys = np.where(
            left[bridging] < right[bridging],
            left[bridging].astype(np.int64) * 100000 + right[bridging],
            right[bridging].astype(np.int64) * 100000 + left[bridging],
        )
        pair_keys.append(keys)
    all_keys = np.concatenate(pair_keys) if pair_keys else np.zeros(0, dtype=np.int64)
    if len(all_keys):
        unique_keys, counts = np.unique(all_keys, return_counts=True)
        count_of = dict(zip(unique_keys.tolist(), counts.tolist()))
    else:
        count_of = {}

    conductances = []
    n_bridging = 0
    for direction in (0, 1, 2):
        left = labels
        right = np.roll(labels, -1, axis=direction)
        left_k = np.where(left >= 0, ks, kf)
        right_k = np.where(right >= 0, ks, kf)
        face = 2.0 * h * left_k * right_k / (left_k + right_k)
        bridging = masks[direction]
        if bridging.any() and pair_conductance > 0.0:
            n_bridging += int(np.count_nonzero(bridging))
            keys_here = pair_keys[direction]
            # voxel-adjacent spheres that are *not* in the contact list only
            # touch through a sub-voxel fluid film: give them the series
            # (harmonic) solid-fluid value instead of a solid bond
            is_contact = np.array([key in contact_keys for key in keys_here.tolist()])
            if is_contact.any():
                per_face = np.array(
                    [pair_conductance / count_of[key] for key in keys_here[is_contact].tolist()]
                )
                idx_bridging = np.flatnonzero(bridging)
                # solid constriction in series with the voxel bridge, plus the
                # interstitial film that wets the interface in parallel (this
                # term is what makes the degenerate k_solid == k_fluid case
                # conduct as a homogeneous medium)
                # a constricted contact cannot conduct better than a fully
                # solid face; clamping keeps the operator inside the Wiener
                # parallel bound at the degenerate contrast
                contact_face = (
                    1.0 / (1.0 / (h * ks) + 1.0 / per_face) + h * kf
                )
                face.reshape(-1)[idx_bridging[is_contact]] = np.minimum(
                    contact_face, h * ks
                )
            if (~is_contact).any():
                idx_bridging = np.flatnonzero(bridging)
                film = 2.0 * h * ks * kf / (ks + kf)
                face.reshape(-1)[idx_bridging[~is_contact]] = film
        conductances.append(face)

    k_eff, _ = _periodic_solve(conductances, h, n, gradient=1.0)
    return {
        "k_eff": float(k_eff),
        "grid": n,
        "voxel_size": float(h),
        "solid_voxel_fraction": float(np.count_nonzero(solid) / solid.size),
        "contact_radius": float(a_c),
        "pair_conductance": float(pair_conductance),
        "n_contact_pairs": int(len(bed.contact_pairs)),
        "n_bridging_faces": n_bridging,
    }


# ---------------------------------------------------------------------------
# Operator A: contact resistance network summation
# ---------------------------------------------------------------------------
def _graph_solid_channel(
    bed: StaticPackedBed, k_solid: float, contact_radius_m: float
) -> dict[str, float]:
    """Kirchhoff network summation of the solid skeleton (operator A, solid part).

    Bond resistance = Maxwell constriction of the contact patch ``1/(2 k_s a_c)``
    plus the intra-sphere spreading ``1/(pi k_s R)``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    ks = _finite("k_solid", k_solid, positive=True)
    a_c = _finite("contact_radius_m", contact_radius_m, positive=True)
    n_spheres = bed.n_spheres
    if len(bed.contact_pairs) == 0:
        return {"k_eff": 0.0, "n_contacts": 0, "bond_conductance": 0.0}
    r_spread = 1.0 / (math.pi * ks * bed.radius)
    r_contact = 1.0 / (2.0 * ks * a_c)
    g_bond = 1.0 / (r_contact + r_spread)

    pairs = bed.contact_pairs
    vectors = bed.contact_vectors  # x_i - x_j (minimum image)
    gradient = 1.0
    drive = gradient * g_bond * vectors[:, 2]
    rhs = np.zeros(n_spheres)
    np.add.at(rhs, pairs[:, 0], drive)
    np.add.at(rhs, pairs[:, 1], -drive)

    diag = np.zeros(n_spheres)
    np.add.at(diag, pairs[:, 0], g_bond)
    np.add.at(diag, pairs[:, 1], g_bond)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n_spheres)])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n_spheres)])
    vals = np.concatenate([
        -np.full(len(pairs), g_bond),
        -np.full(len(pairs), g_bond),
        diag + 1e-16 * g_bond,
    ])
    matrix = coo_matrix((vals, (rows, cols)), shape=(n_spheres, n_spheres)).tocsr()
    rhs = rhs - rhs.mean()
    tprime = spsolve(matrix, rhs)
    tprime -= tprime.mean()

    q = g_bond * (tprime[pairs[:, 0]] - tprime[pairs[:, 1]] - gradient * vectors[:, 2])
    flux_sum = float(np.sum(q * (-vectors[:, 2])))
    return {
        "k_eff": float(flux_sum / (bed.volume * gradient)),
        "n_contacts": int(len(pairs)),
        "bond_conductance": float(g_bond),
        "r_contact": float(r_contact),
        "r_spread": float(r_spread),
    }


def conductivity_contact_network(
    bed: StaticPackedBed,
    k_solid: float,
    k_fluid: float,
    *,
    grid: int = 32,
    contact_model: ContactModel = DEFAULT_CONTACT_MODEL,
    contact_radius_m: float | None = None,
) -> dict[str, float]:
    """Operator A: contact-resistance network summation + interstitial channel.

    The solid skeleton is a Kirchhoff resistor network on the sphere contact
    graph; the interstitial channel is a field solve with the solid made
    insulating.  The two channels are combined in parallel, the standard
    two-channel approximation.
    """
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    a_c = float(contact_radius_m) if contact_radius_m is not None else contact_model.contact_radius(bed)
    solid_channel = _graph_solid_channel(bed, ks, a_c)
    fluid_only = conductivity_flux_average(
        bed, k_solid=1e-9 * kf, k_fluid=kf, grid=grid, contact_radius_m=a_c
    )
    return {
        "k_eff": float(solid_channel["k_eff"] + fluid_only["k_eff"]),
        "k_solid_channel": float(solid_channel["k_eff"]),
        "k_fluid_channel": float(fluid_only["k_eff"]),
        "n_contacts": solid_channel["n_contacts"],
        "bond_conductance": float(solid_channel["bond_conductance"]),
        "contact_radius": float(a_c),
        "grid": int(grid),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def effective_conductivity(
    bed: StaticPackedBed,
    k_solid: float,
    k_fluid: float,
    *,
    grid: int = 44,
    contact_model: ContactModel = DEFAULT_CONTACT_MODEL,
    contact_radius_m: float | None = None,
) -> dict[str, float]:
    """Evaluate every model at one (porosity, contrast) point.

    Returns the field-averaged effective conductivity (operator B) together with
    its decomposition into solid-skeleton and interstitial channels, the
    network-summation skeleton (operator A) and both literature correlations.
    The two operators are compared on the *solid skeleton* (same configuration):
    operator A is the contact-network summation, operator B's skeleton value is
    the field solve with the fluid phase insulating.
    """
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    bounds = wiener_bounds(bed.porosity, ks, kf)
    if contact_radius_m is None:
        contact_radius_m = contact_model.contact_radius(bed)
    k_eff = conductivity_flux_average(
        bed, ks, kf, grid=grid, contact_radius_m=contact_radius_m
    )["k_eff"]
    skeleton_fvm = conductivity_flux_average(
        bed, ks, 1e-9 * ks, grid=grid, contact_radius_m=contact_radius_m
    )["k_eff"]
    fluid_channel = _fluid_only_channel(bed, kf, grid, contact_radius_m)
    skeleton_network = _graph_solid_channel(bed, ks, contact_radius_m)["k_eff"]
    k_krupiczka = krupiczka_conductivity(bed.porosity, ks, kf)
    k_zs = zehner_schluender_conductivity(bed.porosity, ks, kf)
    lo, hi = min(k_krupiczka, k_zs), max(k_krupiczka, k_zs)
    two_operator = abs(skeleton_network - skeleton_fvm) / skeleton_fvm
    return {
        "porosity": float(bed.porosity),
        "contrast": float(ks / kf),
        "k_solid": ks,
        "k_fluid": kf,
        "k_eff": float(k_eff),
        "k_skeleton_fvm": float(skeleton_fvm),
        "k_skeleton_network": float(skeleton_network),
        "k_fluid_channel": float(fluid_channel),
        "k_parallel_superposition": float(skeleton_network + fluid_channel),
        "two_operator_rel_diff": float(two_operator),
        "k_series": bounds["k_series"],
        "k_parallel": bounds["k_parallel"],
        "k_krupiczka": float(k_krupiczka),
        "k_zehner_schluender": float(k_zs),
        "rel_error_krupiczka": abs(k_eff - k_krupiczka) / k_eff,
        "rel_error_zehner_schluender": abs(k_eff - k_zs) / k_eff,
        "literature_bracket_low": float(lo),
        "literature_bracket_high": float(hi),
        "within_literature_band": bool(0.85 * lo <= k_eff <= 1.15 * hi),
        "band_low": float(0.85 * lo),
        "band_high": float(1.15 * hi),
        "within_bounds": bool(
            bounds["k_series"] * (1.0 - 1e-6) <= k_eff <= bounds["k_parallel"] * (1.0 + 1e-6)
        ),
        "coordination_number": float(bed.coordination_number),
        "mean_contact_overlap": float(bed.mean_overlap),
        "contact_radius": float(contact_radius_m),
        "contact_stress_pa": float(contact_model.confining_stress_pa),
        "n_contacts": int(len(bed.contact_pairs)),
    }


# ---------------------------------------------------------------------------
# SOR solver for contact network (alternative to direct spsolve)
# ---------------------------------------------------------------------------
def solve_contact_network_sor(
    bed: StaticPackedBed,
    k_solid: float,
    contact_radius_m: float,
    *,
    gradient: float = 1.0,
    omega: float = 1.8,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> dict[str, float]:
    """SOR solver for the contact-resistance network with Dirichlet boundary.

    Solves the sparse linear system arising from the Kirchhoff network on the
    sphere contact graph, using Successive Over-Relaxation.  The background
    field ``T0 = -gradient * z`` is imposed; the periodic image carries the
    jump ``+gradient * L`` which is unwrapped in the flux sum.

    Parameters
    ----------
    bed : StaticPackedBed
        The relaxed packing.
    k_solid : float
        Solid thermal conductivity (W/m/K).
    contact_radius_m : float
        Hertz contact radius (m).
    gradient : float
        Background temperature gradient (K/m).
    omega : float
        SOR relaxation parameter (1.0 = Gauss-Seidel, 1.8 typical optimum).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on residual norm.

    Returns
    -------
    dict with k_eff, n_contacts, bond_conductance, iterations, converged.
    """
    from scipy.sparse import coo_matrix

    ks = _finite("k_solid", k_solid, positive=True)
    a_c = _finite("contact_radius_m", contact_radius_m, positive=True)
    n_spheres = bed.n_spheres
    pairs = bed.contact_pairs
    vectors = bed.contact_vectors

    # Bond conductance: series of constriction + spread resistance
    r_spread = 1.0 / (math.pi * ks * bed.radius)
    r_contact = 1.0 / (2.0 * ks * a_c)
    g_bond = 1.0 / (r_contact + r_spread)

    # Build diagonal and off-diagonal
    diag = np.zeros(n_spheres)
    np.add.at(diag, pairs[:, 0], g_bond)
    np.add.at(diag, pairs[:, 1], g_bond)

    # RHS from background gradient
    drive = gradient * g_bond * vectors[:, 2]
    rhs = np.zeros(n_spheres)
    np.add.at(rhs, pairs[:, 0], drive)
    np.add.at(rhs, pairs[:, 1], -drive)
    rhs = rhs - rhs.mean()  # fix gauge

    # SOR iteration
    t = np.zeros(n_spheres)
    converged = False
    n_iter = 0
    for it in range(max_iter):
        t_old = t.copy()
        for i in range(n_spheres):
            sigma = 0.0
            for j_idx in range(len(pairs)):
                if pairs[j_idx, 0] == i:
                    sigma -= g_bond * t[pairs[j_idx, 1]]
                elif pairs[j_idx, 1] == i:
                    sigma -= g_bond * t[pairs[j_idx, 0]]
            t[i] = (1.0 - omega) * t_old[i] + omega * (rhs[i] - sigma) / diag[i]
        n_iter = it + 1
        residual = np.max(np.abs(t - t_old))
        if residual < tol:
            converged = True
            break

    t -= t.mean()

    # Compute effective conductivity from flux
    q = g_bond * (t[pairs[:, 0]] - t[pairs[:, 1]] - gradient * vectors[:, 2])
    flux_sum = float(np.sum(q * (-vectors[:, 2])))
    k_eff = flux_sum / (bed.volume * gradient)

    return {
        "k_eff": float(k_eff),
        "n_contacts": int(len(pairs)),
        "bond_conductance": float(g_bond),
        "r_contact": float(r_contact),
        "r_spread": float(r_spread),
        "iterations": n_iter,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Isothermal sphere approximation (explicit declaration)
# ---------------------------------------------------------------------------
def isothermal_sphere_conductance(
    radius: float, k_solid: float, k_fluid: float, porosity: float
) -> float:
    """Particle-fluid equivalent conductance under isothermal sphere approximation.

    **Explicit declaration**: This model assumes each sphere is isothermal
    (Biot number << 1), so the intra-particle temperature gradient is neglected.
    The sphere acts as a lumped thermal capacitor connected to neighbours via
    contact conductances and to the fluid via a film conductance.

    The equivalent conductance is the parallel sum of:
    - Contact conductance to neighbours: ``N_c * 2 * k_s * a_c``
    - Fluid film conductance: ``4 * pi * R * k_f`` (Stokes-Einstein-like)

    Parameters
    ----------
    radius : float
        Sphere radius (m).
    k_solid : float
        Solid conductivity (W/m/K).
    k_fluid : float
        Fluid conductivity (W/m/K).
    porosity : float
        Bed porosity.

    Returns
    -------
    float : Equivalent conductance per sphere (W/K).
    """
    r = _finite("radius", radius, positive=True)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)
    phi = _porosity(porosity)

    # Fluid film conductance (spherical conduction in infinite medium)
    g_fluid = 4.0 * math.pi * r * kf

    # Contact conductance estimate (using typical coordination number)
    # Z ~ 6 for random close packing, scales with (1-phi)
    z_coord = 6.0 * (1.0 - phi) / 0.36  # normalised to RCP
    # Contact radius from typical overlap
    a_c = r * 0.05  # 5% overlap typical
    g_contact = z_coord * 2.0 * ks * a_c

    return g_fluid + g_contact


# ---------------------------------------------------------------------------
# Particle-fluid equivalent conductivity (two-phase homogenisation)
# ---------------------------------------------------------------------------
def particle_fluid_effective_conductivity(
    porosity: float, k_solid: float, k_fluid: float,
    *,
    model: str = "maxwell_eucken",
) -> float:
    """Particle-fluid equivalent conductivity with explicit model selection.

    Parameters
    ----------
    porosity : float
        Bed porosity (fluid volume fraction).
    k_solid : float
        Solid conductivity.
    k_fluid : float
        Fluid conductivity.
    model : str
        Homogenisation model: 'maxwell_eucken', 'wiener', 'krupiczka',
        'zehner_schluender'.

    Returns
    -------
    float : Effective conductivity (W/m/K).
    """
    phi = _porosity(porosity)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, positive=True)

    if model == "maxwell_eucken":
        # Use ME1 for fluid-continuous (typical for packed beds with gas)
        return maxwell_eucken_1(phi, ks, kf)
    elif model == "wiener":
        bounds = wiener_bounds(phi, ks, kf)
        return math.sqrt(bounds["k_series"] * bounds["k_parallel"])  # geometric mean
    elif model == "krupiczka":
        return krupiczka_conductivity(phi, ks, kf)
    elif model == "zehner_schluender":
        return zehner_schluender_conductivity(phi, ks, kf)
    else:
        raise ValueError(f"Unknown model: {model}")


# ---------------------------------------------------------------------------
# Downstream self-heating model: critical thickness delta_crit
# ---------------------------------------------------------------------------
def critical_thickness_delta_crit(
    k_eff: float,
    heat_generation_w_m3: float,
    t_surface: float,
    t_critical: float,
    *,
    geometry: str = "slab",
) -> float:
    """Critical bed thickness for thermal runaway (Frank-Kamenetskii criterion).

    For a self-heating porous bed with uniform volumetric heat generation,
    the critical thickness below which the bed remains stable is:

        delta_crit = sqrt(k_eff * (T_crit - T_surf) / q_gen) * xi

    where xi is a geometry factor (pi/2 for slab, 2.038 for cylinder, 3.0 for sphere).

    Parameters
    ----------
    k_eff : float
        Effective thermal conductivity (W/m/K).
    heat_generation_w_m3 : float
        Volumetric heat generation rate (W/m^3).
    t_surface : float
        Surface temperature (K).
    t_critical : float
        Critical temperature for runaway (K).
    geometry : str
        Geometry: 'slab', 'cylinder', 'sphere'.

    Returns
    -------
    float : Critical thickness (m).
    """
    k = _finite("k_eff", k_eff, positive=True)
    q = _finite("heat_generation_w_m3", heat_generation_w_m3, positive=True)
    ts = _finite("t_surface", t_surface)
    tc = _finite("t_critical", t_critical)
    if tc <= ts:
        raise ValueError("t_critical must exceed t_surface")

    delta_t = tc - ts
    if geometry == "slab":
        xi = math.pi / 2.0
    elif geometry == "cylinder":
        xi = 2.038
    elif geometry == "sphere":
        xi = math.pi
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    return xi * math.sqrt(k * delta_t / q)


# ---------------------------------------------------------------------------
# Kozeny-Carman cross-validation
# ---------------------------------------------------------------------------
def kozeny_carman_permeability(
    porosity: float, particle_diameter: float, *, kozeny_constant: float = 180.0
) -> float:
    """Kozeny-Carman permeability for packed bed of spheres.

    ``K = phi^3 * d^2 / (180 * (1-phi)^2)``

    Parameters
    ----------
    porosity : float
        Bed porosity.
    particle_diameter : float
        Particle diameter (m).
    kozeny_constant : float
        Kozeny constant (180 for spheres, 150-180 typical).

    Returns
    -------
    float : Permeability (m^2).
    """
    phi = _porosity(porosity)
    d = _finite("particle_diameter", particle_diameter, positive=True)
    kc = _finite("kozeny_constant", kozeny_constant, positive=True)
    return phi**3 * d**2 / (kc * (1.0 - phi)**2)


def thermal_formation_factor(
    porosity: float, k_fluid: float
) -> float:
    """Thermal formation factor F = k_fluid / k_eff_fluid_only.

    Analogous to electrical formation factor in petrophysics.
    For a packed bed, F ~ phi^(-m) where m is the cementation exponent.
    Uses Maxwell-Eucken model for fluid-only channel (solid ~ insulating).
    """
    phi = _porosity(porosity)
    kf = _finite("k_fluid", k_fluid, positive=True)
    # ME1 with k_s -> 0: k_eff = kf * 2*phi / (3 - phi)
    k_eff_fluid = kf * 2.0 * phi / (3.0 - phi)
    return kf / k_eff_fluid


def archie_cementation_exponent(
    porosities: list[float], k_fluid: float, k_eff_values: list[float]
) -> float:
    """Fit Archie's law exponent m from porosity vs formation factor data.

    ``F = phi^(-m)`` => ``log(F) = -m * log(phi)``

    Parameters
    ----------
    porosities : list[float]
        Porosity values.
    k_fluid : float
        Fluid conductivity.
    k_eff_values : list[float]
        Effective conductivity values (fluid-only channel).

    Returns
    -------
    float : Cementation exponent m.
    """
    if len(porosities) < 2:
        raise ValueError("Need at least 2 data points")
    log_phi = np.log(porosities)
    log_F = np.log([k_fluid / k for k in k_eff_values])
    m = -np.polyfit(log_phi, log_F, 1)[0]
    return float(m)
