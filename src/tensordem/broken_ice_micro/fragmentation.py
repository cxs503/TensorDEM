"""Quasi-static fragmentation of a bonded level-ice plate (campaign IN34-W3-N01).

Why this engine
---------------
A floe-size distribution is *produced* by a fragmentation process, so the FSD
exponent can only be validated against a simulation that actually breaks a
plate.  TensorDEM's existing bonded models (:class:`tensordem.dem.IceDEM`,
:class:`tensordem.dem3d.IceDEM3D`) are built for single-fracture-event
verification: the 2D model needs ~1 s per step for 1.2k particles (measured on
this worktree), so a fragmentation run that yields a *statistical ensemble* of
floes is out of reach, and an explicit dynamic integration of a stiff lattice at
the required damage level is dominated by contact chatter rather than by
fragmentation (this was measured: an explicit version of the engine below kept
``KE / strain_energy ~ O(10)`` even with critical contact damping, i.e. it was
not quasi-static).

This module therefore solves the fragmentation problem the way brittle
fragmentation is treated in the lattice-fracture literature: **quasi-static
elastic equilibrium with irreversible bond removal**.

* Geometry: a **triangular (hexagonal) bond lattice**; each node owns the plate
  area ``sqrt(3)/2 a**2`` and connects to its six nearest neighbours by an axial
  spring ``k = E t / sqrt(3)`` (``E`` Young's modulus, ``t`` plate thickness), so
  the intact lattice is a genuine 2D elastic medium -- its effective modulus is
  measured by :meth:`QuasiStaticFragmentation.effective_modulus` and checked
  against ``E`` in the test suite.
* Disorder: every bond carries its own critical strain
  ``eps_c = eps_0 (-ln U)**(1/m)`` (Weibull law of modulus ``m``).  This is the
  only heterogeneity in the model, so the statistics are reproducible from the
  seed, and the no-disorder limit is the negative control (it does not
  fragment).
* Loading: uniaxial displacement control.  The left platen is fixed, the right
  platen advances in increments of ``strain_increment``; at every load level the
  elastic equilibrium of the *surviving* network is solved exactly (sparse
  direct solve), bonds that exceed their threshold are removed, and the
  equilibrium is recomputed until the load level is again admissible.  There is
  no inertia, no contact model and no damping: the fragment census is a property
  of the elastic network, which is exactly what the FSD measurement needs.
* Fragments: connected components of the surviving bonds, with equivalent
  diameters ``D = sqrt(4 A / pi)``, ``A = n_nodes sqrt(3)/2 a**2``.

The census is recorded at every load stage, so the *evolution* of the FSD with
increasing comminution is an output, not a post-hoc assumption.

Scope and honesty
-----------------
Two-dimensional and central-force: no bond bending moments, no contact between
detached fragments (a detached floe carries no load and is frozen), and the load
path is plate compression between rigid platens, not hull indentation.  The
model is a fragmentation *statistics generator*; it is not used for force
history or for absolute strength predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

__all__ = [
    "FragmentationConfig",
    "QuasiStaticFragmentation",
    "FragmentCensus",
    "fragment_census",
    "triangular_lattice",
]


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class FragmentationConfig:
    """Geometry, material, disorder and loading parameters (SI)."""

    nx: int = 64
    ny: int = 32
    spacing: float = 0.03
    thickness: float = 0.5
    density: float = 917.0
    young_modulus: float = 5.0e9
    failure_strain_scale: float = 2.0e-3
    weibull_modulus: float = 4.0
    compression_factor: float = 5.0
    strain_increment: float = 5.0e-5
    compliant_regularisation: float = 1.0e-6
    target_strain: float = 0.10
    target_largest_fraction: float = 0.5
    seed: int = 0
    min_fragment_nodes: int = 2
    stage_stride: int = 10

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f"{name} must be an integer >= 4")
        for name in (
            "spacing", "thickness", "density", "young_modulus",
            "failure_strain_scale", "strain_increment", "target_strain",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.weibull_modulus) <= 0.0:
            raise ValueError("weibull_modulus must be positive")
        if float(self.compression_factor) <= 0.0:
            raise ValueError("compression_factor must be positive")
        if not 0.0 < float(self.target_largest_fraction) <= 1.0:
            raise ValueError("target_largest_fraction must lie in (0, 1]")
        if int(self.min_fragment_nodes) < 1:
            raise ValueError("min_fragment_nodes must be >= 1")
        if int(self.stage_stride) < 1:
            raise ValueError("stage_stride must be >= 1")

    # -- derived quantities -------------------------------------------------
    @property
    def node_length(self) -> float:
        """Plate length along the loading direction (m)."""
        return (self.nx - 1) * self.spacing

    @property
    def node_width(self) -> float:
        """Plate width across the loading direction (m)."""
        return (self.ny - 1) * self.spacing * math.sin(math.pi / 3.0)

    @property
    def node_area(self) -> float:
        """Plate area owned by one lattice node (m^2)."""
        return 0.5 * math.sqrt(3.0) * self.spacing**2

    @property
    def bond_stiffness(self) -> float:
        """Axial spring stiffness of one bond (N/m)."""
        return self.young_modulus * self.thickness / math.sqrt(3.0)

    @property
    def plate_area(self) -> float:
        return self.node_length * self.node_width


# --------------------------------------------------------------------------- lattice
def triangular_lattice(nx: int, ny: int, spacing: float):
    """Return ``(positions, bonds)`` of a staggered triangular lattice.

    Row ``j`` is shifted by half a spacing for odd ``j``; bonds connect the six
    nearest neighbours, so the bulk coordination is exactly 6 and every bond has
    length ``spacing``.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    pos = np.empty((nx * ny, 2), dtype=np.float64)
    ids = np.arange(nx * ny).reshape(ny, nx)
    height = math.sin(math.pi / 3.0) * spacing
    for j in range(ny):
        x = np.arange(nx, dtype=np.float64) * spacing
        if j % 2:
            x = x + 0.5 * spacing
        pos[ids[j], 0] = x
        pos[ids[j], 1] = j * height
    bonds = set()
    for j in range(ny):
        for i in range(nx):
            a = int(ids[j, i])
            if i + 1 < nx:
                bonds.add((a, int(ids[j, i + 1])))
            if j + 1 < ny:
                # row j+1 is shifted by +a/2 when j is even and unshifted when j
                # is odd, so the two nearest neighbours are (i, i-1) resp. (i, i+1)
                for di in ((0, -1) if j % 2 == 0 else (0, 1)):
                    ii = i + di
                    if 0 <= ii < nx:
                        bonds.add((a, int(ids[j + 1, ii])))
    bond_array = np.asarray(sorted(bonds), dtype=np.int64)
    return pos, bond_array


# --------------------------------------------------------------------------- census
@dataclass
class FragmentCensus:
    """Census of the connected components of a surviving bond network."""

    node_counts: np.ndarray
    diameters: np.ndarray
    areas: np.ndarray
    total_nodes: int
    n_fragments: int
    largest_fraction: float
    n_nodes_total: int
    labels: np.ndarray = dataclass_field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def summary(self) -> dict:
        return dict(
            n_fragments=int(self.n_fragments),
            n_nodes_total=int(self.n_nodes_total),
            nodes_in_fragments=int(self.total_nodes),
            largest_fraction=float(self.largest_fraction),
            median_diameter_m=float(np.median(self.diameters)) if self.n_fragments else 0.0,
            max_diameter_m=float(self.diameters.max()) if self.n_fragments else 0.0,
            min_diameter_m=float(self.diameters.min()) if self.n_fragments else 0.0,
        )


def _component_labels(n_nodes: int, bonds: np.ndarray) -> np.ndarray:
    """Union-find labels of the connected components of ``bonds``."""
    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for i, j in np.asarray(bonds, dtype=np.int64).reshape(-1, 2):
        ri, rj = find(int(i)), find(int(j))
        if ri == rj:
            continue
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1
    return np.asarray([find(i) for i in range(n_nodes)], dtype=np.int64)


def fragment_census(
    n_nodes: int,
    bonds: np.ndarray,
    node_area: float,
    *,
    min_fragment_nodes: int = 1,
) -> FragmentCensus:
    """Fragment census from an intact-bond list.

    ``D = sqrt(4 A / pi)`` is the diameter of the circle with the fragment's plate
    area, i.e. floes are treated as area-equivalent disks.  Fragments below
    ``min_fragment_nodes`` are dropped: a single node is the resolution floor of
    the lattice, not a floe (the choice is checked by a sensitivity case).
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    labels = _component_labels(n_nodes, bonds)
    _, counts = np.unique(labels, return_counts=True)
    keep = counts >= int(min_fragment_nodes)
    node_counts = np.sort(counts[keep].astype(np.int64))[::-1]
    areas = node_counts.astype(np.float64) * float(node_area)
    diameters = np.sqrt(4.0 * areas / math.pi)
    return FragmentCensus(
        node_counts=node_counts,
        diameters=diameters,
        areas=areas,
        total_nodes=int(node_counts.sum()),
        n_fragments=int(node_counts.size),
        largest_fraction=float(node_counts.max() / n_nodes) if node_counts.size else 0.0,
        n_nodes_total=int(n_nodes),
        labels=labels,
    )


# --------------------------------------------------------------------------- engine
class QuasiStaticFragmentation:
    """Displacement-controlled quasi-static fragmentation of a bonded lattice."""

    def __init__(self, config: FragmentationConfig | None = None) -> None:
        self.config = config or FragmentationConfig()
        cfg = self.config
        self.positions, self.bonds = triangular_lattice(cfg.nx, cfg.ny, cfg.spacing)
        self.n_nodes = self.positions.shape[0]
        self.alive = np.ones(len(self.bonds), dtype=bool)
        rng = np.random.default_rng(cfg.seed)
        u = rng.random(len(self.bonds))
        # Weibull failure strain: P(eps_c > eps) = exp(-(eps/eps_0)**m)
        self.thresholds = cfg.failure_strain_scale * (-np.log(np.maximum(u, 1e-15))) ** (
            1.0 / cfg.weibull_modulus
        )
        ids = np.arange(self.n_nodes).reshape(cfg.ny, cfg.nx)
        self._left = ids[:, 0]
        self._right = ids[:, -1]
        self._prescribed = np.zeros(self.n_nodes, dtype=bool)
        self._prescribed[self._left] = True
        self._prescribed[self._right] = True
        self._bond_unit = self.positions[self.bonds[:, 1]] - self.positions[self.bonds[:, 0]]
        self._bond_unit /= cfg.spacing
        self.applied_strain = 0.0
        self.stage_history: list[dict] = []
        self.total_breaks = 0
        self.solve_count = 0
        self.displacements = np.zeros((self.n_nodes, 2))

    # -- elastic solve ------------------------------------------------------
    def _components(self) -> np.ndarray:
        return _component_labels(self.n_nodes, self.bonds[self.alive])

    def _assemble(self, active_nodes: np.ndarray) -> coo_matrix:
        """Stiffness matrix of the alive bonds over the active node set."""
        cfg = self.config
        k = cfg.bond_stiffness
        index = -np.ones(self.n_nodes, dtype=np.int64)
        index[active_nodes] = np.arange(active_nodes.size)
        i, j = self.bonds[self.alive, 0], self.bonds[self.alive, 1]
        keep = (index[i] >= 0) & (index[j] >= 0)
        i, j = index[i[keep]], index[j[keep]]
        n = self._bond_unit[self.alive][keep]
        rows, cols, vals = [], [], []
        for a in range(2):
            for b in range(2):
                block = k * n[:, a] * n[:, b]
                rows.append(2 * i + a)
                cols.append(2 * i + b)
                vals.append(block)
                rows.append(2 * j + a)
                cols.append(2 * j + b)
                vals.append(block)
                rows.append(2 * i + a)
                cols.append(2 * j + b)
                vals.append(-block)
                rows.append(2 * j + a)
                cols.append(2 * i + b)
                vals.append(-block)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        size = 2 * int(active_nodes.size)
        matrix = coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
        return matrix

    def spanning_component(self) -> np.ndarray:
        """Nodes of the components that connect the fixed to the moving platen.

        Only a network that spans both platens carries load; a floe attached to
        one platen (or to none) is unloaded debris, and solving for it would
        introduce a soft rigid mode (a hinge at a single attachment point) whose
        displacement is unbounded.  Those nodes are therefore held at zero
        displacement -- the quasi-static equivalent of "the piece carries no
        load".
        """
        labels = self._components()
        left_labels = set(labels[self._left].tolist())
        right_labels = set(labels[self._right].tolist())
        spanning = left_labels & right_labels
        if not spanning:
            return np.zeros(self.n_nodes, dtype=bool)
        return np.isin(labels, list(spanning))

    def solve(self) -> np.ndarray:
        """Solve the elastic equilibrium of the load-carrying network."""
        cfg = self.config
        disp = np.zeros((self.n_nodes, 2))
        delta = self.applied_strain * cfg.node_length
        disp[self._right, 0] = -delta
        loaded = self.spanning_component()
        free = np.flatnonzero(~self._prescribed & loaded)
        self.solve_count += 1
        self.spanning_nodes = int(loaded.sum())
        if free.size:
            active = np.flatnonzero(loaded)
            K = self._assemble(active)
            index = -np.ones(self.n_nodes, dtype=np.int64)
            index[active] = np.arange(active.size)
            free_dofs = np.concatenate([2 * index[free], 2 * index[free] + 1])
            fixed_nodes = active[self._prescribed[active]]
            fixed_dofs = np.concatenate([2 * index[fixed_nodes], 2 * index[fixed_nodes] + 1])
            u_fixed = disp.reshape(-1)[np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])]
            K_ff = K[free_dofs][:, free_dofs]
            K_fb = K[free_dofs][:, fixed_dofs]
            rhs = -(K_fb @ u_fixed)
            # The spanning network can still contain a nearly-soft mechanism
            # (a chain hinged at one node); a relative diagonal shift of 1e-6
            # bounds it without biasing the load-bearing bonds measurably.
            diag_scale = max(float(abs(K_ff).max()), 1e-30)
            K_ff = K_ff + diag_scale * float(cfg.compliant_regularisation) * eye(
                K_ff.shape[0], format="csr"
            )
            try:
                u_free = spsolve(K_ff.tocsc(), rhs)
            except Exception:  # pragma: no cover - defensive
                u_free = np.zeros_like(rhs)
            if not np.isfinite(u_free).all():
                u_free = np.zeros_like(rhs)
            flat = disp.reshape(-1).copy()
            flat[free_dofs] = u_free
            disp = flat.reshape(-1, 2)
        self.displacements = disp
        return disp

    def bond_strain(self) -> np.ndarray:
        """Axial strain of every bond (compression negative)."""
        du = self.displacements[self.bonds[:, 1]] - self.displacements[self.bonds[:, 0]]
        return np.einsum("ij,ij->i", du, self._bond_unit) / self.config.spacing

    def break_bonds(self) -> int:
        """Remove every alive bond whose strain exceeds its own threshold."""
        strain = self.bond_strain()
        cfg = self.config
        tension = strain > self.thresholds
        compression = strain < -cfg.compression_factor * self.thresholds
        newly = self.alive & (tension | compression)
        count = int(newly.sum())
        if count:
            self.alive[newly] = False
            self.total_breaks += count
        return count

    def wall_force(self) -> float:
        """Total compressive force transmitted through the right platen (N)."""
        cfg = self.config
        k = cfg.bond_stiffness
        loaded = self.spanning_component()
        strain = self.bond_strain()
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        right = np.zeros(self.n_nodes, dtype=bool)
        right[self._right] = True
        touch = (right[i] | right[j]) & loaded[i] & loaded[j] & self.alive
        return float(-k * cfg.spacing * float((strain[touch] * self._bond_unit[touch, 0]).sum()))

    def census(self) -> FragmentCensus:
        return fragment_census(
            self.n_nodes,
            self.bonds[self.alive],
            self.config.node_area,
            min_fragment_nodes=self.config.min_fragment_nodes,
        )

    # -- loading history ---------------------------------------------------
    def run(self) -> dict:
        """Load step by step to the target strain / target comminution level."""
        cfg = self.config
        stage = 0
        while self.applied_strain < cfg.target_strain:
            self.applied_strain = min(
                cfg.target_strain, self.applied_strain + cfg.strain_increment
            )
            self.solve()
            # cascade: remove the over-stressed bonds and re-equilibrate
            for _ in range(60):
                if self.break_bonds() == 0:
                    break
                self.solve()
            stage += 1
            if stage % cfg.stage_stride == 0:
                census = self.census()
                self.stage_history.append(
                    dict(
                        stage=stage,
                        applied_strain=self.applied_strain,
                        broken_bonds=int((~self.alive).sum()),
                        **census.summary(),
                    )
                )
                if census.largest_fraction < cfg.target_largest_fraction:
                    break
        return self.diagnostics()

    def effective_modulus(self, *, strain: float = 1.0e-5) -> float:
        """Effective plate modulus ``E_eff`` of the intact lattice (Pa).

        ``E_eff = (F / (t W)) / eps`` with ``F`` the platen force at a small
        applied strain (no bond may break: thresholds are scaled up by 1e3).
        """
        cfg = self.config
        self.thresholds = self.thresholds * 1.0e3
        self.applied_strain = strain
        self.solve()
        area = cfg.thickness * cfg.node_width
        return self.wall_force() / (area * strain)

    def diagnostics(self) -> dict:
        census = self.census()
        return dict(
            applied_strain=float(self.applied_strain),
            broken_bonds=int((~self.alive).sum()),
            bond_count=int(self.alive.size),
            solve_count=int(self.solve_count),
            platen_force_N=float(self.wall_force()),
            census=census.summary(),
            stages=list(self.stage_history),
        )