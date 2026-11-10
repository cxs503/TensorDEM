"""Fracture-face extraction and floe partition on the FEM lattice.

What is computed here
---------------------
Given the moment field of :mod:`tensordem.fem_dem_fracture.plate_fem`, this
module produces the two things a DEM handoff needs:

1. **a crack network** -- polylines that live on the *edges* of the FEM mesh, and
2. **floe polygons** -- the resulting partition of the plate into pieces, with
   area, centroid, perimeter, equivalent diameter and mass per piece.

The crack network lives on mesh edges on purpose.  In a lattice fracture model
the fracture surface *is* a set of removed bonds/edges, and if the crack graph is
edge-aligned then the floe partition is exact: every cell belongs to exactly one
floe, the shoelace area of the traced floe polygon equals the sum of its cell
areas to machine precision, and ``sum(A_floe) = A_plate`` is verified rather than
assumed.  A crack crossing cell interiors would need polygon clipping and would
make the area ledger approximate; the price of the edge-aligned choice is that
crack paths are staircase approximations of the smooth paths (documented, and
measured in the mesh-refinement study of
``docs/FEM_DEM_FRACTURE_HANDOFF.md``: the floe *count* is stable, the crack
azimuth carries an O(h) quantisation).

The crack law (frozen-field mode-I growth)
------------------------------------------
This is deliberately a *first-order* fracture model, stated as such:

* **Nucleation**: a crack nucleates at a node where the flexural stress reaches
  the nucleation fraction of ``sigma_flexure`` and which is a local maximum
  within ``min_seed_separation``.  The separation is the model's process-zone
  parameter; a sensitivity study over it is part of the report, not a hidden
  constant.
* **Propagation**: a crack opens under the tensile bending stress whose axis is
  the principal moment of larger magnitude, so it runs *along* the direction
  perpendicular to that axis (mode-I: crack plane normal = tensile axis).  The
  tracer takes one cell-edge step at a time, choosing among the four lattice
  directions the one best aligned with the local perpendicular of the principal
  moment, with the current direction as tie-breaker (momentum).
* **Arrest**: propagation stops at a plate edge, at another crack
  (coalescence), or where the local stress has decayed below
  ``arrest_fraction * sigma_flexure``.

The stress field is *frozen* at the collapse load for all tips: there is no
crack-tip redistribution, no crack-crack elastic interaction and no inertia.
This is the standard "stress-field-guided" (uncoupled) fragmentation
approximation; it is good for the *pattern topology* and for the floe census, and
it is not a fracture-mechanics prediction of crack speed or of the post-collapse
load path.  What it does give is a physically conservative datum for the DEM: the
sheet breaks where the FEM says the moment peaks, and the piece sizes follow from
the geometry of the intersection of those crack paths.

References for the physics: Timoshenko & Woinowsky-Krieger (1959) for the plate
moment criterion and the "radial cracks around a loaded area" pattern; Dempsey
(1991) and Bazant (2002) for the size-effect caveat that the strength-based
nucleation criterion used here is a simplification of the cohesive process zone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .plate_fem import PlateMesh, PlateSolution

__all__ = [
    "CrackTracerConfig",
    "SplitState",
    "find_splitting_state",
    "Crack",
    "Floe",
    "CrackNetwork",
    "trace_crack_network",
    "polygon_area",
    "polygon_centroid",
    "polygon_perimeter",
    "point_in_polygon",
]

_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --------------------------------------------------------------------------
# polygon utilities
# --------------------------------------------------------------------------
def polygon_area(polygon: np.ndarray) -> float:
    """Signed shoelace area of a closed polygon [m**2] (CCW positive)."""
    p = np.asarray(polygon, dtype=float)
    if len(p) < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """Area centroid of a closed polygon."""
    p = np.asarray(polygon, dtype=float)
    area = polygon_area(p)
    if abs(area) < 1e-15:
        return p.mean(axis=0)
    x = p[:, 0]
    y = p[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    return np.array([float(np.sum((x + xn) * cross)), float(np.sum((y + yn) * cross))]) / (6.0 * area)


def polygon_perimeter(polygon: np.ndarray) -> float:
    """Closed-polygon perimeter [m]."""
    p = np.asarray(polygon, dtype=float)
    if len(p) < 2:
        return 0.0
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def point_in_polygon(point, polygon) -> bool:
    """Ray-casting containment test (used to prove the floe partition)."""
    x, y = float(point[0]), float(point[1])
    p = np.asarray(polygon, dtype=float)
    inside = False
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


# --------------------------------------------------------------------------
# configuration and records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CrackTracerConfig:
    """Model parameters of the frozen-field crack tracer.

    ``weibull_modulus``
        Scatter of the *local* flexural strength, ``sigma_c = sigma_f
        (-ln U)**(1/m) / (ln 2)**(1/m)`` (``U`` uniform, median ``= sigma_f``).
        Ice is a brittle disordered material and its strength scatter is what
        decides where cracks actually nucleate; with no scatter (``m -> inf``)
        a symmetric load produces a single, perfectly symmetric nucleation site,
        which is both unphysical and numerically degenerate.  ``m = 4`` is inside
        the 3--6 band quoted for sea ice in the literature (Timco & Weeks 2010
        review; Weibull-modulus values of 3--6 for first-year ice flexural
        strength).  This is the same disorder idiom the package already uses for
        bonded-lattice fragmentation
        (:mod:`tensordem.broken_ice_micro.fragmentation`).
    ``min_seed_separation``
        Closest two nucleation sites may be [m].  A continuum model cannot
        nucleate two cracks closer than the size of the process zone, so this is
        the process-zone surrogate; the report gives the sensitivity of the floe
        census to it.
    ``seed``
        RNG seed for the strength field, so the whole handoff is reproducible
        bit for bit.
    """

    weibull_modulus: float = 4.0
    min_seed_separation: float = 1.5   # [m]
    arrest_fraction: float = 0.25      # stop when sigma < fraction * sigma_f
    max_steps_per_tip: int = 400       # safety net (~2x plate diagonal in cells)
    seed: int = 20261110

    def __post_init__(self) -> None:
        if self.weibull_modulus <= 0.0:
            raise ValueError("weibull_modulus must be positive")
        if self.min_seed_separation <= 0.0:
            raise ValueError("min_seed_separation must be positive")
        if not 0.0 <= self.arrest_fraction < 1.0:
            raise ValueError("arrest_fraction must lie in [0, 1)")


@dataclass
class Crack:
    """One crack polyline on the mesh lattice."""

    crack_id: int
    nodes: list[int]              # lattice node ids, ordered start -> end
    points: np.ndarray            # (n, 2) coordinates of those nodes
    seed_node: int
    tip_end: str                  # "boundary" | "coalescence" | "arrest" | "max_steps"

    @property
    def length(self) -> float:
        """Arc length of the polyline [m] (open polyline, no closing segment)."""
        if len(self.points) < 2:
            return 0.0
        d = np.diff(self.points, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

    def as_polyline(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.points]


@dataclass
class Floe:
    """One DEM body: the polygon of a detached piece of the sheet."""

    floe_id: int
    cells: list[int]
    outer: np.ndarray             # (m, 2) outer boundary polygon
    holes: list[np.ndarray]
    area: float                   # [m**2]
    centroid: np.ndarray          # (2,) [m]
    perimeter: float              # [m]
    equivalent_diameter: float    # [m] sqrt(4 A / pi)
    mass: float                   # [kg]
    max_sigma: float              # [Pa] largest FEM stress inside the floe
    mean_sigma: float

    def all_polygons(self) -> list[list[list[float]]]:
        """Outer ring first, then holes -- the format written to JSON."""
        rings = [[[float(x), float(y)] for x, y in self.outer]]
        rings += [[[float(x), float(y)] for x, y in hole] for hole in self.holes]
        return rings

    def contains(self, point) -> bool:
        if not point_in_polygon(point, self.outer):
            return False
        return not any(point_in_polygon(point, hole) for hole in self.holes)


@dataclass
class CrackNetwork:
    """Output of :func:`trace_crack_network`."""

    cracks: list[Crack]
    cut_edges: set
    floes: list[Floe]
    seed_nodes: list[int]
    sigma_flexure: float
    config: CrackTracerConfig
    meta: dict = field(default_factory=dict)

    @property
    def n_floe(self) -> int:
        return len(self.floes)

    @property
    def total_crack_length(self) -> float:
        return float(sum(c.length for c in self.cracks))

    def areas(self) -> np.ndarray:
        return np.array([f.area for f in self.floes])

    def diameters(self) -> np.ndarray:
        return np.array([f.equivalent_diameter for f in self.floes])

    def masses(self) -> np.ndarray:
        return np.array([f.mass for f in self.floes])


# --------------------------------------------------------------------------
# nodal fields
# --------------------------------------------------------------------------
def _nodal_crack_direction(solution: PlateSolution) -> tuple[np.ndarray, np.ndarray]:
    """Node-level stress and crack-propagation direction.

    The crack-propagation direction at a node is the perpendicular of the
    principal moment of larger magnitude, averaged over the adjacent cells with
    the local stress as weight.  Directions are lines (``mod pi``), so the
    average is taken in doubled angle space.
    """
    mesh = solution.mesh
    cells = mesh.cell_ids()
    sigma = solution.sigma_max
    angle = solution.principal_angle
    vx = np.zeros(mesh.n_nodes)
    vy = np.zeros(mesh.n_nodes)
    wsum = np.zeros(mesh.n_nodes)
    node_sigma = np.zeros(mesh.n_nodes)
    # crack direction = principal axis + pi/2 -> doubled angle = 2*angle + pi
    cx = np.cos(2.0 * angle + math.pi)
    cy = np.sin(2.0 * angle + math.pi)
    for k in range(4):
        ids = cells[:, k]
        np.add.at(vx, ids, sigma * cx)
        np.add.at(vy, ids, sigma * cy)
        np.add.at(wsum, ids, sigma)
        np.maximum.at(node_sigma, ids, sigma)
    norm = np.hypot(vx, vy)
    ok = norm > 1e-30
    ux = np.zeros(mesh.n_nodes)
    uy = np.zeros(mesh.n_nodes)
    ux[ok] = vx[ok] / norm[ok]
    uy[ok] = vy[ok] / norm[ok]
    ux[~ok] = 1.0
    return node_sigma, np.stack([ux, uy], axis=1)


def _cell_ahead_direction(solution: PlateSolution, node: int, direction: tuple[int, int]):
    """Average crack direction of the two cells adjacent to edge(node, node+dir)."""
    mesh = solution.mesh
    i, j = mesh.node_index(node)
    sigma = solution.sigma_max
    angle = solution.principal_angle
    di, dj = direction
    edge_i = min(i, i + di)
    edge_j = min(j, j + dj)
    if di != 0:      # vertical edge at x = edge_i, spanning cell rows edge_j-1 and edge_j
        candidates = [(edge_i - 1, edge_j), (edge_i, edge_j)]
    else:            # horizontal edge at y = edge_j, columns edge_i-1 and edge_i
        candidates = [(edge_i, edge_j - 1), (edge_i, edge_j)]
    vx = vy = wsum = 0.0
    for (ci, cj) in candidates:
        if 0 <= ci < mesh.nx and 0 <= cj < mesh.ny:
            cell = cj * mesh.nx + ci
            vx += sigma[cell] * math.cos(2.0 * angle[cell] + math.pi)
            vy += sigma[cell] * math.sin(2.0 * angle[cell] + math.pi)
            wsum += sigma[cell]
    if wsum <= 0.0 or math.hypot(vx, vy) < 1e-30:
        return None, 0.0
    return np.array([vx, vy]) / math.hypot(vx, vy), wsum


# --------------------------------------------------------------------------
# crack tracing
# --------------------------------------------------------------------------
def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _nodal_strength(mesh: PlateMesh, sigma_flexure: float, cfg: CrackTracerConfig) -> np.ndarray:
    """Weibull-scattered local flexural strength per node [Pa] (median = sigma_f)."""
    rng = np.random.default_rng(cfg.seed)
    u = np.clip(rng.random(mesh.n_nodes), 1e-12, 1.0 - 1e-12)
    weib = (-np.log(u)) ** (1.0 / cfg.weibull_modulus)
    weib /= (math.log(2.0)) ** (1.0 / cfg.weibull_modulus)
    return sigma_flexure * weib


def _select_seeds(
    node_sigma: np.ndarray,
    node_strength: np.ndarray,
    mesh: PlateMesh,
    cfg: CrackTracerConfig,
) -> list[int]:
    """Nucleation sites: nodes where ``sigma >= sigma_c``, strongest ratio first.

    Greedy non-maximum suppression at ``min_seed_separation`` removes the sites
    that would sit inside the process zone of an already nucleated crack.
    """
    xy = mesh.node_xy()
    ratio = node_sigma / node_strength
    order = np.argsort(-ratio, kind="stable")
    chosen: list[int] = []
    for node in order:
        if ratio[node] < 1.0:
            break
        if chosen:
            d = np.hypot(*(xy[np.asarray(chosen)] - xy[node]).T)
            if np.any(d < cfg.min_seed_separation):
                continue
        chosen.append(int(node))
    return chosen


def trace_crack_network(
    solution: PlateSolution,
    config: CrackTracerConfig | None = None,
) -> CrackNetwork:
    """Trace the crack network of a plate at its collapse load and build floes.

    The solution is expected to be scaled so that ``max(sigma) = sigma_flexure``
    (see :meth:`PlateSolution.scale_to_failure`); the seed and arrest thresholds
    are fractions of ``sigma_flexure``.
    """
    cfg = config or CrackTracerConfig()
    mesh = solution.mesh
    sigma_f = solution.material.sigma_flexure
    node_sigma, node_dir = _nodal_crack_direction(solution)
    boundary = mesh.boundary_node_mask()
    nx, ny = mesh.nx, mesh.ny

    node_strength = _nodal_strength(mesh, sigma_f, cfg)
    seeds = _select_seeds(node_sigma, node_strength, mesh, cfg)
    if not seeds:
        raise RuntimeError(
            "no node reached its local flexural strength: the plate is below its "
            "fracture load (honest no-fracture outcome)"
        )

    cut_edges: set[tuple[int, int]] = set()
    crack_nodes: set[int] = set()
    cracks: list[Crack] = []

    def step_direction(node: int, current: tuple[int, int]) -> tuple[tuple[int, int], float] | None:
        """Best lattice direction to continue a tip at ``node``."""
        ahead, weight = _cell_ahead_direction(solution, node, current)
        reference = ahead if ahead is not None else node_dir[node]
        best = None
        for cand in _DIRS4:
            if (cand[0] == -current[0] and cand[1] == -current[1]):
                continue                      # never walk back along the crack
            i, j = mesh.node_index(node)
            ti, tj = i + cand[0], j + cand[1]
            if not (0 <= ti <= nx and 0 <= tj <= ny):
                continue
            score = abs(cand[0] * reference[0] + cand[1] * reference[1])
            if cand == current:               # momentum breaks the diagonal ties
                score += 1e-6
            score += 1e-9 * weight
            if best is None or score > best[0]:
                best = (score, cand)
        if best is None:
            return None
        return best[1], best[0]

    def grow(seed: int, direction: tuple[int, int]) -> tuple[list[int], str]:
        path = [seed]
        current = direction
        node = seed
        reason = "max_steps"
        for _ in range(cfg.max_steps_per_tip):
            choice = step_direction(node, current)
            if choice is None:
                reason = "boundary"
                break
            cand = choice[0]
            i, j = mesh.node_index(node)
            nxt = mesh.node_id(i + cand[0], j + cand[1])
            if node_sigma[nxt] < cfg.arrest_fraction * sigma_f and not boundary[nxt]:
                reason = "arrest"
                break
            cut_edges.add(_edge_key(node, nxt))
            path.append(nxt)
            current = cand
            if boundary[nxt]:
                node = nxt
                reason = "boundary"
                break
            if nxt in crack_nodes:
                node = nxt
                reason = "coalescence"
                break
            node = nxt
        crack_nodes.update(path)
        return path, reason

    xy = mesh.node_xy()
    for seed in seeds:
        primary = node_dir[seed]
        # Snap the two initial tips to the lattice.
        cands = sorted(
            _DIRS4,
            key=lambda c: (-abs(c[0] * primary[0] + c[1] * primary[1]), _DIRS4.index(c)),
        )
        first = cands[0]
        second = (-first[0], -first[1])
        paths = []
        for direction in (first, second):
            path, reason = grow(seed, direction)
            paths.append((path, reason))
        # Merge the two arms through the seed into one polyline.
        arm_a, reason_a = paths[0]
        arm_b, reason_b = paths[1]
        nodes = list(reversed(arm_a[1:])) + [seed] + arm_b[1:]
        reason = reason_a if reason_a == reason_b else f"{reason_a}|{reason_b}"
        cracks.append(
            Crack(
                crack_id=len(cracks),
                nodes=nodes,
                points=xy[np.asarray(nodes)],
                seed_node=int(seed),
                tip_end=reason,
            )
        )

    floes = _build_floes(solution, cut_edges)
    return CrackNetwork(
        cracks=cracks,
        cut_edges=cut_edges,
        floes=floes,
        seed_nodes=[int(s) for s in seeds],
        sigma_flexure=sigma_f,
        config=cfg,
        meta={
            "mesh": (mesh.nx, mesh.ny),
            "cut_edges": len(cut_edges),
            "weibull_modulus": cfg.weibull_modulus,
            "seed": cfg.seed,
            "min_seed_separation": cfg.min_seed_separation,
            "arrest_threshold": cfg.arrest_fraction * sigma_f,
            "max_sigma_over_strength": float(np.max(node_sigma / node_strength)),
        },
    )


# --------------------------------------------------------------------------
# floe partition
# --------------------------------------------------------------------------
def _cell_edges(mesh: PlateMesh, cell: int) -> list[tuple[int, int, int]]:
    """The four (node_a, node_b) edges of a cell, CCW, with their neighbour cell.

    Returns ``(n0, n1, neighbour_cell_or_-1)`` quadruples in the *directed* CCW
    sense of the cell.
    """
    i, j = mesh.cell_index(cell)
    n00 = mesh.node_id(i, j)
    n10 = mesh.node_id(i + 1, j)
    n11 = mesh.node_id(i + 1, j + 1)
    n01 = mesh.node_id(i, j + 1)
    below = (j - 1) * mesh.nx + i if j > 0 else -1          # shares edge n00-n10
    right = j * mesh.nx + (i + 1) if i + 1 < mesh.nx else -1  # shares n10-n11
    above = (j + 1) * mesh.nx + i if j + 1 < mesh.ny else -1  # shares n11-n01
    left = j * mesh.nx + (i - 1) if i > 0 else -1             # shares n01-n00
    return [(n00, n10, below), (n10, n11, right), (n11, n01, above), (n01, n00, left)]


def _trace_loops(directed_edges: list[tuple[int, int]]) -> list[list[int]]:
    """Chain directed boundary edges into closed node loops.

    Every node of a cell-complex boundary has as many outgoing as incoming
    boundary edges, so walking and consuming edges always closes a loop; the
    walk is edge-consuming (never node-consuming) so the procedure terminates
    even where two floes touch at a single node (a pinch, which is common when a
    staircase crack passes a cell corner).
    """
    outgoing: dict[int, list[int]] = {}
    for a, b in directed_edges:
        outgoing.setdefault(a, []).append(b)
    loops: list[list[int]] = []
    for start in list(outgoing):
        while outgoing.get(start):
            loop = [start]
            current = start
            while True:
                options = outgoing.get(current)
                if not options:
                    loop = []          # open chain: numerically impossible, drop
                    break
                nxt = options.pop(0)
                if nxt == start:
                    break
                loop.append(nxt)
                current = nxt
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _build_floes(solution: PlateSolution, cut_edges: set) -> list[Floe]:
    """Connected components of cells under the cut-edge blocking."""
    mesh = solution.mesh
    material = solution.material
    n_cells = mesh.n_cells
    parent = list(range(n_cells))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for cell in range(n_cells):
        for (n0, n1, neighbour) in _cell_edges(mesh, cell):
            if neighbour < 0:
                continue
            if _edge_key(n0, n1) in cut_edges:
                continue
            if neighbour > cell:
                union(cell, neighbour)

    groups: dict[int, list[int]] = {}
    for cell in range(n_cells):
        groups.setdefault(find(cell), []).append(cell)

    xy = mesh.node_xy()
    floes: list[Floe] = []
    for root, cells in sorted(groups.items()):
        cell_set = set(cells)
        edge_use: dict[tuple[int, int], int] = {}
        directed: list[tuple[int, int]] = []
        for cell in cells:
            for (n0, n1, neighbour) in _cell_edges(mesh, cell):
                key = _edge_key(n0, n1)
                edge_use[key] = edge_use.get(key, 0) + 1
                if neighbour < 0 or neighbour not in cell_set:
                    directed.append((n0, n1))
        loops = _trace_loops(directed)
        if not loops:
            continue
        rings = [np.array([xy[n] for n in loop]) for loop in loops]
        signed = np.array([polygon_area(r) for r in rings])
        outer_idx = int(np.argmax(np.abs(signed)))
        outer = rings[outer_idx]
        holes = [r for k, r in enumerate(rings) if k != outer_idx and abs(signed[k]) > 1e-12]
        area = float(abs(signed[outer_idx]) - sum(abs(polygon_area(h)) for h in holes))
        cell_area_sum = len(cells) * mesh.cell_area
        if abs(area - cell_area_sum) > 1e-9 * max(1.0, cell_area_sum):
            raise RuntimeError(
                f"floe polygon area {area!r} disagrees with its cell sum {cell_area_sum!r}; "
                "the boundary tracing is inconsistent"
            )
        centre = np.mean([xy[mesh.cell_ids()[c]] for c in cells], axis=(0, 1))
        floes.append(
            Floe(
                floe_id=len(floes),
                cells=sorted(cells),
                outer=outer,
                holes=holes,
                area=area,
                centroid=centre,
                perimeter=polygon_perimeter(outer) + sum(polygon_perimeter(h) for h in holes),
                equivalent_diameter=math.sqrt(4.0 * area / math.pi),
                mass=area * material.area_density,
                max_sigma=float(np.max(solution.sigma_max[cells])),
                mean_sigma=float(np.mean(solution.sigma_max[cells])),
            )
        )
    return floes


# --------------------------------------------------------------------------
# load ramp: the state handed to DEM
# --------------------------------------------------------------------------
@dataclass
class SplitState:
    """The fracture state that splits the sheet, plus how it was found."""

    load_factor: float
    network: CrackNetwork
    largest_floe_fraction: float
    evaluations: int
    history: list[dict] = field(default_factory=list)


def _largest_fraction(network: CrackNetwork, plate_area: float) -> float:
    return float(np.max(network.areas()) / plate_area)


def find_splitting_state(
    solution_at_first_cracking: PlateSolution,
    config: CrackTracerConfig | None = None,
    *,
    target_largest_fraction: float = 0.8,
    tol: float = 0.02,
    max_load_factor: float = 50.0,
) -> SplitState:
    """Smallest load factor at which the crack network splits the sheet.

    ``solution_at_first_cracking`` must be scaled so that
    ``max(sigma) = sigma_flexure`` (call :meth:`PlateSolution.scale_to_failure`
    first).  The load factor ``lambda`` multiplies that field.  The *handoff*
    state of the campaign is the first state in which no single piece keeps more
    than ``target_largest_fraction`` of the plate -- i.e. the sheet is broken up,
    not merely cracked.

    Why a ramp at all: at ``lambda = 1`` only the immediate neighbourhood of the
    load has reached the strength, so the trace gives a few arrested cracks and a
    still-intact sheet (measured: 9 pieces, largest 91 % of the area).  A level
    ice sheet does not stop there, and the DEM needs the broken-up state.  The
    stress field is *frozen* (linear, no redistribution), so ``lambda`` is a
    damage-accumulation parameter, not a load history; the largest-floe fraction
    decreases monotonically with it in every case measured, which is what makes
    the bisection well posed.  The trap is that ``lambda`` is **not** a bearing
    capacity: a redistributing model would predict a lower load at this degree of
    damage -- this is stated again in the report's limitation section.
    """
    cfg = config or CrackTracerConfig()
    plate_area = solution_at_first_cracking.mesh.plate_area
    base_amplitude = solution_at_first_cracking.load_amplitude
    history: list[dict] = []

    def evaluate(lam: float) -> tuple[CrackNetwork, float]:
        net = trace_crack_network(
            solution_at_first_cracking.scaled(lam * base_amplitude), cfg
        )
        frac = _largest_fraction(net, plate_area)
        history.append({"load_factor": round(lam, 6), "largest_floe_fraction": round(frac, 6),
                        "n_floe": net.n_floe})
        return net, frac

    lo = 1.0
    net, frac = evaluate(lo)
    if frac <= target_largest_fraction:
        return SplitState(load_factor=lo, network=net, largest_floe_fraction=frac,
                          evaluations=1, history=history)
    hi = lo
    while hi < max_load_factor:
        hi = min(max_load_factor, hi * 1.5)
        net, frac = evaluate(hi)
        if frac <= target_largest_fraction:
            break
    else:
        raise RuntimeError(
            f"the crack network never splits the plate up to load factor {max_load_factor}"
        )
    while (hi - lo) / hi > tol:
        mid = 0.5 * (lo + hi)
        net_mid, frac_mid = evaluate(mid)
        if frac_mid <= target_largest_fraction:
            hi, net, frac = mid, net_mid, frac_mid
        else:
            lo = mid
    return SplitState(load_factor=hi, network=net, largest_floe_fraction=frac,
                      evaluations=len(history), history=history)
