"""The FEM -> DEM handoff artefact: ``failure_faces.json`` and its DEM reader.

Contract
--------
The continuous solver (:mod:`tensordem.fem_dem_fracture.plate_fem` +
:mod:`tensordem.fem_dem_fracture.crack_network`) writes one JSON file that
contains everything the discrete side needs to build initial conditions:

* ``meta``   -- plate geometry, material, mesh, load case, the stress state at
  which fracture was declared (``sigma_max`` vs ``sigma_flexure``), and the code
  provenance (version + payload SHA-256);
* ``cracks`` -- the fracture **polylines** (crack paths, in metres);
* ``fractured_elements`` -- the FEM cells whose flexural stress reached the
  strength, as polygons (the "断裂单元");
* ``floes``  -- the floe **polygons** with area, centroid, equivalent diameter,
  perimeter and mass: the碎块 size distribution is simply the list of
  equivalent diameters;
* ``fsd``    -- the floe-size census and a power-law fit (from
  :mod:`tensordem.broken_ice_micro.fsd`);
* ``conservation`` -- the closed ledger (mass and area of the plate vs the sum
  over floes, plus a boundary-length ledger).

The reader (:func:`read_failure_faces`) **recomputes** the ledger from the file
alone -- it does not trust the stored numbers -- and only then emits DEM bodies.
That is the point of the handoff: the discrete solver can be started from a file
written by a different process, on a different day, and still prove that no mass
and no area were created or lost.

Conservation statements (both are exact, not approximate)
---------------------------------------------------------
Mass: ``sum_j m_j = t rho sum_j A_j`` and ``sum_j A_j = A_plate`` because the
floe polygons partition the plate exactly (the crack network runs on mesh edges,
see :mod:`tensordem.fem_dem_fracture.crack_network`).  Residuals are therefore at
round-off level (~1e-12 kg on ~4.6e4 kg) and are *reported*, not assumed.

Geometry: the boundary ledger
``sum_j perimeter_j = perimeter_plate + 2 * L_internal_crack``
holds for any partition of a polygon: every internal fracture face is shared by
two floes and the outer boundary by one.  This is checked separately from the
area sum, so a polygon-tracing bug (double-counted or missing edges) cannot hide
behind a correct area total.

Bodies handed to DEM
--------------------
Each floe becomes one DEM body at its area centroid with

* ``diameter = sqrt(4 A / pi)`` (the equivalent-diameter disc, whose area is
  *exactly* the floe area, so disc-based mass bookkeeping also conserves mass),
* ``mass = A t rho_ice``,
* ``velocity = 0`` for a quasi-static FEM solve (the field is static; a velocity
  can be injected by the caller), and
* its polygon, for shape-resolved contact models.

Known limitation, stated up front: discs of equivalent area placed at the
centroids of *non-convex* floes can overlap.  The polygon is shipped precisely so
that a shape-resolved DEM does not have to use the disc.  The handoff therefore
verifies mass and area conservation and *not* non-overlap of the disc
idealisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

from .crack_network import CrackNetwork, polygon_area, polygon_perimeter

__all__ = [
    "SCHEMA",
    "FloeRecord",
    "CrackRecord",
    "FailureFaces",
    "ConservationLedger",
    "DemInitialConditions",
    "build_failure_faces",
    "read_failure_faces",
    "dem_initial_conditions",
    "check_conservation",
]

SCHEMA = "tensordem.failure_faces/1"
_CODE_VERSION = "tensordem.fem_dem_fracture 1.0 (campaign p302w1-n03)"


def _round(x, ndigits: int = 12):
    return float(round(float(x), ndigits))


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ConservationLedger:
    """Closed mass / area / boundary ledger of a fracture state.

    ``rim_length_from_polygons`` and ``internal_face_length`` are rebuilt from
    the floe polygons by segment bookkeeping (a segment shared by two floes is a
    fracture face, a segment used once is on the plate rim), so the rim residual
    ``rim_length_from_polygons - plate_perimeter`` is a genuine independent check
    of the boundary tracing.  (The first version of this ledger *defined* the
    internal length as ``(sum(perimeter) - P_plate)/2``, which makes the boundary
    residual identically zero -- a tautology that cannot fail.  That is replaced
    here.)
    """

    plate_area: float
    floes_area_sum: float
    area_residual: float
    plate_mass: float
    floes_mass_sum: float
    mass_residual: float
    plate_perimeter: float
    floes_perimeter_sum: float
    rim_length_from_polygons: float
    rim_residual: float
    internal_face_length: float
    n_floe: int
    n_segments_used_three_times: int = 0

    def as_dict(self) -> dict:
        return {
            "plate_area_m2": _round(self.plate_area),
            "floes_area_sum_m2": _round(self.floes_area_sum),
            "area_residual_m2": _round(self.area_residual),
            "plate_mass_kg": _round(self.plate_mass),
            "floes_mass_sum_kg": _round(self.floes_mass_sum),
            "mass_residual_kg": _round(self.mass_residual),
            "plate_perimeter_m": _round(self.plate_perimeter),
            "floes_perimeter_sum_m": _round(self.floes_perimeter_sum),
            "rim_length_from_polygons_m": _round(self.rim_length_from_polygons),
            "rim_residual_m": _round(self.rim_residual),
            "internal_face_length_m": _round(self.internal_face_length),
            "n_floe": self.n_floe,
            "n_segments_used_three_times": int(self.n_segments_used_three_times),
        }


@dataclass(frozen=True)
class FloeRecord:
    floe_id: int
    area: float
    mass: float
    centroid: tuple[float, float]
    equivalent_diameter: float
    perimeter: float
    n_cells: int
    polygons: list[list[list[float]]]     # outer ring first, then holes
    max_sigma: float
    mean_sigma: float

    def polygon_area(self) -> float:
        """Area recomputed from the polygons alone (outer minus holes)."""
        outer = polygon_area(np.asarray(self.polygons[0]))
        holes = sum(abs(polygon_area(np.asarray(h))) for h in self.polygons[1:])
        return abs(outer) - holes

    def contains(self, point) -> bool:
        """Inside the outer ring and outside every hole."""
        from .crack_network import point_in_polygon as _pip

        if not _pip(point, np.asarray(self.polygons[0])):
            return False
        return not any(_pip(point, np.asarray(h)) for h in self.polygons[1:])


@dataclass(frozen=True)
class CrackRecord:
    crack_id: int
    polyline: list[list[float]]
    length: float
    seed_xy: tuple[float, float]
    tip_end: str
    n_nodes: int


@dataclass
class FailureFaces:
    """In-memory image of ``failure_faces.json``."""

    meta: dict
    cracks: list[CrackRecord]
    floes: list[FloeRecord]
    fractured_elements: list[dict]
    fsd: dict
    conservation: ConservationLedger
    provenance: dict = field(default_factory=dict)
    crack_cells: list[dict] = field(default_factory=list)

    # -- serialisation -----------------------------------------------------
    def to_payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "meta": self.meta,
            "cracks": [
                {
                    "crack_id": c.crack_id,
                    "polyline": c.polyline,
                    "length_m": _round(c.length),
                    "seed_xy": [_round(c.seed_xy[0]), _round(c.seed_xy[1])],
                    "tip_end": c.tip_end,
                    "n_nodes": c.n_nodes,
                }
                for c in self.cracks
            ],
            "fractured_elements": [
                {
                    "cell": e["cell"],
                    "polygon": e["polygon"],
                    "sigma_pa": _round(e["sigma_pa"]),
                    "sigma_over_strength": _round(e["sigma_over_strength"], 6),
                }
                for e in self.fractured_elements
            ],
            "crack_cells": [
                {
                    "cell": e["cell"],
                    "polygon": e["polygon"],
                    "sigma_pa": _round(e["sigma_pa"]),
                }
                for e in self.crack_cells
            ],
            "floes": [
                {
                    "floe_id": f.floe_id,
                    "area_m2": _round(f.area),
                    "mass_kg": _round(f.mass),
                    "centroid_xy": [_round(f.centroid[0]), _round(f.centroid[1])],
                    "equivalent_diameter_m": _round(f.equivalent_diameter),
                    "perimeter_m": _round(f.perimeter),
                    "n_cells": f.n_cells,
                    "polygons": f.polygons,
                    "max_sigma_pa": _round(f.max_sigma),
                    "mean_sigma_pa": _round(f.mean_sigma),
                }
                for f in self.floes
            ],
            "fsd": self.fsd,
            "conservation": self.conservation.as_dict(),
            "provenance": dict(self.provenance),
        }

    def to_json(self, indent: int | None = 2) -> str:
        payload = self.to_payload()
        payload["provenance"] = {**payload["provenance"],
                                 "payload_sha256": payload_sha256(payload)}
        return json.dumps(payload, indent=indent, sort_keys=True)

    def write(self, path) -> str:
        text = self.to_json()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
        return text


def payload_sha256(payload: dict) -> str:
    """SHA-256 over the canonical payload (provenance for the audit chain)."""
    body = {k: v for k, v in payload.items() if k != "provenance"}
    body["provenance"] = {k: v for k, v in payload.get("provenance", {}).items()
                          if k != "payload_sha256"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def _cell_polygon(mesh, cell: int) -> list[list[float]]:
    xy = mesh.node_xy()[mesh.cell_ids()[cell]]
    return [[float(x), float(y)] for x, y in xy]


def _polygon_segments(ring) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    pts = np.asarray(ring, dtype=float)
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 0.0:
            continue
        key = (round(float(a[0]), 9), round(float(a[1]), 9),
               round(float(b[0]), 9), round(float(b[1]), 9))
        out.append((key, seg))
    return out


def check_conservation(
    floes: list[FloeRecord],
    plate_area: float,
    plate_perimeter: float,
    density: float,
    thickness: float,
) -> ConservationLedger:
    """Rebuild the ledger from the floe polygons alone (independent of the FEM).

    Segment bookkeeping: every boundary segment of a floe appears once if it lies
    on the plate rim and twice if it is a fracture face shared with a neighbouring
    floe.  That gives the rim length (which must equal the plate perimeter) and
    the fracture-face length (which must be positive and no longer than the total
    traced crack length).
    """
    if not floes:
        raise ValueError("no floes: the fracture state is empty")
    areas = np.array([f.polygon_area() for f in floes])
    perims = np.array([polygon_perimeter(np.asarray(f.polygons[0])) for f in floes])
    holes = np.array([sum(polygon_perimeter(np.asarray(h)) for h in f.polygons[1:])
                      for f in floes])
    total_perimeter = float(np.sum(perims + holes))

    usage: dict[tuple, int] = {}
    length: dict[tuple, float] = {}
    for floe in floes:
        for ring in floe.polygons:
            for key, seg in _polygon_segments(ring):
                # Undirected key: an internal face is traversed in opposite
                # directions by the two floes that share it.
                canon = (min(key[0:2], key[2:4]), max(key[0:2], key[2:4]))
                usage[canon] = usage.get(canon, 0) + 1
                length[canon] = seg
    rim = sum(length[k] for k, c in usage.items() if c == 1)
    # Each segment that is used twice IS one fracture face: its *unique* length is
    # the face length (the factor 1/2 belongs to the naive (sum(perimeter) -
    # P_plate)/2 bookkeeping, where the internal faces are counted twice; here the
    # unique lengths are summed, so halving would under-report the face length by
    # 2x -- measured 21.75 m instead of 43.5 m before this was fixed).
    internal = sum(length[k] for k, c in usage.items() if c == 2)
    over = sum(1 for c in usage.values() if c > 2)
    return ConservationLedger(
        plate_area=float(plate_area),
        floes_area_sum=float(areas.sum()),
        area_residual=float(areas.sum() - plate_area),
        plate_mass=float(plate_area * thickness * density),
        floes_mass_sum=float(areas.sum() * thickness * density),
        mass_residual=float((areas.sum() - plate_area) * thickness * density),
        plate_perimeter=float(plate_perimeter),
        floes_perimeter_sum=total_perimeter,
        rim_length_from_polygons=float(rim),
        rim_residual=float(rim - plate_perimeter),
        internal_face_length=float(internal),
        n_floe=len(floes),
        n_segments_used_three_times=int(over),
    )


def _cells_touched_by_fracture(mesh, cut_edges, solution) -> list[dict]:
    """FEM cells whose boundary carries a fracture face (the "断裂单元")."""
    touched: set[int] = set()
    for (a, b) in cut_edges:
        i_a, j_a = mesh.node_index(a)
        i_b, j_b = mesh.node_index(b)
        if i_a == i_b:                     # horizontal edge at y = j_a * hy
            row = j_a - 1
            for col in (i_a - 1, i_a):
                if 0 <= row < mesh.ny and 0 <= col < mesh.nx:
                    touched.add(row * mesh.nx + col)
        else:                              # vertical edge at x = i_a * hx
            col = i_a - 1
            for row in (j_a - 1, j_a):
                if 0 <= row < mesh.ny and 0 <= col < mesh.nx:
                    touched.add(row * mesh.nx + col)
    out = [
        {"cell": [int(c) % mesh.nx, int(c) // mesh.nx],
         "polygon": _cell_polygon(mesh, int(c)),
         "sigma_pa": float(solution.sigma_max[c])}
        for c in sorted(touched)
    ]
    return out


def build_failure_faces(
    network: CrackNetwork,
    solution,
    *,
    case_meta: dict,
    fsd: dict | None = None,
    fractured_sigma: float | None = None,
) -> FailureFaces:
    """Turn a :class:`CrackNetwork` + FEM solution into a handoff artefact."""
    mesh = solution.mesh
    material = solution.material
    sigma_f = material.sigma_flexure if fractured_sigma is None else fractured_sigma

    floes = [
        FloeRecord(
            floe_id=f.floe_id,
            area=f.area,
            mass=f.mass,
            centroid=(float(f.centroid[0]), float(f.centroid[1])),
            equivalent_diameter=f.equivalent_diameter,
            perimeter=f.perimeter,
            n_cells=len(f.cells),
            polygons=f.all_polygons(),
            max_sigma=float(f.max_sigma),
            mean_sigma=float(f.mean_sigma),
        )
        for f in network.floes
    ]
    cracks = [
        CrackRecord(
            crack_id=c.crack_id,
            polyline=c.as_polyline(),
            length=c.length,
            seed_xy=(float(c.points[0][0]) if len(c.points) else 0.0,
                     float(c.points[0][1]) if len(c.points) else 0.0),
            tip_end=c.tip_end,
            n_nodes=len(c.nodes),
        )
        for c in network.cracks
    ]
    over = np.nonzero(solution.sigma_max >= sigma_f)[0]
    fractured = [
        {
            "cell": [int(c) % mesh.nx, int(c) // mesh.nx],
            "polygon": _cell_polygon(mesh, int(c)),
            "sigma_pa": float(solution.sigma_max[c]),
            "sigma_over_strength": float(solution.sigma_max[c] / sigma_f),
        }
        for c in over
    ]

    if fsd is None:
        diameters = network.diameters()
        fsd = {
            "n_floe": int(len(diameters)),
            "diameters_m": [_round(d) for d in sorted(diameters, reverse=True)],
            "mean_diameter_m": _round(np.mean(diameters)),
            "median_diameter_m": _round(np.median(diameters)),
        }

    ledger = check_conservation(
        floes,
        plate_area=mesh.plate_area,
        plate_perimeter=2.0 * (mesh.length_x + mesh.length_y),
        density=material.density,
        thickness=material.thickness,
    )

    crack_cells = _cells_touched_by_fracture(mesh, network.cut_edges, solution)

    meta = {
        "case": case_meta.get("case", "level_ice_contact_load"),
        "plate": {
            "length_x_m": mesh.length_x,
            "length_y_m": mesh.length_y,
            "thickness_m": material.thickness,
            "area_m2": mesh.plate_area,
        },
        "material": {
            "young_modulus_pa": material.young_modulus,
            "poisson_ratio": material.poisson_ratio,
            "density_kg_m3": material.density,
            "sigma_flexure_pa": material.sigma_flexure,
            "flexural_rigidity_nm": material.flexural_rigidity,
        },
        "mesh": {"nx": mesh.nx, "ny": mesh.ny,
                 "n_cells": mesh.n_cells, "cell_area_m2": mesh.cell_area},
        "load": case_meta.get("load", {}),
        "stress_state": {
            "sigma_max_pa": float(solution.max_sigma),
            "sigma_flexure_pa": sigma_f,
            "overstress_ratio": float(solution.max_sigma / sigma_f),
            "fracture_declared": bool(solution.max_sigma >= sigma_f),
            "n_fractured_elements": len(fractured),
            "n_crack_cells": len(crack_cells),
        },
        "crack_model": {
            "kind": "frozen-field mode-I lattice tracing",
            "weibull_modulus": network.config.weibull_modulus,
            "seed": network.config.seed,
            "min_seed_separation_m": network.config.min_seed_separation,
            "arrest_fraction": network.config.arrest_fraction,
            "n_seeds": len(network.seed_nodes),
            "n_cracks": len(network.cracks),
            "total_crack_length_m": _round(network.total_crack_length),
        },
        "fem": {
            "element": "ACM 12-dof Kirchhoff rectangle",
            "boundary": solution.extra.get("boundary", "free"),
            "foundation_modulus_n_m3": solution.foundation_modulus,
            "max_deflection_m": float(solution.max_deflection),
        },
    }
    return FailureFaces(
        meta=meta,
        cracks=cracks,
        floes=floes,
        fractured_elements=fractured,
        crack_cells=crack_cells,
        fsd=fsd,
        conservation=ledger,
        provenance={
            "code": _CODE_VERSION,
            "generator": "tensordem.fem_dem_fracture",
            "notes": "frozen-field crack tracing; area/mass ledger exact by mesh-edge partition",
        },
    )


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
def read_failure_faces(path_or_payload) -> FailureFaces:
    """Read and *validate* a ``failure_faces.json`` file.

    Recomputes the whole ledger from the stored polygons and raises if the file
    is internally inconsistent (area, mass or boundary ledger), so a corrupted
    or hand-edited artefact cannot silently feed the DEM.
    """
    if isinstance(path_or_payload, dict):
        payload = path_or_payload
    else:
        with open(path_or_payload, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unexpected schema {schema!r}, expected {SCHEMA!r}")

    meta = payload["meta"]
    floes = [
        FloeRecord(
            floe_id=f["floe_id"],
            area=f["area_m2"],
            mass=f["mass_kg"],
            centroid=(f["centroid_xy"][0], f["centroid_xy"][1]),
            equivalent_diameter=f["equivalent_diameter_m"],
            perimeter=f["perimeter_m"],
            n_cells=f["n_cells"],
            polygons=f["polygons"],
            max_sigma=f["max_sigma_pa"],
            mean_sigma=f["mean_sigma_pa"],
        )
        for f in payload["floes"]
    ]
    cracks = [
        CrackRecord(
            crack_id=c["crack_id"],
            polyline=c["polyline"],
            length=c["length_m"],
            seed_xy=(c["seed_xy"][0], c["seed_xy"][1]),
            tip_end=c["tip_end"],
            n_nodes=c["n_nodes"],
        )
        for c in payload["cracks"]
    ]
    material = meta["material"]
    plate = meta["plate"]
    ledger = check_conservation(
        floes,
        plate_area=plate["area_m2"],
        plate_perimeter=2.0 * (plate["length_x_m"] + plate["length_y_m"]),
        density=material["density_kg_m3"],
        thickness=plate["thickness_m"],
    )
    stored = payload["conservation"]
    tol_area = 1e-9 * max(1.0, plate["area_m2"])
    tol_mass = 1e-6 * max(1.0, ledger.plate_mass)
    if abs(ledger.area_residual) > tol_area:
        raise ValueError(
            f"artifact violates area conservation: residual {ledger.area_residual:g} m2"
        )
    if abs(ledger.mass_residual) > tol_mass:
        raise ValueError(
            f"artifact violates mass conservation: residual {ledger.mass_residual:g} kg"
        )
    if abs(ledger.rim_residual) > 1e-9 * max(1.0, plate["area_m2"]):
        raise ValueError(
            "artifact violates the boundary ledger: the floe polygons' outer rim "
            f"({ledger.rim_length_from_polygons:g} m) is not the plate perimeter "
            f"({ledger.plate_perimeter:g} m)"
        )
    if ledger.n_segments_used_three_times:
        raise ValueError("floe polygons overlap: a boundary segment is used 3+ times")
    if int(stored["n_floe"]) != ledger.n_floe:
        raise ValueError("stored floe count disagrees with the artefact")
    for floe in floes:
        poly_area = floe.polygon_area()
        if abs(poly_area - floe.area) > 1e-9 * max(1.0, floe.area):
            raise ValueError(
                f"floe {floe.floe_id}: declared area {floe.area:g} m2 disagrees with its "
                f"polygons ({poly_area:g} m2)"
            )

    return FailureFaces(
        meta=meta,
        cracks=cracks,
        floes=floes,
        fractured_elements=payload["fractured_elements"],
        crack_cells=payload.get("crack_cells", []),
        fsd=payload["fsd"],
        conservation=ledger,
        provenance=payload.get("provenance", {}),
    )


# --------------------------------------------------------------------------
# DEM initial conditions
# --------------------------------------------------------------------------
@dataclass
class DemInitialConditions:
    """Floe bodies ready to be inserted into a DEM solver."""

    positions: np.ndarray        # (N, 3) area centroids at the plate mid-plane
    diameters: np.ndarray        # (N,) equivalent diameters [m]
    masses: np.ndarray           # (N,) [kg]
    velocities: np.ndarray       # (N, 3) [m/s]
    thickness: float
    polygons: list[list[list[list[float]]]]
    ledger: ConservationLedger

    def __len__(self) -> int:
        return len(self.diameters)

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses))

    def as_payload(self) -> dict:
        return {
            "schema": "tensordem.dem_initial_conditions/1",
            "n_bodies": len(self),
            "thickness_m": _round(self.thickness),
            "total_mass_kg": _round(self.total_mass),
            "bodies": [
                {
                    "id": i,
                    "position_m": [_round(v) for v in self.positions[i]],
                    "diameter_m": _round(self.diameters[i]),
                    "mass_kg": _round(self.masses[i]),
                    "velocity_m_s": [_round(v) for v in self.velocities[i]],
                    "polygons": self.polygons[i],
                }
                for i in range(len(self))
            ],
            "ledger": self.ledger.as_dict(),
        }


def dem_initial_conditions(
    faces: FailureFaces,
    *,
    velocity_field=None,
) -> DemInitialConditions:
    """Convert the fracture artefact into DEM body initial conditions.

    ``velocity_field`` is an optional callable ``(x, y) -> (vx, vy)``; a
    quasi-static FEM run has no velocity field, so the default is zero initial
    velocity (stated in the payload).  Passing a field lets a dynamic FEM/DEM
    coupling inject the FEM nodal velocity at each floe centroid.
    """
    n = len(faces.floes)
    positions = np.zeros((n, 3))
    diameters = np.zeros(n)
    masses = np.zeros(n)
    velocities = np.zeros((n, 3))
    polygons = []
    for i, floe in enumerate(faces.floes):
        cx, cy = floe.centroid
        positions[i] = (cx, cy, 0.0)
        diameters[i] = floe.equivalent_diameter
        masses[i] = floe.mass
        if velocity_field is not None:
            vx, vy = velocity_field(cx, cy)
            velocities[i] = (vx, vy, 0.0)
        polygons.append(floe.polygons)
    return DemInitialConditions(
        positions=positions,
        diameters=diameters,
        masses=masses,
        velocities=velocities,
        thickness=float(faces.meta["plate"]["thickness_m"]),
        polygons=polygons,
        ledger=facets_ledger(faces),
    )


# Kept separate so :func:`dem_initial_conditions` stays readable.
def facets_ledger(faces: FailureFaces) -> ConservationLedger:
    return faces.conservation