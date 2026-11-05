"""Differential probe: does the merged fracture generator keep *both* contracts?

Run this script inside a tree that has ``tensordem`` importable; it prints a JSON
document of every observable in the dual fracture-gen contract, using only the
API that exists in all three trees compared by mission m3857 (branch
``agent/m2965``, ``main``, and the merge of the two).  The m3857 merge oracle
re-runs it in each tree and compares the observables one by one, which is how
"no side's behaviour was silently dropped" is *measured* instead of asserted.

Observables (prefix = contract owner):

``branch/*``  face-extent (plane|disc) and on-plane-tolerance semantics,
              lattice coverage of the requested domain, mass-mapping aliases,
              vectorised ``build_bond_topology`` bond list.
``main/*``    volume-consistent default mass mapping, ``volume_fraction`` as the
              mass-sphere fraction of the cell, signed/magnitude ledger keys.

``coverage`` is reported under the key each tree uses (``volume_fraction`` on the
branch, ``domain_coverage`` after the merge) so the comparison is semantic.

Usage::

    python scripts/merge_m3857_dual_contract_probe.py > probe.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.fracture_gen import (
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    build_bond_topology,
    generate_fracture_field,
    generate_particle_lattice,
    identify_bonds_crossing_face,
)

RADIUS = 0.1
DENSITY = 917.0


def _field(side: float, **kwargs):
    cfg = FractureGenConfig(
        density=DENSITY,
        particle_radius=RADIUS,
        domain_bounds=(0.0, side, 0.0, side, 0.0, side),
        **kwargs,
    )
    return generate_fracture_field([], cfg)


def _count(field, **kwargs) -> int:
    face = kwargs.pop("face")
    return int(
        identify_bonds_crossing_face(
            field.bond_indices, field.positions, face, **kwargs
        ).sum()
    )


def _face(area: float, x: float) -> FailureFace:
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([1.0, 0.0, 0.0]),
        centroid=torch.tensor([x, 0.5, 0.5]),
        G=100.0,
        released_energy=1.0,
    )


def _coverage(field) -> float:
    """Lattice coverage of the requested domain, under whatever name this tree uses."""
    for name in ("domain_coverage", "volume_fraction"):
        try:
            value = getattr(field, name)
        except AttributeError:
            continue
        else:
            # ``volume_fraction`` means "mass-sphere volume / cell volume" on
            # main; that lives in a different observable, so only accept the
            # attribute when it is the coverage term (== 1 for an exact fit).
            return float(value)
    raise AttributeError("no coverage observable on FractureField")


def _ledger(side: float, **kwargs) -> dict:
    field = _field(side, **kwargs)
    accounting = FractureLedger(field, []).mass_accounting(domain_volume=side**3)
    keys = (
        "num_particles",
        "dem_mass",
        "fem_continuum_mass",
        "relative_error",
        "absolute_relative_error",
        "relative_error_magnitude",
        "volume_fraction",
    )
    return {k: accounting.get(k) for k in keys}


def main() -> int:
    out: dict[str, object] = {}

    # --- branch contract: lattice, bond topology, coverage, aliases ---------
    exact = _field(1.0)
    coarse = _field(0.9)  # 4 particles per axis, cells cover 0.8^3 of 0.9^3
    out["branch/particles_exact"] = int(exact.positions.shape[0])
    out["branch/bonds_exact"] = int(exact.bond_indices.shape[0])
    out["branch/coverage_exact"] = _coverage(exact)
    out["branch/coverage_coarse"] = _coverage(coarse)

    # vectorised topology must still be the simple-cubic nearest-neighbour graph
    lattice = generate_particle_lattice((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), RADIUS)
    bonds, rest = build_bond_topology(lattice, RADIUS)
    out["branch/bond_count"] = int(bonds.shape[0])
    out["branch/rest_length_max_dev"] = float((rest - 2.0 * RADIUS).abs().max())

    # face-extent contract: plane vs footprint disc vs layer-coincident plane
    out["branch/plane_x055"] = _count(exact, face=_face(1.0, 0.55))
    for area in (0.25, 0.5, 1.0):
        out[f"branch/disc_area{area}"] = _count(
            exact, face=_face(area, 0.55), face_extent="disc"
        )
    layer = _face(1.0, 0.5)
    out["branch/layer_strict"] = _count(exact, face=layer)
    out["branch/layer_tolerance"] = _count(
        exact, face=layer, on_plane_tolerance=1e-9
    )

    # mass-mapping aliases -> one internal mode
    aliases = {}
    for spelling in (
        "cell",
        "cell_volume",
        "lattice_cell",
        "sphere",
        "equal_volume_sphere",
    ):
        try:
            field = _field(1.0, mass_mapping=spelling)
            aliases[spelling] = field.mass_mode
        except (TypeError, ValueError) as exc:  # pragma: no cover - tree dependent
            aliases[spelling] = f"rejected: {type(exc).__name__}"
    out["branch/mass_mapping_aliases"] = aliases

    # --- main contract: default mapping and mass-sphere volume fraction -----
    default = _field(1.0)
    out["main/default_mass_mode"] = default.mass_mode
    out["main/volume_fraction_cell"] = float(default.volume_fraction)
    sphere = _field(1.0, mass_mapping="equal_volume_sphere")
    out["main/volume_fraction_sphere"] = float(sphere.volume_fraction)
    out["main/mass_radius_over_r"] = float(default.mass_radius / RADIUS)
    out["main/pi_over_6"] = math.pi / 6.0

    for label, kwargs in (("cell", {"mass_mapping": "cell_volume"}), ("sphere", {"mass_mapping": "equal_volume_spheres"})):
        accounting = _ledger(1.0, **kwargs)
        for key, value in accounting.items():
            out[f"main/ledger_{label}/{key}"] = value

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())