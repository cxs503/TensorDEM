"""DEM export skeleton: continuum fracture fragments -> DEM particle input.

Provides:
- A JSON schema for serialising fracture fragments (centroid, mass, velocity,
  radius) into a format that can be loaded as DEM initial conditions.
- A serialiser that takes lists/arrays of fragment properties and writes JSON.
- A loader that reads the JSON and returns tensors suitable for DEM initialisation.

This is a data-contract layer only — no physics coupling is performed here.
It is a P2 audit item (SI-2 prerequisite).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import torch


# ---------------------------------------------------------------------------
# JSON Schema (as a Python dict; can be serialised for documentation)
# ---------------------------------------------------------------------------

DEM_EXPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "tensordem/dem-export/1",
    "title": "TensorDEM Fragment Export",
    "description": "Continuum fracture fragments serialised as DEM particle input.",
    "type": "object",
    "required": ["schema", "fragments"],
    "properties": {
        "schema": {"const": "tensordem.dem-export/1"},
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "centroid_m", "mass_kg", "radius_m", "velocity_m_s"],
                "properties": {
                    "id": {"type": "integer", "minimum": 0},
                    "centroid_m": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                    "mass_kg": {"type": "number", "exclusiveMinimum": 0},
                    "radius_m": {"type": "number", "exclusiveMinimum": 0},
                    "velocity_m_s": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """A single fracture fragment."""
    id: int
    centroid_m: list[float]       # [x, y] or [x, y, z]
    mass_kg: float
    radius_m: float
    velocity_m_s: list[float]     # [vx, vy] or [vx, vy, vz]

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        dim = len(self.centroid_m)
        if dim not in (2, 3):
            raise ValueError("centroid_m must have 2 or 3 components")
        if len(self.velocity_m_s) != dim:
            raise ValueError("velocity_m_s must match centroid dimension")


@dataclass
class DemExport:
    """Full export record."""
    schema: str = "tensordem.dem-export/1"
    fragments: list[Fragment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fragments": [asdict(f) for f in self.fragments],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DemExport":
        if data.get("schema") != "tensordem.dem-export/1":
            raise ValueError(f"Unknown schema: {data.get('schema')}")
        fragments = [
            Fragment(**f) for f in data.get("fragments", [])
        ]
        return cls(schema=data["schema"], fragments=fragments)

    @classmethod
    def from_json(cls, text: str) -> "DemExport":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Serialiser / Loader utilities
# ---------------------------------------------------------------------------

def serialize_fragments(
    centroids: list[list[float]],
    masses: list[float],
    radii: list[float],
    velocities: list[list[float]],
) -> DemExport:
    """Build a DemExport from parallel lists.

    Args:
        centroids:   list of [x, y] or [x, y, z].
        masses:      list of masses (kg).
        radii:       list of equivalent radii (m).
        velocities:  list of [vx, vy] or [vx, vy, vz].

    Returns:
        DemExport instance.
    """
    if not (len(centroids) == len(masses) == len(radii) == len(velocities)):
        raise ValueError("All input lists must have the same length")

    fragments = [
        Fragment(
            id=i,
            centroid_m=c,
            mass_kg=m,
            radius_m=r,
            velocity_m_s=v,
        )
        for i, (c, m, r, v) in enumerate(zip(centroids, masses, radii, velocities))
    ]
    return DemExport(fragments=fragments)


def load_to_tensors(
    export: DemExport,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a DemExport into tensors ready for DEM initialisation.

    Returns:
        dict with keys:
            'positions':   (N, D) float64
            'velocities':  (N, D) float64
            'masses':      (N,)   float64
            'radii':       (N,)   float64
    """
    fragments = export.fragments
    N = len(fragments)
    if N == 0:
        return {
            "positions": torch.empty((0, 2), dtype=torch.float64, device=device),
            "velocities": torch.empty((0, 2), dtype=torch.float64, device=device),
            "masses": torch.empty((0,), dtype=torch.float64, device=device),
            "radii": torch.empty((0,), dtype=torch.float64, device=device),
        }

    dim = len(fragments[0].centroid_m)
    positions = torch.tensor(
        [f.centroid_m for f in fragments], dtype=torch.float64, device=device
    )
    velocities = torch.tensor(
        [f.velocity_m_s for f in fragments], dtype=torch.float64, device=device
    )
    masses = torch.tensor(
        [f.mass_kg for f in fragments], dtype=torch.float64, device=device
    )
    radii = torch.tensor(
        [f.radius_m for f in fragments], dtype=torch.float64, device=device
    )
    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "radii": radii,
    }


def save_json(export: DemExport, path: str | Path) -> None:
    """Write DemExport to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(export.to_json())


def load_json(path: str | Path) -> DemExport:
    """Read DemExport from a JSON file."""
    with open(path, "r") as f:
        return DemExport.from_json(f.read())
