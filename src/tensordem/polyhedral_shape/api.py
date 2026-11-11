"""HTTP endpoints for the polyhedral shape / packing validation face.

Two routes, both pure functions ``payload -> dict`` so that the FastAPI layer is
a thin wrapper and structural errors surface as HTTP 422:

``POST /api/v1/dem/polyhedral/shape_descriptor``
    Convex particle -> integral shape descriptors (sphericity, roundness,
    aspect ratio, inradius/circumradius, volume, area, vertex/face/edge counts).

``POST /api/v1/dem/polyhedral/packing_fraction``
    Particle family + protocol -> packing fraction ``phi`` with the protocol and
    the shape descriptors recorded alongside it.

The route layer maps ``ValueError`` (physical/geometric rejection: negative
volume, degenerate vertex cloud, unknown shape name, ``n_particles`` below 8,
...) to HTTP 422; pydantic handles the structural validation (missing fields,
wrong types, out-of-range integers) and also returns 422.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .packing import pack_deposition, pack_spheres_compaction
from .shapes import (
    Ellipsoid,
    PolyhedralShape,
    describe,
    describe_ellipsoid,
    named_shape,
    volume_equivalent_radius,
)

__all__ = [
    "ShapeRequest",
    "DescriptorRequest",
    "PackingRequest",
    "handle_shape_descriptor",
    "handle_packing_fraction",
    "build_app",
]


class ShapeRequest(BaseModel):
    """A convex particle: either a named canonical shape or an explicit cloud."""

    name: Optional[str] = None
    vertices: Optional[List[List[float]]] = None
    orientation: Optional[List[List[float]]] = None
    semi_axes: Optional[List[float]] = None
    target_volume: Optional[float] = Field(default=None, gt=0.0)
    size: float = Field(default=1.0, gt=0.0, le=1.0e6)

    @field_validator("vertices")
    @classmethod
    def _vertices_shape(cls, value):
        if value is None:
            return value
        if len(value) < 4:
            raise ValueError("need at least 4 vertices")
        for row in value:
            if len(row) != 3:
                raise ValueError("each vertex needs exactly 3 coordinates")
            if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in row):
                raise ValueError("vertex coordinates must be finite numbers")
        return value

    @field_validator("semi_axes")
    @classmethod
    def _semi_axes(cls, value):
        if value is None:
            return value
        if len(value) != 3:
            raise ValueError("semi_axes needs exactly 3 values")
        if any(float(a) <= 0.0 or not math.isfinite(float(a)) for a in value):
            raise ValueError("semi-axes must be finite and positive")
        return value

    @model_validator(mode="after")
    def _one_description(self):
        if self.name is None and self.vertices is None and self.semi_axes is None:
            raise ValueError("provide one of: name, vertices, semi_axes")
        if self.vertices is not None and self.semi_axes is not None:
            raise ValueError("provide either vertices or semi_axes, not both")
        return self


class DescriptorRequest(BaseModel):
    shape: ShapeRequest


class PackingRequest(BaseModel):
    shape: ShapeRequest
    protocol: Literal["deposition", "compaction"] = "deposition"
    n_particles: int = Field(default=120, ge=8, le=2000)
    trials: int = Field(default=1, ge=1, le=8)
    seed: int = Field(default=0, ge=0, le=2 ** 31 - 1)


def _build_particle(spec: ShapeRequest):
    """ShapeRequest -> (support body, ShapeDescriptors, volume)."""
    name = (spec.name or "").strip().lower()
    if spec.semi_axes is None and spec.vertices is None and name in {"sphere", "ball"}:
        radius = 0.5 * float(spec.size)
        sphere = Ellipsoid([radius] * 3, orientation=spec.orientation)
        return sphere, describe_ellipsoid(sphere)
    if spec.semi_axes is not None:
        ell = Ellipsoid(spec.semi_axes, orientation=spec.orientation)
        desc = describe_ellipsoid(ell)
        return ell, desc
    if spec.vertices is not None:
        verts = np.asarray(spec.vertices, dtype=float)
        shape = PolyhedralShape(verts, orientation=spec.orientation, name=spec.name or "custom")
    else:
        shape = named_shape(spec.name, spec.size)
        if spec.orientation is not None:
            shape = PolyhedralShape(shape.local_vertices, orientation=spec.orientation,
                                    name=shape.name)
    if spec.target_volume is not None:
        scale = (float(spec.target_volume) / shape.volume) ** (1.0 / 3.0)
        shape = PolyhedralShape(shape.local_vertices * scale, name=shape.name)
    return shape, describe(shape)


def handle_shape_descriptor(payload: dict) -> dict:
    """``POST /api/v1/dem/polyhedral/shape_descriptor`` body."""
    req = DescriptorRequest.model_validate(payload)
    shape, desc = _build_particle(req.shape)
    out = desc.as_dict()
    out["equivalent_sphere_radius"] = volume_equivalent_radius(desc.volume)
    return {"status": "ok", "descriptor": out}


def handle_packing_fraction(payload: dict) -> dict:
    """``POST /api/v1/dem/polyhedral/packing_fraction`` body."""
    req = PackingRequest.model_validate(payload)
    shape, desc = _build_particle(req.shape)
    seed = int(req.seed)
    if req.protocol == "compaction":
        if not (spec_is_sphere(req.shape)):
            raise ValueError("compaction protocol supports spherical particles only")
        res = pack_spheres_compaction(n=req.n_particles, rng=np.random.default_rng(seed))
    else:
        base_volume = desc.volume

        def factory(rng: np.random.Generator):
            rot = _random_rotation(rng)
            if isinstance(shape, Ellipsoid):
                return Ellipsoid(shape.semi_axes, orientation=rot)
            return PolyhedralShape(shape.local_vertices, orientation=rot, name=shape.name)

        res = pack_deposition(factory, n_particles=req.n_particles, trials=req.trials,
                              rng=np.random.default_rng(seed))
    out = res.as_dict()
    out["shape"] = desc.as_dict()
    out["shape"]["volume_player"] = base_volume
    return {"status": "ok", "packing": out}


def spec_is_sphere(spec: ShapeRequest) -> bool:
    """True when the request describes an exactly spherical particle."""
    if spec.semi_axes is not None:
        a = np.asarray(spec.semi_axes, dtype=float)
        return bool(np.allclose(a, a[0], rtol=0.0, atol=0.0))
    name = (spec.name or "").strip().lower()
    return name in {"sphere", "ball"}


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def build_app():
    """FastAPI application exposing the two endpoints (422 on invalid input)."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="TensorDEM polyhedral shape / packing", version="1.0")

    @app.post("/api/v1/dem/polyhedral/shape_descriptor")
    def shape_descriptor(payload: DescriptorRequest):  # pragma: no cover - thin
        try:
            return handle_shape_descriptor(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/dem/polyhedral/packing_fraction")
    def packing_fraction(payload: PackingRequest):  # pragma: no cover - thin
        try:
            return handle_packing_fraction(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app