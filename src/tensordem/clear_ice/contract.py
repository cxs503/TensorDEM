"""``clear-ice-contract/1.0`` -- interface between level-ice breakup and the floe field.

Ownership
---------
The level-ice breakup (``tensordem.level_ice``) owns *how* a sheet breaks; the
clear-ice field (this package) owns *how the resulting floes move and load the
hull*.  Neither may import the other's internals, so the hand-over is an
explicit, versioned, serialisable contract:

``FloePopulation``  (breakup -> field)
    Fragment centroids, equivalent radii, thicknesses and the parent-sheet area
    ledger.  Area conservation is checked here, not assumed: a population that
    does not conserve the sheet area is rejected, because an area leak would
    silently change the ice concentration and hence every statistic downstream.

``ClearIceRequest``  (driver -> field)
    The population plus the channel/drift/material request.

``ClearIceResult``  (field -> consumer)
    The five report blocks the campaign requires: ``floe_kinematics``,
    ``force_chain_stats``, ``porosity``, ``convergence_report`` (plus
    ``uncertainty``), with a provenance hash over the population so a result can
    always be traced back to the fragment set that produced it.

All payloads are plain JSON-able dicts at the boundary; the dataclasses are
frozen views over those dicts, so adding a field is a minor version bump and
removing one is a major version bump.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CONTRACT_VERSION",
    "ClearIceRequest",
    "ClearIceResult",
    "FloePopulation",
    "population_from_breakup",
    "validate_population",
    "population_hash",
]

CONTRACT_VERSION = "clear-ice-contract/1.0"


def population_hash(x: np.ndarray, radius: np.ndarray, thickness: np.ndarray) -> str:
    """Deterministic SHA-256 over the population arrays (float64, C order)."""
    h = hashlib.sha256()
    for name, arr in (("x", x), ("r", radius), ("t", thickness)):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FloePopulation:
    """Fragments handed over by the level-ice breakup module."""

    x: np.ndarray                 # (N, 2) centroids in the channel frame, m
    radius: np.ndarray            # (N,) equivalent radius r = sqrt(A/pi), m
    thickness: np.ndarray         # (N,) thickness, m
    source: str = "level_ice"
    parent_area: float | None = None      # m^2 of the parent sheet area broken
    contract_version: str = CONTRACT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        r = np.asarray(self.radius, dtype=float)
        t = np.asarray(self.thickness, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape (N, 2)")
        if r.shape != (len(x),) or t.shape != (len(x),):
            raise ValueError("radius and thickness must have shape (N,)")
        if np.any(r <= 0.0) or np.any(t <= 0.0):
            raise ValueError("radius and thickness must be positive")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "radius", r)
        object.__setattr__(self, "thickness", t)

    @property
    def area(self) -> float:
        return float(np.sum(np.pi * self.radius ** 2))

    @property
    def hash(self) -> str:
        return population_hash(self.x, self.radius, self.thickness)

    def as_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "source": self.source,
            "n_floes": int(len(self.radius)),
            "x": self.x.tolist(),
            "radius": self.radius.tolist(),
            "thickness": self.thickness.tolist(),
            "area_m2": self.area,
            "parent_area_m2": self.parent_area,
            "population_hash": self.hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FloePopulation":
        version = payload.get("contract_version", CONTRACT_VERSION)
        if not str(version).startswith("clear-ice-contract/1."):
            raise ValueError(f"unsupported contract version {version!r}")
        return cls(
            x=np.asarray(payload["x"], dtype=float),
            radius=np.asarray(payload["radius"], dtype=float),
            thickness=np.asarray(payload["thickness"], dtype=float),
            source=payload.get("source", "unknown"),
            parent_area=payload.get("parent_area_m2"),
            contract_version=version,
            metadata=payload.get("metadata", {}),
        )


def validate_population(pop: FloePopulation, *, domain: tuple[float, float] | None = None,
                        area_tolerance: float = 1.0e-3,
                        overlap_tolerance: float = 5.0e-2) -> dict:
    """Validate a hand-over and return the audit record.

    Checks (all machine-checkable):

    1. **area ledger** -- ``sum(pi r^2)`` against the declared parent area,
       relative error ``<= area_tolerance`` (default 0.1 %);
    2. **containment** -- centroids inside the delivery domain (if given);
    3. **non-overlap** -- pairwise overlap ``delta = r_i + r_j - d`` at most
       ``overlap_tolerance * min(r_i, r_j)``; a breakup module that emits
       interpenetrating fragments is rejected rather than fixed silently;
    4. **concentration** -- the areal concentration implied by the population in
       the domain, which is the quantity the field statistics are conditioned
       on.
    """
    record: dict[str, Any] = {
        "contract_version": pop.contract_version,
        "population_hash": pop.hash,
        "n_floes": int(len(pop.radius)),
        "area_m2": pop.area,
        "checks": {},
    }
    ok = True

    if pop.parent_area is not None:
        rel = abs(pop.area - pop.parent_area) / max(pop.parent_area, 1e-30)
        passed = rel <= area_tolerance
        record["checks"]["area_ledger"] = {
            "parent_area_m2": pop.parent_area, "population_area_m2": pop.area,
            "relative_error": rel, "tolerance": area_tolerance, "pass": bool(passed)}
        ok &= passed

    if domain is not None:
        Lx, Ly = float(domain[0]), float(domain[1])
        inside = bool(np.all((pop.x[:, 0] >= 0.0) & (pop.x[:, 0] <= Lx)
                             & (pop.x[:, 1] >= 0.0) & (pop.x[:, 1] <= Ly)))
        record["checks"]["containment"] = {"pass": inside,
                                           "domain_m": [Lx, Ly]}
        ok &= inside
        record["concentration"] = pop.area / (Lx * Ly)

    from scipy.spatial import cKDTree
    tree = cKDTree(pop.x, boxsize=np.asarray(domain, dtype=float) if domain else None)
    pairs = tree.query_pairs(r=2.0 * float(pop.radius.max()), output_type="ndarray")
    worst = 0.0
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        d = pop.x[j] - pop.x[i]
        if domain is not None:
            L = np.asarray(domain, dtype=float)
            d -= L * np.round(d / L)
        dist = np.linalg.norm(d, axis=1)
        overlap = pop.radius[i] + pop.radius[j] - dist
        worst = float(np.max(overlap / np.minimum(pop.radius[i], pop.radius[j])))
    passed = worst <= overlap_tolerance
    record["checks"]["non_overlap"] = {"max_relative_overlap": worst,
                                       "tolerance": overlap_tolerance,
                                       "pass": bool(passed)}
    ok &= passed
    record["pass"] = bool(ok)
    return record


def population_from_breakup(fragment_area, centroids, thickness, *,
                            parent_area: float | None = None,
                            source: str = "level_ice",
                            metadata: dict | None = None) -> FloePopulation:
    """Build a population from a breakup event, **area-conserving**.

    Fragment *areas* (what a bonded-particle breakup produces) are converted to
    equivalent radii ``r = sqrt(A/pi)``, so the total area is preserved by
    construction; ``parent_area`` defaults to that same total and
    :func:`validate_population` checks it.
    """
    area = np.asarray(fragment_area, dtype=float).ravel()
    if np.any(area <= 0.0):
        raise ValueError("fragment areas must be positive")
    x = np.asarray(centroids, dtype=float).reshape(-1, 2)
    if len(x) != len(area):
        raise ValueError("centroids and areas differ in length")
    t = np.broadcast_to(np.asarray(thickness, dtype=float), area.shape).copy()
    total = float(area.sum())
    return FloePopulation(
        x=x, radius=np.sqrt(area / np.pi), thickness=t, source=source,
        parent_area=total if parent_area is None else float(parent_area),
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class ClearIceRequest:
    """What a driver asks the clear-ice field to compute."""

    population: FloePopulation
    domain: tuple[float, float] = (48.0, 32.0)
    gate_width: float = 6.0
    drift_speed: float = 0.30
    duration: float = 8.0
    dt: float = 5.0e-4
    contract_version: str = CONTRACT_VERSION

    def as_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "domain_m": list(self.domain),
            "gate_width_m": self.gate_width,
            "drift_speed_m_per_s": self.drift_speed,
            "duration_s": self.duration,
            "dt_s": self.dt,
            "population": self.population.as_dict(),
        }


@dataclass(frozen=True)
class ClearIceResult:
    """The five report blocks the campaign requires, in one object."""

    floe_kinematics: dict
    force_chain_stats: dict
    porosity: dict
    convergence_report: dict
    uncertainty_quantification: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    REQUIRED = ("floe_kinematics", "force_chain_stats", "porosity", "convergence_report")

    def as_dict(self) -> dict:
        payload = {
            "contract_version": self.contract_version,
            "floe_kinematics": self.floe_kinematics,
            "force_chain_stats": self.force_chain_stats,
            "porosity": self.porosity,
            "convergence_report": self.convergence_report,
            "uncertainty_quantification": self.uncertainty_quantification,
            "provenance": self.provenance,
        }
        self.validate_payload(payload)
        return payload

    @staticmethod
    def validate_payload(payload: dict) -> None:
        missing = [k for k in ClearIceResult.REQUIRED if k not in payload]
        if missing:
            raise ValueError(f"result payload missing required blocks: {missing}")
        if not str(payload.get("contract_version", "")).startswith("clear-ice-contract/1."):
            raise ValueError("bad contract version")

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)