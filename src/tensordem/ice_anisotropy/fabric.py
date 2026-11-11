"""Crystal-axis orientation distribution function (ODF) and fabric ledger.

Why this module exists
----------------------
Columnar (S2) sea ice is a *fabric* material: the crystallographic c-axes lie
essentially in the horizontal plane, so a vertical core sees an almost
single-crystal response along the column axis and a randomly-oriented
polycrystal response across it.  Every anisotropic strength statement therefore
has to carry the orientation distribution it was derived from -- a strength
ratio without the fabric it belongs to is not a material property.

This module provides

* :class:`FabricODF` -- a discrete, weighted c-axis orientation distribution
  (the ODF is carried as samples, so a measured EBSD/fabric-analyser pole
  figure can be recorded verbatim; no parametric model has to be assumed),
* the second- and fourth-order orientation (fabric) tensors
  ``a_ij = <n_i n_j>`` / ``a_ijkl = <n_i n_j n_k n_l>``, which are the
  quantities a continuum anisotropy law actually consumes,
* :class:`FabricLedger` -- a JSON record/query store for ODFs, so an
  anisotropic parameter set can always be traced back to the fabric it was
  fitted on.

Conventions
-----------
* ``axis`` is the **column axis** (the grain-elongation / growth direction),
  i.e. the symmetry axis of the transverse isotropy.  For an S2 fabric the
  c-axes are *perpendicular* to it, and the texture is transversely isotropic
  about it.
* All directions are unit vectors in the specimen frame, +z up.
* The anisotropy index ``g`` is normalised so that a **uniform** ODF gives
  ``g = 0`` exactly and the ideal columnar (planar-random) fabric gives
  ``g = 1``; see :meth:`FabricODF.anisotropy_index`.

References
----------
- Weeks, W. F. & Assur, A. (1967). "The mechanical properties of sea ice."
  CRREL Monograph II-C3.  (S2 columnar fabric, c-axes in the horizontal plane.)
- Cole, D. M. (1987). "Strain-rate and grain-size effects in ice."
  J. Glaciol. 33(115), 252-259.  (fabric tensors for ice.)
- Kanatani, K. (1984). "Distribution of directional data and fabric tensors."
  Int. J. Eng. Sci. 22(2), 149-164.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "FabricODF",
    "FabricLedger",
    "cubic_orbit_directions",
    "fibonacci_sphere",
    "columnar_fabric",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n <= 0.0:
        raise ValueError("direction must be nonzero")
    return a / n


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` quasi-uniform unit vectors on S^2 (deterministic, no RNG)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    k = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def cubic_orbit_directions(seeds: Sequence[Sequence[float]]) -> np.ndarray:
    """Images of ``seeds`` under the 48-element cubic rotation group.

    The group average of ``n (x) n`` over such an orbit is proportional to the
    identity *exactly* (the only cubic-invariant second-order tensor is
    ``delta_ij``).  Sampling a uniform ODF this way therefore makes its fabric
    tensor isotropic to round-off, so the isotropic negative control is an
    identity rather than a statistical approximation.
    """
    out: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for seed in seeds:
        x, y, z = (abs(float(v)) for v in seed)
        for perm in ((x, y, z), (y, z, x), (z, x, y),
                     (x, z, y), (z, y, x), (y, x, z)):
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    for sz in (1.0, -1.0):
                        key = (round(sx * perm[0], 12), round(sy * perm[1], 12),
                               round(sz * perm[2], 12))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(key)
    return np.asarray(out, dtype=float)


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``axis``."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


# --------------------------------------------------------------------------
# ODF
# --------------------------------------------------------------------------
@dataclass
class FabricODF:
    """Discrete weighted c-axis orientation distribution (a fabric/pole figure).

    Parameters
    ----------
    directions
        ``(N, 3)`` array of unit c-axis directions (specimen frame).
    weights
        ``(N,)`` non-negative sample weights; normalised on construction.
    axis
        Column (grain-elongation) axis, i.e. the symmetry axis of the
        transverse isotropy.  Defaults to ``+z``.
    kind
        Free-form provenance tag (``"uniform"``, ``"columnar"``, ``"measured"``,
        ...) -- recorded in the ledger, never used arithmetically.
    label
        Optional human-readable identifier.
    provenance
        Optional dict describing where the ODF came from (e.g. thin-section
        count, ``{"source": "core 3", "method": "Rigsby stage"}``).
    """

    directions: np.ndarray
    weights: np.ndarray | None = None
    axis: Sequence[float] = (0.0, 0.0, 1.0)
    kind: str = "measured"
    label: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    def __post_init__(self) -> None:
        d = np.asarray(self.directions, dtype=float)
        if d.ndim != 2 or d.shape[1] != 3:
            raise ValueError("directions must have shape (N, 3)")
        if d.shape[0] < 1:
            raise ValueError("at least one c-axis direction is required")
        norms = np.linalg.norm(d, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("c-axis directions must be nonzero")
        self.directions = d / norms[:, None]
        if self.weights is None:
            w = np.full(len(self.directions), 1.0 / len(self.directions))
        else:
            w = np.asarray(self.weights, dtype=float).reshape(-1)
            if len(w) != len(self.directions):
                raise ValueError("weights must match directions")
            if np.any(w < 0.0):
                raise ValueError("weights must be non-negative")
            total = float(w.sum())
            if total <= 0.0:
                raise ValueError("weights must not all be zero")
            w = w / total
        self.weights = w
        self.axis = _unit(self.axis)

    # -- named fabrics -----------------------------------------------------
    @classmethod
    def uniform(cls, n_seeds: int = 20, *, axis: Sequence[float] = (0.0, 0.0, 1.0),
                label: str = "uniform") -> "FabricODF":
        """Isotropic fabric: c-axes uniform on the sphere (the negative control).

        Sampled as orbits of the cubic rotation group (:func:`cubic_orbit_directions`)
        so that ``a_ij = delta_ij/3`` holds to round-off and
        :meth:`anisotropy_index` returns ``0`` up to 1e-15 instead of a
        sampling-error offset.  The *second*-order tensor is exact by group
        symmetry; the fourth-order tensor keeps a cubic residual of order 1e-2
        (asserted in the tests) because the cubic group has non-isotropic
        fourth-order invariants -- the ODF is therefore isotropic as a fabric,
        not in every higher moment.
        """
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        grid = np.arange(0.5, 4.0, 0.7)
        seeds = [(a, b, c) for a in grid for b in grid for c in grid
                 if abs(a - b) > 0.4 and abs(b - c) > 0.4 and abs(a - c) > 0.4]
        seeds = seeds[:max(n_seeds, 1)]
        return cls(cubic_orbit_directions(seeds), axis=axis, kind="uniform", label=label)

    @classmethod
    def uniform_dense(cls, n: int = 24000, *, seed: int = 20261111,
                      axis: Sequence[float] = (0.0, 0.0, 1.0),
                      label: str = "uniform-dense") -> "FabricODF":
        """Isotropic fabric by dense pseudo-random sampling (fixed seed).

        :meth:`uniform` is exact in its *second* moment (cubic symmetry) but its
        orbits are clumped, so a *nonlinear* angular average -- the mean Schmid
        factor of the slip reference -- carries a quadrature bias of a few
        percent that does not fall monotonically with the sample count.  A dense
        random set is the right tool for that average: with ``n = 24000`` the
        directional variation of the mean-field slip strength is ~0.4 %.

        The price is statistical noise in the fabric tensor: ``g`` of this ODF is
        ~1e-2, not 0.  Use :meth:`uniform` when the *tensor* has to be exact and
        this one when an angular integral has to be accurate -- the report states
        which one each number came from.
        """
        if n < 1000:
            raise ValueError("n must be >= 1000 for a usable isotropic average")
        rng = np.random.default_rng(seed)
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        return cls(d, axis=axis, kind="uniform", label=label)

    @classmethod
    def columnar(cls, *, axis: Sequence[float] = (0.0, 0.0, 1.0), n_azimuth: int = 360,
                 spread_deg: float = 0.0, n_tilt: int = 1,
                 label: str = "columnar") -> "FabricODF":
        """Columnar S2 fabric: c-axes in the plane normal to ``axis``.

        Parameters
        ----------
        spread_deg
            Half-width of a uniform out-of-plane tilt of the c-axes.  ``0`` is
            the ideal S2 fabric (c-axes exactly in the plane); ``spread_deg``
            gives the belt a finite thickness.  ``spread_deg = 90`` blends
            towards the isotropic fabric (checked in the tests).
        n_tilt
            Number of tilt samples used to discretise ``spread_deg``.
        """
        ax = _unit(axis)
        e1, e2 = _orthonormal_basis(ax)
        az = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
        if n_tilt < 1:
            raise ValueError("n_tilt must be >= 1")
        if spread_deg < 0.0 or spread_deg > 90.0:
            raise ValueError("spread_deg must lie in [0, 90]")
        tilts = (np.array([0.0]) if spread_deg == 0.0 or n_tilt == 1
                 else np.linspace(-np.radians(spread_deg), np.radians(spread_deg), n_tilt))
        dirs: list[np.ndarray] = []
        wts: list[float] = []
        for tilt in tilts:
            w_tilt = np.cos(tilt)  # solid-angle weight of a belt element
            for a in az:
                in_plane = np.cos(a) * e1 + np.sin(a) * e2
                d = np.cos(tilt) * in_plane + np.sin(tilt) * ax
                dirs.append(d)
                wts.append(w_tilt)
        return cls(np.asarray(dirs), np.asarray(wts), axis=ax, kind="columnar", label=label)

    @classmethod
    def fisher(cls, *, axis: Sequence[float] = (0.0, 0.0, 1.0), kappa: float = 0.0,
               n: int = 2000, label: str = "fisher") -> "FabricODF":
        """Axial von Mises-Fisher fabric about ``axis`` (``kappa = 0`` -> uniform).

        Used to sweep continuously between the isotropic (``kappa = 0``) and the
        aligned (``kappa -> inf``) limits; the sampling is deterministic
        (importance-free inverse-CDF on a Fibonacci lattice), so the ODF is
        reproducible without an RNG.
        """
        if kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        ax = _unit(axis)
        e1, e2 = _orthonormal_basis(ax)
        u = np.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n)  # cos(theta) strata
        if kappa > 0.0:
            cdf = lambda c: (1.0 - np.exp(-kappa * (1.0 - c))) / (1.0 - np.exp(-2.0 * kappa))  # noqa: E731
            target = (np.arange(n) + 0.5) / n
            cos_t = np.empty(n)
            grid = np.linspace(-1.0, 1.0, 4096)
            vals = cdf(grid)
            for i, t in enumerate(target):
                j = int(np.searchsorted(vals, t))
                j = min(max(j, 1), len(grid) - 1)
                f = (t - vals[j - 1]) / max(vals[j] - vals[j - 1], 1e-300)
                cos_t[i] = grid[j - 1] + f * (grid[j] - grid[j - 1])
        else:
            cos_t = u
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
        phi = np.pi * (1.0 + 5.0 ** 0.5) * (np.arange(n) + 0.5)
        dirs = (cos_t[:, None] * ax[None, :]
                + sin_t[:, None] * (np.cos(phi)[:, None] * e1[None, :]
                                    + np.sin(phi)[:, None] * e2[None, :]))
        return cls(dirs, axis=ax, kind="fisher", label=label)

    # -- tensors -----------------------------------------------------------
    def fabric_tensor(self) -> np.ndarray:
        """Second-order orientation tensor ``a_ij = <n_i n_j>`` (trace 1)."""
        return (self.weights[:, None, None] * (self.directions[:, :, None]
                                              * self.directions[:, None, :])).sum(axis=0)

    def fourth_order_tensor(self) -> np.ndarray:
        """Fourth-order orientation tensor ``a_ijkl = <n_i n_j n_k n_l>``."""
        d = self.directions
        return (self.weights[:, None, None, None, None]
                * d[:, :, None, None, None] * d[:, None, :, None, None]
                * d[:, None, None, :, None] * d[:, None, None, None, :]).sum(axis=0)

    def eigenpairs(self) -> tuple[np.ndarray, np.ndarray]:
        """(eigenvalues descending, eigenvectors as columns) of the fabric tensor."""
        vals, vecs = np.linalg.eigh(self.fabric_tensor())
        order = np.argsort(vals)[::-1]
        return vals[order], vecs[:, order]

    def anisotropy_index(self) -> float:
        """Scalar fabric intensity ``g`` in ``[0, 1]``.

        ``g = (a_max - 1/3) / (1/2 - 1/3)`` clipped to ``[0, 1]``, where
        ``a_max`` is the largest eigenvalue of ``a_ij``.  A uniform ODF has
        ``a_max = 1/3`` hence ``g = 0`` **exactly**; the ideal columnar (planar
        random) fabric has ``a_max = 1/2`` hence ``g = 1``; a fully aligned
        fabric (``a_max = 1``) clips to ``g = 1``.
        """
        a_max = float(self.eigenpairs()[0][0])
        return float(np.clip((a_max - 1.0 / 3.0) / (0.5 - 1.0 / 3.0), 0.0, 1.0))

    def misorientation_from_axis_deg(self) -> np.ndarray:
        """Angle (deg) between each c-axis sample and the column axis."""
        c = np.clip(np.abs(self.directions @ self.axis), -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def mean_c_axis(self) -> np.ndarray | None:
        """Weighted mean c-axis direction, or ``None`` when it cancels (belt)."""
        m = (self.weights[:, None] * self.directions).sum(axis=0)
        n = float(np.linalg.norm(m))
        return None if n < 1e-9 else m / n

    # -- I/O ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "directions": self.directions.tolist(),
            "weights": self.weights.tolist(),
            "axis": list(map(float, self.axis)),
            "kind": self.kind,
            "label": self.label,
            "provenance": dict(self.provenance),
            "fabric_tensor": self.fabric_tensor().tolist(),
            "anisotropy_index": self.anisotropy_index(),
            "n_samples": int(len(self.directions)),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FabricODF":
        return cls(
            np.asarray(payload["directions"], dtype=float),
            None if payload.get("weights") is None else np.asarray(payload["weights"], dtype=float),
            axis=payload.get("axis", (0.0, 0.0, 1.0)),
            kind=payload.get("kind", "measured"),
            label=payload.get("label", ""),
            provenance=dict(payload.get("provenance", {})),
        )


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------
class FabricLedger:
    """JSON-backed record/query store for c-axis ODFs.

    The ledger is the traceability spine of the anisotropic strength model: an
    ``(sigma_parallel, sigma_perp)`` pair is only meaningful together with the
    ``sample_id`` of the ODF it was derived from.

    >>> led = FabricLedger()
    >>> sid = led.record(FabricODF.columnar(), meta={"core": "A"})
    >>> led.query(sid)["kind"]
    'columnar'
    """

    def __init__(self, records: Iterable[dict[str, Any]] | None = None) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        for r in records or ():
            self._records[str(r["sample_id"])] = dict(r)

    # -- write -------------------------------------------------------------
    def record(self, odf: FabricODF, *, sample_id: str | None = None,
               meta: dict[str, Any] | None = None) -> str:
        sid = sample_id or f"odf-{len(self._records) + 1:04d}"
        if sid in self._records:
            raise KeyError(f"sample_id {sid!r} already recorded")
        self._records[sid] = {
            "sample_id": sid,
            "odf": odf.to_dict(),
            "meta": dict(meta or {}),
        }
        return sid

    # -- read --------------------------------------------------------------
    def query(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self._records:
            raise KeyError(f"unknown sample_id {sample_id!r}")
        return self._records[sample_id]

    def odf(self, sample_id: str) -> FabricODF:
        return FabricODF.from_dict(self.query(sample_id)["odf"])

    def sample_ids(self) -> list[str]:
        return sorted(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def summary(self) -> list[dict[str, Any]]:
        """One row per sample: kind, label, n_samples, anisotropy index."""
        rows = []
        for sid in self.sample_ids():
            rec = self._records[sid]
            rows.append({
                "sample_id": sid,
                "kind": rec["odf"].get("kind", ""),
                "label": rec["odf"].get("label", ""),
                "n_samples": rec["odf"].get("n_samples", 0),
                "anisotropy_index": float(rec["odf"].get("anisotropy_index", 0.0)),
                "meta": rec.get("meta", {}),
            })
        return rows

    def select(self, kind: str) -> list[str]:
        return [sid for sid in self.sample_ids() if self._records[sid]["odf"].get("kind") == kind]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"schema": "tensordem.fabric_ledger.v1",
                                 "records": [self._records[s] for s in self.sample_ids()]},
                                indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "FabricLedger":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload.get("records", []))


def columnar_fabric(**kwargs: Any) -> FabricODF:
    """Shortcut for the reference S2 columnar fabric used by the strength model."""
    return FabricODF.columnar(**kwargs)