"""Kinematic transfer: FEM node field -> rigid-body initial conditions per floe.

A quasi-static plate solve leaves the DEM with nothing to start from: the floes
are in equilibrium at the fracture load, so the obvious (and wrong) choice is
"velocity = 0".  The floes are not motionless relative to each other -- the plate
is *bending*, and the moment the fracture state is reached the released floes
carry the kinematic signature of that bending: a heave offset, a velocity
proportional to the load rate, and pitch/roll rates proportional to the slope
field.

This module extracts exactly that, and reports the number that decides whether a
rigid-body (DEM) representation is legitimate at all: the **rigidity residual**

.. math::  \\varepsilon_\\text{rig} = \\sqrt{\\frac{1}{n}\\sum_n (w_n - w_0 - p x_n - q y_n)^2} \\Big/ \\Delta w_\\text{floe}

i.e. how much of the floe's own deflection field the best-fitting rigid plane
cannot reproduce.  Small ``eps_rig`` means the fragment really is translating and
tilting as a rigid body (so the continuum -> discrete handoff loses nothing
dynamically); large ``eps_rig`` means the floe is still bending and a rigid floe
DEM would be a modelling error that no mass ledger can detect.

The FE slope field ``(w,x, w,y)`` is used as an *independent* check on the fitted
plane: the plane's gradients must agree with the area-mean of the FE slopes.  Two
different interpolations agreeing is evidence; one interpolation agreeing with
itself is not.

Sign convention (small rotations, right-handed ``z`` up)
------------------------------------------------------
A rigid plane ``w(x, y) = w0 + p (x - x0) + q (y - y0)`` about the floe reference
point ``(x0, y0)`` -- the centroid of the floe's own FEM nodes -- is a rotation
vector ``omega = (-q, p, 0)``, because a rotation about ``y`` by ``theta_y`` lifts
``z`` in proportion to ``x`` and a rotation about ``x`` by ``theta_x`` lowers ``z``
in proportion to ``y``.  The reference point matters: ``w0`` is the *translation of
the floe*, i.e. exactly the ``z0`` a DEM body needs, whereas the value of the same
plane at the plate origin is a mix of translation and tilt.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np


__all__ = [
    "RigidPlaneFit",
    "BodyKinematics",
    "rigid_plane_fit",
    "floe_node_mask",
    "kinematic_transfer",
    "read_node_state",
]


@dataclass(frozen=True)
class RigidPlaneFit:
    """Least-squares rigid-plane description of a floe's deflection field."""

    w0: float                     # rigid heave offset at (x0, y0) [m]
    p: float                      # slope dw/dx [-]
    q: float                      # slope dw/dy [-]
    rms: float                    # residual rms of the plane fit [m]
    rel_rms: float                # rms / floe deflection range [-]
    n_points: int
    w_range: float                # max - min of the floe deflection [m]
    theta_x_mean: float           # area-mean of the FE slope w,x [-]
    theta_y_mean: float           # area-mean of the FE slope w,y [-]
    length_scale: float = 0.0     # sqrt(floe area) [m], sets the natural slope
    x0: float = 0.0               # reference point of the rigid plane [m]
    y0: float = 0.0

    def evaluate(self, x, y):
        """Rigid-plane deflection at ``(x, y)`` [m]."""
        return self.w0 + self.p * (np.asarray(x) - self.x0) + self.q * (np.asarray(y) - self.y0)

    @property
    def pitch(self) -> float:
        """Rotation about ``y`` [rad] (small angle)."""
        return self.p

    @property
    def roll(self) -> float:
        """Rotation about ``x`` [rad] (small angle)."""
        return -self.q

    def slope_check(self) -> dict:
        """Compare the fitted slopes with the mean FE slopes.

        The gap is normalised by the **natural slope of the fragment**,
        ``w_range / sqrt(A)`` -- the tilt a floe of that size would have if its
        deflection range happened to be a straight ramp.  Normalising by
        ``max(|p|, |theta|)`` instead is meaningless for a fragment whose slopes
        are small: two nearly-zero numbers differ by "100 %" trivially.  The
        absolute gap is reported as well so the reader can judge both ways.
        """
        scale = (self.w_range / self.length_scale) if self.length_scale > 0.0 else 0.0
        floor = 0.05 * scale

        def gap(a, b):
            d = abs(a - b)
            denom = max(abs(a), abs(b), floor, 1e-300)
            return float(d), float(d / denom)

        dx, rx = gap(self.p, self.theta_x_mean)
        dy, ry = gap(self.q, self.theta_y_mean)
        return {
            "p": self.p,
            "theta_x_mean": self.theta_x_mean,
            "q": self.q,
            "theta_y_mean": self.theta_y_mean,
            "abs_gap_x": dx,
            "abs_gap_y": dy,
            "rel_x": rx,
            "rel_y": ry,
            "natural_slope": float(scale),
            "rel_to_natural_x": float(dx / scale) if scale > 0.0 else float("inf"),
            "rel_to_natural_y": float(dy / scale) if scale > 0.0 else float("inf"),
        }


@dataclass(frozen=True)
class BodyKinematics:
    """Initial state handed to one DEM body."""

    floe_id: int
    z0: float                     # heave offset at the fracture state [m]
    v_z: float                    # heave velocity at the load rate [m/s]
    v_x: float
    v_y: float
    omega_x: float
    omega_y: float
    fit: RigidPlaneFit
    n_nodes: int

    @property
    def speed(self) -> float:
        return float(np.hypot(self.v_x, self.v_y))

    def as_payload(self) -> dict:
        return {
            "floe_id": self.floe_id,
            "z0_m": self.z0,
            "v_z_m_s": self.v_z,
            "v_x_m_s": self.v_x,
            "v_y_m_s": self.v_y,
            "omega_x_rad_s": self.omega_x,
            "omega_y_rad_s": self.omega_y,
            "rigidity_rel_residual": self.fit.rel_rms,
            "floe_deflection_range_m": self.fit.w_range,
            "n_fem_nodes": self.n_nodes,
            "slope_cross_check": self.fit.slope_check(),
        }


def rigid_plane_fit(xy, w, theta_x=None, theta_y=None, *, weights=None,
                    length_scale: float = 0.0) -> RigidPlaneFit:
    """Fit ``w ~ w0 + p (x - x0) + q (y - y0)`` by (weighted) least squares.

    The reference point ``(x0, y0)`` is the weighted centroid of the sample, so
    ``w0`` is the rigid translation of the floe and ``p, q`` its tilt; without
    centring the design matrix the three coefficients are strongly correlated for
    a fragment far from the plate origin.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    w = np.asarray(w, dtype=float).reshape(-1)
    if xy.shape[0] != w.size:
        raise ValueError("one deflection per node is required")
    if xy.shape[0] < 3:
        raise ValueError("a rigid plane needs at least three nodes")
    if weights is None:
        ww = np.ones(xy.shape[0])
    else:
        ww = np.asarray(weights, dtype=float).reshape(-1)
        if ww.size != w.size or np.any(ww <= 0.0):
            raise ValueError("weights must be positive and one per node")
    # Centre the design matrix for conditioning, then map the slopes back.
    x0 = float(np.average(xy[:, 0], weights=ww))
    y0 = float(np.average(xy[:, 1], weights=ww))
    dx = xy[:, 0] - x0
    dy = xy[:, 1] - y0
    A = np.column_stack([np.ones_like(dx), dx, dy])
    sw = np.sqrt(ww)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], w * sw, rcond=None)
    resid = w - A @ coef
    rms = float(np.sqrt(np.average(resid * resid, weights=ww)))
    w_range = float(np.max(w) - np.min(w))
    rel = float(rms / w_range) if w_range > 0.0 else 0.0
    tx = float(np.average(np.asarray(theta_x, dtype=float), weights=ww)) \
        if theta_x is not None else float("nan")
    ty = float(np.average(np.asarray(theta_y, dtype=float), weights=ww)) \
        if theta_y is not None else float("nan")
    return RigidPlaneFit(
        w0=float(coef[0]),
        p=float(coef[1]),
        q=float(coef[2]),
        rms=rms,
        rel_rms=rel,
        n_points=int(xy.shape[0]),
        w_range=w_range,
        theta_x_mean=tx,
        theta_y_mean=ty,
        length_scale=float(length_scale),
        x0=float(x0),
        y0=float(y0),
    )


def _inside_ring(pts: np.ndarray, ring) -> np.ndarray:
    """Vectorised ray casting (same rule as the handoff's scalar version)."""
    v = np.asarray(ring, dtype=float)
    px = pts[:, 0]
    py = pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    n = v.shape[0]
    for i in range(n):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        straddles = (y1 > py) != (y2 > py)
        if not np.any(straddles):
            continue
        x_cross = np.empty_like(px)
        x_cross[straddles] = x1 + (py[straddles] - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (px < x_cross)
    return inside


def floe_node_mask(xy, rings) -> np.ndarray:
    """Boolean mask of the nodes strictly inside the floe footprint."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    rings = list(rings)
    inside = _inside_ring(xy, rings[0])
    for hole in rings[1:]:
        inside &= ~_inside_ring(xy, hole)
    return inside


def read_node_state(path_or_payload) -> dict:
    """Read ``fem_node_state.json`` (the kinematic companion of the handoff)."""
    if isinstance(path_or_payload, dict):
        payload = path_or_payload
    else:
        with open(path_or_payload, encoding="utf-8") as fh:
            payload = json.load(fh)
    nodes = np.asarray(payload["nodes"], dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] < 5:
        raise ValueError("nodes must be an (n, >=5) array of x, y, w, theta_x, theta_y")
    return {
        "node_xy": nodes[:, :2].copy(),
        "w": nodes[:, 2].copy(),
        "theta_x": nodes[:, 3].copy(),
        "theta_y": nodes[:, 4].copy(),
        "load_factor": float(payload["load_factor"]),
        "load_rate_per_s": float(payload["load_rate_per_s"]),
        "mesh": payload.get("mesh", {}),
        "source": payload.get("source", "tensordem.fem_dem_fracture"),
    }


def kinematic_transfer(faces, node_state, *, load_rate=None, current=None) -> list[BodyKinematics]:
    """Rigid-body initial conditions per floe from the FE node field.

    ``load_rate`` overrides the rate stored in the node state [1/s]; the default
    is the stored value.  ``current`` is an optional callable ``(x, y) ->
    (vx, vy)`` [m/s] supplying the in-plane velocity (an ambient drift, since a
    plate-bending solve has no membrane kinematics); the default is rest.
    """
    xy = node_state["node_xy"]
    w = node_state["w"]
    rate = float(node_state["load_rate_per_s"] if load_rate is None else load_rate)
    out: list[BodyKinematics] = []
    for floe in faces.floes:
        rings = floe.polygons
        mask = floe_node_mask(xy, rings)
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"floe {floe.floe_id} has fewer than 3 FEM nodes inside it")
        fit = rigid_plane_fit(xy[mask], w[mask],
                              node_state["theta_x"][mask], node_state["theta_y"][mask],
                              length_scale=math.sqrt(max(floe.area, 1e-30)))
        vx = vy = 0.0
        if current is not None:
            vx, vy = (float(c) for c in current(*floe.centroid))
        out.append(BodyKinematics(
            floe_id=floe.floe_id,
            z0=fit.w0,
            v_z=rate * fit.w0,
            v_x=vx,
            v_y=vy,
            omega_x=rate * fit.roll,
            omega_y=rate * fit.pitch,
            fit=fit,
            n_nodes=int(np.count_nonzero(mask)),
        ))
    out.sort(key=lambda k: k.floe_id)
    return out