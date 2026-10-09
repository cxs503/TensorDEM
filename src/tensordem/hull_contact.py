"""Penalty contact between DEM disks and a prescribed 2-D polyline hull."""
from __future__ import annotations
import math
import torch


def polyline_contact_forces(positions, velocities, hull_vertices, hull_velocity, *,
                            particle_radius, hull_radius, stiffness, damping=0.0):
    """Return particle forces (N,2) and the equal/opposite hull reaction (2,).

    Hull vertices must already be translated to their global positions at this
    time step. Each disk interacts with the nearest point on the polyline,
    including its endpoints. Units are SI: m, m/s, N/m, N*s/m, and N.
    """
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not isinstance(velocities, torch.Tensor) or velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")
    if not isinstance(hull_vertices, torch.Tensor) or hull_vertices.ndim != 2 or hull_vertices.shape[1] != 2 or hull_vertices.shape[0] < 2:
        raise ValueError("hull_vertices must have shape (M, 2), M >= 2")
    if not isinstance(hull_velocity, torch.Tensor) or hull_velocity.numel() != 2:
        raise ValueError("hull_velocity must contain two components")
    for name, value in (("particle_radius", particle_radius), ("stiffness", stiffness)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("hull_radius", hull_radius), ("damping", damping)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    device, dtype = positions.device, positions.dtype
    p = positions
    v = velocities.to(device=device, dtype=dtype)
    vertices = hull_vertices.to(device=device, dtype=dtype)
    hv = hull_velocity.to(device=device, dtype=dtype).reshape(2)
    if not bool(torch.isfinite(p).all() and torch.isfinite(v).all()
                and torch.isfinite(vertices).all() and torch.isfinite(hv).all()):
        raise ValueError("inputs must contain only finite values")

    starts, segments = vertices[:-1], vertices[1:] - vertices[:-1]
    length_sq = segments.square().sum(dim=1)
    if bool((length_sq <= torch.finfo(dtype).eps).any()):
        raise ValueError("hull polyline must not contain zero-length segments")
    rel = p[:, None, :] - starts[None, :, :]
    fraction = ((rel * segments[None, :, :]).sum(dim=2) / length_sq[None, :]).clamp(0.0, 1.0)
    closest = starts[None, :, :] + fraction[:, :, None] * segments[None, :, :]
    delta = p[:, None, :] - closest
    distance_sq = delta.square().sum(dim=2)
    nearest = distance_sq.argmin(dim=1)
    rows = torch.arange(p.shape[0], device=device)
    d = delta[rows, nearest]
    distance = torch.sqrt(distance_sq[rows, nearest])
    tangent = segments[nearest] / torch.sqrt(length_sq[nearest])[:, None]
    fallback = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
    normal = d / distance.clamp_min(torch.finfo(dtype).eps)[:, None]
    normal = torch.where((distance > torch.finfo(dtype).eps)[:, None], normal, fallback)

    overlap = (particle_radius + hull_radius - distance).clamp_min(0.0)
    normal_speed = ((v - hv) * normal).sum(dim=1)
    magnitude = (stiffness * overlap - damping * normal_speed).clamp_min(0.0)
    magnitude = torch.where(overlap > 0, magnitude, torch.zeros_like(magnitude))
    forces = magnitude[:, None] * normal
    return forces, -forces.sum(dim=0)
