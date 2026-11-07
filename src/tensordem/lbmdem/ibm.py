"""Immersed-boundary schemes for the two-way LBM--DEM drag closure.

Two independent force-coupling paths are implemented, because the campaign has
to prove that the *drag number* does not depend on which one produced it:

``MomentumExchangeIBM``
    Noble--Torczynski (1998) partially saturated bounce-back (MEM).  Each
    interface node carries a solid fraction ``eps_s`` and blends the fluid
    post-collision state with a moving-wall bounce-back:

    ``f_i <- (1-B) f_i^post + B (f_{i-bar}^post + 6 w_i rho (c_i.u_b))``,
    ``B = eps_s (tau - 1/2) / ((1-eps_s) + (tau - 1/2))``.

``DirectForcingIBM``
    Uhlmann (2005) direct forcing on Lagrangian surface markers: interpolate
    the Eulerian velocity to the markers, force each marker toward the wall
    velocity, spread the marker forces back with the same 4-point Peskin kernel
    (so the fluid/particle momentum exchange is antisymmetric by construction).

Both schemes report the force *on the fluid*; the force on the particle is its
negative, which is the quantity a DEM solver consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .lattice import OPPOSITE, Lattice

__all__ = [
    "Sphere",
    "solid_fraction",
    "MomentumExchangeIBM",
    "DirectForcingIBM",
]


@dataclass(frozen=True)
class Sphere:
    """A rigid sphere in lattice units."""

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        if len(self.center) != 3:
            raise ValueError("center must have three components")
        if not self.radius > 0.0:
            raise ValueError("radius must be positive")

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius**3

    @property
    def area(self) -> float:
        return 4.0 * math.pi * self.radius**2

    def shifted(self, delta: Sequence[float]) -> "Sphere":
        c = self.center
        return Sphere((c[0] + float(delta[0]), c[1] + float(delta[1]), c[2] + float(delta[2])), self.radius)


def solid_fraction(
    shape: tuple[int, int, int],
    sphere: Sphere,
    *,
    sub: int = 4,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Solid volume fraction per cell by ``sub**3`` sub-point sampling.

    Only the bounding box of the sphere is touched; cells outside it keep zero.
    ``sub=1`` would be a binary mask; ``sub>=4`` gives the smooth fraction that
    the MEM blending needs to keep the interface second-order accurate.
    """
    if sub < 1:
        raise ValueError("sub must be >= 1")
    nx, ny, nz = shape
    eps = torch.zeros((nx, ny, nz), dtype=dtype)
    cx, cy, cz = sphere.center
    r = sphere.radius
    lo = [max(0, int(math.floor(c - r)) - 1) for c in (cx, cy, cz)]
    hi = [min(n - 1, int(math.ceil(c + r)) + 1) for c, n in zip((cx, cy, cz), (nx, ny, nz))]
    if lo[0] > hi[0] or lo[1] > hi[1] or lo[2] > hi[2]:
        return eps
    offsets = (torch.arange(sub, dtype=dtype) + 0.5) / sub
    gx = torch.arange(lo[0], hi[0] + 1, dtype=dtype).view(-1, 1, 1, 1, 1, 1) - 0.5 + offsets.view(
        1, 1, 1, sub, 1, 1
    )
    gy = torch.arange(lo[1], hi[1] + 1, dtype=dtype).view(1, -1, 1, 1, 1, 1) - 0.5 + offsets.view(
        1, 1, 1, 1, sub, 1
    )
    gz = torch.arange(lo[2], hi[2] + 1, dtype=dtype).view(1, 1, -1, 1, 1, 1) - 0.5 + offsets.view(
        1, 1, 1, 1, 1, sub
    )
    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    inside = (dist2 <= r * r).to(dtype)
    eps[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = inside.mean(dim=(3, 4, 5))
    return eps


class MomentumExchangeIBM:
    """Partially saturated bounce-back (MEM) with solid-fraction blending."""

    def __init__(
        self,
        lattice: Lattice,
        sphere: Sphere,
        *,
        sub: int = 4,
        wall_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.lattice = lattice
        self.sphere = sphere
        self.wall_velocity = torch.tensor(
            tuple(float(v) for v in wall_velocity), dtype=lattice.dtype
        )
        self.eps = solid_fraction(
            (lattice.nx, lattice.ny, lattice.nz), sphere, sub=sub, dtype=lattice.dtype
        )
        # Every node touched by the solid participates: fully solid nodes get
        # B = 1, i.e. pure bounce-back.  Parking fully solid nodes at rest
        # instead would silently destroy the momentum of the populations
        # streamed into them, which biases the measured drag low (an error that
        # grows with tau) and breaks the Newton's-third-law audit.
        self.interface = self.eps > 1e-12
        self.inside = self.eps >= 1.0 - 1e-12
        self.n_interface = int(self.interface.sum().item())
        if self.n_interface == 0:
            raise ValueError("sphere resolved no interface cells; increase the diameter")
        coords = self.interface.nonzero(as_tuple=False)
        self.ix, self.iy, self.iz = coords[:, 0], coords[:, 1], coords[:, 2]
        inside = self.inside.nonzero(as_tuple=False)
        if inside.numel():
            self.iix, self.iiy, self.iiz = inside[:, 0], inside[:, 1], inside[:, 2]
        else:
            self.iix = self.iiy = self.iiz = None
        eps_iface = self.eps[self.interface]
        tau_term = lattice.tau0 - 0.5
        blending = eps_iface * tau_term / ((1.0 - eps_iface) + tau_term)
        self.blending = blending.clamp(0.0, 1.0).unsqueeze(0)  # (1, n)
        # Fully solid nodes act as an absorbing wall: they are parked at their
        # own density moving with the wall velocity.  Their momentum loss is part
        # of the momentum exchange and is counted in the reported force, which
        # makes ``F = g V - dP_fluid/dt`` hold to round-off (see the audit test).
        self.solid_node = (eps_iface >= 1.0 - 1e-12).clone()
        self.n_solid = int(self.solid_node.sum().item())
        self.ratio = (1.0 - lattice.tau0) / tau_term  # unused, kept explicit
        self._opposite_idx = torch.tensor(OPPOSITE, dtype=torch.long)
        self._c_dot_ub = (lattice.c * self.wall_velocity.unsqueeze(0)).sum(dim=-1)  # (19,)
        self._force = torch.zeros(3, dtype=lattice.dtype)

    def reset_force(self) -> None:
        self._force = torch.zeros(3, dtype=self.lattice.dtype)

    @property
    def force_on_particle(self) -> torch.Tensor:
        """Force on the sphere (negative of the momentum given to the fluid)."""
        return -self._force

    def apply(self, f_post: torch.Tensor) -> torch.Tensor:
        lat = self.lattice
        f_iface = f_post[:, self.ix, self.iy, self.iz]  # (19, n)
        rho_local = f_iface.sum(dim=0)  # (n,)
        f_opp = f_iface[self._opposite_idx]  # (19, n)
        bounce = f_opp + (
            6.0 * lat.w.unsqueeze(-1) * self._c_dot_ub.unsqueeze(-1)
        ) * rho_local.unsqueeze(0)
        f_new = (1.0 - self.blending) * f_iface + self.blending * bounce
        if self.n_solid:
            count = rho_local.shape[0]
            u_wall = self.wall_velocity.reshape(1, 3).expand(count, 3)
            parked = lat.equilibrium(rho_local, u_wall)
            f_new = torch.where(self.solid_node.unsqueeze(0), parked, f_new)
        delta = f_new - f_iface
        # Momentum given to the fluid at the interface (per unit time).
        self._force = torch.einsum("ia,in->a", lat.c, delta)
        out = f_post.clone()
        out[:, self.ix, self.iy, self.iz] = f_new
        return out


def _fibonacci_directions(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Quasi-uniform directions on the unit sphere (Fibonacci spiral)."""
    if n < 4:
        raise ValueError("need at least 4 markers")
    k = torch.arange(n, dtype=dtype) + 0.5
    z = 1.0 - 2.0 * k / n
    rho = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = golden * k
    return torch.stack((rho * torch.cos(phi), rho * torch.sin(phi), z), dim=-1)


def _peskin4(r: torch.Tensor) -> torch.Tensor:
    """Four-point Peskin kernel ``phi(r)``, support ``|r| <= 2``."""
    a = r.abs()
    outer_zone = a > 1.0
    inner = (3.0 - 2.0 * a + torch.sqrt((1.0 + 4.0 * a - 4.0 * a * a).clamp_min(0.0))) / 8.0
    outer = (5.0 - 2.0 * a - torch.sqrt((-7.0 + 12.0 * a - 4.0 * a * a).clamp_min(0.0))) / 8.0
    return torch.where(outer_zone, outer, inner)


class DirectForcingIBM:
    """Uhlmann (2005) direct-forcing immersed boundary on Lagrangian markers."""

    def __init__(
        self,
        lattice: Lattice,
        sphere: Sphere,
        *,
        n_markers: int | None = None,
        marker_density: float = 1.4,
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.lattice = lattice
        self.sphere = sphere
        if n_markers is None:
            area = 4.0 * math.pi * sphere.radius**2
            n_markers = max(64, int(round(marker_density * area)))
        self.n_markers = int(n_markers)
        self.dirs = _fibonacci_directions(self.n_markers, dtype=lattice.dtype)
        self.marker_area = sphere.area / self.n_markers
        self.velocity = torch.tensor(tuple(float(v) for v in velocity), dtype=lattice.dtype)
        self._force = torch.zeros(3, dtype=lattice.dtype)
        self.marker_force = torch.zeros((self.n_markers, 3), dtype=lattice.dtype)

    def reset_force(self) -> None:
        self._force = torch.zeros(3, dtype=self.lattice.dtype)

    @property
    def force_on_particle(self) -> torch.Tensor:
        return -self._force

    def marker_positions(self) -> torch.Tensor:
        c = torch.tensor(self.sphere.center, dtype=self.lattice.dtype)
        return c + self.sphere.radius * self.dirs

    def _stencil(self) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Marker positions, integer stencil base and per-axis kernel weights."""
        pos = self.marker_positions()
        base = torch.floor(pos).to(torch.long) - 1  # four nodes per axis: base..base+3
        offsets = torch.arange(4, dtype=self.lattice.dtype)
        weights = []
        for axis in range(3):
            rel = pos[:, axis:axis + 1] - (
                base[:, axis:axis + 1].to(self.lattice.dtype) + offsets
            )
            weights.append(_peskin4(rel))  # (N, 4)
        return pos, base, weights

    def apply(self, u_field: torch.Tensor) -> torch.Tensor:
        """Return the Eulerian force density that drives the fluid to the wall velocity."""
        lat = self.lattice
        nx, ny, nz = lat.nx, lat.ny, lat.nz
        _, base, weights = self._stencil()
        n_m = self.n_markers
        axis_idx = []
        for axis, n in enumerate((nx, ny, nz)):
            idx = (base[:, axis].unsqueeze(-1) + torch.arange(4, dtype=torch.long)) % n
            axis_idx.append(idx)
        ix, iy, iz = axis_idx
        wx, wy, wz = weights
        idx_flat = (ix[:, :, None, None] * (ny * nz)) + (iy[:, None, :, None] * nz) + iz[:, None, None, :]
        w_flat = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(n_m, 64)
        u_flat = u_field.reshape(nx * ny * nz, 3)
        gathered = u_flat[idx_flat.reshape(-1)]  # (N*64, 3)
        u_interp = (gathered * w_flat.reshape(-1, 1)).reshape(n_m, 64, 3).sum(dim=1)
        target = self.velocity.reshape(1, 3).expand(n_m, 3)
        marker_force = (target - u_interp) * self.marker_area
        self.marker_force = marker_force
        self._force = marker_force.sum(dim=0)
        density = torch.zeros(nx * ny * nz, 3, dtype=lat.dtype)
        contrib = marker_force.reshape(n_m, 1, 3) * w_flat.reshape(n_m, 64, 1)
        density.index_add_(0, idx_flat.reshape(-1), contrib.reshape(-1, 3))
        return density.reshape(nx, ny, nz, 3)

    def step(self, *, extra_hook=None) -> None:
        """Advance the coupled system by one lattice step with direct forcing."""
        lat = self.lattice
        _, u = lat.macroscopic(lat.f)
        force_density = self.apply(u)
        lat.step(force_density=force_density, post_collision_hook=extra_hook)