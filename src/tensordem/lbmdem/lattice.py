"""D3Q19 lattice-Boltzmann core for two-way LBM--DEM drag calibration.

Lattice units throughout (``dx = dt = 1``, ``c_s^2 = 1/3``).  The solver is
deliberately small and explicit: BGK collision with an optional Smagorinsky
sub-grid viscosity, momentum-kick forcing, and periodic streaming.  It is the
fluid half of the LBM--DEM coupling; the immersed-boundary half lives in
:mod:`tensordem.lbmdem.ibm`.

Why this shape
--------------
The drag-closure calibration needs three things from the fluid solver:

* **exact momentum bookkeeping** -- periodic streaming conserves
  ``sum_i c_i f_i`` identically, BGK collision conserves it identically, and the
  force is injected as the momentum kick ``f_i += 3 w_i (c_i.F)`` whose momentum
  sum is exactly ``F`` (``sum_i w_i c_i c_i = I/3``, ``sum_i w_i c_i = 0``).
  Hence ``d(m v)/dt + d(P_fluid)/dt - F_external`` closes to round-off, which is
  what the ``<= 1e-10`` audit of the campaign measures.  (Guo forcing was
  rejected here: its split ``F/2`` velocity shift plus ``(1 - 1/2tau)`` source
  does *not* inject exactly ``F`` per step, so a 1e-10 momentum audit of a Guo
  solver would only measure its own bookkeeping defect.  The accuracy price of
  the kick scheme is O(F^2), negligible at the forcing levels used, and the
  calibration checks the resulting drag against the analytic anchors.)
* an explicit, controllable viscosity (``tau0``) at ``Re -> 0``, so the Stokes
  anchor is reached by lowering the forcing rather than by tuning anything;
* an optional eddy viscosity, so that the Smagorinsky constant ``Cs`` can be
  shown to be *irrelevant* in the Stokes limit -- the physical-consistency gate
  of the campaign.

The population array is stored as ``(19, nx, ny, nz)``: every population is then
a contiguous 3-D block, which makes the 19 periodic shifts of the streaming step
contiguous copies instead of strided gathers (about 10x faster at the grid sizes
used by the benchmark).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = ["D3Q19_C", "D3Q19_W", "OPPOSITE", "Lattice", "LatticeShape"]

# fmt: off
D3Q19_C: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)
# fmt: on
D3Q19_W: tuple[float, ...] = (1.0 / 3.0,) + (1.0 / 18.0,) * 6 + (1.0 / 36.0,) * 12


def _opposite() -> tuple[int, ...]:
    lookup = {c: i for i, c in enumerate(D3Q19_C)}
    return tuple(lookup[(-cx, -cy, -cz)] for (cx, cy, cz) in D3Q19_C)


OPPOSITE: tuple[int, ...] = _opposite()


@dataclass(frozen=True)
class LatticeShape:
    """Lattice dimensions plus the indexing convention used everywhere."""

    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 4:
                raise ValueError(f"{name} must be at least 4 cells")

    @property
    def volume(self) -> int:
        return self.nx * self.ny * self.nz


class Lattice:
    """D3Q19 BGK + Smagorinsky solver in lattice units, ``f`` shaped (19, nx, ny, nz)."""

    def __init__(
        self,
        shape: LatticeShape | Sequence[int],
        *,
        tau0: float = 0.8,
        smagorinsky_cs: float = 0.0,
        body_force: Sequence[float] = (0.0, 0.0, 0.0),
        dtype: str = "float64",
    ) -> None:
        self.shape = shape if isinstance(shape, LatticeShape) else LatticeShape(*shape)
        self.nx, self.ny, self.nz = self.shape.nx, self.shape.ny, self.shape.nz
        if not tau0 > 0.5:
            raise ValueError("tau0 must exceed 0.5 for a positive viscosity")
        if smagorinsky_cs < 0.0:
            raise ValueError("smagorinsky_cs must be nonnegative")
        self.tau0 = float(tau0)
        self.smagorinsky_cs = float(smagorinsky_cs)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.c = torch.tensor(D3Q19_C, dtype=self.dtype)
        self.cx = self.c[:, 0].contiguous()
        self.cy = self.c[:, 1].contiguous()
        self.cz = self.c[:, 2].contiguous()
        self._cx = self.cx.view(19, 1, 1, 1)
        self._cy = self.cy.view(19, 1, 1, 1)
        self._cz = self.cz.view(19, 1, 1, 1)
        self.w = torch.tensor(D3Q19_W, dtype=self.dtype)
        self._w = self.w.view(19, 1, 1, 1)
        self.opposite = torch.tensor(OPPOSITE, dtype=torch.long)
        self.body_force = torch.tensor(tuple(float(v) for v in body_force), dtype=self.dtype)
        self._body = self.body_force.view(3, 1, 1, 1)
        # Off-diagonals enter Pi_ab Pi_ab twice because the stress is symmetric.
        self._stress_weight = torch.tensor((1.0, 1.0, 1.0, 2.0, 2.0, 2.0), dtype=self.dtype)
        self.rho = torch.ones((self.nx, self.ny, self.nz), dtype=self.dtype)
        self.u = torch.zeros((self.nx, self.ny, self.nz, 3), dtype=self.dtype)
        self.f = self.equilibrium(self.rho, self.u)
        self.impulse_body_force = 0.0
        self.steps = 0

    # -- equilibrium / moments -------------------------------------------
    def equilibrium(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Second-order Maxwellian equilibrium, returned with a leading stencil axis.

        Works for both a 3-D lattice field (``(nx, ny, nz)`` -> ``(19, nx, ny, nz)``)
        and a flat list of nodes (``(n,)`` -> ``(19, n)``), which the immersed
        boundary needs for its interior nodes.
        """
        # u carries a trailing component axis, so the stencil axis needs one
        # more dimension than the field itself.
        rank = rho.dim()
        shape = (19,) + (1,) * rank
        cx = self.cx.view(shape)
        cy = self.cy.view(shape)
        cz = self.cz.view(shape)
        w = self.w.view(shape)
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        cu = cx * ux + cy * uy + cz * uz
        uu = ux * ux + uy * uy + uz * uz
        return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu.unsqueeze(0))

    def macroscopic(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Density and velocity moments."""
        rho = f.sum(dim=0)
        mx = (f * self._cx).sum(dim=0)
        my = (f * self._cy).sum(dim=0)
        mz = (f * self._cz).sum(dim=0)
        u = torch.stack((mx, my, mz), dim=-1) / rho.unsqueeze(-1)
        return rho, u

    def initialize_velocity(self, velocity: Sequence[float], *, density: float = 1.0) -> None:
        """Reset the field to a uniform equilibrium flow (transient shortcut)."""
        u = torch.tensor(tuple(float(v) for v in velocity), dtype=self.dtype)
        self.rho = torch.full((self.nx, self.ny, self.nz), float(density), dtype=self.dtype)
        self.u = u.reshape(1, 1, 1, 3).expand(self.nx, self.ny, self.nz, 3).clone()
        self.f = self.equilibrium(self.rho, self.u)

    def _neq_stress(self, f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
        df = f - feq
        return torch.stack(
            [
                (df * (self._cx * self._cx)).sum(dim=0),
                (df * (self._cy * self._cy)).sum(dim=0),
                (df * (self._cz * self._cz)).sum(dim=0),
                (df * (self._cx * self._cy)).sum(dim=0),
                (df * (self._cx * self._cz)).sum(dim=0),
                (df * (self._cy * self._cz)).sum(dim=0),
            ],
            dim=-1,
        )

    def effective_tau(self, f: torch.Tensor, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """BGK + Smagorinsky: ``tau = 1/2 + 3 (nu0 + Cs^2 |S|)``.

        ``|S| = sqrt(2 S_ab S_ab)`` with the Chapman--Enskog relation
        ``S_ab = -3 Pi_ab^neq / (2 tau rho)``.  The closure is explicit (``tau``
        on the right-hand side is ``tau0``), the standard inexpensive variant.
        At ``Re -> 0`` the correction is O(|Pi^neq|) and vanishes, which is what
        the "Stokes drag is Cs-independent" gate measures.
        """
        if self.smagorinsky_cs == 0.0:
            return torch.full_like(rho, self.tau0)
        feq = self.equilibrium(rho, u)
        pi = self._neq_stress(f, feq)
        norm = torch.sqrt((self._stress_weight * pi * pi).sum(dim=-1).clamp_min(0.0))
        strain = 3.0 * norm / (2.0 * self.tau0 * rho.clamp_min(1e-12))
        nu_t = self.smagorinsky_cs**2 * math.sqrt(2.0) * strain
        return torch.clamp(0.5 + 3.0 * ((self.tau0 - 0.5) / 3.0 + nu_t), min=0.5001)

    # -- collision / streaming -------------------------------------------
    def collide(self, f: torch.Tensor, *, force_density: torch.Tensor | None = None) -> torch.Tensor:
        """BGK collision plus an exact-momentum force kick.

        The kick ``f_i += 3 w_i (c_i . F)`` adds momentum ``F`` and mass ``0``
        per cell per step exactly, which is what makes the global audit exact.
        """
        rho, u = self.macroscopic(f)
        feq = self.equilibrium(rho, u)
        tau = self.effective_tau(f, rho, u)
        omega = (1.0 / tau).unsqueeze(0)
        f_post = f - omega * (f - feq)
        force = self._body
        if force_density is not None:
            force = force + force_density.permute(3, 0, 1, 2)
        if force_density is not None or float(self.body_force.abs().sum()) > 0.0:
            kick = self._cx * force[0] + self._cy * force[1] + self._cz * force[2]
            f_post = f_post + 3.0 * self._w * kick
        return f_post

    def stream(self, f: torch.Tensor) -> torch.Tensor:
        """Periodic pull-streaming: ``f_i(x) <- f_i(x - c_i)`` (momentum exact)."""
        out = torch.empty_like(f)
        for i, (cx, cy, cz) in enumerate(D3Q19_C):
            if cx == 0 and cy == 0 and cz == 0:
                out[i] = f[i]
            else:
                out[i] = torch.roll(f[i], shifts=(cx, cy, cz), dims=(0, 1, 2))
        return out

    # -- momentum bookkeeping --------------------------------------------
    def fluid_momentum(self, f: torch.Tensor | None = None) -> torch.Tensor:
        """Total fluid momentum ``sum_x sum_i c_i f_i``."""
        field = self.f if f is None else f
        return torch.stack(
            ((field * self._cx).sum(), (field * self._cy).sum(), (field * self._cz).sum())
        )

    def step(self, *, force_density: torch.Tensor | None = None, post_collision_hook=None) -> None:
        """One full step: collide -> optional boundary blend -> stream."""
        f_post = self.collide(self.f, force_density=force_density)
        if post_collision_hook is not None:
            f_post = post_collision_hook(f_post)
        self.f = self.stream(f_post)
        self.impulse_body_force += float(self.body_force[0]) * self.shape.volume
        self.steps += 1