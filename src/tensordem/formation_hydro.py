"""Hydrodynamic interference coefficients for a multi-ship formation.

Model
-----
At the low advance speeds of ice-going operation (``Fn = U / sqrt(gL) < 0.1``)
the free surface is effectively rigid, so the flow about a formation in a channel
is a *double-body* potential flow in the horizontal plane.  Each hull is
represented by its waterline contour, discretised with constant-strength Rankine
(source) panels; the no-penetration condition on every hull and on the optional
channel walls is satisfied by one dense linear system.  The source strengths give
the disturbance velocity of each ship, from which the pairwise interference
coefficient of ship ``j`` on ship ``i`` follows as a first-order perturbation of
the ship's effective advance speed:

    C_ij = beta * <u_x^(j)> / U          (beta = 2 for quadratic resistance)

``<.>`` is the mean over the hull's panels; the coefficient is positive when the
neighbour accelerates the local inflow (added resistance) and negative when it
decelerates it (drafting benefit).  This inflow-perturbation argument is the
standard first-order model for ship-ship and shallow-water interaction.  Because
it is first order, ``|C_ij|`` is capped by
:attr:`FormationFlowConfig.max_induced_fraction` and the result reports whether
the cap was active.

Verification (``tests/test_formation_hydro.py``):
* no-penetration residual of the panel solution,
* circular waterline against the exact potential flow ``u_theta = 2 U sin(theta)``
  and the exact dipole far field,
* panel-count convergence,
* asymptotics ``C_ij -> 0`` at large separation and boundedness at contact.

Wave-making interference is deliberately excluded: at ice-going Froude numbers
(``Fn <= 0.06`` for the cases in this campaign) the wave resistance of a
displacement hull is a fraction of a percent of the level-ice resistance, and the
hulls additionally operate in a confined channel.  The Froude number of every
case is reported so that the applicability window is explicit.

All quantities are SI (m, m/s, kg/m3).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def _number(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class HullWaterline:
    """Elliptic waterline contour used as the interference-model hull.

    ``b(x) = (B/2) sqrt(1 - (2x/L)^2)`` for ``x in [-L/2, L/2]``; the bow is at
    ``x = -L/2`` because the onset flow of the resistance frame runs towards +x.
    A circle (``L = B``) reproduces the classical cylinder benchmark.
    """

    length: float
    beam: float
    panels_per_side: int = 20

    def __post_init__(self) -> None:
        _number(self.length, "length", positive=True)
        _number(self.beam, "beam", positive=True)
        if isinstance(self.panels_per_side, bool) or not isinstance(self.panels_per_side, int):
            raise TypeError("panels_per_side must be an integer")
        if self.panels_per_side < 6:
            raise ValueError("panels_per_side must be >= 6")

    def contour(self) -> torch.Tensor:
        """Closed waterline polygon ``(M, 2)`` with the bow first."""
        n = self.panels_per_side
        upper: list[tuple[float, float]] = []
        for i in range(n + 1):
            xi = -math.cos(math.pi * i / n)
            half = 0.5 * self.beam * math.sqrt(max(0.0, 1.0 - xi * xi))
            upper.append((0.5 * self.length * xi, half))
        lower = [(x, -y) for x, y in upper[n - 1:0:-1]]
        return torch.tensor(upper + lower, dtype=torch.float64)


@dataclass(frozen=True)
class FormationFlowConfig:
    """Configuration of the double-body panel solution."""

    speed: float
    water_density: float = 1025.0
    resistance_exponent: float = 2.0
    max_induced_fraction: float = 0.6
    channel_half_width: float | None = None
    channel_panels: int = 60
    channel_length_factor: float = 6.0

    def __post_init__(self) -> None:
        _number(self.speed, "speed", positive=True)
        _number(self.water_density, "water_density", positive=True)
        _number(self.resistance_exponent, "resistance_exponent", positive=True)
        _number(self.max_induced_fraction, "max_induced_fraction", positive=True)
        if self.channel_half_width is not None:
            _number(self.channel_half_width, "channel_half_width", positive=True)
        if isinstance(self.channel_panels, bool) or not isinstance(self.channel_panels, int) \
                or self.channel_panels < 8:
            raise ValueError("channel_panels must be an integer >= 8")
        _number(self.channel_length_factor, "channel_length_factor", positive=True)


@torch.no_grad()
def panel_velocity_tensor(points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
                          strengths: torch.Tensor) -> torch.Tensor:
    """Per-panel velocity contributions, shape ``(N, M, 2)``."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape != ends.shape:
        raise ValueError("starts and ends must both have shape (M, 2)")
    if tuple(strengths.shape) != (starts.shape[0],):
        raise ValueError("strengths must have shape (M,)")
    segments = ends - starts
    lengths = torch.linalg.vector_norm(segments, dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("zero-length source panel")
    points = points.to(torch.float64)
    tangents = segments / lengths[:, None]
    base_normals = torch.stack((tangents[:, 1], -tangents[:, 0]), dim=1)
    rel = points[:, None, :] - starts[None, :, :]
    xi = (rel * tangents[None, :, :]).sum(dim=2)
    eta = (rel * base_normals[None, :, :]).sum(dim=2)
    r1sq = (xi * xi + eta * eta).clamp_min(1.0e-24)
    xi2 = xi - lengths[None, :]
    r2sq = (xi2 * xi2 + eta * eta).clamp_min(1.0e-24)
    u_t = strengths[None, :] / (4.0 * math.pi) * torch.log(r1sq / r2sq)
    theta1 = torch.atan2(eta, xi)
    theta2 = torch.atan2(eta, xi2)
    u_n = strengths[None, :] / (2.0 * math.pi) * (theta2 - theta1)
    return u_t[:, :, None] * tangents[None, :, :] + u_n[:, :, None] * base_normals[None, :, :]


@torch.no_grad()
def source_panel_velocity(points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
                          strengths: torch.Tensor) -> torch.Tensor:
    """Velocity induced at ``points`` by straight constant-strength source panels.

    The 2-D influence of a panel of length ``L`` at a field point located at
    ``(xi, eta)`` in the panel frame is

        ``u_t = sigma/(4 pi) ln(r1^2/r2^2)``,  ``u_n = sigma/(2 pi) (theta2 - theta1)``

    with ``r1 = |P - A|``, ``r2 = |P - B|``, ``theta1 = atan2(eta, xi)`` and
    ``theta2 = atan2(eta, xi - L)``.  Field points on a panel take the outward-side
    limit ``u_n = sigma/2``.
    """
    return panel_velocity_tensor(points, starts, ends, strengths).sum(dim=1)


class FormationFlow:
    """Multi-body double-body panel solution for a formation of hulls.

    Panels are tagged by owner: ``owner[i] = k < N`` means a hull panel of ship
    ``k``, ``owner[i] = N`` means a channel-wall panel.
    """

    def __init__(self, waterlines: Sequence[HullWaterline], config: FormationFlowConfig) -> None:
        if not waterlines:
            raise ValueError("at least one hull waterline is required")
        self.waterlines = list(waterlines)
        self.config = config
        self._build()

    # -- geometry --------------------------------------------------------------
    @staticmethod
    def _segments_from_contour(contour: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return contour, torch.roll(contour, shifts=-1, dims=0)

    def _build(self) -> None:
        starts: list[torch.Tensor] = []
        ends: list[torch.Tensor] = []
        frame_normals: list[torch.Tensor] = []
        signs: list[torch.Tensor] = []
        fluid_normals: list[torch.Tensor] = []
        collocation: list[torch.Tensor] = []
        owner: list[torch.Tensor] = []
        for index, hull in enumerate(self.waterlines):
            contour = hull.contour()
            a, b = self._segments_from_contour(contour)
            tangent = (b - a) / torch.linalg.vector_norm(b - a, dim=1)[:, None]
            normal = torch.stack((tangent[:, 1], -tangent[:, 0]), dim=1)
            mid = a + 0.5 * (b - a)
            centroid = contour.mean(dim=0)
            # The fluid is outside the hull: orient the BC normal outwards, and
            # record the sign that maps it onto the (tangent, normal) source frame.
            sign = 1.0 if float(((mid - centroid) * normal).sum()) > 0.0 else -1.0
            starts.append(a)
            ends.append(b)
            fluid_normals.append(sign * normal)
            frame_normals.append(normal)
            signs.append(torch.full((mid.shape[0],), sign, dtype=torch.float64))
            collocation.append(mid)
            owner.append(torch.full((mid.shape[0],), index, dtype=torch.long))
        self._body_panel_count = int(sum(int(c.shape[0]) for c in collocation))
        self.wall_panel_count = 0
        if self.config.channel_half_width is not None:
            half = float(self.config.channel_half_width)
            length_max = max(hull.length for hull in self.waterlines)
            x_min = min(-0.5 * hull.length for hull in self.waterlines) \
                - self.config.channel_length_factor * length_max
            x_max = max(0.5 * hull.length for hull in self.waterlines) \
                + self.config.channel_length_factor * length_max
            n = self.config.channel_panels
            xs = torch.linspace(x_min, x_max, n + 1, dtype=torch.float64)
            dx = float(xs[1] - xs[0])
            a = torch.stack((xs[:-1], torch.full((n,), half, dtype=torch.float64)), dim=1)
            b = torch.stack((xs[:-1] + dx, torch.full((n,), half, dtype=torch.float64)), dim=1)
            a2 = torch.stack((xs[:-1], torch.full((n,), -half, dtype=torch.float64)), dim=1)
            b2 = torch.stack((xs[:-1] + dx, torch.full((n,), -half, dtype=torch.float64)), dim=1)
            starts.append(torch.cat((a, a2)))
            ends.append(torch.cat((b, b2)))
            up = torch.tensor([0.0, -1.0], dtype=torch.float64)
            low = torch.tensor([0.0, 1.0], dtype=torch.float64)
            fluid_normals.append(torch.cat((up.expand(n, 2), low.expand(n, 2))))
            frame_normals.append(torch.stack((torch.zeros(2 * n, dtype=torch.float64),
                                              torch.full((2 * n,), -1.0, dtype=torch.float64)),
                                             dim=1))
            signs.append(torch.cat((torch.ones(n, dtype=torch.float64),
                                    -torch.ones(n, dtype=torch.float64))))
            collocation.append(torch.cat((a + 0.5 * (b - a), a2 + 0.5 * (b2 - a2))))
            self.wall_panel_count = 2 * n
            owner.append(torch.full((2 * n,), len(self.waterlines), dtype=torch.long))
        self.owner = torch.cat(owner)
        self._base_starts = torch.cat(starts)
        self._base_ends = torch.cat(ends)
        self._base_collocation = torch.cat(collocation)
        self._fluid_normals = torch.cat(fluid_normals)
        self._frame_normals = torch.cat(frame_normals)
        self._signs = torch.cat(signs)

    # -- solution --------------------------------------------------------------
    @torch.no_grad()
    def solve(self, offsets: Sequence[tuple[float, float]]) -> dict[str, torch.Tensor]:
        """Solve the formation flow for hull offsets ``(dx, dy)`` in metres.

        ``dx`` is positive downstream, ``dy`` positive to starboard; the lead ship
        sits at the origin.
        """
        offsets = [(float(dx), float(dy)) for dx, dy in offsets]
        if len(offsets) != len(self.waterlines):
            raise ValueError("one offset is required per hull waterline")
        hull_shift = torch.tensor(offsets, dtype=torch.float64)
        shift = torch.cat((hull_shift, torch.zeros((1, 2), dtype=torch.float64)))[self.owner]
        collocation = self._base_collocation + shift
        starts = self._base_starts + shift
        ends = self._base_ends + shift
        normals = self._fluid_normals
        count = starts.shape[0]
        velocity = panel_velocity_tensor(collocation, starts, ends,
                                         torch.ones(count, dtype=torch.float64))
        matrix = torch.einsum("ikc,ic->ik", velocity, normals)
        # Self influence of a constant-strength source panel: outward-side u_n = sigma/2,
        # and the BC normal is oriented towards the fluid, so the diagonal is +0.5.
        matrix.fill_diagonal_(0.5)
        onset = torch.tensor([self.config.speed, 0.0], dtype=torch.float64)
        rhs = -(normals * onset).sum(dim=1)
        strengths = torch.linalg.solve(matrix, rhs)
        residual = float((matrix @ strengths - rhs).abs().max())
        contribution = panel_velocity_tensor(collocation, starts, ends, strengths)
        # A panel's own contribution to the velocity at its collocation point is
        # the source-sheet jump u_n = sigma/2 on the fluid side, i.e. sigma/2 along
        # the outward normal (same convention as the Neumann diagonal).  It is set
        # explicitly because the atan2 form of the u_n integral is ambiguous
        # (sign(when the point lies exactly on the sheet).
        diagonal = torch.arange(count)
        contribution[diagonal, diagonal, :] = (0.5 * strengths[:, None]
                                               * self._fluid_normals)
        return {
            "strengths": strengths,
            "collocation": collocation,
            "starts": starts,
            "ends": ends,
            "normals": normals,
            "panel_velocity": contribution,
            "induced": contribution.sum(dim=1),
            "onset": onset,
            "neumann_residual": residual,
            "normalized_neumann_residual": residual / max(self.config.speed, 1.0e-12),
        }

    @torch.no_grad()
    def interference(self, offsets: Sequence[tuple[float, float]], *,
                     clamp: bool = True) -> dict[str, object]:
        """Pairwise interference-coefficient matrix plus diagnostics."""
        solution = self.solve(offsets)
        panel_velocity = solution["panel_velocity"]
        owner = self.owner
        hull_count = len(self.waterlines)
        speed = self.config.speed
        beta = self.config.resistance_exponent
        wall_selection = owner >= hull_count
        confinement = torch.zeros(hull_count, dtype=torch.float64)
        if bool(wall_selection.any()):
            wall_velocity = panel_velocity[:, wall_selection, :].sum(dim=1)
            for index in range(hull_count):
                selection = owner == index
                confinement[index] = beta * float(wall_velocity[selection, 0].mean()) / speed
        matrix = torch.zeros((hull_count, hull_count), dtype=torch.float64)
        raw = torch.zeros((hull_count, hull_count), dtype=torch.float64)
        capped = False
        limit = self.config.max_induced_fraction
        for target in range(hull_count):
            selection = owner == target
            for source in range(hull_count):
                if source == target:
                    continue
                source_selection = owner == source
                contribution = panel_velocity[:, source_selection, :].sum(dim=1)
                mean_fraction = float(contribution[selection, 0].mean()) / speed
                raw[target, source] = beta * mean_fraction
                if clamp:
                    capped = capped or abs(mean_fraction) > limit
                    mean_fraction = max(-limit, min(limit, mean_fraction))
                matrix[target, source] = beta * mean_fraction
        return {
            "coefficients": matrix,
            "coefficients_unclamped": raw,
            "confinement": confinement,
            "clamped": capped,
            "neumann_residual": float(solution["normalized_neumann_residual"]),
            "strengths": solution["strengths"],
            "owner": owner,
            "hull_panel_count": int(self._body_panel_count),
            "wall_panel_count": int(self.wall_panel_count),
        }

    def interference_tables(self, spacings: Sequence[float],
                            bearings_deg: Sequence[float]) -> list[dict[str, float]]:
        """Tabulate a two-ship coefficient over spacing (in lengths) and bearing.

        ``bearing_deg = 0`` is the column (astern) configuration, 90 deg is abeam.
        """
        rows: list[dict[str, float]] = []
        length = self.waterlines[0].length
        for spacing in spacings:
            spacing = _number(spacing, "spacing", positive=True)
            for bearing in bearings_deg:
                angle = math.radians(_number(bearing, "bearing"))
                dx = spacing * length * math.cos(angle)
                dy = spacing * length * math.sin(angle)
                result = self.interference([(0.0, 0.0), (dx, dy)])
                table = result["coefficients"]
                rows.append({
                    "spacing_over_length": spacing,
                    "bearing_deg": float(bearing),
                    "dx_over_length": dx / length,
                    "dy_over_length": dy / length,
                    "C_leader_from_follower": float(table[0, 1]),
                    "C_follower_from_leader": float(table[1, 0]),
                    "clamped": bool(result["clamped"]),
                })
        return rows


def froude_number(speed: float, length: float) -> float:
    """Length-based Froude number ``U / sqrt(g L)``."""
    speed = _number(speed, "speed", positive=True)
    length = _number(length, "length", positive=True)
    return speed / math.sqrt(9.80665 * length)