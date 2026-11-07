"""Ice-floe DEM dynamics for channel navigation in a broken-ice field.

Minimal usable surface for a broken-ice field in which every floe is a
**flat-lying circular disk rigid body**:

* hydrostatics -- buoyancy ``F_b = rho_w g A_wp d_sub`` (waterplane area times
  submerged thickness, i.e. Archimedes for a flat-lying disk) plus a
  metacentric capsizing (righting) moment ``M = -rho_w g V_sub GM sin(phi)``
  about the tilt degree of freedom;
* contacts -- disk-disk (and hull-disk) Kelvin-Voigt normal spring/dashpot with
  a Coulomb cone on the tangential direction, applied at the contact point so
  that the tangential force also spins the floes;
* environment -- quadratic wind drag on the above-water freeboard and quadratic
  current drag on the submerged side wall.

Orientation convention (see ``docs/ice_channel_dem.md``):
a floe floats **flat**, so the submerged volume is the waterplane area times the
submerged thickness ``d_sub = clamp(t/2 - z, 0, t)``.  The circular-segment
"submerged fraction" kernel of :mod:`tensordem.buoyancy_wet` describes a disk
seen *edge-on* (2-D vertical section) and is therefore **not** used here: for a
flat floe it would give a draught that is wrong by the segment-to-thickness
mapping.  :func:`analytic_equilibrium_draught` returns ``(rho_i/rho_w) t``.

Contact stiffness anchor (self-built; declared in ``artifacts/dem_channel.json``):
``k_n = 2 sigma_c t_eff sqrt(2 R_eff / delta_ref)`` with
``delta_ref = reference_penetration_fraction * 2 R_eff``.  With the nominal
line-contact area ``A_co = 2 t_eff sqrt(2 R_eff delta)`` the nominal contact
pressure becomes exactly ``sigma_c`` at the reference penetration.
``sigma_c = 1.5 MPa`` is the Keinonen / ISO-19906 order local ice pressure for
ship-ice contact.  A repository anchor table ``benchmarks/ice_nav_refs`` is not
present on ``main`` when this module was written, so the anchor is derived here
from the closed form plus the citation and re-verified numerically by
:func:`contact_pressure_at`.

All state is ``float64`` torch tensors and every reduction is deterministic, so
two runs with the same seed are bit-identical (verified through
:meth:`IceChannelSimulation.state_digest`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "WaterEnvironment",
    "ContactParameters",
    "HullGeometry",
    "IceChannelSimulation",
    "analytic_equilibrium_draught",
    "submerged_draught",
    "buoyancy_force",
    "heave_stiffness",
    "analytic_heave_period",
    "heave_potential",
    "metacentric_height",
    "capsize_restoring_moment",
    "contact_stiffness",
    "contact_pressure_at",
    "nominal_contact_area",
    "wind_drag_force",
    "current_drag_force",
]

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value


@dataclass(frozen=True)
class WaterEnvironment:
    """Hydrostatic / hydrodynamic environment of the ice field."""

    water_density: float = 1025.0          # kg/m3, sea water
    ice_density: float = 917.0             # kg/m3, columnar sea ice
    gravity: float = 9.80665               # m/s2
    air_density: float = 1.225             # kg/m3
    wind_velocity: Tuple[float, float] = (8.0, 0.0)     # m/s
    current_velocity: Tuple[float, float] = (0.0, 0.0)  # m/s
    wind_drag_coefficient: float = 1.2
    water_drag_coefficient: float = 1.0
    sail_area_fraction: float = 0.15       # above-water share of the side wall
    heave_damping_ratio: float = 0.2
    tilt_damping_ratio: float = 0.1

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        for name in ("water_density", "ice_density", "gravity", "air_density",
                     "wind_drag_coefficient", "water_drag_coefficient"):
            _require_positive(name, getattr(self, name))
        if not self.ice_density < self.water_density:
            raise ValueError(
                "ice_density must be smaller than water_density for a floating floe")
        if not 0.0 < self.sail_area_fraction <= 1.0:
            raise ValueError("sail_area_fraction must lie in (0, 1]")
        for name in ("heave_damping_ratio", "tilt_damping_ratio"):
            value = float(getattr(self, name))
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in ("wind_velocity", "current_velocity"):
            vec = getattr(self, name)
            if len(vec) != 2:
                raise ValueError(f"{name} must be a 2-vector")

    @property
    def density_ratio(self) -> float:
        return self.ice_density / self.water_density


@dataclass(frozen=True)
class ContactParameters:
    """Contact law parameters, including the self-built stiffness anchor."""

    crushing_strength: float = 1.5e6       # Pa, Keinonen local ice pressure
    reference_penetration_fraction: float = 0.01   # fraction of 2*R_eff
    normal_damping_ratio: float = 0.2
    tangential_damping_ratio: float = 0.2
    friction: float = 0.3

    def __post_init__(self) -> None:
        _require_positive("crushing_strength", self.crushing_strength)
        frac = float(self.reference_penetration_fraction)
        if not 0.0 < frac < 0.5:
            raise ValueError("reference_penetration_fraction must lie in (0, 0.5)")
        for name in ("normal_damping_ratio", "tangential_damping_ratio"):
            value = float(getattr(self, name))
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must lie in [0, 1)")
        if not (0.0 <= float(self.friction) < 1.5):
            raise ValueError("friction must lie in [0, 1.5)")


@dataclass(frozen=True)
class HullGeometry:
    """Prescribed equal-width (rectangular) hull used for the channel transit."""

    length: float = 80.0
    beam: float = 20.0
    speed: float = 1.6
    y_position: float = 0.0
    x_start: float = 0.0

    def __post_init__(self) -> None:
        _require_positive("length", self.length)
        _require_positive("beam", self.beam)
        if not math.isfinite(float(self.speed)):
            raise ValueError("speed must be finite")

    @property
    def half_beam(self) -> float:
        return 0.5 * float(self.beam)

    @property
    def half_length(self) -> float:
        return 0.5 * float(self.length)

    def x_centre(self, time: float) -> float:
        return float(self.x_start) + float(self.speed) * float(time)

    def velocity(self) -> torch.Tensor:
        return torch.tensor([float(self.speed), 0.0], dtype=_DTYPE)


# ---------------------------------------------------------------------------
# Hydrostatics of a flat-lying circular disk
# ---------------------------------------------------------------------------

def submerged_draught(centre_height, thickness):
    """Submerged thickness ``d = clamp(t/2 - z, 0, t)`` [m] for a flat disk.

    ``z`` is the disk centre height above the still-water surface (positive up).
    Accepts python floats or tensors; returns the same kind.
    """
    if isinstance(centre_height, torch.Tensor):
        t = torch.as_tensor(thickness, dtype=centre_height.dtype,
                            device=centre_height.device)
        half = 0.5 * t
        return torch.clamp(half - centre_height, min=torch.zeros_like(half), max=t)
    thick = _require_positive("thickness", thickness)
    return min(max(0.5 * thick - float(centre_height), 0.0), thick)


def analytic_equilibrium_draught(rho_ice: float, rho_water: float,
                                 thickness: float) -> float:
    """Archimedes draught of a flat-lying disk: ``d_eq = (rho_i/rho_w) * t``."""
    thick = _require_positive("thickness", thickness)
    rho_i = _require_positive("rho_ice", rho_ice)
    rho_w = _require_positive("rho_water", rho_water)
    if not rho_i < rho_w:
        raise ValueError("floe cannot float: rho_ice >= rho_water")
    return (rho_i / rho_w) * thick


def waterplane_area(radius: float) -> float:
    _require_positive("radius", radius)
    return math.pi * float(radius) ** 2


def buoyancy_force(centre_height, radius: float, thickness: float,
                   rho_water: float, gravity: float):
    """Archimedes force ``rho_w g A_wp d_sub`` (positive upward) [N]."""
    area = waterplane_area(radius)
    _require_positive("rho_water", rho_water)
    _require_positive("gravity", gravity)
    return rho_water * gravity * area * submerged_draught(centre_height, thickness)


def heave_stiffness(radius: float, rho_water: float, gravity: float) -> float:
    """Heave restoring stiffness of a flat disk over a free surface [N/m].

    ``K = rho_w g A_wp`` while the floe is partially submerged (constant).
    """
    return _require_positive("rho_water", rho_water) * _require_positive(
        "gravity", gravity) * waterplane_area(radius)


def analytic_heave_period(radius: float, thickness: float, rho_ice: float,
                          rho_water: float, gravity: float) -> float:
    """Linearised heave period ``T = 2 pi sqrt(m / K)`` [s]."""
    mass = floe_mass(radius, thickness, rho_ice)
    return 2.0 * math.pi * math.sqrt(mass / heave_stiffness(radius, rho_water, gravity))


def floe_mass(radius: float, thickness: float, rho_ice: float) -> float:
    return _require_positive("rho_ice", rho_ice) * waterplane_area(radius) * _require_positive(
        "thickness", thickness)


def heave_potential(centre_height, radius: float, thickness: float,
                    rho_ice: float, rho_water: float, gravity: float):
    """Potential energy of the heave mode [J], referenced to the disk bottom.

    ``U(z) = m g z + U_b(z)`` with ``U_b = -int F_b dz``; ``dU/dz = -F_net`` so
    the equilibrium sits at the Archimedes draught and the total energy
    ``E = 0.5 m vz^2 + U`` is conserved by a symplectic integrator.
    """
    thick = float(thickness)
    rho_i = _require_positive("rho_ice", rho_ice)
    rho_w = _require_positive("rho_water", rho_water)
    g = _require_positive("gravity", gravity)
    area = waterplane_area(radius)
    k = rho_w * g * area
    mass = rho_i * area * thick
    tensor = isinstance(centre_height, torch.Tensor)
    z = centre_height if tensor else float(centre_height)
    half = 0.5 * thick
    zero = torch.zeros_like(z) if tensor else 0.0
    # U_b(z) = -k * int_{-t/2}^{z} (t/2 - z') dz'
    # ``integral = int_{-t/2}^{z} d_sub dz'``; below the bottom the submerged
    # thickness stays at ``t`` (constant buoyancy), which keeps U_b smooth so a
    # symplectic integrator conserves the total energy there as well.
    if tensor:
        zc = torch.clamp(z, min=-half, max=half)
        partial = half * zc - 0.5 * zc * zc + 3.0 * thick * thick / 8.0
        integral = torch.where(z > half, torch.full_like(z, thick * thick / 2.0),
                               torch.where(z < -half, thick * (z + half), partial))
        ub = -k * integral
        return mass * g * z + ub
    if z <= -half:
        integral = thick * (z + half)
    elif z >= half:
        integral = thick * thick / 2.0
    else:
        integral = half * z - 0.5 * z * z + 3.0 * thick * thick / 8.0
    return mass * g * z - k * integral


def centre_of_buoyancy_height(centre_height, thickness):
    """Height of the centre of buoyancy above the still-water surface [m].

    Uniform box section: the submerged part spans ``[z - t/2, 0]`` so its
    centroid sits at ``(z - t/2)/2``.
    """
    if isinstance(centre_height, torch.Tensor):
        thick = torch.as_tensor(thickness, dtype=centre_height.dtype,
                                device=centre_height.device)
        return 0.5 * (centre_height - 0.5 * thick)
    return 0.5 * (float(centre_height) - 0.5 * float(thickness))


def metacentric_height(centre_height, radius: float, thickness: float,
                       rho_ice: float, rho_water: float) -> float:
    """Metacentric height ``GM = BM - BG`` [m] of a flat-lying disk.

    ``BM = I_wl / V_sub`` with ``I_wl = pi R^4 / 4`` (full waterplane circle),
    ``BG`` = vertical distance from the centre of buoyancy up to the centre of
    gravity (the disk centre).  For a thin floe ``BM`` is O(R^2/(4t)) ~ metres,
    so a flat floe is very stable and the tilt mode is stiff -- which is the
    physical reason a level-ice floe does not capsize in a channel.
    """
    thick = _require_positive("thickness", thickness)
    area = waterplane_area(radius)
    d_sub = submerged_draught(centre_height, thick)
    if isinstance(d_sub, torch.Tensor):
        if bool((d_sub <= 0).any()):
            raise ValueError("floe is dry: no submerged volume, GM undefined")
        volume = area * d_sub
        i_wl = math.pi * float(radius) ** 4 / 4.0
        z_cb = centre_of_buoyancy_height(centre_height, thick)
        z = centre_height
        return i_wl / volume - (z - z_cb)
    if d_sub <= 0.0:
        raise ValueError("floe is dry: no submerged volume, GM undefined")
    volume = area * d_sub
    i_wl = math.pi * float(radius) ** 4 / 4.0
    z_cb = centre_of_buoyancy_height(centre_height, thick)
    return i_wl / volume - (float(centre_height) - z_cb)


def capsize_restoring_moment(tilt, centre_height, radius: float, thickness: float,
                             rho_ice: float, rho_water: float, gravity: float):
    """Righting moment ``M = -rho_w g V_sub GM sin(phi)`` [N m].

    Positive tilt is a rotation about a horizontal axis; the moment opposes it.
    Small-angle form would be ``-K_phi phi`` with ``K_phi = rho_w g V_sub GM``.
    """
    d_sub = submerged_draught(centre_height, thickness)
    gm = metacentric_height(centre_height, radius, thickness, rho_ice, rho_water)
    stiffness = rho_water * gravity * waterplane_area(radius) * d_sub * gm
    if isinstance(tilt, torch.Tensor):
        return -stiffness * torch.sin(tilt)
    return -float(stiffness) * math.sin(float(tilt))


# ---------------------------------------------------------------------------
# Contact law and its anchor
# ---------------------------------------------------------------------------

def nominal_contact_area(penetration: float, r_eff: float, t_eff: float) -> float:
    """Nominal line-contact area of two overlapping disks ``2 t sqrt(2 R d)``.

    This is the chord width of the lens-shaped overlap of two circles of radius
    ``R`` at relative approach ``d`` (plane-strain ice contact over the floe
    thickness), i.e. the area that carries the crushing pressure.
    """
    pen = float(penetration)
    if pen <= 0.0:
        return 0.0
    return 2.0 * _require_positive("t_eff", t_eff) * math.sqrt(2.0 * _require_positive(
        "r_eff", r_eff) * pen)


def contact_stiffness(r_eff: float, t_eff: float,
                      params: ContactParameters = ContactParameters()) -> float:
    """Normal contact stiffness from the Keinonen local-pressure anchor [N/m]."""
    r_eff = _require_positive("r_eff", r_eff)
    t_eff = _require_positive("t_eff", t_eff)
    delta_ref = params.reference_penetration_fraction * 2.0 * r_eff
    return 2.0 * params.crushing_strength * t_eff * math.sqrt(2.0 * r_eff / delta_ref)


def contact_pressure_at(penetration: float, r_eff: float, t_eff: float,
                        params: ContactParameters = ContactParameters()) -> float:
    """Nominal contact pressure ``F_n / A_co`` [Pa] under the anchored spring."""
    k = contact_stiffness(r_eff, t_eff, params)
    force = k * float(penetration)
    area = nominal_contact_area(penetration, r_eff, t_eff)
    if area <= 0.0:
        return 0.0
    return force / area


# ---------------------------------------------------------------------------
# Environmental drag
# ---------------------------------------------------------------------------

def wind_drag_force(velocity, radius: float, freeboard: float,
                    env: WaterEnvironment = WaterEnvironment()):
    """Quadratic wind drag on the above-water side wall [N, 2 components].

    ``area = 2 R * freeboard * sail_area_fraction``; the sail fraction accounts
    for the fact that only part of the side wall is exposed above the water.
    """
    area = 2.0 * float(radius) * max(float(freeboard), 0.0) * env.sail_area_fraction
    return _quadratic_drag(velocity, env.wind_velocity, env.air_density,
                           env.wind_drag_coefficient, area)


def current_drag_force(velocity, radius: float, draught: float,
                       env: WaterEnvironment = WaterEnvironment()):
    """Quadratic current (water) drag on the submerged side wall [N, 2]."""
    area = 2.0 * float(radius) * max(float(draught), 0.0)
    return _quadratic_drag(velocity, env.current_velocity, env.water_density,
                           env.water_drag_coefficient, area)


def _quadratic_drag(velocity, ambient: Sequence[float], density: float,
                    coefficient: float, area: float):
    v = torch.as_tensor(velocity, dtype=_DTYPE) if not torch.is_tensor(velocity) else velocity
    ambient_vec = torch.tensor(list(ambient), dtype=v.dtype, device=v.device)
    rel = v - ambient_vec
    speed = torch.linalg.norm(rel, dim=-1, keepdim=True)
    return -0.5 * density * coefficient * area * speed * rel


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class IceChannelSimulation:
    """Disk-floe DEM in the horizontal plane with heave, tilt and yaw modes.

    State (all shape ``(N,)`` except ``pos``/``vel`` which are ``(N, 2)``):
    ``pos``, ``vel``, ``yaw``, ``omega``, ``z``, ``vz``, ``tilt``, ``tilt_rate``.
    """

    def __init__(self, radius: Sequence[float], thickness: Sequence[float], *,
                 water: Optional[WaterEnvironment] = None,
                 contact: Optional[ContactParameters] = None,
                 hull: Optional[HullGeometry] = None,
                 dt: float = 5.0e-3,
                 integrator: str = "euler",
                 dtype: torch.dtype = _DTYPE) -> None:
        radius_t = torch.as_tensor(list(radius), dtype=dtype)
        thickness_t = torch.as_tensor(list(thickness), dtype=dtype)
        if radius_t.ndim != 1 or thickness_t.ndim != 1:
            raise ValueError("radius and thickness must be 1-D sequences")
        if radius_t.numel() == 0:
            raise ValueError("at least one floe is required")
        if radius_t.numel() != thickness_t.numel():
            raise ValueError("radius and thickness must have the same length")
        for name, values in (("radius", radius_t), ("thickness", thickness_t)):
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} must be finite")
            if (values <= 0).any():
                bad = float(values.min())
                raise ValueError(f"{name} must be strictly positive, got {bad!r} "
                                 "(zero-size particles are rejected)")
        self.water = water or WaterEnvironment()
        self.contact = contact or ContactParameters()
        self.hull = hull or HullGeometry()
        self.dt = _require_positive("dt", dt)
        if integrator not in ("euler", "verlet"):
            raise ValueError("integrator must be 'euler' or 'verlet'")
        self.integrator = integrator
        self.dtype = dtype

        self.radius = radius_t
        self.thickness = thickness_t
        self.n = int(radius_t.numel())
        area = math.pi * radius_t ** 2
        self.mass = self.water.ice_density * area * thickness_t
        self.inertia_yaw = 0.5 * self.mass * radius_t ** 2
        self.inertia_tilt = 0.25 * self.mass * radius_t ** 2
        self.waterplane_area = area

        self.pos = torch.zeros((self.n, 2), dtype=dtype)
        self.vel = torch.zeros((self.n, 2), dtype=dtype)
        self.yaw = torch.zeros(self.n, dtype=dtype)
        self.omega = torch.zeros(self.n, dtype=dtype)
        self.z = torch.zeros(self.n, dtype=dtype)
        self.vz = torch.zeros(self.n, dtype=dtype)
        self.tilt = torch.zeros(self.n, dtype=dtype)
        self.tilt_rate = torch.zeros(self.n, dtype=dtype)
        self.time = 0.0
        self.steps = 0

    # -- field construction -------------------------------------------------
    @classmethod
    def from_field(cls, concentration: float, *, domain_length: float,
                   domain_width: float, seed: int,
                   mean_radius: float = 5.0,
                   radius_jitter: float = 0.08,
                   radius_range: Tuple[float, float] = (2.0, 8.0),
                   thickness_range: Tuple[float, float] = (0.4, 1.0),
                   **kwargs) -> "IceChannelSimulation":
        """Build a floe field of prescribed ice concentration (tenths of area).

        ``concentration`` is the fraction of the domain area covered by floe
        material, i.e. 0.6/0.8 for 6/10, 8/10 ice.  A random sequential
        placement jams near 0.55 coverage and therefore cannot represent 6-8/10
        fields, so the field is a **jittered hexagonal tessellation**: the cell
        size follows from the requested coverage and mean floe radius, each
        floe radius is the circle of equal area to its cell times a seeded
        ``radius_jitter``, and the radii are rescaled once so that the realised
        coverage equals the target.  Consequently the initial field has (i) the
        requested concentration to <1 % and (ii) no initial disk overlap
        (``max_initial_overlap`` ~ 0), so no spurious contact impulse at ``t=0``.
        """
        conc = float(concentration)
        if not math.isfinite(conc) or not (0.0 <= conc <= 1.0):
            raise ValueError(
                f"concentration must lie in [0, 1] (got {concentration!r}); "
                "negative or over-unity ice coverage is rejected")
        length = _require_positive("domain_length", domain_length)
        width = _require_positive("domain_width", domain_width)
        r_mean = _require_positive("mean_radius", mean_radius)
        jitter = float(radius_jitter)
        if not (0.0 <= jitter < 0.5):
            raise ValueError("radius_jitter must lie in [0, 0.5)")
        r_lo, r_hi = float(radius_range[0]), float(radius_range[1])
        if not (0.0 < r_lo < r_hi):
            raise ValueError("radius_range must satisfy 0 < lo < hi")
        t_lo, t_hi = float(thickness_range[0]), float(thickness_range[1])
        if not (0.0 < t_lo <= t_hi):
            raise ValueError("thickness_range must satisfy 0 < lo <= hi")

        generator = torch.Generator().manual_seed(int(seed))
        area = length * width
        # Triangular lattice: rows separated by sqrt(3)/2 of the column spacing so
        # that every nearest-neighbour pair is s_x apart.  The floe radius then
        # follows from the requested coverage and the *actual* lattice cell, which
        # keeps the realised concentration equal to the requested tenths and keeps
        # 2r < s_x (no initial overlap) up to the hexagonal limit 0.907.
        cell_area = math.pi * r_mean * r_mean / conc
        s_lat = math.sqrt(2.0 * cell_area / math.sqrt(3.0))
        cols = max(1, int(round(length / s_lat)))
        s_x = length / cols
        rows = max(1, int(math.ceil(width / (0.5 * math.sqrt(3.0) * s_x))))
        s_y = width / rows
        points = []
        for row in range(rows):
            y = (row + 0.5) * s_y
            offset = 0.5 * s_x * (row % 2)
            n_col = cols + (row % 2)
            for col in range(n_col):
                x = offset + (col + 0.5) * s_x
                if x < length:
                    points.append((x, y))
        count = len(points)
        cell = s_x * s_y
        nn = min(s_x, math.hypot(0.5 * s_x, s_y))
        cap = 0.49 * nn
        base = math.sqrt(conc * cell / math.pi)
        jit_r = 1.0 + jitter * (torch.rand(count, generator=generator, dtype=_DTYPE) * 2.0 - 1.0)
        radii_t = (base * jit_r).clamp(min=r_lo, max=min(r_hi, cap))
        for _ in range(4):
            realised = float((math.pi * radii_t ** 2).sum()) / area
            if realised <= 0.0:
                break
            radii_t = (radii_t * math.sqrt(conc / realised)).clamp(min=r_lo, max=min(r_hi, cap))
        strata_t = (torch.arange(count, dtype=_DTYPE) + 0.5) / count
        strata_t = strata_t[torch.randperm(count, generator=generator)]
        thick_t = t_lo + (t_hi - t_lo) * strata_t
        pos_t = torch.tensor(points, dtype=_DTYPE)
        jitter_xy = (torch.rand(count, 2, generator=generator, dtype=_DTYPE) - 0.5) \
            * 2.0 * (0.02 * nn)
        pos_t = pos_t + jitter_xy
        pos_t[:, 0] = torch.clamp(pos_t[:, 0], min=radii_t, max=length - radii_t)
        pos_t[:, 1] = torch.clamp(pos_t[:, 1], min=radii_t, max=width - radii_t)
        # deterministic de-overlap projection (positions only; radii untouched)
        for _ in range(400):
            delta = pos_t[:, None, :] - pos_t[None, :, :]
            dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
            overlap = (radii_t[:, None] + radii_t[None, :]) - dist
            overlap = overlap - torch.diag_embed(torch.diagonal(overlap))
            if float(overlap.max()) <= 1e-6:
                break
            active = overlap > 0.0
            normal = delta / dist[:, :, None]
            push = torch.where(active[:, :, None], normal * (0.6 * overlap)[:, :, None],
                               torch.zeros_like(delta))
            pos_t = pos_t + push.sum(dim=1)
            pos_t[:, 0] = torch.clamp(pos_t[:, 0], min=radii_t, max=length - radii_t)
            pos_t[:, 1] = torch.clamp(pos_t[:, 1], min=radii_t, max=width - radii_t)

        sim = cls([float(v) for v in radii_t], [float(v) for v in thick_t], **kwargs)
        sim.pos = pos_t
        # start every floe at its Archimedes draught: no startup heave transient
        ratio = sim.water.ice_density / sim.water.water_density
        sim.z = 0.5 * sim.thickness - ratio * sim.thickness
        sim.domain = (length, width)
        sim.target_concentration = conc
        sim.seed = int(seed)
        return sim

    # -- diagnostics --------------------------------------------------------
    def field_statistics(self) -> Dict[str, float]:
        length, width = getattr(self, "domain", (float("nan"), float("nan")))
        covered = float((math.pi * self.radius ** 2).sum())
        target = getattr(self, "target_concentration", float("nan"))
        return {
            "floe_count": int(self.n),
            "target_concentration": float(target),
            "realised_concentration": covered / (length * width),
            "mean_radius": float(self.radius.mean()),
            "radius_min": float(self.radius.min()),
            "radius_max": float(self.radius.max()),
            "mean_thickness": float(self.thickness.mean()),
            "total_floe_mass": float(self.mass.sum()),
            "max_initial_overlap": self.initial_overlap,
        }

    @property
    def initial_overlap(self) -> float:
        """Largest remaining disk overlap of the initial placement [m]."""
        if self.n < 2:
            return 0.0
        delta = self.pos[:, None, :] - self.pos[None, :, :]
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
        r_sum = self.radius[:, None] + self.radius[None, :]
        overlap = r_sum - dist
        overlap = overlap - torch.diag_embed(torch.diagonal(overlap))
        return float(overlap.max().clamp(min=0.0))

    def heave_energy(self) -> torch.Tensor:
        """Per-floe conserved energy of the heave mode [J] (no contacts/drag)."""
        water = self.water
        potential = torch.stack([
            heave_potential(self.z[i], float(self.radius[i]), float(self.thickness[i]),
                            water.ice_density, water.water_density, water.gravity)
            for i in range(self.n)])
        return 0.5 * self.mass * self.vz ** 2 + potential

    def kinetic_energy(self) -> torch.Tensor:
        return 0.5 * self.mass * (self.vel ** 2).sum(dim=1) \
            + 0.5 * self.inertia_yaw * self.omega ** 2

    def state_digest(self) -> str:
        """SHA-256 over every state component: bit-identical runs share it."""
        hasher = hashlib.sha256()
        for tensor in (self.pos, self.vel, self.yaw, self.omega, self.z, self.vz,
                       self.tilt, self.tilt_rate):
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hasher.update(str(self.steps).encode())
        return hasher.hexdigest()

    def velocity_statistics(self) -> Dict[str, float]:
        speed = torch.linalg.norm(self.vel, dim=1)
        if self.n == 1:
            return {"speed_mean": float(speed[0]), "speed_rms": float(speed[0]),
                    "speed_p90": float(speed[0]), "speed_max": float(speed[0]),
                    "moving_fraction": float(speed[0] > 0.05)}
        return {
            "speed_mean": float(speed.mean()),
            "speed_rms": float(torch.sqrt((speed ** 2).mean())),
            "speed_p90": float(torch.quantile(speed, 0.9)),
            "speed_max": float(speed.max()),
            "moving_fraction": float((speed > 0.05).to(_DTYPE).mean()),
            "vx_mean": float(self.vel[:, 0].mean()),
            "vy_mean": float(self.vel[:, 1].mean()),
            "vy_abs_mean": float(self.vel[:, 1].abs().mean()),
        }

    def cleared_width(self, behind_length: Optional[float] = None,
                      lateral_limit: Optional[float] = None) -> float:
        """Cleared channel width behind the hull [m].

        Definition: over the swept corridor ``x in [x_ship - behind_length,
        x_ship]`` the local lateral clearance of every floe is
        ``|y_i - y_ship| - R_i``; the cleared *half* width is the minimum of
        those clearances (clipped at zero), and the reported width is twice it,
        clipped at the domain width and at ``2*(lateral_limit + ...)``.  It is
        therefore the widest symmetric band around the track that contains no
        floe material.
        """
        x_ship = self.hull.x_centre(self.time)
        behind = float(behind_length if behind_length is not None
                       else self.hull.length)
        y_ship = float(self.hull.y_position)
        in_corridor = (self.pos[:, 0] >= x_ship - behind) & (self.pos[:, 0] <= x_ship)
        if not bool(in_corridor.any()):
            length, width = getattr(self, "domain", (float("nan"), 2.0 * float(
                lateral_limit if lateral_limit is not None else self.hull.beam)))
            return float(width)
        clearance = (self.pos[in_corridor, 1] - y_ship).abs() - self.radius[in_corridor]
        half = float(clearance.min().clamp(min=0.0))
        width = 2.0 * half
        if lateral_limit is not None:
            width = min(width, 2.0 * float(lateral_limit))
        return width

    # -- force assembly -----------------------------------------------------
    def _floe_floe_contacts(self):
        pos, vel = self.pos, self.vel
        n = self.n
        if n < 2:
            empty = torch.zeros(0, dtype=self.dtype)
            return (empty, empty, empty, empty, empty)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
        r_sum = self.radius[:, None] + self.radius[None, :]
        overlap = r_sum - dist
        eye = torch.eye(n, dtype=torch.bool)
        # unique pairs only (i < j), otherwise every contact is applied twice
        upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
        active = (overlap > 0.0) & (~eye) & upper
        idx = active.nonzero(as_tuple=False)
        if idx.numel() == 0:
            empty = torch.zeros(0, dtype=self.dtype)
            return (empty, empty, empty, empty, empty)
        i, j = idx[:, 0], idx[:, 1]
        d = delta[i, j]
        dist_ij = dist[i, j]
        normal = d / dist_ij[:, None]
        pen = overlap[i, j]
        r_i = self.radius[i]
        r_j = self.radius[j]
        r_eff = (r_i * r_j) / (r_i + r_j)
        t_eff = torch.minimum(self.thickness[i], self.thickness[j])
        k_n = 2.0 * self.contact.crushing_strength * t_eff * torch.sqrt(
            2.0 * r_eff / (self.contact.reference_penetration_fraction * 2.0 * r_eff))
        m_eff = (self.mass[i] * self.mass[j]) / (self.mass[i] + self.mass[j])
        c_n = 2.0 * self.contact.normal_damping_ratio * torch.sqrt(k_n * m_eff)
        arm_i = -(r_i - 0.5 * pen)
        arm_j = (r_j - 0.5 * pen)
        # contact point velocity including spin: v + omega x r, omega = (0,0,w)
        v_i = self.vel[i] + self.omega[i][:, None] * torch.stack(
            (-normal[:, 1], normal[:, 0]), dim=1) * arm_i[:, None]
        v_j = self.vel[j] + self.omega[j][:, None] * torch.stack(
            (-normal[:, 1], normal[:, 0]), dim=1) * arm_j[:, None]
        v_rel = v_i - v_j
        v_n = (v_rel * normal).sum(dim=1)
        f_mag = torch.clamp(k_n * pen - c_n * v_n, min=torch.zeros_like(pen))
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
        v_t = (v_rel * tangent).sum(dim=1)
        c_t = 2.0 * self.contact.tangential_damping_ratio * torch.sqrt(k_n * m_eff)
        f_t = torch.clamp(c_t * v_t, min=-self.contact.friction * f_mag,
                          max=self.contact.friction * f_mag)
        f_pair = normal * f_mag[:, None] + tangent * (-f_t)[:, None]
        return idx, f_pair, (normal * arm_i[:, None]), (normal * arm_j[:, None]), f_pair

    def _accumulate_contacts(self, force, torque):
        if self.n < 2:
            return
        idx, f_pair, arm_pt_i, arm_pt_j, _ = self._floe_floe_contacts()
        if idx.numel() == 0:
            return
        i, j = idx[:, 0], idx[:, 1]
        force.index_add_(0, i, f_pair)
        force.index_add_(0, j, -f_pair)
        torque.index_add_(0, i, arm_pt_i[:, 0] * f_pair[:, 1] - arm_pt_i[:, 1] * f_pair[:, 0])
        torque.index_add_(0, j, arm_pt_j[:, 0] * (-f_pair[:, 1]) - arm_pt_j[:, 1] * (-f_pair[:, 0]))

    def _hull_force(self, t: float, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Penalty contact of every floe with the prescribed rectangular hull."""
        hull = self.hull
        x_c = hull.x_centre(t)
        y_c = float(hull.y_position)
        half_l, half_b = hull.half_length, hull.half_beam
        local = pos - torch.tensor([x_c, y_c], dtype=self.dtype)
        closest = torch.stack((
            torch.clamp(local[:, 0], -half_l, half_l),
            torch.clamp(local[:, 1], -half_b, half_b)), dim=1)
        offset = local - closest
        dist = torch.linalg.norm(offset, dim=1)
        inside = dist < 1e-12
        normal = torch.where(inside[:, None],
                             torch.zeros_like(offset), offset / torch.clamp(
                                 dist, min=1e-30)[:, None])
        dx = half_l - local[:, 0].abs()
        dy = half_b - local[:, 1].abs()
        use_x = dx < dy
        sx = torch.sign(local[:, 0])
        sy = torch.sign(local[:, 1])
        sx = torch.where(sx == 0, torch.ones_like(sx), sx)
        sy = torch.where(sy == 0, torch.ones_like(sy), sy)
        push_x = torch.stack((sx, torch.zeros_like(sx)), dim=1)
        push_y = torch.stack((torch.zeros_like(sy), sy), dim=1)
        push = torch.where(use_x[:, None], push_x, push_y)
        normal = torch.where(inside[:, None], push, normal)
        pen_inside = torch.where(use_x, dx, dy) + self.radius
        pen = torch.where(inside, pen_inside, self.radius - dist)
        pen = torch.clamp(pen, min=0.0)
        active = pen > 0.0
        r_eff = self.radius
        t_eff = self.thickness
        k_n = 2.0 * self.contact.crushing_strength * t_eff * torch.sqrt(
            2.0 * r_eff / (self.contact.reference_penetration_fraction * 2.0 * r_eff))
        m_eff = self.mass
        c_n = 2.0 * self.contact.normal_damping_ratio * torch.sqrt(k_n * m_eff)
        v_rel = self.vel - hull.velocity()
        v_n = (v_rel * normal).sum(dim=1)
        f_mag = torch.clamp(k_n * pen - c_n * v_n, min=0.0)
        f_mag = torch.where(active, f_mag, torch.zeros_like(f_mag))
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
        v_t = (v_rel * tangent).sum(dim=1)
        c_t = 2.0 * self.contact.tangential_damping_ratio * torch.sqrt(k_n * m_eff)
        f_t = torch.clamp(c_t * v_t, min=-self.contact.friction * f_mag,
                          max=self.contact.friction * f_mag)
        force = normal * f_mag[:, None] + tangent * (-f_t)[:, None]
        arm = -(self.radius - 0.5 * pen)
        torque = arm * (normal[:, 0] * force[:, 1] - normal[:, 1] * force[:, 0])
        return force, torque

    def net_forces(self, t: float):
        """Return (planar force, yaw torque, vertical force) at time ``t``."""
        water = self.water
        force = torch.zeros_like(self.pos)
        torque = torch.zeros(self.n, dtype=self.dtype)
        self._accumulate_contacts(force, torque)
        hull_force, hull_torque = self._hull_force(t, self.pos)
        force = force + hull_force
        torque = torque + hull_torque

        draught = submerged_draught(self.z, self.thickness)
        freeboard = self.thickness - draught
        sail = 2.0 * self.radius * torch.clamp(freeboard, min=0.0) * water.sail_area_fraction
        side = 2.0 * self.radius * draught
        wind_rel = self.vel - torch.tensor(list(water.wind_velocity), dtype=self.dtype)
        cur_rel = self.vel - torch.tensor(list(water.current_velocity), dtype=self.dtype)
        wind_speed = torch.linalg.norm(wind_rel, dim=1, keepdim=True)
        cur_speed = torch.linalg.norm(cur_rel, dim=1, keepdim=True)
        force = force - 0.5 * water.air_density * water.wind_drag_coefficient * sail[:, None] \
            * wind_speed * wind_rel
        force = force - 0.5 * water.water_density * water.water_drag_coefficient * side[:, None] \
            * cur_speed * cur_rel

        upward = water.water_density * water.gravity * self.waterplane_area * draught
        vertical = upward - self.mass * water.gravity
        # heave radiation damping: c = 2 zeta sqrt(m K), K = rho_w g A_wp
        if water.heave_damping_ratio > 0.0:
            k_heave = water.water_density * water.gravity * self.waterplane_area
            c_heave = 2.0 * water.heave_damping_ratio * torch.sqrt(self.mass * k_heave)
            vertical = vertical - c_heave * self.vz
        return force, torque, vertical

    # -- integration --------------------------------------------------------
    def step(self) -> None:
        dt = self.dt
        if self.integrator == "verlet":
            force, torque, vertical = self.net_forces(self.time)
            acc = force / self.mass[:, None]
            acc_yaw = torque / self.inertia_yaw
            acc_z = vertical / self.mass
            self.vel = self.vel + 0.5 * dt * acc
            self.omega = self.omega + 0.5 * dt * acc_yaw
            self.vz = self.vz + 0.5 * dt * acc_z
            self.pos = self.pos + dt * self.vel
            self.yaw = self.yaw + dt * self.omega
            self.z = self.z + dt * self.vz
            self.time += dt
            force, torque, vertical = self.net_forces(self.time)
            self.vel = self.vel + 0.5 * dt * force / self.mass[:, None]
            self.omega = self.omega + 0.5 * dt * torque / self.inertia_yaw
            self.vz = self.vz + 0.5 * dt * vertical / self.mass
        else:
            force, torque, vertical = self.net_forces(self.time)
            self.vel = self.vel + dt * force / self.mass[:, None]
            self.omega = self.omega + dt * torque / self.inertia_yaw
            self.vz = self.vz + dt * vertical / self.mass
            self.pos = self.pos + dt * self.vel
            self.yaw = self.yaw + dt * self.omega
            self.z = self.z + dt * self.vz
            self.time += dt

        # tilt mode (hydrostatic righting + optional damping), guarded so a dry
        # floe cannot produce a NaN GM
        draught = submerged_draught(self.z, self.thickness)
        wet = draught > 0.0
        if bool(wet.any()):
            volume = self.waterplane_area * torch.clamp(draught, min=0.0)
            i_wl = math.pi * torch.clamp(self.radius, min=1e-12) ** 4 / 4.0
            z_cb = 0.5 * (self.z - 0.5 * self.thickness)
            gm = torch.where(volume > 0, i_wl / torch.clamp(volume, min=1e-30)
                             - (self.z - z_cb), torch.zeros_like(volume))
            k_phi = self.water.water_density * self.water.gravity * volume * gm
            c_phi = 2.0 * self.water.tilt_damping_ratio * torch.sqrt(
                torch.clamp(k_phi * self.inertia_tilt, min=0.0))
            moment = torch.where(wet, -k_phi * torch.sin(self.tilt) - c_phi * self.tilt_rate,
                                 torch.zeros_like(k_phi))
            self.tilt_rate = self.tilt_rate + dt * moment / self.inertia_tilt
            self.tilt = self.tilt + dt * self.tilt_rate
        self.steps += 1

    def run(self, n_steps: int, record_every: int = 1) -> Dict[str, list]:
        """Advance ``n_steps`` steps, recording diagnostics every ``record_every``."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        record_every = int(record_every)
        if record_every <= 0:
            raise ValueError("record_every must be positive")
        history: Dict[str, list] = {
            "time": [], "cleared_width": [], "max_speed": [], "mean_speed": [],
            "kinetic_energy": [], "max_abs_tilt": [], "max_abs_heave": [],
        }
        for step in range(int(n_steps)):
            self.step()
            if (step + 1) % record_every == 0:
                speed = torch.linalg.norm(self.vel, dim=1)
                history["time"].append(self.time)
                history["cleared_width"].append(self.cleared_width())
                history["max_speed"].append(float(speed.max()))
                history["mean_speed"].append(float(speed.mean()))
                history["kinetic_energy"].append(float(self.kinetic_energy().sum()))
                history["max_abs_tilt"].append(float(self.tilt.abs().max()))
                history["max_abs_heave"].append(float(self.z.abs().max()))
        return history