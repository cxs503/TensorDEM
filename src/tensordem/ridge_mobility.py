"""Grounded ice-ridge mobility: three-state buoyant equilibrium with a DEM closure.

Physical problem
----------------
A first-year ice ridge hangs as a *keel* from the surrounding ice sheet.  The
keel is idealised as a triangular prism in plane strain (unit out-of-plane
thickness): keel depth ``h_k`` below the waterline, flank slope ``s = w/(2 h_k)``
(half width gained per metre of depth) and rubble porosity ``n``.  Only the ice
*skeleton* displaces water (the pore water inside the rubble is neutrally
buoyant), so the submerged unit weight is ``(1-n)(rho_i - rho_w) g``.

Three states are distinguished:

``floating``
    water depth ``H`` exceeds the free-float draft
    ``t_ff = h_k sqrt(rho_i/rho_w)``: the keel hangs without seabed contact.
``grounded``
    ``H < t_ff``: the apex rests on the seabed and the seabed carries
    ``N = (1-n) g s (rho_i h_k^2 - rho_w H^2)`` per metre of ridge.
``mobile`` / ``immobile``
    a grounded keel pushed by a driving force per metre ``q`` (ship thrust
    spread over the loaded ridge length plus current drag) slides when
    ``q > R = mu_b N + c_a b``.  ``mu* = (q - c_a b)/N`` is the critical seabed
    friction coefficient, i.e. the ``(h_k/d, mu_b)`` mobility boundary.

``d`` is the *local draft of the ambient level ice sheet*; the literature band
``h_k/d = 4...6`` (Timco & Burden 1997 ridge keel depth 4-5 x level ice
thickness, draft = 0.894 x thickness) marks the validated regime of the
triangular-prism / uniform-porosity idealisation.  Cases outside the band are
computed but carry a ``model_form`` attribution flag.

The DEM closure
---------------
:class:`DemKeelRig` packs the same triangular prism out of 2-D disks at the
prescribed porosity and integrates it with

* per-grain buoyancy from the exact circular-segment submerged fraction,
* linear-spring normal contact with viscous damping,
* incrementally accumulated tangential springs capped by Coulomb friction, so a
  genuine *static* equilibrium exists and can be measured,
* a seabed plane with normal stiffness, Coulomb friction ``mu_b`` and optional
  refreezing adhesion ``c_a`` acting over the real grain contact,
* a horizontal body force representing the driving force per metre.

The static state is reached by overdamped dynamic relaxation and is audited by
the per-grain force residual ``max_i |F_i|/F_char`` (characteristic force: ridge
weight per metre) and the moment residual about the section centroid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

__all__ = [
    "RHO_ICE",
    "RHO_WATER",
    "GRAVITY",
    "LITERATURE_KEEL_DRAFT_BAND",
    "RidgeMobilityCase",
    "assess",
    "mobility_boundary",
    "free_float_draft_ratio",
    "submerged_fraction",
    "DemKeelSpec",
    "DemKeelRig",
    "dem_free_float",
    "dem_grounding",
    "dem_mobility_sweep",
    "mobility_criteria",
]

RHO_ICE = 917.0
"""Density of columnar sea ice [kg/m^3]."""

RHO_WATER = 1025.0
"""Density of seawater [kg/m^3]."""

GRAVITY = 9.81
"""Standard gravity [m/s^2]."""

LITERATURE_KEEL_DRAFT_BAND = (4.0, 6.0)
"""First-year ridge keel depth over ambient level-ice draft (Timco & Burden 1997)."""


def _require(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def free_float_draft_ratio(ice_density: float = RHO_ICE,
                           water_density: float = RHO_WATER) -> float:
    """Free-floating draft ratio ``t_ff/h_k`` of a self-similar porous wedge.

    For a triangular keel the submerged area fraction is ``(t/h_k)^2`` and the
    ice skeleton (not the pore water) displaces water, so the buoyancy balance
    ``(1-n) rho_i A g = (1-n) rho_w A_sub g`` gives ``t/h_k = sqrt(rho_i/rho_w)``
    independent of porosity and of absolute size.
    """
    _require(ice_density, "ice_density", positive=True)
    _require(water_density, "water_density", positive=True)
    return math.sqrt(ice_density / water_density)


# ---------------------------------------------------------------------------
# Closed-form three-state model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RidgeMobilityCase:
    """Input deck of one ridge-mobility assessment (SI units, per metre)."""

    keel_depth: float                       # h_k [m]
    porosity: float = 0.30                  # n [-]
    seabed_friction: float = 0.35           # mu_b [-]
    local_draft: float = 1.0                # d   [m] ambient level-ice draft
    water_depth: float = 5.0                # H   [m] water depth at the ridge
    flank_angle_deg: float = 30.0           # keel flank angle from horizontal
    keel_width: float | None = None         # w   [m], default 2 s h_k
    refreeze_adhesion: float = 0.0          # c_a [Pa] refrozen keel-seabed bond
    current_speed: float = 0.0              # u   [m/s]
    drag_coefficient: float = 1.0           # C_D [-]
    ship_thrust: float = 0.0                # [N] total thrust applied to ridge
    ship_contact_length: float = 30.0       # [m] ridge length loaded by the ship
    subgrade_modulus: float = 2.0e7         # k_s [N/m^3] elastic seabed modulus
    bearing_capacity: float | None = None   # q_ult [Pa]
    ice_density: float = RHO_ICE
    water_density: float = RHO_WATER
    gravity: float = GRAVITY
    label: str = ""

    def __post_init__(self) -> None:
        _require(self.keel_depth, "keel_depth", positive=True)
        _require(self.porosity, "porosity", nonnegative=True)
        if self.porosity >= 1.0:
            raise ValueError("porosity must be < 1")
        _require(self.seabed_friction, "seabed_friction", nonnegative=True)
        _require(self.local_draft, "local_draft", positive=True)
        _require(self.water_depth, "water_depth", nonnegative=True)
        _require(self.flank_angle_deg, "flank_angle_deg", positive=True)
        if not 0.0 < self.flank_angle_deg <= 90.0:
            raise ValueError("flank_angle_deg must be in (0, 90]")
        if self.keel_width is not None:
            _require(self.keel_width, "keel_width", positive=True)
        _require(self.refreeze_adhesion, "refreeze_adhesion", nonnegative=True)
        _require(self.current_speed, "current_speed", nonnegative=True)
        _require(self.drag_coefficient, "drag_coefficient", nonnegative=True)
        _require(self.ship_thrust, "ship_thrust", nonnegative=True)
        _require(self.ship_contact_length, "ship_contact_length", positive=True)
        _require(self.subgrade_modulus, "subgrade_modulus", positive=True)
        if self.bearing_capacity is not None:
            _require(self.bearing_capacity, "bearing_capacity", positive=True)
        _require(self.gravity, "gravity", positive=True)

    # -- geometry ---------------------------------------------------------
    @property
    def flank_slope(self) -> float:
        """Half width gained per metre of depth, ``s = w/(2 h_k)``."""
        return 1.0 / math.tan(math.radians(self.flank_angle_deg))

    @property
    def width(self) -> float:
        return (self.keel_width if self.keel_width is not None
                else 2.0 * self.flank_slope * self.keel_depth)

    @property
    def section_area(self) -> float:
        """Triangular keel cross-section area per metre of ridge [m^2/m]."""
        return 0.5 * self.width * self.keel_depth

    @property
    def solid_area(self) -> float:
        return (1.0 - self.porosity) * self.section_area

    @property
    def keel_draft_ratio(self) -> float:
        """``h_k/d``: keel depth over ambient level-ice draft."""
        return self.keel_depth / self.local_draft

    @property
    def draft_ratio_in_band(self) -> bool:
        lo, hi = LITERATURE_KEEL_DRAFT_BAND
        return lo <= self.keel_draft_ratio <= hi

    # -- buoyancy ---------------------------------------------------------
    @property
    def free_float_draft(self) -> float:
        return self.keel_depth * free_float_draft_ratio(self.ice_density, self.water_density)

    @property
    def weight_per_length(self) -> float:
        """Ice skeleton weight [N/m] (pore water excluded)."""
        return (1.0 - self.porosity) * self.ice_density * self.section_area * self.gravity

    @property
    def submerged_depth(self) -> float:
        """Depth of keel below the waterline, truncated by the water column."""
        return min(self.keel_depth, self.water_depth)

    @property
    def submerged_area(self) -> float:
        """Keel area below the waterline [m^2/m] (self-similar taper)."""
        h = self.submerged_depth
        return 0.5 * self.width * h * h / self.keel_depth

    @property
    def buoyancy_per_length(self) -> float:
        """Buoyancy on the ice skeleton [N/m] (pore water is neutrally buoyant)."""
        return ((1.0 - self.porosity) * self.water_density * self.submerged_area
                * self.gravity)

    @property
    def normal_force(self) -> float:
        """Seabed contact normal force per metre [N/m]; 0 when floating."""
        return max(0.0, self.weight_per_length - self.buoyancy_per_length)

    @property
    def grounded(self) -> bool:
        return self.normal_force > 0.0

    # -- contact width and bearing ---------------------------------------
    @property
    def contact_width(self) -> float:
        """Effective keel-seabed contact width [m].

        Wedge geometry ``b = 2 s delta`` plus an elastic Winkler bed
        ``N = k_s b delta`` gives ``b = sqrt(2 s N / k_s)``.  If the resulting
        pressure exceeds the bearing capacity the contact is plastic and
        ``b = N/q_ult``.
        """
        n = self.normal_force
        if n <= 0.0:
            return 0.0
        b_elastic = math.sqrt(2.0 * self.flank_slope * n / self.subgrade_modulus)
        if self.bearing_capacity is None:
            return b_elastic
        return max(b_elastic, n / self.bearing_capacity)

    @property
    def bearing_pressure(self) -> float:
        b = self.contact_width
        return self.normal_force / b if b > 0.0 else 0.0

    @property
    def elastic_contact_width(self) -> float:
        n = self.normal_force
        if n <= 0.0:
            return 0.0
        return math.sqrt(2.0 * self.flank_slope * n / self.subgrade_modulus)

    @property
    def plastic_contact(self) -> bool:
        """True when bearing capacity, not elasticity, sets the contact width."""
        if self.bearing_capacity is None:
            return False
        return self.normal_force / self.bearing_capacity > self.elastic_contact_width

    @property
    def bearing_pressure_ratio(self) -> float:
        """``p/q_ult``; > 1 flags local bearing failure (the keel beds in)."""
        if self.bearing_capacity is None:
            return 0.0
        return self.bearing_pressure / self.bearing_capacity

    @property
    def adhesion_force(self) -> float:
        """Refreezing adhesion per metre of ridge, ``c_a b`` [N/m]."""
        return self.refreeze_adhesion * self.contact_width

    # -- driving force ---------------------------------------------------
    @property
    def current_force(self) -> float:
        """Current drag per metre on the keel frontal (vertical) projection."""
        return (0.5 * self.water_density * self.drag_coefficient
                * self.submerged_depth * self.current_speed ** 2)

    @property
    def ship_force(self) -> float:
        return self.ship_thrust / self.ship_contact_length

    @property
    def driving_force(self) -> float:
        """Total driving force per metre of ridge [N/m]."""
        return self.ship_force + self.current_force

    # -- resistance and mobility ------------------------------------------
    @property
    def friction_resistance(self) -> float:
        return self.seabed_friction * self.normal_force

    @property
    def resistance(self) -> float:
        """Bearing limit: maximum horizontal force held by the grounded keel."""
        return self.friction_resistance + self.adhesion_force

    @property
    def critical_friction(self) -> float:
        """``mu* = (q - c_a b)/N``: the mobility boundary in ``mu_b``."""
        n = self.normal_force
        if n <= 0.0:
            return math.inf
        return (self.driving_force - self.adhesion_force) / n

    @property
    def mobility_margin(self) -> float:
        """``q - R`` [N/m]; positive => the keel is pushed off its bed."""
        return self.driving_force - self.resistance

    @property
    def mobile(self) -> bool:
        if not self.grounded:
            return True
        if self.resistance <= 0.0:
            return self.driving_force >= 0.0
        return self.driving_force > self.resistance

    @property
    def status(self) -> str:
        if not self.grounded:
            return "floating"
        return "grounded_mobile" if self.mobile else "grounded_immobile"

    @property
    def model_form_flags(self) -> tuple[str, ...]:
        flags: list[str] = []
        if not self.draft_ratio_in_band:
            lo, hi = LITERATURE_KEEL_DRAFT_BAND
            flags.append(
                f"keel_depth_over_draft_out_of_band:{self.keel_draft_ratio:.3f}"
                f"_not_in_[{lo:g},{hi:g}]::triangular_prism_uniform_porosity_unvalidated"
            )
        if self.water_depth < self.keel_depth and self.grounded:
            flags.append("keel_truncated_by_seabed:sail_and_keel_not_self_similar")
        if self.refreeze_adhesion > 0.0:
            flags.append("refreeze_adhesion_contact_width_closure_dependent")
        if self.grounded and self.plastic_contact:
            flags.append("seabed_bearing_failure:keel_bedded_in")
        return tuple(flags)


def assess(case: RidgeMobilityCase) -> dict:
    """Evaluate one case; every number the report cites comes from here."""
    row = {
        "label": case.label,
        "keel_depth_m": case.keel_depth,
        "keel_width_m": case.width,
        "flank_slope": case.flank_slope,
        "porosity": case.porosity,
        "seabed_friction": case.seabed_friction,
        "local_draft_m": case.local_draft,
        "water_depth_m": case.water_depth,
        "keel_over_draft": case.keel_draft_ratio,
        "draft_ratio_in_band": case.draft_ratio_in_band,
        "section_area_m2_per_m": case.section_area,
        "solid_area_m2_per_m": case.solid_area,
        "weight_per_length_N_per_m": case.weight_per_length,
        "free_float_draft_m": case.free_float_draft,
        "free_float_draft_over_keel": case.free_float_draft / case.keel_depth,
        "submerged_depth_m": case.submerged_depth,
        "submerged_area_m2_per_m": case.submerged_area,
        "buoyancy_per_length_N_per_m": case.buoyancy_per_length,
        "normal_force_N_per_m": case.normal_force,
        "grounded": case.grounded,
        "contact_width_m": case.contact_width,
        "bearing_pressure_Pa": case.bearing_pressure,
        "bearing_pressure_ratio": case.bearing_pressure_ratio,
        "adhesion_force_N_per_m": case.adhesion_force,
        "current_force_N_per_m": case.current_force,
        "ship_force_N_per_m": case.ship_force,
        "driving_force_N_per_m": case.driving_force,
        "friction_resistance_N_per_m": case.friction_resistance,
        "resistance_N_per_m": case.resistance,
        "mobility_margin_N_per_m": case.mobility_margin,
        "critical_friction": case.critical_friction,
        "status": case.status,
        "model_form_flags": list(case.model_form_flags),
    }
    return row


def mobility_boundary(keel_over_draft,
                      *,
                      local_draft: float = 1.0,
                      water_depth_ratio: float = 0.85,
                      porosity: float = 0.30,
                      refreeze_adhesion: float = 0.0,
                      ship_thrust: float = 0.0,
                      ship_contact_length: float = 30.0,
                      current_speed: float = 0.0,
                      drag_coefficient: float = 1.0,
                      flank_angle_deg: float = 30.0,
                      subgrade_modulus: float = 2.0e7,
                      seabed_frictions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)) -> list[dict]:
    """Closed-form ``(h_k/d, mu_b)`` mobility map plus the ``mu*(h_k/d)`` curve.

    The water depth is tied to the keel by ``H = water_depth_ratio * h_k`` so
    that every row is grounded but not deeply truncated.
    """
    if isinstance(keel_over_draft, (int, float)):
        ratios = [float(keel_over_draft)]
    else:
        ratios = [float(r) for r in keel_over_draft]
    rows: list[dict] = []
    for ratio in ratios:
        h_k = ratio * local_draft
        case = RidgeMobilityCase(
            keel_depth=h_k,
            porosity=porosity,
            local_draft=local_draft,
            water_depth=water_depth_ratio * h_k,
            flank_angle_deg=flank_angle_deg,
            refreeze_adhesion=refreeze_adhesion,
            current_speed=current_speed,
            drag_coefficient=drag_coefficient,
            ship_thrust=ship_thrust,
            ship_contact_length=ship_contact_length,
            subgrade_modulus=subgrade_modulus,
            label=f"hk/d={ratio:g}",
        )
        row = assess(case)
        row["mu_star"] = case.critical_friction
        row["mobility_map"] = {
            f"{mu:.2f}": ("mobile" if case.grounded and mu * case.normal_force
                          + case.adhesion_force < case.driving_force else
                          "floating" if not case.grounded else "immobile")
            for mu in seabed_frictions
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# DEM closure: a 2-D disk packing of the same triangular prism
# ---------------------------------------------------------------------------

def submerged_fraction(centres: torch.Tensor, radius: float,
                       waterline: float = 0.0) -> torch.Tensor:
    """Exact circular-segment submerged area fraction of disks (0..1)."""
    h = (centres - waterline) / radius
    out = torch.zeros_like(h)
    full = h <= -1.0
    dry = h >= 1.0
    mid = ~(full | dry)
    if bool(mid.any()):
        theta = 2.0 * torch.acos(h[mid].clamp(-1.0, 1.0))
        out[mid] = (theta - torch.sin(theta)) / (2.0 * math.pi)
    out = torch.where(full, torch.ones_like(out), out)
    out = torch.where(dry, torch.zeros_like(out), out)
    return out


@dataclass(frozen=True)
class DemKeelSpec:
    """Micro-parameters of the DEM keel packing (SI units, 1 m thick slice)."""

    keel_depth: float = 5.0
    porosity: float = 0.30
    flank_angle_deg: float = 30.0
    keel_width: float | None = None
    radius: float = 0.075
    seed: int = 20261110
    ice_density: float = RHO_ICE
    water_density: float = RHO_WATER
    gravity: float = GRAVITY
    normal_stiffness: float = 2.0e6
    tangential_ratio: float = 0.5
    contact_damping_ratio: float = 0.2
    particle_friction: float = 0.40
    ground_stiffness_scale: float = 1.0
    jitter: float = 0.02
    bond_stiffness_scale: float = 5.0
    bonded: bool = True

    @property
    def width(self) -> float:
        if self.keel_width is not None:
            return self.keel_width
        return 2.0 * self.keel_depth / math.tan(math.radians(self.flank_angle_deg))

    @property
    def particle_area(self) -> float:
        return math.pi * self.radius ** 2

    @property
    def particle_mass(self) -> float:
        return self.ice_density * self.particle_area

    @property
    def target_solid_area(self) -> float:
        return (1.0 - self.porosity) * 0.5 * self.width * self.keel_depth


class DemKeelRig:
    """Overdamped 2-D disk packing of a triangular ridge keel."""

    def __init__(self, spec: DemKeelSpec):
        self.spec = spec
        self.dtype = torch.float64
        self.positions, self.radii = self._build_packing()
        self.n = self.positions.shape[0]
        self.velocities = torch.zeros_like(self.positions)
        self.inv_mass = torch.full((self.n,), 1.0 / spec.particle_mass, dtype=self.dtype)
        self.pairs: torch.Tensor | None = None
        self.xi_t: torch.Tensor | None = None
        self.bonds, self.bond_rest = self._build_bonds()
        self.xi_g = torch.zeros(self.n, dtype=self.dtype)
        g = spec.gravity
        self.kn = spec.normal_stiffness
        self.kt = spec.tangential_ratio * spec.normal_stiffness
        self.cn = spec.contact_damping_ratio * 2.0 * math.sqrt(
            spec.normal_stiffness * spec.particle_mass)
        self.kn_g = spec.normal_stiffness * spec.ground_stiffness_scale
        self.kt_g = self.kt
        self.cn_g = self.cn
        self.kb = spec.bond_stiffness_scale * spec.normal_stiffness
        self.weight = spec.particle_mass * g
        self.characteristic_force = (1.0 - spec.porosity) * spec.ice_density * (
            0.5 * spec.width * spec.keel_depth) * g
        self.steps = 0

    # -- packing generation ----------------------------------------------
    def _build_packing(self) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.spec
        r = spec.radius
        # A monodisperse disk packing cannot hold an areal solid fraction of
        # 1-porosity = 0.70 (random disk packings saturate near 0.85), so the
        # packing is generated at near-contact spacing and its *measured*
        # porosity is fed back into the closed-form comparison (like-for-like).
        spacing = 2.0 * r * 1.02
        gen = torch.Generator().manual_seed(spec.seed)
        h_k, w = spec.keel_depth, spec.width
        s = w / (2.0 * h_k)
        xs, ys = [], []
        rows = int(math.ceil(h_k / (spacing * math.sqrt(3.0) / 2.0))) + 2
        cols = int(math.ceil(s * h_k / spacing)) + 2
        for row in range(-1, rows + 1):
            y = -h_k + row * spacing * math.sqrt(3.0) / 2.0
            x_off = 0.5 * spacing * (row % 2)
            for col in range(-cols, cols + 1):
                x = col * spacing + x_off
                if abs(x) <= s * (y + h_k) - r * 0.5 and -h_k <= y <= 0.0:
                    xs.append(x)
                    ys.append(y)
        pos = torch.tensor([xs, ys], dtype=self.dtype).t().contiguous()
        jitter = (torch.rand(pos.shape, generator=gen, dtype=self.dtype) - 0.5) * 2.0 * spec.jitter * spacing
        pos = pos + jitter
        # keep centres inside the prism
        inside = (pos[:, 1] >= -h_k) & (pos[:, 1] <= 0.0) & (
            pos[:, 0].abs() <= s * (pos[:, 1] + h_k) - r * 0.6)
        pos = pos[inside]
        radii = torch.full((pos.shape[0],), r, dtype=self.dtype)
        return pos, radii

    def _build_bonds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Freeze bonds of the consolidated/refrozen keel skeleton.

        A loose disk pile cannot hold the triangular-prism geometry: it spreads
        to the angle of repose.  A *consolidated* first-year keel (refrozen
        rubble, the case that survives the summer and grounds) is a coherent
        body, so the skeleton is bonded at packing time with axial springs
        (tension + compression, no strength limit).  The bonds are internal,
        so the external force ledger is unchanged; they only preserve the
        prescribed prism geometry and porosity.
        """
        if not self.spec.bonded:
            empty = torch.zeros((0,), dtype=torch.int64)
            return torch.zeros((2, 0), dtype=torch.int64), torch.zeros((0,), dtype=self.dtype)
        pos = self.positions
        dist = torch.cdist(pos, pos)
        idx = torch.triu_indices(self.n, self.n, offset=1)
        d = dist[idx[0], idx[1]]
        keep = d < 2.0 * self.spec.radius * 1.06
        bonds = torch.stack([idx[0][keep], idx[1][keep]])
        return bonds, d[keep]

    @property
    def measured_height(self) -> float:
        return float(self.positions[:, 1].max() - self.positions[:, 1].min() + 2.0 * self.spec.radius)

    @property
    def measured_porosity(self) -> float:
        solid = self.n * self.spec.particle_area
        geom = 0.5 * self.spec.width * self.spec.keel_depth
        return 1.0 - solid / geom

    # -- neighbour bookkeeping -------------------------------------------
    def rebuild_pairs(self) -> None:
        pos = self.positions
        dist = torch.cdist(pos, pos)
        idx = torch.triu_indices(self.n, self.n, offset=1)
        cutoff = 2.0 * self.spec.radius * 1.02
        keep = dist[idx[0], idx[1]] < cutoff
        self.pairs = torch.stack([idx[0][keep], idx[1][keep]])
        self.xi_t = torch.zeros(self.pairs.shape[1], dtype=self.dtype)

    # -- force assembly ---------------------------------------------------
    def forces(self, *, waterline: float, ground_depth: float | None,
               seabed_friction: float, adhesion_pa: float,
               drive_force_per_length: float, dt: float) -> torch.Tensor:
        spec = self.spec
        pos, vel = self.positions, self.velocities
        f = torch.zeros_like(pos)
        # gravity + buoyancy per grain
        frac = submerged_fraction(pos[:, 1], spec.radius, waterline)
        f[:, 1] -= self.weight
        f[:, 1] += spec.water_density * spec.particle_area * frac * spec.gravity
        # grain-grain contacts
        if self.pairs is None:
            self.rebuild_pairs()
        i, j = self.pairs
        d = pos[j] - pos[i]
        dist = torch.linalg.norm(d, dim=1).clamp_min(1e-12)
        overlap = 2.0 * spec.radius - dist
        active = overlap > 0.0
        if bool(active.any()):
            ia, ja = i[active], j[active]
            n_vec = d[active] / dist[active].unsqueeze(1)
            v_rel = vel[ja] - vel[ia]
            vn = (v_rel * n_vec).sum(dim=1)
            fn = (self.kn * overlap[active] - self.cn * vn).clamp_min(0.0)
            t_vec = torch.stack([-n_vec[:, 1], n_vec[:, 0]], dim=1)
            vt = (v_rel * t_vec).sum(dim=1)
            xi = self.xi_t[active] + vt * dt
            ft = -self.kt * xi
            cap = spec.particle_friction * fn
            over = ft.abs() > cap
            ft = torch.where(over, torch.sign(ft) * cap, ft)
            xi = torch.where(over, -ft / self.kt, xi)
            self.xi_t[active] = xi
            f_pair = fn.unsqueeze(1) * n_vec + ft.unsqueeze(1) * t_vec
            f.index_add_(0, ia, -f_pair)
            f.index_add_(0, ja, f_pair)
        # freeze-bonded skeleton (internal forces, axial springs)
        if self.bonds.shape[1] > 0:
            bi, bj = self.bonds
            bd = pos[bj] - pos[bi]
            bl = torch.linalg.norm(bd, dim=1).clamp_min(1e-12)
            bdir = bd / bl.unsqueeze(1)
            bf = self.kb * (bl - self.bond_rest)
            f_bond = bf.unsqueeze(1) * bdir
            f.index_add_(0, bi, f_bond)
            f.index_add_(0, bj, -f_bond)
        # seabed plane
        if ground_depth is not None:
            pen = ground_depth - (pos[:, 1] - spec.radius)
            bed = pen > 0.0
            if bool(bed.any()):
                fn_g = (self.kn_g * pen[bed] - self.cn_g * vel[bed, 1]).clamp_min(0.0)
                xi_g = self.xi_g[bed] + vel[bed, 0] * dt
                cap_g = seabed_friction * fn_g + adhesion_pa * 2.0 * spec.radius
                ft_g = -self.kt_g * xi_g
                over_g = ft_g.abs() > cap_g
                ft_g = torch.where(over_g, torch.sign(ft_g) * cap_g, ft_g)
                xi_g = torch.where(over_g, -ft_g / self.kt_g, xi_g)
                self.xi_g[bed] = xi_g
                f[bed, 1] += fn_g
                f[bed, 0] += ft_g
        # driving force per metre, distributed by mass
        if drive_force_per_length != 0.0:
            f[:, 0] += drive_force_per_length / self.n
        return f

    # -- relaxation -------------------------------------------------------
    def relax(self, *, waterline: float = 0.0, ground_depth: float | None = None,
              seabed_friction: float = 0.35, adhesion_pa: float = 0.0,
              drive_force_per_length: float = 0.0, dt: float | None = None,
              max_steps: int = 40000, damping: float = 0.05, local_damping: float = 0.0,
              bulk_damping: float = 0.0, kinetic_damping: bool = False,
              tol: float = 1e-8, rebuild_every: int = 100,
              rebuild_until: int | None = 4000, verbose: bool = False) -> dict:
        """Dynamic relaxation to static equilibrium (local damping, PFC-style).

        ``local_damping`` subtracts ``alpha |F| sign(v)`` per grain, i.e. it
        removes energy in proportion to the *unbalanced* force instead of the
        velocity, which leaves a static state (``v = 0``) as an exact fixed
        point.  ``damping`` is an additional global velocity scale.
        """
        if dt is None:
            dt = 0.25 * math.sqrt(self.spec.particle_mass / self.kn)
        self.rebuild_pairs()
        residual = math.inf
        hist = []
        self.x_track = []
        ke_best = math.inf
        for step in range(max_steps):
            f = self.forces(waterline=waterline, ground_depth=ground_depth,
                            seabed_friction=seabed_friction, adhesion_pa=adhesion_pa,
                            drive_force_per_length=drive_force_per_length, dt=dt)
            residual = float(f.abs().max()) / self.characteristic_force
            if local_damping > 0.0:
                f_eff = f - local_damping * f.abs() * torch.sign(self.velocities)
            else:
                f_eff = f
            self.velocities = (self.velocities + f_eff * self.inv_mass.unsqueeze(1) * dt) * (1.0 - damping)
            if kinetic_damping:
                # kinetic damping (dynamic relaxation): quench all velocities when
                # the kinetic energy peaks.  Converges to the static state far
                # faster than viscous damping, and leaves v = 0 as the fixed point.
                ke = float((self.velocities ** 2).sum())
                if ke < ke_best:
                    ke_best = ke
                elif ke > 1e-30:
                    self.velocities = torch.zeros_like(self.velocities)
                    ke_best = math.inf
            if bulk_damping > 0.0:
                # damp only the bulk (rigid-body) drift, leave internal modes alone:
                # a free-floating or sliding section otherwise creeps for ever
                self.velocities = self.velocities - bulk_damping * self.velocities.mean(dim=0)
            self.positions = self.positions + self.velocities * dt
            self.steps += 1
            limit = self.n if rebuild_until is None else rebuild_until
            if rebuild_every and step < limit and step % rebuild_every == rebuild_every - 1:
                self.rebuild_pairs()
            if step % 200 == 0:
                hist.append(residual)
                self.x_track.append(float(self.positions[:, 0].mean()))
                if verbose:
                    print(f"  step {step} residual {residual:.3e} F(x) {float(f[:,0].sum()):.3g}", flush=True)
            if residual < tol:
                break
        f = self.forces(waterline=waterline, ground_depth=ground_depth,
                        seabed_friction=seabed_friction, adhesion_pa=adhesion_pa,
                        drive_force_per_length=drive_force_per_length, dt=dt)
        return self.ledger(f, waterline=waterline, ground_depth=ground_depth,
                           seabed_friction=seabed_friction, adhesion_pa=adhesion_pa,
                           drive_force_per_length=drive_force_per_length, history=hist)

    def ledger(self, f: torch.Tensor, *, waterline: float, ground_depth: float | None,
               seabed_friction: float, adhesion_pa: float,
               drive_force_per_length: float, history=None) -> dict:
        spec = self.spec
        frac = submerged_fraction(self.positions[:, 1], spec.radius, waterline)
        weight = float(self.weight) * self.n
        buoyancy = float(spec.water_density * spec.particle_area * spec.gravity) * float(frac.sum())
        pen = (ground_depth - (self.positions[:, 1] - spec.radius)) if ground_depth is not None else None
        if pen is not None:
            bed = pen > 0
            fn_g = (self.kn_g * pen[bed] - self.cn_g * self.velocities[bed, 1]).clamp_min(0.0)
            normal_bed = float(fn_g.sum())
            tangential_bed = float((-self.kt_g * self.xi_g[bed]).sum())
            bed_count = int(bed.sum())
            contact_width = 2.0 * spec.radius * bed_count
            penetration = float(pen[bed].max()) if bed_count else 0.0
        else:
            normal_bed = tangential_bed = contact_width = penetration = 0.0
            bed_count = 0
        f_char = self.characteristic_force
        centroid = self.positions.mean(dim=0)
        r = self.positions - centroid
        moment = float((r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]).abs().max())
        residual_force = float(f.abs().max()) / f_char
        residual_moment = moment / (f_char * spec.keel_depth)
        solid_sub = spec.particle_area * float(frac.sum())
        solid_tot = self.n * spec.particle_area
        out = {
            "particles": int(self.n),
            "measured_porosity": self.measured_porosity,
            "weight_per_length_N_per_m": weight,
            "buoyancy_per_length_N_per_m": buoyancy,
            "net_force_y_N_per_m": float(f[:, 1].sum()),
            "net_force_x_N_per_m": float(f[:, 0].sum()),
            "ground_normal_N_per_m": normal_bed,
            "ground_tangential_N_per_m": tangential_bed,
            "ground_contact_grains": bed_count,
            "ground_contact_width_m": contact_width,
            "ground_penetration_m": penetration,
            "seabed_friction": seabed_friction,
            "adhesion_pa": adhesion_pa,
            "drive_force_per_length_N_per_m": drive_force_per_length,
            "residual_force_ratio": residual_force,
            "residual_moment_ratio": residual_moment,
            "residual_history": history or [],
            "steps": int(self.steps),
            "solid_submerged_fraction": solid_sub / solid_tot,
            "draft_ratio_area": math.sqrt(max(solid_sub / solid_tot, 0.0)),
            "draft_ratio_geometric": float(waterline - (self.positions[:, 1].min() - spec.radius)) / spec.keel_depth,
            "top_y": float(self.positions[:, 1].max()),
            "bottom_y": float(self.positions[:, 1].min()),
            "mean_x": float(self.positions[:, 0].mean()),
            "mean_velocity_x": float(self.velocities[:, 0].mean()),
            "drift_late_m": (self.x_track[int(0.75 * len(self.x_track))] - self.x_track[-1]
                             if len(self.x_track) >= 4 else 0.0),
            "mean_velocity_y": float(self.velocities[:, 1].mean()),
        }
        return out


def _rig_for(case: RidgeMobilityCase, spec: DemKeelSpec | None) -> DemKeelRig:
    spec = spec or DemKeelSpec(keel_depth=case.keel_depth, porosity=case.porosity,
                              flank_angle_deg=case.flank_angle_deg,
                              keel_width=case.keel_width)
    return DemKeelRig(spec)


def dem_free_float(case: RidgeMobilityCase, spec: DemKeelSpec | None = None,
                   *, max_steps: int = 20000, tol: float = 1e-7) -> tuple[dict, dict]:
    """DEM free-float check: no seabed, the packing finds its own waterline.

    The keel hangs from the ice sheet, so the mouldline is held at the
    waterline and the whole body is free to find the draft where buoyancy
    balances weight.  Two draft measures are reported: the *geometric* draft of
    the packing surface (noisy at the ``O(r/h_k)`` grain scale) and the draft
    implied by the emergent submerged solid fraction, ``sqrt(phi)``, which is
    exact for a self-similar prism and is the quantity the closed form predicts.
    """
    rig = _rig_for(case, spec)
    rig.positions = rig.positions - rig.positions[:, 1].max()
    ledger = rig.relax(waterline=0.0, ground_depth=None, seabed_friction=0.0,
                       max_steps=max_steps, tol=tol)
    h_meas = rig.measured_height
    cf = {
        "free_float_draft_m": case.free_float_draft,
        "free_float_draft_ratio_cf": free_float_draft_ratio(case.ice_density, case.water_density),
        "keel_depth_m": case.keel_depth,
        "particle_radius_over_keel": rig.spec.radius / case.keel_depth,
        "measured_height_m": h_meas,
        "archimedes_ratio_cf": case.ice_density / case.water_density,
        "archimedes_ratio_dem": ledger["solid_submerged_fraction"],
    }
    return ledger, cf


def dem_grounding(case: RidgeMobilityCase, spec: DemKeelSpec | None = None,
                  *, max_steps: int = 20000, tol: float = 1e-7,
                  loaded_steps: int = 6000,
                  seabed_friction: float | None = None,
                  drive_force_per_length: float | None = None) -> tuple[dict, dict]:
    """DEM grounded state: seabed at ``-water_depth``, measure N, T and mu*."""
    rig = _rig_for(case, spec)
    mu = case.seabed_friction if seabed_friction is None else seabed_friction
    drive = case.driving_force if drive_force_per_length is None else drive_force_per_length
    ground = -abs(case.water_depth)
    rig.positions = rig.positions - rig.positions[:, 1].min() + ground + rig.spec.radius * 0.5
    x0 = float(rig.positions[:, 0].mean())
    # phase 1: static (undriven) contact state -> the normal force the mobility
    # criterion is defined on, and the state that C1 audits
    static_ledger = rig.relax(waterline=0.0, ground_depth=ground, seabed_friction=mu,
                              adhesion_pa=case.refreeze_adhesion,
                              drive_force_per_length=0.0, max_steps=max_steps, tol=tol)
    # phase 2: continue from the static state with the drive applied
    x1 = float(rig.positions[:, 0].mean())
    ledger = rig.relax(waterline=0.0, ground_depth=ground, seabed_friction=mu,
                       adhesion_pa=case.refreeze_adhesion,
                       drive_force_per_length=drive, max_steps=max(loaded_steps, 1), tol=tol)
    ledger["static"] = {k: v for k, v in static_ledger.items() if k != "residual_history"}
    ledger["drift_m"] = float(rig.positions[:, 0].mean()) - x1
    h_meas = rig.measured_height
    n_dem = static_ledger["ground_normal_N_per_m"]
    n_dem_loaded = ledger["ground_normal_N_per_m"]
    adh_dem = case.refreeze_adhesion * static_ledger["ground_contact_width_m"]
    # closed form re-evaluated on the *realised* prism height (the packing
    # cannot resolve the last grain of the apex): geometry-injected closure.
    s = case.flank_slope
    n_cf_meas = ((1.0 - ledger["measured_porosity"]) * case.gravity * s
                 * (case.ice_density * h_meas ** 2 - case.water_density * case.water_depth ** 2))
    cf = {
        "normal_force_cf_N_per_m": case.normal_force,
        "normal_force_cf_measured_geometry_N_per_m": max(0.0, n_cf_meas),
        "normal_ratio": n_dem / case.normal_force if case.normal_force > 0 else math.inf,
        "normal_ratio_measured_geometry": n_dem / n_cf_meas if n_cf_meas > 0 else math.inf,
        "friction_resistance_cf_N_per_m": case.friction_resistance,
        "adhesion_cf_N_per_m": case.adhesion_force,
        "adhesion_dem_N_per_m": adh_dem,
        "mu_star_cf": case.critical_friction,
        "mu_star_dem": ((drive - adh_dem) / n_dem) if n_dem > 0 else math.inf,
        "measured_height_m": h_meas,
        "measured_porosity": ledger["measured_porosity"],
        "water_depth_over_keel": case.water_depth / case.keel_depth,
        "normal_force_dem_static_N_per_m": n_dem,
        "normal_force_dem_loaded_N_per_m": n_dem_loaded,
        "residual_force_ratio_static": static_ledger["residual_force_ratio"],
        "residual_moment_ratio_static": static_ledger["residual_moment_ratio"],
        "drift_m": ledger["drift_m"],
        "drift_late_m": ledger["drift_late_m"],
        "unbalanced_force_N_per_m": drive + ledger["ground_tangential_N_per_m"],
        "drive_force_per_length_N_per_m": drive,
        "mobilised_friction": (-ledger["ground_tangential_N_per_m"] / n_dem_loaded
                               if n_dem_loaded > 0 else 0.0),
    }
    cf["mu_star_ratio"] = (cf["mu_star_dem"] / cf["mu_star_cf"]
                           if cf["mu_star_cf"] not in (0.0, math.inf) else math.inf)
    return ledger, cf


def dem_mobility_sweep(case: RidgeMobilityCase, mu_values, *,
                       spec: DemKeelSpec | None = None,
                       max_steps: int = 20000, tol: float = 1e-7,
                       loaded_steps: int = 6000,
                       slide_tolerance_grains: float = 0.25) -> list[dict]:
    """Load a grounded keel with the driving force at each ``mu_b`` and classify.

    A keel is ``mobile`` when its section centroid drifts by more than
    ``slide_tolerance_grains`` grain radii along the drive; it is ``held`` when
    the mobilised seabed friction stops it.
    """
    rows: list[dict] = []
    for mu in mu_values:
        c = RidgeMobilityCase(**{**case.__dict__, "seabed_friction": float(mu)})
        rig_spec = spec or DemKeelSpec(keel_depth=case.keel_depth,
                                       porosity=case.porosity,
                                       flank_angle_deg=case.flank_angle_deg,
                                       keel_width=case.keel_width)
        ledger, cf = dem_grounding(c, rig_spec, max_steps=max_steps, tol=tol,
                                   loaded_steps=loaded_steps)
        tol_m = slide_tolerance_grains * rig_spec.radius
        n_dem = ledger["ground_normal_N_per_m"]
        drive = cf["drive_force_per_length_N_per_m"]
        # Unbalanced horizontal force: the drive minus what the seabed friction
        # actually mobilised.  A held keel has ~0; a sliding keel keeps the
        # excess (the friction saturates at mu_b N).
        unbalanced = drive + ledger["ground_tangential_N_per_m"]
        rows.append({
            "seabed_friction": float(mu),
            # A held keel needs only part of the available friction; a sliding
            # one saturates the cap mu_b N and keeps an unbalanced force.  The
            # (damped) terminal velocity is a poor discriminator because the
            # relaxation damping caps it, so the friction saturation is used.
            "mobile": bool(abs(unbalanced) > 0.02 * max(drive, 1e-12)
                           and ledger["ground_normal_N_per_m"] > 0.0
                           and (-ledger["ground_tangential_N_per_m"]
                                / ledger["ground_normal_N_per_m"]) > 0.98 * mu),
            "unbalanced_force_ratio": unbalanced / max(drive, 1e-12),
            "unbalanced_force_N_per_m": unbalanced,
            "drift_m": cf["drift_m"],
            "mean_velocity_x": ledger["mean_velocity_x"],
            "drift_exceeds_grains": abs(cf["drift_m"]) > tol_m,
            "slide_tolerance_m": tol_m,
            "ground_normal_N_per_m": ledger["ground_normal_N_per_m"],
            "ground_normal_static_N_per_m": n_dem,
            "ground_tangential_N_per_m": ledger["ground_tangential_N_per_m"],
            "drift_late_m": ledger["drift_late_m"],
            "mobilised_friction": (-ledger["ground_tangential_N_per_m"] / ledger["ground_normal_N_per_m"]
                                   if ledger["ground_normal_N_per_m"] > 0 else 0.0),
            "required_friction": (cf["drive_force_per_length_N_per_m"] - cf["adhesion_dem_N_per_m"]) / n_dem,
            "residual_force_ratio": ledger["residual_force_ratio"],
            "residual_moment_ratio": ledger["residual_moment_ratio"],
            "mu_star_cf": case.critical_friction,
        })
    return rows


# ---------------------------------------------------------------------------
# Machine-checkable criteria C1..C5
# ---------------------------------------------------------------------------

def mobility_criteria(ledgers: dict, cf: dict) -> dict:
    """Aggregate the C1..C5 verdicts from DEM ledgers and closed-form values."""
    out = {"C1": {}, "C2": {}, "C3": {}, "C4": {}, "C5": {}}
    out["C1"] = {
        "reward": "per-case max|F|/F_char < 1e-6 and max|M|/(F_char h_k) < 1e-6",
        "cases": {k: {"residual_force_ratio": v["residual_force_ratio"],
                      "residual_moment_ratio": v["residual_moment_ratio"]}
                  for k, v in ledgers.items()},
    }
    out["C1"]["pass"] = all(v["residual_force_ratio"] < 1e-6
                            and v["residual_moment_ratio"] < 1e-6
                            for v in out["C1"]["cases"].values())
    return out