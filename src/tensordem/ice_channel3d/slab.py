"""Slab (layered) channel DEM: geometry, contact, ledger, resistance accounting.

See the package docstring for the model statement and the evidence grades.
Everything here is deterministic given ``SlabConfig.seed``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WATER_DENSITY = 1025.0
ICE_DENSITY = 917.0
GRAVITY = 9.80665
POISSON_ICE = 0.3

#: Lindqvist bending efficiency, inherited verbatim from the graded ledger in
#: :mod:`tensordem.level_ice.formula` (``bending_efficiency = 0.25``,
#: grade CONVENTION, lever arm ``L_c / 1.5``).  Re-used rather than re-invented
#: so that this module carries no new free coefficient.
BENDING_EFFICIENCY = 0.25
#: Submersion closure constants, also inherited from that ledger
#: (``submersion_buoyancy = 0.9``, ``submersion_friction = 0.05``).
SUBMERSION_BUOYANCY = 0.9
SUBMERSION_FRICTION = 0.05


def characteristic_length_m(thickness: float, elastic_modulus: float,
                            rho_water: float = WATER_DENSITY,
                            poisson: float = POISSON_ICE,
                            gravity: float = GRAVITY) -> float:
    """Plate characteristic length ``L_c = (E h^3 / (12 (1-nu^2) rho g))^(1/4)``."""
    if thickness <= 0 or elastic_modulus <= 0:
        raise ValueError("thickness and elastic modulus must be positive")
    return (elastic_modulus * thickness ** 3
            / (12.0 * (1.0 - poisson ** 2) * rho_water * gravity)) ** 0.25


def bending_capacity_per_width_n_per_m(flexural_strength: float,
                                       thickness: float,
                                       char_length: float) -> float:
    """Breaking force per unit width of sheet, ``0.25 sigma_f h^2 / L_c`` [N/m]."""
    if char_length <= 0:
        raise ValueError("char_length must be positive")
    return BENDING_EFFICIENCY * flexural_strength * thickness ** 2 / char_length


def layer_moment_shares(n_layers: int, thickness: float) -> np.ndarray:
    """Share of the full-slab bending capacity carried by each layer.

    The layer occupying ``[z_k, z_{k+1}]`` (``z`` from ``-h/2`` to ``h/2``)
    carries ``M_k = 2 sigma_f (z_{k+1}^3 - z_k^3) / (3 h)`` of the slab moment
    ``sigma_f h^2 / 6``, hence ``s_k = 4 (z_{k+1}^3 - z_k^3) / h^3``.  The
    shares sum to one exactly, which is what makes the layered slab converge to
    the full-thickness sheet (independent lamina bending would instead scale as
    ``n t^2 / h^2 = 1/n``).
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    edges = np.linspace(-0.5 * thickness, 0.5 * thickness, n_layers + 1)
    shares = 4.0 * (edges[1:] ** 3 - edges[:-1] ** 3) / thickness ** 3
    return shares


@dataclass(frozen=True)
class IceSheetProps:
    """Level-ice sheet material state for one experimental case (SI units)."""

    thickness_m: float
    flexural_strength_pa: float
    compressive_strength_pa: float
    elastic_modulus_pa: float
    poisson_ratio: float = POISSON_ICE
    ice_density: float = ICE_DENSITY

    def __post_init__(self) -> None:
        for name in ("thickness_m", "flexural_strength_pa",
                     "compressive_strength_pa", "elastic_modulus_pa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

    @property
    def char_length_m(self) -> float:
        return characteristic_length_m(self.thickness_m, self.elastic_modulus_pa,
                                       poisson=self.poisson_ratio)

    @property
    def bending_per_width_n_per_m(self) -> float:
        return bending_capacity_per_width_n_per_m(
            self.flexural_strength_pa, self.thickness_m, self.char_length_m)


@dataclass(frozen=True)
class HullFootprint:
    """Horizontal footprint of the hull: a stem wedge plus parallel sides.

    Stem at the origin, shoulder at ``(-L_bow, +-B/2)``, stern at
    ``(-L_ship, +-B/2)``.  ``bow_half_angle_deg`` is the half entrance angle of
    the wedge (the angle between the centreline and the bow face in plan view).
    """

    beam_m: float = 22.5
    length_m: float = 100.0
    bow_half_angle_deg: float = 30.0

    def __post_init__(self) -> None:
        for name in ("beam_m", "length_m", "bow_half_angle_deg"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def bow_length_m(self) -> float:
        return 0.5 * self.beam_m / math.tan(math.radians(self.bow_half_angle_deg))

    def vertices(self) -> np.ndarray:
        half = 0.5 * self.beam_m
        return np.array([[0.0, 0.0],
                         [-self.bow_length_m, half],
                         [-self.length_m, half],
                         [-self.length_m, -half],
                         [-self.bow_length_m, -half]], dtype=float)

    def edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(start_points, outward_normals)`` of the five convex edges."""
        v = self.vertices()
        nxt = np.roll(v, -1, axis=0)
        centroid = v.mean(axis=0)
        starts = v.copy()
        d = nxt - v
        norms = np.stack([d[:, 1], -d[:, 0]], axis=1)
        norms /= np.linalg.norm(norms, axis=1, keepdims=True)
        mid = 0.5 * (v + nxt)
        flip = np.einsum("ij,ij->i", mid - centroid, norms) < 0.0
        norms[flip] *= -1.0
        return starts, norms

    def signed_distance(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Outside-positive signed distance to the convex footprint and normal.

        ``distance = max_j dot(p - v_j, n_j)`` over the edges (exact for a
        convex polygon: the maximising edge is the nearest face).
        """
        pts = np.asarray(points, dtype=float)
        starts, norms = self.edges()
        d = np.einsum("njk,jk->nj", pts[:, None, :] - starts[None, :, :], norms)
        idx = np.argmax(d, axis=1)
        return d[np.arange(len(pts)), idx], norms[idx]


# --------------------------------------------------------------------------
# ledgers
# --------------------------------------------------------------------------
@dataclass
class MassLedger:
    """Mass bookkeeping of the slice: nothing may vanish without a line item."""

    ice_density: float
    initial_mass_kg: float
    inflow_kg: float = 0.0
    outflow_kg: float = 0.0        # floes that left the domain downstream
    deleted_kg: float = 0.0        # legacy instant clearance (destructive)
    departure_kg: float = 0.0      # channel -> reservoir band (lateral)
    return_kg: float = 0.0         # reservoir band -> channel (lateral)
    departure_vertical_kg: float = 0.0   # subducted under the hull
    return_vertical_kg: float = 0.0      # re-surfaced in the wake
    field_mass_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)

    def register_departure(self, mass: float) -> None:
        self.departure_kg += mass

    def register_return(self, mass: float) -> None:
        self.return_kg += mass

    def register_outflow(self, mass: float) -> None:
        self.outflow_kg += mass

    def register_deletion(self, mass: float) -> None:
        self.deleted_kg += mass

    def register_inflow(self, mass: float) -> None:
        self.inflow_kg += mass

    def register_departure_vertical(self, mass: float) -> None:
        self.departure_vertical_kg += mass

    def register_return_vertical(self, mass: float) -> None:
        self.return_vertical_kg += mass

    def record_frame(self, field_mass_kg: float) -> float:
        """Append a frame sample; return the *physical* conservation error.

        ``error = |initial + inflow - outflow - field| / initial``.

        Only the domain boundaries (inflow, outflow) are admissible transfers.
        Deleted mass is therefore **not** subtracted: it is destroyed inside the
        domain, so it shows up as a conservation error -- which is precisely the
        legacy defect the mission is about (the ledger knows the amount, but the
        simulation cannot restore the floe).  Lateral and vertical departures are
        internal transfers to the reservoir band / under-hull FIFO, so the floes
        stay in the field and do not enter this balance at all.  A corrected run
        has ``deleted_kg == 0`` and the error sits at round-off level.
        """
        expected = self.initial_mass_kg + self.inflow_kg - self.outflow_kg
        err = abs(expected - field_mass_kg) / self.initial_mass_kg
        self.field_mass_history.append(float(field_mass_kg))
        self.error_history.append(float(err))
        return float(err)

    @property
    def max_frame_error(self) -> float:
        return max(self.error_history) if self.error_history else 0.0


@dataclass
class FluxLedger:
    """Lateral-flux closure of the opening.

    ``closure = (departure - return) / (reservoir_mass - reservoir_mass_0)``.
    Because every crossing is registered at the exact frame it happens, the
    identity is exact up to floating point; the reported number is the
    machine-checkable version of "the band is a reservoir, not a bin".
    """

    reservoir_mass_0_kg: float = 0.0
    reservoir_mass_kg: float = 0.0
    samples: List[float] = field(default_factory=list)

    def closure(self, departure_kg: float, return_kg: float) -> float:
        net = departure_kg - return_kg
        delta = self.reservoir_mass_kg - self.reservoir_mass_0_kg
        if abs(net) <= 1e-12 and abs(delta) <= 1e-12:
            return 1.0
        if abs(net) <= 1e-12:
            return 0.0
        return delta / net

    def record(self, departure_kg: float, return_kg: float) -> float:
        value = self.closure(departure_kg, return_kg)
        self.samples.append(float(value))
        return float(value)

    @property
    def worst_deviation(self) -> float:
        if not self.samples:
            return 0.0
        return max(abs(v - 1.0) for v in self.samples)


# --------------------------------------------------------------------------
# configuration / result
# --------------------------------------------------------------------------
@dataclass
class SlabConfig:
    """Controls of one slab-slice run.  The three legacy switches are explicit."""

    n_layers: int = 3
    floe_radius_m: float = 2.0
    channel_width_over_beam: float = 1.5
    lateral_open: bool = True          # legacy 2-D slice sets this False
    instant_removal: bool = False      # legacy behaviour sets this True
    reservoir_band_m: float = 10.0
    upstream_length_m: float = 26.0
    downstream_length_m: float = 12.0
    drift_speed_m_s: float = 0.2
    frames: int = 200
    advance_per_frame_m: float = 0.06
    relax_iterations: int = 10
    relax_omega: float = 0.45
    reference_penetration_m: float = 0.01
    friction_mu: float = 0.10
    jitter: float = 0.25
    layer_moment_coupling: bool = True
    subduction: bool = True
    subduction_delay_frames: int = 10
    seed: int = 0
    steady_window_fraction: float = 0.4

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        for name in ("floe_radius_m", "channel_width_over_beam", "drift_speed_m_s",
                     "frames", "advance_per_frame_m", "relax_iterations",
                     "reference_penetration_m", "steady_window_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.relax_omega <= 1.0:
            raise ValueError("relax_omega must be in (0, 1]")
        if not 0.0 <= self.friction_mu < 1.0:
            raise ValueError("friction_mu must be in [0, 1)")

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RunSummary:
    """Numbers a criterion can be evaluated on (no derived prose)."""

    case_id: str
    config: Dict[str, object]
    resistance_mean_kN: float
    resistance_std_kN: float
    resistance_series_kN: List[float]
    break_component_kN: float
    crush_component_kN: float
    friction_component_kN: float
    submersion_closure_kN: float
    drag_closure_kN: float
    total_with_closures_kN: float
    max_frame_mass_error: float
    flux_closure_worst_deviation: float
    flux_closure_ratio: float
    departure_kg: float
    return_kg: float
    departure_vertical_kg: float
    return_vertical_kg: float
    subduction_events: int
    deleted_kg: float
    outflow_kg: float
    initial_mass_kg: float
    n_floes: int
    n_layers: int
    breaking_events: int
    contacts_in_contact_set: float
    opening_open: bool
    floe_radius_m: float
    steady_frames: int

    def as_dict(self) -> Dict[str, object]:
        out = dict(self.__dict__)
        out["resistance_series_kN"] = list(self.resistance_series_kN)
        return out


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
class SlabChannelSim:
    """Layered slab of a level-ice channel with an open lateral boundary."""

    def __init__(self, config: SlabConfig, ice: IceSheetProps,
                 hull: Optional[HullFootprint] = None,
                 case_id: str = "synthetic") -> None:
        self.config = config
        self.ice = ice
        self.hull = hull or HullFootprint()
        self.case_id = case_id
        self.n_layers = config.n_layers
        self.t_layer = ice.thickness_m / self.n_layers
        if config.layer_moment_coupling:
            self.shares = layer_moment_shares(self.n_layers, ice.thickness_m)
        else:  # negative control: independent lamina bending
            self.shares = np.full(self.n_layers, 1.0 / self.n_layers)

        beam = self.hull.beam_m
        self.channel_width_m = config.channel_width_over_beam * beam
        self.half_channel = 0.5 * self.channel_width_m
        self.free_span_m = 0.5 * self.channel_width_m - 0.5 * beam
        self.opening_open = bool(config.lateral_open and self.free_span_m > 0.0)
        band = config.reservoir_band_m if self.opening_open else 0.0
        self.half_total = self.half_channel + band
        self.x_min = -self.hull.length_m - config.downstream_length_m
        self.x_max = config.upstream_length_m

        #: a channel no wider than the beam leaves no room for a wedge in plan
        #: view: the 2-D slice degenerates to a flat pushing wall at the stem
        self.flat_bow = bool(self.channel_width_m <= beam + 1e-9)
        #: the vertical departure face only exists when the thickness is
        #: resolved: a single horizontal slice has no subduction route by
        #: construction (that is what "2-D" means for this model)
        self.subduction_enabled = bool(config.subduction and self.n_layers >= 2)
        self.bend_per_width = ice.bending_per_width_n_per_m
        self._build_field()
        self.mass_ledger = MassLedger(ice_density=ice.ice_density,
                                      initial_mass_kg=self._field_mass())
        self.flux_ledger = FluxLedger(reservoir_mass_0_kg=self._reservoir_mass(),
                                      reservoir_mass_kg=self._reservoir_mass())
        self.breaking_events = 0
        self.subduction_events = 0
        self.contact_set_size: List[float] = []
        self.frame_break: List[float] = []
        self.frame_crush: List[float] = []
        self.frame_friction: List[float] = []
        self.frame_total: List[float] = []

    # -- construction -----------------------------------------------------
    def _build_field(self) -> None:
        cfg, r = self.config, self.config.floe_radius_m
        spacing = 2.0 * r
        rng = np.random.default_rng(cfg.seed)
        pos: List[np.ndarray] = []
        layer_ids: List[np.ndarray] = []
        y0 = -self.half_total + r
        n_y = max(1, int(math.floor(2.0 * self.half_total / spacing)))
        yy = y0 + spacing * np.arange(n_y)
        # the intact sheet fills everything ahead of the hull *including* the
        # wedge region: floes are laid on a lattice and the ones that fall
        # inside the footprint are dropped, so the sheet touches both bow faces
        x0 = -self.hull.bow_length_m
        n_x = max(1, int(math.floor((self.x_max - x0) / spacing)))
        xx = x0 + spacing * np.arange(n_x)
        for k in range(self.n_layers):
            gx, gy = np.meshgrid(xx, yy, indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
            # Column-wise phase jitter: a real floe field is not a lattice, and a
            # lattice makes every column reach the bow in the same frame, which
            # would manufacture a duty cycle for the breaking front instead of
            # resolving it.  Each column gets its own x phase (and a y offset),
            # so the perimeter is loaded continuously.
            n_col = len(xx)
            phase = rng.uniform(0.0, spacing, n_col)
            y_off = rng.uniform(-0.4, 0.4, n_col) * spacing
            col = (np.arange(len(pts)) // len(yy)) % n_col
            pts[:, 0] += phase[col]
            pts[:, 1] += y_off[col]
            pts += rng.uniform(-cfg.jitter, cfg.jitter, pts.shape) * r
            pts[:, 1] = np.clip(pts[:, 1], -self.half_total + r, self.half_total - r)
            if self.flat_bow:
                keep = pts[:, 0] > 0.35 * r
            else:
                dist, _ = self.hull.signed_distance(pts)
                keep = dist > 0.1 * r
            pts = pts[keep]
            pos.append(pts)
            layer_ids.append(np.full(len(pts), k, dtype=int))
        self.pos = np.concatenate(pos, axis=0)
        self.layer_id = np.concatenate(layer_ids)
        self.radius = np.full(len(self.pos), r)
        self.thickness = np.full(len(self.pos), self.t_layer)
        # every floe placed is part of the intact sheet ahead of the stem
        self.intact = np.ones(len(self.pos), dtype=bool)
        self.active = np.ones(len(self.pos), dtype=bool)
        self.in_reservoir = np.zeros(len(self.pos), dtype=bool)
        self.subducted = np.zeros(len(self.pos), dtype=bool)
        self.slide_m = np.zeros(len(self.pos))
        self.transit_left = np.zeros(len(self.pos), dtype=int)
        share = np.asarray([self.shares[k] for k in self.layer_id])
        self.break_capacity = share * self.bend_per_width * 2.0 * self.radius
        self.k_anchor = self._contact_stiffness(self.radius, self.thickness,
                                                self.config.reference_penetration_m)

# -- constitutive helpers --------------------------------------------
    def _contact_stiffness(self, radius: np.ndarray, thickness: np.ndarray,
                           delta: float) -> np.ndarray:
        """Anchor: at ``delta_ref`` the nominal line-contact area crushes at sigma_c."""
        area = 2.0 * thickness * np.sqrt(2.0 * radius * delta)
        return self.ice.compressive_strength_pa * area / delta

    def _crush_force(self, radius: np.ndarray, thickness: np.ndarray,
                     penetration: np.ndarray) -> np.ndarray:
        """Crushing cap ``sigma_c A_c(delta)`` with ``delta`` clipped to ``0.5 r``.

        The clip is the physical ceiling of local crushing: a floe cannot be
        crushed deeper than half its own radius before it stops being a body.
        """
        delta_eff = np.minimum(penetration, 0.5 * radius)
        area = 2.0 * thickness * np.sqrt(2.0 * np.maximum(radius, 1e-9)
                                         * np.maximum(delta_eff, 1e-9))
        return self.ice.compressive_strength_pa * area

    def _mass_of(self, idx: np.ndarray) -> float:
        return float(np.sum(self.ice.ice_density * math.pi * self.radius[idx] ** 2
                            * self.thickness[idx]))

    def _field_mass(self) -> float:
        return self._mass_of(np.flatnonzero(self.active))

    def _reservoir_mass(self) -> float:
        return self._mass_of(np.flatnonzero(self.active & self.in_reservoir))

    # -- contact geometry --------------------------------------------------
    def hull_contacts(self) -> Dict[str, np.ndarray]:
        """Penalty contact of every active floe with the hull footprint.

        When the channel is not wider than the beam (``flat_bow``) the slice has
        no room for a wedge: the footprint is clipped to a flat wall at the stem
        -- exactly what a vertical slice with reflective lateral walls means.
        """
        pos, radius = self.pos, self.radius
        active = self.active & (~self.subducted)
        if self.flat_bow:
            dist = pos[:, 0].copy()
            normal = np.zeros_like(pos)
            normal[:, 0] = 1.0
        else:
            dist, normal = self.hull.signed_distance(pos)
        gap = dist - radius
        penetration = np.where(active, np.clip(-gap, 0.0, None), 0.0)
        contact = penetration > 0.0
        # An intact floe is part of the sheet: it transmits at most its flexural
        # capacity and detaches when that is reached.  A *free* floe has no such
        # ceiling -- once detached it can be crushed locally, and its ceiling is
        # the crushing limit.  That asymmetry is what makes a laterally closed
        # channel expensive: jammed rubble is crushed at sigma_c instead of
        # yielding by bending.
        crush = self._crush_force(radius, self.thickness, np.maximum(penetration, 1e-9))
        if self.subduction_enabled:
            # a slab floe yields by its own bending breakup (subduction removes it
            # from the slice before it can be crushed, see _apply_subduction)
            free_cap = np.minimum(crush, self.break_capacity)
        else:
            # a 2-D slice has no vertical face: the only way out for jammed ice is
            # local crushing, and the crushing load grows with the penetration that
            # cannot be relieved -- this is the term that makes a closed slice
            # explode
            free_cap = crush
        cap = np.where(self.intact, np.minimum(crush, self.break_capacity), free_cap)
        # Rigid-plastic contact: a floe in contact transmits its yield load.  The
        # elastic penetration is *not* the carrier of the load (its magnitude in
        # a projection-relaxed quasi-static solver would be a solver artefact);
        # the resolved quantity is the contact SET and the yield loads, which is
        # standard limit-analysis practice for ice-structure contact.
        # Only a face that the drifting sheet actually runs *into* transmits load:
        # the approach rate of the sheet relative to the footprint is V n_x, so a
        # floe sliding along the parallel sides carries no normal load (only
        # friction), while every contacted bow face is loaded at its yield load.
        driven = contact & (normal[:, 0] > 1e-6)
        n = np.where(driven, cap, 0.0)
        # Coulomb friction with the drift as the only slip direction: in the
        # ship-fixed frame the sheet moves at (-V, 0); with t = (-n_y, n_x) the
        # slip is -V t_x, so the friction on the floe is mu N |t_x| along +x and
        # its reaction adds the same amount to the resistance.
        t_x = -normal[:, 1]
        friction_x = self.config.friction_mu * n * np.abs(t_x)
        force = -n[:, None] * normal
        force[:, 0] += friction_x
        relax = penetration[:, None] * normal
        at_cap = driven & (n > 0.0)
        return {"normal": n, "contact": contact, "driven": driven, "normal_dir": normal,
                "friction_x": friction_x, "force": force, "relax": relax,
                "penetration": penetration, "at_cap": at_cap,
                "resistance_n": n * normal[:, 0] + friction_x}

    def floe_contacts(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Disk-disk repulsion within each layer, evaluated on the free set only.

        Returns ``(payload, relax_delta, load_on_intact)``.  Only pairs with at
        least one *free* floe carry force (two pinned floes belong to the intact
        sheet, whose resistance is carried by the capacity closure).
        ``load_on_intact`` is the x-load that free floes transmit into the sheet;
        it erodes the sheet away from the bow.  A sheet that could never yield
        would turn the field into a rigid wall and build the legacy artefact
        into the model.
        """
        pos, r = self.pos, self.radius
        n_tot = len(pos)
        free = np.flatnonzero(self.active & (~self.intact) & (~self.subducted))
        rest = np.flatnonzero(self.active & (~self.subducted))
        zeros2 = np.zeros((n_tot, 2))
        zeros1 = np.zeros(n_tot)
        if free.size == 0 or rest.size == 0:
            return ({"force": zeros2, "at_cap": np.zeros(n_tot, dtype=bool),
                     "mag": np.zeros((n_tot, n_tot))}, zeros2, zeros1)
        delta = pos[free][:, None, :] - pos[rest][None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        overlap = r[free][:, None] + r[rest][None, :] - dist
        same_layer = self.layer_id[free][:, None] == self.layer_id[rest][None, :]
        not_self = free[:, None] != rest[None, :]
        mask = (overlap > 0.0) & same_layer & not_self
        if not mask.any():
            return ({"force": zeros2, "at_cap": np.zeros(n_tot, dtype=bool),
                     "mag": np.zeros((n_tot, n_tot))}, zeros2, zeros1)
        r_i = r[free][:, None]
        r_j = r[rest][None, :]
        r_eff = r_i * r_j / (r_i + r_j)
        t_eff = 0.5 * (self.thickness[free][:, None] + self.thickness[rest][None, :])
        k = self.ice.compressive_strength_pa * 2.0 * t_eff * np.sqrt(
            2.0 * r_eff * self.config.reference_penetration_m) / self.config.reference_penetration_m
        cap = np.minimum(
            self._crush_force(r_i * np.ones_like(r_j), t_eff,
                              np.clip(overlap, 1e-9, None)),
            np.minimum(self.break_capacity[free][:, None],
                       self.break_capacity[rest][None, :]))
        mag = np.where(mask, np.minimum(k * np.clip(overlap, 0.0, None), cap), 0.0)
        unit = delta / np.maximum(dist, 1e-12)[:, :, None]
        pair_force = mag[:, :, None] * unit
        force_rows = pair_force.sum(axis=1)
        force = zeros2.copy()
        force[free] = force_rows
        overlap_pos = np.where(mask, np.clip(overlap, 0.0, None), 0.0)
        relax = zeros2.copy()
        relax[free] = (overlap_pos[:, :, None] * unit).sum(axis=1)
        intact_j = self.intact[rest]
        load_intact = zeros1.copy()
        load_intact[rest] = np.where(intact_j[None, :],
                                     mag * np.abs(unit[:, :, 0]), 0.0).sum(axis=0)
        at_cap = np.zeros(n_tot, dtype=bool)
        at_cap[rest] = ((mag >= 0.999 * cap) & mask & (mag > 0.0)).any(axis=0)
        return ({"force": force, "at_cap": at_cap, "mag": mag}, relax, load_intact)

    # -- one frame ---------------------------------------------------------
    def step(self, dt: float) -> Dict[str, float]:
        cfg = self.config
        active = self.active
        # 1. the sheet drifts toward the (ship-fixed) hull
        self.pos[active, 0] -= cfg.drift_speed_m_s * dt
        free = active & (~self.intact)
        # 2. non-smooth projection relaxation: displace by the geometric overlap
        #    (well conditioned, no stiffness-limited time step)
        for _ in range(cfg.relax_iterations):
            hull = self.hull_contacts()
            _, relax_pair, _ = self.floe_contacts()
            disp = cfg.relax_omega * (hull["relax"] + relax_pair)
            self.pos += np.where(free[:, None], disp, 0.0)
        hull = self.hull_contacts()
        _, _, load_intact = self.floe_contacts()
        self.contact_set_size.append(float(np.count_nonzero(hull["contact"])))
        # resistance split recorded *before* the detachment update, so that the
        # force carried by an intact floe is booked as breaking, not as crush
        active_now = self.active & (~self.subducted)
        intact_now = hull["contact"] & self.intact & active_now
        free_now = hull["contact"] & (~self.intact) & active_now
        self.frame_break.append(float(np.sum(hull["resistance_n"][intact_now])) / 1e3)
        self.frame_crush.append(float(np.sum(hull["resistance_n"][free_now])) / 1e3)
        self.frame_friction.append(float(np.sum(hull["friction_x"][active_now])) / 1e3)
        self.frame_total.append(float(np.sum(hull["resistance_n"][active_now])) / 1e3)
        # 3. flexural detachment: an intact floe leaves the sheet when the load
        #    it carries (hull + transmitted from the pile) reaches its capacity
        load = hull["normal"] + load_intact
        detach = self.intact & (load >= 0.999 * self.break_capacity) & (load > 0.0)
        n_break = int(np.count_nonzero(detach))
        if n_break:
            self.intact[detach] = False
            self.breaking_events += n_break
        # 4. vertical departure (subduction under the hull) -- only in the slab
        n_sub = self._apply_subduction(hull)
        self.subduction_events += n_sub
        # 5. boundary management: opening / reservoir / legacy clearance
        self._manage_boundaries()
        return {"contacts": float(np.count_nonzero(hull["contact"])),
                "break": float(n_break), "subduct": float(n_sub)}

    def _apply_subduction(self, hull: Dict[str, np.ndarray]) -> int:
        """Push broken floes under the hull: the vertical departure face.

        A free floe held against the bow by a normal load larger than its own
        buoyancy (the force needed to press a floe under the shoulder) is
        submerged.  It leaves the horizontal slice but **not** the books: it is
        parked ``subduction_delay_frames`` frames and re-surfaces aft of the
        shoulder, where it drifts out with the wake.  Mass is therefore
        conserved exactly; the vertical flux is a ledger line item.
        """
        if not self.subduction_enabled:
            return 0
        cfg = self.config
        buoyancy = (WATER_DENSITY * GRAVITY * math.pi * self.radius ** 2
                    * self.thickness)
        touching = self.active & (~self.intact) & (~self.subducted) & hull["driven"]
        # A broken floe is dragged along the bow face and is pressed under the
        # shoulder once it has reached it: the release is geometric (the floe has
        # traversed the entrance length), not a timer, so the residence on the
        # bow is the physical sliding time L_bow / (V cos(theta)) and the loaded
        # front is populated at every resolution.
        cand = touching & (self.pos[:, 0] < -0.80 * self.hull.bow_length_m)
        n = int(np.count_nonzero(cand))
        if n:
            self.subducted[cand] = True
            self.transit_left[cand] = cfg.subduction_delay_frames
            self.mass_ledger.register_departure_vertical(self._mass_of(np.flatnonzero(cand)))
        return n

    def _manage_boundaries(self) -> None:
        cfg = self.config
        active = self.active
        # under-hull FIFO: count down and re-surface aft of the shoulder
        transit = self.subducted & active
        if transit.any():
            self.transit_left[transit] -= 1
            release = transit & (self.transit_left <= 0)
            if release.any():
                idx = np.flatnonzero(release)
                self.subducted[idx] = False
                half = 0.5 * self.channel_width_m - self.radius[idx]
                self.pos[idx, 0] = self.x_min + 1.5 * self.radius[idx]
                self.pos[idx, 1] = np.clip(self.pos[idx, 1], -half, half)
                self.mass_ledger.register_return_vertical(self._mass_of(idx))
        free = active & (~self.intact) & (~self.subducted)
        if self.opening_open:
            leave = free & (~self.in_reservoir) & (np.abs(self.pos[:, 1]) > self.half_channel)
            if leave.any():
                self.mass_ledger.register_departure(self._mass_of(np.flatnonzero(leave)))
                self.in_reservoir[leave] = True
            back = free & self.in_reservoir & (np.abs(self.pos[:, 1]) <= self.half_channel)
            if back.any():
                self.mass_ledger.register_return(self._mass_of(np.flatnonzero(back)))
                self.in_reservoir[back] = False
            limit = self.half_total - self.radius
        else:
            limit = self.half_channel - self.radius
        over = active & (~self.subducted) & (np.abs(self.pos[:, 1]) > limit)
        if over.any():
            self.pos[over, 1] = np.sign(self.pos[over, 1]) * limit[over]
        # downstream loss: the physical outflow through the aft boundary
        out = active & (~self.subducted) & (self.pos[:, 0] < self.x_min)
        if out.any():
            self.mass_ledger.register_outflow(self._mass_of(np.flatnonzero(out)))
            self.active[out] = False
        # legacy instant clearance: delete everything in the ship wake channel
        if cfg.instant_removal:
            wake = self.active & (self.pos[:, 0] < -0.5 * self.hull.length_m) & (
                np.abs(self.pos[:, 1]) < 0.5 * self.hull.beam_m)
            if wake.any():
                self.mass_ledger.register_deletion(self._mass_of(np.flatnonzero(wake)))
                self.active[wake] = False

    # -- driver ------------------------------------------------------------
    def run(self) -> RunSummary:
        cfg = self.config
        dt = cfg.advance_per_frame_m / cfg.drift_speed_m_s
        series: List[float] = []
        break_force: List[float] = []
        crush_force: List[float] = []
        friction_series: List[float] = []
        for _ in range(cfg.frames):
            self.step(dt)
            series.append(self.frame_total[-1])
            break_force.append(self.frame_break[-1])
            crush_force.append(self.frame_crush[-1])
            friction_series.append(self.frame_friction[-1])
            self.mass_ledger.record_frame(self._field_mass())
            self.flux_ledger.reservoir_mass_kg = self._reservoir_mass()
            self.flux_ledger.record(self.mass_ledger.departure_kg,
                                    self.mass_ledger.return_kg)
        arr = np.asarray(series)
        n_steady = max(1, int(round(cfg.steady_window_fraction * len(arr))))
        steady = arr[-n_steady:]
        closure = self.flux_ledger.samples[-1] if self.flux_ledger.samples else 1.0
        h, B, V = self.ice.thickness_m, self.hull.beam_m, cfg.drift_speed_m_s
        submersion = (WATER_DENSITY * GRAVITY * h ** 2 * B
                      * (SUBMERSION_BUOYANCY
                         + SUBMERSION_FRICTION * cfg.friction_mu * self.hull.length_m / B)) / 1e3
        drag = 0.5 * WATER_DENSITY * B * h * V ** 2 / 1e3
        mean_r = float(steady.mean())
        return RunSummary(
            case_id=self.case_id,
            config=cfg.as_dict(),
            resistance_mean_kN=mean_r,
            resistance_std_kN=float(steady.std()),
            resistance_series_kN=series,
            break_component_kN=float(np.mean(break_force[-n_steady:])),
            crush_component_kN=float(np.mean(crush_force[-n_steady:])),
            friction_component_kN=float(np.mean(friction_series[-n_steady:])),
            submersion_closure_kN=float(submersion),
            drag_closure_kN=float(drag),
            total_with_closures_kN=float(mean_r + submersion + drag),
            max_frame_mass_error=float(self.mass_ledger.max_frame_error),
            flux_closure_worst_deviation=float(self.flux_ledger.worst_deviation),
            flux_closure_ratio=float(closure),
            departure_kg=float(self.mass_ledger.departure_kg),
            return_kg=float(self.mass_ledger.return_kg),
            departure_vertical_kg=float(self.mass_ledger.departure_vertical_kg),
            return_vertical_kg=float(self.mass_ledger.return_vertical_kg),
            subduction_events=int(self.subduction_events),
            deleted_kg=float(self.mass_ledger.deleted_kg),
            outflow_kg=float(self.mass_ledger.outflow_kg),
            initial_mass_kg=float(self.mass_ledger.initial_mass_kg),
            n_floes=int(np.count_nonzero(self.active)),
            n_layers=self.n_layers,
            breaking_events=int(self.breaking_events),
            contacts_in_contact_set=float(np.mean(self.contact_set_size[-n_steady:])),
            opening_open=bool(self.opening_open),
            floe_radius_m=float(cfg.floe_radius_m),
            steady_frames=int(n_steady),
        )