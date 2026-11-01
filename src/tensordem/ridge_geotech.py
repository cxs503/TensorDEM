"""Geotechnical and structural building blocks for ice-ridge / pipeline grounding.

This module is the *continuum and structural* side of the ridge-grounding
workflow implemented in :mod:`tensordem.ridge_dem` and orchestrated in
:mod:`tensordem.ridge_grounding`:

* ``MohrCoulombSoil`` -- effective-stress Mohr-Coulomb (or undrained total
  stress) subgrade with an initial elastic stiffness, a hyperbolic
  (elastic-plastic) vertical reaction curve and a lateral limit pressure.
  Bearing-capacity factors follow Vesic (1973); undrained embedment follows
  Skempton (1951); lateral limit pressure uses the usual pipe-soil practice of
  an embedment-dependent bearing factor ``N_p = 2 + 1.5 z/D`` capped at 9.
* ``PipeSection`` / ``dnv_os_f101_utilization`` -- DNV-OS-F101 style plastic
  capacities ``M_p = f_y (D-t)^2 t`` and ``N_p = f_y pi (D-t) t`` plus the
  load-controlled combined-loading interaction check.
* ``solve_beam_on_winkler`` -- Euler-Bernoulli beam on a Winkler foundation with
  consistent Hermitian soil stiffness, together with the closed-form Hetenyi
  (1946) infinite-beam solutions used as the independent reference.

All quantities are SI (m, s, N, Pa).  Nothing here is DEM: every function is a
closed-form or finite-element continuum model, and each carries an analytic
limit that the test-suite re-checks.

Scope limits (deliberate; see ``docs/ice_ridge_pipeline_grounding.md``): the
Winkler subgrade modulus is a chosen elastic model (``c_e E_s / ((1-nu^2) B)``),
not a derived value, so soil stiffness scales displacements much more strongly
than capacities; capacities are the verified quantities.  Pipeline buckling,
cyclic degradation, trench geometry and pipe-soil gapping are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

__all__ = [
    "GRAVITY",
    "MohrCoulombSoil",
    "SofteningVerticalSpring",
    "PipeSection",
    "bearing_capacity_factors",
    "dnv_os_f101_utilization",
    "hetenyi_infinite_beam_patch_load",
    "hetenyi_infinite_beam_point_load",
    "hetenyi_verification",
    "solve_beam_on_winkler",
]

GRAVITY = 9.81
"""Standard gravity, m/s^2."""


def _require_finite(name: str, value: float, *, positive: bool = False,
                    nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def bearing_capacity_factors(phi_rad: float) -> tuple[float, float, float]:
    """Vesic (1973) bearing-capacity factors ``(N_c, N_q, N_gamma)``.

    ``N_q = exp(pi tan phi) tan^2(45 + phi/2)``,
    ``N_c = (N_q - 1) / tan phi`` with the Prandtl limit ``N_c = 5.14`` at
    ``phi = 0``, and ``N_gamma = 2 (N_q + 1) tan phi``.
    """
    _require_finite("phi_rad", phi_rad)
    if phi_rad < 0.0 or phi_rad >= 0.5 * math.pi:
        raise ValueError("phi_rad must lie in [0, pi/2)")
    if phi_rad < 1e-9:
        return 5.14, 1.0, 0.0
    tan_phi = math.tan(phi_rad)
    n_q = math.exp(math.pi * tan_phi) * math.tan(0.25 * math.pi + 0.5 * phi_rad) ** 2
    n_c = (n_q - 1.0) / tan_phi
    n_gamma = 2.0 * (n_q + 1.0) * tan_phi
    return n_c, n_q, n_gamma


@dataclass(frozen=True)
class MohrCoulombSoil:
    """Effective-stress Mohr-Coulomb subgrade beneath a subsea pipeline.

    ``friction_angle_deg`` and ``cohesion`` are the *drained effective* strength
    parameters ``(phi', c')``.  ``undrained=True`` switches to a total-stress
    phi = 0 model in which ``cohesion`` is read as the undrained shear strength
    ``s_u``.  ``pore_pressure_ratio`` reduces the effective unit weight by
    ``(1 - r_u)``, i.e. a simple partially-drained pore-pressure model rather
    than a coupled seepage solution.
    """

    friction_angle_deg: float = 30.0
    cohesion: float = 5.0e3
    saturated_unit_weight: float = 18.0e3
    young_modulus: float = 10.0e6
    poisson_ratio: float = 0.3
    water_unit_weight: float = 10.06e3
    pore_pressure_ratio: float = 0.0
    undrained: bool = False
    subgrade_modulus_factor: float = 0.65

    def __post_init__(self) -> None:
        _require_finite("friction_angle_deg", self.friction_angle_deg, nonnegative=True)
        if self.friction_angle_deg >= 90.0:
            raise ValueError("friction_angle_deg must be < 90")
        _require_finite("cohesion", self.cohesion, nonnegative=True)
        _require_finite("saturated_unit_weight", self.saturated_unit_weight, positive=True)
        _require_finite("young_modulus", self.young_modulus, positive=True)
        _require_finite("poisson_ratio", self.poisson_ratio, nonnegative=True)
        if self.poisson_ratio >= 0.5:
            raise ValueError("poisson_ratio must be < 0.5")
        _require_finite("water_unit_weight", self.water_unit_weight, positive=True)
        _require_finite("pore_pressure_ratio", self.pore_pressure_ratio, nonnegative=True)
        if self.pore_pressure_ratio > 1.0:
            raise ValueError("pore_pressure_ratio must be <= 1")
        _require_finite("subgrade_modulus_factor", self.subgrade_modulus_factor, positive=True)
        if not isinstance(self.undrained, bool):
            raise ValueError("undrained must be a boolean")
        if self.undrained and self.friction_angle_deg != 0.0:
            raise ValueError("undrained (phi = 0) model requires friction_angle_deg == 0")

    @property
    def phi_rad(self) -> float:
        return math.radians(self.friction_angle_deg)

    @property
    def effective_unit_weight(self) -> float:
        """Submerged unit weight ``gamma' = gamma_sat - gamma_w`` in N/m^3."""
        return self.saturated_unit_weight - self.water_unit_weight

    @property
    def shear_strength(self) -> float:
        """Undrained shear strength ``s_u`` (= ``cohesion`` in total stress)."""
        return self.cohesion

    def effective_vertical_stress(self, depth: float) -> float:
        """Effective vertical stress (Pa) ``depth`` below the mudline."""
        _require_finite("depth", depth, nonnegative=True)
        gamma_prime = self.effective_unit_weight
        if gamma_prime <= 0.0:
            raise ValueError("soil must have a positive effective unit weight")
        return gamma_prime * (1.0 - self.pore_pressure_ratio) * depth

    def vertical_pressure_capacity(self, width: float, depth_below_mudline: float) -> float:
        """Ultimate bearing pressure (Pa) for a footing of projected ``width``.

        Drained: ``q = c' N_c d_c + sigma'_v N_q d_q + 0.5 gamma' B N_gamma``.
        Undrained: ``q = s_u N_c d_c`` with Skempton ``N_c = 5.14 (1 + 0.2 z/B)``
        capped at 9.  Depth factors use Meyerhof/Hansen ``k = min(z/B, 1)``.
        """
        _require_finite("width", width, positive=True)
        _require_finite("depth_below_mudline", depth_below_mudline, nonnegative=True)
        phi = self.phi_rad
        k = min(depth_below_mudline / width, 1.0)
        if self.undrained:
            n_c = min(5.14 * (1.0 + 0.2 * depth_below_mudline / width), 9.0)
            return self.shear_strength * n_c * (1.0 + 0.2 * k)
        n_c, n_q, n_gamma = bearing_capacity_factors(phi)
        d_c = 1.0 + 0.4 * k
        d_q = 1.0 + 2.0 * math.tan(phi) * (1.0 - math.sin(phi)) ** 2 * k
        sigma_v = self.effective_vertical_stress(depth_below_mudline)
        gamma_prime = self.effective_unit_weight * (1.0 - self.pore_pressure_ratio)
        return (self.cohesion * n_c * d_c
                + sigma_v * n_q * d_q
                + 0.5 * gamma_prime * width * n_gamma)

    def vertical_capacity_per_length(self, diameter: float, embedment: float) -> float:
        """Ultimate vertical resistance per unit pipe length (N/m).

        A buried cylinder is treated as a strip footing of projected width ``D``
        with its base at the pipe invert, ``embedment + D`` below the mudline.
        """
        _require_finite("diameter", diameter, positive=True)
        _require_finite("embedment", embedment, nonnegative=True)
        return self.vertical_pressure_capacity(diameter, embedment + diameter) * diameter

    def subgrade_modulus(self, width: float) -> float:
        """Elastic subgrade modulus ``k_v`` in N/m^3 (see class docstring)."""
        _require_finite("width", width, positive=True)
        return (self.subgrade_modulus_factor * self.young_modulus
                / ((1.0 - self.poisson_ratio ** 2) * width))

    def initial_vertical_stiffness(self, diameter: float) -> float:
        """Initial slope ``dp/dw`` of the vertical reaction curve (N/m per m)."""
        _require_finite("diameter", diameter, positive=True)
        return self.subgrade_modulus(diameter) * diameter

    def vertical_reaction(self, displacement: float, diameter: float, embedment: float) -> float:
        """Hyperbolic vertical reaction per unit length (N/m) at ``displacement``."""
        _require_finite("displacement", displacement)
        if displacement < 0.0:
            raise ValueError("vertical_reaction requires displacement >= 0")
        p_ult = self.vertical_capacity_per_length(diameter, embedment)
        k_v = self.initial_vertical_stiffness(diameter)
        if p_ult <= 0.0:
            return 0.0
        return displacement / (1.0 / k_v + displacement / p_ult)

    def displacement_for_load(self, load: float, diameter: float, embedment: float) -> float:
        """Inverse of :meth:`vertical_reaction`; ``inf`` once capacity is exhausted."""
        _require_finite("load", load, nonnegative=True)
        p_ult = self.vertical_capacity_per_length(diameter, embedment)
        if p_ult <= 0.0:
            return 0.0
        if load >= p_ult:
            return math.inf
        k_v = self.initial_vertical_stiffness(diameter)
        return load / (k_v * (1.0 - load / p_ult))

    def lateral_limit_pressure(self, embedment: float, diameter: float) -> float:
        """Lateral limit line load (N/m) on the buried pipe at ``embedment``.

        Undrained: ``N_p s_u D`` with ``N_p = min(2 + 1.5 z/D, 9)``.
        Drained: Coulomb passive wedge ``0.5 gamma' z^2 K_p + 2 c' z sqrt(K_p)``
        at the pipe centreline depth ``z = embedment + D/2``.
        """
        _require_finite("embedment", embedment, nonnegative=True)
        _require_finite("diameter", diameter, positive=True)
        if embedment <= 0.0:
            return 0.0
        z = embedment + 0.5 * diameter
        if self.undrained:
            n_p = min(2.0 + 1.5 * z / diameter, 9.0)
            return n_p * self.shear_strength * diameter
        k_p = math.tan(0.25 * math.pi + 0.5 * self.phi_rad) ** 2
        gamma_prime = self.effective_unit_weight * (1.0 - self.pore_pressure_ratio)
        return (0.5 * gamma_prime * z ** 2 * k_p
                + 2.0 * self.cohesion * z * math.sqrt(k_p))


class SofteningVerticalSpring:
    """Peak-to-residual vertical subgrade spring with memory (pipe-soil coupling).

    The hyperbolic Mohr-Coulomb reaction of :meth:`MohrCoulombSoil.vertical_reaction`
    is monotone and asymptotic, so on its own it can never produce the
    peak-then-residual shape that pipe-soil practice (DNV-RP-F114 / DNV-RP-F105
    style peak and residual resistance) and ice load-displacement curves
    (ISO 19906 style) both show.  This spring adds the documented post-peak
    branch:

    * ``w <= w_yield``: the Mohr-Coulomb hyperbolic reaction;
    * ``w > w_yield``: linear softening from the peak to ``residual_ratio`` of
      the peak over ``softening_length``;
    * beyond that: the residual plateau.

    ``w_yield`` is the displacement at which the hyperbolic reaction first
    reaches ``yield_ratio`` of the ultimate bearing capacity, so the peak of the
    spring is ``yield_ratio * p_ult`` by construction.  The softening length is
    taken as ``softening_length_factor * D``; both are explicit modelling
    choices, not calibrated values, and are recorded in every artifact.
    """

    def __init__(self, soil: MohrCoulombSoil, diameter: float, embedment: float, *,
                 yield_ratio: float = 0.90, residual_ratio: float = 0.80,
                 softening_length_factor: float = 0.5):
        if not isinstance(soil, MohrCoulombSoil):
            raise TypeError("soil must be a MohrCoulombSoil")
        _require_finite("diameter", diameter, positive=True)
        _require_finite("embedment", embedment, nonnegative=True)
        if not 0.0 < yield_ratio < 1.0:
            raise ValueError("yield_ratio must lie in (0, 1)")
        if not 0.0 <= residual_ratio <= 1.0:
            raise ValueError("residual_ratio must lie in [0, 1]")
        _require_finite("softening_length_factor", softening_length_factor, positive=True)
        object.__setattr__ if False else None
        self.soil = soil
        self.diameter = float(diameter)
        self.embedment = float(embedment)
        self.yield_ratio = float(yield_ratio)
        self.residual_ratio = float(residual_ratio)
        self.softening_length = softening_length_factor * float(diameter)
        self.ultimate = soil.vertical_capacity_per_length(diameter, embedment)
        stiffness = soil.initial_vertical_stiffness(diameter)
        # displacement at which the hyperbolic curve reaches the yield fraction
        self.yield_displacement = (yield_ratio * self.ultimate
                                   / (stiffness * (1.0 - yield_ratio)))
        self.peak = yield_ratio * self.ultimate
        self.residual = residual_ratio * self.peak
        self.max_displacement = 0.0

    def reaction(self, displacement: float) -> float:
        """Vertical reaction per unit length (N/m), peak-then-residual."""
        _require_finite("displacement", displacement)
        if displacement <= 0.0:
            return 0.0
        self.max_displacement = max(self.max_displacement, displacement)
        hyperbolic = self.soil.vertical_reaction(displacement, self.diameter, self.embedment)
        if displacement <= self.yield_displacement:
            return hyperbolic
        excess = (displacement - self.yield_displacement) / self.softening_length
        factor = max(self.residual_ratio, 1.0 - (1.0 - self.residual_ratio) * excess)
        return self.peak * factor

    def stiffness(self, displacement: float) -> float:
        """Secant stiffness ``p / w`` (N/m per m) for a nonlinear beam solve."""
        if displacement <= 0.0:
            stiffness = self.soil.initial_vertical_stiffness(self.diameter)
            return stiffness
        return max(self.reaction(displacement), 0.0) / displacement

    def describe(self) -> dict:
        return {
            "ultimate_bearing_capacity": self.ultimate,
            "peak_reaction": self.peak,
            "yield_displacement": self.yield_displacement,
            "residual_reaction": self.residual,
            "softening_length": self.softening_length,
            "yield_ratio": self.yield_ratio,
            "residual_ratio": self.residual_ratio,
        }


@dataclass(frozen=True)
class PipeSection:
    """Geometric and plastic section properties of a line pipe."""

    outer_diameter: float
    wall_thickness: float
    young_modulus: float
    yield_strength: float

    def __post_init__(self) -> None:
        _require_finite("outer_diameter", self.outer_diameter, positive=True)
        _require_finite("wall_thickness", self.wall_thickness, positive=True)
        _require_finite("young_modulus", self.young_modulus, positive=True)
        _require_finite("yield_strength", self.yield_strength, positive=True)
        if 2.0 * self.wall_thickness >= self.outer_diameter:
            raise ValueError("wall_thickness must satisfy 2t < D")

    @property
    def inner_diameter(self) -> float:
        return self.outer_diameter - 2.0 * self.wall_thickness

    @property
    def area(self) -> float:
        """Steel area, m^2."""
        return math.pi * self.wall_thickness * (self.outer_diameter - self.wall_thickness)

    @property
    def inertia(self) -> float:
        """Second moment of area, m^4."""
        return math.pi / 64.0 * (self.outer_diameter ** 4 - self.inner_diameter ** 4)

    @property
    def elastic_section_modulus(self) -> float:
        """Elastic section modulus ``W_el = 2I/D``, m^3."""
        return 2.0 * self.inertia / self.outer_diameter

    @property
    def plastic_moment(self) -> float:
        """DNV-OS-F101 plastic moment capacity ``M_p = f_y (D-t)^2 t``, N m."""
        return (self.yield_strength * (self.outer_diameter - self.wall_thickness) ** 2
                * self.wall_thickness)

    @property
    def plastic_axial_force(self) -> float:
        """DNV-OS-F101 plastic axial capacity ``N_p = f_y pi (D-t) t``, N."""
        return (self.yield_strength * math.pi
                * (self.outer_diameter - self.wall_thickness) * self.wall_thickness)

    def bending_stress(self, moment: float) -> float:
        """Extreme-fibre bending stress (Pa) for bending moment ``moment`` (N m)."""
        _require_finite("moment", moment)
        return moment / self.elastic_section_modulus

    def axial_stress(self, force: float) -> float:
        """Uniform axial stress (Pa) for axial force ``force`` (N)."""
        _require_finite("force", force)
        return force / self.area


def dnv_os_f101_utilization(moment: float, axial_force: float, section: PipeSection,
                            *, alpha_c: float = 0.85) -> float:
    """Load-controlled combined-loading utilization per DNV-OS-F101.

    ``U = M / (alpha_c M_p) + (N / (alpha_c N_p))^2``; the section is acceptable
    when ``U <= 1``.  ``alpha_c`` is a code coefficient supplied as input (0.85
    is the usual load-controlled value in normal safety class), not a derived
    quantity of this model.
    """
    _require_finite("moment", moment)
    _require_finite("axial_force", axial_force)
    _require_finite("alpha_c", alpha_c, positive=True)
    if alpha_c > 1.0:
        raise ValueError("alpha_c must be <= 1")
    return (abs(moment) / (alpha_c * section.plastic_moment)
            + (axial_force / (alpha_c * section.plastic_axial_force)) ** 2)


def _hetenyi_a(bx: float) -> float:
    return math.exp(-bx) * (math.cos(bx) + math.sin(bx))


def _hetenyi_c(bx: float) -> float:
    return math.exp(-bx) * (math.cos(bx) - math.sin(bx))


def hetenyi_infinite_beam_point_load(x, *, load: float, beta: float, k: float):
    """Hetenyi (1946) infinite beam on elastic foundation under a point load.

    ``beta = (k / (4 EI))^(1/4)``, ``k`` in N/m^2 (foundation modulus times
    width).  Returns ``(w, M)`` arrays: deflection (m) with downward positive
    and bending moment (N m) with sagging positive at ``M(0) = P/(4 beta)``.
    """
    if not isinstance(load, float) and not isinstance(load, int):
        raise TypeError("load must be a real number")
    _require_finite("beta", beta, positive=True)
    _require_finite("k", k, positive=True)
    x_t = torch.as_tensor(x, dtype=torch.float64)
    bx = beta * x_t.abs()
    deflection = load * beta / (2.0 * k) * torch.exp(-bx) * (torch.cos(bx) + torch.sin(bx))
    moment = load / (4.0 * beta) * torch.exp(-bx) * (torch.cos(bx) - torch.sin(bx))
    return deflection, moment


def hetenyi_infinite_beam_patch_load(x, *, load_per_length: float, half_length: float,
                                     beta: float, k: float, samples: int = 401):
    """Hetenyi infinite beam under a uniform patch load of half-length ``a``.

    Implemented as the exact superposition integral of the point-load kernel,
    ``w(x) = int_{-a}^{a} q0 beta/(2k) A(beta|x-s|) ds`` (and likewise for
    ``M``), evaluated with the midpoint rule on a fine grid -- an independent
    reference that shares no code with the finite-element solver.
    """
    _require_finite("load_per_length", load_per_length)
    _require_finite("half_length", half_length, positive=True)
    _require_finite("beta", beta, positive=True)
    _require_finite("k", k, positive=True)
    if not isinstance(samples, int) or samples < 20:
        raise ValueError("samples must be an integer >= 20")
    x_t = torch.as_tensor(x, dtype=torch.float64)
    ds = 2.0 * half_length / samples
    s = -half_length + ds * (torch.arange(samples, dtype=torch.float64) + 0.5)
    bx = beta * (x_t[:, None] - s[None, :]).abs()
    a = torch.exp(-bx) * (torch.cos(bx) + torch.sin(bx))
    c = torch.exp(-bx) * (torch.cos(bx) - torch.sin(bx))
    deflection = load_per_length * beta / (2.0 * k) * a.sum(dim=1) * ds
    moment = load_per_length / (4.0 * beta) * c.sum(dim=1) * ds
    return deflection, moment


def solve_beam_on_winkler(loads, *, inertia: float, young_modulus: float,
                          length: float, node_count: int, foundation_modulus,
                          end_condition: str = "infinite", end_spring: float | None = None,
                          max_iterations: int = 60, tolerance: float = 1e-9):
    """Euler-Bernoulli beam on a Winkler foundation by finite elements.

    ``loads`` is the line load (N/m) sampled on ``node_count`` equally spaced
    nodes over ``[0, length]`` (downward positive).  ``foundation_modulus`` is
    either a scalar subgrade modulus per unit length (N/m^2) or a callable
    ``k(w) -> (k_secant, p)`` returning the secant stiffness and the nonlinear
    reaction; nonlinear problems are solved by secant iteration.

    ``end_condition = "infinite"`` adds the exact semi-infinite boundary
    stiffnesses ``2 EI beta^3`` (translation) and ``4 EI beta`` (rotation) with
    ``beta = (k/(4 EI))**(1/4)``, so a long finite model reproduces the Hetenyi
    infinite-beam solution; ``"pinned"`` fixes the deflection at both ends;
    ``"free"`` leaves the ends free.

    Bending moment and shear are evaluated from the Hermite interpolation of the
    converged nodal solution as ``M = -EI wxx`` and ``V = -EI wxxx`` (sagging
    moment positive, downward deflection positive), one sign convention for the
    whole module.

    Returns a dict with ``x``, ``deflection`` (m), ``moment`` (N m), ``shear``
    (N), ``reaction`` (N/m), ``soil_modulus``, ``iterations``, ``converged`` and
    ``element_length``.
    """
    if not isinstance(node_count, int) or node_count < 3:
        raise ValueError("node_count must be an integer >= 3")
    _require_finite("inertia", inertia, positive=True)
    _require_finite("young_modulus", young_modulus, positive=True)
    _require_finite("length", length, positive=True)
    if end_condition not in ("infinite", "pinned", "free"):
        raise ValueError("end_condition must be 'infinite', 'pinned' or 'free'")
    loads_t = torch.as_tensor(loads, dtype=torch.float64).reshape(-1)
    if loads_t.numel() != node_count:
        raise ValueError("loads must have node_count entries")
    if not bool(torch.isfinite(loads_t).all()):
        raise ValueError("loads must be finite")

    h = length / (node_count - 1)
    ei = young_modulus * inertia
    n_dof = 2 * node_count

    ke = ei / h ** 3 * torch.tensor(
        [[12.0, 6.0 * h, -12.0, 6.0 * h],
         [6.0 * h, 4.0 * h * h, -6.0 * h, 2.0 * h * h],
         [-12.0, -6.0 * h, 12.0, -6.0 * h],
         [6.0 * h, 2.0 * h * h, -6.0 * h, 4.0 * h * h]], dtype=torch.float64)

    gauss_xi = (-math.sqrt(0.6), 0.0, math.sqrt(0.6))
    gauss_w = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)

    def hermite(xi: float):
        n1 = 0.25 * (1 - xi) ** 2 * (2 + xi)
        n2 = 0.125 * h * (1 - xi) ** 2 * (1 + xi)
        n3 = 0.25 * (1 + xi) ** 2 * (2 - xi)
        n4 = -0.125 * h * (1 + xi) ** 2 * (1 - xi)
        return torch.tensor([n1, n2, n3, n4], dtype=torch.float64)

    soil_unit = torch.zeros(4, 4, dtype=torch.float64)
    for xi, weight in zip(gauss_xi, gauss_w):
        shape = hermite(xi)
        soil_unit += weight * torch.outer(shape, shape)
    soil_unit *= h / 2.0

    nodal_load = loads_t * h
    nodal_load[0] *= 0.5
    nodal_load[-1] *= 0.5
    f = torch.zeros(n_dof, dtype=torch.float64)
    f[0::2] = nodal_load

    support = torch.zeros(n_dof, dtype=torch.float64)
    fixed = torch.empty(0, dtype=torch.long)
    if end_condition == "infinite":
        if isinstance(foundation_modulus, (int, float)):
            k_ref = float(foundation_modulus)
        elif end_spring is not None:
            k_ref = float(end_spring)
        else:
            raise ValueError("end_condition='infinite' needs a scalar foundation_modulus")
        beta = (k_ref / (4.0 * ei)) ** 0.25
        rigid = 1.0e4 * k_ref
        support[0] = rigid
        support[-2] = rigid
        support[1] = rigid * h
        support[-1] = rigid * h
    elif end_condition == "pinned":
        fixed = torch.tensor([0, 2 * (node_count - 1)], dtype=torch.long)

    free = torch.ones(n_dof, dtype=torch.bool)
    free[fixed] = False
    free_idx = torch.arange(n_dof)[free]

    deflection = torch.zeros(node_count, dtype=torch.float64)
    k_soil = torch.zeros(node_count, dtype=torch.float64)
    rotations = torch.zeros(node_count, dtype=torch.float64)
    iterations = 0
    converged = False
    for _ in range(max_iterations):
        iterations += 1
        if callable(foundation_modulus):
            k_secant, _reaction = foundation_modulus(deflection)
            k_soil = torch.as_tensor(k_secant, dtype=torch.float64).reshape(-1)
            if k_soil.numel() != node_count:
                raise ValueError("foundation_modulus callback must return node_count values")
        else:
            k_soil = torch.full((node_count,), float(foundation_modulus), dtype=torch.float64)

        stiffness = torch.zeros(n_dof, n_dof, dtype=torch.float64)
        for element in range(node_count - 1):
            rows = torch.tensor([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
            stiffness[rows[:, None], rows[None, :]] += ke
            stiffness[rows[:, None], rows[None, :]] += (
                soil_unit * 0.5 * (k_soil[element] + k_soil[element + 1]))
        stiffness[torch.arange(n_dof), torch.arange(n_dof)] += support

        red = stiffness[free_idx[:, None], free_idx[None, :]]
        solution = torch.linalg.solve(red, f[free_idx])
        updated = torch.zeros(n_dof, dtype=torch.float64)
        updated[free_idx] = solution
        new_deflection = updated[0::2]
        change = float(torch.max(torch.abs(new_deflection - deflection)))
        deflection = new_deflection
        rotations = updated[1::2]
        if not callable(foundation_modulus) or change < tolerance:
            converged = True
            break

    moment, shear = _hermite_moment_shear(deflection, rotations, h, ei)
    x = torch.linspace(0.0, length, node_count, dtype=torch.float64)
    return {
        "x": x,
        "deflection": deflection,
        "moment": moment,
        "shear": shear,
        "reaction": k_soil * deflection,
        "soil_modulus": k_soil,
        "iterations": iterations,
        "converged": converged,
        "element_length": h,
    }


def _hermite_moment_shear(deflection: torch.Tensor, rotation: torch.Tensor,
                          h: float, ei: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodal bending moment and shear from Hermite interpolation of the solution."""
    node_count = int(deflection.numel())
    scale2 = (2.0 / h) ** 2
    scale3 = (2.0 / h) ** 3
    curvature_left = torch.tensor([-1.5, -0.5 * h, 1.5, -0.5 * h], dtype=torch.float64)
    curvature_right = torch.tensor([1.5, 0.5 * h, -1.5, 0.5 * h], dtype=torch.float64)
    third_left = torch.tensor([3.0, 1.5 * h, -3.0, 1.5 * h], dtype=torch.float64)
    third_right = torch.tensor([-3.0, 1.5 * h, 3.0, 1.5 * h], dtype=torch.float64)
    w2 = torch.zeros(node_count, dtype=torch.float64)
    w3 = torch.zeros(node_count, dtype=torch.float64)
    count = torch.zeros(node_count, dtype=torch.float64)
    for element in range(node_count - 1):
        d = torch.stack((deflection[element], rotation[element],
                         deflection[element + 1], rotation[element + 1]))
        w2[element] += scale2 * float(curvature_left @ d)
        w2[element + 1] += scale2 * float(curvature_right @ d)
        w3[element] += scale3 * float(third_left @ d)
        w3[element + 1] += scale3 * float(third_right @ d)
        count[element] += 1.0
        count[element + 1] += 1.0
    return -ei * w2 / count, -ei * w3 / count


def hetenyi_verification(*, inertia: float = 0.0123, young_modulus: float = 2.1e11,
                         foundation_modulus: float = 5.0e6, load: float = 1.0e6,
                         patch_half_length: float | None = None,
                         elements_per_beta: float = 12.0,
                         half_span_betas: float = 12.0) -> dict:
    """Compare the finite-element Winkler beam with the closed-form Hetenyi results.

    Returns peak deflection / moment from both models and their relative
    differences.  The FE mesh is refined in units of ``1/beta`` so the error is
    dominated by discretisation of the load, not by mesh coarseness.
    """
    ei = young_modulus * inertia
    beta = (foundation_modulus / (4.0 * ei)) ** 0.25
    half_span = half_span_betas / beta
    node_count = int(2 * half_span_betas * elements_per_beta) + 1
    length = 2.0 * half_span
    x = torch.linspace(0.0, length, node_count, dtype=torch.float64)
    centre = 0.5 * length
    nodal = torch.zeros(node_count, dtype=torch.float64)
    if patch_half_length is None:
        nearest = int(torch.argmin(torch.abs(x - centre)))
        element_length = length / (node_count - 1)
        nodal[nearest] = load / element_length
        w_ref, m_ref = hetenyi_infinite_beam_point_load(
            x - centre, load=load, beta=beta, k=foundation_modulus)
    else:
        inside = torch.abs(x - centre) <= patch_half_length + 1e-12
        nodal[inside] = load
        w_ref, m_ref = hetenyi_infinite_beam_patch_load(
            x - centre, load_per_length=load, half_length=patch_half_length,
            beta=beta, k=foundation_modulus)
    fe = solve_beam_on_winkler(nodal, inertia=inertia, young_modulus=young_modulus,
                               length=length, node_count=node_count,
                               foundation_modulus=foundation_modulus,
                               end_condition="infinite")
    w_fe = fe["deflection"]
    m_fe = fe["moment"]
    w_peak_fe = float(w_fe.max())
    m_peak_fe = float(m_fe.max())
    w_peak_ref = float(w_ref.max())
    m_peak_ref = float(m_ref.max())
    return {
        "beta": beta,
        "node_count": node_count,
        "element_length": fe["element_length"],
        "deflection_fe": w_peak_fe,
        "deflection_reference": w_peak_ref,
        "deflection_relative_error": abs(w_peak_fe - w_peak_ref) / abs(w_peak_ref),
        "moment_fe": m_peak_fe,
        "moment_reference": m_peak_ref,
        "moment_relative_error": abs(m_peak_fe - m_peak_ref) / abs(m_peak_ref),
        "profile_deflection_relative_error": float(
            torch.max(torch.abs(w_fe - w_ref)) / torch.max(torch.abs(w_ref))),
        "profile_moment_relative_error": float(
            torch.max(torch.abs(m_fe - m_ref)) / torch.max(torch.abs(m_ref))),
    }