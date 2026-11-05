"""Finite-amplitude (nonlinear) heave of a floating disk floe: exact reference.

Why this module exists
----------------------
``docs/ice_floe_physics_validation.md`` closed the *linear* heave anchors of the
2-D floe model but recorded one explicit open item: "the period anchor uses
``2e-3 R``; the nonlinear (finite-amplitude) heave period is not verified and is
expected to differ."  Nothing in the tree measured how the heave period of the
shipped model depends on amplitude, so any statement of the form "the floe heave
period is verified" was true only in the infinitesimal-amplitude limit.

This module supplies the missing reference.  For the same discrete floe the
production kernels act on -- ``N`` extruded disks of radius ``R`` at prescribed
y-offsets, each carrying its own submerged fraction -- the rigid-heave equation
is an exactly known one-degree-of-freedom oscillator::

    m y'' = rho_w g t * sum_i A(y + o_i - water_level) - m g

with ``A(h)`` the circular-segment area of a disk whose centre is ``h`` above
the free surface.  Because ``A`` is available in closed form, so is the
potential ``U``, and the exact period of the conservative oscillator follows
from energy quadrature

    T(E) = sqrt(m/2) * int_{x_lo}^{x_hi} dx / sqrt(E - U(x)),   U(x_lo)=U(x_hi)=E,

where the turning points bracket the equilibrium.  The well is *asymmetric*
(``U'''(0) != 0``), so both turning points are needed; the ``T = 4 int_0^{x_max}``
form that assumes symmetry is not used here.

Independence of the reference
-----------------------------
* Geometry enters through the chord antiderivative ``P(u) = int sqrt(R^2-u^2) du``
  and its second integral ``Q(u) = -(R^2-u^2)^{3/2}/3``, not through the
  production ``(theta - sin theta)`` submerged fraction nor through
  :func:`tensordem.floe_buoyancy.buoyancy_potential`'s Simpson accumulation.
* The quadrature is Gauss-Legendre on the analytic integrand.  The two intervals
  touching a turning point use the ``x = x_turn +/- (1-u)^2`` substitution that
  removes the square-root endpoint singularity, so the accuracy is limited by
  the node count (checked by convergence, not asserted).
* The solver side is the shipped :class:`tensordem.floe_buoyancy.RigidHeaveFloe`
  driving the shipped ``compute_buoyancy_force``; nothing here re-implements the
  measured path.

Model scope (deliberate, inherited from the 2-D buoyancy model)
--------------------------------------------------------------
Per-particle Archimedes on an extruded disk: no added mass, no free-surface
deformation, no rotation, no horizontal drift, no contact.  The nonlinear
reference therefore verifies the *model's own* amplitude dependence, not a real
floe's heave period: added mass would lengthen the physical period by
``sqrt(1 + Ca)`` and is absent from both sides of the comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from .buoyancy_wet import WetCouplingConfig
from .floe_buoyancy import (
    RigidHeaveFloe,
    mean_period,
    upward_crossing_periods,
)


# ---------------------------------------------------------------------------
# Closed-form disk geometry (independent of the production fraction formula)
# ---------------------------------------------------------------------------

def _chord_antiderivative(u: float, radius: float) -> float:
    """``P(u) = int_0^u sqrt(R^2 - t^2) dt`` for ``|u| <= R``."""
    r2 = radius * radius
    uu = min(max(u, -radius), radius)
    radicand = max(r2 - uu * uu, 0.0)
    return 0.5 * (uu * math.sqrt(radicand) + r2 * math.asin(min(max(uu / radius, -1.0), 1.0)))


def segment_area(h: float, radius: float) -> float:
    """Area of a disk of ``radius`` lying below the free surface, centre ``h`` above it.

    ``A(h) = 2 * int_h^R sqrt(R^2-u^2) du = pi R^2/2 - 2 P(h)``.
    """
    if h >= radius:
        return 0.0
    if h <= -radius:
        return math.pi * radius * radius
    return 0.5 * math.pi * radius * radius - 2.0 * _chord_antiderivative(h, radius)


def segment_area_antiderivative(h: float, radius: float) -> float:
    """``I_A(h) = int_{-R}^{h} A(s) ds``, the volume (2-D: area sweep) below ``h``.

    Obtained by exchanging the order of integration in
    ``int_{-R}^{h} int_{s}^{R} 2 sqrt(R^2-u^2) du ds``::

        I_A(h) = 2[Q(h) + R (P(h) + pi R^2/4)] + 2 (h + R) (pi R^2/4 - P(h))

    with ``Q(u) = -(R^2-u^2)^{3/2}/3``.  Endpoints: ``I_A(-R) = 0`` and
    ``I_A(R) = pi R^3`` (the area under the whole submerged profile), and
    ``dI_A/dh = A(h)`` -- both are asserted in the test module.

    Outside ``[-R, R]`` the antiderivative does *not* saturate at the endpoints:
    above ``R`` the disk is dry and ``A = 0`` so ``I_A = pi R^3`` is constant,
    while below ``-R`` the disk is fully submerged and ``A = pi R^2``, so
    ``I_A`` keeps growing linearly as ``pi R^2 (h + R)``.  Saturating it there
    would silently flatten the potential of a deeply submerged floe; the
    derivative test in the test module covers exactly that branch.
    """
    r = radius
    if h >= r:
        return math.pi * r * r * r
    if h <= -r:
        return math.pi * r * r * (h + r)
    hc = h
    p = _chord_antiderivative(hc, r)
    q = -(max(r * r - hc * hc, 0.0)) ** 1.5 / 3.0
    quarter = 0.25 * math.pi * r * r
    return 2.0 * (q + r * (p + quarter)) + 2.0 * (hc + r) * (quarter - p)


def waterplane_chord(h: float, radius: float) -> float:
    """Free-surface chord ``c(h) = 2 sqrt(R^2 - h^2)`` of one disk (0 outside)."""
    if abs(h) >= radius:
        return 0.0
    return 2.0 * math.sqrt(max(radius * radius - h * h, 0.0))


def segment_area_second_derivative(h: float, radius: float) -> float:
    """``A''(h) = 2h / sqrt(R^2 - h^2)`` inside the disk, ``0`` outside."""
    radicand = radius * radius - h * h
    if radicand <= 0.0:
        return 0.0
    return 2.0 * h / math.sqrt(radicand)


def mean_segment_area(h_lo: float, h_hi: float, radius: float) -> float:
    """Mean of ``A`` over ``[h_lo, h_hi]``, stable when the span is tiny.

    For a span ``d <= 1e-3 R`` with a midpoint away from the disk edges (where
    ``A`` is analytic) the midpoint value plus the curvature term
    ``A'' d^2/24`` is used; otherwise the exact difference
    ``(I_A(h_hi) - I_A(h_lo)) / d``.  This is what keeps ``E - U`` free of the
    ``~1e-13 J`` cancellation error of subtracting aggregate potentials.
    """
    d = h_hi - h_lo
    if d == 0.0:
        return segment_area(h_lo, radius)
    r = radius
    if h_lo >= r:
        return 0.0
    if h_hi <= -r:
        return math.pi * r * r
    if abs(d) <= 1e-3 * r:
        mid = 0.5 * (h_lo + h_hi)
        if -0.99 * r <= mid <= 0.99 * r:
            return segment_area(mid, r) + segment_area_second_derivative(mid, r) * d * d / 24.0
    return (segment_area_antiderivative(h_hi, r) - segment_area_antiderivative(h_lo, r)) / d


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature (self-contained; no numpy dependency)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def gauss_legendre(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``n``-point Gauss-Legendre nodes/weights on ``[-1, 1]`` via Newton iteration."""
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        deriv = 0.0
        for _ in range(100):
            p_prev, p_cur = 1.0, x
            for k in range(2, n + 1):
                p_prev, p_cur = p_cur, ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
            deriv = n * (x * p_cur - p_prev) / (x * x - 1.0)
            dx = -p_cur / deriv
            x += dx
            if abs(dx) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * deriv * deriv))
    return tuple(nodes), tuple(weights)


# ---------------------------------------------------------------------------
# The rigid-heave oscillator of the shipped model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeaveOscillator:
    """Analytic 1-DOF heave model of a discrete disk floe.

    ``offsets`` are the particle y-offsets of the floe relative to its centroid
    (as produced by :func:`tensordem.floe_buoyancy.hex_lattice_floe`), so the
    aggregate force is a sum of ``N`` independent circular segments.  Every
    quantity here is the closed form of the force the shipped kernel evaluates
    numerically; the two are compared in the test module.
    """

    offsets: tuple[float, ...]
    radius: float
    thickness: float
    density_ice: float
    water_density: float = 1025.0
    gravity: float = 9.81
    water_level: float = 0.0

    @classmethod
    def from_config(cls, offsets: Sequence[float], radius: float, thickness: float,
                    density_ice: float, cfg: WetCouplingConfig) -> "HeaveOscillator":
        if cfg.model != "2d":
            raise ValueError("finite-amplitude heave reference is 2-D only")
        return cls(
            offsets=tuple(float(o) for o in offsets),
            radius=float(radius),
            thickness=float(thickness),
            density_ice=float(density_ice),
            water_density=float(cfg.water_density),
            gravity=float(cfg.gravity),
            water_level=float(cfg.water_level),
        )

    # -- geometry / mass -------------------------------------------------
    @property
    def n_particles(self) -> int:
        return len(self.offsets)

    @property
    def particle_area(self) -> float:
        return math.pi * self.radius * self.radius

    @property
    def mass(self) -> float:
        return self.density_ice * self.particle_area * self.thickness * self.n_particles

    def heights(self, y: float) -> tuple[float, ...]:
        """Per-particle centre heights above the free surface at centroid height ``y``."""
        return tuple(y + o - self.water_level for o in self.offsets)

    # -- force and potential --------------------------------------------
    def buoyancy(self, y: float) -> float:
        """Analytic aggregate buoyancy force [N] (positive up)."""
        fac = self.water_density * self.gravity * self.thickness
        return fac * math.fsum(segment_area(h, self.radius) for h in self.heights(y))

    def force(self, y: float) -> float:
        """Net vertical force [N]: aggregate buoyancy minus weight."""
        return self.buoyancy(y) - self.mass * self.gravity

    def potential(self, y: float) -> float:
        """``-int F dy`` up to a constant (``-U_buoy + m g y``, reference fully dry)."""
        fac = -self.water_density * self.gravity * self.thickness
        swept = math.fsum(segment_area_antiderivative(h, self.radius) for h in self.heights(y))
        return fac * swept + self.mass * self.gravity * y

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    def relative_potential(self, x: float, y_eq: float) -> float:
        """``U(y_eq + x) - U(y_eq)``: the well the quadrature integrates over.

        Evaluated as a *centred* difference (one ``I_A`` subtraction per
        particle) instead of subtracting two aggregate potentials.  The absolute
        potentials are of order ``rho_w g t N R^3 ~ 5e2 J``, so the naive
        difference loses ~``1e-13 J`` to cancellation; for the tiny amplitudes
        (``1e-4 R``) that the linear-limit check uses the well depth is only
        ``~1e-7 J`` and the loss would dominate the near-turning-point integrand.
        """
        fac = -self.water_density * self.gravity * self.thickness
        swept = math.fsum(
            segment_area_antiderivative(y_eq + o - self.water_level + x, self.radius)
            - segment_area_antiderivative(y_eq + o - self.water_level, self.radius)
            for o in self.offsets
        )
        return fac * swept + self.weight * x

    def well_depth(self, x: float, y_eq: float, anchor: float) -> float:
        """``U(y_eq+anchor) - U(y_eq+x)`` as a mean-force form.

        ``U(a) - U(b) = -(a - b) * mean(F)`` over the span, so the result carries
        the ``~1e-9 N`` accuracy of the mean-force bracket all the way down to
        spans of ``1e-11 m`` -- exactly where the plain potential difference is
        swamped by cancellation.  ``x`` and ``anchor`` are displacements from the
        equilibrium centroid height ``y_eq``.
        """
        d = anchor - x
        if d <= 0.0:
            return 0.0
        fac = self.water_density * self.gravity * self.thickness
        mean_area = math.fsum(
            mean_segment_area(h - d, h, self.radius)
            for h in self.heights(y_eq + anchor)
        )
        return d * (self.weight - fac * mean_area)

    # -- equilibrium -----------------------------------------------------
    def equilibrium(self, tol: float = 1e-14) -> float:
        """Centroid height ``y_eq`` with ``sum_i A(h_i) = m g / (rho_w g t)``.

        The aggregate force is strictly decreasing in ``y`` (each ``A`` is
        non-increasing), so a bracketed bisection is unconditionally safe.
        """
        target_area = self.mass * self.gravity / (self.water_density * self.gravity * self.thickness)
        span = self.radius + (max(abs(o) for o in self.offsets) if self.offsets else 0.0) + 1.0
        lo = self.water_level - span
        hi = self.water_level + span

        def residual(y: float) -> float:
            total = math.fsum(segment_area(h, self.radius) for h in self.heights(y))
            return total - target_area

        if residual(lo) < 0.0 or residual(hi) > 0.0:
            raise ValueError("equilibrium not bracketed; check offsets/geometry")
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    # -- Taylor coefficients of the restoring force ----------------------
    def curve_coefficients(self, y_eq: float) -> tuple[float, float, float]:
        """``k1, k2, k3`` of ``F(x) = -k1 x - k2 x^2 - k3 x^3`` at ``y_eq``.

        Written out rather than finite-differenced: ``A' = -c``, ``A'' = 2h/sqrt(R^2-h^2)``,
        ``A''' = 2 R^2/(R^2-h^2)^{3/2}``.
        """
        r = self.radius
        fac = self.water_density * self.gravity * self.thickness
        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
        for h in self.heights(y_eq):
            radicand = max(r * r - h * h, 0.0)
            if radicand <= 0.0:
                continue  # disk fully above/below water: all derivatives vanish
            root = math.sqrt(radicand)
            f1 -= waterplane_chord(h, r)
            f2 += 2.0 * h / root
            f3 += 2.0 * r * r / (radicand * root)
        return -fac * f1, -0.5 * fac * f2, -fac * f3 / 6.0

    def linear_period(self, y_eq: float | None = None) -> float:
        """Small-amplitude period ``2 pi sqrt(m / k1)``."""
        y_eq = self.equilibrium() if y_eq is None else y_eq
        k1 = self.curve_coefficients(y_eq)[0]
        return 2.0 * math.pi * math.sqrt(self.mass / k1)


# ---------------------------------------------------------------------------
# Exact (nonlinear) period by energy quadrature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodReference:
    """Exact nonlinear heave period and the quantities it was built from."""

    period: float
    amplitude: float
    y_eq: float
    energy: float
    x_lo: float
    x_hi: float
    k1: float
    k2: float
    k3: float
    linear_period: float
    series_period: float
    n_nodes: int
    n_intervals: int

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "amplitude": self.amplitude,
            "y_eq": self.y_eq,
            "energy": self.energy,
            "x_lo": self.x_lo,
            "x_hi": self.x_hi,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "linear_period": self.linear_period,
            "series_period": self.series_period,
            "n_nodes": self.n_nodes,
            "n_intervals": self.n_intervals,
        }


def _quad(gap, xa: float, xb: float, n_nodes: int,
          left_singular: bool, right_singular: bool) -> float:
    """``int_{xa}^{xb} dx / sqrt(gap(x))`` with optional endpoint de-singularisation.

    A turning point of a conservative oscillator has ``gap ~ c (x - x_t)``, so a
    plain Gauss rule sees a ``1/sqrt`` singularity.  The substitutions
    ``x = xa + (xb-xa) u^2`` (left end) and ``x = xb - (xb-xa)(1-u)^2`` (right
    end) map the integrand to a smooth function whose endpoint value is fixed by
    the analytic force.  Interior panels use the identity map.
    """
    nodes, weights = gauss_legendre(n_nodes)
    span = xb - xa
    total = 0.0
    for xi, wi in zip(nodes, weights):
        u = 0.5 * (xi + 1.0)  # map [-1, 1] -> [0, 1]
        if left_singular:
            x = xa + span * u * u
            jac = 2.0 * span * u
        elif right_singular:
            x = xb - span * (1.0 - u) * (1.0 - u)
            jac = 2.0 * span * (1.0 - u)
        else:
            x = xa + span * u
            jac = span
        value = gap(x)
        if value <= 0.0:
            raise ValueError("quadrature node left the potential well")
        total += 0.5 * wi * jac / math.sqrt(value)
    return total


def _turning_point(gap, x_start: float, direction: float, max_reach: float) -> float:
    """Solve ``gap(x) = 0`` on the side of ``x_start`` given by ``direction``.

    ``gap`` (the energy above ``U(x)``) decreases monotonically away from the
    equilibrium on each side, and the returned point is the *inside* bracket end,
    so every node of the well keeps a strictly positive gap.
    """
    x_inside = x_start
    x_outside = x_start + direction * max_reach
    f_inside = gap(x_inside)
    f_outside = gap(x_outside)
    if f_inside <= 0.0:
        raise ValueError("start point is not inside the well")
    if f_outside > 0.0:
        raise ValueError("turning point not bracketed; increase amplitude or reach")
    for _ in range(300):
        mid = 0.5 * (x_inside + x_outside)
        f_mid = gap(mid)
        if f_mid > 0.0:
            x_inside = mid
        else:
            x_outside = mid
        if abs(x_outside - x_inside) < 1e-16 * (1.0 + abs(x_inside)):
            break
    return x_inside


def exact_heave_period(osc: HeaveOscillator, amplitude: float, y_eq: float | None = None,
                       n_nodes: int = 160) -> PeriodReference:
    """Exact period of the released-from-rest oscillation of amplitude ``amplitude``.

    The initial condition is ``y(0) = y_eq + amplitude``, ``v(0) = 0``, so
    ``E = U(amplitude)`` and ``x_hi = amplitude``; the opposite turning point
    ``x_lo < 0`` follows from ``E - U(x_lo) = 0``.  The energy is handled
    entirely through :meth:`HeaveOscillator.well_depth`, anchored at ``x_hi``.
    """
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    y_eq = osc.equilibrium() if y_eq is None else y_eq
    x_hi = amplitude

    def gap(x: float) -> float:
        return osc.well_depth(x, y_eq, x_hi)

    energy = gap(0.0)
    reach = 2.0 * (abs(amplitude) + osc.radius + abs(y_eq) + 1.0)
    x_lo = _turning_point(gap, 0.0, -1.0, reach)

    # Breakpoints: a particle entering or leaving the water makes U only C^1;
    # split the interval there so each Gauss panel sees a smooth integrand.
    kinks: list[float] = []
    for o in osc.offsets:
        for edge in (osc.radius, -osc.radius):
            x = osc.water_level + edge - y_eq - o
            if x_lo < x < x_hi:
                kinks.append(x)
    knots = sorted({x_lo, 0.0, x_hi, *kinks})
    knots = [k for k in knots if x_lo <= k <= x_hi]
    if knots[0] > x_lo:
        knots.insert(0, x_lo)
    if knots[-1] < x_hi:
        knots.append(x_hi)

    total = 0.0
    n_intervals = 0
    for xa, xb in zip(knots[:-1], knots[1:]):
        if xb - xa <= 0.0:
            continue
        total += _quad(gap, xa, xb, n_nodes,
                       left_singular=(xa == x_lo),
                       right_singular=(xb == x_hi))
        n_intervals += 1

    # The sweep from x_lo to x_hi is a *half* period; hence sqrt(2m), not sqrt(m/2).
    period = math.sqrt(2.0 * osc.mass) * total
    k1, k2, k3 = osc.curve_coefficients(y_eq)
    linear = 2.0 * math.pi * math.sqrt(osc.mass / k1)
    correction = 3.0 * k3 / (8.0 * k1) - 5.0 * k2 * k2 / (12.0 * k1 * k1)
    series = linear * (1.0 - correction * amplitude * amplitude)
    return PeriodReference(
        period=period,
        amplitude=amplitude,
        y_eq=y_eq,
        energy=energy,
        x_lo=x_lo,
        x_hi=x_hi,
        k1=k1,
        k2=k2,
        k3=k3,
        linear_period=linear,
        series_period=series,
        n_nodes=n_nodes,
        n_intervals=n_intervals,
    )


def amplitude_series_period(osc: HeaveOscillator, amplitude: float,
                            y_eq: float | None = None) -> float:
    """Lindstedt-Poincare prediction ``T = T0 (1 - c A^2)`` for the same well.

    A weakly nonlinear oscillator with ``F(x) = -k1 x - k2 x^2 - k3 x^3`` has
    ``T/T0 = 1 - (3 k3/(8 k1) - 5 k2^2/(12 k1^2)) A^2 + O(A^4)``.  It is quoted
    only as a small-amplitude check on the quadrature; at finite amplitude the
    quadrature -- not the series -- is the reference.
    """
    y_eq = osc.equilibrium() if y_eq is None else y_eq
    k1, k2, k3 = osc.curve_coefficients(y_eq)
    linear = 2.0 * math.pi * math.sqrt(osc.mass / k1)
    correction = 3.0 * k3 / (8.0 * k1) - 5.0 * k2 * k2 / (12.0 * k1 * k1)
    return linear * (1.0 - correction * amplitude * amplitude)


# ---------------------------------------------------------------------------
# Measuring the shipped solver
# ---------------------------------------------------------------------------

@dataclass
class HeaveMeasurement:
    """Period measured on the shipped ``RigidHeaveFloe`` integrator."""

    period: float
    amplitude: float
    dt: float
    y_eq: float
    periods: list[float] = field(default_factory=list)
    drift: float = 0.0

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "amplitude": self.amplitude,
            "dt": self.dt,
            "y_eq": self.y_eq,
            "periods": list(self.periods),
            "drift": self.drift,
        }


def measure_heave_period(osc: HeaveOscillator, cfg: WetCouplingConfig, amplitude: float,
                         dt: float, n_cycles: int = 12, skip_cycles: int = 2,
                         y_eq: float | None = None) -> HeaveMeasurement:
    """Run :class:`RigidHeaveFloe` from rest at ``y_eq + amplitude`` and time the cycles.

    The drag term is forced off (``cfg.wet_drag_coeff = 0``) so the measured
    system is the same conservative oscillator the quadrature integrates; a
    damped run would not have a well-defined single period.
    """
    y_eq = osc.equilibrium() if y_eq is None else y_eq
    damped = WetCouplingConfig(
        enabled=cfg.enabled,
        water_density=osc.water_density,
        gravity=osc.gravity,
        water_level=osc.water_level,
        wet_drag_coeff=0.0,
        model="2d",
    )
    floe = RigidHeaveFloe(
        offsets=_offsets_tensor(osc.offsets),
        particle_radius=osc.radius,
        thickness=osc.thickness,
        density_ice=osc.density_ice,
        cfg=damped,
        dt=dt,
        y0=y_eq + amplitude,
    )
    period_guess = osc.linear_period(y_eq)
    n_steps = int(math.ceil((n_cycles + 1) * period_guess / dt))
    traj = floe.run(n_steps, record_every=1)
    periods = upward_crossing_periods(traj.t, traj.y, y_eq)
    measured = mean_period(periods, skip=min(skip_cycles, max(len(periods) - 1, 0)))
    drift = max(traj.y) - min(traj.y)
    return HeaveMeasurement(period=float(measured), amplitude=amplitude, dt=dt,
                            y_eq=y_eq, periods=list(periods), drift=float(drift))


def _offsets_tensor(offsets: Iterable[float]):
    import torch

    return torch.tensor([[0.0, float(o)] for o in offsets], dtype=torch.float64)


def richardson_period(osc: HeaveOscillator, cfg: WetCouplingConfig, amplitude: float,
                      dt: float, **kwargs) -> float:
    """Second-order (velocity-Verlet) extrapolation ``(4 T(dt/2) - T(dt))/3``."""
    coarse = measure_heave_period(osc, cfg, amplitude, dt, **kwargs).period
    fine = measure_heave_period(osc, cfg, amplitude, 0.5 * dt, **kwargs).period
    return (4.0 * fine - coarse) / 3.0