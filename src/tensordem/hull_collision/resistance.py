"""Side-shell resistance and energy allocation for ice-region ship collisions.

Model chain
-----------
1. Each structural layer of the struck side (outer shell, inner shell, ...) is
   idealised as stiffened plating spanning one transverse frame bay.  Its
   resistance is the sum of two rigid-plastic terms:

   * hinge term  ``E_hinge(d) = [ 8/span * ( N_long*M_p,long + m_p*b_eff ) ] * d``
     with ``m_p = sigma_y*t^2/4`` the plate plastic moment per unit width and
     ``M_p,long = sigma_y*Z_long`` the fully plastic moment of one longitudinal
     with its attached plating (see :mod:`tensordem.hull_collision.plastic_hinge`).
     This is the ``E = M_p*theta`` mechanism of a clamped-clamped beam bay.
   * membrane term ``E_membrane(d) = sigma_y * b_eff * t * d^2 / span``,
     i.e. fully plastic membrane stretching of the same bay: the mean plastic
     strain of the panel is ``(d/span)^2`` (clamped strip kinematics), so the
     plastic work density is ``sigma_y*(d/span)^2`` over the panel volume
     ``span*b_eff*t``.  This term is what limits the penetration; without it the
     pure hinge mechanism predicts penetration depths of the order of the frame
     spacing for a few tens of MJ.

   Both terms are *geometrically* motivated, not fitted: the only free inputs
   are scantlings, yield stress and the contact-patch width.

2. Layers engage in series: outer shell first, then (once its crush depth is
   spent) the inner shell, etc.  ``E_total(d) = sum_k E_k(d)`` and the inner
   shell is *intact* until the indenter has passed through its plating
   thickness.

3. Friction: the normal contact force is in equilibrium with the structural
   resistance, so the work-equivalent friction dissipation over the
   penetration is ``E_friction(d) = mu * E_structural(d)`` with ``mu`` the
   ice-steel friction coefficient.  This is the discrete form of
   ``E_friction = mu*N*d`` with ``N`` the *work-average* normal force
   ``N_bar = E_structural/d``.

4. ``solve_penetration`` finds the penetration depth that balances the
   collision kinetic energy; the residual is reported, never forced to zero by
   construction of the reported number.

Limitations
-----------
* Quasi-static, monotone loading; no dynamics, no added-mass coupling, no ice
  crushing energy (the indenter is rigid).  For ice collisions the ice itself
  absorbs a large share of the energy - this surface deliberately isolates the
  *structural* share, so ``d`` here is an upper bound for a rigid indenter.
* No fracture/tearing energy, no buckling, no strain hardening or rate effects.
* The membrane term is the rigid-plastic limit; below ``d/span ~ 0.1`` the real
  panel response is elastic and this model under-predicts stiffness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .plastic_hinge import plastic_section_modulus_rect

__all__ = [
    "SideShellLayer",
    "DoubleHullSide",
    "PenetrationSolution",
    "DamageGeometry",
    "CapacityExhausted",
    "solve_penetration",
    "verify_layer_decomposition",
    "damage_geometry",
]


def _real(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class SideShellLayer:
    """One rigid-plastic structural layer of the struck side shell.

    Attributes
    ----------
    name : human readable layer id (``"outer_shell"``, ``"inner_shell"``, ...).
    thickness : plating thickness ``t`` [m].
    sigma_y : static yield stress [Pa].
    frame_span : transverse frame spacing ``span`` [m] (the bending span).
    effective_width : width of plating participating in the mechanism [m]
        (the contact patch height, widened by one stiffener spacing on each
        side, is a typical choice and is *given by the caller*).
    n_longitudinals : number of longitudinals inside ``effective_width``.
    longitudinal_Z : fully plastic section modulus of one longitudinal
        *including attached plating* [m^3].
    longitudinal_area : bare steel cross-section of one longitudinal [m^2],
        used only for damaged-volume accounting.
    crush_depth : penetration distance over which this layer deforms before it
        is spent [m].  For the outer shell of a double hull this is the distance
        to the inner shell; for the inner shell it is the plastic deformation
        capacity of the inner boundary structure.
    """

    name: str
    thickness: float
    sigma_y: float
    frame_span: float
    effective_width: float
    n_longitudinals: int
    longitudinal_Z: float
    longitudinal_area: float
    crush_depth: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("layer name must be a non-empty string")
        _real("thickness", self.thickness, positive=True)
        _real("sigma_y", self.sigma_y, positive=True)
        _real("frame_span", self.frame_span, positive=True)
        _real("effective_width", self.effective_width, positive=True)
        _real("longitudinal_Z", self.longitudinal_Z, positive=True)
        _real("longitudinal_area", self.longitudinal_area, nonnegative=True)
        _real("crush_depth", self.crush_depth, positive=True)
        if isinstance(self.n_longitudinals, bool) or not isinstance(self.n_longitudinals, int):
            raise TypeError("n_longitudinals must be an integer")
        if self.n_longitudinals < 1:
            raise ValueError("n_longitudinals must be >= 1")

    # -- mechanics -------------------------------------------------------
    @property
    def plate_plastic_moment_per_width(self) -> float:
        """``m_p = sigma_y*t^2/4`` [N*m/m]."""
        return self.sigma_y * plastic_section_modulus_rect(1.0, self.thickness)

    @property
    def hinge_coefficient(self) -> float:
        """Hinge stiffness ``k1 = dE/dd`` of the layer [N]."""
        per_mechanism = (
            self.n_longitudinals * self.sigma_y * self.longitudinal_Z
            + self.plate_plastic_moment_per_width * self.effective_width
        )
        return 8.0 * per_mechanism / self.frame_span

    @property
    def membrane_coefficient(self) -> float:
        """Membrane stiffness ``k2`` with ``E_membrane = k2*d^2`` [N/m]."""
        return self.sigma_y * self.effective_width * self.thickness / self.frame_span

    def energy(self, local_penetration: float) -> float:
        """Energy absorbed by this layer for a *local* penetration [J].

        Saturates at :meth:`capacity` once ``crush_depth`` is spent.
        """
        u = min(max(_real("local_penetration", local_penetration, nonnegative=True), 0.0), self.crush_depth)
        return self.hinge_coefficient * u + self.membrane_coefficient * u * u

    @property
    def capacity(self) -> float:
        return self.energy(self.crush_depth)

    def resistance(self, local_penetration: float) -> float:
        """``dE/du`` [N]; zero once the layer is spent."""
        u = _real("local_penetration", local_penetration, nonnegative=True)
        if u >= self.crush_depth:
            return 0.0
        return self.hinge_coefficient + 2.0 * self.membrane_coefficient * u


@dataclass(frozen=True)
class DoubleHullSide:
    """Stack of structural layers engaged in series by the indenter.

    ``friction_coefficient`` is the ice-steel friction coefficient ``mu``; the
    friction dissipation is ``mu`` times the structural dissipation (see the
    module docstring for the derivation).
    """

    layers: tuple
    friction_coefficient: float = 0.0

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("at least one structural layer is required")
        for layer in layers:
            if not isinstance(layer, SideShellLayer):
                raise TypeError("layers must be SideShellLayer instances")
        _real("friction_coefficient", self.friction_coefficient, nonnegative=True)
        object.__setattr__(self, "layers", layers)

    # -- geometry --------------------------------------------------------
    @property
    def window_starts(self) -> tuple:
        starts, acc = [], 0.0
        for layer in self.layers:
            starts.append(acc)
            acc += layer.crush_depth
        return tuple(starts)

    @property
    def max_depth(self) -> float:
        return sum(layer.crush_depth for layer in self.layers)

    def local_penetrations(self, depth: float) -> tuple:
        d = _real("depth", depth, nonnegative=True)
        starts = self.window_starts
        out = []
        for layer, start in zip(self.layers, starts):
            out.append(min(max(d - start, 0.0), layer.crush_depth))
        return tuple(out)

    # -- energy ----------------------------------------------------------
    def layer_energies(self, depth: float) -> tuple:
        return tuple(
            layer.energy(u) for layer, u in zip(self.layers, self.local_penetrations(depth))
        )

    def structural_energy(self, depth: float) -> float:
        return math.fsum(self.layer_energies(depth))

    def friction_energy(self, depth: float) -> float:
        return self.friction_coefficient * self.structural_energy(depth)

    def absorbed(self, depth: float) -> float:
        return self.structural_energy(depth) + self.friction_energy(depth)

    def resistance(self, depth: float) -> float:
        """Structural resistance force at a penetration depth [N].

        A layer that the indenter has not yet reached contributes *nothing*; the
        first layer is engaged from ``d = 0`` onwards.
        """
        d = _real("depth", depth, nonnegative=True)
        forces = []
        for layer, start in zip(self.layers, self.window_starts):
            if start > 0.0 and d <= start:
                continue
            forces.append(layer.resistance(d - start))
        return math.fsum(forces)

    @property
    def total_capacity(self) -> float:
        return self.absorbed(self.max_depth)

    def inner_shell_intact(self, depth: float) -> bool:
        """True while no *inner* layer has been penetrated through its plating.

        The outer shell is the layer subscript 0; it is expected to be breached
        (that is what "penetration" means).  Any layer ``i >= 1`` counts as
        breached once its local penetration reaches its own plating thickness.
        """
        locals_ = self.local_penetrations(depth)
        for layer, u in zip(self.layers[1:], locals_[1:]):
            if u >= layer.thickness:
                return False
        return True

    def breached_layers(self, depth: float) -> tuple:
        names, locals_ = [], self.local_penetrations(depth)
        for layer, u in zip(self.layers, locals_):
            if u >= layer.thickness:
                names.append(layer.name)
        return tuple(names)


class CapacityExhausted(RuntimeError):
    """Raised when the collision energy exceeds the structural capacity."""


@dataclass(frozen=True)
class PenetrationSolution:
    penetration_depth: float
    kinetic_energy: float
    structural_energy: float
    friction_energy: float
    residual: float
    residual_relative: float
    layer_energies: tuple
    layer_local_penetrations: tuple
    layer_names: tuple
    structural_resistance_N: float
    inner_shell_intact: bool
    breached_layers: tuple
    iterations: int
    capacity_J: float


def solve_penetration(
    kinetic_energy: float,
    structure: DoubleHullSide,
    *,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> PenetrationSolution:
    """Solve ``E_kinetic = E_structural(d) + E_friction(d)`` for the depth ``d``.

    ``E_absorbed(d)`` is continuous, non-decreasing and goes to zero at ``d = 0``,
    so a bracketed bisection on ``[0, max_depth]`` is used (no derivative needed,
    no tuning).  ``V = 0`` returns exactly ``d = 0`` without any iteration - this
    is the negative control of the energy surface.
    """
    if not isinstance(structure, DoubleHullSide):
        raise TypeError("structure must be a DoubleHullSide")
    e_k = _real("kinetic_energy", kinetic_energy, nonnegative=True)
    tol = _real("tol", tol, positive=True)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    capacity = structure.total_capacity
    if e_k == 0.0:
        return PenetrationSolution(
            penetration_depth=0.0,
            kinetic_energy=0.0,
            structural_energy=0.0,
            friction_energy=0.0,
            residual=0.0,
            residual_relative=0.0,
            layer_energies=tuple(0.0 for _ in structure.layers),
            layer_local_penetrations=tuple(0.0 for _ in structure.layers),
            layer_names=tuple(layer.name for layer in structure.layers),
            structural_resistance_N=structure.resistance(0.0),
            inner_shell_intact=True,
            breached_layers=(),
            iterations=0,
            capacity_J=capacity,
        )
    if e_k > capacity:
        raise CapacityExhausted(
            f"kinetic energy {e_k/1e6:.3f} MJ exceeds structural capacity {capacity/1e6:.3f} MJ"
        )

    lo, hi = 0.0, structure.max_depth
    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        if structure.absorbed(mid) < e_k:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    depth = 0.5 * (lo + hi)

    structural = structure.structural_energy(depth)
    friction = structure.friction_energy(depth)
    residual = e_k - (structural + friction)
    return PenetrationSolution(
        penetration_depth=depth,
        kinetic_energy=e_k,
        structural_energy=structural,
        friction_energy=friction,
        residual=residual,
        residual_relative=abs(residual) / e_k,
        layer_energies=structure.layer_energies(depth),
        layer_local_penetrations=structure.local_penetrations(depth),
        layer_names=tuple(layer.name for layer in structure.layers),
        structural_resistance_N=structure.resistance(depth),
        inner_shell_intact=structure.inner_shell_intact(depth),
        breached_layers=structure.breached_layers(depth),
        iterations=iterations,
        capacity_J=capacity,
    )


def verify_layer_decomposition(structure: DoubleHullSide, depth: float, *, n_intervals: int = 20000) -> dict:
    """Independent check of ``E_total = sum_k E_k`` by quadrature of the force curve.

    Integrates ``int_0^d R(s) ds`` with composite Simpson's rule from the
    resistance function (i.e. from ``dE/dd``, not from the layer energy
    expressions) and compares it with ``sum_k E_k(d)``.
    """
    if not isinstance(structure, DoubleHullSide):
        raise TypeError("structure must be a DoubleHullSide")
    d = _real("depth", depth, nonnegative=True)
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, int) or n_intervals < 2:
        raise ValueError("n_intervals must be an integer >= 2")
    if n_intervals % 2 == 1:
        n_intervals += 1
    if d == 0.0:
        return {
            "depth_m": 0.0,
            "energy_layers_J": 0.0,
            "energy_quadrature_J": 0.0,
            "rel_error": 0.0,
            "layer_energies_J": tuple(0.0 for _ in structure.layers),
        }
    h = d / n_intervals
    total = structure.resistance(0.0) + structure.resistance(d)
    for i in range(1, n_intervals):
        coeff = 4.0 if i % 2 == 1 else 2.0
        total += coeff * structure.resistance(i * h)
    quadrature = total * h / 3.0
    layer_sum = structure.structural_energy(d)
    return {
        "depth_m": d,
        "energy_layers_J": layer_sum,
        "energy_quadrature_J": quadrature,
        "rel_error": abs(quadrature - layer_sum) / layer_sum if layer_sum else 0.0,
        "layer_energies_J": structure.layer_energies(d),
    }


@dataclass(frozen=True)
class DamageGeometry:
    """Damage geometry handed to SOLAS damage-stability input (SI units)."""

    length_m: float
    width_m: float
    depth_m: float
    volume_m3: float
    inner_shell_intact: bool
    breached_layers: tuple
    layer_volumes_m3: tuple

    def as_dict(self) -> dict:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "volume_m3": self.volume_m3,
            "inner_shell_intact": self.inner_shell_intact,
            "breached_layers": list(self.breached_layers),
            "layer_volumes_m3": list(self.layer_volumes_m3),
        }


def damage_geometry(
    structure: DoubleHullSide,
    solution: PenetrationSolution,
    *,
    contact_length: float,
) -> DamageGeometry:
    """Damage extents implied by the plastic mechanism.

    * ``length`` - contact patch length plus one yielded frame bay at each end of
      the contact zone (the rigid-plastic membrane mechanism spreads over the
      bays adjacent to the contact), for the *outermost* layer.
    * ``width``  - largest effective width among the engaged layers.
    * ``depth``  - solved penetration depth.
    * ``volume`` - bare steel volume inside the yielded region of every engaged
      layer: plating ``L*W_k*t_k`` plus longitudinals ``n_k*A_k*L``.

    Only layers whose local penetration is non-zero contribute, so a collision
    that does not reach the inner shell reports no inner-shell volume.
    """
    if not isinstance(structure, DoubleHullSide):
        raise TypeError("structure must be a DoubleHullSide")
    if not isinstance(solution, PenetrationSolution):
        raise TypeError("solution must be a PenetrationSolution")
    l_c = _real("contact_length", contact_length, positive=True)

    length = l_c + 2.0 * structure.layers[0].frame_span
    volumes = []
    width = 0.0
    for layer, u in zip(structure.layers, solution.layer_local_penetrations):
        if u <= 0.0:
            volumes.append(0.0)
            continue
        width = max(width, layer.effective_width)
        volumes.append(
            length * layer.effective_width * layer.thickness
            + layer.n_longitudinals * layer.longitudinal_area * length
        )
    if width == 0.0:
        width = structure.layers[0].effective_width
    return DamageGeometry(
        length_m=length,
        width_m=width,
        depth_m=solution.penetration_depth,
        volume_m3=math.fsum(volumes),
        inner_shell_intact=solution.inner_shell_intact,
        breached_layers=solution.breached_layers,
        layer_volumes_m3=tuple(volumes),
    )
