"""Maxwell-Wiechert (Prony-series) delayed-elastic bond law for ice DEM bonds.

Physics
-------
The default :class:`~tensordem.dem3d.IceDEM3D` bond is a pure linear spring:
one elastic constant, no memory, therefore no rate dependence at all.  Ice, in
contrast, creeps: its compliance grows with the *time the load is applied*, and
that memory is the reason the uniaxial compressive strength of ice depends on
the strain rate (ductile ``sigma_f ~ edot**(1/n)`` at low rates, rate-insensitive
brittle plateau at high rates).

The smallest constitutive upgrade that carries that memory is a
Maxwell-Wiechert element: an equilibrium spring ``k0`` in parallel with ``M``
Maxwell branches ``(k_a, tau_a)``.  Written on the bond elongation ``d``
(``d = r - r0``, metres) the axial bond force is

    F(d) = k0 * d + sum_a k_a * (d - d_a),      d(d_a)/dt = (d - d_a) / tau_a,

i.e. the relaxation modulus is the Prony series

    G(t) = k0 + sum_a k_a * exp(-t / tau_a)         [N/m],
    G(0) = k0 + sum_a k_a = k_unrelaxed,   G(inf) = k0.

``tau_a`` are the *delay times* (``tau_1``, ``tau_2`` in the mission
specification); ``k_unrelaxed / k0`` is the delayed-elastic stiffening that makes
the instantaneous (wave-speed) response of ice stiffer than its relaxed
(quasi-static, creeping) response.

Exact incremental integration
-----------------------------
Within one time increment the strain path is linear, ``d(t) = d_old + dd*t/dt``,
so the relaxation ODE for ``s_a = d - d_a`` has the closed-form solution

    s_a(t) = A_a (1 - u_a(t)) + s_a(0) u_a(t),
    A_a = (dd/dt) * tau_a,      u_a(t) = exp(-t / tau_a).

All three quantities the energy ledger needs are therefore available in closed
form and the ledger closes algebraically (to round-off, not to truncation
error):

* work done on the bond     ``W  = int F dd``
* stored spring energy      ``dU = k0/2 (d^2 - d_old^2) + sum k_a/2 (s_a^2 - s_a0^2)``
* dashpot dissipation       ``Phi = sum (k_a/tau_a) * int s_a^2 dt``  (exactly ``W - dU``)

with ``int s^2 dt`` and ``int s dd`` both elementary; see
:func:`advance_branches`.  This is what makes a ``< 1e-10`` energy closure
achievable instead of a truncation-limited ``~1e-3``: it is measured at
``1.5e-12`` relative for ``dt/tau >= 3e-4`` (see
``docs/ice_bond_viscoelastic_verification.md``).  The closure is *not* uniform
in ``dt/tau``: the ``int s^2 dt`` summand is a difference of terms of order
``(dd/dt)^2 tau^2 dt``, so for ``dt/tau ~ 1e-6`` with large strain increments
the residual rises to ``~1e-7`` relative (cancellation, not truncation).

Failure / damage
----------------
Two failure mechanisms share the bond:

1. **Brittle (instantaneous) cleavage** -- ``|F| >= F_b``, with ``F_b`` drawn
   per bond from a Weibull distribution (heterogeneity / grain-scale scatter).
2. **Creep damage (sub-critical)** -- Kachanov-style scalar damage
   ``dD/dt = (1/t_ref) * (|F| / F_ref) ** m`` with failure at ``D >= 1``.

The two together are what produce the ductile-to-brittle transition: at a slow
strain rate the bond spends enough *time* at a given load for ``D`` to reach 1
first, so the specimen fails by distributed creep damage at a stress well below
the cleavage strength; at a fast rate the bond reaches ``F_b`` before the damage
clock can run, so it fails in a brittle, load-redistribution dominated way.

Because ``D`` is an integral over the *real* time and ``F`` grows linearly with
strain, a specimen whose bonds follow mechanism 2 fails at
``sigma_f ~ t_ref**(1/(m+1))`` for a fixed strain rate, i.e. the emergent
envelope is ``sigma_f ~ edot**(1/(m+1))``.  Setting ``m = 2`` therefore
recovers the classical ductile exponent ``1/3`` (Golding/Schulson) *without*
hard-coding the envelope; see ``docs/ice_bond_rate_envelope.md``.

All quantities are SI.  Nothing in this module mutates the solver state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

#: Default delayed-elastic spectrum for columnar/granular fresh-water ice at
#: about -10 degC, split by the observed transient-creep spectrum rather than
#: fitted to the strength envelope: a short (sub-second) branch that dominates
#: the first 10 % of a loading ramp and a long (tens of seconds) branch that
#: carries the transient creep tail (Sinha 1978 delayed elasticity, tau ~ 1-30 s).
DEFAULT_DELAYED_ELASTIC_FRACTION: tuple[float, ...] = (0.45, 0.20)
DEFAULT_RELAXATION_TIMES_S: tuple[float, ...] = (0.5, 30.0)
#: Equilibrium (fully relaxed) share of the unrelaxed stiffness.  Ice's relaxed
#: modulus inferred from transient-creep plateaus is 30-40 % of the elastic one.
DEFAULT_EQUILIBRIUM_FRACTION: float = 0.35


@dataclass(frozen=True)
class MaxwellWiechertBond:
    """Prony-series bond law (equilibrium spring + ``M`` Maxwell branches)."""

    equilibrium_stiffness: float
    branch_stiffness: tuple[float, ...]
    relaxation_time: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.equilibrium_stiffness)
                and self.equilibrium_stiffness >= 0.0):
            raise ValueError("equilibrium_stiffness must be finite and nonnegative")
        if len(self.branch_stiffness) != len(self.relaxation_time):
            raise ValueError("branch_stiffness and relaxation_time must have equal length")
        for k in self.branch_stiffness:
            if not (math.isfinite(k) and k > 0.0):
                raise ValueError("branch stiffnesses must be finite and positive")
        for tau in self.relaxation_time:
            if not (math.isfinite(tau) and tau > 0.0):
                raise ValueError("relaxation times must be finite and positive")
        if self.unrelaxed_stiffness <= 0.0:
            raise ValueError("total (unrelaxed) bond stiffness must be positive")

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_unrelaxed_stiffness(
        cls,
        unrelaxed_stiffness: float,
        *,
        equilibrium_fraction: float = DEFAULT_EQUILIBRIUM_FRACTION,
        delayed_elastic_fraction: tuple[float, ...] = DEFAULT_DELAYED_ELASTIC_FRACTION,
        relaxation_time: tuple[float, ...] = DEFAULT_RELAXATION_TIMES_S,
    ) -> "MaxwellWiechertBond":
        """Split a target instantaneous stiffness into equilibrium + branches.

        ``equilibrium_fraction`` is ``k0 / k_unrelaxed``; the remaining
        ``1 - equilibrium_fraction`` is shared between the Maxwell branches in
        the proportions of ``delayed_elastic_fraction``.  The shares are
        *proportions*, so they are normalised by their sum: both the
        proportions convention (shares summing to 1) and the fraction-of-
        unrelaxed-stiffness convention used by the module defaults (which sum to
        ``1 - equilibrium_fraction``) give the same bond law.  Requiring the
        sum to be exactly 1 rejected the module's own defaults
        ``(0.45, 0.20)`` against ``equilibrium_fraction = 0.35``.
        """
        if not (0.0 < equilibrium_fraction < 1.0):
            raise ValueError("equilibrium_fraction must be in (0, 1)")
        shares = tuple(float(f) for f in delayed_elastic_fraction)
        if len(shares) != len(relaxation_time):
            raise ValueError("delayed_elastic_fraction must match the tau count")
        if any(not (math.isfinite(f) and f > 0.0) for f in shares):
            raise ValueError("delayed_elastic_fraction entries must be finite and positive")
        share_sum = sum(shares)
        if not (math.isfinite(share_sum) and share_sum > 0.0):
            raise ValueError("delayed_elastic_fraction must have a positive finite sum")
        k_total = float(unrelaxed_stiffness)
        k0 = equilibrium_fraction * k_total
        branches = tuple((1.0 - equilibrium_fraction) * k_total * (f / share_sum)
                         for f in shares)
        return cls(equilibrium_stiffness=k0, branch_stiffness=branches,
                   relaxation_time=tuple(float(t) for t in relaxation_time))

    # -- properties ----------------------------------------------------------
    @property
    def n_branches(self) -> int:
        return len(self.branch_stiffness)

    @property
    def unrelaxed_stiffness(self) -> float:
        return float(self.equilibrium_stiffness + sum(self.branch_stiffness))

    @property
    def delayed_elastic_stiffness(self) -> float:
        return float(sum(self.branch_stiffness))

    def relaxation_modulus(self, time: float | torch.Tensor) -> torch.Tensor:
        """``G(t) = k0 + sum k_a exp(-t/tau_a)`` in N/m (the relaxation test)."""
        t = time if isinstance(time, torch.Tensor) else torch.tensor(
            float(time), dtype=torch.float64)
        total = torch.full_like(t, float(self.equilibrium_stiffness))
        for k, tau in zip(self.branch_stiffness, self.relaxation_time):
            total = total + k * torch.exp(-t / tau)
        return total

    def relaxed_stiffness_at(self, time: float) -> float:
        return float(self.relaxation_modulus(time).item())


@torch.no_grad()
def advance_branches(
    model: MaxwellWiechertBond,
    branch_offsets: torch.Tensor,
    elongation_old: torch.Tensor,
    elongation_new: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the Maxwell branches over one increment, exactly.

    Parameters
    ----------
    branch_offsets
        ``(n_bonds, M)`` dashpot elongations ``d_a`` at the start of the step.
    elongation_old, elongation_new
        ``(n_bonds,)`` bond elongation ``d`` before and after the increment.
    dt
        Increment length in seconds (must be positive).

    Returns
    -------
    (branch_offsets_new, force_new, work, dissipation)
        ``work`` is ``int F dd`` and ``dissipation`` is ``sum_a (k_a/tau_a)
        int s_a^2 dt``; their difference equals the change of stored spring
        energy to round-off (the ledger identity of the module docstring).
    """
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    dtype = elongation_new.dtype
    device = elongation_new.device
    d_old = elongation_old.to(dtype)
    d_new = elongation_new.to(dtype)
    delta_d = d_new - d_old
    d_dot = delta_d / dt

    k0 = float(model.equilibrium_stiffness)
    stored_new = 0.5 * k0 * d_new.square()
    stored_old = 0.5 * k0 * d_old.square()
    work = stored_new - stored_old
    dissipation = torch.zeros_like(d_new)
    force = k0 * d_new
    new_offsets = [] if model.n_branches else branch_offsets
    for a, (k_a, tau_a) in enumerate(zip(model.branch_stiffness,
                                         model.relaxation_time)):
        k = float(k_a)
        tau = float(tau_a)
        off_old = branch_offsets[:, a].to(dtype)
        s0 = d_old - off_old
        amp = d_dot * tau
        decay = torch.exp(torch.tensor(-dt / tau, dtype=dtype, device=device))
        one_minus = -torch.expm1(torch.tensor(-dt / tau, dtype=dtype, device=device))
        s_new = amp * one_minus + s0 * decay
        new_offsets.append(d_new - s_new)
        # int s dd  and  int s^2 dt  in closed form (linear strain path).
        integral_s_dd = d_dot * (amp * dt + (s0 - amp) * tau * one_minus)
        # (1 - u^2) is formed as (1-u)(1+u): computing 1 - decay*decay loses
        # the whole significance of the factor when dt << tau, and that error
        # propagates into the dissipation and breaks the energy ledger at small
        # dt/tau (measured 6.6e-8 relative at dt/tau = 2e-4, vs <1e-10 below).
        integral_s2 = (amp * amp * dt
                       + 2.0 * amp * (s0 - amp) * tau * one_minus
                       + 0.5 * (s0 - amp).square() * tau
                       * (one_minus * (1.0 + decay)))
        work = work + k * integral_s_dd
        dissipation = dissipation + (k / tau) * integral_s2
        force = force + k * s_new
    offsets_new = (torch.stack(new_offsets, dim=1) if model.n_branches
                   else torch.zeros((len(d_new), 0), dtype=dtype, device=device))
    return offsets_new, force, work, dissipation


@torch.no_grad()
def elastic_force(
    model: MaxwellWiechertBond,
    branch_offsets: torch.Tensor,
    elongation: torch.Tensor,
) -> torch.Tensor:
    """Bond force ``F = k0 d + sum k_a (d - d_a)`` at frozen branch state."""
    force = float(model.equilibrium_stiffness) * elongation
    for a, k_a in enumerate(model.branch_stiffness):
        force = force + float(k_a) * (elongation - branch_offsets[:, a])
    return force


@torch.no_grad()
def stored_energy(
    model: MaxwellWiechertBond,
    branch_offsets: torch.Tensor,
    elongation: torch.Tensor,
) -> torch.Tensor:
    """Elastic energy stored in the equilibrium spring plus the branch springs."""
    energy = 0.5 * float(model.equilibrium_stiffness) * elongation.square()
    for a, k_a in enumerate(model.branch_stiffness):
        s = elongation - branch_offsets[:, a]
        energy = energy + 0.5 * float(k_a) * s.square()
    return energy


@torch.no_grad()
def damage_increment(
    force: torch.Tensor,
    reference_force: torch.Tensor,
    dt: float,
    damage_time: float,
    exponent: float,
) -> torch.Tensor:
    """Kachanov damage increment ``dD = dt/t_ref * (|F|/F_ref)**m``."""
    if damage_time <= 0.0 or not math.isfinite(damage_time):
        raise ValueError("damage_time must be finite and positive")
    if exponent <= 0.0 or not math.isfinite(exponent):
        raise ValueError("exponent must be finite and positive")
    ratio = (force.abs() / reference_force.clamp_min(1e-30)).pow(float(exponent))
    return ratio * (dt / float(damage_time))


def weibull_bond_strengths(
    n: int,
    median_force: float,
    modulus: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Draw ``n`` Weibull bond strength forces with the requested median.

    ``P(F_b < f) = 1 - exp(-(f/scale)**m)``; the median is
    ``scale * ln(2)**(1/m)``, so ``scale`` is fixed by ``median_force`` and the
    seed controls only the scatter realisations (acceptance criterion C7).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if median_force <= 0.0 or not math.isfinite(median_force):
        raise ValueError("median_force must be finite and positive")
    if modulus <= 0.0 or not math.isfinite(modulus):
        raise ValueError("modulus must be finite and positive")
    scale = float(median_force) / math.log(2.0) ** (1.0 / float(modulus))
    u = torch.rand(n, generator=generator, dtype=dtype)
    u = u.clamp_min(torch.finfo(dtype).tiny)
    return scale * (-torch.log1p(-u)).pow(1.0 / float(modulus))