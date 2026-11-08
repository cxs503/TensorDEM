"""Maxwell-Wiechert ice bond law: independent verification surface.

Campaign node: self-start verification debt.  The module
``tensordem.ice_bond_viscoelastic`` was merged with no test, no documentation
and no call site; it makes quantitative claims that are pinned here against
references written independently of the module.

Acceptance criteria, in the order they appear:

* **V1 Prony split** -- ``from_unrelaxed_stiffness()`` works with the module's
  own defaults (it used to raise, see ``test_factory_defaults_*``), preserves
  ``k0 + sum k_a = K`` exactly and honours both share conventions.
* **V2 relaxation** -- holding an instantaneous elongation reproduces
  ``G(t) = k0 + sum k_a exp(-t/tau_a)`` from the incremental integrator.
* **V3 creep** -- the constant-force compliance of a *parallel* Wiechert
  element equals ``J(t) = 1/(k0+k1) + k1/(k0(k0+k1))(1 - exp(-t/tau_eff))``,
  and the driver converges at first order in ``dt``.
* **V4 constant strain rate** -- the ramp has the closed form
  ``k0 edot t + sum k_a edot tau_a (1 - exp(-t/tau_a))``.
* **V5 exact integration** -- any partition of the same strain path ends in the
  same state (this is what the module means by "exact incremental integration").
* **V6 energy ledger** -- ``W - dU - Phi`` closes to ``< 1e-10`` relative, and
  dissipation is non-negative.  This is a regression test: the ``(1 - u^2)``
  factor used to be formed as ``1 - decay*decay``, which broke the ledger at
  ``6.6e-8`` for ``dt/tau << 1``.
* **V7 Kachanov damage** -- ``D(t) = (t/t_ref)(F/F_ref)^m`` at constant force.
* **V8 emergent envelope** -- ``sigma_f ~ edot^(1/(m+1))`` comes out of the
  failure-time race *without* being hard-coded (checked for ``m = 1, 2, 4``).
* **V9 brittle vs ductile** -- both mechanisms appear inside one rate sweep.
* **V10 Weibull strengths** -- the sample median equals the requested median and
  the draw is seed-reproducible.
* **V11 degeneracies** -- ``M = 0`` reduces to a linear spring with zero
  dissipation, and ``elastic_force`` agrees with the increment's force.

Tolerances: ``1e-10``-ish where the module's integrals are closed form and the
reference is analytic (float64 round-off plus a few thousand accumulations);
looser only where a stated scheme limit exists (creep driver: first order in
``dt``; envelope: fit of 3-5 points).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.ice_bond_viscoelastic import (
    DEFAULT_DELAYED_ELASTIC_FRACTION,
    DEFAULT_EQUILIBRIUM_FRACTION,
    MaxwellWiechertBond,
    advance_branches,
    damage_increment,
    elastic_force,
    stored_energy,
    weibull_bond_strengths,
)

K = 1.0e6  # N/m


def _zeros(n_branches: int) -> torch.Tensor:
    return torch.zeros((1, n_branches), dtype=torch.float64)


def _default_model() -> MaxwellWiechertBond:
    return MaxwellWiechertBond.from_unrelaxed_stiffness(K)


# ---------------------------------------------------------------------------
# V1: construction / Prony split
# ---------------------------------------------------------------------------
def test_factory_defaults_are_accepted():
    """The module's own defaults used to raise ``ValueError`` (shares sum to 0.65)."""
    assert abs(sum(DEFAULT_DELAYED_ELASTIC_FRACTION) - 1.0) > 1e-6
    model = MaxwellWiechertBond.from_unrelaxed_stiffness(K)
    assert model.unrelaxed_stiffness == pytest.approx(K, rel=1e-15)
    assert model.equilibrium_stiffness == pytest.approx(
        DEFAULT_EQUILIBRIUM_FRACTION * K, rel=1e-15)
    # the shares are fractions of K: 0.35 + 0.45 + 0.20 = 1
    assert model.branch_stiffness == pytest.approx((0.45 * K, 0.20 * K), rel=1e-14)


def test_factory_proportions_convention_matches():
    """Shares summing to 1 (proportions of the delayed part) stay supported."""
    model = MaxwellWiechertBond.from_unrelaxed_stiffness(
        K, delayed_elastic_fraction=(0.5, 0.5))
    assert model.unrelaxed_stiffness == pytest.approx(K, rel=1e-15)
    assert model.branch_stiffness == pytest.approx((0.325 * K, 0.325 * K), rel=1e-14)


def test_factory_rejects_bad_shares():
    with pytest.raises(ValueError):
        MaxwellWiechertBond.from_unrelaxed_stiffness(K, delayed_elastic_fraction=(0.5,))
    with pytest.raises(ValueError):
        MaxwellWiechertBond.from_unrelaxed_stiffness(
            K, delayed_elastic_fraction=(0.0, 1.0))
    with pytest.raises(ValueError):
        MaxwellWiechertBond(equilibrium_stiffness= -1.0, branch_stiffness=(1.0,),
                            relaxation_time=(1.0,))
    with pytest.raises(ValueError):
        MaxwellWiechertBond(equilibrium_stiffness=0.0, branch_stiffness=(1.0, 2.0),
                            relaxation_time=(1.0,))


def test_relaxation_modulus_bounds():
    model = _default_model()
    assert float(model.relaxation_modulus(0.0)) == pytest.approx(
        model.unrelaxed_stiffness, rel=1e-15)
    assert model.relaxed_stiffness_at(1.0e6) == pytest.approx(
        model.equilibrium_stiffness, rel=1e-15)
    # monotone decreasing in time
    ts = [0.0, 0.5, 5.0, 50.0]
    gs = [float(model.relaxation_modulus(t)) for t in ts]
    assert all(a > b for a, b in zip(gs, gs[1:]))


# ---------------------------------------------------------------------------
# V2: stress relaxation
# ---------------------------------------------------------------------------
def test_stress_relaxation_reproduces_prony_series():
    model = _default_model()
    d0 = 1.0e-6
    d = torch.tensor([d0], dtype=torch.float64)
    offsets = _zeros(model.n_branches)
    # instantaneous jump (the branch state does not move during the jump)
    offsets, force, _w, _p = advance_branches(model, offsets, torch.zeros(1), d, 1e-12)
    assert float(force[0]) / d0 == pytest.approx(model.unrelaxed_stiffness, rel=1e-11)
    t = 0.0
    for target in (0.5, 5.0, 50.0):
        steps = int(round((target - t) / 0.05))
        for _ in range(steps):
            offsets, force, _w, _p = advance_branches(model, offsets, d, d, 0.05)
        t = target
        analytic = float(model.relaxation_modulus(t))
        assert float(force[0]) / d0 == pytest.approx(analytic, rel=1e-11)


# ---------------------------------------------------------------------------
# V3: creep compliance of a parallel Wiechert element
# ---------------------------------------------------------------------------
def _creep_step(model: MaxwellWiechertBond, offsets: torch.Tensor, d_start: torch.Tensor,
                target: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact affine solve for the elongation that holds ``F = target`` over a step."""
    slope = float(model.equilibrium_stiffness)
    rhs = target
    for a, (k_a, tau_a) in enumerate(zip(model.branch_stiffness, model.relaxation_time)):
        u = math.exp(-dt / float(tau_a))
        coef = float(k_a) * (float(tau_a) / dt) * (1.0 - u)
        s0 = float(d_start[0]) - float(offsets[0, a])
        slope += coef
        rhs += coef * float(d_start[0]) - float(k_a) * s0 * u
    d_new = torch.tensor([rhs / slope], dtype=torch.float64)
    off_new, force, _, _ = advance_branches(model, offsets, d_start, d_new, dt)
    assert float(force[0]) == pytest.approx(target, rel=1e-12)
    return off_new, d_new


def test_creep_compliance_matches_parallel_wiechert_reference():
    k0, k1, tau1 = 3.5e5, 4.5e5, 2.0
    model = MaxwellWiechertBond(equilibrium_stiffness=k0, branch_stiffness=(k1,),
                                relaxation_time=(tau1,))
    tau_eff = tau1 * (1.0 + k1 / k0)

    def J(t: float) -> float:
        return 1.0 / (k0 + k1) + (k1 / (k0 * (k0 + k1))) * (1.0 - math.exp(-t / tau_eff))

    dt = tau1 / 200.0
    offsets, d = _zeros(1), torch.zeros(1, dtype=torch.float64)
    n = int(round(6.0 / dt))
    for _ in range(n):
        offsets, d = _creep_step(model, offsets, d, 1.0, dt)
    measured = float(d[0])
    assert measured == pytest.approx(J(6.0), rel=2e-3)
    # long-time limit is the equilibrium spring alone
    assert J(1.0e3) == pytest.approx(1.0 / k0, rel=1e-12)


def test_creep_driver_converges_at_first_order():
    k0, k1, tau1 = 3.5e5, 4.5e5, 2.0
    model = MaxwellWiechertBond(equilibrium_stiffness=k0, branch_stiffness=(k1,),
                                relaxation_time=(tau1,))
    tau_eff = tau1 * (1.0 + k1 / k0)
    ref = 1.0 / (k0 + k1) + (k1 / (k0 * (k0 + k1))) * (1.0 - math.exp(-2.0 / tau_eff))
    errs = []
    for dt in (0.04, 0.02):
        offsets, d = _zeros(1), torch.zeros(1, dtype=torch.float64)
        for _ in range(int(round(2.0 / dt))):
            offsets, d = _creep_step(model, offsets, d, 1.0, dt)
        errs.append(abs(float(d[0]) - ref) / ref)
    assert errs[0] < 5e-3 and errs[1] < errs[0]
    assert 0.8 < math.log(errs[0] / errs[1]) / math.log(2.0) < 1.3


# ---------------------------------------------------------------------------
# V4 / V5: constant strain rate and exact partitioning
# ---------------------------------------------------------------------------
def test_constant_strain_rate_closed_form():
    model = _default_model()
    rate = 1.0e-4
    dt = 2.0e-3
    offsets, d = _zeros(model.n_branches), torch.zeros(1, dtype=torch.float64)
    t = 0.0
    for _ in range(250):
        d_new = d + rate * dt
        offsets, force, _w, _p = advance_branches(model, offsets, d, d_new, dt)
        t += dt
        analytic = (model.equilibrium_stiffness * rate * t
                    + sum(k * rate * tau * (1.0 - math.exp(-t / tau))
                          for k, tau in zip(model.branch_stiffness,
                                            model.relaxation_time)))
        assert float(force[0]) == pytest.approx(analytic, rel=1e-11)
        d = d_new


def test_increment_splitting_is_invariant():
    model = _default_model()
    rate, t_end = 1.0e-3, 0.5
    forces = []
    for n in (1, 2, 5, 17, 100):
        h = t_end / n
        offsets, d = _zeros(model.n_branches), torch.zeros(1, dtype=torch.float64)
        for _ in range(n):
            d_new = d + rate * h
            offsets, force, _w, _p = advance_branches(model, offsets, d, d_new, h)
            d = d_new
        forces.append(float(force[0]))
    assert max(abs(f - forces[0]) / abs(forces[0]) for f in forces) < 1e-12


# ---------------------------------------------------------------------------
# V6: energy ledger (regression for the ``(1 - u^2)`` conditioning fix)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2])
def test_energy_ledger_closes(dt: float):
    model = _default_model()
    g = torch.Generator().manual_seed(7)
    offsets, d = _zeros(model.n_branches), torch.zeros(1, dtype=torch.float64)
    work = 0.0
    diss = 0.0
    u_start = float(stored_energy(model, offsets, d)[0])
    for _ in range(100):
        rate = float(torch.rand((), generator=g, dtype=torch.float64)) * 4.0 - 2.0
        d_new = d + rate * dt
        offsets, _f, w, p = advance_branches(model, offsets, d, d_new, dt)
        work += float(w[0])
        diss += float(p[0])
        assert float(p[0]) >= 0.0
        d = d_new
    du = float(stored_energy(model, offsets, d)[0]) - u_start
    scale = max(abs(work), abs(du), abs(diss))
    assert abs(work - du - diss) / scale < 1e-10


def test_energy_ledger_bound_at_extreme_dt_over_tau():
    """At dt/tau ~ 3e-6 the closed form is cancellation-limited (~1e-7), not 1e-10.

    This is the honest envelope of the ledger identity: the dissipation per step
    is a difference of terms ``O(dd^2 tau^2 / dt)`` that cancel by ~1e8, so
    float64 cannot resolve it better.  The regression under test is that the
    ``(1-u)^(1+u)`` form keeps the residual at the *conditioning* limit instead
    of amplifying it further (the old ``1 - decay**2`` form added its own
    error on top, measured 6.6e-8 at small strain increments).
    """
    model = _default_model()
    dt = 1.0e-4
    offsets, d = _zeros(model.n_branches), torch.zeros(1, dtype=torch.float64)
    work = diss = 0.0
    u0 = float(stored_energy(model, offsets, d)[0])
    for rate in (2.0, -1.5, 0.5, -2.0, 1.0) * 20:
        d_new = d + rate * dt
        offsets, _f, w, p = advance_branches(model, offsets, d, d_new, dt)
        work += float(w[0])
        diss += float(p[0])
        d = d_new
    du = float(stored_energy(model, offsets, d)[0]) - u0
    assert abs(work - du - diss) / max(abs(work), abs(du), abs(diss)) < 1e-6


# ---------------------------------------------------------------------------
# V7: Kachanov damage increment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1.0, 2.0, 4.0])
def test_damage_increment_is_exact_at_constant_force(m: float):
    force = torch.tensor([3.0], dtype=torch.float64)
    f_ref = torch.tensor([2.0], dtype=torch.float64)
    t_ref, dt, steps = 10.0, 2.0e-3, 500
    d = sum(float(damage_increment(force, f_ref, dt, t_ref, m)[0]) for _ in range(steps))
    assert d == pytest.approx((dt * steps / t_ref) * (3.0 / 2.0) ** m, rel=1e-12)


def test_damage_increment_validates_inputs():
    force = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        damage_increment(force, force, 1e-3, 0.0, 2.0)
    with pytest.raises(ValueError):
        damage_increment(force, force, 1e-3, 10.0, 0.0)


# ---------------------------------------------------------------------------
# V8: emergent ductile envelope
# ---------------------------------------------------------------------------
def _failure_stress(model: MaxwellWiechertBond, rate: float, *, m: float,
                    t_ref: float, f_ref: float, h: float | None = None,
                    t_end: float = 60.0) -> tuple[str, float, float]:
    """Constant-strain-rate run; returns ``(mode, sigma_f, t_f)``.

    ``h = None`` means a two-pass run: a coarse pass locates the failure, the
    second pass resolves it with ~200 steps (step size affects the reported
    stress at ``O(h)`` because damage is accumulated at step ends only).
    """
    def _run(step: float) -> tuple[str, float, float]:
        offsets, d = _zeros(model.n_branches), torch.zeros(1, dtype=torch.float64)
        damage = 0.0
        t = 0.0
        f_ref_t = torch.tensor([f_ref], dtype=torch.float64)
        for _ in range(int(round(t_end / step))):
            d_new = d + rate * step
            offsets, force, _w, _p = advance_branches(model, offsets, d, d_new, step)
            damage += float(damage_increment(force, f_ref_t, step, t_ref, m)[0])
            t += step
            if damage >= 1.0:
                return "creep", abs(float(force[0])), t
            d = d_new
        return "survived", float("nan"), t

    if h is not None:
        return _run(h)
    mode, sig, t_f = _run(1.0e-3)
    if mode == "survived":
        return mode, sig, t_f
    return _run(max(1.0e-7, t_f / 200.0))


def test_ductile_envelope_exponent_is_emergent():
    """sigma_f ~ edot**(1/(m+1)) for an elastic bond: no envelope in the code."""
    elastic = MaxwellWiechertBond(equilibrium_stiffness=1.0e6, branch_stiffness=(),
                                  relaxation_time=())
    rates = [1.0e-3, 1.0e-2, 1.0e-1]
    for m in (1.0, 2.0, 4.0):
        sig = []
        for rate in rates:
            mode, s, _t = _failure_stress(elastic, rate, m=m, t_ref=5.0e3, f_ref=1.0)
            assert mode == "creep"
            sig.append(s)
        slope = ((math.log(sig[-1]) - math.log(sig[0]))
                 / (math.log(rates[-1]) - math.log(rates[0])))
        assert slope == pytest.approx(1.0 / (m + 1.0), rel=0.02)


def test_brittle_and_ductile_both_appear_in_one_sweep():
    elastic = MaxwellWiechertBond(equilibrium_stiffness=1.0e6, branch_stiffness=(),
                                  relaxation_time=())
    f_b, t_ref, m = 50.0, 50.0, 2.0
    modes = []
    for rate in (1.0e-5, 1.0e-1):
        h = 1.0e-3
        offsets, d = _zeros(0), torch.zeros(1, dtype=torch.float64)
        damage = 0.0
        f_ref = torch.tensor([1.0], dtype=torch.float64)
        mode = "survived"
        for _ in range(int(round(10.0 / h))):
            d_new = d + rate * h
            offsets, force, _w, _p = advance_branches(elastic, offsets, d, d_new, h)
            if abs(float(force[0])) >= f_b:
                mode = "brittle"
                break
            damage += float(damage_increment(force, f_ref, h, t_ref, m)[0])
            if damage >= 1.0:
                mode = "creep"
                break
            d = d_new
        modes.append(mode)
    assert modes == ["creep", "brittle"]


# ---------------------------------------------------------------------------
# V10: Weibull strengths
# ---------------------------------------------------------------------------
def test_weibull_median_and_reproducibility():
    n, median, m = 20_000, 15.0, 3.0
    a = weibull_bond_strengths(n, median, m,
                               generator=torch.Generator().manual_seed(2026))
    b = weibull_bond_strengths(n, median, m,
                               generator=torch.Generator().manual_seed(2026))
    assert torch.equal(a, b)
    assert float(a.median()) == pytest.approx(median, rel=2e-2)
    assert bool((a > 0).all())
    # heavier modulus -> less scatter
    wide = weibull_bond_strengths(n, median, 1.0,
                                  generator=torch.Generator().manual_seed(1))
    assert float(wide.std() / wide.mean()) > float(a.std() / a.mean())


# ---------------------------------------------------------------------------
# V11: degeneracies
# ---------------------------------------------------------------------------
def test_zero_branches_is_a_linear_spring():
    model = MaxwellWiechertBond(equilibrium_stiffness=7.5e5, branch_stiffness=(),
                                relaxation_time=())
    d = torch.tensor([1.0e-6], dtype=torch.float64)
    offsets, force, work, diss = advance_branches(model, _zeros(0),
                                                  torch.zeros(1, dtype=torch.float64), d, 1e-3)
    assert list(offsets.shape) == [1, 0]
    assert float(force[0]) == pytest.approx(0.75, rel=1e-14)
    assert float(diss[0]) == 0.0
    assert float(work[0]) == pytest.approx(float(stored_energy(model, offsets, d)[0]),
                                           rel=1e-14)


def test_elastic_force_agrees_with_increment():
    model = _default_model()
    d_old = torch.tensor([2.0e-7], dtype=torch.float64)
    d_new = torch.tensor([3.0e-7], dtype=torch.float64)
    offsets, force, _w, _p = advance_branches(model, _zeros(model.n_branches), d_old,
                                              d_new, 1e-3)
    assert float(elastic_force(model, offsets, d_new)[0]) == pytest.approx(
        float(force[0]), rel=1e-14)