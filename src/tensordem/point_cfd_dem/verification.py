"""Verification faces for the point-particle CFD-DEM coupling.

Five machine-checkable gates, all against closed-form references:

============  =========================================================  ==========
gate          reference                                                  threshold
============  =========================================================  ==========
``stokes_terminal``      ``v_t = (rho_p-rho_f) g d^2 / (18 mu)``         2 %
``added_mass_frequency`` ``omega = sqrt(k/(m_p + C_A m_a))``            5 %
``momentum_exchange``    ``d(sum m_p u_p + p_fluid)/dt = F_ext``         1e-6
``richardson_zaki``      ``u_s = u_t eps^n``, n from Richardson-Zaki    10 %
``pressure_gradient``    ``F_pg = -V_p grad p`` (analytic mode)          1e-10 / 2 %
============  =========================================================  ==========

The numbers produced here are the ones reported in
``docs/point-particle-cfd-dem.md`` and ``benchmarks/point_particle_cfd_dem``.
"""

from __future__ import annotations

import math

import numpy as np

from . import closures
from .coupling import PointParticleCFDDEM
from .fluid import SpectralPeriodicFluid
from .particles import ParticleSet, PointParticleModel

__all__ = [
    "gate",
    "stokes_terminal_velocity_case",
    "added_mass_oscillation_case",
    "momentum_exchange_case",
    "richardson_zaki_case",
    "pressure_gradient_case",
    "run_all",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def gate(name, criterion, measured, threshold, evidence=""):
    """Build a gate record (logical structure shared by all cases)."""
    value = float(measured)
    return {
        "gate": name,
        "criterion": criterion,
        "measured": value,
        "threshold": float(threshold),
        "passed": bool(abs(value) <= float(threshold)),
        "evidence": evidence,
    }


def _richardson_zaki_hindrance(particles, voidage, model=None):
    """RZ drag multiplier ``eps**(1-n)`` with ``n`` from the single-particle Re."""
    rho_f, mu = model.rho_f, model.mu
    d = particles.diameter.mean()
    rho_p = particles.density.mean()
    # single-particle terminal Reynolds number (Stokes terminal velocity)
    v_t = closures.terminal_velocity_stokes(
        rho_p, rho_f, abs(model.gravity[2]), d, mu)
    Re_t = float(closures.particle_reynolds(rho_f, mu, d, v_t))
    n = closures.richardson_zaki_exponent(Re_t)
    return closures.richardson_zaki_hindrance(voidage, n)


def _zero_crossing_frequency(t, y):
    """Frequency from linear-interpolated zero crossings of ``y(t)``."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    crossings = []
    for i in range(len(y) - 1):
        if y[i] == 0.0:
            crossings.append(t[i])
        elif y[i] * y[i + 1] < 0.0:
            w = y[i] / (y[i] - y[i + 1])
            crossings.append(t[i] + w * (t[i + 1] - t[i]))
    if len(crossings) < 3:
        return math.nan, crossings
    # half period between successive crossings
    half = np.diff(np.asarray(crossings))
    period = 2.0 * float(np.mean(half))
    return 2.0 * math.pi / period, crossings


# ---------------------------------------------------------------------------
# gate 1: single sphere Stokes terminal velocity (two-way)
# ---------------------------------------------------------------------------
def stokes_terminal_velocity_case(diameter=1.0, rho_p=2.0, rho_f=1.0, mu=0.5,
                                  g=0.5, cells=16, h=2.0, dt=0.01,
                                  t_end=2.0, window=0.4,
                                  drag_filter_width=None,
                                  self_response="spectral",
                                  baseline_filter_width=None,
                                  baseline_self_response=None):
    """One sphere settling in a periodic box, two-way coupled.

    Reference: the unbounded Stokes terminal velocity
    ``v_t = (rho_p-rho_f) g d^2 / (18 mu)``.

    The grid is deliberately coarse (``h = 2 d``, i.e. the point-particle
    regime) and the box is ``32 d`` wide, so the *physical* finite-domain effect
    (``3 pi d / L ~ 0.3 %``) is far below the 2 % gate.  Two model artefacts are
    handled explicitly and reported separately:

    * ``self_response="spectral"`` -- the particle's own induced flow
      ``R(t) * (-F_drag)`` is removed from the velocity that enters the drag
      (time-dependent, discretisation-consistent self-response; see
      :meth:`tensordem.point_cfd_dem.fluid.SpectralPeriodicFluid.self_response`).
      The drag relation is then solved implicitly, ``C_eff = C/(1 - C R)``,
      which makes the steady terminal velocity *exactly* the analytic one
      independently of ``R``;
    * ``self_disturbance`` -- removes the velocity the particle induces on
      itself within the current time step (exact adjoint self-weight sum).

    The same case is repeated with both corrections switched off; that
    "uncorrected point-particle" run is the discriminating baseline of the gate.
    """
    v_t = closures.terminal_velocity_stokes(rho_p, rho_f, g, diameter, mu)
    tau_p = closures.relaxation_time_stokes(rho_p, diameter, mu)

    def run(filter_width, self_dist, response):
        fluid = SpectralPeriodicFluid((cells,) * 3, h, rho=rho_f, mu=mu,
                                      advect=True)
        centre = np.array(fluid.extent) / 2.0
        part = ParticleSet.create([centre], [[0.0, 0.0, 0.0]],
                                  radius=diameter / 2.0, density=rho_p)
        model = PointParticleModel(rho_f=rho_f, mu=mu,
                                   gravity=np.array([0.0, 0.0, -g]))
        sim = PointParticleCFDDEM(fluid, part, model, zero_net_flux=True,
                                  self_disturbance=self_dist,
                                  exchange_terms=("drag",),
                                  drag_filter_width=filter_width,
                                  self_response=response)
        n_steps = int(round(t_end / dt))
        t = np.empty(n_steps + 1)
        vz = np.empty(n_steps + 1)
        ufz = np.empty(n_steps + 1)
        t[0], vz[0], ufz[0] = 0.0, 0.0, 0.0
        for k in range(n_steps):
            sim.step(dt)
            t[k + 1] = sim.time
            vz[k + 1] = part.velocity[0, 2]
            ufz[k + 1] = fluid.mean_velocity(sim.u)[2]
        return t, vz, ufz, part, fluid, sim

    t, vz, ufz, part, fluid, sim = run(drag_filter_width, True, self_response)
    steady = t >= (t_end - window)
    v_meas = abs(float(np.mean(vz[steady])))
    v_rel = abs(float(np.mean(vz[steady] - ufz[steady])))
    rel_err = abs(v_meas - v_t) / v_t

    _, vz0, _, _, _, _ = run(baseline_filter_width, False,
                             baseline_self_response)
    v_base = abs(float(np.mean(vz0[steady])))
    err_base = abs(v_base - v_t) / v_t

    phi = float(part.volume.sum()) / (
        fluid.cell_volume * int(np.prod(fluid.shape)))
    eps = 1.0 - phi
    rec = gate(
        "stokes_terminal_velocity",
        "|U_container - v_t| / v_t <= 2 %  with v_t = "
        "(rho_p-rho_f) g d^2 / (18 mu)",
        rel_err, 0.02,
        f"U={v_meas:.6f}, v_t={v_t:.6f}, Re_t="
        f"{float(closures.particle_reynolds(rho_f, mu, diameter, v_t)):.4f}; "
        f"uncorrected baseline U={v_base:.6f} (err={err_base:.3f})",
    )
    return {
        "gate": rec,
        "v_terminal_analytic": v_t,
        "v_terminal_measured": v_meas,
        "v_relative_measured": v_rel,
        "rel_error": rel_err,
        "rel_error_uncorrected": err_base,
        "v_terminal_uncorrected": v_base,
        "baseline_configuration": "no self-response correction, no drag filter",
        "relaxation_time": tau_p,
        "reynolds": float(closures.particle_reynolds(rho_f, mu, diameter, v_t)),
        "voidage": eps,
        "grid": list(fluid.shape),
        "h": h,
        "box_length_in_diameters": float(fluid.extent[0] / diameter),
        "drag_filter_width": sim.drag_filter_width,
        "self_response_used": self_response,
        "self_disturbance_correction": True,
        "time": t.tolist(),
        "vz": vz.tolist(),
        "ufz": ufz.tolist(),
        "vz_uncorrected": vz0.tolist(),
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# gate 2: added-mass oscillation
# ---------------------------------------------------------------------------
def added_mass_oscillation_case(diameter=1.0, rho_p=2.0, rho_f=1.0, mu=0.0,
                                added_mass=0.5, omega_target=1.0, amplitude=0.2,
                                cells=8, h=1.0, dt=0.01, periods=2.5):
    """Sphere on a spring in an inviscid fluid: ``omega = sqrt(k/(m_p+C_A m_a))``.

    The added-mass force is integrated implicitly, so the discrete frequency is
    the analytic one up to the O((omega dt)^2) time-integration error.  The same
    case is repeated with ``C_A = 0`` to show that the gate actually
    discriminates (a model without added mass is ~12 % off, i.e. FAIL).
    """
    vol = closures.sphere_volume(diameter)
    m_p = vol * rho_p
    m_a = added_mass * rho_f * vol
    m_eff = m_p + m_a
    k = m_eff * omega_target ** 2
    omega_theory = math.sqrt(k / m_eff)

    def run(c_a):
        fluid = SpectralPeriodicFluid((cells,) * 3, h, rho=rho_f,
                                      mu=mu, advect=False)
        x0 = np.array(fluid.extent) / 2.0
        pos0 = x0 + np.array([0.0, 0.0, amplitude])
        part = ParticleSet.create([pos0], [[0.0, 0.0, 0.0]],
                                  radius=diameter / 2.0, density=rho_p)
        model = PointParticleModel(rho_f=rho_f, mu=mu,
                                   gravity=np.array([0.0, 0.0, 0.0]),
                                   added_mass=c_a)
        sim = PointParticleCFDDEM(fluid, part, model, zero_net_flux=False,
                                  self_disturbance=True,
                                  exchange_terms=("drag", "added_mass"))

        def spring(s):
            return -k * (s.particles.position - x0[None, :])

        n_steps = int(round(periods * (2.0 * math.pi / omega_theory) / dt))
        t = np.empty(n_steps + 1)
        y = np.empty(n_steps + 1)
        t[0] = 0.0
        y[0] = part.position[0, 2] - x0[2]
        for n in range(n_steps):
            sim.step(dt, external_force=spring(sim))
            t[n + 1] = sim.time
            y[n + 1] = part.position[0, 2] - x0[2]
        omega_meas, crossings = _zero_crossing_frequency(t, y)
        return {
            "added_mass": c_a,
            "omega_measured": omega_meas,
            "periods": periods,
            "time": t.tolist(),
            "displacement": y.tolist(),
            "crossings": [float(c) for c in crossings],
            "amplitude_initial": amplitude,
            "amplitude_final": float(np.max(np.abs(y[-len(y) // 4:]))),
            "m_eff": m_p + c_a * rho_f * vol,
        }

    with_am = run(added_mass)
    without = run(0.0)
    rel_err = abs(with_am["omega_measured"] - omega_theory) / omega_theory
    counter_err = abs(without["omega_measured"] - omega_theory) / omega_theory

    rec = gate(
        "added_mass_oscillation_frequency",
        "|omega_num - sqrt(k/(m_p + C_A m_a))| / omega <= 5 %",
        rel_err, 0.05,
        f"omega_num={with_am['omega_measured']:.6f}, "
        f"omega_theory={omega_theory:.6f}, C_A={added_mass}, "
        f"counterfactual C_A=0 error={counter_err:.4f}",
    )
    return {
        "gate": rec,
        "omega_theory": omega_theory,
        "omega_measured": with_am["omega_measured"],
        "rel_error": rel_err,
        "counterfactual_no_added_mass_error": counter_err,
        "stiffness": k,
        "mass_particle": m_p,
        "added_mass_mass": m_a,
        "effective_mass": m_eff,
        "num_periods": periods,
        "dt": dt,
        "displacement": with_am["displacement"],
        "time": with_am["time"],
        "displacement_no_added_mass": without["displacement"],
        "time_no_added_mass": without["time"],
        "amplitude_initial": amplitude,
    }


# ---------------------------------------------------------------------------
# gate 3: global interphase momentum conservation
# ---------------------------------------------------------------------------
def momentum_exchange_case(n_particles=24, rho_f=1.0, mu=0.2, rho_p=5.0,
                           diameter=0.6, cells=12, h=0.5, dt=0.004,
                           n_steps=100, seed=7, spread_mode="adjoint",
                           exchange_scale=1.0):
    """Closed (periodic) box, no gravity: only drag is exchanged.

    Then ``sum_p m_p u_p + p_fluid`` must be constant to machine precision.
    The gate is repeated with a deliberately miss-scaled exchange
    (``exchange_scale=1.02``, i.e. the fluid receives 2 % more momentum than the
    particles lose) so that the test is shown to be able to fail.  Note that a
    non-adjoint (nearest-node) spreading operator does *not* break total
    momentum conservation -- it only breaks the local adjointness -- so it is
    not a valid negative control for this gate.
    """
    def run(mode, scale=1.0):
        fluid = SpectralPeriodicFluid((cells,) * 3, h, rho=rho_f,
                                      mu=mu, advect=True)
        rng = np.random.default_rng(seed)
        pos = rng.uniform(0.0, 1.0, size=(n_particles, 3)) * fluid.extent
        vel = rng.normal(scale=0.05, size=(n_particles, 3))
        part = ParticleSet.create(pos, vel, radius=diameter / 2.0, density=rho_p)
        # added mass is switched off here: only the drag is exchanged, so the
        # particle + fluid momentum must balance to round-off
        model = PointParticleModel(rho_f=rho_f, mu=mu, added_mass=0.0,
                                   gravity=np.array([0.0, 0.0, 0.0]))
        sim = PointParticleCFDDEM(fluid, part, model, zero_net_flux=False,
                                  self_disturbance=True,
                                  exchange_terms=("drag",), spread_mode=mode,
                                  exchange_scale=scale)
        budget0 = sim.momentum_budget()
        for _ in range(n_steps):
            sim.step(dt)
        budget = sim.momentum_budget()
        return sim, budget, budget0

    sim, budget, budget0 = run(spread_mode)
    residual = budget["residual"]
    _, budget_biased, _ = run("adjoint", scale=1.02)
    residual_biased = budget_biased["residual"]

    rec = gate(
        "interphase_momentum_conservation",
        "|d(sum m_p u_p) + d(p_fluid)| / max(|d p_p|,|d p_f|) <= 1e-6",
        residual, 1e-6,
        f"adjoint residual={residual:.3e}; "
        f"+2 % exchange-bias counterfactual residual={residual_biased:.3e}",
    )
    return {
        "gate": rec,
        "residual_adjoint": residual,
        "residual_exchange_bias_2pct": residual_biased,
        "budget": budget,
        "budget_initial": budget0,
        "history_time": [float(x) for x in sim.history["time"]],
        "history_momentum_particle": [list(map(float, x)) for x in
                                      sim.history["momentum_particle"]],
        "history_momentum_fluid": [list(map(float, x)) for x in
                                   sim.history["momentum_fluid"]],
        "particles": n_particles,
        "steps": n_steps,
        "dt": dt,
        "energy_particle": float(sim.particles.kinetic_energy()),
        "energy_fluid": float(sim.fluid.kinetic_energy(sim.u)),
    }


# ---------------------------------------------------------------------------
# gate 4: Richardson-Zaki hindered settling
# ---------------------------------------------------------------------------
def richardson_zaki_case(diameter=1.0, rho_p=2.0, rho_f=1.0, mu=0.5, g=0.5,
                         lattice=(8, 8, 8), jitter=0.15, cells=12, h=1.5,
                         dt=0.005, t_end=2.0, window=0.6, seed=11,
                         voidage_mode="global", drag_filter_width=None,
                         baseline_filter_width=None, self_response=None):
    """Multi-particle settling in a periodic box, container (zero-flux) frame.

    Reference: Richardson & Zaki, ``u_s / u_t = eps**n`` with ``eps`` the
    measured voidage and ``n`` from the single-particle terminal Reynolds
    number.  The drag closure carries the matching hindrance ``eps**(1-n)``
    (first-principles derivation in
    :func:`tensordem.point_cfd_dem.closures.richardson_zaki_hindrance`).

    Resolution note: the grid is ``h = 1.5 d`` because the self-response
    correction ``C/(1-C R)`` is ill conditioned for ``h <= d`` (see
    :meth:`SpectralPeriodicFluid.self_response`); ``h = d`` gives a larger
    deviation (see ``sensitivity.richardson_zaki_resolution`` in
    ``result.json``).
    """
    vol = closures.sphere_volume(diameter)
    fluid = SpectralPeriodicFluid((cells,) * 3, h, rho=rho_f, mu=mu, advect=True)
    nx, ny, nz = lattice
    extent = fluid.extent
    base = np.stack(np.meshgrid(
        (np.arange(nx) + 0.5) * extent[0] / nx,
        (np.arange(ny) + 0.5) * extent[1] / ny,
        (np.arange(nz) + 0.5) * extent[2] / nz,
        indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    base = base + rng.uniform(-jitter, jitter, size=base.shape) * diameter
    n = base.shape[0]

    v_t = closures.terminal_velocity_stokes(rho_p, rho_f, g, diameter, mu)
    phi = n * vol / (fluid.cell_volume * int(np.prod(fluid.shape)))
    eps = 1.0 - phi
    Re_t = float(closures.particle_reynolds(rho_f, mu, diameter, v_t))
    rz_n = float(closures.richardson_zaki_exponent(Re_t))
    u_rz = v_t * eps ** rz_n

    def run(filter_width):
        part = ParticleSet.create(base.copy(), np.zeros((n, 3)),
                                  radius=diameter / 2.0, density=rho_p)
        model = PointParticleModel(rho_f=rho_f, mu=mu,
                                   gravity=np.array([0.0, 0.0, -g]),
                                   hindrance=_richardson_zaki_hindrance)
        sim = PointParticleCFDDEM(fluid.__class__((cells,) * 3, h, rho=rho_f,
                                                  mu=mu, advect=True),
                                  part, model, zero_net_flux=True,
                                  self_disturbance=True,
                                  exchange_terms=("drag",),
                                  voidage_mode=voidage_mode,
                                  drag_filter_width=filter_width,
                                  self_response=self_response)
        n_steps = int(round(t_end / dt))
        t = np.empty(n_steps + 1)
        vz = np.empty(n_steps + 1)
        vfz = np.empty(n_steps + 1)
        t[0], vz[0], vfz[0] = 0.0, 0.0, 0.0
        for k in range(n_steps):
            sim.step(dt)
            t[k + 1] = sim.time
            vz[k + 1] = float(np.mean(sim.particles.velocity[:, 2]))
            vfz[k + 1] = float(sim.fluid.mean_velocity(sim.u)[2])
        return t, vz, vfz, sim

    t, vz, vfz, sim = run(drag_filter_width)
    steady = t >= (t_end - window)
    u_meas = abs(float(np.mean(vz[steady])))
    u_fluid = abs(float(np.mean(vfz[steady])))
    rel_err = abs(u_meas - u_rz) / u_rz

    _, vz0, _, _ = run(baseline_filter_width)
    err_base = abs(abs(float(np.mean(vz0[steady]))) - u_rz) / u_rz

    rec = gate(
        "richardson_zaki_settling",
        "|u_s,container - u_t eps^n| / (u_t eps^n) <= 10 %",
        rel_err, 0.10,
        f"u_s={u_meas:.6f}, u_t eps^n={u_rz:.6f}, eps={eps:.4f}, n={rz_n:.4f}, "
        f"phi={phi:.4f}, N={n}; uncorrected baseline err={err_base:.4f}",
    )
    return {
        "gate": rec,
        "v_terminal_single": v_t,
        "settling_velocity_measured": u_meas,
        "settling_velocity_rz": u_rz,
        "rel_error": rel_err,
        "rel_error_uncorrected": err_base,
        "voidage": eps,
        "solids_fraction": phi,
        "rz_exponent": rz_n,
        "reynolds_terminal": Re_t,
        "hindrance_factor": float(eps ** (1.0 - rz_n)),
        "fluid_mean_velocity": u_fluid,
        "particle_count": int(n),
        "grid": list(fluid.shape),
        "h": h,
        "drag_filter_width": sim.drag_filter_width,
        "self_response_used": self_response,
        "dt": dt,
        "t_end": t_end,
        "time": t.tolist(),
        "vz_mean": vz.tolist(),
        "vz_fluid": vfz.tolist(),
        "vz_uncorrected": vz0.tolist(),
    }


# ---------------------------------------------------------------------------
# gate 5: pressure-gradient operator / force
# ---------------------------------------------------------------------------
def pressure_gradient_case(cells=24, h=0.5, amplitude=2.0, n_modes=1,
                           seed=3, n_particles=8):
    """Analytic mode ``p = A sin(2 pi z / L)``.

    Two checks: (i) the spectral gradient on the grid reproduces the analytic
    derivative (machine precision), (ii) the trilinear interpolation of the
    gradient at the particle centres gives ``F_pg = -V_p grad p`` to the
    interpolation error, ``O((k h)^2)``.
    """
    fluid = SpectralPeriodicFluid((cells,) * 3, h)
    L = fluid.extent[2]
    k = 2.0 * math.pi * n_modes / L
    z = np.arange(fluid.shape[2]) * h
    p = np.broadcast_to(
        amplitude * np.sin(k * z)[None, None, :], fluid.shape).copy()

    grad = fluid.gradient(p)
    analytic = np.zeros_like(grad)
    zz = z[None, None, :]
    analytic[2] = (amplitude * k * np.cos(k * zz))[np.newaxis, :, :] * np.ones(
        (fluid.shape[0], fluid.shape[1], 1))
    op_err = float(np.max(np.abs(grad - analytic)) / (amplitude * k))

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n_particles, 3)) * fluid.extent
    grad_at_p = fluid.interpolate(grad, pos)
    analytic_at_p = np.zeros((3, n_particles))
    analytic_at_p[2] = amplitude * k * np.cos(k * pos[:, 2])
    rel = np.abs(grad_at_p - analytic_at_p) / (amplitude * k)
    interp_err = float(np.max(rel))

    vol = closures.sphere_volume(1.0)
    diameter = 1.0
    part = ParticleSet.create(pos, np.zeros((n_particles, 3)),
                              radius=diameter / 2.0, density=1.0)
    model = PointParticleModel(rho_f=1.0, mu=0.1,
                               gravity=np.array([0.0, 0.0, 0.0]))
    from .particles import evaluate_forces
    forces = evaluate_forces(part, model, np.zeros((3, n_particles)),
                             du_dt_fluid=None, grad_p=grad_at_p)
    fpg = forces["pressure_gradient"].T  # (3, N)
    fpg_analytic = -vol * analytic_at_p
    force_err = float(np.max(np.abs(fpg - fpg_analytic)) /
                      np.max(np.abs(fpg_analytic)))

    rec = gate(
        "pressure_gradient_force",
        "grid: |grad p - analytic| / (A k) <= 1e-10 ; "
        "interpolated: residual / |F_pg| <= 2 %",
        interp_err, 0.02,
        f"operator_error={op_err:.3e}, interpolation_error={interp_err:.3e}, "
        f"force_error={force_err:.3e} (k h = {k * h:.3f})",
    )
    rec["operator_passed"] = bool(op_err <= 1e-10)
    rec["passed"] = bool(rec["passed"] and rec["operator_passed"])
    return {
        "gate": rec,
        "operator_error": op_err,
        "interpolation_error": interp_err,
        "force_error": force_err,
        "kh": k * h,
        "amplitude": amplitude,
    }


# ---------------------------------------------------------------------------
# all gates
# ---------------------------------------------------------------------------
def run_all(quick=False):
    """Run every verification face and return a JSON-serialisable summary."""
    if quick:
        st = stokes_terminal_velocity_case(cells=12, dt=0.01, t_end=1.5,
                                           window=0.4)
        am = added_mass_oscillation_case(cells=6, h=1.0, dt=0.02, periods=2.5,
                                         mu=0.0)
        me = momentum_exchange_case(n_particles=10, cells=10, dt=0.004,
                                    n_steps=40)
        rz = richardson_zaki_case(lattice=(4, 4, 3), cells=8, dt=0.01,
                                  t_end=1.0, window=0.4)
        pg = pressure_gradient_case(cells=12)
    else:
        st = stokes_terminal_velocity_case()
        am = added_mass_oscillation_case()
        me = momentum_exchange_case()
        rz = richardson_zaki_case()
        pg = pressure_gradient_case()

    cases = {"stokes_terminal": st, "added_mass": am,
             "momentum_exchange": me, "richardson_zaki": rz,
             "pressure_gradient": pg}
    gates = [c["gate"] for c in cases.values()]
    summary = {
        "cases": {k: {kk: vv for kk, vv in v.items() if kk != "gate"}
                  for k, v in cases.items()},
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
        "n_passed": sum(1 for g in gates if g["passed"]),
        "n_gates": len(gates),
    }
    return summary