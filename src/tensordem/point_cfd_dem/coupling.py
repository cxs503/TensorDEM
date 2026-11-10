"""Two-way point-particle CFD-DEM coupling driver.

Loop (one time step)
--------------------
1. estimate the local voidage ``eps`` seen by every particle;
2. interpolate the fluid velocity / pressure gradient / fluid acceleration at the
   particle centres (adjoint trilinear operator, with the particle's *own*
   induced disturbance removed -- see ``self_disturbance``);
3. assemble the force budget and integrate the particle momentum equation with
   the added-mass term treated implicitly;
4. hand the reaction ``-(F_drag [+ F_am])`` back to the fluid as a momentum
   source, spread with the transpose of the interpolation operator, and advance
   the fluid;
5. optional Galilean shift into the *container frame* (zero net volumetric
   flux), needed for Richardson-Zaki comparisons;
6. book-keeping: cumulative particle/fluid momentum and the exchange impulses.

Momentum conservation
---------------------
Because the interpolation and spreading operators are exact transposes, the
total force spread on the grid equals the sum of the particle forces to
round-off; because the spectral projection and the viscous integrating factor
act only on ``k != 0`` modes, the volume-averaged fluid momentum changes *only*
by that impulse.  Hence ::

    d/dt [ sum_p m_p u_p + rho_f sum_cells u_cell dV ] = sum_p F_ext

holds to machine precision.  :meth:`PointParticleCFDDEM.momentum_residual`
returns the normalised drift, which is the quantity gated at ``1e-6``.
"""

from __future__ import annotations

import numpy as np

from .particles import ParticleSet, PointParticleModel, evaluate_forces

__all__ = ["PointParticleCFDDEM"]


class PointParticleCFDDEM:
    """Coupled point-particle / spectral-fluid solver.

    Parameters
    ----------
    fluid : :class:`tensordem.point_cfd_dem.fluid.SpectralPeriodicFluid`
    particles : :class:`ParticleSet`
    model : :class:`PointParticleModel`
    zero_net_flux : bool
        If True, after every step the whole system (particles + fluid) is shifted
        by ``c = phi <u_p,z> + eps <u_f,z>`` so that the net volumetric flux in
        the ``z`` direction vanishes (container frame).
    self_disturbance : bool
        Remove the velocity that a particle induces on *itself* through the
        spreading/interpolation of its own reaction force.  Without this, a
        point-particle model double counts its own singular disturbance and the
        single-particle terminal velocity is systematically under-predicted.
    exchange_terms : tuple of str
        Which forces are handed back to the fluid (``"drag"`` and optionally
        ``"added_mass"``).  The pressure-gradient force is *not* included by
        default because it is already represented by the fluid pressure field.
    spread_mode : {"adjoint", "nearest"}
        ``"adjoint"`` is the conservative (transpose) operator.  ``"nearest"``
        exists only to show that the conservation gate discriminates.
    drag_filter_width : float or {"h", "h/4", None}
        Width of the Gaussian low-pass filter applied to the fluid velocity that
        enters the *drag* closure.  The Eulerian velocity at a particle
        contains the flow induced by the particle's own momentum source (and by
        the cell-scale part of its neighbours'); with an unresolved particle
        this is a modelling artefact that must not enter the drag.  The filter
        removes the sub-cell part only and leaves the volume-averaged
        (interstitial) velocity untouched.  ``None`` disables the correction
        and reproduces the uncorrected point-particle behaviour (used as the
        discriminating baseline of the verification gates).
    self_response : {None, "spectral"}
        Optional *steady-state* self-response correction
        (``C_eff = C/(1-C R)``).  Only well conditioned for ``h >= d``; see the
        limitations section of ``docs/point-particle-cfd-dem.md``.
    voidage_mode : {"control_volume", "global"}
        ``control_volume``: local average over a sphere of radius
        ``kernel_factor * <d>/2`` excluding the particle itself.
        ``global``: ``1 - N V_p / V_domain`` (statistically homogeneous cell).
    """

    def __init__(self, fluid, particles, model, zero_net_flux=False,
                 self_disturbance=True, exchange_terms=("drag",),
                 spread_mode="adjoint", voidage_mode="control_volume",
                 kernel_factor=3.0, track_budget=True,
                 self_response=None, self_response_refresh=100,
                 drag_filter_width="h/4", exchange_scale=1.0):
        self.fluid = fluid
        self.particles = particles
        self.model = model
        self.zero_net_flux = bool(zero_net_flux)
        self.self_disturbance = bool(self_disturbance)
        self.exchange_terms = tuple(exchange_terms)
        self.spread_mode = spread_mode
        self.voidage_mode = voidage_mode
        self.kernel_factor = float(kernel_factor)
        self.track_budget = bool(track_budget)
        self.self_response_mode = self_response
        self.self_response_refresh = int(self_response_refresh)
        if drag_filter_width == "h":
            drag_filter_width = fluid.h
        elif drag_filter_width == "h/4":
            drag_filter_width = 0.25 * fluid.h
        self.exchange_scale = float(exchange_scale)
        self.drag_filter_width = drag_filter_width
        self._R = None

        self.u = fluid.zeros()
        self.grad_p = np.zeros((3,) + fluid.shape)
        self.time = 0.0
        self.step_count = 0

        self._u_at_p_prev = None
        self._du_dt_at_p = np.zeros((3, particles.count))
        self._self_disturbance_velocity = np.zeros((particles.count, 3))
        self._last_forces = None
        self._last_acceleration = np.zeros((particles.count, 3))

        self.history = {"time": [], "momentum_particle": [],
                        "momentum_fluid": [], "kinetic_particle": [],
                        "kinetic_fluid": [], "voidage": [],
                        "mean_velocity_particle": [], "mean_velocity_fluid": []}

        self._p_particle0 = particles.momentum()
        self._p_fluid0 = fluid.momentum(self.u)
        self._external_impulse = np.zeros(3)
        self._exchange_impulse = np.zeros(3)

    # ------------------------------------------------------------------ setup
    def set_fluid_velocity(self, u):
        self.u = np.asarray(u, dtype=float).copy()
        self._p_fluid0 = self.fluid.momentum(self.u)
        self._u_at_p_prev = None

    def set_pressure_gradient(self, grad_p):
        self.grad_p = np.asarray(grad_p, dtype=float)

    # ---------------------------------------------------------------- voidage
    def voidage(self):
        """Local voidage seen by the particles (see ``voidage_mode``)."""
        p = self.particles
        if self.voidage_mode == "global":
            total = p.volume.sum()
            domain = self.fluid.cell_volume * int(np.prod(self.fluid.shape))
            return np.full(p.count, max(1.0 - total / domain, 1e-6))
        if self.voidage_mode != "control_volume":
            raise ValueError(f"unknown voidage_mode {self.voidage_mode!r}")
        # local average excluding self, minimum image convention
        rk = self.kernel_factor * float(np.mean(p.radius))
        v_cv = 4.0 / 3.0 * np.pi * rk ** 3
        extent = self.fluid.extent
        d = p.position[:, None, :] - p.position[None, :, :]
        d -= extent * np.round(d / extent)
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        inside = dist <= rk
        occupied = np.sum(inside * p.volume[None, :], axis=1)
        return np.clip(1.0 - occupied / v_cv, 1e-6, 1.0)

    # ------------------------------------------------------------------- step
    def step(self, dt, external_force=None):
        """Advance the coupled system by ``dt``."""
        fluid = self.fluid
        part = self.particles
        model = self.model

        eps = self.voidage()

        if self.drag_filter_width:
            u_drag_field = fluid.filter_field(self.u, self.drag_filter_width)
        else:
            u_drag_field = self.u
        u_at_p = fluid.interpolate(u_drag_field, part.position)   # (3, N)
        if self.self_disturbance:
            u_at_p = u_at_p - self._self_disturbance_velocity.T

        if self.self_response_mode == "spectral":
            if self._R is None or (self.step_count % self.self_response_refresh == 0):
                self._R = fluid.self_response(part.position, dt,
                                              elapsed=max(self.time, dt))
        elif self.self_response_mode in (None, "none", False):
            self._R = None
        else:
            raise ValueError(
                f"unknown self_response mode {self.self_response_mode!r}")

        grad_p_at_p = fluid.interpolate(self.grad_p, part.position)

        if self._u_at_p_prev is not None:
            self._du_dt_at_p = (u_at_p - self._u_at_p_prev) / dt
        else:
            self._du_dt_at_p = np.zeros_like(u_at_p)

        ext = None if external_force is None else np.asarray(external_force, dtype=float)
        forces = evaluate_forces(
            part, model, u_at_p, du_dt_fluid=self._du_dt_at_p,
            grad_p=grad_p_at_p, voidage=eps, external_force=ext,
            self_response=self._R,
        )

        accel = forces["total"] / forces["effective_mass"][:, None]
        v_new = part.velocity + accel * dt
        x_new = part.position + v_new * dt
        x_new = np.mod(x_new, fluid.extent)

        # ---- reaction handed to the fluid -------------------------------
        reaction = np.zeros((part.count, 3))
        for term in self.exchange_terms:
            if term == "drag":
                reaction += -forces["drag"]
            elif term == "added_mass":
                reaction += -model.added_mass * model.rho_f * part.volume[:, None] * (
                    accel - self._du_dt_at_p.T)
            else:
                raise ValueError(f"unknown exchange term {term!r}")
        f_cells = self.exchange_scale * fluid.spread(reaction.T, part.position,
                                                     mode=self.spread_mode)
        force_density = f_cells / fluid.cell_volume

        self.u = fluid.step(self.u, dt, force_density=force_density)

        # self-induced velocity for the next step (analytic, exact adjoint)
        sw = fluid.self_weight_sum(part.position)
        if self.self_disturbance:
            self._self_disturbance_velocity = (
                dt / (fluid.rho * fluid.cell_volume)
            ) * reaction * sw[:, None]
        else:
            self._self_disturbance_velocity = np.zeros_like(reaction)

        part.velocity = v_new
        part.position = x_new
        self._u_at_p_prev = u_at_p
        self.time += dt
        self.step_count += 1

        # ---- container frame (zero net volumetric flux) ------------------
        if self.zero_net_flux:
            phi = float(np.sum(part.volume)) / (
                fluid.cell_volume * int(np.prod(fluid.shape)))
            eps_mean = 1.0 - phi
            c = phi * part.velocity.mean(axis=0) + eps_mean * fluid.mean_velocity(self.u)
            part.velocity = part.velocity - c
            self.u = self.u - c[:, None, None, None]
            self._u_at_p_prev = self._u_at_p_prev - c[:, None]

        if external_force is not None:
            self._external_impulse = self._external_impulse + (
                np.sum(forces["weight"] + forces["external"], axis=0) * dt)

        self._last_forces = forces
        self._last_acceleration = accel
        if self.track_budget:
            self._record(eps)
        return forces

    # -------------------------------------------------------------- measures
    def _record(self, eps):
        h = self.history
        h["time"].append(self.time)
        h["momentum_particle"].append(self.particles.momentum())
        h["momentum_fluid"].append(self.fluid.momentum(self.u))
        h["kinetic_particle"].append(self.particles.kinetic_energy())
        h["kinetic_fluid"].append(self.fluid.kinetic_energy(self.u))
        h["voidage"].append(float(np.mean(eps)))
        h["mean_velocity_particle"].append(self.particles.mean_velocity())
        h["mean_velocity_fluid"].append(self.fluid.mean_velocity(self.u))

    def run(self, t_end, dt, external_force=None, record_step=1):
        """Integrate until ``time >= t_end``; returns the number of steps."""
        n = int(round(t_end / dt))
        for k in range(n):
            ext = external_force(self) if callable(external_force) else external_force
            self.step(dt, external_force=ext)
            if record_step > 1:
                self.history.setdefault("subsample", []).append(k % record_step)
        return n

    # ---------------------------------------------------------------- budgets
    def momentum_residual(self):
        """Normalised drift of the total (particle + fluid) momentum.

        ``residual = |dp_total - impulse_ext| / max(|dp_particle| + |dp_fluid|, eps)``
        """
        p_p = self.particles.momentum()
        p_f = self.fluid.momentum(self.u)
        dp_p = p_p - self._p_particle0
        dp_f = p_f - self._p_fluid0
        num = np.linalg.norm(dp_p + dp_f - self._external_impulse)
        den = max(np.linalg.norm(dp_p), np.linalg.norm(dp_f), 1e-300)
        return float(num / den)

    def momentum_budget(self):
        p_p = self.particles.momentum()
        p_f = self.fluid.momentum(self.u)
        dp_p = p_p - self._p_particle0
        dp_f = p_f - self._p_fluid0
        return {
            "particle_momentum": p_p.tolist(),
            "fluid_momentum": p_f.tolist(),
            "delta_particle": dp_p.tolist(),
            "delta_fluid": dp_f.tolist(),
            "external_impulse": self._external_impulse.tolist(),
            "sum_exchange": (dp_p + dp_f).tolist(),
            "residual": self.momentum_residual(),
        }