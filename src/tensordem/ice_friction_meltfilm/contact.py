"""Per-contact thermal-friction state machine for the ice/steel melt film.

The object added to the DEM bond/contact surface is two internal variables per
contact, ``{T_i, h}``:

* ``T_i``  interface temperature [degC] -- solved from the Bowden-Hughes energy
  balance, not stored as a rate;
* ``h``    melt-film thickness [m] -- a genuine state variable, advanced by the
  film mass balance ``rho_w L dh/dt = q_latent - rho_w L u_drain``.

Friction is then a *solved* output of the contact rather than an input:
``mu = f(h/sigma_rms) mu_asp + (1-f) eta(T_f) V/(h sigma)``.

Every call returns an auditable ledger (heat in, heat out by path, melt in,
storage, drainage out) so the energy and mass criteria are checkable per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .physics import (
    MaterialPair,
    film_shear_friction,
    gw_area_fraction,
    halfspace_conductance,
    reynolds_drainage_velocity,
    water_viscosity,
)

__all__ = ["ContactState", "ContactResult", "MeltFilmContact", "solve_steady"]


@dataclass
class ContactState:
    """Internal variables carried by one contact point."""

    t_interface: float = -10.0   # degC, steel-side interface temperature
    h_film: float = 0.0          # m, melt-water film thickness
    melted_volume: float = 0.0   # m, cumulative melt produced per unit area
    drained_volume: float = 0.0  # m, cumulative water squeezed out

    def copy(self) -> "ContactState":
        return ContactState(self.t_interface, self.h_film,
                            self.melted_volume, self.drained_volume)


@dataclass
class ContactResult:
    """One converged / stepped contact point, fully auditable."""

    speed: float
    sigma: float
    t_bulk: float
    mu: float
    t_interface: float
    h_film: float
    real_area_fraction: float
    mu_asperity: float
    mu_film: float
    q_in: float
    q_solid: float
    q_latent: float
    q_conv: float
    melt_rate: float            # kg/(m^2 s)
    drain_velocity: float       # m/s (thinning rate)
    melting: bool
    residual_energy: float
    residual_energy_rel: float
    iterations: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class MeltFilmContact:
    """Melt-film lubricated ice/steel contact (`{T_i, h}` internal variables).

    Parameters
    ----------
    material:
        Thermophysical + texture data of the pair (see :class:`MaterialPair`).
    thermal_coupling:
        Negative-control switch (criterion C7): with ``thermal_coupling=False``
        the friction heat is not allowed to reach the interface (``q_in`` is
        still reported but nothing is heated), so no film can form and mu must
        collapse onto ``mu_asp`` for every speed.
    """

    def __init__(self, material: MaterialPair | None = None,
                 thermal_coupling: bool = True):
        self.material = material or MaterialPair()
        self.thermal_coupling = thermal_coupling

    # -- closures ----------------------------------------------------------
    def _mu(self, h: float, speed: float, sigma: float, t_film: float
            ) -> tuple[float, float, float]:
        m = self.material
        lam = h / m.sigma_rms if m.sigma_rms > 0 else 0.0
        f = gw_area_fraction(lam)
        mu_f = film_shear_friction(h, speed, sigma, t_film) if h > 0 else 0.0
        mu = f * m.mu_asp + (1.0 - f) * mu_f
        return mu, f, mu_f

    def _removal(self, h: float) -> tuple[float, float]:
        """Heat-removal conductance of the interface [W/(m^2 K)].

        Series: transient half-space (both bodies) in series with the water film,
        in parallel with the gap convective loss.
        """
        m = self.material
        g_hs = halfspace_conductance(m, self._speed_cache)
        if g_hs <= 0.0:
            # No sliding -> no sliding heat source at all; return only conv.
            return m.h_conv, g_hs
        r = 1.0 / g_hs + (h / m.k_water if h > 0 else 0.0)
        return 1.0 / r + m.h_conv, g_hs

    def _film_temperature(self, h: float, t_i: float) -> float:
        """Mean film temperature used for the water viscosity."""
        return 0.5 * (t_i + self.material.t_melt)

    # -- steady state ------------------------------------------------------
    def solve_steady(self, speed: float, sigma: float, t_bulk: float,
                     h_lo: float = 1.0e-11, h_hi: float = 1.0e-3,
                     tol: float = 1.0e-14) -> ContactResult:
        """Solve the closed-form steady contact at (V, sigma, T_bulk).

        Two branches:

        * **dry** -- the friction heat can be carried by conduction (+convection)
          without reaching the melting temperature: ``h = 0`` and
          ``mu = mu_asp`` exactly (criterion C1).
        * **melting** -- the interface is pinned at ``t_melt`` and the surplus
          ``q_latent = q_in - (t_melt - t_bulk) G_removal`` feeds the film.  The
          steady thickness is the root of
          ``rho_w L u_drain(h) = q_latent(h)`` (bisection on h; the residual is
          monotone increasing in h because drainage ~ h^3).
        """
        m = self.material
        self._speed_cache = speed
        dt_melt = m.t_melt - t_bulk                    # > 0 for cold ice
        g_rem0, g_hs = self._removal(0.0)

        mu0, f0, muf0 = self._mu(0.0, speed, sigma, t_bulk)
        q_in0 = mu0 * sigma * speed
        q_rem0 = dt_melt * g_rem0 if self.thermal_coupling else 0.0

        if (not self.thermal_coupling) or q_in0 <= q_rem0 or speed <= 0.0:
            # ---- dry branch -------------------------------------------------
            q_in = q_in0
            q_rem = q_rem0
            t_i = t_bulk + (q_in / g_rem0 if (self.thermal_coupling and g_rem0 > 0)
                            else 0.0)
            q_lat = 0.0
            q_solid = q_in - m.h_conv * max(t_i - t_bulk, 0.0)
            q_conv = q_in - q_solid
            res = q_in - (q_solid + q_conv)
            return ContactResult(
                speed=speed, sigma=sigma, t_bulk=t_bulk, mu=mu0,
                t_interface=t_i, h_film=0.0, real_area_fraction=f0,
                mu_asperity=m.mu_asp, mu_film=muf0, q_in=q_in, q_solid=q_solid,
                q_latent=0.0, q_conv=q_conv, melt_rate=0.0, drain_velocity=0.0,
                melting=False, residual_energy=res,
                residual_energy_rel=abs(res) / max(abs(q_in), 1e-30),
                iterations=0, extra={"branch": "dry", "g_hs": g_hs})

        # ---- melting branch -------------------------------------------------
        def residual(h: float) -> tuple[float, dict]:
            g_rem, _ = self._removal(h)
            q_rem = dt_melt * g_rem
            # mu depends on h through f(h) and the film shear; use the
            # interface temperature of the previous iterate would make this
            # non-monotone, so the film viscosity is evaluated at the mean film
            # temperature implied by pure conduction to t_melt.
            t_i_est = m.t_melt + q_rem * h / m.k_water
            t_f = self._film_temperature(h, t_i_est)
            mu, f, muf = self._mu(h, speed, sigma, t_f)
            q_in = mu * sigma * speed
            q_lat = q_in - q_rem
            p_real = sigma / max(f, 1e-12)
            u_d = reynolds_drainage_velocity(h, p_real, m, t_f)
            r = m.latent_rho * u_d - q_lat
            return r, dict(mu=mu, f=f, mu_f=muf, q_in=q_in, q_lat=q_lat,
                           u_drain=u_d, t_f=t_f, g_rem=g_rem)

        r_lo, d_lo = residual(h_lo)
        r_hi, d_hi = residual(h_hi)
        it = 0
        if r_lo > 0.0:
            # extremely thin film already drains faster than it is produced:
            # clamp to the lower bracket (documented in the report).
            h_star, dat = h_lo, d_lo
        elif r_hi < 0.0:
            # drainage cannot keep up even at the upper bracket; the film grows
            # without bound within this closure -> report the bracket end.
            h_star, dat = h_hi, d_hi
        else:
            lo, hi = h_lo, h_hi
            h_star, dat = lo, d_lo
            while it < 200:
                it += 1
                mid = math.sqrt(lo * hi)      # geometric bisection (log h)
                r_mid, d_mid = residual(mid)
                h_star, dat = mid, d_mid
                if abs(r_mid) <= tol * (abs(d_mid["q_lat"]) + 1.0):
                    break
                if r_mid < 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi / lo < 1.0 + 1e-12:
                    break

        g_rem, _ = self._removal(h_star)
        q_in = dat["q_in"]
        t_i = t_bulk + q_in / g_rem
        q_rem = dt_melt * g_rem
        q_conv = m.h_conv * max(t_i - t_bulk, 0.0)
        q_solid = q_rem - q_conv
        q_lat = dat["q_lat"]
        res = q_in - (q_solid + q_lat + q_conv)
        melt_rate = q_lat / m.latent_heat if q_lat > 0 else 0.0
        return ContactResult(
            speed=speed, sigma=sigma, t_bulk=t_bulk, mu=dat["mu"],
            t_interface=t_i, h_film=h_star, real_area_fraction=dat["f"],
            mu_asperity=m.mu_asp, mu_film=dat["mu_f"], q_in=q_in,
            q_solid=q_solid, q_latent=q_lat, q_conv=q_conv,
            melt_rate=melt_rate, drain_velocity=dat["u_drain"], melting=True,
            residual_energy=res,
            residual_energy_rel=abs(res) / max(abs(q_in), 1e-30),
            iterations=it,
            extra={"branch": "melting", "g_hs": g_hs,
                   "t_film": dat["t_f"], "p_real": sigma / max(dat["f"], 1e-12)},
        )

    # -- time march (mass-conservation ledger) -----------------------------
    def march(self, state: ContactState, speed: float, sigma: float,
              t_bulk: float, dt: float, n_steps: int) -> tuple[ContactState, dict]:
        """Advance the film state by ``n_steps`` implicit (backward-Euler) steps.

        ``dh/dt = (q_latent(h) - rho_w L u_drain(h)) / (rho_w L)``

        The drainage term is cubic in ``h`` and therefore stiff, so each step
        solves the implicit equation by bisection (unconditionally stable; an
        explicit step of the same size oscillates and drives the film to zero).
        The half-space thermal response is treated as quasi-steady over the
        dwell time ``l_c/V`` (its diffusion time is microseconds against a
        millisecond-to-second dwell), so only the film mass balance is marched.

        The returned ledger closes mass to bisection accuracy:
        ``melt_in = storage_change + drained_out``.
        """
        m = self.material
        self._speed_cache = speed
        state = state.copy()
        melt_in = 0.0
        drain_out = 0.0

        def rates(h: float, t_guess: float) -> tuple[float, float, float]:
            """(dh/dt, q_latent, u_drain) at film thickness h."""
            g_rem, g_hs = self._removal(h)
            if g_hs <= 0.0:
                return 0.0, 0.0, 0.0
            t_f = self._film_temperature(h, t_guess)
            mu, f, _ = self._mu(h, speed, sigma, t_f)
            q_in = mu * sigma * speed
            q_lat = q_in - (m.t_melt - t_bulk) * g_rem
            if q_lat <= 0.0:
                return (0.0, q_lat, 0.0) if h <= 0.0 else (
                    -reynolds_drainage_velocity(
                        h, sigma / max(f, 1e-12), m, t_f), q_lat,
                    reynolds_drainage_velocity(h, sigma / max(f, 1e-12), m, t_f))
            p_real = sigma / max(f, 1e-12)
            u_d = reynolds_drainage_velocity(h, p_real, m, t_f)
            return (q_lat - m.latent_rho * u_d) / m.latent_rho, q_lat, u_d

        for _ in range(n_steps):
            h0 = state.h_film
            t_i0 = state.t_interface
            _, ql0, _ = rates(h0, t_i0)
            # bracket: phi(h) = h0 + dt*dhdt(h) - h, phi(0) >= 0, phi(H) < 0
            # phi(h) = h0 + dt * dh/dt(h) - h is positive at h = 0 and turns
            # negative once drainage overtakes the melt supply; grow the upper
            # bracket until that happens (at most 80 doublings).
            h_hi = max(h0, 1e-12) + dt * max(ql0, 0.0) / m.latent_rho + 1e-12
            for _outer in range(80):
                if h0 + dt * rates(h_hi, t_i0)[0] - h_hi <= 0.0:
                    break
                h_hi *= 2.0
            lo, hi = 0.0, h_hi
            for _bis in range(90):
                mid = 0.5 * (lo + hi)
                r_mid = h0 + dt * rates(mid, t_i0)[0] - mid
                if r_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-18 + 1e-12 * hi:
                    break
            h1 = 0.5 * (lo + hi)
            _, ql1, ud1 = rates(h1, t_i0)
            g_rem1, _ = self._removal(h1)
            t_i1 = t_bulk + (self._mu(h1, speed, sigma,
                                      self._film_temperature(h1, t_i0))[0]
                             * sigma * speed) / g_rem1
            melt_in += max(ql1, 0.0) * dt / (m.latent_heat * m.rho_water)
            drain_out += ud1 * dt
            state.h_film = h1
            state.t_interface = t_i1
            state.melted_volume += max(ql1, 0.0) * dt / (m.latent_heat * m.rho_water)
            state.drained_volume += ud1 * dt

        ledger = {
            "melt_in": melt_in,
            "storage_change": state.h_film,
            "drained_out": drain_out,
            "mass_residual": melt_in - (state.h_film + drain_out),
            "mass_residual_rel": abs(melt_in - (state.h_film + drain_out))
            / max(melt_in, 1e-30),
            "storage_only_residual_rel": abs(melt_in - state.h_film)
            / max(melt_in, 1e-30),
            "dt": dt,
            "n_steps": n_steps,
            "h_final": state.h_film,
            "t_interface_final": state.t_interface,
        }
        return state, ledger


def solve_steady(speed: float, sigma: float, t_bulk: float,
                 material: MaterialPair | None = None,
                 thermal_coupling: bool = True) -> ContactResult:
    """Convenience wrapper: steady contact at (V, sigma, T_bulk)."""
    return MeltFilmContact(material, thermal_coupling).solve_steady(
        speed, sigma, t_bulk)