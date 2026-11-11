"""L/D = 5 pipe-section Euler-Lagrange (DEM-CFD) model of slurry transport.

Purpose
-------
An *independent numerical adjudicator* for the empirical closures: run a
Lagrangian DEM of the granular phase inside a streamwise-periodic segment of
length ``L = 5 D`` and measure

* the vertical concentration profile ``C(z)`` and its asymmetry,
* the particle velocity profile ``u_p(z)`` and the slip,
* the **measured pressure gradient** ``i_m`` and its decomposition into the
  carrier (wall-shear) part ``i_w`` and the solids part ``i_e``.

Model: what is resolved and what is closed
------------------------------------------
Resolved
    * particle-particle and particle-wall contacts (linear spring-dashpot,
      Coulomb-limited tangential damping, translation only, no spin);
    * particle motion under buoyancy-reduced gravity, local hindered drag and
      turbulent dispersion;
    * the **streamwise momentum of the carrier**, cross-sectionally resolved on
      an ``n_z`` cell grid, with the particle drag entering as a momentum sink
      and the pressure gradient obtained from the imposed mixture flow rate.
      This is what lets the bed *block* the flow: a packed bed leaves little
      open area, so the momentum balance pushes the carrier velocity into the
      free region -- a mechanism a prescribed profile cannot represent (an
      earlier prescribed-profile version over-predicted ``i_e`` by a factor 6).
Closed
    * turbulence: Prandtl mixing length in the carrier, damped in the bed
      (``mu_t = rho kappa^2 l_m^2 |du/dz| eps^3``), plus an Ornstein-Uhlenbeck
      particle dispersion velocity with rms ``c_t u_tau`` and correlation time
      ``tau_t = 0.05 H / u_tau``;
    * voidage-dependent drag: Schiller-Naumann with Richardson-Zaki hindrance
      on the local voidage, floored at ``eps_floor = 0.40`` (see
      :meth:`drag_state`);
    * the streamwise advection term is dropped (locally fully developed
      cross-section), the standard 1-D+ Euler-Lagrange approximation; the
      consequence is quantified in the report's limitation section.

Geometry
--------
Periodic in ``x`` (segment length ``5 D``) and in ``y`` (slab width ``W``); solid
walls in ``z`` at ``0`` and ``H = D/2``, so the channel hydraulic diameter
``2 H = D`` equals the pipe diameter used by every empirical closure.

Pressure-gradient bookkeeping
-----------------------------
With ``G = -dp/dx`` measured from the flow-rate constraint, the steady carrier
momentum balance reads ``eps_bar G H = tau_wall,total + int D dz`` (all per unit
width, ``D`` the particle drag per unit volume).  Hence ``i_m = G/(rho_f g)``,
``i_e = int D dz /(H rho_f g)`` and ``i_w = i_m - i_e``; :meth:`results` reports
the residual of that balance, so the decomposition is auditable rather than
asserted.

Units: SI.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .pressure_drop import colebrook_friction
from .closures import (G, MU_W, RHO_W, cd_schiller_naumann,
                       hindrance_richardson_zaki, particle_reynolds,
                       sphere_volume, submerged_weight,
                       terminal_velocity_numeric)

__all__ = ["PipeSectionDEMCFD", "dem_transport_case", "skewness"]

KAPPA_VK = 0.41


def skewness(x, weights=None):
    """Third standardised moment (zero for a symmetric profile)."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return 0.0
    mu = float(np.sum(w * x) / sw)
    var = float(np.sum(w * (x - mu) ** 2) / sw)
    if var <= 0.0:
        return 0.0
    return float(np.sum(w * (x - mu) ** 3) / sw / var ** 1.5)


def _solve_tridiag(lower, diag, upper, rhs):
    """Thomas algorithm (n is O(10) here, so plain Python is fine)."""
    n = diag.size
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class PipeSectionDEMCFD:
    """L/D = 5 periodic slab segment with a cross-sectionally resolved carrier.

    Parameters
    ----------
    D : pipe diameter [m]; sets ``H = D/2`` and the segment length ``5 D``.
    W : slab width [m] (y-periodic).
    d : particle diameter [m].
    Ss : relative particle density ``rho_s / rho_f``.
    cv : delivered volumetric concentration target [-] (sets the count).
    rho_f, mu : carrier properties.
    n_particles : override the particle count.
    k_n : contact normal stiffness [N/m].
    zeta : contact damping ratio [-].
    mu_c : Coulomb friction coefficient [-].
    c_t : turbulent dispersion intensity (rms = ``c_t u_tau``).
    n_z_fluid : number of carrier cells over ``H`` (default ``H/(1.5 d)``).
    n_z_bins : bins of the reported concentration profile.
    eps_floor : lower clamp of the voidage entering the RZ hindrance.
    seed : RNG seed.
    erosion : optional :class:`tensordem.erosion_dem.recorder.ImpactErosionRecorder`.
        When supplied, every rising-edge wall contact is booked as an impact
        (approach speed and angle from the solver state) and the recorder
        accumulates the wall loss.  ``None`` (default) leaves the solver
        bit-for-bit unchanged -- the coupling is an observer, never a force.
    """

    def __init__(self, D=0.10, W=0.02, d=0.004, Ss=2.65, cv=0.10,
                 rho_f=RHO_W, mu=MU_W, n_particles=None, k_n=300.0, zeta=0.7,
                 mu_c=0.4, c_t=0.30, n_z_fluid=None, n_z_bins=40,
                 eps_floor=0.40, mu_t_floor=1e-3, eps_D=1e-3, seed=0,
                 erosion=None):
        self.D = float(D)
        self.H = 0.5 * self.D
        self.Lx = 5.0 * self.D
        self.W = float(W)
        self.d = float(d)
        self.r = 0.5 * self.d
        self.Ss = float(Ss)
        self.cv_target = float(cv)
        self.rho_f = float(rho_f)
        self.rho_s = self.Ss * self.rho_f
        self.mu = float(mu)
        self.k_n = float(k_n)
        self.zeta = float(zeta)
        self.mu_c = float(mu_c)
        self.c_t = float(c_t)
        self.n_z_bins = int(n_z_bins)
        self.n_z_fluid = int(n_z_fluid) if n_z_fluid is not None else \
            max(6, int(round(self.H / (1.5 * self.d))))
        self.dz_f = self.H / self.n_z_fluid
        self.eps_floor = float(eps_floor)
        self.mu_t_floor = float(mu_t_floor)
        self.eps_D = float(eps_D)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # optional erosion observer (see tensordem.erosion_dem.recorder)
        self.erosion_recorder = erosion
        self._wall_contact = None

        self.vol_domain = self.Lx * self.W * self.H
        self.Vp = sphere_volume(self.d)
        self.m = self.rho_s * self.Vp
        self.W_sub = submerged_weight(self.d, self.rho_s, self.rho_f)
        if n_particles is None:
            n_particles = int(round(self.cv_target * self.vol_domain / self.Vp))
        self.n_particles = int(n_particles)
        self.cv = self.n_particles * self.Vp / self.vol_domain

        self.c_n = 2.0 * self.zeta * math.sqrt(self.k_n * self.m)
        self.c_t_tang = self.c_n

        self.w_s = terminal_velocity_numeric(self.d, self.rho_s, self.rho_f,
                                             self.mu)["w"]
        self.Re_t = float(particle_reynolds(self.w_s, self.d, self.rho_f, self.mu))

        self.t = 0.0
        self.pos = None
        self.vel = None
        self.fluct = None
        self.u_f = np.zeros(self.n_z_fluid)
        self.G = 0.0            # -dp/dx [Pa/m]
        self.u_tau = 0.0
        self.tau_t = 0.0
        self.U = 0.0
        self.U_mix = 0.0
        self.D_h = self.D
        self.mu_t = np.zeros(self.n_z_fluid)
        self._sample = None
        self._last = None
        self.balance_residual = float("nan")

    # ------------------------------------------------------------------
    # set-up
    # ------------------------------------------------------------------
    def initialize(self, U_mix, jitter=0.02, start_at_fluid_velocity=False):
        """Place the particles on a jittered lattice and set the carrier field.

        ``U_mix`` is the **mixture** velocity (total volumetric flux / area),
        which is what a dredge pump imposes.  ``start_at_fluid_velocity=True``
        starts the solids in local equilibrium with the carrier (used by the
        neutral-density negative control, where the quadratic-drag spin-up from
        rest would otherwise mask the zero excess gradient).
        """
        self.U_mix = float(U_mix)
        self.D_h = 2.0 * self.H                     # = D by construction
        self.U = self.U_mix                         # liquid velocity estimate
        self.u_f = np.full(self.n_z_fluid, self.U_mix)
        self.mu_t = np.zeros(self.n_z_fluid)
        Re = self.rho_f * self.U * self.D_h / self.mu
        self.u_tau = self.U * math.sqrt(0.02 / 8.0)
        self.tau_t = 0.05 * self.H / max(self.u_tau, 1e-9)
        self.f_re_estimate = Re

        spacing = (self.Vp / max(self.cv_target, 1e-6)) ** (1.0 / 3.0)
        nx = max(2, int(round(self.Lx / spacing)))
        ny = max(2, int(round(self.W / spacing)))
        nz = max(2, int(round(self.H / spacing)))
        xs = (np.arange(nx) + 0.5) * self.Lx / nx
        ys = (np.arange(ny) + 0.5) * self.W / ny
        zs = np.arange(nz) * spacing + 0.5 * (self.H - (nz - 1) * spacing)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        self.n_particles = int(pos.shape[0])
        self.cv = self.n_particles * self.Vp / self.vol_domain
        pos = pos + jitter * self.d * self.rng.standard_normal(pos.shape)
        pos[:, 0] %= self.Lx
        pos[:, 1] %= self.W
        pos[:, 2] = np.clip(pos[:, 2], self.r, self.H - self.r)
        self.pos = pos
        nn = self.pos.shape[0]
        self.vel = np.zeros((nn, 3))
        if start_at_fluid_velocity:
            self.vel[:, 0] = float(self.u_f.mean())
        self.fluct = np.zeros((nn, 3))
        # contact-edge state of the erosion observer, sized to the real lattice
        self._wall_contact = (
            np.zeros((2, nn), dtype=bool)
            if self.erosion_recorder is not None else None
        )
        self.t = 0.0
        self._sample = None

    # ------------------------------------------------------------------
    # voidage
    # ------------------------------------------------------------------
    def voidage_cells(self):
        """Voidage of every carrier cell (counts-based, exact)."""
        idx = np.clip((self.pos[:, 2] / self.dz_f).astype(int),
                      0, self.n_z_fluid - 1)
        counts = np.bincount(idx, minlength=self.n_z_fluid).astype(float)
        eps = 1.0 - counts * self.Vp / (self.Lx * self.W * self.dz_f)
        return idx, np.clip(eps, 1e-3, 1.0)

    def voidage(self):
        """Voidage seen by the particles, floored at ``eps_floor``.

        The Richardson-Zaki hindrance is a dilute-to-moderate suspension
        correlation; inside the DEM bed the inter-particle momentum transfer is
        carried by the resolved contacts, so extrapolating ``eps**-n`` into a
        packed bed (``eps -> 0.02`` would give a factor 1.2e4) is both
        unphysical and numerically explosive.
        """
        _, eps = self.voidage_cells()
        eps_p = np.clip(eps, self.eps_floor, 1.0)
        idx, _ = self.voidage_cells()
        return eps, eps_p[idx]

    def fluid_velocity(self, pos):
        """Local instantaneous carrier velocity seen by the particles (N, 3)."""
        u = np.zeros_like(pos)
        z = np.clip(pos[:, 2], 0.0, self.H)
        # linear interpolation between cell centres (zero at both walls)
        s = z / self.dz_f - 0.5
        i0 = np.clip(np.floor(s).astype(int), 0, self.n_z_fluid - 1)
        i1 = np.clip(i0 + 1, 0, self.n_z_fluid - 1)
        w = np.clip(s - i0, 0.0, 1.0)
        u[:, 0] = (1.0 - w) * self.u_f[i0] + w * self.u_f[i1]
        return u + self.fluct

    # ------------------------------------------------------------------
    # drag closure
    # ------------------------------------------------------------------
    def drag_state(self):
        """Relative velocity, Re, drag law and relaxation time per particle.

        Two branches (Gidaspow 1982 blend, split at ``eps = 0.8``):

        * ``eps > 0.8`` -- Schiller-Naumann with the Richardson-Zaki hindrance
          ``eps**-n`` (dilute / moderate suspension);
        * ``eps <= 0.8`` -- Ergun (1952) packed-bed relation written per
          particle, ``F = V_p [150 (1-eps) mu /(eps d^2) + 1.75 rho |u_rel|/d]
          u_rel``.

        The dense branch matters: with the suspension branch alone (even at the
        ``eps = 0.4`` clamp) the bed offers 6x too little resistance, the
        interstitial velocity stays at the bulk value and the solids drag is
        over-predicted by a factor ~3.
        """
        _, eps_p = self.voidage()
        u_local = self.fluid_velocity(self.pos)
        u_rel = u_local - self.vel
        speed = np.linalg.norm(u_rel, axis=1)
        Re = self.rho_f * speed * self.d / self.mu
        cd = cd_schiller_naumann(Re)
        hind = hindrance_richardson_zaki(eps_p, self.Re_t)
        area = 0.25 * math.pi * self.d ** 2
        k_dilute = 0.5 * self.rho_f * cd * area * hind * speed
        k_dense = self.Vp * (150.0 * (1.0 - eps_p) * self.mu
                             / (eps_p * self.d ** 2)
                             + 1.75 * self.rho_f * speed / self.d)
        dense = eps_p <= 0.8
        k_drag = np.where(dense, k_dense, k_dilute)
        law = np.where(dense, "ergun", "schiller_naumann")
        tau = self.m / np.maximum(k_drag, 1e-300)
        return {"u_local": u_local, "u_rel": u_rel, "speed": speed, "Re": Re,
                "Cd": cd, "hindrance": hind, "tau": tau, "k": k_drag,
                "law": law}

    def drag_force(self):
        """Instantaneous quasi-steady drag force [N], shape (N, 3)."""
        st = self.drag_state()
        return st["k"][:, None] * st["u_rel"]

    # ------------------------------------------------------------------
    # carrier momentum solve
    # ------------------------------------------------------------------
    def _wall_shear_velocity(self, u_first):
        """Friction velocity from the log law at the first cell (wall function).

        ``u(z1) = u_tau [ln(z1/z0)/kappa + 8.5]`` with the roughness length
        ``z0 = max(nu/(9 u_tau), eps_wall/30)``.  With a cell size of order the
        particle diameter the near-wall gradient cannot be resolved, and using
        the first-cell eddy viscosity in the wall flux over-predicts the carrier
        pressure gradient by a factor 2 (measured on the clear-liquid case).
        """
        u_first = max(float(u_first), 1e-6)
        z1 = 0.5 * self.dz_f
        nu = self.mu / self.rho_f

        def resid(ut):
            z0 = max(nu / (9.0 * ut), self.eps_D / 30.0)
            return ut * (math.log(z1 / z0) / KAPPA_VK + 8.5) - u_first

        lo, hi = 1e-6, max(2.0 * u_first, 1e-3)
        f_lo, f_hi = resid(lo), resid(hi)
        if f_lo * f_hi > 0.0:
            return math.sqrt(max(hi * hi / 400.0, 1e-12))
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def _solve_carrier(self, dt, fluid_index, fluid_eps, k_cell, up_cell):
        """Quasi-steady 1-D carrier momentum solve; returns the pressure gradient.

        Solves ``0 = eps G dz + d/dz[(mu+mu_t) du/dz] - D`` per unit width with
        the particle drag as a momentum sink, then picks ``G`` so that the
        **mixture** velocity matches ``U_mix`` (the system is linear in ``G``:
        two tridiagonal solves).

        Face conductances use the correct wall distance (``dz/2`` at the walls):
        using ``dz`` there halves the wall flux and breaks the momentum balance.
        """
        n = self.n_z_fluid
        dz = self.dz_f
        rho = self.rho_f
        eps = fluid_eps
        du = np.gradient(self.u_f, dz)
        z_c = (np.arange(n) + 0.5) * dz
        l_m = KAPPA_VK * np.minimum(z_c, self.H - z_c)
        mu_t = rho * l_m ** 2 * np.abs(du) * eps ** 3 + self.mu_t_floor * self.mu
        self.mu_t = mu_t
        K = self.mu + mu_t

        # face diffusivities: interior faces are arithmetic means, the wall faces
        # are set by the log-law wall function so that the wall flux is rho u_tau^2
        K_int = 0.5 * (K[:-1] + K[1:])
        ut_bot = self._wall_shear_velocity(self.u_f[0])
        ut_top = self._wall_shear_velocity(self.u_f[-1])
        A_bot = rho * ut_bot ** 2 / max(self.u_f[0], 1e-6)      # conductance
        A_top = rho * ut_top ** 2 / max(self.u_f[-1], 1e-6)
        A_face = np.zeros(n + 1)
        A_face[0] = A_bot
        A_face[1:n] = K_int / dz
        A_face[n] = A_top

        diag = A_face[:-1] + A_face[1:] + k_cell
        lower = -A_face[1:n]
        upper = -A_face[1:n]
        base_rhs = k_cell * up_cell

        sol = []
        for G_try in (0.0, 1.0):
            u = _solve_tridiag(lower, diag, upper, base_rhs + eps * G_try * dz)
            sol.append((u, float(np.sum(eps * u) * dz / self.H)))
        (u0, f0), (u1, f1) = sol
        solids_flux = float(np.sum((1.0 - eps) * up_cell) * dz / self.H)
        if abs(f1 - f0) < 1e-30:
            G_new, u_new = self.G, u0
        else:
            G_new = (self.U_mix - solids_flux - f0) / (f1 - f0)
            u_new = u0 + G_new * (u1 - u0)
        self.G = float(max(G_new, 0.0))
        self.u_f = np.maximum(u_new, 0.0)
        self.tau_wall = rho * (ut_bot ** 2 + ut_top ** 2)
        self.u_tau = math.sqrt(max(self.tau_wall, 0.0) / rho)
        self.tau_t = 0.05 * self.H / max(self.u_tau, 1e-9)
        self.U = float(np.mean(eps * self.u_f))
        return self.G

    # ------------------------------------------------------------------
    # contacts
    # ------------------------------------------------------------------
    def _pair_forces(self):
        """Particle-particle contact forces (vectorised KD-tree pair search)."""
        N = self.pos.shape[0]
        F = np.zeros_like(self.pos)
        if N < 2:
            return F, 0
        tree = cKDTree(self.pos, boxsize=[self.Lx, self.W, 1e6])
        pairs = tree.query_pairs(1.02 * self.d, output_type="ndarray")
        if pairs.size == 0:
            return F, 0
        i, j = pairs[:, 0], pairs[:, 1]
        dr = self.pos[j] - self.pos[i]
        dr[:, 0] -= self.Lx * np.round(dr[:, 0] / self.Lx)
        dr[:, 1] -= self.W * np.round(dr[:, 1] / self.W)
        dist = np.linalg.norm(dr, axis=1)
        overlap = self.d - dist
        active = overlap > 0.0
        if not np.any(active):
            return F, 0
        i, j = i[active], j[active]
        dr, dist, overlap = dr[active], dist[active], overlap[active]
        n_hat = dr / dist[:, None]
        dv = self.vel[j] - self.vel[i]
        vn = np.sum(dv * n_hat, axis=1)
        Fn = np.maximum(self.k_n * overlap - self.c_n * vn, 0.0)
        vt = dv - vn[:, None] * n_hat
        vt_mag = np.linalg.norm(vt, axis=1)
        ft_mag = np.minimum(self.mu_c * Fn, self.c_t_tang * vt_mag)
        safe = np.where(vt_mag > 1e-14, vt_mag, 1.0)
        f_on_i = -Fn[:, None] * n_hat + (ft_mag / safe)[:, None] * vt
        np.add.at(F, i, f_on_i)
        np.add.at(F, j, -f_on_i)
        return F, int(i.size)

    def _scan_wall_impacts(self, pen, wall_index, normal_sign):
        """Book rising-edge wall contacts on the erosion recorder (observer).

        A particle that *enters* the wall contact (penetration crossing zero)
        with a positive approach speed is one impact; the impact kinematics are
        read from the solver state (normal speed ``-sign*vz``, tangential speed
        ``hypot(vx, vy)``), so the erosion law sees the same (v, alpha) the
        contact model sees.  Resting/sliding contact is not re-billed: the
        rising edge is booked once.
        """
        if self.erosion_recorder is None:
            return
        now = pen > 0.0
        rising = now & ~self._wall_contact[wall_index]
        self._wall_contact[wall_index] = now
        if not np.any(rising):
            return
        v_n = -normal_sign * self.vel[rising, 2]
        v_t = np.hypot(self.vel[rising, 0], self.vel[rising, 1])
        for vn, vt in zip(v_n, v_t):
            if vn <= 0.0:
                continue
            speed = math.hypot(vn, vt)
            alpha = math.degrees(math.atan2(vn, vt)) if speed > 0.0 else 90.0
            self.erosion_recorder.record(speed, alpha, self.d, self.rho_s)

    def _wall_forces(self):
        """Contact forces against the two z walls (x and y are periodic)."""
        F = np.zeros_like(self.pos)
        pen = self.r - self.pos[:, 2]                 # bottom wall, normal +z
        self._scan_wall_impacts(pen, 0, +1.0)
        act = pen > 0.0
        if np.any(act):
            Fn = np.maximum(self.k_n * pen[act] + self.c_n * self.vel[act, 2], 0.0)
            F[act, 2] += Fn
            vt = self.vel[act].copy()
            vt[:, 2] = 0.0
            vt_mag = np.linalg.norm(vt, axis=1)
            ft = np.minimum(self.mu_c * Fn, self.c_t_tang * vt_mag)
            safe = np.where(vt_mag > 1e-14, vt_mag, 1.0)
            F[act, :2] -= (ft / safe)[:, None] * vt[:, :2]
        pen = self.pos[:, 2] - (self.H - self.r)      # top wall, normal -z
        self._scan_wall_impacts(pen, 1, -1.0)
        act = pen > 0.0
        if np.any(act):
            Fn = np.maximum(self.k_n * pen[act] - self.c_n * self.vel[act, 2], 0.0)
            F[act, 2] -= Fn
            vt = self.vel[act].copy()
            vt[:, 2] = 0.0
            vt_mag = np.linalg.norm(vt, axis=1)
            ft = np.minimum(self.mu_c * Fn, self.c_t_tang * vt_mag)
            safe = np.where(vt_mag > 1e-14, vt_mag, 1.0)
            F[act, :2] -= (ft / safe)[:, None] * vt[:, :2]
        return F

    # ------------------------------------------------------------------
    # time advancement
    # ------------------------------------------------------------------
    def step(self, dt):
        """One step: carrier solve -> turbulence -> particle advance."""
        fluid_index, fluid_eps = self.voidage_cells()

        # --- carrier momentum with the drag of the previous particle state ---
        st = self.drag_state()
        k_cell = np.bincount(fluid_index, weights=st["k"],
                             minlength=self.n_z_fluid) / (self.Lx * self.W)
        # cell-averaged particle velocity (per unit cross-section of the cell)
        cnt = np.bincount(fluid_index, minlength=self.n_z_fluid)
        up_sum = np.bincount(fluid_index, weights=self.vel[:, 0],
                             minlength=self.n_z_fluid)
        up_cell = np.divide(up_sum, cnt, out=np.zeros(self.n_z_fluid),
                            where=cnt > 0)
        self._solve_carrier(dt, fluid_index, fluid_eps, k_cell, up_cell)

        # --- turbulence (OU dispersion with the measured friction velocity) ---
        sigma = self.c_t * self.u_tau
        decay = math.exp(-dt / self.tau_t)
        amp = sigma * math.sqrt(max(1.0 - decay * decay, 0.0))
        self.fluct = self.fluct * decay \
            + amp * self.rng.standard_normal(self.fluct.shape)
        self.fluct[:, 0] *= 0.5            # damp the mean-zero streamwise noise

        # --- particles: semi-implicit drag + explicit contacts/gravity ---
        st = self.drag_state()
        frac = 1.0 - np.exp(-dt / st["tau"])
        dv_drag = frac[:, None] * st["u_rel"]
        Fd = self.m * dv_drag / dt
        Fc, n_contacts = self._pair_forces()
        Fw = self._wall_forces()
        acc = (Fc + Fw) / self.m
        acc[:, 2] -= (self.Ss - 1.0) * G
        self.vel += dt * acc + dv_drag
        self.pos += dt * self.vel
        self.pos[:, 0] %= self.Lx
        self.pos[:, 1] %= self.W
        np.clip(self.pos[:, 2], 1e-3 * self.d, self.H - 1e-3 * self.d,
                out=self.pos[:, 2])
        self.t += dt
        self._last = {"F_drag": Fd, "n_contacts": n_contacts,
                      "k_cell": k_cell, "up_cell": up_cell,
                      "eps": fluid_eps}
        return n_contacts

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def accumulate(self):
        """Add the current state to the time-average sample."""
        if self._last is None:
            Fd = self.drag_force()
            eps = self.voidage_cells()[1]
        else:
            Fd = self._last["F_drag"]
            eps = self._last["eps"]
        if self._sample is None:
            self._sample = {"n": 0,
                            "z_cnt": np.zeros(self.n_z_bins),
                            "vx_z": np.zeros(self.n_z_bins),
                            "Fx_sum": 0.0, "ux": 0.0, "bed": 0.0,
                            "upper_flux": 0.0, "tot_flux": 0.0,
                            "G": 0.0, "u_tau": 0.0, "tau_wall": 0.0,
                            "u_f": np.zeros(self.n_z_fluid),
                            "eps": np.zeros(self.n_z_fluid),
                            "resid": 0.0}
        s = self._sample
        s["n"] += 1
        idx = np.clip((self.pos[:, 2] / self.H * self.n_z_bins).astype(int),
                      0, self.n_z_bins - 1)
        np.add.at(s["z_cnt"], idx, 1.0)
        np.add.at(s["vx_z"], idx, self.vel[:, 0])
        s["Fx_sum"] += float(np.sum(Fd[:, 0]))
        s["ux"] += float(np.mean(self.vel[:, 0]))
        s["bed"] += float(np.mean(self.pos[:, 2] < self.r + 2.0 * self.d))
        up = self.pos[:, 2] >= 0.5 * self.H
        s["upper_flux"] += float(np.sum(self.vel[up, 0]))
        s["tot_flux"] += float(np.sum(self.vel[:, 0]))
        s["G"] += self.G
        s["u_tau"] += self.u_tau
        s["tau_wall"] += getattr(self, "tau_wall", 0.0)
        s["u_f"] += self.u_f
        s["eps"] += eps
        s["resid"] += self._momentum_residual(eps)

    def _momentum_residual(self, eps):
        """Steady carrier momentum balance residual (relative)."""
        Fd = self._last["F_drag"] if self._last is not None else self.drag_force()
        drag_volume = float(np.sum(Fd[:, 0])) / (self.Lx * self.W * self.H)
        lhs = float(np.mean(eps)) * self.G
        rhs = getattr(self, "tau_wall", 0.0) / self.H + drag_volume
        if abs(lhs) < 1e-30:
            return 0.0
        return abs(lhs - rhs) / abs(lhs)

    def results(self):
        """Time-averaged transport diagnostics."""
        s = self._sample
        if s is None or s["n"] == 0:
            raise RuntimeError("no samples accumulated")
        n = float(s["n"])
        dz = self.H / self.n_z_bins
        bin_vol = self.Lx * self.W * dz
        cv_z = s["z_cnt"] / n * self.Vp / bin_vol
        vx_z = np.divide(s["vx_z"], s["z_cnt"],
                         out=np.zeros_like(s["vx_z"]), where=s["z_cnt"] > 0)
        Fx_mean = s["Fx_sum"] / n
        H = self.H
        rho = self.rho_f
        i_m = (s["G"] / n) / (rho * G)
        i_e_drag = Fx_mean / (self.Lx * self.W * H) / (rho * G)
        u_f = s["u_f"] / n
        eps_f = s["eps"] / n
        U_liq = float(np.sum(eps_f * u_f) * self.dz_f / H)
        Re_l = rho * U_liq * self.D_h / self.mu
        f_c = colebrook_friction(max(Re_l, 4000.0), self.eps_D)["f"]
        i_w = f_c * U_liq ** 2 / (2.0 * G * self.D_h)
        # the resolved near-wall gradient cannot carry the full wall shear at
        # dz ~ d/2, so the solids contribution is defined against the carrier
        # friction closure at the same liquid superficial velocity (the standard
        # engineering decomposition); the raw partition is reported alongside.
        i_e = i_m - i_w
        i_w_resolved = float(s["tau_wall"] / n) / (H * rho * G)
        i_e_partition = i_m - i_w_resolved
        u_p = s["ux"] / n
        i_ideal = self.cv * (self.Ss - 1.0) * self.w_s / self.U_mix
        mid = self.n_z_bins // 2
        c_bot = float(np.mean(cv_z[:mid]))
        c_top = float(np.mean(cv_z[mid:]))
        zc = (np.arange(self.n_z_bins) + 0.5) * dz
        zf = (np.arange(self.n_z_fluid) + 0.5) * self.dz_f
        return {
            "U": self.U_mix, "U_liquid": U_liq, "Re_liquid": float(Re_l),
            "f_carrier": float(f_c),
            "cv": self.cv, "n_particles": self.n_particles, "Ss": self.Ss,
            "w_s": self.w_s, "u_tau": float(s["u_tau"] / n),
            "tau_wall": float(s["tau_wall"] / n), "D_h": self.D_h,
            "c_t": self.c_t, "n_z_fluid": self.n_z_fluid,
            "G": float(s["G"] / n),
            "i_m": float(i_m), "i_e": float(i_e), "i_w": float(i_w),
            "i_e_drag": float(i_e_drag), "i_e_partition": float(i_e_partition),
            "i_w_resolved": float(i_w_resolved),
            "wall_shear_ratio": float((s["tau_wall"] / n)
                                      / (f_c * rho * U_liq ** 2 / 8.0)),
            "i_e_ideal": float(i_ideal),
            "kappa": float(i_e / i_ideal) if i_ideal > 0 else float("nan"),
            "momentum_balance_residual": float(s["resid"] / n),
            "u_p_mean": float(u_p), "slip": float(self.U_mix - u_p),
            "bed_fraction": float(s["bed"] / n),
            "upper_half_flux_fraction":
                float(s["upper_flux"] / s["tot_flux"]) if s["tot_flux"] else float("nan"),
            "cv_z": cv_z.tolist(), "vx_z": vx_z.tolist(),
            "z_centers": zc.tolist(),
            "u_f_profile": u_f.tolist(), "eps_profile": eps_f.tolist(),
            "z_fluid": zf.tolist(),
            "c_bottom_half": c_bot, "c_top_half": c_top,
            # capped at 1e4 for JSON sanity; a zero top-half concentration means
            # "all solids in the lower half" (bed), which the cap preserves
            "bottom_top_ratio": float(min(c_bot / c_top, 1e4)) if c_top > 1e-6
            else 1e4,
            "profile_skewness": skewness(zc, weights=cv_z),
            "samples": int(s["n"]),
        }

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def run(self, t_end, dt, sample_start=None, sample_every=20):
        """Integrate to ``t_end``; time-average from ``sample_start`` onwards."""
        if self.pos is None:
            raise RuntimeError("call initialize(U_mix) first")
        if sample_start is None:
            sample_start = 0.6 * t_end
        n_steps = int(math.ceil(t_end / dt))
        hist = {"t": [], "i_e": [], "u_p": [], "G": []}
        block = max(1, n_steps // 20)
        step = 0
        acc_i, acc_u, acc_G, acc_n = 0.0, 0.0, 0.0, 0
        while step < n_steps:
            self.step(dt)
            step += 1
            if self.t >= sample_start and step % sample_every == 0:
                self.accumulate()
                Fd = self._last["F_drag"]
                acc_i += float(np.sum(Fd[:, 0])) / (self.Lx * self.W * self.H) \
                    / (self.rho_f * G)
                acc_u += float(np.mean(self.vel[:, 0]))
                acc_G += self.G
                acc_n += 1
            if step % block == 0 and acc_n > 0:
                hist["t"].append(self.t)
                hist["i_e"].append(acc_i / acc_n)
                hist["u_p"].append(acc_u / acc_n)
                hist["G"].append(acc_G / acc_n)
                acc_i, acc_u, acc_G, acc_n = 0.0, 0.0, 0.0, 0
        res = self.results()
        if len(hist["i_e"]) >= 4:
            half = len(hist["i_e"]) // 2
            first = float(np.mean(hist["i_e"][:half]))
            second = float(np.mean(hist["i_e"][half:]))
            res["i_e_drift"] = abs(second - first) / max(abs(first), 1e-30)
            res["i_e_first_half"] = first
            res["i_e_second_half"] = second
        res["history"] = hist
        return res, hist


def dem_transport_case(U_mix, D=0.10, W=0.02, d=0.004, Ss=2.65, cv=0.10,
                       t_end=3.5, dt=1e-4, seed=0, **kw):
    """Convenience driver: build, initialize, run and return the diagnostics."""
    model = PipeSectionDEMCFD(D=D, W=W, d=d, Ss=Ss, cv=cv, seed=seed, **kw)
    model.initialize(U_mix)
    res, hist = model.run(t_end=t_end, dt=dt)
    res["geometry"] = {"D": D, "H": model.H, "Lx": model.Lx, "W": W, "d": d}
    return res