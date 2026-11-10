"""2D granular layer as a particle impact damper: transmissibility ``T(f)``.

Physical setting
----------------
A 2D (in-plane) packing of ``nrows`` layers resting on a driven base platen is
capped by a rigid lid of mass ``M_lid``.  The base is driven kinematically,
``z_b(t) = A sin(2 pi f t)``; the lid acceleration divided by the base
acceleration is the **transmissibility** ``T(f)`` of the granular layer plus lid,
i.e. the quantity a particle damper is specified by.

Damping is the DEM dashpot ``c_k = 2 zeta_c sqrt(k_t m_eff)`` with ``k_t`` the
tangent Hertz stiffness at a 2% of ``R`` reference overlap (constant per
contact, so the ledger closes exactly).  Two *independent* damping estimates are
produced from the same sweep:

``zeta_energy``  energy ledger per steady cycle,
                 ``zeta = dE_cycle / (4 pi E_peak)``, ``E_peak = k_eff X^2 / 2``
                 with ``X`` the lid-to-base relative displacement amplitude;
``zeta_curve``   least-squares fit of the base-excitation SDOF transmissibility
                 ``|T| = sqrt((1+(2 zeta r)^2)/((1-r^2)^2+(2 zeta r)^2))`` to the
                 whole measured ``T(f)`` curve (``f_n`` and ``zeta`` free).

The campaign cross-checks those two against each other (``<= 15%``).

Known limitations (documented, not hidden): the packing is frictionless and
2D (no out-of-plane rattling, no rotation), so the damper's absolute damping is
a model quantity rather than a prediction for a specific real damper; the
*band structure* of ``T(f)`` and the *consistency* between the two damping
routes are what the campaign scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

BASE = -1
LID = -2
WALL_L = -3
WALL_R = -4


@dataclass
class LayerConfig:
    ncols: int = 5
    nrows: int = 5
    radius: float = 3.5e-3
    density: float = 1200.0
    young_modulus: float = 5.0e7
    poisson_ratio: float = 0.30
    lid_mass: float = 2.0e-2
    gravity: float = 9.81
    drive_amplitude: float = 5.0e-5
    damping_ratio: float = 0.15        # dashpot ratio at the 2%-of-R tangent
    sigma_radius: float = 0.0
    dt_factor: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        if self.ncols < 2 or self.nrows < 2:
            raise ValueError("need at least a 2x2 layer")
        if min(self.radius, self.density, self.young_modulus) <= 0.0:
            raise ValueError("radius, density, young_modulus must be positive")
        if self.lid_mass <= 0.0:
            raise ValueError("lid_mass must be positive")
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be in [0, 1)")

    @property
    def e_star(self) -> float:
        return self.young_modulus / (2.0 * (1.0 - self.poisson_ratio ** 2))

    @property
    def e_star_wall(self) -> float:
        return self.young_modulus / (1.0 - self.poisson_ratio ** 2)

    @property
    def thickness(self) -> float:
        """Layer height from the base plate to the lid seat [m]."""
        return 2.0 * self.radius + (self.nrows - 1) * math.sqrt(3.0) * self.radius

    @property
    def width(self) -> float:
        return 2.0 * (self.ncols + 1) * self.radius

    @property
    def filling_ratio(self) -> float:
        area = self.ncols * self.nrows * math.pi * self.radius ** 2
        return area / (self.width * self.thickness)


def _sine_amplitude(t: np.ndarray, y: np.ndarray, omega: float) -> float:
    """Least-squares amplitude of the ``sin(omega t)`` / ``cos`` / offset fit."""
    basis = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return float(math.hypot(coef[0], coef[1]))


def sdof_transmissibility(f: Sequence[float], fn: float, zeta: float) -> np.ndarray:
    """Base-excitation transmissibility of a damped SDOF system."""
    r = np.asarray(f, dtype=float) / fn
    num = 1.0 + (2.0 * zeta * r) ** 2
    den = (1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2
    return np.sqrt(num / den)


def fit_sdof_to_curve(f: Sequence[float], T: Sequence[float]) -> dict[str, float]:
    """Least-squares ``(f_n, zeta)`` of the SDOF transmissibility to data."""
    f = np.asarray(f, dtype=float)
    y = np.log(np.maximum(np.asarray(T, dtype=float), 1e-6))
    f0 = float(f[np.argmax(y)])
    z0 = 0.1

    def resid(p: np.ndarray) -> np.ndarray:
        fn, zeta = float(np.exp(p[0])), float(np.exp(p[1]))
        return np.log(np.maximum(sdof_transmissibility(f, fn, zeta), 1e-9)) - y

    sol = least_squares(resid, x0=np.array([math.log(f0), math.log(z0)]),
                        bounds=([math.log(f[0] / 5.0), math.log(1e-3)],
                                [math.log(f[-1] * 5.0), math.log(0.95)]))
    fn, zeta = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    pred = sdof_transmissibility(f, fn, zeta)
    ss_res = float(np.sum((np.log(np.maximum(pred, 1e-9)) - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"fn_Hz": fn, "zeta": zeta,
            "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


class GranularLayer:
    """Frictionless 2D Hertzian packing between a driven base, side walls and a lid."""

    def __init__(self, cfg: LayerConfig | None = None) -> None:
        self.cfg = cfg or LayerConfig()
        c = self.cfg
        rng = np.random.default_rng(int(c.seed))
        R = c.radius
        pos: list[tuple[float, float]] = []
        rad: list[float] = []
        for j in range(c.nrows):
            for i in range(c.ncols):
                x = (i - 0.5 * (c.ncols - 1)) * 2.0 * R + (R if j % 2 else 0.0)
                z = R + j * math.sqrt(3.0) * R
                rr = R * (1.0 + c.sigma_radius * (2.0 * rng.random() - 1.0))
                pos.append((x, z))
                rad.append(rr)
        self.pos = np.asarray(pos, dtype=float) + \
            (rng.random((len(pos), 2)) - 0.5) * (0.02 * R)
        self.radius = np.asarray(rad, dtype=float)
        self.n = len(self.pos)
        self.mass = c.density * math.pi * self.radius ** 2          # unit depth
        self.vel = np.zeros_like(self.pos)

        # particle-particle candidate pairs (initial distance < 2.4 R)
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.hypot(*(self.pos[j] - self.pos[i])))
                if d < 2.4 * R:
                    pairs.append((i, j))
        self.pairs = np.asarray(pairs, dtype=int)
        k_pp = (4.0 / 3.0) * c.e_star * np.sqrt(0.5 * R)
        self.k_pp = k_pp
        self.k_wall = (4.0 / 3.0) * c.e_star_wall * math.sqrt(R)
        delta_ref = 0.02 * R
        k_t = 1.5 * k_pp * math.sqrt(delta_ref)
        m_eff = 0.5 * float(self.mass.mean())
        self.c_damp = 2.0 * c.damping_ratio * math.sqrt(k_t * m_eff)
        k_tw = 1.5 * self.k_wall * math.sqrt(delta_ref)
        self.c_damp_wall = 2.0 * c.damping_ratio * math.sqrt(k_tw * float(self.mass.mean()))

        self.z_base = 0.0
        self.v_base = 0.0
        self.z_lid = float(self.pos[:, 1].max() + R)
        self.v_lid = 0.0
        self.lid_mass = c.lid_mass
        self.dt = self._recommended_dt()
        self.dissipated = 0.0
        self._gravity = c.gravity

    # ------------------------------------------------------------------
    def _recommended_dt(self) -> float:
        """Conservative step from the stiffest *effective pair* in the layer.

        The lid is lighter than a grain, so its contact with a grain is the
        stiffest oscillator even though ``k_wall`` is only ~3x ``k_pp``; using
        ``m_min / k_max`` alone (particle mass only) over-predicts ``dt`` by
        more than an order of magnitude and the run diverges -- the reference
        overlap is 5% of R so the estimate also covers the settling transient.
        """
        c = self.cfg
        delta_ref = 0.05 * c.radius
        k_t_max = 1.5 * max(self.k_pp, self.k_wall) * math.sqrt(delta_ref)
        m_pp = 0.5 * float(self.mass.min())
        m_lid = self.lid_mass * float(self.mass.min()) / (self.lid_mass + float(self.mass.min()))
        m_min = min(m_pp, m_lid, float(self.mass.min()))
        return c.dt_factor * 2.0 * math.pi * math.sqrt(m_min / k_t_max)

    @property
    def layer_height(self) -> float:
        return float(self.pos[:, 1].max() + self.cfg.radius)

    # ------------------------------------------------------------------
    def _contact_forces(self, pos: np.ndarray, vel: np.ndarray,
                        z_base: float, v_base: float, z_lid: float, v_lid: float
                        ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Return (particle accel, lid accel, dashpot work rate, overlaps)."""
        c = self.cfg
        acc = np.zeros_like(pos)
        acc[:, 1] = -self._gravity
        work = 0.0
        overlaps = []

        # particle-particle
        if self.pairs.size:
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            dvec = pos[j] - pos[i]
            d = np.hypot(dvec[:, 0], dvec[:, 1])
            d = np.maximum(d, 1e-12)
            delta = self.radius[i] + self.radius[j] - d
            act = delta > 0.0
            if act.any():
                ii, jj = i[act], j[act]
                nvec = dvec[act] / d[act][:, None]
                dv = vel[jj] - vel[ii]
                vn = np.sum(dv * nvec, axis=1)
                delta_a = delta[act]
                # Kelvin-Voigt: the dashpot resists the *separation* rate, so its
                # contribution enters with a minus sign (a plus sign makes it a
                # negative damper that pumps energy -- verified numerically).
                f = self.k_pp * delta_a ** 1.5 - self.c_damp * vn
                f = np.maximum(f, 0.0)
                # repulsion: bead i is pushed along -n, bead j along +n
                # bincount instead of np.add.at: same result, ~20x faster
                fx = -f * nvec[:, 0]
                fz = -f * nvec[:, 1]
                acc[:, 0] += (np.bincount(ii, weights=fx, minlength=self.n)
                              / self.mass)
                acc[:, 1] += (np.bincount(ii, weights=fz, minlength=self.n)
                              / self.mass)
                acc[:, 0] -= (np.bincount(jj, weights=fx, minlength=self.n)
                              / self.mass)
                acc[:, 1] -= (np.bincount(jj, weights=fz, minlength=self.n)
                              / self.mass)
                work += float(np.sum((-self.c_damp * vn) * vn)) * -1.0
                overlaps.append(delta_a)

        # base platen (driven)
        delta_b = self.radius - (pos[:, 1] - z_base)
        act = delta_b > 0.0
        if act.any():
            vn = vel[act, 1] - v_base
            f = self.k_wall * delta_b[act] ** 1.5 - self.c_damp_wall * vn
            f = np.maximum(f, 0.0)
            acc[act, 1] += f / self.mass[act]
            work += float(np.sum((-self.c_damp_wall * vn) * vn)) * -1.0
            overlaps.append(delta_b[act])
            # base reaction (upward on particles, so work on base is -f*v_base)

        # side walls
        half = 0.5 * c.width
        for sign, xw in ((1.0, half), (-1.0, -half)):
            delta_w = self.radius - (half - sign * pos[:, 0])
            act = delta_w > 0.0
            if act.any():
                # outward normal of the wall is -sign*x: approaching => vn < 0
                vn = -sign * vel[act, 0]
                f = self.k_wall * delta_w[act] ** 1.5 - self.c_damp_wall * vn
                f = np.maximum(f, 0.0)
                acc[act, 0] += -sign * f / self.mass[act]
                work += float(np.sum((-self.c_damp_wall * vn) * vn)) * -1.0
                overlaps.append(delta_w[act])

        # lid
        lid_force = -self.lid_mass * self._gravity
        delta_l = self.radius - (z_lid - pos[:, 1])
        act = delta_l > 0.0
        if act.any():
            vn = v_lid - vel[act, 1]
            f = self.k_wall * delta_l[act] ** 1.5 - self.c_damp_wall * vn
            f = np.maximum(f, 0.0)
            # the lid sits *above* the grains: it pushes them down (-z) and is
            # itself pushed up (+z) by the sum of the contact forces
            acc[act, 1] -= f / self.mass[act]
            lid_force += float(np.sum(f))
            work += float(np.sum((-self.c_damp_wall * vn) * vn)) * -1.0
            overlaps.append(delta_l[act])

        ov = np.concatenate(overlaps) if overlaps else np.zeros(0)
        return acc, lid_force / self.lid_mass, work, ov

    # ------------------------------------------------------------------
    def energy(self, pos: np.ndarray | None = None, vel: np.ndarray | None = None
               ) -> dict[str, float]:
        pos = self.pos if pos is None else pos
        vel = self.vel if vel is None else vel
        pe_g = float(np.sum(self.mass * self._gravity * pos[:, 1])
                     + self.lid_mass * self._gravity * self.z_lid)
        ke = float(np.sum(0.5 * self.mass * np.sum(vel ** 2, axis=1))
                   + 0.5 * self.lid_mass * self.v_lid ** 2)
        pe_e = 0.0
        if self.pairs.size:
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            d = np.maximum(np.hypot(*(pos[j] - pos[i]).T), 1e-12)
            delta = np.maximum(self.radius[i] + self.radius[j] - d, 0.0)
            pe_e += float(np.sum((2.0 / 5.0) * self.k_pp * delta ** 2.5))
        delta_b = np.maximum(self.radius - (pos[:, 1] - self.z_base), 0.0)
        pe_e += float(np.sum((2.0 / 5.0) * self.k_wall * delta_b ** 2.5))
        half = 0.5 * self.cfg.width
        for sign in (1.0, -1.0):
            dw = np.maximum(self.radius - (half - sign * pos[:, 0]), 0.0)
            pe_e += float(np.sum((2.0 / 5.0) * self.k_wall * dw ** 2.5))
        dl = np.maximum(self.radius - (self.z_lid - pos[:, 1]), 0.0)
        pe_e += float(np.sum((2.0 / 5.0) * self.k_wall * dl ** 2.5))
        return {"kinetic": ke, "gravity": pe_g, "elastic": pe_e,
                "total": ke + pe_g + pe_e, "dissipated": self.dissipated}

    # ------------------------------------------------------------------
    def step(self, dt: float, z_base_fn=None) -> None:
        c = self.cfg
        if z_base_fn is not None:
            self.z_base, self.v_base = z_base_fn
        acc, acc_lid, work, _ = self._contact_forces(
            self.pos, self.vel, self.z_base, self.v_base, self.z_lid, self.v_lid)
        v_half = self.vel + 0.5 * dt * acc
        v_lid_half = self.v_lid + 0.5 * dt * acc_lid
        self.pos = self.pos + dt * v_half
        self.z_lid += dt * v_lid_half
        acc, acc_lid, work, _ = self._contact_forces(
            self.pos, v_half, self.z_base, self.v_base, self.z_lid, v_lid_half)
        self.vel = v_half + 0.5 * dt * acc
        self.v_lid = v_lid_half + 0.5 * dt * acc_lid
        self.dissipated += work * dt       # work = sum(c vn^2) >= 0 = dissipated rate

    # ------------------------------------------------------------------
    def settle(self, duration: float) -> dict[str, float]:
        """Let the packing come to rest under gravity and the lid weight."""
        n = int(duration / self.dt)
        e0 = self.energy()
        d0 = self.dissipated          # the ledger is cumulative across calls
        dyn = max(abs(e0["kinetic"]) + abs(e0["elastic"]), 1e-12)
        for k in range(n):
            self.step(self.dt)
            if k % 10 == 0:
                e_now = self.energy()
                dyn = max(dyn, abs(e_now["kinetic"]) + abs(e_now["elastic"]))
        e1 = self.energy()
        imbalance = e1["total"] + (e1["dissipated"] - d0) - e0["total"]
        return {"duration_s": duration, "steps": n,
                "e_initial_J": e0["total"], "e_final_J": e1["total"],
                "dissipated_J": e1["dissipated"] - d0,
                "imbalance_J": imbalance,
                "residual": imbalance / dyn,   # normalised by the dynamic energy scale
                "dynamic_scale_J": dyn}

    def static_stiffness(self, extra_load_ratio: float = 0.5,
                         relax_time: float = 0.004) -> dict[str, float]:
        """``k_eff = dF/dz`` of the layer under the lid (finite difference)."""
        z0 = self.z_lid
        f0 = self.lid_mass * self._gravity
        saved = (self.pos.copy(), self.vel.copy(), self.z_lid, self.v_lid)
        # extra load applied as an equivalent lid weight through a pseudo-gravity
        g_save = self._gravity
        try:
            self._gravity = g_save * (1.0 + extra_load_ratio)
            self.vel[:] = 0.0
            self.v_lid = 0.0
            for _ in range(int(relax_time / self.dt)):
                self.step(self.dt)
            z1 = self.z_lid
            f1 = self.lid_mass * self._gravity
        finally:
            self._gravity = g_save
        k_eff = (f1 - f0) / (z0 - z1) if z0 != z1 else float("inf")
        self.pos, self.vel, self.z_lid, self.v_lid = saved
        return {"k_eff_N_m": float(k_eff), "dz_m": float(z0 - z1),
                "fn_Hz": float(math.sqrt(k_eff / self.lid_mass) / (2.0 * math.pi))
                if k_eff > 0 else float("nan")}

    # ------------------------------------------------------------------
    def drive(self, frequency: float, *, cycles_measure: int = 12,
              cycles_warmup: int = 6, amplitude: float | None = None) -> dict[str, Any]:
        """Steady-state base excitation at ``frequency``; return one sweep point."""
        c = self.cfg
        A = c.drive_amplitude if amplitude is None else float(amplitude)
        omega = 2.0 * math.pi * frequency
        dt = self.dt
        z_b0 = self.z_base
        n_warm = int(round(cycles_warmup / frequency / dt))
        n_meas = int(round(cycles_measure / frequency / dt))

        def step_base(n: int) -> None:
            for k in range(n):
                t = self._t
                self._t = t + dt
                self.step(dt, z_base_fn=(z_b0 + A * math.sin(omega * t),
                                         A * omega * math.cos(omega * t)))

        self._t = 0.0
        step_base(n_warm)
        t = np.empty(n_meas)
        z_lid = np.empty(n_meas)
        v_lid = np.empty(n_meas)
        e_diss = np.empty(n_meas)
        ke_hist = np.empty(n_meas)
        ov_max = np.empty(n_meas)
        for k in range(n_meas):
            tt = self._t
            self._t = tt + dt
            self.step(dt, z_base_fn=(z_b0 + A * math.sin(omega * tt),
                                     A * omega * math.cos(omega * tt)))
            t[k] = self._t
            z_lid[k] = self.z_lid
            v_lid[k] = self.v_lid
            e_diss[k] = self.dissipated
            ke_hist[k] = (float(np.sum(0.5 * self.mass * np.sum(self.vel ** 2, axis=1)))
                          + 0.5 * self.lid_mass * self.v_lid ** 2)
            if k % 10 == 0:
                acc, _, _, ov = self._contact_forces(
                    self.pos, self.vel, self.z_base, self.v_base, self.z_lid, self.v_lid)
                ov_max[k] = float(ov.max()) if ov.size else 0.0
            else:
                ov_max[k] = ov_max[k - 1] if k else 0.0

        acc_lid = np.gradient(v_lid, dt)
        a_lid = _sine_amplitude(t, acc_lid, omega)
        a_lid_via_v = omega * _sine_amplitude(t, v_lid, omega)
        a_base = A * omega ** 2
        T = a_lid / a_base if a_base > 0 else float("nan")
        T_v = a_lid_via_v / a_base if a_base > 0 else float("nan")
        x_rel = _sine_amplitude(t, z_lid - (z_b0 + A * np.sin(omega * t)), omega)
        dE = float(e_diss[-1] - e_diss[0]) / cycles_measure
        return {
            "frequency_Hz": float(frequency),
            "transmissibility": float(T),
            "transmissibility_dB": float(20.0 * math.log10(max(T, 1e-6))),
            "transmissibility_from_velocity": float(T_v),
            "phase_deg": float("nan"),
            "lid_amplitude_m": float(_sine_amplitude(t, z_lid, omega)),
            "relative_amplitude_m": float(x_rel),
            "energy_dissipated_per_cycle_J": dE,
            "kinetic_energy_max_J": float(np.max(ke_hist)) if ke_hist.size else float("nan"),
            "kinetic_energy_mean_J": float(np.mean(ke_hist)) if ke_hist.size else float("nan"),
            "max_overlap_m": float(ov_max.max()),
            "drive_amplitude_m": A,
        }

    def state(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "n_particles": int(self.n),
            "ncols": int(c.ncols), "nrows": int(c.nrows),
            "thickness_m": c.thickness,
            "width_m": c.width,
            "filling_ratio": c.filling_ratio,
            "radius_m": c.radius,
            "lid_mass_kg": self.lid_mass,
            "gravity_m_s2": c.gravity,
            "dt_s": self.dt,
            "damping_ratio_dashpot": c.damping_ratio,
            "k_pp_N_m15": self.k_pp,
            "k_wall_N_m15": self.k_wall,
            "drive_amplitude_m": c.drive_amplitude,
        }


def energy_method_zeta(dissipated_per_cycle: float, k_eff: float, x_amplitude: float) -> float:
    """``zeta = dE_cycle / (4 pi E_peak)`` with ``E_peak = k_eff X^2 / 2``."""
    e_peak = 0.5 * k_eff * x_amplitude ** 2
    if e_peak <= 0.0:
        return float("nan")
    return float(dissipated_per_cycle / (4.0 * math.pi * e_peak))


def zeta_from_peak(transmissibility_peak: float) -> float:
    """SDOF damping from the peak amplification: ``zeta = 1/(2 sqrt(T^2-1))``.

    Exact inversion of ``|T|(r=1) = sqrt(1+(2 zeta)^2)/(2 zeta)``.
    """
    T = float(transmissibility_peak)
    if T <= 1.0:
        return float("nan")
    return float(1.0 / (2.0 * math.sqrt(T * T - 1.0)))


def zeta_from_bandwidth(f: Sequence[float], T: Sequence[float],
                        f_peak: float | None = None) -> dict[str, float]:
    """SDOF damping from the half-power bandwidth: ``zeta = df / (2 f_peak)``.

    Half-power crossings are found by log-log interpolation between sweep
    points, which is the resolution-limited part of the estimate (the sweep has
    ~6 points per octave in the modal band).
    """
    fa = np.asarray(f, dtype=float)
    Ta = np.asarray(T, dtype=float)
    i_pk = int(np.argmax(Ta))
    Tp = float(Ta[i_pk])
    fp = float(fa[i_pk]) if f_peak is None else float(f_peak)
    level = Tp / math.sqrt(2.0)

    def crossing(lo: int, hi: int) -> float:
        y0, y1 = math.log(Ta[lo]), math.log(Ta[hi])
        x0, x1 = math.log(fa[lo]), math.log(fa[hi])
        if y0 == y1:
            return float(fa[lo])
        xh = x0 + (math.log(level) - y0) * (x1 - x0) / (y1 - y0)
        return float(math.exp(xh))

    f_lo = float("nan")
    for i in range(i_pk, 0, -1):
        if Ta[i - 1] <= level <= Ta[i] or Ta[i] <= level <= Ta[i - 1]:
            f_lo = crossing(i - 1, i)
            break
    f_hi = float("nan")
    for i in range(i_pk, Ta.size - 1):
        if Ta[i] >= level >= Ta[i + 1] or Ta[i] <= level <= Ta[i + 1]:
            f_hi = crossing(i, i + 1)
            break
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return {"zeta": float("nan"), "f_lo_Hz": f_lo, "f_hi_Hz": f_hi,
                "bandwidth_Hz": float("nan"), "f_peak_Hz": fp}
    return {"zeta": float((f_hi - f_lo) / (2.0 * fp)),
            "f_lo_Hz": f_lo, "f_hi_Hz": f_hi, "bandwidth_Hz": float(f_hi - f_lo),
            "f_peak_Hz": fp}


def stopband(points: Sequence[dict[str, Any]], threshold_db: float = -20.0) -> dict[str, Any]:
    """Contiguous frequency band where ``T(f) <= threshold_db``."""
    pts = sorted(points, key=lambda d: d["frequency_Hz"])
    f = np.array([p["frequency_Hz"] for p in pts])
    db = np.array([p["transmissibility_dB"] for p in pts])
    ok = db <= threshold_db
    if not ok.any():
        return {"found": False, "threshold_dB": threshold_db,
                "min_dB": float(np.min(db)), "f_lo_Hz": float("nan"),
                "f_hi_Hz": float("nan"), "width_Hz": 0.0}
    idx = np.nonzero(ok)[0]
    return {
        "found": True,
        "threshold_dB": float(threshold_db),
        "f_lo_Hz": float(f[idx[0]]),
        "f_hi_Hz": float(f[idx[-1]]),
        "width_Hz": float(f[idx[-1]] - f[idx[0]]),
        "min_dB": float(np.min(db)),
        "n_points": int(idx.size),
    }