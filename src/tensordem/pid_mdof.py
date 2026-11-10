"""Particle impact damper (PID) on MDOF structures — insertion loss, tuning and
detuning envelope (campaign node p307w1-n010).

Motivation
----------
The IN32-D02 delivery (``tensordem.pid``) identified the equivalent viscous
damping ratio ``zeta_eff(filling, frequency)`` that a granular particle impact
damper contributes to a **single-degree-of-freedom (SDOF)** host.  A damping
ratio alone is not a design quantity: a ship structural engineer needs the
**insertion loss** IL(f) — how many dB the resonant response of a real
multi-degree-of-freedom (MDOF) structure drops when the device is installed —
plus the tuning optimum and the robustness of that optimum to frequency
mistuning.

This module builds that layer.  It **consumes** the delivered zeta_eff surface
(no DEM is re-run) and adds:

1. `zeta_eff_delivered` / `ZetaEffProvider` — faithful evaluation of the
   IN32-D02 surface for an arbitrary installation mass ratio mu (the delivered
   model's own mass dependence is used; no invented scaling law).
2. `two_dof_response`, `den_hartog_optimum`, `den_hartog_peak_amplification` —
   the analytic Den Hartog main-system / absorber closed form, which is the
   verification anchor for everything numerical below.
3. `cantilever_structure`, `stiffened_plate_structure` — mass-normalised MDOF
   modal models (Euler-Bernoulli cantilever; orthotropic smeared-stiffener
   Kirchhoff panel).
4. `modal_response`, `insertion_loss`, `peak_insertion_loss`,
   `optimise_tuning` — exact frequency-domain solve of the (N+1) coupled
   system (N structural modes + 1 damper DOF).  For N = 1 it must reproduce
   the Den Hartog closed form to machine precision (tested).
5. `inject_modal_damping` + `identify_zeta_half_power` — the SDOF bridge used
   for the IN32-D02 regression cross-check.
6. `floating_raft_il_db` — two-stage (floating raft) isolation reference for
   the per-unit-mass IL comparison table.

Model conventions (documented because they carry physical meaning)
------------------------------------------------------------------
* The PID is treated as a **damped absorber equivalence**: the granular bed of
  mass ``m_d`` has an internal rattling resonance ``omega_d = nu * omega_1``
  (set by cavity length / particle size), and its dissipative capacity is the
  delivered ``zeta_eff``.  This reading is forced by the delivered surface
  itself: its frequency dependence is a Lorentzian with Q = 5 (see
  ``tensordem.pid._frequency_tuning``), i.e. the bed behaves as a resonant
  element of quality factor Q = 5, not as a frequency-independent dashpot.
* Mass ratio ``mu = m_d / M_att`` where ``M_att = 1 / phi_1(x_d)**2`` is the
  *effective modal mass at the attachment point* of the mass-normalised mode.
  With this convention an N = 1 modal model is exactly Den Hartog's 2-DOF
  system with mass ratio mu.
* Filling ratio and mass are physically coupled: an empty cavity has neither
  mass nor dissipation, so ``m_d`` and ``zeta_eff`` both vanish as
  phi -> 0.  This is what makes the negative control (criterion d) meaningful.
* ``detuning_coupling=True`` (default) evaluates ``zeta_eff`` at the *actual*
  tuning ratio nu, i.e. a mistuned bed also dissipates less (what the surface
  says).  ``False`` freezes zeta_eff at its peak value and isolates the purely
  mechanical impedance effect of mistuning.  Both are reported.

All quantities SI.  Angles/frequencies in rad/s unless a name ends in ``_hz``.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from tensordem.pid import (
    MultiParticlePIDModel,
    identify_damping_ratio_from_model,
)

G = 9.81

#: Mass ratio of the *delivered* IN32-D02 reference configuration
#: (n_particles = 20, particle mass = 0.01 kg, primary mass = 1.0 kg).
MU_REF = 0.20

#: Number of particles held fixed at the delivered calibration point.
N_PARTICLES_REF = 20

#: Filling ratio at which the design mass ratio mu is quoted (delivered optimum).
PHI_REF = 0.30


# ===================================================================
# 1. IN32-D02 zeta_eff surface consumption (no DEM recomputation)
# ===================================================================

def delivered_natural_frequency(primary_mass: float = 1.0,
                                primary_stiffness: float = 1000.0) -> float:
    """Natural frequency (Hz) of the IN32-D02 reference SDOF host."""
    return 0.5 * math.sqrt(primary_stiffness / primary_mass) / math.pi


@dataclass
class ZetaEffProvider:
    """Evaluates the delivered IN32-D02 zeta_eff surface for one design point.

    The delivered calibration used ``n_particles = 20`` particles of 0.01 kg on
    a 1.0 kg host (mass ratio 0.20).  An installation with a different mass
    ratio is represented by *keeping the delivered particle count* and scaling
    the per-particle mass so that ``n * m_p / M = mass_ratio``.  This uses only
    the mass dependence already present in the delivered model (the energy
    method gives zeta_0 proportional to m_p); no new scaling law is invented.
    At ``mass_ratio = MU_REF`` the provider reproduces the delivered surface
    exactly (regression check, criterion (a)).
    """
    filling_ratio: float
    gamma: float = 1.5
    restitution: float = 0.6
    mass_ratio: float = MU_REF
    n_particles: int = N_PARTICLES_REF
    primary_mass: float = 1.0
    primary_stiffness: float = 1000.0
    cavity_length: float = 0.1
    gravity: float = G

    @property
    def f_n(self) -> float:
        return delivered_natural_frequency(self.primary_mass,
                                           self.primary_stiffness)

    def _model(self, freq_ratio: float) -> MultiParticlePIDModel:
        f_n = self.f_n
        f = freq_ratio * f_n
        omega = 2.0 * math.pi * f
        amplitude = self.gamma * self.gravity / omega ** 2 if omega > 0 else 0.0
        particle_mass = self.mass_ratio * self.primary_mass / self.n_particles
        model = MultiParticlePIDModel(
            n_particles=self.n_particles,
            mass=particle_mass,
            primary_mass=self.primary_mass,
            primary_stiffness=self.primary_stiffness,
            cavity_length=self.cavity_length,
            amplitude=amplitude,
            frequency=f,
            gravity=self.gravity,
            e=self.restitution,
        )
        return model

    def zeta(self, freq_ratio: float) -> float:
        """zeta_eff at excitation/tuning ratio ``freq_ratio = f / f_n``."""
        if self.filling_ratio <= 0.0 or self.mass_ratio <= 0.0:
            return 0.0
        model = self._model(freq_ratio)
        return identify_damping_ratio_from_model(
            model, self.filling_ratio, self.gamma, freq_ratio,
            self.restitution).zeta_mean

    def zeta_at_peak(self) -> float:
        """zeta_eff at the delivered optimum tuning (f/f_n = 1)."""
        return self.zeta(1.0)


def zeta_eff_delivered(filling_ratio: float, freq_ratio: float,
                       gamma: float = 1.5, restitution: float = 0.6,
                       mass_ratio: float = MU_REF, **kwargs) -> float:
    """Functional form of the IN32-D02 surface: zeta_eff(phi, f/f_n, Gamma, e)."""
    return ZetaEffProvider(filling_ratio=filling_ratio, gamma=gamma,
                           restitution=restitution,
                           mass_ratio=mass_ratio, **kwargs).zeta(freq_ratio)


# ===================================================================
# 2. Analytic anchor: Den Hartog two-degree-of-freedom system
# ===================================================================

def two_dof_response(r, mu: float, nu: float, zeta_d: float,
                     zeta_1: float = 0.0):
    """Closed-form 2-DOF response of the primary mass, force-excited.

    Primary mass ``m1`` carries the absorber ``m2 = mu * m1`` (tuning
    ``nu = omega_2 / omega_1``, damping ``zeta_d``); ``r = omega / omega_1``.
    Returns the complex ratio ``X1 / (P/k1)``.  With ``zeta_1 = 0`` this is the
    textbook Den Hartog / Rao response function.
    """
    r = np.asarray(r, dtype=float)
    d1 = (1.0 - r ** 2) + 1j * 2.0 * zeta_1 * r
    num = (nu ** 2 - r ** 2) + 1j * 2.0 * zeta_d * nu * r
    den = (d1 * (nu ** 2 - r ** 2) - mu * nu ** 2 * r ** 2) \
        + 1j * 2.0 * zeta_d * nu * r * (d1 - mu * r ** 2)
    return num / den


def den_hartog_optimum(mu: float) -> tuple[float, float]:
    """Den Hartog closed-form optimum (nu_opt, zeta_opt) for mass ratio ``mu``.

    nu_opt   = 1 / (1 + mu)
    zeta_opt = sqrt(3 mu / (8 (1 + mu)))
    """
    nu_opt = 1.0 / (1.0 + mu)
    zeta_opt = math.sqrt(3.0 * mu / (8.0 * (1.0 + mu)))
    return nu_opt, zeta_opt


def den_hartog_peak_amplification(mu: float) -> float:
    """Den Hartog peak dynamic magnification of the optimally tuned system."""
    return math.sqrt(1.0 + 2.0 / mu)


def _response_peak(response_fn, r_lo: float, r_hi: float,
                   n_grid: int = 2001, n_refine: int = 60) -> tuple[float, float]:
    """Peak of |response| over [r_lo, r_hi] with local golden-section refine."""
    r = np.linspace(r_lo, r_hi, n_grid)
    mag = np.abs(response_fn(r))
    i = int(np.argmax(mag))
    peak_r, peak_mag = float(r[i]), float(mag[i])
    if 0 < i < len(r) - 1:
        a, b = float(r[i - 1]), float(r[i + 1])
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(n_refine):
            fc = float(np.abs(response_fn(np.array([c]))[0]))
            fd = float(np.abs(response_fn(np.array([d]))[0]))
            if fc > fd:
                b, d = d, c
                c = b - gr * (b - a)
            else:
                a, c = c, d
                d = a + gr * (b - a)
        x = 0.5 * (a + b)
        fx = float(np.abs(response_fn(np.array([x]))[0]))
        if fx > peak_mag:
            peak_r, peak_mag = x, fx
    return peak_r, peak_mag


def residual_background_compliance(structure: ModalStructure,
                                   target_mode: int = 0) -> float:
    """Static compliance of the non-target modes, normalised by k_1.

    ``beta = sum_{n != target} phi_n(x_o) phi_n(x_f) (omega_1/omega_n)^2``.
    Adding beta to the 2-DOF closed form makes the analytic anchor represent
    the same physics as the MDOF solve (target mode + modal background), so
    that the comparison in criterion (b) tests the implementation rather than
    the well-known mode-interaction effect.
    """
    omega_1 = structure.modes[target_mode].omega
    beta = 0.0
    for k, m in enumerate(structure.modes):
        if k == target_mode:
            continue
        beta += m.phi_o * m.phi_f * (omega_1 / m.omega) ** 2
    return beta


def two_dof_insertion_loss_peak(mu: float, nu: float, zeta_d: float,
                                zeta_1: float = 0.0,
                                band: tuple[float, float] = (0.5, 1.5),
                                background_compliance: float = 0.0) -> dict:
    """Band-peak insertion loss (dB) of the closed-form 2-DOF system.

    ``IL_peak = 20 log10( max|X_without| / max|X_with| )`` over ``band``,
    where ``X_without`` is the bare primary system.  Both peaks are searched on
    the same band, so a device that merely shifts the resonance out of the band
    is not credited.
    """
    def with_damper(r):
        return two_dof_response(r, mu, nu, zeta_d, zeta_1) + background_compliance

    def without(r):
        return 1.0 / ((1.0 - np.asarray(r, float) ** 2)
                      + 1j * 2.0 * zeta_1 * np.asarray(r, float)) \
            + background_compliance

    r0, m0 = _response_peak(without, *band)
    r1, m1 = _response_peak(with_damper, *band)
    il = 20.0 * math.log10(m0 / m1) if m1 > 0 else float("inf")
    return {"il_peak_db": il, "r_peak_without": r0, "r_peak_with": r1,
            "peak_without": m0, "peak_with": m1}


# ===================================================================
# 3. MDOF modal structures (mass-normalised)
# ===================================================================

@dataclass
class Mode:
    """One mass-normalised structural mode sampled at the three key points."""
    index: int
    omega: float        # rad/s
    phi_d: float        # mode shape at the damper attachment point
    phi_o: float        # mode shape at the observation point
    phi_f: float        # mode shape at the force application point
    participation: float  # base-excitation participation Gamma_n
    label: str = ""

    @property
    def frequency_hz(self) -> float:
        return self.omega / (2.0 * math.pi)


@dataclass
class ModalStructure:
    """Mass-normalised MDOF structure: modal mass = 1, k_n = omega_n^2."""
    name: str
    modes: list[Mode]
    zeta: list[float]
    meta: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def omega_1(self) -> float:
        return self.modes[0].omega

    def omegas(self) -> np.ndarray:
        return np.array([m.omega for m in self.modes])

    def phi_d(self) -> np.ndarray:
        return np.array([m.phi_d for m in self.modes])

    def phi_o(self) -> np.ndarray:
        return np.array([m.phi_o for m in self.modes])

    def phi_f(self) -> np.ndarray:
        return np.array([m.phi_f for m in self.modes])

    def participation(self) -> np.ndarray:
        return np.array([m.participation for m in self.modes])

    def modal_damping(self) -> np.ndarray:
        return np.array(self.zeta, dtype=float)

    def effective_mass_at_damper(self, mode: int = 0) -> float:
        """M_att = 1 / phi_n(x_d)^2 (mass-normalised modal mass = 1)."""
        phi = self.modes[mode].phi_d
        if abs(phi) < 1e-12:
            raise ValueError("damper attached at a node of the target mode")
        return 1.0 / phi ** 2


_CANT_BETA_L = [1.8751040687119611, 4.694091132974174, 7.854757438237613,
                10.995540734875467, 14.137168391046471, 17.278759532088236,
                20.420352251041251, 23.561944901923447, 26.703537555513244,
                29.845130209103034, 32.986722862692826, 36.128315516282618]


def _cantilever_psi(x: np.ndarray, beta_l: float, length: float) -> np.ndarray:
    """Un-normalised clamped-free Euler-Bernoulli mode shape."""
    beta = beta_l / length
    s = -(math.cosh(beta_l) + math.cos(beta_l)) / (math.sinh(beta_l)
                                                   + math.sin(beta_l))
    return (np.cosh(beta * x) - np.cos(beta * x)
            + s * (np.sinh(beta * x) - np.sin(beta * x)))


@functools.lru_cache(maxsize=8)
def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def _gauss_legendre_nodes(length: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    xi, w = _leggauss(n)
    xi, w = np.array(xi), np.array(w)
    x = 0.5 * length * (xi + 1.0)
    w = 0.5 * length * w
    return x, w


def cantilever_structure(length: float = 1.0, width: float = 0.05,
                         thickness: float = 0.01, E: float = 210e9,
                         rho: float = 7850.0, n_modes: int = 6,
                         x_damper: float | None = None,
                         x_obs: float | None = None,
                         x_force: float | None = None,
                         zeta: float = 0.005) -> ModalStructure:
    """Uniform steel cantilever beam, mass-normalised Euler-Bernoulli modes."""
    x_damper = length if x_damper is None else x_damper
    x_obs = length if x_obs is None else x_obs
    x_force = length if x_force is None else x_force

    A = width * thickness
    EI = E * width * thickness ** 3 / 12.0
    mass_per_length = rho * A
    xg, wg = _gauss_legendre_nodes(length)

    modes: list[Mode] = []
    for n in range(n_modes):
        if n < len(_CANT_BETA_L):
            beta_l = _CANT_BETA_L[n]
        else:
            beta_l = (2 * n + 1) * math.pi / 2.0
        psi = _cantilever_psi(xg, beta_l, length)
        norm2 = float(np.sum(wg * psi ** 2)) * mass_per_length
        norm = math.sqrt(norm2)
        omega = beta_l ** 2 * math.sqrt(EI / (mass_per_length * length ** 4))
        phi_d = float(_cantilever_psi(np.array([x_damper]), beta_l, length)[0]) / norm
        phi_o = float(_cantilever_psi(np.array([x_obs]), beta_l, length)[0]) / norm
        phi_f = float(_cantilever_psi(np.array([x_force]), beta_l, length)[0]) / norm
        gamma_n = float(np.sum(wg * psi)) * mass_per_length / norm
        modes.append(Mode(index=n, omega=omega, phi_d=phi_d, phi_o=phi_o,
                          phi_f=phi_f, participation=gamma_n,
                          label=f"cantilever mode {n + 1}"))

    return ModalStructure(
        name="cantilever", modes=modes, zeta=[zeta] * n_modes,
        meta={"length_m": length, "width_m": width, "thickness_m": thickness,
              "E_Pa": E, "rho_kgm3": rho,
              "tip_mass_kg": mass_per_length * length,
              "x_damper_m": x_damper, "x_obs_m": x_obs, "x_force_m": x_force,
              "damping_ratio": zeta})


def stiffened_plate_structure(a: float = 1.0, b: float = 0.6,
                              thickness: float = 0.006, E: float = 210e9,
                              rho: float = 7850.0, nu_poisson: float = 0.3,
                              n_stiffeners: int = 3,
                              stiffener_area: float = 1.2e-4,
                              stiffener_inertia: float = 2.0e-9,
                              n_modes: int = 8,
                              x_damper: tuple[float, float] | None = None,
                              x_obs: tuple[float, float] | None = None,
                              x_force: tuple[float, float] | None = None,
                              zeta: float = 0.005) -> ModalStructure:
    """Simply-supported Kirchhoff panel with smeared longitudinal stiffeners.

    Stiffener bending stiffness is smeared into ``D_x``:
    ``D_x = D + E I_s / s`` with spacing ``s = b / n_stiffeners``; the coupling
    stiffness uses the standard orthotropic approximation ``H = sqrt(D_x D_y)``.
    Mode shapes ``sin(m pi x / a) sin(n pi y / b)`` are mass-normalised with
    ``rho h a b / 4 = 1``.
    """
    x_damper = (a / 2.0, b / 2.0) if x_damper is None else x_damper
    x_obs = x_damper if x_obs is None else x_obs
    x_force = x_damper if x_force is None else x_force

    D = E * thickness ** 3 / (12.0 * (1.0 - nu_poisson ** 2))
    spacing = b / max(n_stiffeners, 1)
    D_x = D + E * stiffener_inertia / spacing
    D_y = D
    H = math.sqrt(D_x * D_y)
    rho_h = rho * thickness
    scale = 2.0 / math.sqrt(rho_h * a * b)

    raw: list[tuple[float, int, int, float, float, float, float]] = []
    for m in range(1, n_modes + 2):
        for n in range(1, n_modes + 2):
            km, kn = m * math.pi / a, n * math.pi / b
            omega = math.sqrt((D_x * km ** 4 + 2.0 * H * km ** 2 * kn ** 2
                               + D_y * kn ** 4) / rho_h)
            phi_at = lambda x, y: scale * math.sin(km * x) * math.sin(kn * y)
            phi_d = phi_at(*x_damper)
            phi_o = phi_at(*x_obs)
            phi_f = phi_at(*x_force)
            # base-excitation participation Gamma_n = int int rho h phi_n dA
            gamma_n = (rho_h * scale * (a / (m * math.pi)) * (b / (n * math.pi))
                       * (1.0 - math.cos(m * math.pi))
                       * (1.0 - math.cos(n * math.pi)))
            raw.append((omega, m, n, phi_d, phi_o, phi_f, gamma_n))

    raw.sort(key=lambda t: t[0])
    modes = [Mode(index=i, omega=w, phi_d=p_d, phi_o=p_o, phi_f=p_f,
                  participation=g, label=f"panel mode ({m},{n})")
             for i, (w, m, n, p_d, p_o, p_f, g) in enumerate(raw[:n_modes])]

    return ModalStructure(
        name="stiffened_plate", modes=modes, zeta=[zeta] * len(modes),
        meta={"a_m": a, "b_m": b, "thickness_m": thickness, "E_Pa": E,
              "rho_kgm3": rho, "poisson": nu_poisson,
              "n_stiffeners": n_stiffeners,
              "stiffener_area_m2": stiffener_area,
              "stiffener_inertia_m4": stiffener_inertia,
              "D_Nm": D, "D_x_Nm": D_x, "D_y_Nm": D_y, "H_Nm": H,
              "damping_ratio": zeta})


# ===================================================================
# 4. PID model on an MDOF structure (absorber equivalence)
# ===================================================================

@dataclass
class ParticleDamperModel:
    """PID as a damped absorber whose damping is the IN32-D02 zeta_eff.

    ``mass_ratio`` is mu = m_d / M_att (M_att = effective modal mass at the
    attachment point).  ``detuning_coupling`` selects whether a mistuned bed
    also loses dissipative capacity (per the delivered Lorentzian surface).
    """
    filling_ratio: float = 0.30
    mass_ratio: float = 0.03
    gamma: float = 1.5
    restitution: float = 0.6
    detuning_coupling: bool = True
    n_particles: int = N_PARTICLES_REF
    primary_mass: float = 1.0
    primary_stiffness: float = 1000.0
    cavity_length: float = 0.1
    gravity: float = G

    def provider(self) -> ZetaEffProvider:
        return ZetaEffProvider(
            filling_ratio=self.filling_ratio, gamma=self.gamma,
            restitution=self.restitution, mass_ratio=self.mass_ratio,
            n_particles=self.n_particles, primary_mass=self.primary_mass,
            primary_stiffness=self.primary_stiffness,
            cavity_length=self.cavity_length, gravity=self.gravity)

    def zeta_d(self, nu: float = 1.0) -> float:
        """Absorber damping ratio at tuning ratio ``nu``."""
        if self.filling_ratio <= 0.0 or self.mass_ratio <= 0.0:
            return 0.0
        prov = self.provider()
        if self.detuning_coupling:
            return prov.zeta(nu)
        return prov.zeta_at_peak()

    def effective_mass_ratio(self, structure: ModalStructure,
                             mode: int = 0) -> float:
        """Mass ratio versus the *structural* modal mass (mu_eff = mu phi^2)."""
        return self.mass_ratio * structure.modes[mode].phi_d ** 2

    def fill_fraction(self) -> float:
        """Ratio of the actual granular fill to the nominal design fill.

        ``mu`` is quoted at the nominal filling ``PHI_REF``; an emptying cavity
        loses both mass and dissipation proportionally, which is what makes the
        zero-filling negative control physically meaningful.
        """
        return max(self.filling_ratio, 0.0) / PHI_REF

    def damper_mass(self, structure: ModalStructure, mode: int = 0) -> float:
        """Physical damper mass: m_d = mu * M_att * (phi / PHI_REF)."""
        return (self.mass_ratio * self.fill_fraction()
                * structure.effective_mass_at_damper(mode))

    def elements(self, structure: ModalStructure, nu: float, mode: int = 0):
        """(m_d, k_d, c_d) of the equivalent absorber."""
        m_d = self.damper_mass(structure, mode)
        omega_d = nu * structure.modes[mode].omega
        zeta_d = self.zeta_d(nu)
        k_d = m_d * omega_d ** 2
        c_d = 2.0 * zeta_d * m_d * omega_d
        return m_d, k_d, c_d


# ===================================================================
# 5. Coupled (N+1)-DOF frequency-domain solver
# ===================================================================

def modal_response(structure: ModalStructure, omega, damper: ParticleDamperModel
                   | None = None, nu: float = 1.0, exciter: str = "force",
                   excite_amplitude: float = 1.0,
                   zetas=None) -> np.ndarray:
    """Complex displacement at the observation point for each ``omega``.

    Solves the coupled system of ``N`` mass-normalised modal coordinates plus
    one absorber DOF exactly (no modal truncation of the coupling).  For
    ``N = 1`` this is identical to the Den Hartog 2-DOF system.

    exciter = "force" : harmonic point force of amplitude ``excite_amplitude``
                        at x_force.
    exciter = "base"  : uniform base acceleration ``excite_amplitude``.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(structure.modes)
    phi_d = structure.phi_d()
    phi_o = structure.phi_o()
    phi_f = structure.phi_f()
    zetas = structure.modal_damping() if zetas is None else np.asarray(zetas, float)
    omegas = structure.omegas()

    m_d = k_d = c_d = 0.0
    if damper is not None:
        m_d, k_d, c_d = damper.elements(structure, nu)
        if m_d <= 0.0 and k_d <= 0.0 and c_d <= 0.0:
            damper = None          # empty cavity: no particles -> bare system

    m_pts = omega.size
    sys_size = n + (1 if damper is not None else 0)
    A = np.zeros((m_pts, sys_size, sys_size), dtype=complex)
    b = np.zeros((m_pts, sys_size), dtype=complex)

    idx = np.arange(n)
    A[:, idx, idx] = (-omega[:, None] ** 2
                      + 1j * 2.0 * zetas * omegas * omega[:, None]
                      + omegas ** 2)

    if exciter == "force":
        b[:, :n] = excite_amplitude * phi_f
    elif exciter == "base":
        b[:, :n] = -excite_amplitude * structure.participation()
    else:
        raise ValueError(f"unknown exciter {exciter!r}")

    if damper is not None:
        Z = k_d + 1j * omega * c_d                      # (m_pts,)
        A[:, :n, :n] += Z[:, None, None] * np.outer(phi_d, phi_d)[None, :, :]
        A[:, :n, n] += -Z[:, None] * phi_d[None, :]
        A[:, n, :n] += -Z[:, None] * phi_d[None, :]
        A[:, n, n] += -omega ** 2 * m_d + Z

    x = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return x[:, :n] @ phi_o


def insertion_loss(structure: ModalStructure, omega, damper: ParticleDamperModel,
                   nu: float = 1.0, exciter: str = "force",
                   zetas=None) -> np.ndarray:
    """IL(f) = 20 log10(|X_without| / |X_with|) in dB (pointwise)."""
    with_d = np.abs(modal_response(structure, omega, damper, nu=nu,
                                   exciter=exciter, zetas=zetas))
    without = np.abs(modal_response(structure, omega, None,
                                    exciter=exciter))
    tiny = 1e-300
    return 20.0 * np.log10(np.maximum(without, tiny) / np.maximum(with_d, tiny))


def rms_insertion_loss(structure: ModalStructure, damper: ParticleDamperModel
                       | None, nu: float = 1.0,
                       band: tuple[float, float] = (0.5, 1.5),
                       exciter: str = "force", zetas=None,
                       n_grid: int = 2001) -> float:
    """Mean-square (broadband) insertion loss in dB.

    ``IL_rms = 10 log10( sum|X_without|^2 / sum|X_with|^2 )`` over the band.
    Unlike the band-peak IL this functional is *smooth* in the tuning ratio,
    because it does not select one of two nearly equal split peaks; it is the
    natural metric for broadband / random excitation and is used to verify the
    MDOF implementation against the closed form.
    """
    w1 = structure.omega_1
    w = np.linspace(band[0] * w1, band[1] * w1, n_grid)
    with_d = np.abs(modal_response(structure, w, damper, nu=nu,
                                   exciter=exciter, zetas=zetas)) ** 2
    without = np.abs(modal_response(structure, w, None, exciter=exciter)) ** 2
    return 10.0 * math.log10(float(without.sum()) / max(float(with_d.sum()), 1e-300))


def rms_insertion_loss_2dof(mu: float, nu: float, zeta_d: float,
                            zeta_1: float = 0.0,
                            band: tuple[float, float] = (0.5, 1.5),
                            n_grid: int = 4001) -> float:
    """Closed-form counterpart of :func:`rms_insertion_loss` (2-DOF anchor)."""
    r = np.linspace(band[0], band[1], n_grid)
    with_d = np.abs(two_dof_response(r, mu, nu, zeta_d, zeta_1)) ** 2
    without = np.abs(1.0 / ((1.0 - r ** 2) + 1j * 2.0 * zeta_1 * r)) ** 2
    return 10.0 * math.log10(float(without.sum()) / max(float(with_d.sum()), 1e-300))


def peak_response(structure: ModalStructure, damper: ParticleDamperModel | None,
                  nu: float = 1.0, band: tuple[float, float] = (0.5, 1.5),
                  exciter: str = "force", n_grid: int = 2001,
                  n_refine: int = 60, zetas=None) -> tuple[float, float]:
    """Maximum |X| over ``band`` (multiples of omega_1), with local refinement.

    The bare-structure peak (``damper=None``) is cached on the structure since
    it is the IL denominator and is recomputed for every tuning ratio.
    """
    cache_key = None if (damper is not None or zetas is not None) \
        else ("bare_peak", band, exciter, n_grid)
    if cache_key is not None and cache_key in structure._cache:
        return structure._cache[cache_key]
    w1 = structure.omega_1
    w = np.linspace(band[0] * w1, band[1] * w1, n_grid)
    mag = np.abs(modal_response(structure, w, damper, nu=nu, exciter=exciter,
                                zetas=zetas))
    i = int(np.argmax(mag))
    best_w, best_mag = float(w[i]), float(mag[i])
    if 0 < i < len(w) - 1:
        a, bb = float(w[i - 1]), float(w[i + 1])
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = bb - gr * (bb - a), a + gr * (bb - a)
        for _ in range(n_refine):
            fc = float(np.abs(modal_response(structure, np.array([c]), damper,
                                             nu=nu, exciter=exciter,
                                             zetas=zetas)[0]))
            fd = float(np.abs(modal_response(structure, np.array([d]), damper,
                                             nu=nu, exciter=exciter,
                                             zetas=zetas)[0]))
            if fc > fd:
                bb, d = d, c
                c = bb - gr * (bb - a)
            else:
                a, c = c, d
                d = a + gr * (bb - a)
        x = 0.5 * (a + bb)
        fx = float(np.abs(modal_response(structure, np.array([x]), damper,
                                         nu=nu, exciter=exciter,
                                         zetas=zetas)[0]))
        if fx > best_mag:
            best_w, best_mag = x, fx
    if cache_key is not None:
        structure._cache[cache_key] = (best_w, best_mag)
    return best_w, best_mag


def peak_insertion_loss(structure: ModalStructure, damper: ParticleDamperModel,
                        nu: float = 1.0, band: tuple[float, float] = (0.5, 1.5),
                        exciter: str = "force", zetas=None) -> dict:
    """Band-peak insertion loss (dB) of the MDOF structure with the PID."""
    w0, m0 = peak_response(structure, None, band=band, exciter=exciter)
    w1p, m1 = peak_response(structure, damper, nu=nu, band=band, exciter=exciter,
                            zetas=zetas)
    il = 20.0 * math.log10(m0 / m1) if m1 > 0 else float("inf")
    return {"il_peak_db": il, "omega_peak_without": w0,
            "omega_peak_with": w1p, "peak_without": m0, "peak_with": m1,
            "omega_peak_without_hz": w0 / (2 * math.pi),
            "omega_peak_with_hz": w1p / (2 * math.pi), "nu": nu,
            "zeta_d": damper.zeta_d(nu) if damper is not None else 0.0}


def optimise_tuning(structure: ModalStructure, damper: ParticleDamperModel,
                    nu_values=None, band: tuple[float, float] = (0.5, 1.5),
                    exciter: str = "force") -> dict:
    """Scan the tuning ratio nu and return the IL-maximising point."""
    if nu_values is None:
        nu_values = [0.80 + 0.05 * i for i in range(7)]  # 0.80 ... 1.10
    best = None
    curve = []
    for nu in nu_values:
        res = peak_insertion_loss(structure, damper, nu=nu, band=band,
                                  exciter=exciter)
        curve.append({"nu": nu, "il_peak_db": res["il_peak_db"],
                      "zeta_d": res["zeta_d"]})
        if best is None or res["il_peak_db"] > best["il_peak_db"]:
            best = res
    # golden-section refinement of the tuning optimum around the coarse argmax
    i = min(range(len(curve)), key=lambda k: -curve[k]["il_peak_db"])
    lo, hi = curve[max(i - 1, 0)]["nu"], curve[min(i + 1, len(curve) - 1)]["nu"]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f = lambda x: peak_insertion_loss(structure, damper, nu=x, band=band,
                                      exciter=exciter)["il_peak_db"]
    fc, fd = f(c), f(d)
    for _ in range(15):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    for x in (0.5 * (lo + hi),):
        res = peak_insertion_loss(structure, damper, nu=x, band=band,
                                  exciter=exciter)
        curve.append({"nu": x, "il_peak_db": res["il_peak_db"],
                      "zeta_d": res["zeta_d"], "refined": True})
        if res["il_peak_db"] > best["il_peak_db"]:
            best = res
    best["curve"] = curve
    return best


# ===================================================================
# 6. SDOF bridge: FRF identification (Model A lives in section 6b below)
# ===================================================================

def sdof_response(omega, omega_n: float, zeta: float) -> np.ndarray:
    """Unit-static-displacement SDOF FRF (force excited)."""
    omega = np.asarray(omega, dtype=float)
    return 1.0 / ((1.0 - (omega / omega_n) ** 2) + 1j * 2.0 * zeta * (omega / omega_n))


def identify_zeta_half_power(omega_n: float, zeta_true: float,
                             n_grid: int = 400001,
                             span: float = 0.20) -> float:
    """Recover zeta from an FRF by the half-power (-3 dB) bandwidth method.

    Independent numerical path (does not read the delivered surface): builds
    the SDOF FRF, locates the peak, scans outward until the magnitude drops
    below ``peak/sqrt(2)`` and bisects the two -3 dB crossings, then returns
    ``zeta = df / (2 f_n)`` in ratio units.
    """
    r = np.linspace(1.0 - span, 1.0 + span, n_grid)
    mag = np.abs(sdof_response(r * omega_n, omega_n, zeta_true))
    i = int(np.argmax(mag))
    target = float(mag[i]) / math.sqrt(2.0)

    def bisect(lo: float, hi: float) -> float:
        m_lo = float(np.abs(sdof_response(np.array([lo * omega_n]), omega_n,
                                          zeta_true)[0]))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            m_mid = float(np.abs(sdof_response(np.array([mid * omega_n]),
                                               omega_n, zeta_true)[0]))
            if (m_mid - target) * (m_lo - target) <= 0:
                hi = mid
            else:
                lo, m_lo = mid, m_mid
        return 0.5 * (lo + hi)

    # low crossing: outermost grid point below target, towards the peak
    j_lo = i
    while j_lo > 0 and mag[j_lo] >= target:
        j_lo -= 1
    j_hi = i
    while j_hi < len(r) - 1 and mag[j_hi] >= target:
        j_hi += 1
    if j_lo == 0 or j_hi == len(r) - 1:
        raise ValueError("half-power crossing outside the scanned band")
    r_lo = bisect(float(r[j_lo]), float(r[j_lo + 1]))
    r_hi = bisect(float(r[j_hi - 1]), float(r[j_hi]))
    return (r_hi - r_lo) / 2.0


def inject_modal_damping(structure: ModalStructure, provider: ZetaEffProvider,
                         nu: float = 1.0, target_mode: int = 0) -> np.ndarray:
    """Model A: add the delivered zeta_eff to the modal damping (point device).

    The device is designed for the frequency ``omega_d = nu * omega_1``; each
    mode n sees the surface argument ``omega_n / omega_d`` (so the target mode
    is evaluated at the surface peak when ``nu = 1``).  A point device
    dissipating with the local velocity adds
    ``zeta_eff * (phi_n(x_d) / phi_max)^2`` to mode n.
    """
    omega_d = nu * structure.modes[target_mode].omega
    phi_max = max(abs(m.phi_d) for m in structure.modes)
    zetas = []
    for m, z0 in zip(structure.modes, structure.modal_damping()):
        z_eff = provider.zeta(m.omega / omega_d)
        zetas.append(z0 + z_eff * (m.phi_d / phi_max) ** 2)
    return np.array(zetas)


def injection_peak_insertion_loss(structure: ModalStructure,
                                  provider: ZetaEffProvider, nu: float = 1.0,
                                  band: tuple[float, float] = (0.5, 1.5),
                                  exciter: str = "force",
                                  target_mode: int = 0) -> dict:
    """Band-peak IL of the MDOF structure under Model A (zeta_eff injection)."""
    zetas = inject_modal_damping(structure, provider, nu=nu,
                                 target_mode=target_mode)
    res = peak_insertion_loss(structure, None, band=band, exciter=exciter,
                              zetas=zetas)
    res["zeta_added_target_mode"] = zetas[target_mode] \
        - structure.modal_damping()[target_mode]
    res["nu"] = nu
    return res


def optimise_injection_tuning(structure: ModalStructure,
                              provider: ZetaEffProvider,
                              nu_values=None,
                              band: tuple[float, float] = (0.5, 1.5),
                              exciter: str = "force") -> dict:
    """Model A tuning scan: device design frequency ratio maximising IL."""
    if nu_values is None:
        nu_values = [0.70 + 0.04 * i for i in range(13)]  # 0.70 ... 1.18
    curve, best = [], None
    for nu in nu_values:
        res = injection_peak_insertion_loss(structure, provider, nu=nu,
                                           band=band, exciter=exciter)
        curve.append({"nu": nu, "il_peak_db": res["il_peak_db"],
                      "zeta_added": res["zeta_added_target_mode"]})
        if best is None or res["il_peak_db"] > best["il_peak_db"]:
            best = res
    # golden-section refinement of the tuning optimum around the coarse argmax
    i = min(range(len(curve)), key=lambda k: -curve[k]["il_peak_db"])
    lo, hi = curve[max(i - 1, 0)]["nu"], curve[min(i + 1, len(curve) - 1)]["nu"]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f = lambda x: injection_peak_insertion_loss(structure, provider, nu=x,
                                                band=band, exciter=exciter)
    fc, fd = f(c), f(d)
    for _ in range(12):
        if fc["il_peak_db"] > fd["il_peak_db"]:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    res = f(0.5 * (lo + hi))
    curve.append({"nu": res["nu"], "il_peak_db": res["il_peak_db"],
                  "zeta_added": res["zeta_added_target_mode"], "refined": True})
    if res["il_peak_db"] > best["il_peak_db"]:
        best = res
    best["curve"] = curve
    return best


# ===================================================================
# 7. Floating-raft (two-stage) isolation reference
# ===================================================================

def floating_raft_transmissibility(r, mu_raft: float, r_iso: float,
                                   zeta_iso: float = 0.05) -> np.ndarray:
    """Force transmissibility of a two-stage floating raft (hull rigid).

    Machine ``m1`` -> isolator 1 -> raft ``m2 = mu_raft m1`` -> isolator 2 ->
    rigid hull.  Both isolators share the isolation frequency ratio
    ``r_iso = omega_iso / omega_1`` (``omega_1`` = structural resonance) and
    damping ``zeta_iso``.  ``r = omega / omega_1``.
    """
    r = np.asarray(r, dtype=float)
    r1, r2 = r / r_iso, r / r_iso
    z = 2.0 * zeta_iso * 1j * r1
    # machine + raft mass matrix (unit machine mass)
    a11 = -r1 ** 2 + z + 1.0
    a12 = -z - 1.0
    a21 = -z - 1.0
    a22 = -mu_raft * r2 ** 2 + z + 1.0 + (1j * 2.0 * zeta_iso * r2 + 1.0)
    det = a11 * a22 - a12 * a21
    x2 = -a21 / det
    # force transmitted to the hull through isolator 2
    return np.abs((1.0 + 2.0 * 1j * zeta_iso * r2) * x2)


def floating_raft_il_db(omega_ratio, mu_raft: float, r_iso: float = 0.3,
                        zeta_iso: float = 0.05) -> float:
    """IL (dB) of the floating raft at a single frequency ratio (force based)."""
    T = float(floating_raft_transmissibility(np.array([omega_ratio]), mu_raft,
                                             r_iso, zeta_iso)[0])
    return -20.0 * math.log10(max(T, 1e-300))


# ===================================================================
# 8. Exports
# ===================================================================

__all__ = [
    "G", "MU_REF", "N_PARTICLES_REF", "delivered_natural_frequency",
    "ZetaEffProvider", "zeta_eff_delivered",
    "two_dof_response", "den_hartog_optimum", "den_hartog_peak_amplification",
    "two_dof_insertion_loss_peak",
    "Mode", "ModalStructure", "cantilever_structure",
    "stiffened_plate_structure",
    "ParticleDamperModel",
    "modal_response", "insertion_loss", "peak_response",
    "rms_insertion_loss", "rms_insertion_loss_2dof",
    "peak_insertion_loss", "optimise_tuning",
    "residual_background_compliance",
    "inject_modal_damping", "injection_peak_insertion_loss",
    "optimise_injection_tuning",
    "sdof_response", "identify_zeta_half_power", "PHI_REF",
    "floating_raft_transmissibility", "floating_raft_il_db",
]