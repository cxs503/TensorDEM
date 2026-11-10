"""Rigid floe response to linear waves: time domain vs frequency domain.

What is verified here
---------------------
An ice floe is modelled as a **rigid body with two degrees of freedom** (heave
``z``, pitch ``theta``) resting on the free surface, loaded by (i) per-particle
Archimedes buoyancy evaluated against the *instantaneous local* free-surface
elevation ``eta(x, t)`` and (ii) a per-particle relative-acceleration Morison
load,

    F_i = rho_w * A_i * f_i * [ C_m * a_wz(x_i, z_i, t) - C_a * a_body_i ],

with ``a_body_i = zdotdot + thetadotdot * x_i`` the particle acceleration.  That
is the standard Morison/added-mass decomposition: ``C_m = 1 + C_a`` recovers the
Froude-Krylov + inertia form and the ``C_a`` term *is* the added mass, so the
2x2 mass matrix ``[[m + a33, a35], [a35, I + a55]]`` is assembled implicitly
(operators frozen at the equilibrium submerged fraction, which is what makes the
time-domain system identical to the frequency-domain one).

Particle idealisation
---------------------
The floe section ``L x h`` is tiled by ``n_x * n_y`` **rectangular columns** of
area ``dx*dy = L h/(n_x n_y)``.  Columns tile the rectangle exactly, so the
submerged-area sum, the waterplane chord sum and the displaced volume are exact
by construction and converge monotonically with refinement -- unlike a lattice of
overlapping discs, which over-counts the waterplane by ``O(r/dy)``.  The buoyancy
of a column is the hydrostatic integral over its own vertical extent,
``rho_w g dx * clip(-bottom, 0, dy)``, and its antiderivative is available in
closed form (:func:`cell_potential`), which is what the energy ledger uses.

Two independent routes to the same response are provided and compared:

``FloeResponseModel`` (time domain)
    RK4 on the nonlinear (exact rotation, per-column buoyancy against the local
    ``eta``) two-degree-of-freedom body.  This is the *solver*: it only knows
    per-particle forces, exactly like the DEM side of the shipped wet coupling.

``frequency_response`` (frequency domain, two flavours)
    A complex 2x2 transfer function.  ``reference="continuum"`` uses the exact
    continuum coefficients of the same physical model (``K33 = rho g L``,
    ``K55 = rho g L^3/12``, ``m = rho_i L h``, ``a33 = rho_w C_a L d``, ...) with
    the excitation integrals evaluated by fine rectangle quadrature -- no
    particles anywhere.  ``reference="discrete"`` uses the *same* sums the solver
    uses, linearised about equilibrium.

Excitation has two parts, both present in the solver and in the transfer
function: the inertia load ``rho_w C_m a_wz`` on the submerged part, and the
free-surface buoyancy excess ``rho_w g chord * eta`` (the waterplane pressure of
the incident wave).  The second one is what makes the long-wave limit correct::

    (K33 - (m+a33) w^2) z = rho_w g L A   =>   RAO_3 -> 1 as w -> 0,

i.e. the floe rides the surface -- exact physics, independent of the coefficient
model.

Known limitation (quote it with any number taken from here): with a
*frequency-independent* added mass ``C_a`` and an absolute Froude-Krylov load the
model carries no radiation damping and does not decay at high frequency, so it is
only quoted for ``k L <= ~2``.  Damping is the linearised viscous term
``b = 2 zeta sqrt(K (m + a))`` with ``zeta`` an *input* (drag linearisation), not
a potential-flow radiation coefficient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .waves import RegularWave, solve_wavenumber

GRAVITY = 9.81
RHO_WATER = 1025.0
RHO_ICE = 917.0


# ---------------------------------------------------------------------------
# Column geometry: submerged area, its antiderivative, waterplane chord
# ---------------------------------------------------------------------------

def cell_submerged_area(bottom, dx: float, dy: float) -> np.ndarray:
    """Submerged area of a column whose *bottom* sits ``bottom`` above the surface."""
    depth = np.clip(-np.asarray(bottom, dtype=float), 0.0, dy)
    return dx * depth


def cell_potential(bottom, dx: float, dy: float) -> np.ndarray:
    """``Phi(bottom) = int_{-inf}^{bottom} A_sub(s) ds`` -- closed form.

    ``b >= 0 -> 0``; ``-dy <= b <= 0 -> dx b^2/2``;
    ``b < -dy -> dx (-dy b - dy^2/2)``.  ``dPhi/db = -A_sub(b)``, so
    ``rho_w g Phi`` is the buoyancy potential energy of the column (zero at the
    fully emerged pose).
    """
    b = np.asarray(bottom, dtype=float)
    out = np.zeros_like(b)
    mid = (b <= 0.0) & (b >= -dy)
    out[mid] = dx * 0.5 * b[mid] ** 2
    low = b < -dy
    out[low] = dx * (-dy * b[low] - 0.5 * dy ** 2)
    return out


def cell_chord(bottom, dx: float, dy: float) -> np.ndarray:
    """Waterplane chord of a column: ``dx`` if the surface cuts it, else 0."""
    depth = -np.asarray(bottom, dtype=float)
    return np.where((depth > 0.0) & (depth < dy), dx, 0.0)


# ---------------------------------------------------------------------------
# Lattice discretisation of the floe
# ---------------------------------------------------------------------------

@dataclass
class FloeLattice:
    """A rectangular floe ``L x h`` tiled by ``n_x * n_y`` rectangular columns."""

    length: float = 20.0
    thickness: float = 1.0
    n_x: int = 40
    n_y: int = 2
    rho_ice: float = RHO_ICE
    rho_water: float = RHO_WATER
    gravity: float = GRAVITY

    x: np.ndarray = field(init=False, repr=False)
    z: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)
    dy: float = field(init=False)

    def __post_init__(self) -> None:
        nx, ny = int(self.n_x), int(self.n_y)
        if nx < 1 or ny < 1:
            raise ValueError("n_x and n_y must be >= 1")
        ex = np.linspace(-0.5 * self.length, 0.5 * self.length, nx + 1)
        ez = np.linspace(-0.5 * self.thickness, 0.5 * self.thickness, ny + 1)
        xc = 0.5 * (ex[1:] + ex[:-1])
        zc = 0.5 * (ez[1:] + ez[:-1])
        gx, gz = np.meshgrid(xc, zc, indexing="ij")
        self.x = gx.ravel().copy()
        self.z = gz.ravel().copy()
        self.dx = self.length / nx
        self.dy = self.thickness / ny

    # -- inertial properties -------------------------------------------
    @property
    def n_particles(self) -> int:
        return int(self.x.size)

    @property
    def area_share(self) -> float:
        return self.dx * self.dy

    @property
    def particle_mass(self) -> float:
        return self.rho_ice * self.area_share

    @property
    def mass(self) -> float:
        return self.particle_mass * self.n_particles

    @property
    def inertia(self) -> float:
        """Pitch inertia about the y axis through the centroid (per unit width)."""
        own = self.particle_mass * (self.dx ** 2 + self.dy ** 2) / 12.0
        return float(np.sum(self.particle_mass * (self.x ** 2 + self.z ** 2) + own))

    @property
    def displaced_volume(self) -> float:
        return self.mass / self.rho_water

    @property
    def draft(self) -> float:
        return self.displaced_volume / self.length

    def equilibrium_height(self, tol: float = 1e-14) -> float:
        """Heave offset ``z_eq`` (centroid displacement) at buoyancy balance.

        ``imbalance(z)`` decreases monotonically in ``z``; the bracket is
        ``[-2h, 2h]`` and bisection raises the *lower* bound while the floe is
        still too buoyant.
        """
        weight = self.mass * self.gravity

        def imbalance(z: float) -> float:
            bottom = self.z - 0.5 * self.dy + z
            area = cell_submerged_area(bottom, self.dx, self.dy).sum()
            return self.rho_water * self.gravity * area - weight

        lo, hi = -2.0 * self.thickness - 0.2, 2.0 * self.thickness + 0.2
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if imbalance(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "thickness": self.thickness,
            "n_x": int(self.n_x),
            "n_y": int(self.n_y),
            "n_particles": self.n_particles,
            "dx": self.dx,
            "dy": self.dy,
            "mass": self.mass,
            "inertia": self.inertia,
            "draft": self.draft,
            "displaced_volume": self.displaced_volume,
            "rho_ratio": self.rho_ice / self.rho_water,
        }


# ---------------------------------------------------------------------------
# Coefficients (continuum and discrete)
# ---------------------------------------------------------------------------

@dataclass
class FloeCoefficients:
    """Linear hydrostatic + added-mass coefficients of the 2-D floe."""

    mass: float
    inertia: float
    k33: float
    k35: float
    k55: float
    a33: float
    a35: float
    a55: float
    draft: float

    @property
    def heave_natural_omega(self) -> float:
        return math.sqrt(self.k33 / (self.mass + self.a33))

    @property
    def pitch_natural_omega(self) -> float:
        return math.sqrt(self.k55 / (self.inertia + self.a55))

    def to_dict(self) -> dict:
        return {
            "mass": self.mass, "inertia": self.inertia,
            "k33": self.k33, "k35": self.k35, "k55": self.k55,
            "a33": self.a33, "a35": self.a35, "a55": self.a55,
            "draft": self.draft,
            "heave_natural_omega": self.heave_natural_omega,
            "heave_natural_period": 2 * math.pi / self.heave_natural_omega,
            "pitch_natural_omega": self.pitch_natural_omega,
            "pitch_natural_period": 2 * math.pi / self.pitch_natural_omega,
        }


def continuum_coefficients(lat: FloeLattice,
                           added_mass_coeff: float = 0.5) -> FloeCoefficients:
    """Coefficients of the continuum floe (no lattice anywhere).

    ``K33 = rho_w g L`` (waterplane), ``K55 = rho_w g L^3/12``, ``K35 = 0``
    (symmetric waterplane), ``m = rho_i L h``, ``I = m (L^2 + h^2)/12``,
    added mass on the displaced volume ``a33 = rho_w C_a L d``,
    ``a55 = rho_w C_a L d (L^2 + d^2)/12``, ``a35 = 0``.
    """
    L, h = lat.length, lat.thickness
    d = lat.rho_ice / lat.rho_water * h
    V = L * d
    m = lat.rho_ice * L * h
    return FloeCoefficients(
        mass=m,
        inertia=m * (L ** 2 + h ** 2) / 12.0,
        k33=lat.rho_water * lat.gravity * L,
        k35=0.0,
        k55=lat.rho_water * lat.gravity * L ** 3 / 12.0,
        a33=lat.rho_water * added_mass_coeff * V,
        a35=0.0,
        a55=lat.rho_water * added_mass_coeff * V * (L ** 2 + d ** 2) / 12.0,
        draft=d,
    )


def discrete_coefficients(lat: FloeLattice, z_eq: float,
                          added_mass_coeff: float = 0.5) -> FloeCoefficients:
    """The same coefficients from the column sums (linearised at equilibrium)."""
    bottom = lat.z - 0.5 * lat.dy + z_eq
    chord = cell_chord(bottom, lat.dx, lat.dy)
    depth = np.clip(-bottom, 0.0, lat.dy)
    sub_area = lat.dx * depth
    rho_w, g = lat.rho_water, lat.gravity
    return FloeCoefficients(
        mass=lat.mass,
        inertia=lat.inertia,
        k33=float(rho_w * g * chord.sum()),
        k35=float(rho_w * g * (chord * lat.x).sum()),
        k55=float(rho_w * g * (chord * lat.x ** 2).sum()),
        a33=float(rho_w * added_mass_coeff * sub_area.sum()),
        a35=float(rho_w * added_mass_coeff * (sub_area * lat.x).sum()),
        a55=float(rho_w * added_mass_coeff * (sub_area * lat.x ** 2).sum()),
        draft=lat.draft,
    )


def damping_coefficients(coeffs: FloeCoefficients, zeta_heave: float = 0.10,
                         zeta_pitch: float = 0.10) -> tuple[float, float]:
    """Linearised viscous damping ``b = 2 zeta sqrt(K (m + a))``."""
    b33 = 2.0 * zeta_heave * math.sqrt(coeffs.k33 * (coeffs.mass + coeffs.a33))
    b55 = 2.0 * zeta_pitch * math.sqrt(coeffs.k55 * (coeffs.inertia + coeffs.a55))
    return b33, b55


# ---------------------------------------------------------------------------
# Frequency domain transfer function
# ---------------------------------------------------------------------------

@dataclass
class FrequencyResponse:
    """Complex heave/pitch response of the floe at one wave frequency."""

    omega: float
    wavenumber: float
    amplitude: float
    heave_phasor: complex
    pitch_phasor: complex
    excitation_heave: complex
    excitation_pitch: complex
    buoyancy_excitation_heave: complex
    coeffs: FloeCoefficients
    damping_heave: float
    damping_pitch: float
    reference: str
    floe_length: float = 20.0

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def rao_heave(self) -> float:
        return abs(self.heave_phasor) / self.amplitude

    @property
    def rao_pitch(self) -> float:
        """``|theta| / (k A)`` -- normalised by the surface slope."""
        return abs(self.pitch_phasor) / (self.wavenumber * self.amplitude)

    @property
    def kL(self) -> float:
        return self.wavenumber * self.floe_length

    def to_dict(self) -> dict:
        m, a33 = self.coeffs.mass, self.coeffs.a33
        inertia, a55 = self.coeffs.inertia, self.coeffs.a55
        return {
            "omega": self.omega,
            "period": self.period,
            "kL": self.kL,
            "wavelength": 2 * math.pi / self.wavenumber,
            "rao_heave": self.rao_heave,
            "rao_pitch": self.rao_pitch,
            "heave_phase_deg": math.degrees(np.angle(self.heave_phasor)),
            "pitch_phase_deg": math.degrees(np.angle(self.pitch_phasor)),
            "zeta_heave": self.damping_heave / (
                2.0 * math.sqrt((m + a33) * self.coeffs.k33)),
            "zeta_pitch": self.damping_pitch / (
                2.0 * math.sqrt((inertia + a55) * self.coeffs.k55)),
            "reference": self.reference,
            "heave_natural_period": 2 * math.pi / self.coeffs.heave_natural_omega,
        }


def _decay(wave: RegularWave, z_level) -> np.ndarray:
    k = wave.wavenumber
    z = np.asarray(z_level, dtype=float)
    if not np.isfinite(wave.depth):
        return np.exp(np.clip(k * z, -60.0, 60.0))
    return np.cosh(k * (z + wave.depth)) / math.sinh(k * wave.depth)


def _excitation_discrete(lat: FloeLattice, z_eq: float, wave: RegularWave,
                         added_mass_coeff: float = 0.5,
                         inertia_coeff: float | None = None
                         ) -> tuple[complex, complex, complex, complex]:
    """Phasors ``(F3_inertia, M5_inertia, F3_buoyancy, M5_buoyancy)``.

    Phasor convention ``Re[ q e^{-i w t} ]``; the incident surface elevation is
    ``A e^{i(k x + phi)}`` so the vertical fluid-acceleration phasor is
    ``-A w^2 decay(z) e^{i(k x + phi)}``.  Both terms only act on the submerged
    part of the floe, matching the time-domain force assembly.
    """
    cm = 1.0 + added_mass_coeff if inertia_coeff is None else inertia_coeff
    bottom = lat.z - 0.5 * lat.dy + z_eq
    chord = cell_chord(bottom, lat.dx, lat.dy)
    depth = np.clip(-bottom, 0.0, lat.dy)
    sub_area = lat.dx * depth
    k, w, A, phi = wave.wavenumber, wave.omega, wave.amplitude, wave.phase
    phase = np.exp(1j * (k * lat.x + phi))
    a_phasor = -A * w ** 2 * _decay(wave, lat.z + z_eq) * phase
    w_inertia = lat.rho_water * sub_area * cm
    f3_i = complex(np.sum(w_inertia * a_phasor))
    m5_i = complex(np.sum(w_inertia * lat.x * a_phasor))
    w_buoy = lat.rho_water * lat.gravity * chord * A
    f3_b = complex(np.sum(w_buoy * phase))
    m5_b = complex(np.sum(w_buoy * lat.x * phase))
    return f3_i, m5_i, f3_b, m5_b


def _excitation_continuum(lat: FloeLattice, wave: RegularWave,
                          added_mass_coeff: float = 0.5,
                          inertia_coeff: float | None = None,
                          n_x: int = 401, n_z: int = 41
                          ) -> tuple[complex, complex, complex, complex]:
    """Same phasors from rectangle quadrature over ``[-L/2, L/2] x [-d, 0]``."""
    cm = 1.0 + added_mass_coeff if inertia_coeff is None else inertia_coeff
    L = lat.length
    d = lat.rho_ice / lat.rho_water * lat.thickness
    k, w, A, phi = wave.wavenumber, wave.omega, wave.amplitude, wave.phase
    xs = np.linspace(-0.5 * L, 0.5 * L, n_x)
    zs = np.linspace(-d, 0.0, n_z)
    dx = L / (n_x - 1)
    dz = d / (n_z - 1)
    XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
    phase = np.exp(1j * (k * XX + phi))
    a_phasor = -A * w ** 2 * _decay(wave, ZZ) * phase
    w_inertia = lat.rho_water * cm * dx * dz
    f3_i = complex(np.sum(w_inertia * a_phasor))
    m5_i = complex(np.sum(w_inertia * XX * a_phasor))
    # Waterplane term: rho g int eta dx over the floe length (exact sinc).
    x1 = np.linspace(-0.5 * L, 0.5 * L, 4001)
    eta = A * np.exp(1j * (k * x1 + phi))
    wet = lat.rho_water * lat.gravity * np.gradient(x1)
    f3_b = complex(np.sum(wet * eta))
    m5_b = complex(np.sum(wet * x1 * eta))
    return f3_i, m5_i, f3_b, m5_b


def frequency_response(wave: RegularWave, lat: FloeLattice, z_eq: float,
                       damping_ratio_heave: float = 0.10,
                       damping_ratio_pitch: float = 0.10,
                       added_mass_coeff: float = 0.5,
                       inertia_coeff: float | None = None,
                       reference: str = "continuum") -> FrequencyResponse:
    """Solve the complex 2x2 system for the steady-state heave/pitch response.

    ``[[K33 - (m+a33) w^2 - i w b33, K35 - a35 w^2],
        [K35 - a35 w^2, K55 - (I+a55) w^2 - i w b55]] [z, theta] = [F3, M5]``
    with ``F = F_inertia + F_buoyancy``.
    """
    if reference == "continuum":
        coeffs = continuum_coefficients(lat, added_mass_coeff)
        f3_i, m5_i, f3_b, m5_b = _excitation_continuum(lat, wave, added_mass_coeff,
                                                       inertia_coeff)
    elif reference == "discrete":
        coeffs = discrete_coefficients(lat, z_eq, added_mass_coeff)
        f3_i, m5_i, f3_b, m5_b = _excitation_discrete(lat, z_eq, wave,
                                                      added_mass_coeff, inertia_coeff)
    else:
        raise ValueError(f"unknown reference {reference!r}")

    f3 = f3_i + f3_b
    m5 = m5_i + m5_b
    w = wave.omega
    b33, b55 = damping_coefficients(coeffs, damping_ratio_heave, damping_ratio_pitch)
    m11 = coeffs.k33 - (coeffs.mass + coeffs.a33) * w ** 2 - 1j * w * b33
    m12 = coeffs.k35 - coeffs.a35 * w ** 2
    m22 = coeffs.k55 - (coeffs.inertia + coeffs.a55) * w ** 2 - 1j * w * b55
    det = m11 * m22 - m12 * m12
    z_ph = (m22 * f3 - m12 * m5) / det
    th_ph = (m11 * m5 - m12 * f3) / det
    return FrequencyResponse(
        omega=w, wavenumber=wave.wavenumber, amplitude=wave.amplitude,
        heave_phasor=complex(z_ph), pitch_phasor=complex(th_ph),
        excitation_heave=complex(f3), excitation_pitch=complex(m5),
        buoyancy_excitation_heave=complex(f3_b),
        coeffs=coeffs, damping_heave=b33, damping_pitch=b55,
        reference=reference, floe_length=lat.length,
    )


def heave_rao_analytic(wave: RegularWave, coeffs: FloeCoefficients,
                       buoyancy_excitation: complex,
                       inertia_excitation: complex,
                       damping_heave: float) -> float:
    """Single-degree-of-freedom heave RAO from the supplied phasors (no pitch)."""
    w = wave.omega
    denom = coeffs.k33 - (coeffs.mass + coeffs.a33) * w ** 2 - 1j * w * damping_heave
    return abs((buoyancy_excitation + inertia_excitation) / denom) / wave.amplitude


def rao_sweep(lat: FloeLattice, z_eq: float, periods: Sequence[float],
              wave_kwargs: dict | None = None, **kwargs) -> list[FrequencyResponse]:
    """Frequency sweep of the transfer function (periods in seconds)."""
    wave_kwargs = dict(wave_kwargs or {})
    return [frequency_response(RegularWave(period=T, **wave_kwargs), lat, z_eq, **kwargs)
            for T in periods]


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------

@dataclass
class ResponseTrace:
    """Time history of the two-degree-of-freedom floe plus its energy audit."""

    time: np.ndarray
    heave: np.ndarray
    pitch: np.ndarray
    heave_rate: np.ndarray
    pitch_rate: np.ndarray
    energy: dict

    def to_dict(self) -> dict:
        return {
            "n_samples": int(self.time.size),
            "duration": float(self.time[-1] - self.time[0]),
            "heave_max": float(np.max(np.abs(self.heave))),
            "pitch_max": float(np.max(np.abs(self.pitch))),
            "energy": self.energy,
        }


class FloeResponseModel:
    """Time-domain two-degree-of-freedom floe under a regular wave.

    State ``(z, theta, zdot, thetadot)``; the acceleration solves the implicit
    2x2 system that carries the added mass, then a classical RK4 step is taken.
    Set ``wave=None`` (or ``height=0``) for a free-decay / energy test.
    """

    def __init__(self, lattice: FloeLattice, wave: RegularWave | None = None,
                 z_eq: float | None = None, added_mass_coeff: float = 0.5,
                 inertia_coeff: float | None = None,
                 damping_ratio_heave: float = 0.10,
                 damping_ratio_pitch: float = 0.10) -> None:
        self.lat = lattice
        self.wave = wave if (wave is not None and wave.height > 0.0) else None
        self.z_eq = lattice.equilibrium_height() if z_eq is None else float(z_eq)
        self.ca = added_mass_coeff
        self.cm = 1.0 + added_mass_coeff if inertia_coeff is None else inertia_coeff
        self.coeffs = discrete_coefficients(lattice, self.z_eq, added_mass_coeff)
        self.b33, self.b55 = damping_coefficients(
            self.coeffs, damping_ratio_heave, damping_ratio_pitch)
        bottom = lattice.z - 0.5 * lattice.dy + self.z_eq
        sub_area = cell_submerged_area(bottom, lattice.dx, lattice.dy)
        w_i = lattice.rho_water * sub_area * self.ca
        self.a33 = float(np.sum(w_i))
        self.a35 = float(np.sum(w_i * lattice.x))
        self.a55 = float(np.sum(w_i * lattice.x ** 2))

    # -- force assembly ------------------------------------------------
    def forces(self, z: float, theta: float, t: float) -> dict:
        """Per-column buoyancy (local ``eta``) and explicit wave load."""
        lat = self.lat
        c, s = math.cos(theta), math.sin(theta)
        xw = lat.x * c - lat.z * s
        zw = z + lat.x * s + lat.z * c
        bottom = zw - 0.5 * lat.dy

        if self.wave is None:
            eta = np.zeros_like(xw)
            f_wave = np.zeros_like(xw)
        else:
            eta = self.wave.surface_elevation(xw, t)
            kin = self.wave.kinematics(xw, zw, t)
            sub = cell_submerged_area(bottom - eta, lat.dx, lat.dy)
            f_wave = lat.rho_water * sub * self.cm * kin["dw_dt"]

        area_sub = cell_submerged_area(bottom - eta, lat.dx, lat.dy)
        f_buoy = lat.rho_water * lat.gravity * area_sub
        f_total = f_buoy + f_wave
        return {
            "xw": xw, "zw": zw, "bottom": bottom, "eta": eta,
            "area_sub": area_sub, "f_buoy": f_buoy, "f_wave": f_wave,
            "f_total": f_total,
            "force_z": float(np.sum(f_total)),
            "moment_y": float(np.sum(xw * f_total)),
        }

    def accelerations(self, z: float, theta: float, zdot: float,
                      thetadot: float, t: float) -> tuple[float, float, dict]:
        f = self.forces(z, theta, t)
        fz = f["force_z"] - self.lat.mass * self.lat.gravity - self.b33 * zdot
        my = f["moment_y"] - self.b55 * thetadot
        m11 = self.lat.mass + self.a33
        m12 = self.a35
        m22 = self.lat.inertia + self.a55
        det = m11 * m22 - m12 * m12
        zdd = (m22 * fz - m12 * my) / det
        tdd = (m11 * my - m12 * fz) / det
        return zdd, tdd, f

    # -- energy --------------------------------------------------------
    def kinetic_energy(self, zdot: float, thetadot: float) -> float:
        return 0.5 * ((self.lat.mass + self.a33) * zdot ** 2
                      + 2.0 * self.a35 * zdot * thetadot
                      + (self.lat.inertia + self.a55) * thetadot ** 2)

    def potential_energy(self, z: float, theta: float) -> float:
        """Buoyancy (mean level) + gravity potential -- closed form."""
        lat = self.lat
        c, s = math.cos(theta), math.sin(theta)
        zw = z + lat.x * s + lat.z * c
        bottom = zw - 0.5 * lat.dy
        buoy = lat.rho_water * lat.gravity * float(
            np.sum(cell_potential(bottom, lat.dx, lat.dy)))
        return buoy + lat.mass * lat.gravity * z

    def total_energy(self, z, theta, zdot, thetadot) -> float:
        return self.kinetic_energy(zdot, thetadot) + self.potential_energy(z, theta)

    # -- integration ---------------------------------------------------
    def run(self, dt: float, n_steps: int, record_every: int = 1,
            z0: float = 0.0, theta0: float = 0.0, zdot0: float = 0.0,
            thetadot0: float = 0.0) -> ResponseTrace:
        """RK4 march with a work/energy ledger.

        The ledger closes ``W_explicit = dKE + dU + W_dissipated`` where the
        explicit generalized force is the Morison load plus the *free-surface
        excess* buoyancy ``rho g [A_sub(b - eta) - A_sub(b)]``; the mean-level
        buoyancy lives in ``U`` and the added mass in ``KE``.
        """
        z = self.z_eq + z0
        th = theta0
        vz = zdot0
        vth = thetadot0
        times = [0.0]
        zs, ths, vzs, vths = [z], [th], [vz], [vth]
        work_explicit = 0.0
        work_damped = 0.0
        t = 0.0

        def rhs(zz, tt, vzz, vtt):
            zdd, tdd, _ = self.accelerations(zz, tt, vzz, vtt, t)
            return vzz, vtt, zdd, tdd

        def explicit_power(zz, tt, vzz, vtt, time):
            f = self.forces(zz, tt, time)
            lat = self.lat
            c, s = math.cos(tt), math.sin(tt)
            xw = lat.x * c - lat.z * s
            vzw = vzz + vtt * xw
            excess = lat.rho_water * lat.gravity * (
                f["area_sub"] - cell_submerged_area(f["bottom"], lat.dx, lat.dy))
            return float(np.sum((f["f_wave"] + excess) * vzw))

        for step in range(1, n_steps + 1):
            k1 = rhs(z, th, vz, vth)
            k2 = rhs(z + 0.5 * dt * k1[0], th + 0.5 * dt * k1[1],
                     vz + 0.5 * dt * k1[2], vth + 0.5 * dt * k1[3])
            k3 = rhs(z + 0.5 * dt * k2[0], th + 0.5 * dt * k2[1],
                     vz + 0.5 * dt * k2[2], vth + 0.5 * dt * k2[3])
            k4 = rhs(z + dt * k3[0], th + dt * k3[1],
                     vz + dt * k3[2], vth + dt * k3[3])
            z_n = z + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            th_n = th + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            vzn = vz + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            vthn = vth + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

            # Simpson on the RK4 stage values keeps the work integral 4th order
            # (a plain trapezoid leaves an O(dt^2) mismatch against the RK4 energy).
            p0 = explicit_power(z, th, vz, vth, t)
            pm = explicit_power(z + 0.5 * dt * k1[0], th + 0.5 * dt * k1[1],
                                vz + 0.5 * dt * k1[2], vth + 0.5 * dt * k1[3],
                                t + 0.5 * dt)
            p1 = explicit_power(z_n, th_n, vzn, vthn, t + dt)
            work_explicit += (p0 + 4.0 * pm + p1) * dt / 6.0
            d0 = self.b33 * vz ** 2 + self.b55 * vth ** 2
            dm = (self.b33 * (vz + 0.5 * dt * k1[2]) ** 2
                  + self.b55 * (vth + 0.5 * dt * k1[3]) ** 2)
            d1 = self.b33 * vzn ** 2 + self.b55 * vthn ** 2
            work_damped += (d0 + 4.0 * dm + d1) * dt / 6.0

            z, th, vz, vth = z_n, th_n, vzn, vthn
            t = step * dt
            if step % record_every == 0:
                times.append(t)
                zs.append(z)
                ths.append(th)
                vzs.append(vz)
                vths.append(vth)

        zs_a, ths_a = np.array(zs), np.array(ths)
        vzs_a, vths_a = np.array(vzs), np.array(vths)
        e_start = self.total_energy(zs_a[0], ths_a[0], vzs_a[0], vths_a[0])
        e_end = self.total_energy(zs_a[-1], ths_a[-1], vzs_a[-1], vths_a[-1])
        residual = work_explicit - (e_end - e_start) - work_damped
        scale = max(abs(work_explicit), abs(e_end - e_start), 1e-30)
        energy = {
            "work_explicit": work_explicit,
            "energy_start": e_start,
            "energy_end": e_end,
            "delta_energy": e_end - e_start,
            "work_dissipated": work_damped,
            "residual": residual,
            "residual_fraction": abs(residual) / scale,
        }
        return ResponseTrace(np.array(times), zs_a, ths_a, vzs_a, vths_a, energy)


# ---------------------------------------------------------------------------
# RAO extraction from a time history
# ---------------------------------------------------------------------------

def rao_from_time_series(t: Sequence[float], y: Sequence[float], omega: float,
                         skip: float = 0.5) -> dict:
    """Least-squares harmonic fit of ``y`` at ``omega`` on the steady window.

    Returns the amplitude, the phase of ``y = a cos(wt) + b sin(wt)`` and the
    residual fraction ``rms(residual)/amplitude`` so a bad fit cannot pass as a
    resonance peak.  ``skip`` drops the first fraction of the record (transient).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0 = int(skip * t.size)
    ts, ys = t[i0:], y[i0:]
    A = np.column_stack([np.cos(omega * ts), np.sin(omega * ts), np.ones_like(ts)])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    fitted = A @ coef
    resid = ys - fitted
    amp = math.hypot(coef[0], coef[1])
    return {
        "amplitude": float(amp),
        "phase": float(math.atan2(-coef[1], coef[0])),
        "mean": float(coef[2]),
        "residual_fraction": float(np.sqrt(np.mean(resid ** 2)) / max(amp, 1e-30)),
    }