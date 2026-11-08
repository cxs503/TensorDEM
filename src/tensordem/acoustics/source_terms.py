r"""Lighthill acoustic analogy and Curle dipole source terms (compact far field).

Physics
-------
Lighthill's reformulation of the compressible Navier-Stokes equations is

.. math::

    \\frac{\\partial^2 \\rho'}{\\partial t^2} - c_0^2 \\nabla^2 \\rho'
      = \\frac{\\partial^2 T_{ij}}{\\partial x_i \\partial x_j},
    \\qquad
    T_{ij} = \\rho u_i u_j + \\bigl(p' - c_0^2 \\rho'\\bigr)\\delta_{ij}
             - \\tau_{ij},

with :math:`T_{ij}` the Lighthill stress tensor (momentum flux, the
non-isentropic pressure term and the viscous stress).  For a **compact** source
(source size small against the acoustic wavelength) the free-space solution in
the far field is

.. math::

    p_a(\\mathbf{x}, t) = \\frac{n_i n_j}{4\\pi c_0^2 r}\\,
        \\frac{\\partial^2}{\\partial t^2}\\int_V T_{ij}\\,\\mathrm{d}V
        \\Big|_{t - r/c_0},
    \\qquad n_i = x_i / r,

i.e. the quadrupole radiates the *second* time derivative of the volume integral
of :math:`T_{ij}`.

Curle's extension for a body with a rigid surface :math:`S` adds the dipole
(surface-load) term

.. math::

    p_a(\\mathbf{x}, t) = \\frac{n_i}{4\\pi c_0 r}\\,
        \\frac{\\partial F_i}{\\partial t}\\Big|_{t-r/c_0},
    \\qquad F_i = \\int_S p\\, n_i \\,\\mathrm{d}S,

the force the surface exerts on the fluid.  Its directivity is the classical
dipole figure-eight :math:`|n_i \\hat F_i| = |\\cos\\theta|`, with a null in the
plane normal to the force.

Convention used everywhere in this package: complex spectra are **peak**
amplitudes, so that ``A cos(omega t) <-> A`` and time averages carry a factor
``1/2`` (``<p^2> = |p~|^2 / 2``).  For a one-sided FFT of a real signal whose
bins are scaled to peak amplitude the mean square is ``sum_k |P_k|^2 / 2``.

Analytic power references
-------------------------
Integrating :math:`\\langle p^2\\rangle/(\\rho_0 c_0)` over a sphere of radius
``r`` and using the exact angular integrals

.. math::

    \\int n_i n_j n_k n_l \\,\\mathrm{d}\\Omega =
        \\frac{4\\pi}{15}(\\delta_{ij}\\delta_{kl}+\\delta_{ik}\\delta_{jl}
                         +\\delta_{il}\\delta_{jk}),
    \\qquad \\int n_i n_j \\,\\mathrm{d}\\Omega = \\frac{4\\pi}{3}\\delta_{ij},

gives, for a harmonic quadrupole of peak strength ``Q_ij`` at ``omega``,

.. math::

    P_\\text{quad} = \\frac{\\omega^4}{32\\pi^2 \\rho_0 c_0^5}
        Q_{ij} Q_{kl} \\int n_i n_j n_k n_l \\,\\mathrm{d}\\Omega
      \\;\\Longrightarrow\\;
      \\frac{\\omega^4 Q^2}{40\\pi\\rho_0 c_0^5}\\;(Q_{xx}),
      \\quad
      \\frac{\\omega^4 Q^2}{120\\pi\\rho_0 c_0^5}\\;(Q_{xy}),

and for a harmonic dipole of peak force ``F`` at ``omega``

.. math::

    P_\\text{dip} = \\frac{\\omega^2 F^2}{24 \\pi \\rho_0 c_0^3}.

Independent exact reference (compact dipole = monopole pair)
-----------------------------------------------------------
The compact **monopole** far field of a volume source with peak volume velocity
``Q`` is textbook exact in the compact limit,

.. math::

    p_a = \frac{\rho_0 \omega Q}{4 \pi r},

with ``Q`` the **peak volume velocity** [m^3/s], and two such monopoles of
opposite sign separated by ``d`` along the axis give

.. math::

    p_a = \frac{\rho_0 \omega Q}{4\pi}
        \Bigl(\frac{e^{i k r_1}}{r_1} - \frac{e^{i k r_2}}{r_2}\Bigr)
        \;\to{k d \ll 1}\;
        -\frac{i \omega\, n_i F_i}{4 \pi c_0 r} e^{i k r},
    \qquad F = \rho_0 \omega Q d,

which is exactly Curle's dipole term with an independently derived strength
``F``.  Feeding the pair through :func:`dipole_pair_far_field` therefore binds
the directivity (``cos theta``), the ``1/r`` law, the ``\omega/(4 pi c_0)``
constant and the radiated power ``omega^2 F^2/(24 pi rho_0 c_0^3)`` of the
implementation to a closed-form Green's function that never touches the dipole
code path.

Scope note: only the **loading (dipole) term** is implemented.  For a *moving*
boundary the Ffowcs Williams-Hawkings thickness term is a separate contribution
of the same order and is explicitly out of scope (see the API ``limitations``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "Medium",
    "lighthill_stress_tensor",
    "volume_integral_stress",
    "quadrupole_far_field_spectrum",
    "dipole_far_field_spectrum",
    "quadrupole_power_analytic",
    "dipole_power_analytic",
    "quadrupole_directivity_theory",
    "monopole_far_field",
    "dipole_pair_force",
    "dipole_pair_far_field",
    "dipole_pair_power",
    "sphere_quadrature",
    "acoustic_power_from_spectrum",
    "pwsl_db",
    "spl_db",
    "fit_power_law",
]


# --------------------------------------------------------------------------- #
# medium
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Medium:
    """Acoustic medium.  Defaults: sea water, underwater reference 1 uPa."""

    c0: float = 1500.0
    rho0: float = 1025.0
    p_ref: float = 1.0e-6
    p_power_ref: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in ("c0", "rho0", "p_ref", "p_power_ref"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"medium.{name} must be a positive finite number")
        if self.c0 < 100.0:
            raise ValueError("medium.c0 below 100 m/s is not an acoustic medium")


def spl_db(pressure_rms: Any, medium: Medium) -> np.ndarray:
    """Sound pressure level [dB re ``medium.p_ref``] of an RMS pressure."""
    p = np.asarray(pressure_rms, dtype=float)
    floor = 1.0e-30
    return 20.0 * np.log10(np.maximum(np.abs(p), floor) / medium.p_ref)


def pwsl_db(power_w: Any, medium: Medium) -> np.ndarray:
    """Sound power level [dB re ``medium.p_power_ref``] of an acoustic power."""
    p = np.asarray(power_w, dtype=float)
    floor = 1.0e-30
    return 10.0 * np.log10(np.maximum(np.abs(p), floor) / medium.p_power_ref)


# --------------------------------------------------------------------------- #
# Lighthill stress tensor
# --------------------------------------------------------------------------- #
def lighthill_stress_tensor(
    velocity: Any,
    *,
    rho: Any = 1025.0,
    pressure_perturbation: Any = 0.0,
    density_perturbation: Any = 0.0,
    c0: float = 1500.0,
    viscous_stress: Any = None,
) -> np.ndarray:
    """Lighthill stress tensor ``T_ij = rho u_i u_j + (p' - c0^2 rho')d_ij - tau_ij``.

    Parameters
    ----------
    velocity
        ``(..., 3)`` velocity field [m/s].
    rho
        Density (scalar or ``(...,)``) [kg/m^3]; the *total* density multiplying
        ``u_i u_j`` (Lighthill's original form).
    pressure_perturbation, density_perturbation
        ``p'`` [Pa] and ``rho'`` [kg/m^3], scalars or ``(...,)``.  Their
        combination is the non-isentropic (entropy / cavitation) term.
    c0
        Reference sound speed [m/s].
    viscous_stress
        Optional ``(..., 3, 3)`` viscous stress ``tau_ij``, subtracted.

    Returns
    -------
    ndarray ``(..., 3, 3)``
    """
    u = np.asarray(velocity, dtype=float)
    if u.shape[-1] != 3:
        raise ValueError("velocity must have a trailing axis of length 3")
    rho_a = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(rho_a)) or np.any(rho_a <= 0.0):
        raise ValueError("rho must be positive and finite")
    mom = rho_a[..., None, None] * u[..., :, None] * u[..., None, :]

    p = np.asarray(pressure_perturbation, dtype=float)
    drho = np.asarray(density_perturbation, dtype=float)
    iso = p - float(c0) ** 2 * drho
    eye = np.eye(3)
    tensor = mom + iso[..., None, None] * eye
    if viscous_stress is not None:
        tau = np.asarray(viscous_stress, dtype=float)
        if tau.shape != tensor.shape:
            raise ValueError("viscous_stress must have the same shape as the tensor")
        tensor = tensor - tau
    return tensor


def volume_integral_stress(tensor: Any, volumes: Any) -> np.ndarray:
    """Volume integral ``Q_ij = int_V T_ij dV`` over control volumes.

    ``tensor`` has shape ``(n_cells, 3, 3)`` and ``volumes`` shape ``(n_cells,)``;
    the sum runs over the leading cell axis (stack the frequency axis in front of
    the result for a spectral volume integral).
    """
    T = np.asarray(tensor, dtype=complex if np.iscomplexobj(tensor) else float)
    V = np.asarray(volumes, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (3, 3):
        raise ValueError("tensor must be (n_cells, 3, 3)")
    if T.shape[0] != V.size:
        raise ValueError("volumes must have one entry per cell")
    if np.any(V < 0.0) or not np.all(np.isfinite(V)):
        raise ValueError("volumes must be finite and non-negative")
    return np.sum(T * V[:, None, None], axis=0)


# --------------------------------------------------------------------------- #
# compact-source far field (frequency domain, peak amplitudes)
# --------------------------------------------------------------------------- #
def _blockwise(compute: Any, n_freq: int, n_dir: int, chunk: int = 64) -> np.ndarray:
    """Evaluate a ``(n_freq, m)`` direction block in memory-bounded chunks."""
    out = np.empty((n_freq, n_dir), dtype=complex)
    for lo in range(0, n_dir, chunk):
        hi = min(lo + chunk, n_dir)
        out[:, lo:hi] = compute(lo, hi)
    return out


def _blockwise(compute: Any, n_freq: int, n_dir: int, chunk: int = 8) -> np.ndarray:
    """Evaluate a ``(n_freq, m)`` direction block in memory-bounded chunks."""
    out = np.empty((n_freq, n_dir), dtype=complex)
    for lo in range(0, n_dir, chunk):
        hi = min(lo + chunk, n_dir)
        out[:, lo:hi] = compute(lo, hi)
    return out


def _retarded_phase(omega: np.ndarray, distance: float, positions: np.ndarray,
                    normals: np.ndarray, c0: float) -> np.ndarray:
    """Far-field phase ``exp(-i k (r - n.x_f))`` of every face, ``(n_freq, m, n_face)``.

    ``r`` is the common observer distance (the ``1/r`` prefactor) and ``n.x_f`` is
    the projection of the face centroid on the observation direction: the term
    that turns a face distribution into an *array* (beam pattern / finite extent)
    instead of a point source.  Dropping it would make every equal-radius surface
    look like an exact point source.
    """
    proj = normals @ positions.T                        # (m, n_face)
    # numpy's rfft/irfft pair uses e^{+i omega t}, so a *delayed* wave is the
    # factor e^{-i k (r - n.x_f)} (the conjugate of the e^{-i omega t} physics form)
    return np.exp(-1j * omega[:, None, None] * (distance - proj[None, :, :]) / c0)


def quadrupole_far_field_spectrum(
    q_spectrum: Any,
    freqs: Any,
    normals: Any,
    *,
    distance: float,
    medium: Medium,
    positions: Any = None,
) -> np.ndarray:
    """Far-field pressure spectrum of a compact quadrupole distribution.

    ``p~(f) = -(2 pi f)^2 / (4 pi c0^2 r) * n_i n_j Q_ij(f) * exp(-i k (r - n.x_f))``

    Parameters
    ----------
    q_spectrum
        ``(n_freq, 3, 3)`` (single lumped quadrupole) or ``(n_freq, n_face, 3, 3)``
        (per-face volume integrals, retarded individually).
    freqs
        ``(n_freq,)`` non-negative frequencies [Hz].
    normals
        ``(n_dir, 3)`` unit observation directions (from source to observer).
    distance
        Observer distance ``r`` [m] (``r >> L^2/lambda`` far-field assumed).
    positions
        ``(n_face, 3)`` face centroids; required when a per-face axis is given.

    Returns
    -------
    ``(n_freq, n_dir)`` complex pressure spectrum [Pa].
    """
    Q = np.asarray(q_spectrum, dtype=complex)
    f = np.asarray(freqs, dtype=float)
    n = np.asarray(normals, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("normals must be (n_dir, 3)")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if Q.ndim == 3:
        Q = Q[:, None, :, :]
    if Q.ndim != 4 or Q.shape[2:] != (3, 3):
        raise ValueError("q_spectrum must be (n_freq, 3, 3) or (n_freq, n_face, 3, 3)")
    if Q.shape[0] != f.size:
        raise ValueError("freqs must have one entry per spectrum row")
    omega = 2.0 * math.pi * f
    prefac = -(omega ** 2) / (4.0 * math.pi * medium.c0 ** 2 * distance)
    n_face = Q.shape[1]
    if n_face == 1:
        pos = (np.zeros((1, 3)) if positions is None
               else np.asarray(positions, dtype=float).reshape(1, 3))
        # a lumped source still propagates: it carries the phase e^{+i k(distance - n.x)}
    else:
        if positions is None:
            raise ValueError("per-face quadrupoles need positions")
        pos = np.asarray(positions, dtype=float)
        if pos.shape != (n_face, 3):
            raise ValueError("positions must be (n_face, 3)")

    def block(lo: int, hi: int) -> np.ndarray:
        nd = n[lo:hi]                                        # (m, 3)
        proj = np.einsum("ai,aj,fnij->fan", nd, nd, Q)       # (n_freq, m, n_face)
        phase = _retarded_phase(omega, distance, pos, nd, medium.c0)
        return prefac[:, None] * np.einsum("fan,fan->fa", proj, phase)

    return _blockwise(block, f.size, n.shape[0])


def dipole_far_field_spectrum(
    f_spectrum: Any,
    freqs: Any,
    normals: Any,
    *,
    distance: float,
    medium: Medium,
    positions: Any = None,
) -> np.ndarray:
    """Far-field pressure spectrum of compact Curle dipoles.

    ``p~(f) = +i (2 pi f) / (4 pi c0 r) * n_i F_i(f) * exp(-i k (r - n.x_f))``

    ``f_spectrum`` is ``(n_freq, 3)`` for one lumped force or ``(n_freq, n_face,
    3)`` for per-face loads (then ``positions`` is required).
    """
    F = np.asarray(f_spectrum, dtype=complex)
    f = np.asarray(freqs, dtype=float)
    n = np.asarray(normals, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("normals must be (n_dir, 3)")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if F.ndim == 2:
        F = F[:, None, :]
    if F.ndim != 3 or F.shape[2] != 3:
        raise ValueError("f_spectrum must be (n_freq, 3) or (n_freq, n_face, 3)")
    if F.shape[0] != f.size:
        raise ValueError("freqs must have one entry per spectrum row")
    omega = 2.0 * math.pi * f
    # numpy e^{+i omega t} convention: d/dt -> +i omega
    prefac = 1j * omega / (4.0 * math.pi * medium.c0 * distance)
    n_face = F.shape[1]
    if n_face == 1:
        pos = (np.zeros((1, 3)) if positions is None
               else np.asarray(positions, dtype=float).reshape(1, 3))
    else:
        if positions is None:
            raise ValueError("per-face dipoles need positions")
        pos = np.asarray(positions, dtype=float)
        if pos.shape != (n_face, 3):
            raise ValueError("positions must be (n_face, 3)")

    def block(lo: int, hi: int) -> np.ndarray:
        nd = n[lo:hi]
        proj = np.einsum("ai,fni->fan", nd, F)                # (n_freq, m, n_face)
        phase = _retarded_phase(omega, distance, pos, nd, medium.c0)
        return prefac[:, None] * np.einsum("fan,fan->fa", proj, phase)

    return _blockwise(block, f.size, n.shape[0])


# --------------------------------------------------------------------------- #
# analytic power references
# --------------------------------------------------------------------------- #
def quadrupole_power_analytic(q_peak: Any, omega: float, medium: Medium) -> float:
    """Radiated power of a harmonic compact quadrupole of peak strength ``Q_ij``.

    ``P = omega^4/(32 pi^2 rho0 c0^5) * Q_ij Q_kl
    * 4 pi (d_ij d_kl + d_ik d_jl + d_il d_jk)/15``
    """
    Q = np.asarray(q_peak, dtype=complex)
    if Q.shape != (3, 3):
        raise ValueError("q_peak must be a (3, 3) tensor")
    if not omega > 0.0:
        raise ValueError("omega must be positive")
    angular = 4.0 * math.pi / 15.0
    eye = np.eye(3)
    total = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    coef = (eye[i, j] * eye[k, l] + eye[i, k] * eye[j, l]
                            + eye[i, l] * eye[j, k])
                    if coef == 0.0:
                        continue
                    total += float(np.real(np.conj(Q[i, j]) * Q[k, l])) * coef
    return omega ** 4 * total * angular / (32.0 * math.pi ** 2 * medium.rho0
                                           * medium.c0 ** 5)


def dipole_power_analytic(f_peak: Any, omega: float, medium: Medium) -> float:
    """Radiated power of a harmonic compact dipole of peak force amplitude ``F``.

    ``P = omega^2 |F|^2 / (24 pi rho0 c0^3)``
    """
    F = np.asarray(f_peak, dtype=complex).ravel()
    if F.size != 3:
        raise ValueError("f_peak must be a 3-vector")
    if not omega > 0.0:
        raise ValueError("omega must be positive")
    return omega ** 2 * float(np.sum(np.abs(F) ** 2)) / (24.0 * math.pi
                                                        * medium.rho0 * medium.c0 ** 3)


def quadrupole_directivity_theory(direction: Any, q_peak: Any) -> np.ndarray:
    """``|n_i n_j Q_ij|`` normalised to its maximum over the given directions."""
    n = np.asarray(direction, dtype=float)
    Q = np.asarray(q_peak, dtype=complex)
    amp = np.abs(np.einsum("...i,...j,ij->...", n, n, Q))
    peak = float(np.max(amp)) if amp.size else 0.0
    return amp / peak if peak > 0.0 else amp


# --------------------------------------------------------------------------- #
# oscillating sphere: independent exact reference
# --------------------------------------------------------------------------- #
def monopole_far_field(volume_velocity_peak: float, omega: float, distance: float,
                       medium: Medium) -> float:
    """Compact monopole far field ``rho_0 omega Q / (4 pi r)`` [Pa], peak amplitude.

    ``volume_velocity_peak`` is the peak volume velocity ``Q`` [m^3/s], i.e. the
    amplitude of ``dV/dt``; the second time derivative brings the single ``omega``.
    """
    if volume_velocity_peak <= 0.0 or omega <= 0.0:
        raise ValueError("volume_velocity_peak and omega must be positive")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    return (medium.rho0 * omega * volume_velocity_peak
            / (4.0 * math.pi * distance))


def dipole_pair_force(volume_velocity_peak: float, separation: float, omega: float,
                      medium: Medium) -> float:
    """Equivalent force ``F = rho_0 omega Q d`` [N] of a two-monopole dipole."""
    if separation <= 0.0:
        raise ValueError("separation must be positive")
    return medium.rho0 * omega * volume_velocity_peak * separation


def dipole_pair_far_field(volume_velocity_peak: float, separation: float, omega: float,
                          directions: Any, distance: float, medium: Medium) -> np.ndarray:
    """Exact compact two-monopole dipole field, complex peak amplitude [Pa].

    ``directions`` are unit vectors from the source to the observer; the pair sits
    at ``+/- d/2`` along ``z``.  The ``e^{+i k r}`` (``e^{-i omega t}``) convention
    matches :func:`dipole_far_field_spectrum`, so the two can be compared complex
    to complex.
    """
    n = np.asarray(directions, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("directions must be (n_dir, 3)")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if separation <= 0.0:
        raise ValueError("separation must be positive")
    k = omega / medium.c0
    obs = distance * n / np.linalg.norm(n, axis=1)[:, None]
    half = 0.5 * separation * np.array([0.0, 0.0, 1.0])
    r1 = np.linalg.norm(obs - half, axis=1)
    r2 = np.linalg.norm(obs + half, axis=1)
    pref = medium.rho0 * omega * volume_velocity_peak / (4.0 * math.pi)
    return pref * (np.exp(-1j * k * r1) / r1 - np.exp(-1j * k * r2) / r2)


def dipole_pair_power(volume_velocity_peak: float, separation: float, omega: float,
                      medium: Medium) -> float:
    """Analytic power of the two-monopole dipole, ``omega^2 F^2/(24 pi rho_0 c_0^3)``."""
    f_peak = dipole_pair_force(volume_velocity_peak, separation, omega, medium)
    return dipole_power_analytic(np.array([0.0, 0.0, f_peak]), omega, medium)


# --------------------------------------------------------------------------- #
# angular quadrature and power integration
# --------------------------------------------------------------------------- #
def sphere_quadrature(n_theta: int = 24, n_phi: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre in ``cos(theta)`` x uniform ``phi`` quadrature of the sphere.

    Returns ``(directions (n_dir, 3), solid_angles)``.  The Gauss rule integrates
    the degree-4 polynomial directivities of quadrupoles exactly; the trapezoidal
    ``phi`` rule is exact for the trigonometric polynomials of order ``< n_phi``.
    """
    if n_theta < 2 or n_phi < 4:
        raise ValueError("need n_theta >= 2 and n_phi >= 4")
    x, w = np.polynomial.legendre.leggauss(n_theta)  # x = cos(theta), sum(w) = 2
    phi = 2.0 * math.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * math.pi / n_phi
    dirs = []
    weights = []
    for xi, wi in zip(x, w):
        st = math.sqrt(max(0.0, 1.0 - xi * xi))
        for pj in phi:
            dirs.append([st * math.cos(pj), st * math.sin(pj), xi])
            weights.append(wi * dphi)
    return np.asarray(dirs, dtype=float), np.asarray(weights, dtype=float)


def acoustic_power_from_spectrum(p_spectrum: Any, radius: float, solid_angles: Any,
                                 medium: Medium) -> float:
    """Radiated power from a far-field spectrum on a sphere.

    One-sided peak-amplitude convention: ``<p^2> = sum_k |P_k|^2 / 2``, so

    ``P = r^2 / (rho0 c0) * sum_dir dOmega * sum_freq |P|^2 / 2``.
    """
    P = np.asarray(p_spectrum, dtype=complex)
    dO = np.asarray(solid_angles, dtype=float)
    if P.ndim != 2:
        raise ValueError("p_spectrum must be (n_freq, n_dir)")
    if dO.size != P.shape[1]:
        raise ValueError("solid_angles must match the direction axis")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    mean_square = 0.5 * np.sum(np.abs(P) ** 2, axis=0)  # per direction
    return float(radius ** 2 / (medium.rho0 * medium.c0)
                 * np.sum(mean_square * dO))


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares fit ``y = C x^n``; returns ``(n, C)``."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("x and y need at least two matching samples")
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        raise ValueError("power-law fit needs strictly positive data")
    n, logc = np.polyfit(np.log(xa), np.log(ya), 1)
    return float(n), float(math.exp(logc))