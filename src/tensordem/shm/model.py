"""Reduced-order structural model for an ice-going hull girder (SHM twin).

The digital twin in :mod:`tensordem.shm` inverts measured strain signals back to
the ice load.  That inverse problem needs a *forward* structural operator with
explicit, verifiable properties, so this module supplies a modal ROM

    w(x, t) = sum_n q_n(t) phi_n(x),          n = 1 .. N_modes

built from the free-free Euler-Bernoulli hull-girder modes of
:mod:`tensordem.whipping.modes` (dry modes, mass-normalised against the
structural mass per unit length), a hydro-elastic Ritz reduction with the 2-D
BEM section added mass of :mod:`tensordem.whipping.pipeline` (wet modes), and a
per-mode generalised mass re-normalised to unity so that the equation of motion
of every mode reads

    q_n'' + 2 zeta_n omega_n q_n' + omega_n^2 q_n = phi_n(x_F) F(t).

Strain at a station is linear in curvature, ``eps(x) = -y * w''(x)`` (``y``
positive towards the deck, i.e. sagging gives deck compression), hence the
strain-mode matrix

    H[k, n] = -y_k * phi_n''(x_k)          [strain per unit modal coordinate].

Because the modes are mass-normalised, ``H`` has units 1/m and ``q`` carries the
unit of the modal deflection; both stay consistent with ``w = sum q_n phi_n`` and
``sigma = E * y * sum q_n phi_n''``.  The module deliberately contains *no*
inversion logic: it is the forward operator only, so that the inverse chain can
be tested against closed-form responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..whipping.pipeline import ShipCase, hull_added_mass_per_length
from ..whipping import modes as whipping_modes

__all__ = ["SHMConfig", "HullGirderROM", "STRAIN_SENSORS"]


#: Default ice-belt sensor layout: (x/L, y_m).  ``y`` is measured from the
#: neutral axis, positive upwards: five deck gauges and one bottom gauge.
#: Stations are deliberately spread over the midbody where the higher modes have
#: non-negligible curvature, so that ``H`` is well conditioned.
STRAIN_SENSORS = (
    (0.30, 7.0),
    (0.42, 7.0),
    (0.50, 7.0),
    (0.58, 7.0),
    (0.70, 7.0),
    (0.50, -7.0),
)


@dataclass(frozen=True)
class SHMConfig:
    """Configuration of the structural ROM used by the monitoring twin."""

    case: ShipCase = field(default_factory=ShipCase)
    n_modes: int = 4
    load_station_fraction: float = 0.06
    sensors: tuple = STRAIN_SENSORS
    damping_ratio: float = 0.015
    added_mass_relief: float = 0.40   # 3-D relief factor on the 2-D strip mass

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError("n_modes must be positive")
        if not 0.0 <= self.load_station_fraction <= 1.0:
            raise ValueError("load_station_fraction must lie in [0, 1]")
        if not 0.0 < self.added_mass_relief <= 1.0:
            raise ValueError("added_mass_relief must lie in (0, 1]")
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must lie in [0, 1)")
        if len(self.sensors) < 2:
            raise ValueError("at least two strain sensors are required")


class HullGirderROM:
    """Dry/wet modal ROM plus the strain-mode operator ``H``.

    Attributes
    ----------
    omega : (n_modes,) wet circular frequencies [rad/s]
    zeta  : (n_modes,) modal damping ratios
    length, x_load : girder length and load-station abscissa [m]
    modal_mass : (n_modes,) generalised mass, identically 1 by construction
    """

    def __init__(self, config: SHMConfig | None = None):
        self.config = config or SHMConfig()
        case = self.config.case
        self.case = case
        self.length = float(case.length)
        self.n_modes = int(self.config.n_modes)

        dry = whipping_modes.dry_modes(case.youngs_modulus * case.inertia,
                                       case.mass_per_length, case.length,
                                       n_elements=case.n_elements,
                                       n_modes=self.n_modes)
        self.dry = dry
        self.dry_freqs = np.asarray(dry.freqs, dtype=float)

        added = hull_added_mass_per_length(case, relief=self.config.added_mass_relief)
        self.added_mass_info = {k: v for k, v in added.items() if k != "m_a_of_x"}
        self.m_a_of_x = added["m_a_of_x"]

        # Hydro-elastic Ritz problem in the dry modal basis.  The dry modes are
        # mass-normalised, so the reduced mass is I + M_a (M_a = projected added
        # mass) and the reduced stiffness is diag(omega_dry^2).  It is solved via
        # the symmetric transform A = C^-1 K C^-T with I + M_a = C C^T, which keeps
        # the pencil symmetric and therefore yields exact eigenpairs; an eigh of
        # the non-symmetric (I+M_a)^-1 K would silently use one triangle, and the
        # scatter against that route is reported as an audit number.
        self.added_mass_matrix = whipping_modes.modal_added_mass(dry, self.m_a_of_x)
        reduced_mass = np.eye(self.n_modes) + self.added_mass_matrix
        dry_stiff = (2.0 * math.pi * self.dry_freqs) ** 2
        chol = np.linalg.cholesky(reduced_mass)
        a_mat = np.linalg.solve(chol, np.diag(dry_stiff))
        a_mat = np.linalg.solve(chol, a_mat.T)
        a_mat = 0.5 * (a_mat + a_mat.T)
        lam, vec = np.linalg.eigh(a_mat)
        order = np.argsort(lam)
        lam, vec = lam[order], vec[:, order]
        coeff = np.linalg.solve(chol.T, vec)      # a = C^-T y, a^T (I+M_a) a = 1
        self.coefficients = coeff
        self.omega = np.sqrt(np.maximum(lam, 0.0))
        self.freqs = self.omega / (2.0 * math.pi)
        gen_mass = np.einsum("kn,kl,ln->n", coeff, reduced_mass, coeff)
        self.modal_mass = gen_mass
        self.modal_scale = 1.0 / np.sqrt(gen_mass)
        self.wet_reduction = 1.0 - self.freqs / self.dry_freqs
        upstream = whipping_modes.wet_modes(dry, self.m_a_of_x)
        self.upstream_wet_freqs = np.asarray(upstream.freqs, dtype=float)
        self.freq_upstream_scatter = float(np.max(
            np.abs(self.freqs - self.upstream_wet_freqs) / self.upstream_wet_freqs))
        self.zeta = np.full(self.n_modes, float(self.config.damping_ratio))

        self.x_load = float(self.config.load_station_fraction) * self.length
        self.load_shape = np.array([float(self.shape(self.x_load, n)[0])
                                    for n in range(self.n_modes)])

        sensors = np.asarray(self.config.sensors, dtype=float)
        self.sensor_x = sensors[:, 0] * self.length
        self.sensor_y = sensors[:, 1]
        self.H = self.strain_mode_matrix(self.sensor_x, self.sensor_y)

    # ------------------------------------------------------------------ shapes
    def shape(self, x, mode: int, deriv: int = 0) -> np.ndarray:
        """Wet mode ``mode`` (or its ``deriv``-th derivative) at abscissa ``x``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if not 0 <= mode < self.n_modes:
            raise ValueError("mode index out of range")
        out = np.zeros_like(x)
        for k in range(self.n_modes):
            out = out + self.coefficients[k, mode] * self.dry.shape(x, k, deriv)
        return np.asarray(self.modal_scale[mode] * out, dtype=float)

    def strain_mode_matrix(self, x, y) -> np.ndarray:
        """``H[k, n] = -y_k * phi_n''(x_k)``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if x.size != y.size:
            raise ValueError("x and y must have equal length")
        curvature = np.array([self.shape(x, n, deriv=2)
                              for n in range(self.n_modes)])
        return -y[:, None] * curvature.T

    # ------------------------------------------------------------- operators
    def deflection(self, x, q) -> np.ndarray:
        """Modal superposition ``w(x) = sum_n q_n phi_n(x)`` for one time sample."""
        q = np.asarray(q, dtype=float)
        return sum(q[n] * self.shape(x, n) for n in range(self.n_modes))

    def strain_from_modal(self, q) -> np.ndarray:
        """Sensor strain vector from modal coordinates ``q`` (n_modes,)."""
        return self.H @ np.asarray(q, dtype=float)

    def strain_history(self, q) -> np.ndarray:
        """Sensor strain histories from modal coordinates ``q`` (n_modes, n_t)."""
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[0] != self.n_modes:
            raise ValueError("q must have shape (n_modes, n_time)")
        return self.H @ q

    def curvature_weights(self, x=None, y=None) -> np.ndarray:
        """Bending-stress weights ``E y phi_n''(x)`` at a station [Pa per unit q]."""
        x = self.case.stress_station_fraction * self.length if x is None else x
        y = self.case.deck_height if y is None else y
        return np.array([self.case.youngs_modulus * y * float(self.shape(x, n, deriv=2)[0])
                         for n in range(self.n_modes)])

    def modal_force(self, force, load_x=None) -> np.ndarray:
        """Modal force ``p_n = phi_n(x_F) F`` for a scalar or array ``force``."""
        x = self.x_load if load_x is None else load_x
        phi = np.array([float(self.shape(x, n)[0]) for n in range(self.n_modes)])
        f = np.atleast_1d(np.asarray(force, dtype=float))
        return phi[:, None] * f[None, :]

    def load_projection(self, load_x=None) -> np.ndarray:
        """Least-squares map from a modal-force vector back to a point force.

        For a point load at ``x_F`` the modal forces are ``p = phi(x_F) F``; the
        minimum-residual point force is therefore ``F = phi^T p / (phi^T phi)``.
        """
        x = self.x_load if load_x is None else load_x
        phi = np.array([float(self.shape(x, n)[0]) for n in range(self.n_modes)])
        return phi / float(phi @ phi)

    def load_field(self, x, modal_force_vector) -> np.ndarray:
        """Distributed load estimate ``f(x) = sum_n p_n phi_n(x) / sum_n phi_n(x_F)^2``."""
        p = np.asarray(modal_force_vector, dtype=float)
        phi = np.array([self.shape(x, n) for n in range(self.n_modes)])  # (n_modes, M)
        denom = float(np.sum(self.load_shape**2))
        return (p @ phi) / denom

    # ------------------------------------------------------------- diagnostics
    def condition_number(self) -> float:
        """Condition number of the strain-mode matrix (sensor adequacy check)."""
        return float(np.linalg.cond(self.H))

    def as_dict(self) -> dict:
        return {
            "n_modes": self.n_modes,
            "length_m": self.length,
            "load_station_m": self.x_load,
            "dry_frequencies_Hz": [float(f) for f in self.dry_freqs],
            "wet_frequencies_Hz": [float(f) for f in self.freqs],
            "wet_frequency_reduction": [float(r) for r in self.wet_reduction],
            "upstream_wet_frequencies_Hz": [float(v) for v in self.upstream_wet_freqs],
            "upstream_frequency_scatter": self.freq_upstream_scatter,
            "modal_mass_check": [float(v) for v in self.modal_mass],
            "damping_ratio": float(self.config.damping_ratio),
            "added_mass_relief": float(self.config.added_mass_relief),
            "added_mass_info": {k: float(v) if isinstance(v, (int, float)) else v
                                for k, v in self.added_mass_info.items()},
            "modal_mass": [float(m) for m in self.modal_mass],
            "load_shape_at_load_station": [float(v) for v in self.load_shape],
            "sensor_stations_m": [float(v) for v in self.sensor_x],
            "sensor_offsets_m": [float(v) for v in self.sensor_y],
            "strain_mode_matrix_cond": self.condition_number(),
            "curvature_weights_Pa": [float(v) for v in self.curvature_weights()],
        }