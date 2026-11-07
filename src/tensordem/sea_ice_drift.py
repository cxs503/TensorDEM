"""Sea-ice drift dynamics closure: Coriolis + added-mass tensor + relative-velocity drag.

Physics scope
-------------
Momentum balance of a sea-ice slab (thickness ``h``, density ``rho_i``) per unit
horizontal area, in a frame rotating with the Earth:

.. math::

    \\mathbf{M}\\,\\dot{\\mathbf u} = \\boldsymbol\\tau_a + \\boldsymbol\\tau_w
        - m f\\,\\mathbf{k}\\times\\mathbf{u} - m\\,g\\nabla\\eta

    m = \\rho_i h, \\qquad \\mathbf{M} = m\\,\\mathbf{I} + \\mathbf{M}_a

* ``f = 2\\Omega\\sin\\varphi`` — Coriolis parameter (``Omega = 7.2921e-5`` rad/s).
* ``\\mathbf{M}_a`` — added-mass tensor per unit area [kg/m^2].  It is kept as a
  full 2x2 symmetric tensor so that anisotropic floe shapes / rotated principal
  axes are representable; only ``\\mathbf{M}`` enters the inertia term.
* ``\\boldsymbol\\tau_a = \\rho_a C_a |\\mathbf{U}_a-\\mathbf{u}|(\\mathbf{U}_a-\\mathbf{u})``
  — wind drag, quadratic in the *relative* velocity (air relative to ice).
* ``\\boldsymbol\\tau_w = \\rho_w C_w |\\mathbf{u}_w-\\mathbf{u}|(\\mathbf{u}_w-\\mathbf{u})``
  — water drag, quadratic in the *relative* velocity (ocean relative to ice).

Coriolis convention (documented modelling choice): the Coriolis force is applied
to the ice mass ``m`` only, as in the iceberg-drift and sea-ice literature; the
optional ``coriolis_mass="total"`` switch adds ``\\mathbf{M}_a`` to the Coriolis
term (rotating-frame correction for the entrained fluid) so the sensitivity can
be quantified instead of hidden.

Vector convention: x east, y north, ``k`` up, so
``k x u = (-u_y, u_x)``; for ``f > 0`` (Northern Hemisphere) the free drift is
deflected to the *right* of the wind.  Deflection angle is reported positive to
the right.

Verified closed forms implemented here (each one is anchored by a test):
  1. Added mass only changes the transient: constant-force acceleration is
     ``M^{-1} F``; spin-up is ``u(t) = u_inf (1 - exp(-t/tau))`` with
     ``tau = (m + m_a)/(m C_D)``.
  2. Linear-drag slab (Ekman-type friction): exact steady state
     ``theta = arctan(f / C_D)``, ``|u| = tau_a / (m sqrt(f^2 + C_D^2))``.
  3. Quadratic-water-drag free drift (stress-specified wind): exact steady
     state from ``tan(theta) sqrt(cos theta) = m f / sqrt(rho_w C_w tau_a)``.
  4. Ekman spiral in a constant-viscosity ocean: ``D_E = sqrt(2K/f)``,
     surface velocity 45 deg to the right of the stress, total Ekman transport
     ``tau/(rho_w f)`` at 90 deg right, depth of frictional influence
     ``pi sqrt(2K/f)``, and the closure stress ``tau = rho_w sqrt(fK) w_s
     e^{i pi/4}`` used as an Ekman drag law.
  5. Negative control: ``f = 0`` gives zero deflection.

All quantities SI.  Public API is stable; see ``docs/sea_ice_drift_dynamics_validation.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

OMEGA_EARTH = 7.2921e-5  # [rad/s] Earth rotation rate (sidereal)

ArrayLike2 = Sequence[float]


# ---------------------------------------------------------------------------
# Geometry / rotation helpers
# ---------------------------------------------------------------------------

def coriolis_parameter(latitude_deg: float, omega: float = OMEGA_EARTH) -> float:
    """Coriolis parameter f = 2*Omega*sin(latitude)  [1/s]."""
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude must be in [-90, 90] degrees")
    return 2.0 * omega * math.sin(math.radians(latitude_deg))


def rotate(vec: ArrayLike2, angle_rad: float) -> np.ndarray:
    """Rotate a 2-vector counter-clockwise (positive angle) [CCW]."""
    v = np.asarray(vec, dtype=float)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def rotate_right(vec: ArrayLike2, angle_rad: float) -> np.ndarray:
    """Rotate a 2-vector clockwise by ``angle_rad`` (used for 'to the right')."""
    return rotate(vec, -angle_rad)


def k_cross(vec: ArrayLike2) -> np.ndarray:
    """k x v for k = +z, v in the horizontal plane: (-v_y, v_x)."""
    v = np.asarray(vec, dtype=float)
    return np.array([-v[1], v[0]])


def deflection_angle(u: ArrayLike2, reference: ArrayLike2) -> float:
    """Deflection of ``u`` from ``reference``, positive *clockwise* (to the right).

    Returns an angle in (-180, 180] degrees.  With f > 0 the free sea-ice drift
    relative to the wind gives a positive value (Nansen's rule).
    """
    u = np.asarray(u, dtype=float)
    r = np.asarray(reference, dtype=float)
    nu = np.linalg.norm(u)
    nr = np.linalg.norm(r)
    if nu == 0.0 or nr == 0.0:
        raise ValueError("both vectors must be non-zero")
    cross = r[0] * u[1] - r[1] * u[0]  # >0 => u is CCW (left) of r
    dot = float(np.dot(u, r))
    return -math.degrees(math.atan2(cross, dot))


# ---------------------------------------------------------------------------
# Added-mass tensor
# ---------------------------------------------------------------------------

@dataclass
class AddedMassTensor:
    """Added-mass tensor per unit horizontal area [kg/m^2].

    ``matrix`` is a symmetric positive semi-definite 2x2 array.  The added-mass
    force on the body is ``-M_a @ du/dt``; it never enters the *steady* force
    balance, only the inertia (and, if requested, the rotating-frame Coriolis).
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (2, 2):
            raise ValueError("added-mass tensor must be 2x2")
        scale = max(1.0, float(np.max(np.abs(m))))
        if not np.allclose(m, m.T, atol=1e-12 * scale):
            raise ValueError("added-mass tensor must be symmetric")
        if float(np.min(np.linalg.eigvalsh(m))) < -1e-12 * scale:
            raise ValueError("added-mass tensor must be positive semi-definite")
        self.matrix = np.array((m + m.T) / 2.0)  # exactly symmetric

    # -- constructors -------------------------------------------------------
    @classmethod
    def isotropic(cls, m_a: float) -> "AddedMassTensor":
        if m_a < 0.0:
            raise ValueError("added mass must be non-negative")
        return cls(np.eye(2) * m_a)

    @classmethod
    def from_principals(
        cls, m_a_11: float, m_a_22: float, angle_deg: float = 0.0
    ) -> "AddedMassTensor":
        """Tensor with principal values (m_a_11, m_a_22) rotated by ``angle_deg`` CCW."""
        if m_a_11 < 0.0 or m_a_22 < 0.0:
            raise ValueError("added mass must be non-negative")
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        rot = np.array([[c, -s], [s, c]])
        return cls(rot @ np.diag([m_a_11, m_a_22]) @ rot.T)

    @classmethod
    def from_coefficient(
        cls,
        rho_water: float,
        draught: float,
        c_m: float = 1.0,
        anisotropy: float = 1.0,
        angle_deg: float = 0.0,
    ) -> "AddedMassTensor":
        """Idealised added mass of a floe drifting in its own plane.

        Added mass per unit area is taken as ``c_m * rho_w * draught`` along
        x (the displaced fluid mass per unit area scaled by the added-mass
        coefficient) and ``anisotropy`` times that along y.  ``c_m = 1``
        corresponds to the potential-flow value of a cylinder/slab whose
        displaced mass per area is ``rho_w * draught``.
        """
        if rho_water <= 0.0 or draught < 0.0:
            raise ValueError("rho_water must be positive and draught non-negative")
        base = c_m * rho_water * draught
        return cls.from_principals(base, anisotropy * base, angle_deg)

    @classmethod
    def none(cls) -> "AddedMassTensor":
        return cls(np.zeros((2, 2)))

    # -- queries ------------------------------------------------------------
    def force(self, accel: ArrayLike2) -> np.ndarray:
        """Added-mass force per unit area: ``-M_a @ accel`` [N/m^2]."""
        return -self.matrix @ np.asarray(accel, dtype=float)

    def total_mass_matrix(self, mass_per_area: float) -> np.ndarray:
        if mass_per_area <= 0.0:
            raise ValueError("mass per area must be positive")
        return np.eye(2) * mass_per_area + self.matrix

    @property
    def isotropic_equivalent(self) -> float:
        """Mean of the principal values (trace/2) [kg/m^2]."""
        return 0.5 * float(np.trace(self.matrix))

    @property
    def anisotropy_ratio(self) -> float:
        ev = np.linalg.eigvalsh(self.matrix)
        return float(ev[-1] / ev[0]) if ev[0] > 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Drag laws
# ---------------------------------------------------------------------------

@dataclass
class QuadraticDrag:
    """Quadratic relative-velocity drag law: ``tau = rho C_D |du| du``."""

    rho: float
    c_d: float
    relative: bool = True

    def __post_init__(self) -> None:
        if self.rho <= 0.0 or self.c_d <= 0.0:
            raise ValueError("rho and C_D must be positive")

    @property
    def coefficient(self) -> float:
        """Bulk coefficient rho*C_D [kg/m^3]."""
        return self.rho * self.c_d

    def stress(self, u_body: ArrayLike2, u_fluid: ArrayLike2) -> np.ndarray:
        ub = np.asarray(u_body, dtype=float)
        uf = np.asarray(u_fluid, dtype=float)
        du = (uf - ub) if self.relative else uf
        mag = float(np.linalg.norm(du))
        return self.coefficient * mag * du

    def stress_magnitude(self, u_body: ArrayLike2, u_fluid: ArrayLike2) -> float:
        return float(np.linalg.norm(self.stress(u_body, u_fluid)))


# ---------------------------------------------------------------------------
# Drift model
# ---------------------------------------------------------------------------

@dataclass
class SeaIceDriftModel:
    """Sea-ice drift momentum balance (Coriolis + added mass + relative drag).

    drag_mode
        ``"quadratic"`` — wind and water drag quadratic in the relative
        velocity (default, physical model).
        ``"linear"`` — water drag replaced by ``-m C_D u_rel`` (Ekman-type
        slab friction with rate ``C_D`` [1/s]); used to anchor
        ``theta = arctan(f/C_D)``.
        ``"ekman"`` — water drag from the constant-viscosity Ekman closure
        ``tau_w = -rho_w sqrt(f K) R(+45deg) u_rel`` (needs ``eddy_viscosity``).
    """

    latitude_deg: float
    ice_thickness: float
    wind: Tuple[float, float] = (10.0, 0.0)
    ocean_current: Tuple[float, float] = (0.0, 0.0)
    ice_density: float = 900.0
    rho_air: float = 1.25
    c_d_air: float = 0.0018
    rho_water: float = 1025.0
    c_d_water: float = 0.0045
    added_mass: Optional[AddedMassTensor] = None
    surface_tilt_accel: Tuple[float, float] = (0.0, 0.0)
    drag_mode: str = "quadratic"
    linear_drag_rate: float = 1.0e-4
    eddy_viscosity: Optional[float] = None
    coriolis_mass: str = "ice"  # "ice" or "total"
    wind_relative: bool = True  # wind drag on |W - u| ; False = prescribed stress
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.ice_thickness <= 0.0 or self.ice_density <= 0.0:
            raise ValueError("ice thickness and density must be positive")
        if self.drag_mode not in ("quadratic", "linear", "ekman"):
            raise ValueError("drag_mode must be 'quadratic', 'linear' or 'ekman'")
        if self.drag_mode == "ekman" and (
            self.eddy_viscosity is None or self.eddy_viscosity <= 0.0
        ):
            raise ValueError("drag_mode='ekman' requires a positive eddy_viscosity")
        if self.coriolis_mass not in ("ice", "total"):
            raise ValueError("coriolis_mass must be 'ice' or 'total'")
        if self.added_mass is None:
            self.added_mass = AddedMassTensor.none()
        self.wind = (float(self.wind[0]), float(self.wind[1]))
        self.ocean_current = (float(self.ocean_current[0]), float(self.ocean_current[1]))
        self.surface_tilt_accel = (
            float(self.surface_tilt_accel[0]),
            float(self.surface_tilt_accel[1]),
        )

    # -- parameters ---------------------------------------------------------
    @property
    def f(self) -> float:
        return coriolis_parameter(self.latitude_deg)

    @property
    def mass_per_area(self) -> float:
        """Ice mass per unit horizontal area m = rho_i h [kg/m^2]."""
        return self.ice_density * self.ice_thickness

    @property
    def mass_matrix(self) -> np.ndarray:
        return self.added_mass.total_mass_matrix(self.mass_per_area)

    @property
    def wind_drag(self) -> QuadraticDrag:
        return QuadraticDrag(self.rho_air, self.c_d_air, relative=self.wind_relative)

    @property
    def water_drag(self) -> QuadraticDrag:
        return QuadraticDrag(self.rho_water, self.c_d_water)

    @property
    def ekman_drag_coefficient(self) -> float:
        """Ekman closure drag [kg/(m^2 s)]: |tau| = rho_w sqrt(f K) |u_rel|."""
        if self.eddy_viscosity is None:
            raise ValueError("eddy_viscosity not set")
        return self.rho_water * math.sqrt(abs(self.f) * self.eddy_viscosity)

    # -- forces -------------------------------------------------------------
    def wind_stress(self, u_ice: ArrayLike2) -> np.ndarray:
        """Wind stress on the ice [N/m^2].

        ``wind_relative=True`` (physical default) uses the relative wind
        ``W - u_ice``; ``wind_relative=False`` prescribes ``rho_a C_a |W| W``,
        which is the classical free-drift problem whose closed form is
        :func:`free_drift_state` (used as an exact analytic anchor).
        """
        if not self.wind_relative:
            u_ice = np.zeros(2)
        return self.wind_drag.stress(u_ice, self.wind)

    def water_stress(self, u_ice: ArrayLike2) -> np.ndarray:
        u = np.asarray(u_ice, dtype=float)
        u_ocean = np.asarray(self.ocean_current, dtype=float)
        if self.drag_mode == "quadratic":
            return self.water_drag.stress(u, u_ocean)
        if self.drag_mode == "linear":
            return -self.mass_per_area * self.linear_drag_rate * (u - u_ocean)
        # Ekman closure: tau_w = -rho_w sqrt(fK) R(+45) (u - u_ocean)
        return -self.ekman_drag_coefficient * rotate(u - u_ocean, math.pi / 4.0)

    def coriolis_force(self, u_ice: ArrayLike2) -> np.ndarray:
        """Coriolis force per unit area: ``-m_eff f k x u`` [N/m^2]."""
        u = np.asarray(u_ice, dtype=float)
        if self.coriolis_mass == "ice":
            m_eff = self.mass_per_area
            return -m_eff * self.f * k_cross(u)
        return -self.f * (self.mass_matrix @ k_cross(u))

    def forces(self, u_ice: ArrayLike2, include_inertia=False) -> dict:
        """All force terms per unit area [N/m^2] plus the net residual."""
        u = np.asarray(u_ice, dtype=float)
        terms = {
            "wind": self.wind_stress(u),
            "water": self.water_stress(u),
            "coriolis": self.coriolis_force(u),
            "tilt": -self.mass_per_area * np.asarray(self.surface_tilt_accel, dtype=float),
        }
        net = sum(terms.values())
        if include_inertia:
            terms["inertia_matrix"] = self.mass_matrix
        return {"terms": terms, "net": net}

    # -- dynamics -----------------------------------------------------------
    def rhs(self, u_ice: ArrayLike2) -> np.ndarray:
        """du/dt = M^{-1} (wind + water + Coriolis + tilt) [m/s^2]."""
        net = self.forces(u_ice)["net"]
        return np.linalg.solve(self.mass_matrix, net)

    def integrate(
        self,
        u0: ArrayLike2,
        t_end: float,
        dt: float,
        method: str = "rk4",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate the drift ODE; returns (times, speeds[n,2], accels[n,2])."""
        if dt <= 0.0 or t_end <= 0.0:
            raise ValueError("dt and t_end must be positive")
        n = int(round(t_end / dt))
        if n < 1:
            raise ValueError("t_end must be at least one dt")
        u = np.asarray(u0, dtype=float).copy()
        times = np.zeros(n + 1)
        us = np.zeros((n + 1, 2))
        accs = np.zeros((n + 1, 2))
        us[0], accs[0] = u, self.rhs(u)
        for i in range(n):
            if method == "rk4":
                k1 = self.rhs(u)
                k2 = self.rhs(u + 0.5 * dt * k1)
                k3 = self.rhs(u + 0.5 * dt * k2)
                k4 = self.rhs(u + dt * k3)
                u = u + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif method == "euler":
                u = u + dt * self.rhs(u)
            else:
                raise ValueError("method must be 'rk4' or 'euler'")
            times[i + 1] = (i + 1) * dt
            us[i + 1], accs[i + 1] = u, self.rhs(u)
        return times, us, accs

    def steady_state(
        self,
        u0: Optional[ArrayLike2] = None,
        tol: float = 1e-13,
        max_iter: int = 200,
    ) -> np.ndarray:
        """Steady drift velocity from a damped Newton iteration on the residual.

        Solves ``F(u) = tau_a + tau_w(u) + tau_coriolis(u) + tau_tilt = 0`` with a
        central-difference Jacobian.  Returns the velocity [m/s].  Raises
        ``RuntimeError`` if the residual does not reach ``tol`` (relative to the
        force scale), so a non-converged answer can never be reported as
        converged.
        """
        u = np.zeros(2) if u0 is None else np.asarray(u0, dtype=float).copy()
        if self.drag_mode == "linear" and self.linear_drag_rate > 0.0:
            # exact linear solve available: M_drag u = tau_a + tilt + drag-rate*u_ocean
            Kk = np.array([[0.0, -1.0], [1.0, 0.0]])  # k x u in matrix form
            A = (
                self.mass_per_area * self.linear_drag_rate * np.eye(2)
                - self._coriolis_matrix()  # F_cor = -_coriolis_matrix u, so +m f Kk
            )
            A = self.mass_per_area * self.linear_drag_rate * np.eye(2) + (
                self.mass_per_area * self.f * Kk
                if self.coriolis_mass == "ice"
                else self.f * (self.mass_matrix @ Kk)
            )
            b = (
                self.wind_stress(np.zeros(2))
                + self.mass_per_area
                * self.linear_drag_rate
                * np.asarray(self.ocean_current, dtype=float)
                + np.asarray(self.forces(np.zeros(2))["terms"]["tilt"])
            )
            return np.linalg.solve(A, b)
        scale = max(
            1e-30,
            float(np.linalg.norm(self.wind_stress(u)))
            + float(np.linalg.norm(self.water_stress(u)))
            + 1e-12,
        )
        for _ in range(max_iter):
            res = self.forces(u)["net"]
            if float(np.linalg.norm(res)) / scale < tol:
                return u
            jac = np.zeros((2, 2))
            du = 1e-7 * max(1.0, float(np.linalg.norm(u)))
            for j in range(2):
                e = np.zeros(2)
                e[j] = du
                # derivative of the (velocity dependent) forces only
                jac[:, j] = (
                    self._velocity_dependent_forces(u + e)
                    - self._velocity_dependent_forces(u - e)
                ) / (2.0 * du)
            try:
                step = np.linalg.solve(jac, -res)
            except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"singular Newton Jacobian: {exc}") from exc
            # damped Newton: halve the step until the residual decreases
            lam = 1.0
            for _ in range(30):
                u_new = u + lam * step
                if float(np.linalg.norm(self.forces(u_new)["net"])) < float(
                    np.linalg.norm(res)
                ):
                    u = u_new
                    break
                lam *= 0.5
            else:  # pragma: no cover - defensive
                u = u + 1e-3 * step
        raise RuntimeError(
            f"steady-state Newton did not converge: residual "
            f"{float(np.linalg.norm(self.forces(u)['net'])) / scale:.3e} > {tol:.1e}"
        )

    def _velocity_dependent_forces(self, u_ice: ArrayLike2) -> np.ndarray:
        u = np.asarray(u_ice, dtype=float)
        return self.wind_stress(u) + self.water_stress(u) + self.coriolis_force(u)

    def _coriolis_matrix(self) -> np.ndarray:
        """Matrix C with ``coriolis_force(u) = C u``."""
        if self.coriolis_mass == "ice":
            m_eff = self.mass_per_area
            return -m_eff * self.f * np.array([[0.0, -1.0], [1.0, 0.0]])
        M = self.mass_matrix
        return -self.f * M @ np.array([[0.0, -1.0], [1.0, 0.0]])

    # -- reporting ----------------------------------------------------------
    def report(self, u_ice: ArrayLike2, reference: Optional[ArrayLike2] = None) -> dict:
        """Diagnostics for a drift velocity (deflection angle, speed ratio, forces)."""
        u = np.asarray(u_ice, dtype=float)
        ref = self.wind if reference is None else np.asarray(reference, dtype=float)
        wind_speed = float(np.linalg.norm(self.wind))
        speed = float(np.linalg.norm(u))
        fr = self.forces(u)
        return {
            "f": self.f,
            "mass_per_area": self.mass_per_area,
            "speed": speed,
            "wind_speed": wind_speed,
            "speed_ratio": speed / wind_speed if wind_speed else float("nan"),
            "deflection_deg": deflection_angle(u, ref),
            "u_x": float(u[0]),
            "u_y": float(u[1]),
            "wind_stress_magnitude": float(np.linalg.norm(fr["terms"]["wind"])),
            "water_stress_magnitude": float(np.linalg.norm(fr["terms"]["water"])),
            "coriolis_magnitude": float(np.linalg.norm(fr["terms"]["coriolis"])),
            "residual_magnitude": float(np.linalg.norm(fr["net"])),
        }


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def linear_slab_solution(
    wind_stress: ArrayLike2,
    mass_per_area: float,
    f: float,
    drag_rate: float,
) -> dict:
    """Steady drifting slab with *linear* drag ``tau_w = -m C_D (u - u_ocean)``.

    With ``u_ocean = 0`` the balance
    ``m C_D u + m f k x u = tau_a`` gives the exact Ekman-type result

        theta = arctan(f / C_D)   (to the right for f > 0),
        |u|   = |tau_a| / (m sqrt(f^2 + C_D^2)).

    This is the closed form the free-drift angle is often quoted as.
    """
    if drag_rate <= 0.0:
        raise ValueError("drag_rate must be positive")
    tau = np.asarray(wind_stress, dtype=float)
    # k x u = (-u_y, u_x) = Kk u with Kk = [[0,-1],[1,0]]; the Coriolis force is
    # -m f Kk u, so the steady drag balance is (m C_D I + m f Kk) u = tau.
    Kk = np.array([[0.0, -1.0], [1.0, 0.0]])
    A = mass_per_area * drag_rate * np.eye(2) + mass_per_area * f * Kk
    u = np.linalg.solve(A, tau)
    theta = math.degrees(math.atan2(f, drag_rate))
    speed = float(np.linalg.norm(tau)) / (mass_per_area * math.hypot(f, drag_rate))
    return {
        "u": u,
        "speed": float(np.linalg.norm(u)),
        "angle_deg": theta,
        "speed_closed_form": speed,
        "deflection_deg": deflection_angle(u, tau),
    }


def nansen_angle(f: float, drag_rate: float) -> float:
    """theta = arctan(f / C_D) [deg] for the linear-friction slab (Ekman limit)."""
    if drag_rate <= 0.0:
        raise ValueError("drag_rate must be positive")
    return math.degrees(math.atan2(f, drag_rate))


def free_drift_state(
    wind_stress_magnitude: float,
    mass_per_area: float,
    f: float,
    rho_water: float,
    c_d_water: float,
) -> dict:
    """Exact steady free drift with *quadratic* water drag and prescribed wind stress.

    Wind blow stress ``T_a`` along +x, ocean at rest, no surface tilt::

        T_a cos(theta) = rho_w C_w U^2        (along-wind)
        T_a sin(theta) = m f U                (transverse, Coriolis)

    Eliminating U gives the scalar equation

        tan(theta) sqrt(cos theta) = Lambda,
        Lambda = m f / sqrt(rho_w C_w T_a),

    solved here by bisection on (0, pi/2) to machine precision, with
    ``U = sqrt(T_a cos(theta) / (rho_w C_w))``.
    """
    if wind_stress_magnitude <= 0.0:
        raise ValueError("wind stress magnitude must be positive")
    if mass_per_area <= 0.0 or rho_water <= 0.0 or c_d_water <= 0.0:
        raise ValueError("mass, rho_water and C_D must be positive")
    c_w = rho_water * c_d_water
    if f == 0.0:
        return {
            "angle_deg": 0.0,
            "speed": math.sqrt(wind_stress_magnitude / c_w),
            "lambda": 0.0,
        }
    lam = abs(mass_per_area * f) / math.sqrt(c_w * wind_stress_magnitude)

    def g(a: float) -> float:
        return math.tan(a) * math.sqrt(math.cos(a)) - lam

    lo, hi = 1e-12, math.pi / 2.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15:
            break
    theta = 0.5 * (lo + hi)
    speed = math.sqrt(wind_stress_magnitude * math.cos(theta) / c_w)
    return {
        "angle_deg": math.degrees(theta),
        "speed": speed,
        "lambda": lam,
        "speed_check": mass_per_area * abs(f) / (c_w * math.tan(theta)),
    }


# ---------------------------------------------------------------------------
# Ekman spiral / boundary-layer closure
# ---------------------------------------------------------------------------

def ekman_decay_depth(f: float, K: float) -> float:
    """e-folding depth of the Ekman spiral, D_E = sqrt(2K/f) [m]."""
    if K <= 0.0:
        raise ValueError("eddy viscosity K must be positive")
    if f == 0.0:
        raise ValueError("Ekman depth undefined for f = 0")
    return math.sqrt(2.0 * K / abs(f))


def ekman_frictional_depth(f: float, K: float) -> float:
    """Depth of frictional influence, D_pi = pi sqrt(2K/f) (where the current reverses)."""
    return math.pi * ekman_decay_depth(f, K)


def ekman_surface_velocity(
    wind_stress: ArrayLike2, f: float, K: float, rho_water: float
) -> np.ndarray:
    """Surface (ice-base) ocean velocity of the Ekman spiral.

    ``w_s = tau / (rho_w sqrt(|f| K)) * exp(-i pi/4)``: magnitude
    ``|tau|/(rho_w sqrt(|f|K))``, rotated 45 deg to the right of the stress
    (Northern Hemisphere).
    """
    tau = np.asarray(wind_stress, dtype=float)
    mag = float(np.linalg.norm(tau))
    amp = mag / (rho_water * math.sqrt(abs(f) * K))
    return rotate_right(tau / mag, math.pi / 4.0) * amp if mag > 0 else np.zeros(2)


def ekman_spiral(
    wind_stress: ArrayLike2,
    f: float,
    K: float,
    rho_water: float,
    n_layers: int = 201,
    depth_factor: float = 3.0,
) -> dict:
    """Constant-viscosity Ekman spiral below the ice.

    Analytical solution of ``f k x u = K d^2u/dz^2`` with ``u -> 0`` as
    ``z -> -inf`` and ``rho_w K du/dz|_{z=0} = tau``::

        w(z) = w_s exp(-(1+i)|z|/D_E),  D_E = sqrt(2K/f)

    i.e. amplitude decays as ``exp(-|z|/D_E)`` and the phase rotates clockwise
    by ``|z|/D_E``.  Depth arrays are positive downward.
    """
    f = abs(f)  # spiral handedness: Northern Hemisphere convention
    D = ekman_decay_depth(f, K)
    z = np.linspace(0.0, depth_factor * D, n_layers)
    ws = ekman_surface_velocity(wind_stress, f, K, rho_water)
    dec = np.exp(-z / D)
    phase = -z / D  # clockwise rotation with depth
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    for i in range(z.size):
        rot = rotate(ws, phase[i]) * dec[i]
        u[i], v[i] = rot[0], rot[1]
    amp = dec * float(np.linalg.norm(ws))
    # total Ekman transport by trapezoidal integration of the profile
    trapezoid = getattr(np, "trapezoid", None)  # numpy>=2 renamed trapz
    if trapezoid is None:  # pragma: no cover - numpy<2
        trapezoid = np.trapz
    transport = np.array([trapezoid(u, z), trapezoid(v, z)])
    tau = np.asarray(wind_stress, dtype=float)
    t_ref = rotate_right(tau / (rho_water * f), math.pi / 2.0)  # 90 deg right
    return {
        "z": z,
        "u": u,
        "v": v,
        "amplitude": amp,
        "decay_depth": D,
        "frictional_depth": math.pi * D,
        "surface_velocity": ws,
        "surface_speed": float(np.linalg.norm(ws)),
        "surface_deflection_from_stress_deg": deflection_angle(ws, tau),
        "transport": transport,
        "transport_reference": t_ref,
        "transport_error": float(np.linalg.norm(transport - t_ref)),
    }


def ekman_spiral_residual(
    spiral: dict, f: float, K: float
) -> float:
    """Max residual of ``f k x u = K u_zz`` on the sampled profile (scaled)."""
    z, u, v = spiral["z"], spiral["u"], spiral["v"]
    if z.size < 7:
        raise ValueError("need at least 7 profile points")
    # central differences, interior points only (one-sided stencils are O(h))
    uzz = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (z[1] - z[0]) ** 2
    vzz = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (z[1] - z[0]) ** 2
    ui, vi = u[1:-1], v[1:-1]
    res_u = -f * vi - K * uzz
    res_v = f * ui - K * vzz
    local = np.maximum(np.hypot(f * ui, f * vi), 1e-300)
    return float(np.max(np.maximum(np.abs(res_u), np.abs(res_v)) / local))


def ekman_depth_from_profile(z: np.ndarray, amplitude: np.ndarray) -> float:
    """Least-squares estimate of D_E from a sampled amplitude profile (log-linear fit)."""
    z = np.asarray(z, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    mask = amp > 0.0
    if np.count_nonzero(mask) < 2:
        raise ValueError("need at least two positive amplitudes")
    slope, intercept = np.polyfit(z[mask], np.log(amp[mask]), 1)
    return float(-1.0 / slope)


def ekman_drag_coefficient(f: float, K: float, rho_water: float) -> float:
    """Ekman closure drag coefficient [kg/(m^2 s)]: ``|tau| = rho_w sqrt(fK) |u|``."""
    return rho_water * math.sqrt(abs(f) * K)