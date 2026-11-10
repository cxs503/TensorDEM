"""One-dimensional transient conduction across a sliding ice-steel interface.

The contact model
-----------------
A friction heat flux ``q(t)`` (W/m^2, per unit *real* contact area) is applied at
the plane ``x = 0`` separating an ice column from a steel column.  The interface
is a shared node with **no capacity** (its thickness is zero on the diffusion
scale), so the temperature is continuous and the flux split is *solved*, not
imposed:

    ``k_i (T_1 - T_0)/dx + k_s (T_s1 - T_0)/dx + q = 0``.

Far ends are Dirichlet at the bulk temperatures.  Backward Euler in time, finite
volume in space; the whole (interface + two columns) system is one banded linear
solve, so the time step is not limited by an explicit stability bound and the
scheme is L-stable (the flux is switched on at ``t = 0`` with a jump).

Because the discrete operator is linear and the boundary condition is the
Neumann flux, the interface temperature is **affine in the flux**:

    ``T_0(q) = T_a + Z q``        (:meth:`TwoBodyContactConductor.trial_response`)

two trial solves (``q = 0``, ``q = 1``) give ``T_a`` and ``Z``.  That is what
makes the friction-heat/melting fixed point of
:mod:`tensordem.ice_friction_thermo.selfconsistent` a one-dimensional root in
``mu`` instead of a nested iteration.

Verification (see the test suite)
---------------------------------
* **Early time** -- while the thermal penetration depth ``sqrt(alpha t)`` is much
  smaller than the column depth the numerical interface rise must reproduce
  Blok's semi-infinite result ``2 q sqrt(t/pi) / (e_ice + e_steel)``; the error
  is measured to fall as the columns are deepened.
* **Steady state** -- with a constant flux the interface rise must tend to
  ``q (d_i/k_i + d_s/k_s)`` exactly (the discretisation is exact for the linear
  steady profile, so the residual is round-off).
* **Energy** -- ``int q dt`` must equal the sensible enthalpy of both columns,
  to round-off, at every step.

References
----------
Blok, H. (1937). Theoretical study of temperature rise at surfaces of actual
    contact under oiliness conditions. Proc. Inst. Mech. Eng. 2, 222-235.
Carslaw, H.S. & Jaeger, J.C. (1959). Conduction of Heat in Solids, 2nd ed.
Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .properties import ICE, STEEL, PhaseProperties

__all__ = [
    "TwoBodyContactConductor",
    "blok_interface_rise_k",
    "steady_state_resistance_m2_k_w",
]


def blok_interface_rise_k(time_s: float, flux_w_m2: float,
                          ice: PhaseProperties = ICE,
                          steel: PhaseProperties = STEEL) -> float:
    """Semi-infinite two-body interface rise under a constant flux [K].

    ``T_0 - T_bulk = 2 q sqrt(t/pi) / (e_ice + e_steel)``, the Blok solution for
    two semi-infinite bodies sharing a plane heat source.  Valid while
    ``sqrt(alpha t)`` is smaller than the column thickness; it is the reference
    the finite-volume conductor is graded against at early time.
    """
    t = float(time_s)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError("time_s must be finite and non-negative")
    if t == 0.0:
        return 0.0
    effusivity = ice.effusivity_j_m2_k_sqrt_s + steel.effusivity_j_m2_k_sqrt_s
    return 2.0 * float(flux_w_m2) * math.sqrt(t / math.pi) / effusivity


def steady_state_resistance_m2_k_w(ice_depth_m: float, steel_depth_m: float,
                                   ice: PhaseProperties = ICE,
                                   steel: PhaseProperties = STEEL) -> float:
    """Steady resistance of the contact pair [m^2 K / W].

    The two columns are in **parallel**: both span the same temperature
    difference between the shared interface and their own far reservoir, so

        ``1/R = k_ice/d_ice + k_steel/d_steel``,

    and the interface rise under a flux ``q`` is ``q R``.  (A series
    ``d_i/k_i + d_s/k_s`` would be a stack with the flux crossing both media --
    a different problem, and 20x larger here because steel conducts 20x better
    than ice.  Keeping the wrong formula in this docstring's place is what made
    the steady-state check read as a 96 % discrepancy; the solver was right.)
    """
    parallel = ice.k_w_m_k / float(ice_depth_m) + steel.k_w_m_k / float(steel_depth_m)
    return 1.0 / parallel


@dataclass
class TwoBodyContactConductor:
    """Implicit finite-volume conductor for an ice/steel contact pair.

    The temperature vector is ``[T_interface, ice cells..., steel cells...]``;
    ``state[0]`` is the contact temperature that the friction law reads.
    """

    ice: PhaseProperties = ICE
    steel: PhaseProperties = STEEL
    ice_depth_m: float = 20.0e-3
    steel_depth_m: float = 20.0e-3
    ice_cells: int = 80
    steel_cells: int = 80
    initial_temperature_c: float = -10.0
    steel_temperature_c: float | None = None

    # populated by __post_init__
    dx_ice: float = field(init=False)
    dx_steel: float = field(init=False)
    state: np.ndarray = field(init=False)
    initial_state: np.ndarray = field(init=False)
    heat_in_j_m2: float = field(init=False, default=0.0)
    boundary_outflow_j_m2: float = field(init=False, default=0.0)
    time_s: float = field(init=False, default=0.0)
    _lu: object = field(init=False, default=None, repr=False)
    _lu_dt: float = field(init=False, default=-1.0, repr=False)
    _a: object = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("ice_depth_m", "steel_depth_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ice_cells", "steel_cells"):
            value = int(getattr(self, name))
            if value < 2:
                raise ValueError(f"{name} must be at least 2")
            object.__setattr__(self, name, value)
        if not math.isfinite(self.initial_temperature_c) or self.initial_temperature_c >= 0.0:
            raise ValueError("initial_temperature_c must be finite and below 0 degC")
        if self.steel_temperature_c is not None:
            if not math.isfinite(self.steel_temperature_c) or self.steel_temperature_c >= 0.0:
                raise ValueError("steel_temperature_c must be finite and below 0 degC")
        object.__setattr__(self, "dx_ice", self.ice_depth_m / self.ice_cells)
        object.__setattr__(self, "dx_steel", self.steel_depth_m / self.steel_cells)
        n = 1 + self.ice_cells + self.steel_cells
        state = np.empty(n, dtype=float)
        state[0] = self.initial_temperature_c
        state[1:1 + self.ice_cells] = self.initial_temperature_c
        steel_bulk = (self.initial_temperature_c if self.steel_temperature_c is None
                      else self.steel_temperature_c)
        state[1 + self.ice_cells:] = steel_bulk
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "initial_state", state.copy())
        object.__setattr__(self, "heat_in_j_m2", 0.0)
        object.__setattr__(self, "boundary_outflow_j_m2", 0.0)
        object.__setattr__(self, "time_s", 0.0)

    # -- geometry ----------------------------------------------------------
    @property
    def n_unknowns(self) -> int:
        return 1 + self.ice_cells + self.steel_cells

    @property
    def interface_temperature_c(self) -> float:
        return float(self.state[0])

    # -- operators ---------------------------------------------------------
    def _matrix(self, dt: float) -> np.ndarray:
        """Implicit system matrix (unknowns: interface + both columns)."""
        ice, steel = self.ice, self.steel
        dxi, dxs = self.dx_ice, self.dx_steel
        g_i, g_s = ice.k_w_m_k / dxi, steel.k_w_m_k / dxs
        c_i = ice.capacity_j_m3_k * dxi          # areal capacity [J/(m^2 K)]
        c_s = steel.capacity_j_m3_k * dxs

        n = self.n_unknowns
        a = np.zeros((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        # row 0: interface energy balance (zero capacity, algebraic).  The
        # face conductance interface<->first cell is 2k/dx (half a cell to the
        # node), and *the same* value is used in the first-cell row below: using
        # g there and 2g here would create or destroy heat at the interface.
        i0 = 1                       # first ice cell index
        s0 = 1 + self.ice_cells      # first steel cell index
        a[0, 0] = 2.0 * (g_i + g_s)
        a[0, i0] = -2.0 * g_i
        a[0, s0] = -2.0 * g_s

        # ice cells
        for k in range(self.ice_cells):
            row = i0 + k
            left_g = 2.0 * g_i if k == 0 else g_i
            a[row, row] += left_g + c_i / dt
            a[row, row - 1] -= left_g
            b[row] += c_i / dt * self.state[row]
            if k == self.ice_cells - 1:
                a[row, row] += 2.0 * g_i         # half-cell to the Dirichlet face
                b[row] += 2.0 * g_i * self.initial_temperature_c
            else:
                a[row, row] += g_i
                a[row, row + 1] -= g_i

        # steel cells
        steel_bulk = (self.initial_temperature_c if self.steel_temperature_c is None
                      else self.steel_temperature_c)
        for k in range(self.steel_cells):
            row = s0 + k
            a[row, row] += (2.0 * g_s if k == 0 else g_s) + c_s / dt
            if k == 0:
                # the first steel cell faces the *interface* node (index 0), not
                # the last ice cell: one wrong column here silently decouples the
                # steel column and leaks the whole flux (measured: 95 % of the
                # injected heat vanished before this was fixed).
                a[row, 0] -= 2.0 * g_s
            else:
                a[row, row - 1] -= g_s
            b[row] += c_s / dt * self.state[row]
            if k == self.steel_cells - 1:
                a[row, row] += 2.0 * g_s
                b[row] += 2.0 * g_s * steel_bulk
            else:
                a[row, row] += g_s
                a[row, row + 1] -= g_s
        return a, b

    def trial_response(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Affine thermal response ``T(q) = T_zero + q * sensitivity``.

        Two right-hand sides (``q = 0`` and ``q = 1``) are solved with the same
        matrix, giving the whole temperature field at zero flux and the
        sensitivity of every node to the applied flux.  The returned
        ``sensitivity[0]`` is the interface impedance ``Z`` [m^2 K/W]: the
        contact-temperature rise per unit flux, positive by construction (the
        second law of thermodynamics is a test case).
        """
        self._check_dt(dt)
        lu = self._factor(dt)
        a, b = self._system(dt)
        rhs = np.zeros((self.n_unknowns, 2), dtype=float)
        rhs[:, 0] = b
        rhs[:, 1] = b.copy()
        rhs[0, 1] += 1.0            # unit flux enters the interface balance
        solution = lu.solve(rhs)
        return solution[:, 0], solution[:, 1] - solution[:, 0]

    def _factor(self, dt: float):
        """Sparse LU of the implicit system, cached for a repeated ``dt``."""
        if self._lu is None or self._lu_dt != float(dt):
            a, _b = self._matrix(dt)
            object.__setattr__(self, "_lu", splu(csc_matrix(a)))
            object.__setattr__(self, "_lu_dt", float(dt))
        return self._lu

    def _system(self, dt: float):
        """Cached operator + current right-hand side (``b`` depends on the state)."""
        if self._a is None or self._lu_dt != float(dt):
            a, _b = self._matrix(dt)
            object.__setattr__(self, "_a", a)
        return self._a, self._rhs(dt)

    def _rhs(self, dt: float) -> np.ndarray:
        """Known terms of the implicit system (old state + Dirichlet values)."""
        ice, steel = self.ice, self.steel
        c_i = ice.capacity_j_m3_k * self.dx_ice
        c_s = steel.capacity_j_m3_k * self.dx_steel
        g_i = ice.k_w_m_k / self.dx_ice
        g_s = steel.k_w_m_k / self.dx_steel
        steel_bulk = (self.initial_temperature_c if self.steel_temperature_c is None
                      else self.steel_temperature_c)
        n = self.n_unknowns
        b = np.zeros(n, dtype=float)
        b[1:1 + self.ice_cells] = c_i / dt * self.state[1:1 + self.ice_cells]
        b[1 + self.ice_cells:] = c_s / dt * self.state[1 + self.ice_cells:]
        b[1 + self.ice_cells - 1] += 2.0 * g_i * self.initial_temperature_c
        b[-1] += 2.0 * g_s * steel_bulk
        return b

    def commit(self, dt: float, flux_w_m2: float) -> float:
        """Advance one step with the given interface flux; return ``T_contact``."""
        self._check_dt(dt)
        zero, sensitivity = self.trial_response(dt)
        self.state = zero + float(flux_w_m2) * sensitivity
        self.time_s += dt
        self.heat_in_j_m2 += float(flux_w_m2) * dt
        # heat conducted out through the two Dirichlet faces (positive outward);
        # without this term the ledger "loses" energy as soon as the thermal wave
        # reaches the far boundary, which looks like a conservation failure and is
        # not one.
        n_i = self.ice_cells
        g_i = self.ice.k_w_m_k / self.dx_ice
        g_s = self.steel.k_w_m_k / self.dx_steel
        steel_bulk = (self.initial_temperature_c if self.steel_temperature_c is None
                      else self.steel_temperature_c)
        outflow = (2.0 * g_i * (self.state[n_i] - self.initial_temperature_c)
                   + 2.0 * g_s * (self.state[-1] - steel_bulk))
        self.boundary_outflow_j_m2 += outflow * dt
        return self.interface_temperature_c

    # -- audits ------------------------------------------------------------
    def sensible_energy_j_m2(self) -> float:
        """Sensible enthalpy stored in both columns above the initial state."""
        ice = self.ice.capacity_j_m3_k * self.dx_ice
        steel = self.steel.capacity_j_m3_k * self.dx_steel
        n_i, n_s = self.ice_cells, self.steel_cells
        ice_part = ice * float(np.sum(self.state[1:1 + n_i] - self.initial_state[1:1 + n_i]))
        steel_part = steel * float(np.sum(self.state[1 + n_i:] - self.initial_state[1 + n_i:]))
        return ice_part + steel_part

    def energy_residual_j_m2(self) -> float:
        """``int q dt - sensible - boundary outflow`` (round-off if conservative)."""
        return (self.heat_in_j_m2 - self.sensible_energy_j_m2()
                - self.boundary_outflow_j_m2)

    def _check_dt(self, dt: float) -> None:
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")