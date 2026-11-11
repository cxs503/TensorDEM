"""Transient conduction in an ice block heated by its own collision plastic work.

Step 2 of the node: ``dT/dt = alpha d2T/dx2`` inside the block, with the
collision face carrying either a **heat flux** (friction / plastic work spread
over the contact) or a **volumetric source** (plastic work deposited in a layer
of depth ``delta`` under the contact).  Both are needed: the flux form is what
the sliding-contact chain of :mod:`tensordem.interface_thermal` already speaks,
the volumetric form is what a crushing impact actually produces.

Two discretisations, deliberately independent
---------------------------------------------
``SlabHeat1D``
    Cell-centred **finite volume**, lumped capacity, tridiagonal theta-scheme
    (``theta = 0.5`` is Crank-Nicolson).  Optional geometric cell grading
    towards the contact face, uniform surface flux, volumetric source, Fourier
    (convective) far boundary, and a strongly imposed Dirichlet surface node.

``slab_fem_nodes``
    Nodal **finite element**, linear elements, *consistent* mass matrix,
    theta-scheme, same boundary data.  It shares no code with the FV solver.

The FEM solver exists so that "thermal stress field vs FEM reference" is a
real cross-check of two discretisations and not a restatement of one solver:
the benchmark compares FV, FEM and the closed-form solutions against each
other, and the residual disagreement is what the report quotes.

Exact solutions used as grading references (all classical)
----------------------------------------------------------
1. Constant surface flux into a semi-infinite body (Carslaw & Jaeger 1959,
   eq. 2-46) -- :func:`exact_semi_infinite_flux`.  Implemented by the existing
   :func:`tensordem.interface_thermal.conduction.semi_infinite_temperature_profile`.
2. Uniform volumetric source in a slab held at ``T0`` on both faces
   (steady state ``q''' L^2 /(2k) * [xi(1-xi)]`` approached through the sine
   series of :func:`exact_slab_uniform_source`).
3. Instantaneous plane source in an infinite medium (Gaussian,
   :func:`exact_plane_source`) -- the impulse response, i.e. the short-time
   limit of the deposited patch.
4. Instantaneous point source in an infinite medium
   (:func:`exact_point_source_sphere`) -- the radial kernel quoted for the
   deposition spot, used by the report as the lateral-spreading bound.

Units are SI; temperatures are degC (only differences enter, so the equations
are unaffected).  Surface flux is positive *into* the body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc

from ..interface_thermal.conduction import (
    semi_infinite_surface_temperature,
    semi_infinite_temperature_profile,
)
from .properties import IceProperties, ice_properties

__all__ = [
    "SlabHeat1D",
    "slab_fem_nodes",
    "exact_semi_infinite_flux",
    "exact_slab_uniform_source",
    "exact_plane_source",
    "exact_point_source_sphere",
    "observed_order",
    "fv_fem_disagreement",
]


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and x <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return x


# ---------------------------------------------------------------------------
# finite-volume conductor
# ---------------------------------------------------------------------------
@dataclass
class SlabHeat1D:
    """Cell-centred transient conductor for one ice column (per unit area).

    Parameters
    ----------
    cells : int
        Number of finite-volume cells.
    thickness_m : float
        Total depth of the modelled column (m).  Deep enough that the far end
        stays at ``t_initial`` for the duration of the collision.
    properties : IceProperties
        Material state; ``k``, ``rho cp`` and ``alpha`` are taken from it.
    theta : float
        Time-integration weight: ``0.5`` Crank-Nicolson (2nd order),
        ``1.0`` fully implicit.
    grading : float
        **Total** refinement ratio ``dx_last / dx_first`` (``>= 1``): the cells
        grow geometrically away from the contact face, so the layer the
        collision heats is the best resolved one.  ``1.0`` is a uniform grid;
        values below 1 are rejected (they would coarsen exactly that layer).
    t_initial : float
        Initial / undisturbed temperature (degC).

    Notes
    -----
    The state is per unit *faced* area: ``capacity`` is J/(m^2 K).  Energy
    accounting (:meth:`energy_balance`) therefore closes against the surface
    flux plus the integral of the volumetric source, which is exactly the
    ledger the report uses to prove no heat is created by the discretisation.
    """

    cells: int
    thickness_m: float
    properties: IceProperties
    theta: float = 0.5
    grading: float = 1.0
    t_initial: float = 0.0
    x: np.ndarray = field(init=False, repr=False)
    dx: np.ndarray = field(init=False, repr=False)
    conductance: np.ndarray = field(init=False, repr=False)
    capacity: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False, repr=False)
    _heat_in: float = field(init=False, default=0.0, repr=False)
    _source_in: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.cells = int(self.cells)
        if self.cells < 2:
            raise ValueError(f"cells must be >= 2, got {self.cells!r}")
        self.thickness_m = _finite("thickness_m", self.thickness_m, positive=True)
        self.theta = _finite("theta", self.theta)
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(f"theta must be in [0, 1], got {self.theta!r}")
        g = _finite("grading", self.grading, positive=True)
        if g < 1.0 - 1e-12:
            raise ValueError(
                f"grading must be >= 1 (refine towards the contact face), got {g!r}")
        n = self.cells
        if abs(g - 1.0) < 1e-12:
            widths = np.full(n, self.thickness_m / n)
        else:
            ratio = g ** (1.0 / (n - 1))          # per-cell growth factor
            powers = ratio ** np.arange(n)
            widths = self.thickness_m * powers / powers.sum()
        self.dx = widths
        self.x = np.cumsum(widths) - 0.5 * widths  # cell centres
        h = 0.5 * (widths[:-1] + widths[1:])
        self.conductance = self.properties.k_w_m_k / h          # W/(m^2 K)
        self.capacity = self.properties.volumetric_heat_capacity_j_m3_k * widths
        self.t = np.full(n, float(self.t_initial), dtype=float)
        self._heat_in = 0.0
        self._source_in = 0.0

    # -- accessors ------------------------------------------------------
    @property
    def alpha(self) -> float:
        return self.properties.diffusivity_m2_s

    def reset(self, t_initial: float | None = None) -> None:
        value = self.t_initial if t_initial is None else float(t_initial)
        self.t_initial = value
        self.t = np.full(self.cells, value)
        self._heat_in = 0.0
        self._source_in = 0.0

    def surface_temperature(self) -> float:
        return float(self.t[0])

    def stored_energy(self) -> float:
        """Enthalpy above the initial state per unit area (J/m^2)."""
        return float(np.sum(self.capacity * (self.t - self.t_initial)))

    def source_profile(self, total_power_w_m2: float, depth_m: float) -> np.ndarray:
        """Uniform volumetric source (W/m^3) deposited in the first ``depth_m``.

        ``total_power_w_m2`` is a power *per unit faced area* (W/m^2); it is
        spread over the cells intersecting ``depth_m`` so that
        ``sum(q_vol * dx) == total_power_w_m2`` exactly.
        """
        q = _finite("total_power_w_m2", total_power_w_m2)
        d = _finite("depth_m", depth_m, positive=True)
        q_vol = np.zeros(self.cells)
        remaining = d
        for i in range(self.cells):
            if remaining <= 0.0:
                break
            take = min(remaining, float(self.dx[i]))
            q_vol[i] = q / d
            remaining -= take
        # exact discrete normalisation (guards non-integer cell coverage)
        weight = float(np.sum(q_vol * self.dx))
        if weight > 0.0:
            q_vol *= q / weight
        return q_vol

    # -- operators ------------------------------------------------------
    def _apply_operator(self, t: np.ndarray, q_surface: float,
                        q_vol: np.ndarray | None,
                        t_far: float | None) -> np.ndarray:
        """Net inflow per cell (W/m^2) for the theta scheme."""
        flux = self.conductance * (t[1:] - t[:-1])
        out = np.zeros_like(t)
        out[1:] -= flux
        out[:-1] += flux
        out[0] += q_surface
        if q_vol is not None:
            out += q_vol * self.dx
        if t_far is not None:
            out[-1] += self.conductance[-1] * (t_far - t[-1])
        return out

    def step(self, dt: float, *, q_surface: float = 0.0,
             q_vol: np.ndarray | None = None, t_far: float | None = None,
             t_surface: float | None = None) -> np.ndarray:
        """Advance one theta step and accumulate the energy ledger."""
        dt = _finite("dt", dt, positive=True)
        if q_vol is not None:
            q_vol = np.asarray(q_vol, dtype=float)
            if q_vol.shape != (self.cells,):
                raise ValueError(
                    f"q_vol must have shape ({self.cells},), got {q_vol.shape}"
                )
        theta = self.theta
        cap_dt = self.capacity / dt
        diag = cap_dt.copy()
        upper = -theta * self.conductance
        lower = -theta * self.conductance
        diag[:-1] += theta * self.conductance
        diag[1:] += theta * self.conductance
        rhs = cap_dt * self.t + (1.0 - theta) * self._apply_operator(
            self.t, float(q_surface), q_vol, t_far)
        rhs[0] += theta * float(q_surface)
        if q_vol is not None:
            rhs += theta * q_vol * self.dx
        if t_far is not None:
            diag[-1] += theta * self.conductance[-1]
            rhs[-1] += theta * self.conductance[-1] * float(t_far)
        ab = np.zeros((3, self.cells))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        if t_surface is not None:
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs[0] = float(t_surface)
        new = solve_banded((1, 1), ab, rhs)
        self._heat_in += float(q_surface) * dt
        if t_far is not None:
            self._heat_in += self.conductance[-1] * (float(t_far) - float(new[-1])) * dt
        if q_vol is not None:
            self._source_in += float(np.sum(q_vol * self.dx)) * dt
        self.t = new
        return self.t

    def run(self, t_end: float, dt: float, *,
            q_surface: float | Callable[[float], float] = 0.0,
            q_vol: np.ndarray | Callable[[float], np.ndarray] | None = None,
            t_far: float | None = None,
            t_surface: float | None = None,
            record_times: Sequence[float] | None = None) -> dict[str, np.ndarray]:
        """March to ``t_end``; ``q`` arguments may be functions of time."""
        t_end = _finite("t_end", t_end, positive=True)
        dt = _finite("dt", dt, positive=True)
        flux_fn = q_surface if callable(q_surface) else (lambda _t: float(q_surface))
        src_fn = (q_vol if callable(q_vol)
                  else (None if q_vol is None else (lambda _t: np.asarray(q_vol, float))))
        targets = sorted(float(v) for v in (record_times or ()))
        time = 0.0
        hist: dict[str, list] = {"time": [], "profiles": [], "surface": []}
        pending = list(targets)
        while time < t_end - 1e-15:
            h = dt
            if pending and pending[0] - time < h:
                h = max(pending[0] - time, 1e-15)
            self.step(h, q_surface=flux_fn(time + 0.5 * h),
                      q_vol=None if src_fn is None else src_fn(time + 0.5 * h),
                      t_far=t_far, t_surface=t_surface)
            time += h
            while pending and pending[0] <= time + 1e-12:
                hist["time"].append(pending.pop(0))
                hist["profiles"].append(self.t.copy())
                hist["surface"].append(self.surface_temperature())
        if not targets:
            hist["time"].append(time)
            hist["profiles"].append(self.t.copy())
            hist["surface"].append(self.surface_temperature())
        return {
            "time": np.array(hist["time"]),
            "profiles": np.array(hist["profiles"]),
            "surface": np.array(hist["surface"]),
            "x": self.x.copy(),
        }

    def energy_balance(self) -> dict[str, float]:
        """Discretisation energy ledger: stored vs. injected heat (J/m^2)."""
        stored = self.stored_energy()
        injected = self._heat_in + self._source_in
        return {
            "stored_j_m2": stored,
            "surface_heat_j_m2": self._heat_in,
            "source_heat_j_m2": self._source_in,
            "injected_j_m2": injected,
            "relative_error": (abs(stored - injected) / injected) if injected else 0.0,
        }


# ---------------------------------------------------------------------------
# independent finite-element reference
# ---------------------------------------------------------------------------
def slab_fem_nodes(
    n_elements: int,
    thickness_m: float,
    properties: IceProperties,
    *,
    t_end: float,
    dt: float,
    theta: float = 0.5,
    t_initial: float = 0.0,
    q_surface: float = 0.0,
    q_vol: float | np.ndarray = 0.0,
    t_far: float | None = None,
    t_surface: float | None = None,
    dirichlet_both_ends: bool = False,
    mass: str = "lumped",
    record_times: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Linear finite elements, consistent mass, theta scheme -- reference only.

    Independent of :class:`SlabHeat1D` in space (nodal, ``h1``-Galerkin,
    consistent mass) and shares only the material constants.  Agreeing with the
    finite-volume field to well below the acceptance tolerance is what makes
    the FV field credible; disagreeing would falsify one of them.

    ``mass`` selects ``"lumped"`` (default) or ``"consistent"``.  Mass lumping matters at a **flux**
    boundary: with a concentrated surface flux and a consistent mass matrix the
    nodal solution does *not* converge (measured in this package: the surface
    error saturates near 22 % of the surface rise and grows with refinement,
    see ``docs/ICE_THERMAL_STRESS_COUPLING.md``).  Lumped mass at a flux
    boundary, consistent mass everywhere else (Dirichlet data or a source that
    is not confined to a boundary layer) is the combination used by the
    benchmark; the lumped default is the safe one.

    Returns ``{"x": node coordinates, "profiles": (n_times, n_nodes),
    "times": ...}``.
    """
    n_elements = int(n_elements)
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2")
    L = _finite("thickness_m", thickness_m, positive=True)
    h = L / n_elements
    n_nodes = n_elements + 1
    k = properties.k_w_m_k
    rho_c = properties.volumetric_heat_capacity_j_m3_k
    # assemble tridiagonal K and M
    diag_k = np.zeros(n_nodes)
    off_k = np.full(n_nodes - 1, -k / h)
    diag_k[:-1] += k / h
    diag_k[1:] += k / h
    if mass not in ("consistent", "lumped"):
        raise ValueError(f"mass must be 'consistent' or 'lumped', got {mass!r}")
    if mass == "consistent":
        diag_m = np.full(n_nodes, 2.0 * rho_c * h / 6.0)
        diag_m[0] = rho_c * h / 3.0
        diag_m[-1] = rho_c * h / 3.0
        off_m = np.full(n_nodes - 1, rho_c * h / 6.0)
    else:
        diag_m = np.full(n_nodes, rho_c * h)
        diag_m[0] = diag_m[-1] = rho_c * h / 2.0
        off_m = np.zeros(n_nodes - 1)
    # load vector from the volumetric source (consistent)
    # consistent nodal loads of the volumetric source: element e carries a
    # uniform q_e, so f_i = (q_{i-1} + q_i) h / 2 with q_0 = 0 outside
    q_el = np.asarray(q_vol, dtype=float)
    if q_el.ndim == 0:
        q_el = np.full(n_elements, float(q_el))
    elif q_el.size != n_elements:
        raise ValueError(
            f"q_vol must be scalar or have one value per element ({n_elements})")
    f = np.zeros(n_nodes)
    f[:-1] += q_el * h / 2.0
    f[1:] += q_el * h / 2.0
    f[0] += q_surface
    if t_far is not None:
        f[-1] += k / h * float(t_far)

    T = np.full(n_nodes, float(t_initial))
    if t_far is not None:
        diag_k[-1] += k / h  # Robin (Fourier) far boundary
        f[-1] += k / h * float(t_far)
    targets = sorted(float(v) for v in (record_times or ()))
    pending = list(targets)
    times: list[float] = []
    profiles: list[np.ndarray] = []
    time = 0.0
    while time < t_end - 1e-15:
        step = dt
        if pending and pending[0] - time < step:
            step = max(pending[0] - time, 1e-15)
        a_off = off_m / step + theta * off_k
        a_diag = diag_m / step + theta * diag_k
        b_off = off_m / step - (1.0 - theta) * off_k
        b_diag = diag_m / step - (1.0 - theta) * diag_k
        ab = np.zeros((3, n_nodes))
        ab[0, 1:] = a_off
        ab[1, :] = a_diag
        ab[2, :-1] = a_off
        rhs = b_diag * T
        rhs[:-1] += b_off * T[1:]
        rhs[1:] += b_off * T[:-1]
        rhs += f  # unit-load vector: no implicit (theta) split needed, f is steady
        if t_surface is not None:
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs[0] = float(t_surface)
        if dirichlet_both_ends:
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs[0] = float(t_initial)
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
            rhs[-1] = float(t_initial)
        T = solve_banded((1, 1), ab, rhs)
        time += step
        while pending and pending[0] <= time + 1e-12:
            times.append(pending.pop(0))
            profiles.append(T.copy())
    if not targets:
        times.append(time)
        profiles.append(T.copy())
    return {
        "x": np.linspace(0.0, L, n_nodes),
        "times": np.array(times),
        "profiles": np.array(profiles),
        "theta": theta,
        "elements": n_elements,
    }


# ---------------------------------------------------------------------------
# closed-form grading references
# ---------------------------------------------------------------------------
def exact_semi_infinite_flux(x: Any, t: float, q_surface_w_m2: float,
                             properties: IceProperties,
                             t_initial: float = 0.0) -> Any:
    """``T(x,t)`` for a constant surface flux on a semi-infinite body.

    Carslaw & Jaeger (1959) eq. 2-46::

        T - T0 = (2q/k) sqrt(alpha t / pi) exp(-x^2/(4 alpha t))
                 - (q x / k) erfc(x / (2 sqrt(alpha t)))

    The surface value ``x = 0`` is the repository's existing
    :func:`tensordem.interface_thermal.conduction.semi_infinite_surface_temperature`;
    the depth dependence (the ``erfc`` leg, which is what grades the *shape* of
    the computed profile) is added here because the existing module only owns
    the surface scalar.
    """
    t = _finite("t", t, positive=True)
    q = _finite("q_surface_w_m2", q_surface_w_m2)
    k = properties.k_w_m_k
    alpha = properties.diffusivity_m2_s
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0.0):
        raise ValueError("x must be nonnegative")
    s = np.sqrt(alpha * t)
    out = (float(t_initial)
           + 2.0 * q / k * s * math.sqrt(1.0 / math.pi) * np.exp(-x_arr ** 2 / (4.0 * alpha * t))
           - q * x_arr / k * erfc(x_arr / (2.0 * s)))
    return float(out) if np.ndim(out) == 0 else out


def exact_slab_uniform_source(x: Any, t: float, q_vol_w_m3: float,
                              thickness_m: float, properties: IceProperties,
                              t_initial: float = 0.0, terms: int = 400) -> Any:
    r"""Exact ``T(x,t)`` in a slab ``0<x<L`` with uniform source, ``T=T0`` ends.

    Steady state ``T_ss(x) - T0 = (q''' L^2 / 2k) xi (1 - xi)`` is approached
    from the initial state through the odd sine modes:

    .. math::

        T - T_0 = \frac{q''' L^2}{2k}\xi(1-\xi)
                  - \sum_{n\ \mathrm{odd}} \frac{4 q''' L^2}{k n^3 \pi^3}
                    \sin(n\pi\xi)\, e^{-\alpha (n\pi/L)^2 t}

    (Derivation: expand ``T_ss`` in ``sin(n pi x / L)``;
    ``int_0^L xi(1-xi) sin(n pi xi) dxi = 4/(n^3 pi^3)`` for odd ``n``.)
    Used to grade the *volumetric source* implementation -- a term the pure
    surface-flux reference cannot see.
    """
    L = _finite("thickness_m", thickness_m, positive=True)
    q = _finite("q_vol_w_m3", q_vol_w_m3)
    x_arr = np.asarray(x, dtype=float)
    xi = np.clip(x_arr / L, 0.0, 1.0)
    steady = q * L * L / (2.0 * properties.k_w_m_k) * xi * (1.0 - xi)
    alpha = properties.diffusivity_m2_s
    decay = np.zeros_like(xi)
    for n in range(1, 2 * int(terms) + 1, 2):
        amp = 4.0 * q * L * L / (properties.k_w_m_k * n ** 3 * math.pi ** 3)
        decay = decay + amp * np.sin(n * math.pi * xi) * math.exp(
            -alpha * (n * math.pi / L) ** 2 * float(t))
    out = float(t_initial) + steady - decay
    return float(out) if np.ndim(out) == 0 else out


def exact_plane_source(x: Any, t: float, areal_energy_j_m2: float,
                       properties: IceProperties, t_initial: float = 0.0) -> Any:
    """Instantaneous plane source (impulse response) in an infinite medium.

    ``T - T0 = Q / (rho cp sqrt(4 pi alpha t)) exp(-x^2/(4 alpha t))``.
    The short-time limit of a deposited heat sheet; grades the diffusion
    operator without any boundary interference.
    """
    t = _finite("t", t, positive=True)
    alpha = properties.diffusivity_m2_s
    x_arr = np.asarray(x, dtype=float)
    pref = areal_energy_j_m2 / (
        properties.volumetric_heat_capacity_j_m3_k * math.sqrt(4.0 * math.pi * alpha * t))
    out = float(t_initial) + pref * np.exp(-x_arr ** 2 / (4.0 * alpha * t))
    return float(out) if np.ndim(out) == 0 else out


def exact_point_source_sphere(r: Any, t: float, energy_j: float,
                              properties: IceProperties, t_initial: float = 0.0) -> Any:
    """Instantaneous point source in an infinite medium (3-D radial kernel).

    ``T - T0 = Q / (rho cp (4 pi alpha t)^{3/2}) exp(-r^2/(4 alpha t))``.
    Quoted by the report as the *lateral spreading bound* of the contact patch.
    """
    t = _finite("t", t, positive=True)
    alpha = properties.diffusivity_m2_s
    r_arr = np.asarray(r, dtype=float)
    pref = energy_j / (
        properties.volumetric_heat_capacity_j_m3_k
        * (4.0 * math.pi * alpha * t) ** 1.5)
    out = float(t_initial) + pref * np.exp(-r_arr ** 2 / (4.0 * alpha * t))
    return float(out) if np.ndim(out) == 0 else out


def observed_order(errors: Sequence[float], refinements: Sequence[float]) -> float:
    """Convergence order from two successive errors and a refinement ratio."""
    e = np.asarray(errors, dtype=float)
    r = np.asarray(refinements, dtype=float)
    if e.size < 2:
        raise ValueError("need at least two errors")
    return float(np.log(e[0] / e[1]) / np.log(r[0]))


def fv_fem_disagreement(fv: SlabHeat1D, fem: dict[str, Any]) -> float:
    """Max nodal disagreement ``max|T_fv - T_fem|`` between two fields (K).

    The FEM nodes are averaged back to the FV cell centres (the FV value is the
    cell average, the FEM value a nodal point value) -- the two are compared in
    the same functional, ``max |T_fv - T_fem_interp|``.
    """
    prof_fv = fv.t
    prof_fem = fem["profiles"][-1]
    x_fem = fem["x"]
    interp = np.interp(fv.x, x_fem, prof_fem)
    return float(np.max(np.abs(prof_fv - interp)))


#: module self-check: the semi-infinite reference is the C&J surface value.
_p = ice_properties(-10.0)
assert abs(
    float(exact_semi_infinite_flux(0.0, 1.0, 1.0e4, _p))
    - float(semi_infinite_surface_temperature(
        1.0, _p.diffusivity_m2_s, _p.k_w_m_k, 1.0e4, 0.0))
) < 1e-12
# the flux profile is not the step profile: shape check at depth
assert float(exact_semi_infinite_flux(0.01, 1.0, 1.0e4, _p)) < float(
    exact_semi_infinite_flux(0.0, 1.0, 1.0e4, _p))
# the plane-source reference integrates to the injected areal energy
_x = np.linspace(-0.2, 0.2, 400001)
assert abs(_p.volumetric_heat_capacity_j_m3_k
           * np.trapezoid(exact_plane_source(_x, 0.5, 1.0e3, _p), _x) - 1.0e3) < 1.0
# the slab series tends to the steady parabola, and equals q L^2/(8k) at mid-plane
_steady = exact_slab_uniform_source(0.05, 1.0e6, 1.0e6, 0.1, _p)
assert abs(_steady - 1.0e6 * 0.1 ** 2 / (8.0 * _p.k_w_m_k)) < 1e-12
