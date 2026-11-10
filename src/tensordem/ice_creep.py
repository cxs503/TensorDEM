"""Glen flow law and Norton power-law creep for bonded-sphere ice DEM.

Physics
-------
Ice does not respond elastically to a *sustained* load: it creeps.  The
classical constitutive model is Glen's flow law, an incompressible power-law
fluid,

    edot_ij = (3/2) A(T) sigma_eq^(n-1) s_ij,
    s_ij = sigma_ij - (sigma_kk/3) delta_ij,
    sigma_eq = sqrt(3/2 s_ij s_ij),                       (Glen 1955, Nye 1953)

with the stress exponent ``n = 3`` for dislocation creep and an Arrhenius
temperature dependence of the fluidity

    A(T) = A_ref exp(-Q/R (1/T - 1/T_ref)),   Q ~ 60 kJ/mol.

The uniaxial special case of the same law is the Norton power law
``edot = A(T) sigma^n``.

Normalisation note (a real trap, see ``docs/ice_creep_glen_norton.md``)
----------------------------------------------------------------------
The tensor form is often written ``edot_ij = A sigma_eq^(n-1) s_ij``.  With the
von Mises equivalent stress used above that form gives a *uniaxial* rate of
``(2/3) A sigma^n``; it is the same law only if the fluidity is redefined as
``A_glen = (3/2) A_norton``.  This module uses the contract-consistent
normalisation ``edot_ij = (3/2) A sigma_eq^(n-1) s_ij`` so that the uniaxial
limit reproduces :func:`norton_uniaxial_rate` exactly, and the equivalent strain
rate is ``edot_eq = sqrt(2/3 edot:edot) = A sigma_eq^n``.  The literal mission
expression is available as ``normalisation="glaciological"``.

Bond law of the DEM lattice
---------------------------
A bonded-sphere bond of rest length ``L0`` and cross-section ``A_b`` carries
axial force ``F`` and axial stress ``sigma = F/A_b``.  A Norton dashpot in
series with the bond spring stores a creep elongation

    d(u_creep)/dt = A(T) sigma^n L0 sign(F)     =>   Delta u_creep = A sigma^n Delta t * L0

so the bond force is ``F = k (d - u_creep)`` (``d`` = current elongation).
Two further ingredients give the model all three creep stages:

* a **Kelvin (delayed-elastic) branch** ``k_K, tau_K`` -> bounded decelerating
  strain, the DEM image of primary (transient) creep;
* **Kachanov-Rabotnov damage** ``dD/dt = (1/t_d) (|sigma|/(sigma_ref (1-D)))^m``
  with bond stiffness ``~ (1-D)`` -> tertiary creep, since damage softens bonds,
  redistributes load and finally localises into a rupture path.

Limitations (stated, not hidden)
--------------------------------
* Creep acts on the **axial** bond channel only; the tangential channel is
  elastic.  The lattice is elastically isotropic but creeps through its axial
  bonds.
* The bond cross-section is the sphere section ``pi r^2``, not a Voronoi facet
  area, so the *absolute* macroscopic creep rate carries a lattice
  discretisation factor of order one.  The stress *exponent* is unaffected,
  which is what the Norton verification pins down.
* Explicit (Euler) integration of the dashpot/damage ODEs: the step must keep
  ``A sigma^(n-1) k L0 dt / A_b`` small.  No adaptive sub-stepping here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:                                  # pragma: no cover - optional accelerator
    from scipy.linalg import lu_factor as _lu_factor, lu_solve as _lu_solve
except Exception:                     # pragma: no cover
    _lu_factor = _lu_solve = None

R_GAS = 8.314462618  # J mol^-1 K^-1
T0_K = 273.15

__all__ = [
    "GlenCreepLaw",
    "UniaxialCreepModel",
    "LatticeCreepParameters",
    "BondedSphereCreepLattice",
    "cubic_lattice_block",
    "arrhenius_fluidity_ratio",
    "deviatoric_stress",
    "von_mises_stress",
    "glen_strain_rate",
    "equivalent_strain_rate",
    "norton_uniaxial_rate",
    "power_law_fit",
    "identify_creep_stages",
    "uniaxial_constant_stress",
    "punch_relaxation",
    "single_bond_relaxation_analytic",
]


# --------------------------------------------------------------------------- #
# Glen / Norton continuum constitutive law
# --------------------------------------------------------------------------- #
def deviatoric_stress(sigma) -> np.ndarray:
    """Deviatoric part ``s_ij = sigma_ij - (sigma_kk/3) delta_ij``."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (3, 3):
        raise ValueError("sigma must be a 3x3 tensor")
    return sigma - np.trace(sigma) / 3.0 * np.eye(3)


def von_mises_stress(sigma) -> float:
    """``sigma_eq = sqrt(3/2 s_ij s_ij)``."""
    s = deviatoric_stress(sigma)
    return math.sqrt(1.5 * float(np.sum(s * s)))


def glen_strain_rate(sigma, a: float, n: float, *,
                     normalisation: str = "norton") -> np.ndarray:
    """Glen flow law ``edot_ij = c A sigma_eq^(n-1) s_ij``.

    ``normalisation="norton"`` (default) uses ``c = 3/2`` so the uniaxial limit
    is exactly ``A sigma^n``; ``"glaciological"`` uses ``c = 1``.
    """
    if normalisation not in ("norton", "glaciological"):
        raise ValueError("normalisation must be 'norton' or 'glaciological'")
    s = deviatoric_stress(sigma)
    seq = math.sqrt(1.5 * float(np.sum(s * s)))
    if seq == 0.0 or a == 0.0:
        return np.zeros((3, 3))
    c = 1.5 if normalisation == "norton" else 1.0
    return c * a * seq ** (n - 1.0) * s


def equivalent_strain_rate(rate) -> float:
    """von Mises equivalent strain rate ``sqrt(2/3 edot:edot)``."""
    rate = np.asarray(rate, dtype=float)
    return math.sqrt(2.0 / 3.0 * float(np.sum(rate * rate)))


def norton_uniaxial_rate(sigma, a: float, n: float):
    """Norton power law ``edot = A sign(sigma) |sigma|^n`` (tension positive)."""
    sigma = np.asarray(sigma, dtype=float)
    return a * np.sign(sigma) * np.abs(sigma) ** n


def arrhenius_fluidity_ratio(q_over_r: float, t_k: float, t_ref_k: float) -> float:
    """``A(T)/A(T_ref) = exp(-Q/R (1/T - 1/T_ref))``."""
    return float(np.exp(-q_over_r * (1.0 / t_k - 1.0 / t_ref_k)))


@dataclass(frozen=True)
class GlenCreepLaw:
    """Temperature dependent Norton/Glen fluidity law.

    Defaults: ``n = 3``, ``A(-10 C) = 3.5e-24 Pa^-3 s^-1`` (effective viscosity
    of order 1e13..1e14 Pa s at 0.1 MPa: the textbook ice range) and
    ``Q/R = 7.4e3 K`` (``Q = 61.5 kJ/mol``).
    """

    a_ref: float = 3.5e-24
    n: float = 3.0
    q_over_r: float = 7.4e3
    t_ref_c: float = -10.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.a_ref) and self.a_ref > 0.0):
            raise ValueError("a_ref must be finite and positive")
        if not (math.isfinite(self.n) and self.n > 0.0):
            raise ValueError("n must be finite and positive")
        if not (math.isfinite(self.q_over_r) and self.q_over_r >= 0.0):
            raise ValueError("q_over_r must be finite and non-negative")

    def fluidity(self, t_c: float = -10.0) -> float:
        """``A(T)`` in ``Pa^-n s^-1`` at Celsius temperature ``t_c``."""
        return self.a_ref * arrhenius_fluidity_ratio(
            self.q_over_r, t_c + T0_K, self.t_ref_c + T0_K)

    def uniaxial_rate(self, sigma, t_c: float = -10.0):
        return norton_uniaxial_rate(sigma, self.fluidity(t_c), self.n)

    def strain_rate_tensor(self, sigma, t_c: float = -10.0) -> np.ndarray:
        return glen_strain_rate(sigma, self.fluidity(t_c), self.n)


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def power_law_fit(x, y, *, x_min: float | None = None,
                  x_max: float | None = None) -> dict:
    """Least squares fit of ``ln y = p ln x + c`` over a window."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    if mask.sum() < 2:
        raise ValueError("not enough positive samples in the fit window")
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    design = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    residual = ly - design @ coef
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "exponent": float(coef[0]),
        "prefactor": float(math.exp(coef[1])),
        "r2": r2,
        "points": int(mask.sum()),
        "x_min": float(x[mask].min()),
        "x_max": float(x[mask].max()),
    }


def identify_creep_stages(t, strain_rate, *, plateau_tolerance: float = 0.05,
                          tertiary_factor: float = 2.0) -> dict:
    """Classify a constant-stress creep record into primary/steady/tertiary.

    ``primary``  = leading monotone decreasing part of ``edot(t)``;
    ``steady``   = window around the minimum rate where
                   ``|d ln edot / d ln t| <= plateau_tolerance``;
    ``tertiary`` = everything after the rate has risen to
                   ``tertiary_factor`` times the steady rate.
    """
    t = np.asarray(t, dtype=float)
    rate = np.asarray(strain_rate, dtype=float)
    if t.shape != rate.shape or t.size < 8:
        raise ValueError("t and strain_rate must be equal-length arrays (>=8)")
    positive = rate > 0.0
    i_min = int(np.argmin(np.where(positive, rate, np.inf)))
    steady_rate = float(rate[i_min])
    t_min = float(t[i_min])

    positive_t = t > 0.0
    log_t = np.log(np.where(positive_t, t, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.gradient(np.log(np.where(positive, rate, np.nan)),
                            log_t, edge_order=1)
    lo = i_min
    while lo - 1 >= 0 and abs(slope[lo - 1]) <= plateau_tolerance:
        lo -= 1
    hi = i_min
    while hi + 1 < t.size and abs(slope[hi + 1]) <= plateau_tolerance:
        hi += 1
    steady_start = float(t[lo])
    steady_end = float(t[hi])

    # tertiary is only searched *after* the steady window: before it, the
    # primary (Kelvin) peak is also above the threshold
    above = np.nonzero(rate[i_min:] >= tertiary_factor * steady_rate)[0]
    tertiary_start = float(t[above[0] + i_min]) if above.size else None
    return {
        "steady_rate": steady_rate,
        "steady_time": t_min,
        "steady_window": (steady_start, steady_end),
        "steady_window_decades": (float(math.log10(steady_end / steady_start))
                                  if steady_start > 0 else 0.0),
        "primary_end_time": steady_start,
        "tertiary_start_time": tertiary_start,
        "peak_rate": float(rate.max()),
        "rate_ratio_peak_steady": float(rate.max() / steady_rate),
    }


# --------------------------------------------------------------------------- #
# Material-point (uniaxial) model: primary + steady + tertiary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UniaxialCreepModel:
    """Constant-stress uniaxial creep with the three classical stages.

    Strain decomposition (``sigma0`` held constant)::

        eps(t) = sigma0/E                                    elastic
               + (sigma0/(f_K E)) (1 - exp(-t/tau_K))        primary (Kelvin)
               + creep integral of A sigma_eff^n             steady / tertiary

    with ``sigma_eff = sigma0/(1-D)`` and Rabotnov damage
    ``dD/dt = (1/t_d) (sigma_eff/sigma_ref)^m``, closed form

        t_r = t_d (sigma_ref/sigma0)^m / (m+1),   1-D = (1 - t/t_r)^(1/(m+1)),

    so ``eps_creep(t) = A sigma0^n t_r (m+1)/(m+1-n) [1 - (1-t/t_r)^((m+1-n)/(m+1))]``.
    :meth:`strain` returns this closed form; :meth:`integrate_rk4` integrates
    the ODE numerically, which is how the implementation is verified.
    """

    law: GlenCreepLaw = GlenCreepLaw()
    temperature_c: float = -10.0
    young_modulus: float = 9.0e9
    kelvin_fraction: float = 0.35        # k_K = kelvin_fraction * E
    kelvin_time_s: float = 100.0
    damage_exponent: float = 4.0         # m
    damage_time_s: float = 5.0e5         # t_d
    damage_stress_fraction: float = 1.0  # sigma_ref = fraction * sigma0

    def time_to_rupture(self, sigma0: float) -> float:
        m = self.damage_exponent
        return (self.damage_time_s / (m + 1.0)
                * self.damage_stress_fraction ** m)

    def damage(self, t, sigma0: float):
        m = self.damage_exponent
        tr = self.time_to_rupture(sigma0)
        frac = np.clip(1.0 - np.asarray(t, dtype=float) / tr, 0.0, None)
        return 1.0 - frac ** (1.0 / (m + 1.0))

    def strain(self, t, sigma0: float):
        """Closed-form strain history (``t < t_rupture``)."""
        t = np.asarray(t, dtype=float)
        a = self.law.fluidity(self.temperature_c)
        n = self.law.n
        m = self.damage_exponent
        tr = self.time_to_rupture(sigma0)
        elastic = sigma0 / self.young_modulus
        if self.kelvin_fraction > 0.0:
            delayed = (sigma0 / (self.kelvin_fraction * self.young_modulus)
                       * (1.0 - np.exp(-t / self.kelvin_time_s)))
        else:
            delayed = np.zeros_like(t)
        exponent = (m + 1.0 - n) / (m + 1.0)
        frac = np.clip(1.0 - t / tr, 1e-300, None)
        if abs(exponent) < 1e-12:
            creep = a * sigma0 ** n * tr * (m + 1.0) * (-np.log(frac))
        else:
            creep = (a * sigma0 ** n * tr * (m + 1.0) / (m + 1.0 - n)
                     * (1.0 - frac ** exponent))
        return elastic + delayed + creep

    def strain_rate(self, t, sigma0: float):
        t = np.asarray(t, dtype=float)
        a = self.law.fluidity(self.temperature_c)
        n = self.law.n
        m = self.damage_exponent
        tr = self.time_to_rupture(sigma0)
        frac = np.clip(1.0 - t / tr, 1e-12, None)
        creep_rate = a * (sigma0 / frac ** (1.0 / (m + 1.0))) ** n
        if self.kelvin_fraction > 0.0:
            delayed_rate = (sigma0 / (self.kelvin_fraction * self.young_modulus
                                      * self.kelvin_time_s)
                            * np.exp(-t / self.kelvin_time_s))
        else:
            delayed_rate = np.zeros_like(t)
        return creep_rate + delayed_rate

    def integrate_rk4(self, sigma0: float, t_end: float, n_steps: int) -> dict:
        """Numerically integrate ``(eps, D)`` with fixed RK4 steps."""
        dt = t_end / n_steps
        a = self.law.fluidity(self.temperature_c)
        n = self.law.n
        m = self.damage_exponent
        sigma_ref = self.damage_stress_fraction * sigma0
        t = 0.0
        eps = 0.0
        dmg = 0.0
        ts, epss, dmgs = [0.0], [0.0], [0.0]
        for _ in range(n_steps):
            def rhs(tt, d_):
                kelvin_rate = 0.0
                if self.kelvin_fraction > 0.0:
                    kelvin_rate = (sigma0 / (self.kelvin_fraction
                                             * self.young_modulus
                                             * self.kelvin_time_s)
                                   * math.exp(-tt / self.kelvin_time_s))
                sigma_eff = sigma0 / max(1.0 - d_, 1e-12)
                d_eps = a * sigma_eff ** n + kelvin_rate
                d_d = (1.0 / self.damage_time_s) * (sigma_eff / sigma_ref) ** m
                return d_eps, d_d

            k1 = rhs(t, dmg)
            k2 = rhs(t + dt / 2, dmg + dt / 2 * k1[1])
            k3 = rhs(t + dt / 2, dmg + dt / 2 * k2[1])
            k4 = rhs(t + dt, dmg + dt * k3[1])
            eps += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            dmg += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += dt
            ts.append(t)
            epss.append(eps)
            dmgs.append(dmg)
            if dmg >= 1.0:
                break
        return {"t": np.array(ts), "strain": np.array(epss),
                "damage": np.array(dmgs)}


# --------------------------------------------------------------------------- #
# Bonded-sphere lattice with Norton creep bonds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatticeCreepParameters:
    """Material/algorithm parameters of the bonded-sphere creep lattice."""

    law: GlenCreepLaw = GlenCreepLaw()
    temperature_c: float = -10.0
    young_modulus: float = 9.0e9
    shear_to_axial: float = 0.4
    kelvin_fraction: float = 0.0
    kelvin_time_s: float = 100.0
    damage_exponent: float = 0.0
    damage_time_s: float = 1.0e12
    damage_stress_fraction: float = 1.0

    def fluidity(self) -> float:
        return self.law.fluidity(self.temperature_c)


def cubic_lattice_block(nx: int, ny: int, nz: int, *, spacing: float = 0.02,
                        diagonals: bool = True):
    """Simple-cubic particle block with axial + face-diagonal bonds.

    The face diagonals are *required*: axis-aligned bonds alone give zero shear
    stiffness, hence a singular global stiffness matrix.
    """
    if min(nx, ny, nz) < 2:
        raise ValueError("need at least 2 particles per direction")
    grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    positions = grid * spacing
    cut = spacing * (1.45 if diagonals else 1.05)
    bonds = []
    n = len(positions)
    for i in range(n):
        d = positions[i + 1:] - positions[i]
        dist = np.linalg.norm(d, axis=1)
        for offset in np.nonzero(dist <= cut)[0]:
            bonds.append((i, i + 1 + int(offset)))
    return positions, np.array(bonds, dtype=int)


def single_bond_relaxation_analytic(f0: float, k_axial: float, area: float,
                                    rest: float, a: float, n: float, t):
    """Fixed-elongation relaxation of one Norton bond (closed form).

    ``F = F0 exp(-B t)`` for ``n = 1`` and ``F = (F0^(1-n) + (n-1) B t)^(-1/(n-1))``
    for ``n > 1``, with ``B = k A L0 A_b^(-n)``.
    """
    t = np.asarray(t, dtype=float)
    b = k_axial * a * rest / area ** n
    if abs(n - 1.0) < 1e-12:
        return f0 * np.exp(-b * t)
    return (f0 ** (1.0 - n) + (n - 1.0) * b * t) ** (-1.0 / (n - 1.0))


class BondedSphereCreepLattice:
    """Bonded-sphere DEM lattice with Norton creep, Kelvin branch and damage.

    Quasi-static formulation: at each creep step the mechanical equilibrium
    ``K u = f_ext + f_eigen(u_creep, u_kelvin)`` is solved, then the internal
    variables are advanced explicitly.  Removing inertia is what makes a 1e4 s
    creep test reachable.
    """

    def __init__(self, positions, bonds, radius: float,
                 params: LatticeCreepParameters | None = None, *,
                 penalty_scale: float = 1.0e6):
        self.pos = np.asarray(positions, dtype=float)
        self.bonds = np.asarray(bonds, dtype=int)
        self.params = params or LatticeCreepParameters()
        self.radius = float(radius)
        self.n_nodes = len(self.pos)
        self.i = self.bonds[:, 0]
        self.j = self.bonds[:, 1]
        delta = self.pos[self.j] - self.pos[self.i]
        self.rest = np.linalg.norm(delta, axis=1)
        self.unit = delta / self.rest[:, None]
        self.area = math.pi * self.radius ** 2 * np.ones(len(self.rest))
        self.k_axial = self.params.young_modulus * self.area / self.rest
        self.k_shear = self.params.shear_to_axial * self.k_axial
        self.k_kelvin = self.params.kelvin_fraction * self.k_axial

        self.c_creep = np.zeros(len(self.rest))
        self.c_kelvin = np.zeros(len(self.rest))
        self.damage = np.zeros(len(self.rest))

        self.fixed = np.zeros(3 * self.n_nodes, dtype=bool)
        self.fixed_value = np.zeros(3 * self.n_nodes)
        self.external = np.zeros(3 * self.n_nodes)
        self.u = np.zeros((self.n_nodes, 3))
        self.time = 0.0
        self._K = None
        self._lu = None
        self.penalty_scale = penalty_scale
        self._penalty = None
        self._sigma_ref = None

    # -- assembly ---------------------------------------------------------- #
    def _stiffness(self) -> np.ndarray:
        n = self.n_nodes
        K = np.zeros((3 * n, 3 * n))
        keep = 1.0 - self.damage
        for b in range(len(self.rest)):
            unit = self.unit[b]
            axial = keep[b] * (self.k_axial[b] + self.k_kelvin[b])
            shear = keep[b] * self.k_shear[b]
            block = (axial * np.outer(unit, unit)
                     + shear * (np.eye(3) - np.outer(unit, unit)))
            si = slice(3 * self.i[b], 3 * self.i[b] + 3)
            sj = slice(3 * self.j[b], 3 * self.j[b] + 3)
            K[si, si] += block
            K[sj, sj] += block
            K[si, sj] -= block
            K[sj, si] -= block
        return K

    def _eigen_force(self) -> np.ndarray:
        keep = 1.0 - self.damage
        magnitude = keep * (self.k_axial * self.c_creep
                            + self.k_kelvin * self.c_kelvin)
        f = np.zeros((self.n_nodes, 3))
        vec = magnitude[:, None] * self.unit
        np.add.at(f, self.j, vec)
        np.add.at(f, self.i, -vec)
        return f.reshape(-1)

    def _ensure_stiffness(self) -> None:
        if self._K is None:
            K = self._stiffness()
            self._penalty = self.penalty_scale * float(
                np.max(np.abs(np.diag(K))))
            if self.fixed.any():
                idx = np.nonzero(self.fixed)[0]
                K[idx, idx] += self._penalty
            self._K = K
            # the stiffness matrix does not change while the damage field is
            # frozen, so one factorisation serves every creep step
            self._lu = _lu_factor(K) if _lu_factor is not None else None

    def invalidate(self) -> None:
        """Force re-assembly (call after changing damage by hand)."""
        self._K = None
        self._lu = None

    # -- equilibrium ------------------------------------------------------- #
    def solve(self) -> np.ndarray:
        self._ensure_stiffness()
        rhs = self.external + self._eigen_force()
        if self.fixed.any():
            idx = np.nonzero(self.fixed)[0]
            rhs[idx] += self._penalty * self.fixed_value[idx]
        if self._lu is not None:
            u = _lu_solve(self._lu, rhs)
        else:                          # pragma: no cover - numpy fallback
            u = np.linalg.solve(self._K, rhs)
        self.u = u.reshape(self.n_nodes, 3)
        return self.u

    def reaction(self) -> np.ndarray:
        """Penalty reactions (positive = force the constraint applies on nodes)."""
        react = np.zeros(3 * self.n_nodes)
        if self.fixed.any():
            idx = np.nonzero(self.fixed)[0]
            u = self.u.reshape(-1)
            react[idx] = self._penalty * (u[idx] - self.fixed_value[idx])
        return react.reshape(self.n_nodes, 3)

    # -- bond state -------------------------------------------------------- #
    def bond_extension(self) -> np.ndarray:
        delta = self.u[self.j] - self.u[self.i]
        return np.einsum("ij,ij->i", delta, self.unit)

    def bond_axial_force(self) -> np.ndarray:
        d = self.bond_extension()
        keep = 1.0 - self.damage
        return keep * (self.k_axial * (d - self.c_creep)
                       + self.k_kelvin * (d - self.c_kelvin))

    def bond_stress(self) -> np.ndarray:
        return self.bond_axial_force() / self.area

    def step(self, dt: float) -> None:
        """Advance the internal variables by ``dt`` at frozen equilibrium."""
        a = self.params.fluidity()
        n = self.params.law.n
        d = self.bond_extension()
        sigma = self.bond_stress()
        # Norton dashpot: Delta u_creep = A sigma^n Delta t * L0
        self.c_creep += np.sign(sigma) * a * np.abs(sigma) ** n * self.rest * dt
        # Kelvin branch (exact exponential update)
        if self.params.kelvin_fraction > 0.0:
            lam = 1.0 - math.exp(-dt / self.params.kelvin_time_s)
            self.c_kelvin += (d - self.c_kelvin) * lam
        # Kachanov-Rabotnov damage
        m = self.params.damage_exponent
        if m > 0:
            if self._sigma_ref is None:
                self._sigma_ref = (self.params.damage_stress_fraction
                                   * max(float(np.abs(sigma).max()), 1e-30))
            sigma_ref = self._sigma_ref
            keep = np.clip(1.0 - self.damage, 1e-12, None)
            self.damage += dt * (np.abs(sigma) / (sigma_ref * keep)) ** m \
                / self.params.damage_time_s
            self.damage = np.clip(self.damage, 0.0, 1.0 - 1e-12)
            self.invalidate()
        self.time += dt

    def solve_stress_reference(self) -> None:
        """Fix the damage reference stress from the current bond stresses."""
        self._sigma_ref = None

    def creep_step_ratio(self, dt: float) -> float:
        """``max(Delta u_creep) / max(elastic bond stretch)`` for a step ``dt``.

        Explicit integration is only trustworthy while this number is well
        below one; it is the lattice counterpart of ``dt << 1/(A sigma^(n-1) E)``.
        """
        sigma = self.bond_stress()
        d = self.bond_extension()
        increment = (self.params.fluidity() * np.abs(sigma) ** self.params.law.n
                     * self.rest * dt)
        scale = max(float(np.abs(d - self.c_creep).max()), 1e-30)
        return float(increment.max() / scale)

    def run(self, dt: float, n_steps: int, *, record=None) -> dict:
        """Integrate; record ``record(state)`` at every equilibrium state."""
        series = []
        self.solve()
        if record is not None:
            series.append(record(self))
        for _ in range(n_steps):
            self.step(dt)
            self.solve()
            if record is not None:
                series.append(record(self))
        return {"series": series, "time": self.time}


# --------------------------------------------------------------------------- #
# Driver 1: uniaxial constant-stress creep of a cubic bonded specimen
# --------------------------------------------------------------------------- #
def uniaxial_constant_stress(*, sigma0: float, n_cells: int = 4,
                             spacing: float = 0.02,
                             params: LatticeCreepParameters | None = None,
                             t_end: float = 2000.0, dt: float = 10.0,
                             stress_control: bool = True) -> dict:
    """Constant axial stress ``sigma0`` on a free-standing cubic specimen.

    Bottom face: ``u_z = 0`` (plus one node pinned laterally to remove rigid
    body modes).  Top face: uniform traction ``sigma0`` applied as nodal forces,
    so the *lateral faces stay traction free* (uniaxial stress, not uniaxial
    strain).

    With ``stress_control`` (default) the applied nodal force is re-scaled at
    every step to keep the measured axial reaction equal to
    ``sigma0 * A_cross``: that removes the load relaxation of the Maxwell bonds
    and turns the test into a genuine constant-stress (Norton) creep test, in
    which ``edot`` is the constitutive rate at all times.
    """
    params = params or LatticeCreepParameters()
    positions, bonds = cubic_lattice_block(n_cells, n_cells, n_cells,
                                          spacing=spacing)
    lat = BondedSphereCreepLattice(positions, bonds, radius=spacing / 2.0,
                                   params=params)
    zmax = positions[:, 2].max()
    zmin = positions[:, 2].min()
    height = float(zmax - zmin)
    top = np.nonzero(np.isclose(positions[:, 2], zmax))[0]
    bottom = np.nonzero(np.isclose(positions[:, 2], zmin))[0]

    lat.fixed[3 * bottom + 2] = True          # u_z = 0 on the bottom face
    lat.fixed[0] = True                       # node 0: u_x = 0
    lat.fixed[1] = True                       # node 0: u_y = 0
    top_dof = 3 * top + 2
    bottom_dof = 3 * bottom + 2
    cross_section = (float(positions[:, 0].max() - positions[:, 0].min())
                     * float(positions[:, 1].max() - positions[:, 1].min()))
    target = sigma0 * cross_section
    lat.external[top_dof] = -target / len(top)

    def record(state):
        return {
            "t": state.time,
            "strain": -float(np.mean(state.u[top, 2])) / height,
            "axial_force": float(-state.reaction()[bottom, 2].sum()),
            "creep_ratio": state.creep_step_ratio(dt),
        }

    series = []
    lat.solve()
    series.append(record(lat))
    for _ in range(int(round(t_end / dt))):
        lat.step(dt)
        lat.solve()
        if stress_control:
            react = float(-lat.reaction()[bottom, 2].sum())
            if react > 0.0:
                lat.external[top_dof] *= target / react
        series.append(record(lat))

    t = np.array([s["t"] for s in series])
    strain = np.array([s["strain"] for s in series])
    return {
        "t": t,
        "strain": strain,
        "axial_force": np.array([s["axial_force"] for s in series]),
        "creep_ratio": np.array([s["creep_ratio"] for s in series]),
        "damage_max": lat.damage.copy(),
        "lattice": lat,
        "height": height,
        "cross_section": cross_section,
        "n_nodes": lat.n_nodes,
        "n_bonds": len(bonds),
        "elastic_strain_expected": sigma0 / params.young_modulus,
    }


# --------------------------------------------------------------------------- #
# Driver 2: rigid plate (flat punch) relaxation at constant penetration
# --------------------------------------------------------------------------- #
def punch_relaxation(*, n_cells: int = 6, spacing: float = 0.02,
                     params: LatticeCreepParameters | None = None,
                     depth_fraction: float = 0.005,
                     t_end: float = 2000.0, dt: float = 10.0) -> dict:
    """Rigid plate pressed ``depth = depth_fraction * height`` into the block
    and then held at constant penetration; returns the contact force history.

    The plate constrains only ``u_z`` on the top face (frictionless, rigid);
    ``u_x, u_y`` stay free so the top face can creep laterally.  The bottom
    face is fully clamped in ``u_z`` and the block is laterally free.
    """
    params = params or LatticeCreepParameters()
    positions, bonds = cubic_lattice_block(n_cells, n_cells, n_cells,
                                           spacing=spacing)
    lat = BondedSphereCreepLattice(positions, bonds, radius=spacing / 2.0,
                                   params=params)
    zmax = positions[:, 2].max()
    zmin = positions[:, 2].min()
    height = float(zmax - zmin)
    top = np.nonzero(np.isclose(positions[:, 2], zmax))[0]
    bottom = np.nonzero(np.isclose(positions[:, 2], zmin))[0]
    depth = depth_fraction * height

    lat.fixed[3 * bottom + 1] = False
    lat.fixed[3 * bottom + 2] = True
    lat.fixed[3 * top + 2] = True
    lat.fixed_value[3 * top + 2] = -depth
    lat.fixed[0] = True
    lat.fixed[1] = True
    lat.solve()
    f0 = float(np.abs(lat.reaction()[top, 2]).sum())

    def record(state):
        return {
            "t": state.time,
            "force": float(np.abs(state.reaction()[top, 2]).sum()),
            "creep_ratio": state.creep_step_ratio(dt),
            "compressive_bond_force": float(
                np.abs(state.bond_axial_force()).max()),
            "damage_max": float(state.damage.max()),
        }

    out = lat.run(dt, int(round(t_end / dt)), record=record)
    series = out["series"]
    return {
        "t": np.array([s["t"] for s in series]),
        "force": np.array([s["force"] for s in series]),
        "damage_max": np.array([s["damage_max"] for s in series]),
        "creep_ratio": np.array([s["creep_ratio"] for s in series]),
        "initial_force": f0,
        "depth": depth,
        "height": height,
        "lattice": lat,
        "n_nodes": lat.n_nodes,
        "n_bonds": len(bonds),
    }