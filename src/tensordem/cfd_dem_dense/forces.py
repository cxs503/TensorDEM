"""Inter-phase force closures for *dense* unresolved (point-particle) CFD--DEM.

Scope and conventions
---------------------
Everything here is stated for a **bed-volume force density** ``f``
[N/m^3 of bed] and the equivalent **per-particle force** ``F`` [N], with the
dictionary

    f = beta(u_rel, eps) * u_rel,          F_particle = f * V_p / phi

where ``phi = 1 - eps`` is the solid fraction and

    u_rel = u_superficial - phi * v_particle

is the *superficial* relative velocity (empty-column basis).  This bookkeeping
is the most error-prone part of dense coupling: the classic Ergun relation is
written in terms of the superficial velocity of a *static* bed, and the
per-particle form differs from it by a factor ``eps`` (compare
:func:`ergun_gradient` with :func:`beta` for ``model="ergun"``).

Model A momentum balance (what :func:`fluid_pressure_gradient` returns)
----------------------------------------------------------------------
The two-fluid equations used throughout this package are (``z`` up, gravity
``-g zhat``, quasi-steady fluid, no fluid inertia)::

    fluid:   0 = -eps dp/dz - eps rho_f g - f
    solid:   rho_p phi dv/dt = f - phi dp/dz - phi rho_p g - d(sigma_c)/dz

Eliminating ``dp/dz`` gives the *combined* solid force density

    rho_p phi dv/dt = f/eps - phi (rho_p - rho_f) g - d(sigma_c)/dz ,

so the pressure-gradient force is not an optional extra: with the Ergun/Wen--Yu
family ``f`` already carries the voidage (buoyancy) correction.  Adding a
separate Archimedes term on top of ``f`` double counts the buoyancy, which is
what the ``pressure_gradient=False`` switch is used to expose as negative
control NC3.

Closures provided
-----------------
``ergun``            Ergun (1952) packed-bed form (dense branch).
``wen_yu``           Schiller--Naumann drag with the Wen--Yu voidage factor
                     ``eps**-2.65`` (dilute branch of Gidaspow's blend).
``gidaspow``         Ergun for ``eps < 0.8``, Wen--Yu above (Gidaspow 1994).
``di_felice``        Di Felice (1994): ``0.5 rho_f C_D |u| u A eps**-chi``,
                     ``chi = 3.7 - 0.65 exp[-(1.5 - log10 Re)^2 / 2]``.
``schiller_naumann`` Single-sphere Schiller--Naumann drag, *no* voidage
                     correction (``eps`` exponent 0): dilute-limit control.

Every closure is individually switchable through :class:`ForceModel`; every
switch has a documented physical meaning and a test.

All quantities SI.  Nothing here carries ship, ice or other domain constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "GRAVITY",
    "GIDASPOW_SWITCH",
    "DRAG_MODELS",
    "WEN_YU_EXPONENT",
    "ForceModel",
    "DEFAULT_DENSE",
    "sphere_volume",
    "sphere_projected_area",
    "schiller_naumann_cd",
    "newton_cd",
    "di_felice_chi",
    "voidage_exponent",
    "beta",
    "drag_density",
    "per_particle_drag",
    "ergun_gradient",
    "ergun_minimum_fluidization_velocity",
    "fluid_pressure_gradient",
    "pressure_gradient_force",
    "added_mass_force",
    "buoyant_weight",
    "terminal_velocity",
    "equilibrium_superficial_velocity",
    "linear_deposit_weights",
    "momentum_exchange_audit",
]

GRAVITY = 9.80665
GIDASPOW_SWITCH = 0.8
WEN_YU_EXPONENT = 2.65
DRAG_MODELS = ("ergun", "wen_yu", "gidaspow", "di_felice", "schiller_naumann")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def sphere_volume(diameter: float) -> float:
    """Volume of a sphere [m^3]."""
    return math.pi * diameter**3 / 6.0


def sphere_projected_area(diameter: float) -> float:
    """Projected (frontal) area of a sphere [m^2]."""
    return math.pi * diameter**2 / 4.0


# ---------------------------------------------------------------------------
# single-particle drag coefficients
# ---------------------------------------------------------------------------
def schiller_naumann_cd(reynolds, cd_scale: float = 1.0):
    """Schiller--Naumann (1935) ``C_D = 24/Re (1 + 0.15 Re^0.687)``.

    Valid for ``Re < 800`` (a few percent above measured sphere drag); the
    Stokes limit ``24/Re`` is returned as ``Re -> 0`` without dividing by zero.
    ``cd_scale`` multiplies the coefficient (doubled-drag negative control).
    """
    re = np.maximum(np.asarray(reynolds, dtype=float), 1e-300)
    cd = 24.0 / re * (1.0 + 0.15 * re**0.687)
    return cd_scale * cd


def newton_cd(cd: float = 0.44, cd_scale: float = 1.0) -> float:
    """Newton-regime constant drag coefficient (inertial limit, ``Re > 1000``)."""
    return cd_scale * cd


def di_felice_chi(reynolds):
    """Di Felice (1994) voidage exponent ``chi(Re)``.

    ``chi = 3.7 - 0.65 exp[-(1.5 - log10 Re)^2 / 2]``; it ranges from ~3.13
    near ``Re ~ 10`` to ~3.7 at both very low and very high ``Re``.  The
    Richardson--Zaki index implied by the *combined* Model-A force balance is
    ``2 + chi`` (see :func:`equilibrium_superficial_velocity`).
    """
    re = np.maximum(np.asarray(reynolds, dtype=float), 1e-300)
    return 3.7 - 0.65 * np.exp(-((1.5 - np.log10(re)) ** 2) / 2.0)


def voidage_exponent(model: str, reynolds):
    """Voidage exponent ``chi`` in ``eps**-chi`` for the requested closure."""
    if model in ("ergun", "gidaspow"):
        raise ValueError("Ergun/Gidaspow do not reduce to a single voidage exponent")
    if model == "wen_yu":
        return WEN_YU_EXPONENT
    if model == "di_felice":
        return di_felice_chi(reynolds)
    if model == "schiller_naumann":
        return 0.0
    raise ValueError(f"unknown drag model {model!r}")


# ---------------------------------------------------------------------------
# switch panel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForceModel:
    """Switch panel for the dense inter-phase force assembly.

    Parameters
    ----------
    drag_model : str
        One of :data:`DRAG_MODELS`.
    voidage_exponent : float or None
        Override for ``chi``; ``None`` keeps the closure's native exponent.
    pressure_gradient : bool
        Include the resolved pressure-gradient force ``-V_p grad p`` (Model A).
        Must stay ``True`` for the dense closures, where it carries the
        buoyancy; ``False`` is negative control NC3.
    added_mass : bool
        Include ``C_am rho_f V_p (Du/Dt - dv/dt)`` (Maxey--Riley form).
    added_mass_coefficient : float
        ``C_am = 0.5`` for a sphere in potential flow.
    fluid_reaction : bool
        Deposit the equal-and-opposite force on the fluid (Newton's third law).
        ``False`` is negative control NC1.
    cd_scale : float
        Multiplier on the drag coefficient (``2.0`` = NC2).
    cd_model : str
        ``"closure"`` (Re-dependent) or ``"newton"`` (constant ``C_D = 0.44``,
        the inertial-limit law used by NC2).
    """

    drag_model: str = "gidaspow"
    voidage_exponent: float | None = None
    pressure_gradient: bool = True
    added_mass: bool = True
    added_mass_coefficient: float = 0.5
    fluid_reaction: bool = True
    cd_scale: float = 1.0
    cd_model: str = "closure"

    def __post_init__(self) -> None:
        if self.drag_model not in DRAG_MODELS:
            raise ValueError(f"drag_model must be one of {DRAG_MODELS}, got {self.drag_model!r}")
        if self.cd_model not in ("closure", "newton"):
            raise ValueError("cd_model must be 'closure' or 'newton'")
        if self.cd_scale <= 0.0:
            raise ValueError("cd_scale must be positive")
        if self.added_mass and self.added_mass_coefficient < 0.0:
            raise ValueError("added_mass_coefficient must be non-negative")

    def replace(self, **kwargs) -> "ForceModel":
        """Return a copy with switches changed (the model stays immutable)."""
        return replace(self, **kwargs)

    @property
    def double_counting_risk(self) -> bool:
        """True when an explicit ``grad p`` is added on top of a dense closure."""
        return self.pressure_gradient and self.drag_model in ("ergun", "gidaspow")

    def describe(self) -> dict:
        """JSON-friendly description recorded in the study provenance block."""
        return {
            "drag_model": self.drag_model,
            "voidage_exponent": self.voidage_exponent,
            "pressure_gradient": self.pressure_gradient,
            "added_mass": self.added_mass,
            "added_mass_coefficient": self.added_mass_coefficient,
            "fluid_reaction": self.fluid_reaction,
            "cd_scale": self.cd_scale,
            "cd_model": self.cd_model,
            "double_counting_risk": self.double_counting_risk,
        }


DEFAULT_DENSE = ForceModel(drag_model="gidaspow")


# ---------------------------------------------------------------------------
# inter-phase coefficient beta
# ---------------------------------------------------------------------------
def _cd_of_re(reynolds, model: ForceModel):
    if model.cd_model == "newton":
        return newton_cd(cd_scale=model.cd_scale)
    return schiller_naumann_cd(reynolds, cd_scale=model.cd_scale)


def beta(
    u_rel,
    eps,
    *,
    diameter: float,
    rho_fluid: float,
    mu: float,
    model: "ForceModel | str" = DEFAULT_DENSE,
):
    """Inter-phase coefficient ``beta`` [kg/(m^3 s)] with ``f = beta u_rel``.

    The dilute closures are written as

        f = (phi / V_p) 0.5 rho_f C_D(Re_p) |u_rel| u_rel A_p eps**-chi,
        Re_p = rho_f |u_rel| d eps / mu,

    algebraically identical to Gidaspow's ``beta = 0.75 C_D rho_f phi |u_rel|
    eps**-chi / d``.  The dense closures use the Ergun grouping
    ``phi^2/eps^2`` (viscous) and ``phi/eps`` (inertial), i.e.
    ``beta_ergun = eps * beta_classic``; that factor ``eps`` is what makes the
    Model-A force balance reproduce the textbook minimum fluidisation velocity.
    """
    fm = model if isinstance(model, ForceModel) else ForceModel(drag_model=model)
    u = np.asarray(u_rel, dtype=float)
    e = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    phi = 1.0 - e
    speed = np.abs(u)
    re_p = e * rho_fluid * np.maximum(speed, 1e-300) * diameter / mu

    with np.errstate(divide="ignore", invalid="ignore"):
        # Model-A Ergun group: beta = eps * (classic porous-media coefficient), so
        # that f = beta u gives -dp/dz|Ergun = classic u and the Model-A force
        # balance reproduces the textbook u_mf at every Re.  This puts eps^2 in
        # the inertial grouping; the widely quoted Gidaspow (1994) form uses
        # eps*d there (a factor eps difference in the inertial term).
        ergun = (
            150.0 * phi**2 * mu / (e**2 * diameter**2)
            + 1.75 * phi * rho_fluid * speed / (e**2 * diameter)
        )

    chi = fm.voidage_exponent
    if chi is None:
        if fm.drag_model == "gidaspow":
            # the dilute branch of Gidaspow's blend is Wen--Yu, so it carries
            # the eps^-2.65 correction
            chi = WEN_YU_EXPONENT
        elif fm.drag_model != "ergun":
            chi = voidage_exponent(fm.drag_model, np.maximum(re_p, 1e-300))
    cd = _cd_of_re(np.maximum(re_p, 1e-300), fm)
    dilute = 0.75 * cd * rho_fluid * phi * speed / diameter
    if chi is not None:
        dilute = dilute * e ** (-np.asarray(chi, dtype=float))

    if fm.drag_model == "ergun":
        out = ergun
    elif fm.drag_model == "gidaspow":
        out = np.where(e < GIDASPOW_SWITCH, ergun, dilute)
    else:
        out = dilute
    return np.asarray(out, dtype=float)


def drag_density(
    u_rel,
    eps,
    *,
    diameter: float,
    rho_fluid: float,
    mu: float,
    model: "ForceModel | str" = DEFAULT_DENSE,
):
    """Drag force per unit bed volume ``f = beta u_rel`` [N/m^3]."""
    return beta(u_rel, eps, diameter=diameter, rho_fluid=rho_fluid, mu=mu, model=model) * np.asarray(
        u_rel, dtype=float
    )


def per_particle_drag(
    u_rel,
    eps,
    *,
    diameter: float,
    rho_fluid: float,
    mu: float,
    model: "ForceModel | str" = DEFAULT_DENSE,
):
    """Drag force on one particle [N]: ``f V_p / phi``."""
    phi = 1.0 - np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    dens = drag_density(u_rel, eps, diameter=diameter, rho_fluid=rho_fluid, mu=mu, model=model)
    return dens * sphere_volume(diameter) / np.maximum(phi, 1e-12)


# ---------------------------------------------------------------------------
# Ergun reference relations (classic porous-media form)
# ---------------------------------------------------------------------------
def ergun_gradient(
    u_superficial: float,
    *,
    diameter: float,
    porosity: float,
    rho_fluid: float,
    mu: float,
) -> dict:
    """Classic Ergun (1952) pressure-gradient magnitude ``-dp/dz`` [Pa/m].

    ``-dp/dz = 150 mu (1-eps)^2 u / (eps^3 d^2) + 1.75 rho_f (1-eps) u^2 / (eps^3 d)``

    This is the *porous-media* form and equals ``eps * beta_ergun * u``; the
    viscous/inertial split is reported because criterion 3 uses one velocity
    at each end of the Re range.
    """
    e = float(porosity)
    if not 0.0 < e < 1.0:
        raise ValueError(f"porosity must lie in (0, 1), got {e}")
    u = abs(float(u_superficial))
    phi = 1.0 - e
    viscous = 150.0 * mu * phi**2 / (e**3 * diameter**2) * u
    inertial = 1.75 * rho_fluid * phi / (e**3 * diameter) * u * u
    total = viscous + inertial
    return {
        "total": total,
        "viscous": viscous,
        "inertial": inertial,
        "inertial_fraction": 0.0 if total == 0.0 else inertial / total,
    }


def ergun_minimum_fluidization_velocity(
    *, diameter: float, porosity: float, rho_particle: float, rho_fluid: float, mu: float
) -> float:
    """Textbook minimum fluidisation velocity from the Ergun inversion [m/s].

    Positive root of
    ``150 mu (1-eps)^2 u/(eps^3 d^2) + 1.75 rho_f (1-eps) u^2/(eps^3 d) =
    (1-eps)(rho_p - rho_f) g``.  Independent closed form; reference for
    criterion 4.
    """
    e = float(porosity)
    phi = 1.0 - e
    a = 1.75 * rho_fluid * phi / (e**3 * diameter)
    b = 150.0 * mu * phi**2 / (e**3 * diameter**2)
    c = -phi * (rho_particle - rho_fluid) * GRAVITY
    return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


# ---------------------------------------------------------------------------
# pressure gradient and added mass
# ---------------------------------------------------------------------------
def fluid_pressure_gradient(eps, drag_dens, *, rho_fluid: float, gravity: float = GRAVITY):
    """Model-A quasi-steady ``dp/dz`` [Pa/m] from the fluid momentum balance.

    ``0 = -eps dp/dz - eps rho_f g - f``  ->  ``dp/dz = -(eps rho_f g + f)/eps``.

    In particle-free cells (``eps = 1``) this reduces to hydrostatic
    ``-rho_f g``.  ``drag_dens`` is the bed-volume drag density ``f``.
    """
    e = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    return -(e * rho_fluid * gravity + np.asarray(drag_dens, dtype=float)) / e


def pressure_gradient_force(volume, grad_p):
    """``F = -V_p grad p`` on a particle [N] (Froude--Krylov / buoyancy)."""
    vol = np.asarray(volume, dtype=float)
    gp = np.asarray(grad_p, dtype=float)
    return -vol * gp


def added_mass_force(volume, rho_fluid: float, du_rel_dt, coefficient: float = 0.5):
    """Maxey--Riley added-mass force ``C_am rho_f V_p (Du/Dt - dv/dt)`` [N].

    ``du_rel_dt`` is the relative acceleration between the undisturbed fluid
    and the particle.  For a particle oscillating in a quiescent fluid this is
    an effective mass ``m + C_am rho_f V_p`` (checked in the test-suite).
    """
    return coefficient * rho_fluid * np.asarray(volume, dtype=float) * np.asarray(
        du_rel_dt, dtype=float
    )


def buoyant_weight(
    diameter: float, *, rho_particle: float, rho_fluid: float, gravity: float = GRAVITY
) -> float:
    """Submerged weight ``V_p (rho_p - rho_f) g`` [N]."""
    return sphere_volume(diameter) * (rho_particle - rho_fluid) * gravity


# ---------------------------------------------------------------------------
# terminal velocity and equilibrium voidage (root finding)
# ---------------------------------------------------------------------------
def _bisect(func, lo: float, hi: float, tol: float = 1e-14, max_iter: int = 400) -> float:
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo
    if f_lo * f_hi > 0.0:
        raise ValueError(f"root not bracketed on [{lo}, {hi}] (f={f_lo:.3e}, {f_hi:.3e})")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0 or abs(hi - lo) <= tol * max(1.0, abs(mid)):
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def terminal_velocity(
    *,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    model: "ForceModel | str" = "schiller_naumann",
    eps: float = 1.0,
) -> float:
    """Terminal velocity [m/s] from ``drag = buoyant weight`` for one sphere.

    The closure is evaluated at the free-stream voidage (``eps = 1`` for a
    single sphere in an unbounded fluid).  Bisection means a doubled
    ``cd_scale`` yields an honest, self-consistent new terminal velocity
    (negative control NC2).
    """
    fm = model if isinstance(model, ForceModel) else ForceModel(drag_model=model)
    weight = buoyant_weight(diameter, rho_particle=rho_particle, rho_fluid=rho_fluid)
    if weight <= 0.0:
        return 0.0
    d = float(diameter)
    area = sphere_projected_area(d)
    chi = fm.voidage_exponent
    if chi is None and fm.drag_model not in ("ergun", "gidaspow"):
        chi = voidage_exponent(fm.drag_model, 1e-30)
    elif chi is None:
        # dense closures have no single-sphere limit; use the dilute branch
        chi = WEN_YU_EXPONENT
    chi = 0.0 if float(eps) >= 1.0 else float(chi)

    def residual(u: float) -> float:
        re = rho_fluid * u * d / mu
        cd = _cd_of_re(max(re, 1e-300), fm)
        drag = 0.5 * rho_fluid * float(cd) * u * u * area * float(eps) ** (-chi)
        return drag - weight

    hi = 1.0
    while residual(hi) < 0.0:
        hi *= 2.0
        if hi > 1e6:  # pragma: no cover
            break
    return _bisect(residual, 1e-12, hi)


def equilibrium_superficial_velocity(
    eps: float,
    *,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    model: "ForceModel | str" = DEFAULT_DENSE,
    gravity: float = GRAVITY,
    hi: float = 10.0,
) -> float:
    """Superficial velocity [m/s] that holds a static bed at voidage ``eps``.

    This is the Model-A equilibrium of a *static* bed element: the combined
    inter-phase force balances the submerged weight,

        pressure_gradient=True :  f = eps phi (rho_p - rho_f) g
        pressure_gradient=False:  f = phi rho_p g          (NC3 variant)

    solved for ``u_rel`` with ``v_p = 0``.  Because ``f = beta(u) u`` with a
    Re-dependent ``beta`` the root is found by bisection rather than by
    inverting a polynomial.
    """
    fm = model if isinstance(model, ForceModel) else ForceModel(drag_model=model)
    e = float(eps)
    if not 0.0 < e < 1.0:
        raise ValueError(f"eps must lie in (0, 1), got {e}")
    phi = 1.0 - e
    if fm.pressure_gradient:
        target = e * phi * (rho_particle - rho_fluid) * gravity
    else:
        target = phi * rho_particle * gravity

    d = float(diameter)

    def residual(u: float) -> float:
        dens = float(
            drag_density(u, e, diameter=d, rho_fluid=rho_fluid, mu=mu, model=fm)
        )
        return dens - target

    return _bisect(residual, 1e-14, hi)


# ---------------------------------------------------------------------------
# fluid-side deposition and action-reaction audit
# ---------------------------------------------------------------------------
def linear_deposit_weights(z_positions, n_cells: int, length: float):
    """Linear (hat) deposition weights from particles to ``n_cells`` cells.

    Returns ``(cell_index_lo, weights)`` such that
    ``deposit[i] += w0 * value + w1 * value`` for the two neighbouring cells
    ``i`` and ``i+1`` (periodic-friendly, weights sum to 1 per particle).
    """
    z = np.asarray(z_positions, dtype=float) % length
    dx = length / n_cells
    s = z / dx - 0.5
    lo = np.floor(s).astype(int)
    w = s - lo
    return lo % n_cells, w


def momentum_exchange_audit(
    *,
    z_particle,
    v_particle,
    n_cells: int,
    length: float,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    model: "ForceModel | str" = DEFAULT_DENSE,
    solid_fraction: float | None = None,
    du_rel_dt=None,
    fluid_velocity: float = 0.0,
    kernel_cells: float = 2.0,
    gravity: float = GRAVITY,
):
    """Action--reaction audit of the coupled force assembly (criterion 1).

    Particles deposit their reaction on the fluid cells with a normalised hat
    kernel; the sum of the cell reactions must equal minus the sum of the
    particle forces to roundoff.  The residual is normalised by ``sum |F|``
    (not by ``|sum F|``, which is a near-cancelling difference at equilibrium
    and would let a bookkeeping bug hide).

    Returns a dict with the residual, the force magnitudes, the per-channel
    totals, the cell voidage and the closure used.
    """
    fm = model if isinstance(model, ForceModel) else ForceModel(drag_model=model)
    z = np.asarray(z_particle, dtype=float)
    v = np.asarray(v_particle, dtype=float) * np.ones_like(z)
    n = z.size
    dx = length / n_cells
    cell_centres = (np.arange(n_cells) + 0.5) * dx

    # Voidage seen by each particle.  With a prescribed solid fraction the bed
    # is homogeneous, so every particle sees the same voidage; otherwise the
    # kernel-filtered voidage of a genuine packing is used.
    if solid_fraction is not None:
        eps_particle = np.full(n, 1.0 - float(solid_fraction))
    else:
        h = kernel_cells * dx
        w = np.maximum(0.0, 1.0 - np.abs(z[:, None] - cell_centres[None, :]) / h)
        vol_p = sphere_volume(diameter)
        eps_particle = np.clip(1.0 - vol_p * w.sum(axis=1) / (h * 1.0), 1e-6, 1.0)

    phi = 1.0 - eps_particle
    u_rel = fluid_velocity - phi * v
    vol = sphere_volume(diameter)

    drag = per_particle_drag(u_rel, eps_particle, diameter=diameter, rho_fluid=rho_fluid, mu=mu, model=fm)
    dens = drag_density(u_rel, eps_particle, diameter=diameter, rho_fluid=rho_fluid, mu=mu, model=fm)
    grad_p = fluid_pressure_gradient(eps_particle, dens, rho_fluid=rho_fluid, gravity=gravity)
    pressure = pressure_gradient_force(vol, grad_p) if fm.pressure_gradient else np.zeros(n)
    if fm.added_mass:
        acc = np.zeros(n) if du_rel_dt is None else np.asarray(du_rel_dt, dtype=float) * np.ones(n)
        am = added_mass_force(vol, rho_fluid, acc, fm.added_mass_coefficient)
    else:
        am = np.zeros(n)
    force = drag + pressure + am

    # normalised hat deposition of the reaction onto the cells
    lo, wt = linear_deposit_weights(z, n_cells, length)
    reaction = np.zeros(n_cells)
    if fm.fluid_reaction:
        np.add.at(reaction, lo, -(1.0 - wt) * force)
        np.add.at(reaction, (lo + 1) % n_cells, -wt * force)
    else:
        reaction[:] = 0.0  # NC1: no reaction on the fluid

    total_particle = float(np.sum(force))
    total_fluid = float(np.sum(reaction))
    l1 = float(np.sum(np.abs(force)))
    residual = abs(total_particle + total_fluid) / l1 if l1 > 0.0 else 0.0

    return {
        "n_particles": int(n),
        "solid_fraction": solid_fraction,
        "drag_model": fm.drag_model,
        "sum_force_particle": total_particle,
        "sum_reaction_fluid": total_fluid,
        "force_l1": l1,
        "reaction_residual": residual,
        "drag_total": float(np.sum(drag)),
        "pressure_total": float(np.sum(pressure)),
        "added_mass_total": float(np.sum(am)),
        "n_cells": int(n_cells),
        "kernel_cells": float(kernel_cells),
        "model": fm.describe(),
    }