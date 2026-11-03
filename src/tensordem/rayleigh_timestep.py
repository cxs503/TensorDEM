"""Rayleigh surface-wave time-step estimate for the 3-D bonded-sphere DEM.

The repository's explicit step is already audited against a rigorous Gershgorin
row-sum bound (:mod:`tensordem.stability`).  That page recorded two open items:

* limitation 2 -- the audited ``dt_crit`` could not be *exercised* through the
  engine, because ``IceDEM3D.__init__`` refuses any step above its own
  conservative ``recommended_dt``;
* limitation 4 -- no Rayleigh-wave estimate was adopted, because it "needs a
  calibrated shear modulus of the lattice, which this repository does not
  have".

Limitation 4 is now stale: :func:`tensordem.lattice_elasticity.analytic_box_constants`
supplies the closed-form elastic constants ``C11 = 13k/3a``, ``C12 = C44 = 7k/3a``
of the 26-neighbour bond network that ``IceDEM3D`` actually builds, so an
isotropic-equivalent shear modulus ``G`` and Poisson ratio ``nu`` are available
without any calibration.  This module turns them into

1. ``c_R`` from the *classical Rayleigh secular equation* (exact, no fit), and
2. the classical DEM step criterion ``dt_R = 2 * r_min / c_R`` (the time a
   Rayleigh wave needs to cross one particle diameter).

It also provides the two audit primitives needed to *close* limitation 2:

* :func:`exact_bond_omega_max` -- the exact ``omega_max`` of the live bond
  network from a dense eigen-decomposition (no bound, no approximation), so the
  Gershgorin bound can be measured instead of trusted;
* :func:`divergence_threshold` -- a bisection on ``dt / dt_crit`` that runs the
  **real** engine and locates where the energy history stops being bounded, so
  the predicted threshold becomes a measured one.

Everything is SI and unit-tested in ``tests/test_dem3d_rayleigh_timestep.py``
against three independent anchors: literature values of ``c_R/c_s`` (0.874032,
0.919402, 0.955313 at ``nu = 0, 0.25, 0.5``), the closed-form lattice constants
and the engine's own step matrix.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch

from .lattice_elasticity import analytic_box_constants

DTYPE = torch.float64

#: Classical Rayleigh speed ratios ``c_R / c_s`` used as external anchors.
#: Values for ``nu = 0, 1/4, 1/2`` from the textbook solution of the Rayleigh
#: secular equation (e.g. Achenbach, *Wave Propagation in Elastic Solids*).
RAYLEIGH_RATIO_LITERATURE: dict[float, float] = {
    0.0: 0.874032,
    0.25: 0.919402,
    0.5: 0.955313,
}

#: Denominator fit used by engineering DEM practice for the Rayleigh step.
RAYLEIGH_FIT_INTERCEPT = 0.877
RAYLEIGH_FIT_SLOPE = 0.163


# --------------------------------------------------------------------------- #
# Rayleigh speed from the secular equation
# --------------------------------------------------------------------------- #
def rayleigh_speed_ratio(nu: float, *, tol: float = 1e-15, max_iter: int = 200) -> float:
    """Exact ``c_R / c_s`` from the Rayleigh secular equation.

    With ``xi = c_R / c_s`` and ``kappa = c_s / c_p = sqrt((1-2nu)/(2(1-nu)))``
    the classical equation is the cubic in ``xi**2``

    .. math::

        \\xi^6 - 8\\xi^4 + 8\\xi^2 (3 - 2\\kappa^2) - 16 (1 - \\kappa^2) = 0 ,

    whose physical root lies in ``(0, 1)``: the polynomial is negative at
    ``xi = 0`` (``-16(1-kappa^2) <= 0``, equal only at ``nu = 1/2``) and ``+1``
    at ``xi = 1``.  The root is bracketed and bisected; no series expansion or
    empirical fit is used, so the result is an independent reference for the
    engineering fit checked in :func:`rayleigh_speed_ratio_fit`.
    """
    nu = float(nu)
    if not math.isfinite(nu) or nu < 0.0 or nu > 0.5:
        raise ValueError("nu must be finite and in [0, 0.5]")
    kappa2 = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))

    def residual(xi: float) -> float:
        x2 = xi * xi
        return x2 * x2 * x2 - 8.0 * x2 * x2 + 8.0 * x2 * (3.0 - 2.0 * kappa2) - 16.0 * (1.0 - kappa2)

    lo, hi = 0.0, 1.0
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo > 0.0 or f_hi < 0.0:
        raise RuntimeError("Rayleigh secular equation is not bracketed on (0, 1)")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid == 0.0:
            return mid
        if f_mid < 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
        if hi - lo <= tol:
            break
    return 0.5 * (lo + hi)


def rayleigh_speed_ratio_fit(nu: float) -> float:
    """Engineering denominator fit ``0.163 nu + 0.877`` for ``c_R / c_s``."""
    nu = float(nu)
    if not math.isfinite(nu) or nu < 0.0 or nu > 0.5:
        raise ValueError("nu must be finite and in [0, 0.5]")
    return RAYLEIGH_FIT_SLOPE * nu + RAYLEIGH_FIT_INTERCEPT


# --------------------------------------------------------------------------- #
# Isotropic equivalent of the cubic lattice
# --------------------------------------------------------------------------- #
def cubic_isotropic_moduli(c11: float, c12: float, c44: float) -> dict[str, float]:
    """Voigt / Reuss / Hill average of a cubic stiffness tensor.

    Returns the bulk modulus ``K`` (identical for all three averages in a cubic
    crystal), the shear averages ``G_V``, ``G_R``, ``G_VRH`` and the Poisson
    ratios implied by the VRH pair, by the two-constant ``(C11, C12)`` reduction
    and by the Zener anisotropy ratio.
    """
    for name, value in (("C11", c11), ("C12", c12), ("C44", c44)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    bulk = (c11 + 2.0 * c12) / 3.0
    shear_voigt = (c11 - c12 + 3.0 * c44) / 5.0
    denominator = 4.0 * c44 + 3.0 * (c11 - c12)
    shear_reuss = 5.0 * (c11 - c12) * c44 / denominator if denominator > 0.0 else math.inf
    shear_vrh = 0.5 * (shear_voigt + shear_reuss)

    def poisson(bulk_modulus: float, shear: float) -> float:
        if shear <= 0.0 or 3.0 * bulk_modulus + shear <= 0.0:
            return math.nan
        return (3.0 * bulk_modulus - 2.0 * shear) / (2.0 * (3.0 * bulk_modulus + shear))

    return {
        "K_Pa": bulk,
        "G_voigt_Pa": shear_voigt,
        "G_reuss_Pa": shear_reuss,
        "G_vrh_Pa": shear_vrh,
        "poisson_vrh": poisson(bulk, shear_vrh),
        "poisson_voigt": poisson(bulk, shear_voigt),
        "poisson_reuss": poisson(bulk, shear_reuss),
        "poisson_c11_c12": c12 / (c11 + c12),
        "zener_anisotropy": 2.0 * c44 / (c11 - c12) if c11 != c12 else math.inf,
    }


def lattice_isotropic_moduli(bond_stiffness: float, spacing: float) -> dict[str, float]:
    """Isotropic equivalent of the 26-neighbour bond lattice (``a = spacing``)."""
    constants = analytic_box_constants(bond_stiffness, spacing)
    out: dict[str, float] = dict(constants)
    out.update(cubic_isotropic_moduli(constants["C11"], constants["C12"], constants["C44"]))
    return out


# --------------------------------------------------------------------------- #
# Rayleigh-wave step criterion
# --------------------------------------------------------------------------- #
def rayleigh_wave_speed(
    *,
    bond_stiffness: float,
    mass: float,
    radius: float,
    spacing: float | None = None,
) -> dict[str, float]:
    """Rayleigh, shear and longitudinal wave speeds of the bond lattice.

    ``rho = mass / spacing**3`` is the *lattice* density (the mass of one
    particle times the number of particles per cell), which is the density the
    continuum equivalent of the discrete network must carry.  It is not the
    particle material density ``917 kg/m^3``; for cubic packing of spheres the
    two differ by the packing fraction ``4pi/3 * (r/a)**3``.
    """
    a = 2.0 * float(radius) if spacing is None else float(spacing)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and positive")
    moduli = lattice_isotropic_moduli(bond_stiffness, a)
    rho = float(mass) / a**3
    shear = moduli["G_vrh_Pa"]
    bulk = moduli["K_Pa"]
    poisson = moduli["poisson_vrh"]
    ratio = rayleigh_speed_ratio(poisson)
    c_s = math.sqrt(shear / rho)
    c_p = math.sqrt((bulk + 4.0 * shear / 3.0) / rho)
    return {
        "lattice_density_kg_per_m3": rho,
        "spacing_m": a,
        "C11_Pa": moduli["C11"],
        "C12_Pa": moduli["C12"],
        "C44_Pa": moduli["C44"],
        "K_Pa": bulk,
        "G_vrh_Pa": shear,
        "poisson_vrh": poisson,
        "rayleigh_ratio": ratio,
        "rayleigh_ratio_fit": rayleigh_speed_ratio_fit(poisson),
        "c_s_m_per_s": c_s,
        "c_p_m_per_s": c_p,
        "c_R_m_per_s": ratio * c_s,
    }


def rayleigh_timestep_limit(
    *,
    bond_stiffness: float,
    mass: float,
    radius: float,
    safety: float = 1.0,
    spacing: float | None = None,
) -> dict[str, float]:
    """Classical DEM step criterion ``dt_R = safety * 2 r_min / c_R``.

    The ``2 r`` length is one particle diameter: the step must be shorter than
    the time a surface wave needs to cross the stiffest resolved length scale,
    otherwise the wave cannot be resolved on the mesh and the explicit update
    aliases it.  The widely quoted ``pi r / c_R`` form differs from this one by
    the factor ``pi/2 = 1.571``; both are reported so a comparison with external
    practice is explicit rather than implicit.
    """
    if not math.isfinite(safety) or safety <= 0.0:
        raise ValueError("safety must be finite and positive")
    speeds = rayleigh_wave_speed(
        bond_stiffness=bond_stiffness, mass=mass, radius=radius, spacing=spacing
    )
    travel = 2.0 * float(radius) / speeds["c_R_m_per_s"]
    return {
        **speeds,
        "safety_factor": float(safety),
        "dt_rayleigh_two_radii_s": safety * travel,
        "dt_rayleigh_pi_radius_s": safety * 0.5 * math.pi * travel,
    }


# --------------------------------------------------------------------------- #
# Exact highest mode of the live bond network
# --------------------------------------------------------------------------- #
def _assembled_bond_stiffness(model: "Any") -> torch.Tensor:
    """Dense ``3N x 3N`` stiffness matrix of the alive bonds (no contact)."""
    from .lattice_waves import assemble_stiffness

    return assemble_stiffness(model)


def exact_bond_omega_max(model: "Any", *, amplitude: float = 1e-6) -> dict[str, Any]:
    """Exact ``omega_max`` and unit top mode of the current bonded state.

    The stiffness matrix is assembled from the solver's own ``pairs``,
    ``initial_normals`` and ``alive`` mask with fixed particles eliminated, so
    the reference respects the engine's boundary conditions and damage state.
    ``omega_max = sqrt(lambda_max / m)`` carries no bound and no approximation,
    which is what makes the Gershgorin comparison in :mod:`tensordem.stability`
    measurable.

    ``amplitude`` (metres) only scales the returned mode; the mode itself is
    unit-normalised over the free degrees of freedom.
    """
    stiff = _assembled_bond_stiffness(model)
    n = int(model.positions.shape[0])
    free: list[int] = []
    for particle in range(n):
        if bool(model.fixed[particle]):
            continue
        free.extend((3 * particle, 3 * particle + 1, 3 * particle + 2))
    if not free:
        raise RuntimeError("no free degrees of freedom")
    free_idx = torch.tensor(free, dtype=torch.long)
    reduced = stiff[free_idx][:, free_idx]
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = torch.linalg.eigh(reduced)
    mass = float(model.config.mass)
    lambda_max = float(values[-1].item())
    if lambda_max <= 0.0:
        raise RuntimeError("bond network has no positive stiffness (all bonds broken?)")
    omega_max = math.sqrt(lambda_max / mass)
    return {
        "omega_max_rad_s": omega_max,
        "dt_crit_exact_s": 2.0 / omega_max,
        "free_dof": free_idx,
        "stiffness_matrix": reduced,
        "top_mode": vectors[:, -1] * float(amplitude),
        "n_free_dof": int(free_idx.numel()),
    }


# --------------------------------------------------------------------------- #
# Engine-level threshold experiment
# --------------------------------------------------------------------------- #
def _engine_energy_series(
    sim: "Any",
    *,
    dt: float,
    n_steps: int,
    excitation: dict[str, Any],
) -> list[float]:
    """Run ``sim`` at a prescribed ``dt`` and return the total energy history.

    ``dt`` is written directly onto the model.  ``IceDEM3D.__init__`` rejects
    steps above its conservative ``recommended_dt``; that guard is intentionally
    left untouched, so the audit bypasses it explicitly here (the model is
    private to this call) instead of relaxing the public bound -- see limitation
    2 of ``docs/dem3d_timestep_stability.md``.
    """
    stiff = excitation["stiffness_matrix"]
    mode = excitation["top_mode"]
    free_idx = excitation["free_dof"]
    mass = float(sim.config.mass)

    sim.velocities.zero_()
    u = torch.zeros(3 * int(sim.positions.shape[0]), dtype=DTYPE)
    u[free_idx] = mode
    sim.positions = sim.initial_positions + u.reshape(-1, 3).to(sim.positions.dtype)
    sim.dt = float(dt)
    sim.time = 0.0

    energies: list[float] = [0.5 * float(mode @ (stiff @ mode))]
    for _ in range(int(n_steps)):
        try:
            sim.step()
        except FloatingPointError:
            # The engine refuses to continue once a state is non-finite: an
            # infinite energy closes the record and marks the step as divergent.
            energies.append(math.inf)
            break
        disp = (sim.positions - sim.initial_positions).reshape(-1).to(DTYPE)
        free_disp = disp[free_idx]
        kinetic = 0.5 * mass * float((sim.velocities.reshape(-1) ** 2).sum().item())
        strain = 0.5 * float(free_disp @ (stiff @ free_disp))
        total = kinetic + strain
        if not math.isfinite(total):
            energies.append(math.inf)
            break
        energies.append(total)
    return energies


def energy_amplification_closed_form(dt_over_crit: float) -> float:
    """Closed form of the measured peak amplification, ``1 + d/(2 - d)``.

    ``d = dt * omega_max = 2 * dt/dt_crit`` is the dimensionless step of the
    marginal oscillator.  The relation is what the symplectic-Euler step matrix
    predicts for the peak of the *naive* (quadratic) energy along the top mode,
    and it reproduces the engine measurement to better than 1 % for
    ``0.1 <= dt/dt_crit <= 0.95`` (see the audit tables).  It therefore turns
    "spectral stability" into an accuracy budget: ``d/(2-d) <= eps`` requires
    ``dt/dt_crit <= eps / (2 (1 + eps))``.
    """
    d = 2.0 * float(dt_over_crit)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError("dt_over_crit must be finite and positive")
    if d >= 2.0:
        return math.inf
    return 1.0 + d / (2.0 - d)


def _energy_growth(energies: Sequence[float]) -> float:
    """Growth of the energy envelope over the last three quarters of the run.

    An algebraically growing (marginal) step leaves ``E ~ n**2``, i.e. a factor
    of at most ~3 between the first and the last quarter.  An exponentially
    growing step multiplies the envelope every step, so the same statistic
    explodes.  The ratio therefore separates ``dt <= dt_crit`` from
    ``dt > dt_crit`` without depending on the initial amplitude.
    """
    values = list(energies)
    if not values or not math.isfinite(values[-1]):
        return math.inf
    quarter = max(1, len(values) // 4)
    early = max(values[:quarter]) if values[:quarter] else 0.0
    late = max(values[-quarter:]) if values[-quarter:] else 0.0
    if early <= 0.0:
        return math.inf if late > 0.0 else 1.0
    return late / early


def energy_amplification(
    model_factory: Callable[[], "Any"],
    *,
    dt_over_crit: float,
    n_steps: int = 1200,
    divergence_ratio: float = 50.0,
    divergence_peak: float = 1e6,
) -> dict[str, float | bool]:
    """Peak-energy amplification of the top mode at a prescribed step ratio."""
    sim = model_factory()
    excitation = exact_bond_omega_max(sim)
    dt_crit = float(excitation["dt_crit_exact_s"])
    energies = _engine_energy_series(
        sim, dt=dt_over_crit * dt_crit, n_steps=n_steps, excitation=excitation
    )
    peak = max(energies) if energies else math.nan
    initial = energies[0] if energies else math.nan
    growth = _energy_growth(energies)
    return {
        "dt_over_crit": float(dt_over_crit),
        "dt_s": dt_over_crit * dt_crit,
        "dt_crit_s": dt_crit,
        "n_steps": int(n_steps),
        "peak_over_initial": float(peak / initial) if initial and math.isfinite(initial) else math.inf,
        "envelope_growth": float(growth),
        "diverged": bool(
            (not math.isfinite(growth))
            or growth > divergence_ratio
            or (math.isfinite(peak / initial) and peak / initial > divergence_peak)
        ),
        "closed_form_amplification": energy_amplification_closed_form(dt_over_crit),
    }


def divergence_threshold(
    model_factory: Callable[[], "Any"],
    *,
    n_steps: int = 1200,
    divergence_ratio: float = 50.0,
    divergence_peak: float = 1e6,
    bracket: tuple[float, float] = (0.90, 1.10),
    iterations: int = 24,
) -> dict[str, float | bool]:
    """Measure the engine's instability threshold on ``dt / dt_crit``.

    A bisection is run on the *real* engine: ``IceDEM3D`` is re-instantiated
    (clean state) at every trial step, excited in its exact top mode and run for
    ``n_steps`` steps; a trial counts as unstable when the energy envelope grows
    faster than ``divergence_ratio``.  The result is a measured threshold that
    can be compared with the predicted ``dt_crit = 2 / omega_max``.
    """
    lo, hi = float(bracket[0]), float(bracket[1])
    if not (0.0 < lo < hi):
        raise ValueError("bracket must satisfy 0 < lo < hi")
    low = energy_amplification(
        model_factory, dt_over_crit=lo, n_steps=n_steps,
        divergence_ratio=divergence_ratio, divergence_peak=divergence_peak,
    )
    high = energy_amplification(
        model_factory, dt_over_crit=hi, n_steps=n_steps,
        divergence_ratio=divergence_ratio, divergence_peak=divergence_peak,
    )
    if bool(low["diverged"]):
        raise RuntimeError(f"bracket lower bound already unstable at {lo}")
    if not bool(high["diverged"]):
        raise RuntimeError(f"bracket upper bound still stable at {hi}")
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        trial = energy_amplification(
            model_factory, dt_over_crit=mid, n_steps=n_steps,
            divergence_ratio=divergence_ratio, divergence_peak=divergence_peak,
        )
        if bool(trial["diverged"]):
            hi = mid
        else:
            lo = mid
    measured = hi
    return {
        "measured_threshold_ratio": measured,
        "predicted_threshold_ratio": 1.0,
        "relative_error": (measured - 1.0),
        "abs_relative_error": abs(measured - 1.0),
        "bracket": [float(bracket[0]), float(bracket[1])],
        "iterations": int(iterations),
        "n_steps": int(n_steps),
        "divergence_ratio": float(divergence_ratio),
        "divergence_peak": float(divergence_peak),
        "dt_crit_exact_s": float(low["dt_crit_s"]),
        "measured_threshold_dt_s": measured * float(low["dt_crit_s"]),
    }


def rayleigh_dt_comparison(
    model: "Any",
    *,
    safety: float = 1.0,
) -> dict[str, float]:
    """Compare the Rayleigh criterion, the exact limit and the engine guard.

    ``engine_recommended_dt`` is the conservative heuristic of
    ``DEM3DConfig.recommended_dt``; ``dt_crit_exact`` is the exact semi-implicit
    Euler limit ``2 / omega_max`` of the *current* bond network.  The ratios are
    reported so "the guard is conservative" becomes a measured number.
    """
    exact = exact_bond_omega_max(model)
    rayleigh = rayleigh_timestep_limit(
        bond_stiffness=model.config.bond_stiffness,
        mass=model.config.mass,
        radius=model.config.radius,
        safety=safety,
    )
    dt_crit = float(exact["dt_crit_exact_s"])
    recommended = float(model.config.recommended_dt)
    dt_rayleigh = rayleigh["dt_rayleigh_two_radii_s"]
    return {
        "omega_max_exact_rad_s": float(exact["omega_max_rad_s"]),
        "dt_crit_exact_s": dt_crit,
        "dt_rayleigh_s": dt_rayleigh,
        "engine_recommended_dt_s": recommended,
        "dt_rayleigh_over_dt_crit": dt_rayleigh / dt_crit,
        "engine_recommended_over_dt_crit": recommended / dt_crit,
        "dt_rayleigh_over_engine_recommended": dt_rayleigh / recommended,
        **{f"rayleigh_{key}": value for key, value in rayleigh.items()},
    }