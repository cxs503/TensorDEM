"""Ice-floe jamming of sea-water intake grills: anchors, statistics, envelope.

The module turns the particle-scale question ("do ice floes block the sea-water
box grill, and for how long can the cooling plant run?") into three computable
layers plus one integration layer:

1. **Discharge anchor** -- Beverloo's law
   ``W = C rho_b sqrt(g) (D - k d)**2.5`` for the mass flow of a cohesionless
   granular material out of an orifice.  ``C`` and ``k`` are calibrated against
   explicit soft-sphere DEM silo discharges and, as a separate identifiability
   test, recovered from noisy synthetic data.
2. **Jamming statistics** -- the probability ``p(D/d, inclination)`` that an
   assembly of floes forms a stable arch over an intake grill slot, measured by
   repeated 2-D DEM funnel trials, with Wilson confidence intervals and a
   power-law fit of the decay.
3. **Pressure loss** -- the Ergun equation with its viscous (Blake-Kozeny /
   Kozeny-Carman) and inertial (Burke-Plummer) branches, plus a hydraulic check
   against the dimensionless friction-factor form.
4. **Cooling-availability envelope** -- deposition of floes on the grill,
   growing blocked-area fraction, Ergun loss over the remaining open area, pump
   head limit and heat-exchanger margin, combined into an allowed continuous
   running time ``t_allow(ice concentration, ship speed)``.

Everything is SI unless a DEM helper is called: the DEM studies run in a
consistent dimensionless system where the Beverloo coefficients are invariant
(Froude scaling), so the calibrated ``C`` and ``k`` can be compared with
literature values directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .granular import (
    FunnelWalls2D,
    GranularConfig,
    GranularSystem,
    HoledFloor,
    PlaneWall,
)

__all__ = [
    "BEVERLOO_C_LITERATURE",
    "BEVERLOO_K_LITERATURE",
    "beverloo_flow",
    "fit_beverloo",
    "beverloo_identifiability",
    "beverloo_identifiability_loo",
    "leave_one_out",
    "silo_discharge_dem",
    "funnel_trial_dem",
    "jamming_probability",
    "wilson_interval",
    "fit_power_law",
    "ergun_pressure_gradient",
    "kozeny_carman_gradient",
    "ergun_friction_factor",
    "ergun_branch_audit",
    "IntakeGeometry",
    "PumpModel",
    "EnvelopeRow",
    "availability_envelope",
    "deposition_rate",
]

#: literature band for the Beverloo prefactor (spheres, circular orifice)
BEVERLOO_C_LITERATURE = (0.55, 0.65)
#: literature band for the Beverloo shape constant (in particle diameters)
BEVERLOO_K_LITERATURE = (1.35, 1.50)


# ---------------------------------------------------------------------------
# 1. Beverloo discharge anchor
# ---------------------------------------------------------------------------


def beverloo_flow(D, d, C: float = 0.58, k: float = 1.4,
                  rho_b: float = 650.0, g: float = 9.81):
    """Mass flow [kg/s] (mass units/time in DEM units) out of an orifice.

    ``D`` is the orifice diameter, ``d`` the particle diameter, ``rho_b`` the
    *bulk* density of the bed and ``g`` gravity.  Below the critical size
    ``D <= k d`` no gravity-driven discharge occurs, so the law returns zero
    rather than a negative flow.
    """
    aperture = np.asarray(D, dtype=float) - k * np.asarray(d, dtype=float)
    flow = C * rho_b * math.sqrt(g) * np.maximum(aperture, 0.0) ** 2.5
    return float(flow) if np.ndim(flow) == 0 else flow


def _beverloo_residuals(theta, D, d, Q, rho_b, g):
    C, k = theta
    return beverloo_flow(D, d, C, k, rho_b, g) - Q


def fit_beverloo(D, d, Q, rho_b: float = 650.0, g: float = 9.81,
                 p0=(0.58, 1.4), max_iter: int = 200, tol: float = 1e-13):
    """Levenberg-Marquardt fit of ``(C, k)`` to measured flows.

    Two parameters only, so a damped Gauss-Newton scheme with a numerical
    Jacobian converges in a handful of iterations; staying inside numpy avoids
    a SciPy dependency for the calibration step.
    """
    D = np.asarray(D, dtype=float)
    d = np.asarray(d, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if D.shape != Q.shape:
        raise ValueError("D and Q must have the same shape")
    if not np.all(np.isfinite(Q)):
        raise ValueError("flow data contain non-finite values")
    theta = np.array(p0, dtype=float)
    lam = 1e-3
    eps = 1e-8
    best = theta.copy()
    best_cost = float(np.sum(_beverloo_residuals(theta, D, d, Q, rho_b, g) ** 2))
    iterations = 0
    for iterations in range(1, max_iter + 1):
        r = _beverloo_residuals(theta, D, d, Q, rho_b, g)
        J = np.empty((D.size, 2))
        for j in range(2):
            step = theta.copy()
            step[j] += eps
            J[:, j] = (_beverloo_residuals(step, D, d, Q, rho_b, g) - r) / eps
        JtJ = J.T @ J
        Jtr = J.T @ r
        improved = False
        delta = np.zeros(2)
        for _ in range(40):
            try:
                delta = np.linalg.solve(
                    JtJ + lam * np.diag(np.diag(JtJ) + 1e-30), -Jtr
                )
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                lam *= 10.0
                continue
            trial = theta + delta
            cost = float(np.sum(
                _beverloo_residuals(trial, D, d, Q, rho_b, g) ** 2
            ))
            if np.isfinite(cost) and cost < best_cost:
                theta = trial
                best, best_cost = trial.copy(), cost
                lam = max(lam / 3.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
        if np.max(np.abs(delta)) < tol * (1.0 + np.max(np.abs(theta))):
            break
    rel = _beverloo_residuals(best, D, d, Q, rho_b, g) / np.maximum(Q, 1e-30)
    return {
        "C": float(best[0]),
        "k": float(best[1]),
        "cost": best_cost,
        "max_rel_error": float(np.max(np.abs(rel))),
        "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
        "iterations": int(iterations),
    }


def beverloo_identifiability(orifices=(0.050, 0.060, 0.075, 0.090),
                             particle_diameter: float = 0.0125,
                             C_true: float = 0.58, k_true: float = 1.4,
                             rho_b: float = 650.0, g: float = 9.81,
                             noise: float = 0.02, seed: int = 0,
                             n_draws: int = 400) -> dict:
    """Recover ``(C, k)`` from four orifices carrying 2 % multiplicative noise.

    Two results are reported: one reference realisation (``seed``) and the
    ensemble statistics over ``n_draws`` independent noise draws, which shows
    whether the four-orifice design is identifiable in general rather than
    accidentally well conditioned for one lucky sample.
    """
    D = np.asarray(orifices, dtype=float)
    d = np.full_like(D, particle_diameter)
    Q_true = beverloo_flow(D, d, C_true, k_true, rho_b, g)

    def run(rng_local):
        Q = Q_true * (1.0 + noise * rng_local.uniform(-1.0, 1.0, size=D.size))
        return fit_beverloo(D, d, Q, rho_b, g)

    reference = run(np.random.default_rng(seed))
    draws = [run(np.random.default_rng(10_000 + i)) for i in range(n_draws)]
    Cs = np.array([r["C"] for r in draws])
    ks = np.array([r["k"] for r in draws])
    errs = np.maximum(np.abs(Cs / C_true - 1.0), np.abs(ks / k_true - 1.0))
    return {
        "orifices_m": [float(v) for v in D],
        "particle_diameter_m": particle_diameter,
        "noise_rel": noise,
        "C_true": C_true,
        "k_true": k_true,
        "reference": {
            "C_hat": reference["C"],
            "k_hat": reference["k"],
            "max_rel_error": reference["max_rel_error"],
            "C_rel_error": abs(reference["C"] / C_true - 1.0),
            "k_rel_error": abs(reference["k"] / k_true - 1.0),
        },
        "ensemble": {
            "n_draws": int(n_draws),
            "C_mean": float(Cs.mean()),
            "C_std": float(Cs.std(ddof=1)),
            "k_mean": float(ks.mean()),
            "k_std": float(ks.std(ddof=1)),
            "max_rel_error_median": float(np.median(errs)),
            "max_rel_error_p95": float(np.quantile(errs, 0.95)),
            "C_rel_error_of_mean": float(abs(Cs.mean() / C_true - 1.0)),
            "k_rel_error_of_mean": float(abs(ks.mean() / k_true - 1.0)),
        },
    }


def beverloo_identifiability_loo(orifices=(0.050, 0.060, 0.075, 0.090),
                                 particle_diameter: float = 0.0125,
                                 C_true: float = 0.58, k_true: float = 1.4,
                                 rho_b: float = 650.0, g: float = 9.81,
                                 noise: float = 0.02, seed: int = 0,
                                 n_draws: int = 200) -> dict:
    """Leave-one-out test of the four-orifice design under 2 % noise.

    For every noise realisation one orifice is held out, ``(C, k)`` are fitted
    on the remaining three and the held-out flow is predicted.  The ensemble of
    held-out relative errors answers the acceptance question "is the Beverloo
    rate reproduced within 5 % at 3-4 apertures" independently of any
    calibration privilege.
    """
    D = np.asarray(orifices, dtype=float)
    d = np.full_like(D, particle_diameter)
    Q_true = beverloo_flow(D, d, C_true, k_true, rho_b, g)
    per_draw_max = []
    per_point = np.zeros(D.size)
    for draw in range(n_draws):
        rng = np.random.default_rng(seed + 7919 * draw)
        Q = Q_true * (1.0 + noise * rng.uniform(-1.0, 1.0, size=D.size))
        errs = []
        for i in range(D.size):
            mask = np.ones(D.size, dtype=bool)
            mask[i] = False
            fit = fit_beverloo(D[mask], d[mask], Q[mask], rho_b, g)
            pred = beverloo_flow(D[i], d[i], fit["C"], fit["k"], rho_b, g)
            errs.append(abs(pred - Q[i]) / Q[i])
        per_draw_max.append(max(errs))
        per_point += np.array(errs)
    per_draw_max = np.asarray(per_draw_max)
    return {
        "n_draws": int(n_draws),
        "noise_rel": noise,
        "median_max_rel_error": float(np.median(per_draw_max)),
        "p95_max_rel_error": float(np.quantile(per_draw_max, 0.95)),
        "worst_max_rel_error": float(per_draw_max.max()),
        "per_point_mean_rel_error": [float(v) for v in per_point / n_draws],
        "orifices_m": [float(v) for v in D],
    }


def leave_one_out(D, d, Q, rho_b: float = 650.0, g: float = 9.81) -> dict:
    """Fit on all-but-one orifice and predict the held-out flow."""
    D = np.asarray(D, dtype=float)
    d = np.asarray(d, dtype=float)
    Q = np.asarray(Q, dtype=float)
    preds, errs = [], []
    for i in range(D.size):
        mask = np.ones(D.size, dtype=bool)
        mask[i] = False
        fit = fit_beverloo(D[mask], d[mask], Q[mask], rho_b, g)
        pred = beverloo_flow(D[i], d[i], fit["C"], fit["k"], rho_b, g)
        preds.append(float(pred))
        errs.append(abs(pred - Q[i]) / Q[i])
    return {
        "predicted": preds,
        "relative_errors": errs,
        "max_rel_error": float(max(errs)),
        "mean_rel_error": float(np.mean(errs)),
    }


# ---------------------------------------------------------------------------
# 2. DEM anchors: silo discharge and funnel jamming
# ---------------------------------------------------------------------------


def silo_discharge_dem(D_over_d: float, *, radius: float = 1.0,
                       half_width: float = 9.0, layers: int = 9,
                       gravity: float = 10.0, stiffness: float = 5.0e4,
                       dt: float | None = None, settle_steps: int = 600,
                       window_start: float = 0.5, window_end: float = 1.7,
                       seed: int = 0, friction: float = 0.35) -> dict:
    """Discharge a DEM silo through a square orifice and measure the flow.

    The silo is a vertical box of square cross-section.  The bed is first
    *settled* on a closed floor and only then is the hole (area-equivalent
    diameter ``D``) opened, which removes the initial free-fall burst.  The
    mass flow is counted over a **fixed physical time window** shared by every
    aperture, so the head (bed depth) is comparable between runs and the
    measurement is not biased by the emptying tail.  All quantities are in DEM
    units.

    ``stiffness`` is chosen so that the static overlap of the deepest contact
    stays near 1 % of the radius: soft particles would otherwise squeeze
    through the orifice and inflate the measured flow (see the case document).
    """
    d = 2.0 * radius
    D = D_over_d * d
    side = 0.5 * math.sqrt(math.pi) * D  # square with the same area as circle D
    cfg = GranularConfig(radius=radius, density=1.0, gravity=gravity,
                         friction=friction, normal_stiffness=stiffness)
    rng = np.random.default_rng(seed)
    n_x = int(math.floor(2.0 * half_width / (2.05 * radius)))
    centres = []
    for k in range(layers):
        for j in range(n_x):
            for i in range(n_x):
                centres.append([
                    -half_width + (i + 0.5) * 2.05 * radius + 0.02 * rng.normal(),
                    -half_width + (j + 0.5) * 2.05 * radius + 0.02 * rng.normal(),
                    radius + 2.05 * radius * k,
                ])
    side_walls = [
        PlaneWall((1, 0, 0), -half_width),
        PlaneWall((-1, 0, 0), -half_width),
        PlaneWall((0, 1, 0), -half_width),
        PlaneWall((0, -1, 0), -half_width),
        PlaneWall((0, 0, 1), 0.0),  # closed floor during settling
    ]
    system = GranularSystem(np.array(centres), cfg, side_walls)
    dt = dt or cfg.recommended_dt()

    for _ in range(settle_steps):
        system.step(dt)

    bed_height = float(system.x[:, 2].max() - cfg.radius)
    bulk_density = system.initial_particles * cfg.mass / (
        (2.0 * half_width) ** 2 * bed_height
    )

    # open the orifice and start discharging
    system.walls[-1] = HoledFloor(0.0, 0.5 * side, 0.5 * side)
    n_steps = int(math.ceil(window_end / dt))
    times = [0.0]
    cumulative = [0]
    for step in range(n_steps):
        system.step(dt)
        below = np.nonzero(system.x[:, 2] < -1.0 * radius)[0]
        if below.size:
            system.remove(below)
        times.append((step + 1) * dt)
        cumulative.append(len(system.removed_indices))

    t = np.asarray(times)
    n_out = np.asarray(cumulative, dtype=float)
    window = (t >= window_start) & (t <= window_end)
    count = float(n_out[window][-1] - n_out[window][0]) if window.sum() > 1 else float("nan")
    duration = float(t[window][-1] - t[window][0]) if window.sum() > 1 else float("nan")
    if not np.isfinite(count) or count < 5:
        flow, r2 = float("nan"), float("nan")
    else:
        flow = count * cfg.mass / duration
        coef = np.polyfit(t[window], n_out[window], 1)
        pred = np.polyval(coef, t[window])
        ss_res = float(np.sum((n_out[window] - pred) ** 2))
        ss_tot = float(np.sum((n_out[window] - n_out[window].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    stride = max(1, len(times) // 40)
    return {
        "D_over_d": float(D_over_d),
        "D": float(D),
        "d": float(d),
        "flow": float(flow),
        "window_count": count,
        "window": [float(window_start), float(window_end)],
        "r2": float(r2),
        "particles": system.initial_particles,
        "escaped": len(system.removed_indices),
        "bulk_density": float(bulk_density),
        "bed_height": bed_height,
        "mass_residual": system.mass_residual(),
        "max_wall_penetration": system.max_wall_penetration(),
        "steps": int(n_steps),
        "settle_steps": int(settle_steps),
        "dt": float(dt),
        "gravity": float(gravity),
        "stiffness": float(stiffness),
        "friction": float(friction),
        "time": [float(v) for v in t[::stride]],
        "cumulative_out": [int(v) for v in n_out[::stride]],
    }


def funnel_trial_dem(D_over_d: float, inclination_deg: float, *,
                     radius: float = 1.0, half_width: float = 11.0,
                     layers: int = 8, gravity: float = 10.0,
                     dt: float | None = None, t_max: float = 0.8,
                     seed: int = 0, friction: float = 0.35,
                     exit_threshold: int = 3) -> dict:
    """One 2-D funnel trial: does an arch arrest the flow?

    A planar funnel (walls inclined ``inclination_deg`` from the horizontal,
    outlet width ``D``) is filled with disks.  The trial counts as *flowing*
    as soon as ``exit_threshold`` disks have left the outlet and as *jammed*
    if the flow is still arrested at ``t_max``.
    """
    d = 2.0 * radius
    D = D_over_d * d
    walls = [
        FunnelWalls2D(D, inclination_deg),
        PlaneWall((1, 0, 0), -half_width),
        PlaneWall((-1, 0, 0), -half_width),
    ]
    cfg = GranularConfig(radius=radius, density=1.0, gravity=gravity,
                         friction=friction, dimensionality=2)
    rng = np.random.default_rng(seed)
    centres = []
    spacing = 2.05 * radius
    for k in range(layers):
        z = 3.0 * radius + k * spacing * math.sqrt(3.0) / 2.0
        n_row = int(math.floor(2.0 * half_width / spacing))
        for j in range(n_row):
            centres.append([
                -half_width + (j + 0.5 * (k % 2)) * spacing + 0.05 * rng.normal(),
                0.0,
                z + 0.05 * rng.normal(),
            ])
    system = GranularSystem(np.array(centres), cfg, walls)
    dt = dt or cfg.recommended_dt()
    n_steps = int(math.ceil(t_max / dt))
    escaped = 0
    for step in range(n_steps):
        system.step(dt)
        below = np.nonzero(system.x[:, 2] < -2.0 * radius)[0]
        if below.size:
            escaped += int(below.size)
            system.remove(below)
            if escaped >= exit_threshold:
                return {
                    "D_over_d": float(D_over_d),
                    "inclination_deg": float(inclination_deg),
                    "jammed": False,
                    "escaped": escaped,
                    "time": (step + 1) * dt,
                    "particles": system.initial_particles,
                    "mass_residual": system.mass_residual(),
                    "seed": int(seed),
                }
    return {
        "D_over_d": float(D_over_d),
        "inclination_deg": float(inclination_deg),
        "jammed": True,
        "escaped": escaped,
        "time": n_steps * dt,
        "particles": system.initial_particles,
        "mass_residual": system.mass_residual(),
        "seed": int(seed),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def jamming_probability(D_over_d_values, inclination_deg: float, *,
                        trials: int = 16, base_seed: int = 0,
                        **trial_kwargs) -> dict:
    """Jamming probability and Wilson 95 % interval for each aperture ratio."""
    rows = []
    for idx, ratio in enumerate(D_over_d_values):
        outcomes = []
        for t in range(trials):
            res = funnel_trial_dem(ratio, inclination_deg,
                                   seed=base_seed + 1000 * idx + t,
                                   **trial_kwargs)
            outcomes.append(res["jammed"])
        k = int(sum(outcomes))
        lo, hi = wilson_interval(k, trials)
        rows.append({
            "D_over_d": float(ratio),
            "trials": int(trials),
            "jammed": k,
            "p": k / trials,
            "ci95": [float(lo), float(hi)],
            "inclination_deg": float(inclination_deg),
        })
    return {"inclination_deg": float(inclination_deg), "trials": rows}


def fit_power_law(ratios, probabilities, weights=None, n_boot: int = 2000,
                  seed: int = 0) -> dict:
    """Fit ``p = A (D/d)^(-alpha)`` in log space with a bootstrap interval."""
    x = np.log(np.asarray(ratios, dtype=float))
    p = np.asarray(probabilities, dtype=float)
    mask = p > 0.0
    if mask.sum() < 2:
        return {"alpha": float("nan"), "prefactor": float("nan"),
                "ci95": [float("nan"), float("nan")], "r2": float("nan"),
                "points": int(mask.sum()), "bootstrap_samples": 0}
    y = np.log(p[mask])
    xs = x[mask]
    w = None if weights is None else np.asarray(weights, dtype=float)[mask]
    A = np.vstack([xs, np.ones_like(xs)]).T
    if w is not None:
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha = -float(coef[0])
    rng = np.random.default_rng(seed)
    boots = []
    n = xs.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.ptp(xs[idx]) < 1e-9:
            continue
        c, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
        boots.append(-float(c[0]))
    boots = np.asarray(boots)
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots.size
              else (np.nan, np.nan))
    pred = np.exp(coef[1] + coef[0] * xs)
    ss_res = float(np.sum((p[mask] - pred) ** 2))
    ss_tot = float(np.sum((p[mask] - p[mask].mean()) ** 2))
    return {
        "alpha": alpha,
        "prefactor": float(np.exp(coef[1])),
        "ci95": [float(lo), float(hi)],
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "points": int(mask.sum()),
        "bootstrap_samples": int(boots.size),
    }


# ---------------------------------------------------------------------------
# 3. Ergun pressure loss
# ---------------------------------------------------------------------------


def ergun_pressure_gradient(u, d_p: float, voidage: float, rho: float = 1025.0,
                            mu: float = 1.88e-3) -> dict:
    """Ergun (1952) pressure gradient [Pa/m], split into its two branches.

    ``u`` is the superficial velocity, ``voidage`` the open-area (void)
    fraction and ``d_p`` the characteristic particle diameter.
    """
    u = np.asarray(u, dtype=float)
    eps = float(voidage)
    if not 0.0 < eps <= 1.0:
        raise ValueError("voidage must be in (0, 1]")
    viscous = 150.0 * mu * u * (1.0 - eps) ** 2 / (eps**3 * d_p**2)
    inertial = 1.75 * rho * u * np.abs(u) * (1.0 - eps) / (eps**3 * d_p)
    total = viscous + inertial
    if u.ndim == 0:
        return {"viscous": float(viscous), "inertial": float(inertial),
                "total": float(total)}
    return {"viscous": viscous, "inertial": inertial, "total": total}


def kozeny_carman_gradient(u, d_p: float, voidage: float, mu: float = 1.88e-3,
                           blake_kozeny: float = 150.0 / 36.0):
    """Kozeny-Carman viscous limit written with the specific surface ``S_v``.

    ``S_v = 6 / d_p`` for spheres, so ``k S_v^2 mu u (1-eps)^2 / eps^3``
    reproduces Ergun's viscous coefficient ``150`` when ``k = 150/36``.
    Carman's ``k = 5`` (coefficient 180) gives the well known ~20 % spread
    between the two correlations; both are reported so the spread stays
    visible instead of being hidden.
    """
    eps = float(voidage)
    Sv = 6.0 / d_p
    k = float(blake_kozeny)
    return k * Sv**2 * mu * np.asarray(u, dtype=float) * (1.0 - eps) ** 2 / eps**3


def ergun_friction_factor(reynolds: float) -> float:
    """Dimensionless Ergun form ``f = 150/Re + 1.75``."""
    return 150.0 / reynolds + 1.75


def ergun_branch_audit(u=0.15, d_p=0.025, voidage=0.45, rho=1025.0,
                       mu=1.88e-3) -> dict:
    """Audit both Ergun branches against their analytic limits."""
    branch = ergun_pressure_gradient(u, d_p, voidage, rho, mu)
    kc_blake = float(kozeny_carman_gradient(u, d_p, voidage, mu))
    kc_carman = float(kozeny_carman_gradient(u, d_p, voidage, mu,
                                             blake_kozeny=5.0))
    burke_plummer = 1.75 * rho * u * abs(u) * (1.0 - voidage) / (voidage**3 * d_p)

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    eps = voidage
    reynolds = rho * u * d_p / (mu * (1.0 - eps))
    f_ergun = ergun_friction_factor(reynolds)
    f_rebuilt = branch["total"] / (
        rho * u * abs(u) * (1.0 - eps) / (eps**3 * d_p)
    )
    low = ergun_pressure_gradient(1e-9, d_p, voidage, rho, mu)
    high = ergun_pressure_gradient(1e3, d_p, voidage, rho, mu)
    return {
        "inputs": {"u": u, "d_p": d_p, "voidage": voidage, "rho": rho, "mu": mu},
        "viscous": branch["viscous"],
        "inertial": branch["inertial"],
        "total": branch["total"],
        "kozeny_carman_blake": kc_blake,
        "kozeny_carman_carman_k5": kc_carman,
        "burke_plummer": burke_plummer,
        "residual_viscous_vs_blake": float(abs(branch["viscous"] - kc_blake)),
        "rel_viscous_vs_blake": float(rel(branch["viscous"], kc_blake)),
        "rel_viscous_vs_carman_k5": float(rel(branch["viscous"], kc_carman)),
        "rel_inertial_vs_burke_plummer": float(
            rel(branch["inertial"], burke_plummer)
        ),
        "friction_factor_reynolds": float(reynolds),
        "friction_factor_from_ergun": float(f_ergun),
        "friction_factor_from_gradient": float(f_rebuilt),
        "rel_friction_factor": float(rel(f_rebuilt, f_ergun)),
        "laminar_ratio_inertial_over_viscous": float(
            low["inertial"] / low["viscous"]
        ),
        "turbulent_ratio_viscous_over_inertial": float(
            high["viscous"] / high["inertial"]
        ),
    }


# ---------------------------------------------------------------------------
# 4. Cooling-availability envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntakeGeometry:
    """Sea-water box intake of a polar ship."""

    grill_area: float = 1.2          # m^2, wetted face area of the grill
    open_ratio: float = 0.45         # open area / face area at t = 0
    slot_pitch: float = 0.080        # m, slot width D
    floe_diameter: float = 0.025     # m, d
    inclination_deg: float = 60.0    # grill bars vs. horizontal
    floe_thickness: float = 0.030    # m, deposit layer thickness scale
    voidage: float = 0.45            # deposit void fraction (Ergun eps)
    flow_length: float = 0.10        # m, streamwise length of the deposit /
                                     # grill depth over which Ergun acts
    retention_efficiency: float = 0.05  # net fraction of captured floes that
                                        # stay on the grill (rest is shed by
                                        # re-entrainment / melting); a model
                                        # parameter, not a measurement

    @property
    def D_over_d(self) -> float:
        return self.slot_pitch / self.floe_diameter

    @property
    def initial_open_area(self) -> float:
        return self.grill_area * self.open_ratio


@dataclass(frozen=True)
class PumpModel:
    """Sea-water pump / heat exchanger operating point."""

    volumetric_flow: float = 0.120       # m^3/s design flow
    max_pressure_drop: float = 25_000.0  # Pa, pump head budget
    density: float = 1025.0
    viscosity: float = 1.88e-3
    inlet_temperature: float = -0.5      # deg C (saline water)
    outlet_limit: float = 32.0           # deg C, cooling-water return limit
    heat_duty: float = 1.2e6             # W, required cooling
    specific_heat: float = 3990.0        # J/kg/K


@dataclass
class EnvelopeRow:
    ice_concentration: float
    ship_speed_kn: float
    arrival_flux: float
    jam_probability: float
    blockage_rate_per_s: float
    pressure_drop_pa: float
    hydraulic_time_s: float
    thermal_time_s: float
    allowed_time_s: float
    outlet_temperature_c: float
    cooling_margin: float

    def as_dict(self) -> dict:
        return asdict(self)


def deposition_rate(ice_concentration: float, ship_speed: float,
                    capture_probability: float,
                    layer_thickness: float) -> float:
    """Blocked-area fraction growth rate [1/s].

    Ice arrives at the grill with the volumetric flux ``c_i * v`` (m^3 of ice
    per m^2 of grill per second); the fraction ``p`` is captured and builds a
    layer of thickness ``h``, hence ``dphi/dt = p c_i v / h``.
    """
    return capture_probability * ice_concentration * ship_speed / layer_thickness


def _outlet_temperature(pump: PumpModel, mass_flow: float) -> float:
    if mass_flow <= 1e-9:
        return float("inf")
    return pump.inlet_temperature + pump.heat_duty / (
        mass_flow * pump.specific_heat
    )


def availability_envelope(geometry: IntakeGeometry, pump: PumpModel,
                          jamming_curves, ice_concentrations, ship_speeds_kn,
                          horizon_s: float = 3600.0,
                          deposition_amplification: float = 1.0):
    """Build the ice-concentration / speed / allowed-running-time table.

    ``jamming_curves`` is either a callable ``p(D/d)`` or a ``(ratios, probs)``
    pair interpolated in log-probability.  Blocked area grows as
    ``phi = 1 - exp(-t/tau)``; the hydraulic limit is the time at which the
    Ergun loss over the remaining open area reaches the pump head, and the
    thermal limit the time at which the sea-water outlet temperature exceeds
    its limit after the flow is throttled by the pump curve.

    Returns ``(rows, summary)`` with ``rows`` being a flat list of
    :class:`EnvelopeRow` so it can be dumped to JSON directly.
    """
    if isinstance(jamming_curves, (list, tuple)):
        ratios, probs = jamming_curves
        x = np.log(np.asarray(ratios, dtype=float))
        log_p = np.log(np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0))

        def jam(ratio):
            return float(np.exp(np.interp(np.log(ratio), x, log_p)))
    else:
        jam = jamming_curves
    p_jam = float(jam(geometry.D_over_d))

    def blocked_fraction(rate, t):
        return 1.0 - math.exp(-rate * t) if rate > 0 else 0.0

    def dp_of_time(rate, t):
        phi = blocked_fraction(rate, t)
        free = geometry.initial_open_area * max(1.0 - phi, 1e-6)
        u = pump.volumetric_flow / free
        gradient = ergun_pressure_gradient(u, geometry.floe_diameter,
                                           geometry.voidage, pump.density,
                                           pump.viscosity)["total"]
        # Ergun returns a gradient [Pa/m]; the deposit / grill depth converts it
        # into the pressure loss the pump actually sees
        return gradient * geometry.flow_length

    def state_at(rate, t):
        dp = dp_of_time(rate, t)
        throttle = min(1.0, math.sqrt(pump.max_pressure_drop / max(dp, 1e-9)))
        m_dot = pump.volumetric_flow * throttle * pump.density
        return dp, throttle, _outlet_temperature(pump, m_dot)

    rows = []
    for concentration in ice_concentrations:
        for speed_kn in ship_speeds_kn:
            speed = float(speed_kn) * 0.514444
            rate = deposition_rate(concentration, speed, p_jam,
                                   geometry.floe_thickness)
            rate *= deposition_amplification * geometry.retention_efficiency
            t_hyd = _bisect_time(lambda t, r=rate: dp_of_time(r, t),
                                 pump.max_pressure_drop, horizon_s)
            t_th = _bisect_time(lambda t, r=rate: state_at(r, t)[2],
                                pump.outlet_limit, horizon_s)
            allowed = min(t_hyd, t_th)
            dp, throttle, t_out = state_at(rate, allowed)
            rows.append(EnvelopeRow(
                ice_concentration=float(concentration),
                ship_speed_kn=float(speed_kn),
                arrival_flux=float(concentration * speed),
                jam_probability=float(p_jam),
                blockage_rate_per_s=float(rate),
                pressure_drop_pa=float(dp),
                hydraulic_time_s=float(t_hyd),
                thermal_time_s=float(t_th),
                allowed_time_s=float(allowed),
                outlet_temperature_c=float(t_out),
                cooling_margin=float(
                    (pump.outlet_limit - pump.inlet_temperature)
                    / max(t_out - pump.inlet_temperature, 1e-9)
                ),
            ))
    summary = {
        "geometry": asdict(geometry),
        "pump": asdict(pump),
        "D_over_d": geometry.D_over_d,
        "jam_probability": p_jam,
        "retention_efficiency": geometry.retention_efficiency,
        "horizon_s": horizon_s,
        "rows": len(rows),
        "min_allowed_time_s": min(r.allowed_time_s for r in rows),
        "max_allowed_time_s": max(r.allowed_time_s for r in rows),
    }
    return rows, summary


def _bisect_time(func, target: float, horizon: float, tol: float = 1e-6) -> float:
    """First time in ``[0, horizon]`` at which ``func`` reaches ``target``."""
    if func(0.0) >= target:
        return 0.0
    if func(horizon) < target:
        return float(horizon)
    lo, hi = 0.0, float(horizon)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if func(mid) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)