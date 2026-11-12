"""Three-dimensional frictionless-sphere jamming: the random close-packing (RCP)
volume fraction as an *energy-minimum* property.

Why this module exists
----------------------
``tensordem.bedcoupling.packing.make_random_close_packing`` rescales the cell to
*hit* the requested porosity, so the ``solid_fraction`` it returns equals
``1 - target_porosity`` to machine precision (see
``docs/AUDIT_m16308_CASE-DEM-RCP-PACKING-FRACTION.md``).  Any benchmark that
compares that number with the literature RCP value measures its own input.  The
audit replaced it with a residual-overlap scan, but that scan is tolerance
dependent (``phi_J(tol) = 0.622 ... 0.652`` over ``tol = 1e-4 ... 5e-2``) and its
FIRE prototype never converged, so the anchor stayed open.

This module closes that gap the way the 2-D jamming campaign closed it for
disks (:mod:`tensordem.granular2d.jamming_protocol`): ``phi_J`` is taken at
**zero pressure of an energy minimum**, not at an overlap tolerance.  A
configuration is jammed when the elastic pressure of the (minimised) harmonic
contact potential extrapolates to zero at a well defined density; that density
is a property of the potential, not of the preparation history or of a
tolerance.

Protocol
--------
``periodic_rsa``
    Random sequential adsorption in a *triply periodic* cube (no wall layer --
    the wall layer is one of the known suspects for a depressed ``phi_J``).
``SpherePacking``
    Static harmonic normal-spring potential ``U = 1/2 kn sum_c delta_c^2`` on a
    cubic periodic cell with a skin-based neighbour list; virial pressure,
    coordination number and overlaps are all derived quantities of the same
    pair list.
``fire_minimize`` / ``pg_minimize``
    FIRE 1.0 (Bitzek et al., *PRL* **97**, 170201, 2006) with a monotone
    safeguard, and a Polak-Ribiere+ conjugate-gradient descent.  Running both is
    a minimiser-invariance control: ``phi_J`` must not depend on which one is
    used.  Convergence is tested on the **force** residual, because above
    jamming the energy minimum carries a *finite* overlap -- an overlap
    tolerance would either never converge or, below jamming, silently truncate
    the relaxation.
``growth_protocol``
    Quasi-static inflation: minimise, scale all radii by ``dphi``-sized
    increments, minimise again (warm started), and record ``(phi, P, z, U)``.
``fit_jamming_density``
    ``phi_J`` = zero-pressure intercept of the equation of state on the first
    few positive-pressure levels (with ``R^2`` reported, never hidden).

Reduced units throughout (``kn = 1``, mean radius ``1``): a volume fraction and
the *shape* of ``P(phi)`` are dimensionless.  All arithmetic is float64 NumPy
with deterministic pair ordering, so a ``(n, seed, protocol)`` triple is
bit-reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

__all__ = [
    "ContactModel",
    "SpherePacking",
    "MinimizerResult",
    "periodic_rsa",
    "fire_minimize",
    "pg_minimize",
    "pressure_reduced",
    "growth_protocol",
    "fit_jamming_density",
    "finite_size_scale",
    "PLACEMENT_PHI",
]

#: deposition volume fraction shared by every protocol; 3-D RSA for monodisperse
#: spheres saturates near 0.38, so 0.30 is safely below saturation.
PLACEMENT_PHI = 0.30


@dataclass(frozen=True)
class ContactModel:
    """Static contact law: linear normal spring plus a neighbour-list halo."""

    kn: float = 1.0
    skin: float = 0.40

    def __post_init__(self) -> None:
        if not math.isfinite(self.kn) or self.kn <= 0.0:
            raise ValueError("kn must be finite and positive")
        if not math.isfinite(self.skin) or self.skin <= 0.0:
            raise ValueError("skin must be finite and positive")


@dataclass(frozen=True)
class MinimizerResult:
    """Outcome of one energy minimisation."""

    method: str
    steps: int
    energy: float
    energy_start: float
    max_force: float
    max_overlap_over_diameter: float
    rejected_steps: int
    converged: bool

    def as_dict(self) -> dict:
        return asdict(self)


def periodic_rsa(n: int, radii: np.ndarray, box: float, *, seed: int = 0,
                 max_attempts_factor: float = 400.0) -> np.ndarray:
    """Random sequential adsorption of ``n`` spheres in a cubic periodic cell.

    A trial position is accepted when it does not overlap an accepted sphere
    under the minimum-image convention.  The neighbour tree is rebuilt in
    batches and the spheres inserted since the last rebuild are checked by
    brute force, so the acceptance test stays exact while the placement cost
    stays ``O(n log n)``.
    """
    rng = np.random.default_rng(seed)
    pos = np.empty((n, 3), dtype=np.float64)
    n_acc = 0
    attempts = 0
    max_attempts = int(max_attempts_factor * n) + 1000
    tree = None
    batch_anchor = 0
    r_max = float(radii.max())
    while n_acc < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"periodic_rsa could not place {n} spheres in box {box}: "
                f"stopped at {n_acc} after {attempts} attempts"
            )
        attempts += 1
        p = rng.random(3) * box
        ok = True
        if n_acc - batch_anchor > 0:
            recent = pos[batch_anchor:n_acc]
            d = recent - p
            d -= box * np.round(d / box)
            if np.min(np.linalg.norm(d, axis=1) - (radii[batch_anchor:n_acc] + radii[n_acc])) < 0.0:
                ok = False
        if ok and tree is not None:
            idx = tree.query_ball_point(p, radii[n_acc] + r_max)
            if idx:
                d = pos[idx] - p
                d -= box * np.round(d / box)
                if np.min(np.linalg.norm(d, axis=1) - (radii[idx] + radii[n_acc])) < 0.0:
                    ok = False
        if ok:
            pos[n_acc] = p
            n_acc += 1
            if n_acc - batch_anchor >= 128:
                tree = cKDTree(pos[:n_acc].copy(), boxsize=box)
                batch_anchor = n_acc
    return pos


class SpherePacking:
    """Static potential of a sphere assembly on a cubic periodic cell.

    Parameters
    ----------
    positions, radii
        ``(N, 3)`` and ``(N,)`` arrays; radii may be polydisperse.
    box
        Cubic cell edge ``L``.  Minimum image is applied on all three axes, so
        there is no wall layer and ``V = L^3`` is exact.
    model
        :class:`ContactModel` (``kn`` and the neighbour skin).

    Notes
    -----
    The pair list holds every pair within ``r_i + r_j + skin`` under minimum
    image.  It is rebuilt whenever a particle has moved more than ``skin / 2``
    since the last rebuild, which bounds the neighbour-list error: a pair that
    becomes a contact without being on the list would have to close more than
    ``skin / 2`` between rebuilds, which cannot happen.
    """

    def __init__(self, positions, radii, *, box: float,
                 model: ContactModel | None = None) -> None:
        pos = np.array(positions, dtype=np.float64)
        rad = np.array(radii, dtype=np.float64).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if pos.shape[0] != rad.shape[0]:
            raise ValueError("positions and radii must have the same length")
        if pos.shape[0] < 2:
            raise ValueError("at least two spheres are required")
        if not np.isfinite(pos).all() or not np.isfinite(rad).all():
            raise ValueError("positions and radii must be finite")
        if (rad <= 0.0).any():
            raise ValueError("radii must be positive")
        if not math.isfinite(box) or box <= 0.0:
            raise ValueError("box must be finite and positive")
        self.model = model or ContactModel()
        self.pos = pos
        self.radii = rad
        self.box = float(box)
        self.n = int(pos.shape[0])
        self._pairs = np.zeros((0, 2), dtype=np.int64)
        self._anchor = self.pos.copy()
        self.rebuild()

    # ------------------------------------------------------------------ setup
    @property
    def volume(self) -> float:
        return self.box ** 3

    @property
    def mean_radius(self) -> float:
        return float(self.radii.mean())

    def volume_fraction(self) -> float:
        return float(4.0 / 3.0 * math.pi * np.sum(self.radii ** 3) / self.volume)

    def wrap(self, pos: np.ndarray) -> np.ndarray:
        return np.mod(pos, self.box)

    def rebuild(self) -> None:
        """Rebuild the periodic neighbour list (skin shell included)."""
        cutoff = 2.0 * float(self.radii.max()) + self.model.skin
        tree = cKDTree(self.pos, boxsize=self.box)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._anchor = self.pos.copy()

    def _needs_rebuild(self) -> bool:
        d = self.pos - self._anchor
        d -= self.box * np.round(d / self.box)
        return bool(np.max(np.linalg.norm(d, axis=1)) > 0.5 * self.model.skin)

    def _pair_geometry(self):
        """Minimum-image separation and contact overlap of every listed pair."""
        i = self._pairs[:, 0]
        j = self._pairs[:, 1]
        dvec = self.pos[j] - self.pos[i]
        dvec -= self.box * np.round(dvec / self.box)
        dist = np.linalg.norm(dvec, axis=1)
        sum_r = self.radii[i] + self.radii[j]
        overlap = sum_r - dist
        return i, j, dvec, dist, overlap

    # ------------------------------------------------------------- physics
    def contact_arrays(self):
        """Return ``(i, j, unit j->i, distance, overlap)`` for overlapping pairs."""
        i, j, dvec, dist, overlap = self._pair_geometry()
        mask = overlap > 0.0
        i, j, dvec, dist, overlap = i[mask], j[mask], dvec[mask], dist[mask], overlap[mask]
        safe = np.where(dist > 0.0, dist, 1.0)
        unit = dvec / safe[:, None]
        return i, j, unit, dist, overlap

    def evaluate(self) -> dict:
        """One-shot evaluation of every derived quantity from a single pair list.

        The minimisers call this once per iteration; computing the energy,
        forces, pressure, coordination and overlap from the same pair arrays is
        what makes a step cost roughly one neighbour evaluation.
        """
        i, j, unit, dist, overlap = self.contact_arrays()
        f = np.zeros_like(self.pos)
        if overlap.size:
            # repulsive: sphere i is pushed away from j, i.e. along (x_i - x_j).
            # bincount accumulation is ~3x faster than np.add.at and keeps the
            # result deterministic (integer-keyed sums, no atomics).
            fc = (self.model.kn * overlap)[:, None] * unit
            for axis in range(3):
                f[:, axis] = (np.bincount(i, weights=-fc[:, axis], minlength=self.n)
                              + np.bincount(j, weights=fc[:, axis], minlength=self.n))
            energy = float(0.5 * self.model.kn * np.sum(overlap ** 2))
            pressure = float(self.model.kn * np.sum(overlap * dist) / (3.0 * self.volume))
            max_overlap = float(np.max(overlap) / (2.0 * self.mean_radius))
        else:
            energy = 0.0
            pressure = 0.0
            max_overlap = 0.0
        return {
            "energy": energy,
            "forces": f,
            "max_force": float(np.max(np.linalg.norm(f, axis=1))) if self.n else 0.0,
            "pressure": pressure,
            "max_overlap_over_diameter": max_overlap,
            "n_contacts": int(overlap.size),
            "coordination": float(2.0 * overlap.size / self.n),
        }

    # convenience wrappers (kept for API symmetry with the 2-D module)
    def energy(self) -> float:
        return self.evaluate()["energy"]

    def forces(self) -> np.ndarray:
        return self.evaluate()["forces"]

    def max_overlap_over_diameter(self) -> float:
        return self.evaluate()["max_overlap_over_diameter"]

    def max_force(self) -> float:
        return self.evaluate()["max_force"]

    def coordination(self) -> float:
        return self.evaluate()["coordination"]

    def pressure(self) -> float:
        return self.evaluate()["pressure"]

    # -------------------------------------------------------------- helpers
    def set_volume_fraction(self, phi: float) -> None:
        """Scale all radii (fixed cell) so the solid fraction becomes ``phi``."""
        if phi <= 0.0:
            raise ValueError("phi must be positive")
        current = self.volume_fraction()
        if current <= 0.0:
            raise ValueError("current volume fraction must be positive")
        self.radii = self.radii * (phi / current) ** (1.0 / 3.0)
        self.rebuild()

    def snapshot(self) -> dict:
        ev = self.evaluate()
        return {
            "volume_fraction": self.volume_fraction(),
            "energy": ev["energy"],
            "pressure": ev["pressure"],
            "coordination": ev["coordination"],
            "max_overlap_over_diameter": ev["max_overlap_over_diameter"],
            "max_force": ev["max_force"],
        }


# ------------------------------------------------------------------ minimisers
def fire_minimize(packing: SpherePacking, *, force_tol: float = 1e-6,
                  max_iter: int = 100000, dt: float = 1e-3,
                  dt_max: float = 5e-2, alpha_start: float = 0.1,
                  n_min: int = 5, f_inc: float = 1.1, f_dec: float = 0.5,
                  monotone_guard: bool = True) -> MinimizerResult:
    """FIRE 1.0 with an optional monotone (energy-decreasing) safeguard.

    Convergence is tested on the force residual ``max_i |F_i| <= force_tol`` in
    reduced units (``kn = 1``, mean radius ``1``).  This matters above jamming,
    where the energy minimum carries a finite contact overlap: an overlap
    criterion would there be either unreachable or satisfied only by chance.

    With ``monotone_guard`` a trial step that raises the potential energy is
    rejected and ``dt`` is halved, which turns the damped dynamics into a
    trust-region-like descent.  The number of rejected steps is reported, not
    silently absorbed: it is part of the protocol's cost.
    """
    if force_tol <= 0.0:
        raise ValueError("force_tol must be positive")
    ev = packing.evaluate()
    u_start = ev["energy"]
    u = u_start
    force = ev["forces"]
    vel = np.zeros_like(packing.pos)
    alpha = alpha_start
    steps = 0
    rejected = 0
    converged = ev["max_force"] <= force_tol
    while not converged and steps < max_iter:
        steps += 1
        f_norm = ev["max_force"]
        if f_norm == 0.0:
            converged = True
            break
        vel = vel + dt * force
        p_mix = float(np.sum(vel * force))
        v_norm = float(np.linalg.norm(vel))
        if p_mix > 0.0:
            if v_norm > 0.0:
                vel = (1.0 - alpha) * vel + alpha * (vel / v_norm) * f_norm
            if steps > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_dec
        else:
            vel[:] = 0.0
            dt = max(dt * f_dec, 1e-12)
            alpha = alpha_start
        step_vec = dt * vel
        packing.pos = packing.wrap(packing.pos + step_vec)
        ev = packing.evaluate()
        if monotone_guard and ev["energy"] > u + 1e-300:
            packing.pos = packing.wrap(packing.pos - step_vec)
            rejected += 1
            dt = max(dt * f_dec, 1e-12)
            vel[:] = 0.0
            alpha = alpha_start
            ev = packing.evaluate()
            continue
        u = ev["energy"]
        force = ev["forces"]
        if packing._needs_rebuild():
            packing.rebuild()
        converged = ev["max_force"] <= force_tol
    ev = packing.evaluate()
    return MinimizerResult(
        method="fire",
        steps=steps,
        energy=ev["energy"],
        energy_start=u_start,
        max_force=ev["max_force"],
        max_overlap_over_diameter=ev["max_overlap_over_diameter"],
        rejected_steps=rejected,
        converged=bool(ev["max_force"] <= force_tol),
    )


def pg_minimize(packing: SpherePacking, *, force_tol: float = 1e-6,
                max_iter: int = 100000, step0: float = 1e-3,
                c1: float = 1e-4, shrink: float = 0.5) -> MinimizerResult:
    """Polak-Ribiere+ conjugate-gradient descent with an Armijo line search.

    Minimiser-invariance control against :func:`fire_minimize`: the jammed
    density must not depend on which minimiser produced the energy minimum.
    """
    if force_tol <= 0.0:
        raise ValueError("force_tol must be positive")
    ev = packing.evaluate()
    u_start = ev["energy"]
    u = u_start
    force = ev["forces"]
    direction = force.copy()
    grad_prev_sq = float(np.sum(force * force))
    steps = 0
    rejected = 0
    step = step0
    converged = ev["max_force"] <= force_tol
    while not converged and steps < max_iter:
        steps += 1
        g_sq = float(np.sum(force * force))
        if g_sq == 0.0:
            converged = True
            break
        trial_step = step
        accepted = False
        while trial_step > 1e-16:
            saved = packing.pos.copy()
            packing.pos = packing.wrap(packing.pos + trial_step * direction)
            ev_try = packing.evaluate()
            if ev_try["energy"] <= u - c1 * trial_step * g_sq:
                accepted = True
                ev = ev_try
                break
            packing.pos = saved
            rejected += 1
            trial_step *= shrink
        if not accepted:
            step = max(step * shrink, 1e-16)
            direction = force.copy()
            force = packing.forces()
            continue
        u = ev["energy"]
        force_new = ev["forces"]
        if packing._needs_rebuild():
            packing.rebuild()
        beta = float(np.sum(force_new * (force_new - force)) / grad_prev_sq) if grad_prev_sq > 0 else 0.0
        beta = max(0.0, beta)
        direction = force_new + beta * direction
        if float(np.sum(direction * force_new)) <= 0.0:
            direction = force_new.copy()
            beta = 0.0
        force = force_new
        grad_prev_sq = float(np.sum(force * force))
        step = min(trial_step * 1.5, 0.05)
        converged = ev["max_force"] <= force_tol
    ev = packing.evaluate()
    return MinimizerResult(
        method="pg",
        steps=steps,
        energy=ev["energy"],
        energy_start=u_start,
        max_force=ev["max_force"],
        max_overlap_over_diameter=ev["max_overlap_over_diameter"],
        rejected_steps=rejected,
        converged=bool(ev["max_force"] <= force_tol),
    )


def pressure_reduced(packing: SpherePacking) -> float:
    """Virial pressure normalised by ``kn`` (which factors out of the statics)."""
    return float(packing.pressure() / packing.model.kn)


# ------------------------------------------------------------------- protocols
def make_packing(n: int, phi: float, *, seed: int, box: float | None = None,
                 mean_radius: float = 1.0, ratio: float = 1.0,
                 model: ContactModel | None = None) -> SpherePacking:
    """RSA deposition at :data:`PLACEMENT_PHI`, then inflate to ``phi``."""
    if phi <= PLACEMENT_PHI:
        raise ValueError("target phi must exceed the deposition density")
    if box is None:
        mean_vol = 4.0 / 3.0 * math.pi * mean_radius ** 3
        box = (n * mean_vol / PLACEMENT_PHI) ** (1.0 / 3.0)
    rng = np.random.default_rng(seed + 7919)
    if ratio == 1.0:
        radii0 = np.full(n, float(mean_radius))
    else:
        radii0 = mean_radius * (1.0 + ratio * (2.0 * rng.random(n) - 1.0))
    radii_place = radii0 * (PLACEMENT_PHI * box ** 3
                            / float(np.sum(4.0 / 3.0 * math.pi * radii0 ** 3))) ** (1.0 / 3.0)
    pos = periodic_rsa(n, radii_place, box, seed=seed)
    target = radii_place * (phi / PLACEMENT_PHI) ** (1.0 / 3.0)
    return SpherePacking(pos, target, box=box, model=model or ContactModel())


def growth_protocol(*, n: int, seed: int, phi_grid=None, dphi: float = 0.01,
                    phi_max: float = 0.72, method: str = "fire",
                    force_tol: float = 1e-6, max_iter: int = 100000,
                    mean_radius: float = 1.0, ratio: float = 1.0,
                    model: ContactModel | None = None) -> dict:
    """Quasi-static inflation protocol, returning the ``P(phi)`` equation of state.

    One energy minimum is computed at every inflation step (warm started from
    the previous step), so the recorded pressure is the *equilibrium* pressure
    at that density.  ``phi_J`` is then the zero-pressure intercept of the
    resulting equation of state; it is nowhere an input of this function.
    """
    model = model or ContactModel()
    if phi_grid is None:
        phi_grid = np.arange(PLACEMENT_PHI + dphi, phi_max + 1e-12, dphi)
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    if phi_grid.ndim != 1 or phi_grid.size == 0:
        raise ValueError("phi_grid must be a non-empty 1-d sequence")
    if np.any(np.diff(phi_grid) <= 0.0):
        raise ValueError("phi_grid must be strictly increasing")
    if phi_grid[0] <= PLACEMENT_PHI:
        raise ValueError("phi_grid must start above the deposition density")
    if method not in ("fire", "pg"):
        raise ValueError(f"unknown minimiser {method!r}")

    packing = make_packing(n, float(phi_grid[0]), seed=seed, mean_radius=mean_radius,
                           ratio=ratio, model=model)
    rows = []
    cumulative_steps = 0
    cumulative_rejected = 0
    for phi in phi_grid:
        packing.set_volume_fraction(float(phi))
        res = (fire_minimize(packing, force_tol=force_tol, max_iter=max_iter)
               if method == "fire"
               else pg_minimize(packing, force_tol=force_tol, max_iter=max_iter))
        cumulative_steps += res.steps
        cumulative_rejected += res.rejected_steps
        rows.append({
            "volume_fraction": float(packing.volume_fraction()),
            "pressure": float(pressure_reduced(packing)),
            "coordination": float(packing.coordination()),
            "energy": float(packing.energy()),
            "max_overlap_over_diameter": float(res.max_overlap_over_diameter),
            "max_force": float(res.max_force),
            "minimiser": method,
            "steps": int(res.steps),
            "converged": bool(res.converged),
        })
    return {
        "n": int(n),
        "seed": int(seed),
        "method": method,
        "dphi": float(dphi),
        "force_tol": float(force_tol),
        "polydispersity": float(ratio),
        "placement_phi": float(PLACEMENT_PHI),
        "box": float(packing.box),
        "mean_radius": float(packing.mean_radius),
        "rows": rows,
        "minimiser_steps": int(cumulative_steps),
        "rejected_steps": int(cumulative_rejected),
    }


# --------------------------------------------------------------------- analysis
def pressure_resolution(*, n: int, volume: float, coordination: float,
                       max_force: float, mean_radius: float, kn: float = 1.0) -> float:
    """Pressure that a minimiser residual force can still *fake*.

    A residual force ``f`` corresponds to a residual overlap ``delta = f / kn``
    on every contact, and the virial sum of ``N_c = z N / 2`` such contacts with
    lever arm ``2 r`` is ``P_res = kn * delta * 2 r * N_c / (3 V)
    = f * r * z * N / (3 V)``.  Using it as a floor is what makes the
    ``P(phi) -> 0`` extrapolation honest: levels whose pressure is below this
    value carry no information about jamming, they only restate the
    minimisation tolerance.
    """
    return float(max_force * mean_radius * coordination * n / (3.0 * volume * kn))


def fit_jamming_density(phi, pressure, *, floor: float = 0.0,
                        floor_mult: float = 30.0, model: str = "power_law",
                        min_levels: int = 4) -> dict:
    """``phi_J`` from the equation of state, with every choice made explicit.

    Levels with ``P <= floor_mult * floor`` are discarded (they are below the
    minimisation resolution, see :func:`pressure_resolution`).  Two estimators
    are reported on the same window so the extrapolation can be judged:

    ``power_law``  ``P = A (phi - phi_J)^beta`` with ``beta`` free -- the
                   expected near-jamming form, and the primary estimator.
    ``linear``     ``P = a (phi - phi_J)`` on the first three levels of the
                   window -- the 2-D campaign's estimator, kept as a control.

    The window, its size, both extrapolants and both ``R^2`` are returned; no
    single number is reported without the estimator that produced it.
    """
    phi = np.asarray(phi, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if phi.shape != pressure.shape or phi.ndim != 1:
        raise ValueError("phi and pressure must be 1-d and of equal length")
    if model not in ("power_law", "linear"):
        raise ValueError("model must be 'power_law' or 'linear'")
    threshold = max(floor_mult * float(floor), 0.0)
    sel = np.flatnonzero(pressure > threshold)
    if sel.size < min_levels:
        raise ValueError(
            f"only {sel.size} levels above the pressure resolution "
            f"(threshold {threshold:.3e}); need >= {min_levels}"
        )
    x = phi[sel]
    y = pressure[sel]
    # An equation of state that does not rise in phi carries no jamming
    # information; extrapolating it would turn noise into a "measurement".
    increasing = np.diff(y) > 0.0
    if increasing.size and np.mean(increasing) < 0.8:
        raise ValueError(
            "equation of state is not increasing in phi (only "
            f"{100 * np.mean(increasing):.0f}% of the levels rise); refusing to extrapolate"
        )

    def _power_law(px, a, phi_j, beta):
        return a * np.maximum(px - phi_j, 0.0) ** beta

    p0 = (y[0] / max(x[0] - (x[0] - 0.02), 1e-6) ** 1.5, x[0] - 0.02, 1.5)
    try:
        popt, pcov = curve_fit(
            _power_law, x, y, p0=p0,
            bounds=([0.0, x[0] - 0.20, 0.3], [np.inf, x[-1] + 1e-9, 5.0]),
            maxfev=40000,
        )
        amp, phi_j_pow, beta = (float(v) for v in popt)
        pow_ok = True
        pred = _power_law(x, *popt)
    except (RuntimeError, ValueError):
        amp, phi_j_pow, beta = float("nan"), float("nan"), float("nan")
        pow_ok = False
        pred = np.full_like(y, np.nan)

    n_lin = min(3, x.size)
    slope, intercept = np.polyfit(x[:n_lin], y[:n_lin], 1)
    phi_j_lin = float(-intercept / slope) if slope > 0 else float("nan")
    pred_lin = slope * x[:n_lin] + intercept

    def _r2(obs, pre):
        ss_res = float(np.nansum((obs - pre) ** 2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {
        "phi_j": phi_j_pow if model == "power_law" else phi_j_lin,
        "model": model,
        "phi_j_power_law": phi_j_pow,
        "amplitude": amp,
        "beta": beta,
        "r2_power_law": _r2(y, pred),
        "phi_j_linear": phi_j_lin,
        "linear_slope": float(slope),
        "r2_linear": _r2(y[:n_lin], pred_lin),
        "n_levels": int(x.size),
        "window": [float(x[0]), float(x[-1])],
        "levels": [float(v) for v in x],
        "pressures": [float(v) for v in y],
        "floor": float(floor),
        "threshold": float(threshold),
        "power_law_converged": bool(pow_ok),
    }


def _fss_model(n, phi_inf, c, nu, d=3.0):
    return phi_inf - c * np.asarray(n, dtype=np.float64) ** (-1.0 / (d * nu))


def finite_size_scale(sizes, phi_j, *, sigma=None, d: float = 3.0,
                      nu_bounds=(0.3, 2.0), n_boot: int = 200, seed: int = 0) -> dict:
    """Fit ``phi_J(N) = phi_J(inf) - c N^{-1/(d nu)}`` (``d = 3`` by default).

    ``nu`` is bounded; ``phi_J(inf)`` is returned with the covariance-matrix
    standard error and a seed-bootstrap interval, which must agree before the
    extrapolation is quoted as a number.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    phi_j = np.asarray(phi_j, dtype=np.float64)
    if sizes.shape != phi_j.shape or sizes.ndim != 1 or sizes.size < 3:
        raise ValueError("need at least three (size, phi_j) pairs")
    if np.any(sizes <= 0.0):
        raise ValueError("sizes must be positive")
    sig = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    if sig is not None and sig.shape != phi_j.shape:
        raise ValueError("sigma must match phi_j")
    popt, pcov = curve_fit(
        lambda n, p, c, nu: _fss_model(n, p, c, nu, d=d),
        sizes, phi_j,
        p0=(float(np.max(phi_j)), 0.1,
            float(np.clip(0.7, nu_bounds[0] + 1e-9, nu_bounds[1] - 1e-12))),
        sigma=sig, absolute_sigma=sig is not None,
        bounds=([0.0, -np.inf, nu_bounds[0]], [1.2, np.inf, nu_bounds[1]]),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    rng = np.random.default_rng(seed)
    noise_scale = sig if sig is not None else np.full(phi_j.shape, 3e-4)
    boot = []
    for _ in range(n_boot):
        try:
            p_boot, _ = curve_fit(
                lambda n, p, c, nu: _fss_model(n, p, c, nu, d=d),
                sizes, phi_j + rng.normal(0.0, noise_scale), p0=popt,
                bounds=([0.0, -np.inf, nu_bounds[0]], [1.2, np.inf, nu_bounds[1]]),
                maxfev=20000,
            )
        except RuntimeError:
            continue
        boot.append(p_boot[0])
    boot = np.asarray(boot)
    return {
        "phi_j_inf": float(popt[0]),
        "phi_j_inf_se": float(perr[0]),
        "c": float(popt[1]),
        "nu": float(popt[2]),
        "nu_se": float(perr[2]),
        "d": float(d),
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if boot.size else [float("nan"), float("nan")],
        "n_boot": int(boot.size),
    }