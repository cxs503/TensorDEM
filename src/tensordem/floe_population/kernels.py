"""Closures of the floe population balance, measured on the DEM instrument.

Two kernels close the equation

.. math:: \\frac{\\partial n(D,t)}{\\partial t}
        = \\underbrace{\\tfrac12\\int_0^D K(D',D'')n(D')n(D'')\\,d\\ell}_{coalescence}
        - \\underbrace{\\int_0^\\infty K(D,D')n(D')n(D)\\,dD'}_{coalescence\\ loss}
        - \\underbrace{\\nu(D)\\,n(D)}_{breakage\\ loss}
        + \\underbrace{\\int_D^\\infty \\nu(D')\\,\\beta(D|D')\\,n(D')\\,dD'}_{breakage\\ source}

Collision kernel
----------------
For a dilute gas of bodies of diameter :math:`D` the Smoluchowski collision rate
constant is the geometric cross-section times the mean relative speed times the
contact value of the radial distribution function (Enskog):

.. math:: K(D_1,D_2) = \\frac{\\pi}{4}(D_1+D_2)^2\\,\\langle|v_{rel}|\\rangle\\,
          g_{12}(\\sigma_{12}),

with :math:`\\langle|v_{rel}|\\rangle = \\sqrt{8T/(\\pi\\mu)}` for a Maxwellian gas
of granular temperature :math:`T` and reduced mass :math:`\\mu`.  ``kernels``
*measures* the three ingredients separately -- the birth rate of contacts
(:meth:`SphereAssembly.contact_events`), :math:`g` from the pair distribution,
and :math:`T` from the velocity field -- and only then compares the product with
the measured rate.  Nothing in the prediction is fitted.

Breakage kernel
---------------
A floe of diameter :math:`D` driven into an anvil at speed ``v`` receives the
impact energy ``E = M v**2 / 2`` and shatters into ``B`` fragments.  The measured
map ``(D, E) -> B`` is fitted in log-log space by

.. math:: B(D,E) = k_{break} D^{\\alpha} E^{\\beta},

and the fit is validated on a grid point that was **not** used for the fit.
``B`` is an extensive fragment count, hence ``B >= 1`` and ``nu(D) = (B-1) *``
(the collision frequency) is the breakage rate used by the population balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .soft_sphere import (
    ContactModel,
    PairHistogramAccumulator,
    SphereAssembly,
    Wall,
    build_floe,
    maxwellian_velocities,
    pair_distribution,
    place_floes,
    rotate_about,
)

__all__ = [
    "CollisionKernelTable",
    "measure_collision_kernel",
    "BreakageKernel",
    "impact_fragmentation",
    "calibrate_breakage_kernel",
    "radial_contact_value",
]


# ---------------------------------------------------------------- collision kernel
@dataclass
class CollisionKernelTable:
    """Per-class collision statistics of one DEM run.

    ``k_measured`` is ``K`` inferred from the birth rate of contacts,
    ``k_predicted`` the Enskog expression built from the independently measured
    ``g``, granular temperature and masses.  ``rate`` is the measured number of
    birth events per unit time and volume (the weight used by the L1 error).
    """

    diameters: np.ndarray
    counts: np.ndarray
    rate: np.ndarray
    k_measured: np.ndarray
    k_predicted: np.ndarray
    g_contact: np.ndarray
    v_rel_maxwell: np.ndarray
    v_rel_measured: np.ndarray
    temperature: float
    duration: float
    volume: float

    def _mask(self, min_events: int = 40) -> np.ndarray:
        mask = self.counts >= float(min_events)
        mask &= self.k_measured > 0.0
        mask &= self.k_predicted > 0.0
        return mask

    @property
    def relative_error(self) -> np.ndarray:
        """Per-cell relative error, NaN where the cell has no statistics."""
        mask = self._mask()
        error = np.full(self.k_measured.shape, np.nan)
        error[mask] = np.abs(self.k_predicted[mask] - self.k_measured[mask]) / self.k_measured[mask]
        return error

    @property
    def cells_used(self) -> int:
        return int(self._mask().sum())

    @property
    def weighted_error(self) -> float:
        """Rate-weighted L1 error: the total contact-count error (criterion C1)."""
        mask = self._mask()
        if not np.any(mask):
            return float("nan")
        return float(
            np.sum(self.rate[mask] * np.abs(self.k_predicted[mask] - self.k_measured[mask]) / self.k_measured[mask])
            / np.sum(self.rate[mask])
        )

    @property
    def max_error(self) -> float:
        error = self.relative_error
        mask = np.isfinite(error)
        if not np.any(mask):
            return float("nan")
        return float(error[mask].max())

    def as_dict(self) -> dict:
        return dict(
            diameters=[float(v) for v in self.diameters],
            counts=[[int(v) for v in row] for row in self.counts],
            rate=[[float(v) for v in row] for row in self.rate],
            k_measured=[[float(v) for v in row] for row in self.k_measured],
            k_predicted=[[float(v) for v in row] for row in self.k_predicted],
            g_contact=[[float(v) for v in row] for row in self.g_contact],
            v_rel_maxwell=[[float(v) for v in row] for row in self.v_rel_maxwell],
            v_rel_measured=[[float(v) for v in row] for row in self.v_rel_measured],
            relative_error=[[float(v) for v in row] for row in self.relative_error],
            temperature=float(self.temperature),
            duration=float(self.duration),
            volume=float(self.volume),
            weighted_error=float(self.weighted_error),
            max_error=float(self.max_error),
        )


def radial_contact_value(
    position,
    box,
    *,
    diameters,
    class_id,
    n_bins: int = 80,
):
    """``g_ab`` evaluated at the contact separation of every diameter pair.

    The pair distribution is measured in shells and the shell containing
    ``sigma_ab = (D_a + D_b)/2`` is used; the value is a lower bound when the
    shell is finite and the pair distribution is steep, which is why the shell
    width is reported alongside.
    """
    diameters = np.asarray(diameters, dtype=np.float64).reshape(-1)
    class_id = np.asarray(class_id, dtype=np.int64).reshape(-1)
    d_max = float(diameters.max())
    r_max = 1.10 * d_max
    centres, g, counts = pair_distribution(
        position,
        box,
        class_id=class_id,
        r_max=r_max,
        n_bins=n_bins,
    )
    n_classes = diameters.size
    sigma = 0.5 * (diameters[:, None] + diameters[None, :])
    values = np.zeros((n_classes, n_classes))
    width = float(centres[1] - centres[0]) if centres.size > 1 else 0.0
    for a in range(n_classes):
        for b in range(n_classes):
            index = int(np.argmin(np.abs(centres - sigma[a, b])))
            values[a, b] = g[a, b, index]
    return values, width


def _random_sequential_positions(radius, box, rng, *, clearance: float = 1.02,
                                attempts: int = 200000):
    """Non-overlapping random insertion in a periodic cube.

    A jittered lattice cannot host strongly polydisperse spheres without overlap
    (the largest sphere would dictate the lattice spacing and collapse the packing
    fraction), and an overlapping start deposits a shock that leaves the heavy
    spheres out of equipartition for a long time.  Rejection sampling avoids both.
    """
    radius = np.asarray(radius, dtype=np.float64)
    box_length = float(box[0])
    order = np.argsort(-radius)                     # place the big ones first
    position = np.zeros((radius.size, 3))
    placed = np.zeros((0, 3))
    placed_radius = np.zeros(0)
    for index in order:
        r = radius[index]
        for _ in range(attempts):
            candidate = rng.uniform(0.0, box_length, size=3)
            if placed.shape[0]:
                delta = placed - candidate
                delta -= box_length * np.round(delta / box_length)
                dist = np.linalg.norm(delta, axis=1)
                if np.any(dist < clearance * (r + placed_radius)):
                    continue
            position[index] = candidate
            placed = np.vstack([placed, candidate])
            placed_radius = np.append(placed_radius, r)
            break
        else:
            raise RuntimeError(
                "random sequential insertion failed; lower the packing fraction"
            )
    return position


def measure_collision_kernel(
    *,
    diameters,
    counts_per_class: int,
    temperature: float,
    packing_fraction: float,
    seed: int = 0,
    duration: float = 40.0,
    model: ContactModel | None = None,
    snapshot_interval: int = 10,
) -> CollisionKernelTable:
    """Homogeneous, thermostatted sphere gas: measure ``K`` per diameter class.

    A dilute-ish ensemble (``packing_fraction`` around 0.10--0.35) of free spheres
    with the requested diameters is equilibrated, then three quantities are
    measured over ``duration``:  the birth events of contacts (the rate), the
    time-averaged pair histogram (the contact value ``g``) and the granular
    temperature.  There is no shear, no wall and no gravity, so the state is
    isotropic and the Enskog closure is applied in its own geometry.

    The prediction uses only those measured ingredients:

    ``K = pi * sigma**2 * sqrt(8 T / (pi mu)) * g(sigma)``,

    so the comparison with the measured rate is a genuine test of the closure.
    Masses follow the engine convention ``m = r**3`` (unit density).
    """
    diameters = np.asarray(diameters, dtype=np.float64).reshape(-1)
    if diameters.size < 1:
        raise ValueError("at least one diameter class is required")
    rng = np.random.default_rng(seed)
    radii_per_class = 0.5 * diameters
    radius = np.repeat(radii_per_class, int(counts_per_class))
    class_id = np.repeat(np.arange(diameters.size), int(counts_per_class))
    volume_spheres = float(np.sum(4.0 / 3.0 * math.pi * radius**3))
    box_length = (volume_spheres / packing_fraction) ** (1.0 / 3.0)
    box = np.array([box_length] * 3)
    position = _random_sequential_positions(radius, box, rng)
    velocity = maxwellian_velocities(radius**3, temperature, rng=rng)
    assembly = SphereAssembly(
        radius,
        position,
        box=box,
        model=model,
        velocity=velocity,
        tag=class_id,
        rng=rng,
    )
    assembly.resolve_contacts(passes=5)
    dt = assembly.recommended_dt()
    steps = int(round(duration / dt))
    if steps < 2:
        raise ValueError("duration too short for the stable time step")
    spin = int(round(0.25 * steps))
    assembly.advance(dt, spin, temperature=temperature, thermostat_interval=5)
    accumulator = PairHistogramAccumulator(
        box=box,
        diameters=diameters,
        n_classes=diameters.size,
    )
    accumulator.observe(assembly.position, class_id)

    def sample(_assembly):
        accumulator.observe(_assembly.position, class_id)

    assembly.contact_events.clear()
    t_start = assembly.time
    assembly.advance(
        dt,
        steps - spin,
        temperature=temperature,
        thermostat_interval=5,
        callback=sample,
        callback_interval=max(1, int(snapshot_interval)),
    )
    measured_span = assembly.time - t_start
    volume = float(np.prod(box))
    n_classes = diameters.size
    raw = np.zeros((n_classes, n_classes))
    speed_sum = np.zeros((n_classes, n_classes))
    for _, i, j, v_n in assembly.contact_events:
        a, b = int(class_id[i]), int(class_id[j])
        raw[min(a, b), max(a, b)] += 1.0
        speed_sum[min(a, b), max(a, b)] += abs(float(v_n))
    counts = raw + raw.T - np.diag(np.diag(raw))
    speeds = speed_sum + speed_sum.T - np.diag(np.diag(speed_sum))
    number_density = np.array([np.sum(class_id == k) / volume for k in range(n_classes)])
    rate = counts / (measured_span * volume)
    k_measured = np.zeros((n_classes, n_classes))
    for a in range(n_classes):
        for b in range(n_classes):
            denom = number_density[a] * number_density[b]
            if denom <= 0.0:
                continue
            k_measured[a, b] = rate[a, b] / denom
            if a == b:
                k_measured[a, b] *= 2.0
    # The event ledger is sampled on the shell ``r < sigma + event_gap`` (so that a
    # soft contact cannot register several births); the flux through that shell is
    # scaled back to the contact sphere analytically.
    shell_gap = float((model or ContactModel()).event_gap)
    g_contact = accumulator.contact_value()
    mass_class = radii_per_class**3          # engine convention: unit density
    mu = mass_class[:, None] * mass_class[None, :] / (
        mass_class[:, None] + mass_class[None, :]
    )
    v_rel = np.sqrt(8.0 * temperature / (math.pi * mu))
    sigma = 0.5 * (diameters[:, None] + diameters[None, :])
    k_measured = k_measured * (sigma / (sigma + shell_gap)) ** 2
    k_predicted = math.pi * sigma**2 * v_rel * g_contact
    with np.errstate(invalid="ignore", divide="ignore"):
        v_rel_measured = np.where(counts > 0, speeds / np.maximum(counts, 1.0), np.nan)
    return CollisionKernelTable(
        diameters=diameters,
        counts=counts,
        rate=rate,
        k_measured=k_measured,
        k_predicted=k_predicted,
        g_contact=g_contact,
        v_rel_maxwell=v_rel,
        v_rel_measured=v_rel_measured,
        temperature=float(assembly.granular_temperature()),
        duration=float(measured_span),
        volume=volume,
    )


# ----------------------------------------------------------------- breakage kernel
@dataclass
class ImpactResult:
    """One anvil impact: parent size, impact energy, fragment census."""

    diameter: float
    speed: float
    energy: float
    fragments: int
    largest: float
    bonds_broken: int
    mean_fragment: float
    seed: int
    fragment_diameters: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(
            diameter=float(self.diameter),
            speed=float(self.speed),
            energy=float(self.energy),
            fragments=int(self.fragments),
            largest=float(self.largest),
            bonds_broken=int(self.bonds_broken),
            mean_fragment=float(self.mean_fragment),
            seed=int(self.seed),
            fragment_diameters=[float(v) for v in self.fragment_diameters],
        )


def impact_fragmentation(
    n_side: int,
    speed: float,
    *,
    seed: int = 0,
    radius: float = 0.5,
    model: ContactModel | None = None,
    drop_height: float = 4.0,
    settle_time: float = 2.5,
) -> ImpactResult:
    """Drive one aggregate into a rigid anvil and census the debris.

    The aggregate is a ``n_side**3`` simple-cubic cluster of bonded spheres, so
    its volume-equivalent diameter is ``n_side`` sphere diameters (for
    ``radius = 0.5``).  The impact energy is exactly ``M v**2 / 2``; the run stops
    after ``settle_time`` once the bond network has stopped changing, which is
    when the fragment census is frozen.
    """
    rng = np.random.default_rng(seed)
    grid, bonds, rad = build_floe(n_side, radius=radius)
    mass = float(np.sum(rad**3))
    centre = np.array([0.0, 0.0, drop_height])
    position = rotate_about(grid, np.zeros(3), rng) + centre
    velocity = np.zeros_like(position)
    velocity[:, 2] = -speed
    box_length = max(24.0, 6.0 * n_side * radius)
    assembly = SphereAssembly(
        rad,
        position,
        box=np.array([box_length] * 3),
        model=model,
        velocity=velocity,
        bonds=bonds,
        walls=[Wall(axis=2, coordinate=0.0, sign=1.0)],
        rng=rng,
    )
    dt = assembly.recommended_dt()
    travel = (drop_height - 1.5 * n_side * radius) / speed
    total = travel + settle_time + 2.0 * (n_side * radius) / speed
    steps = max(1, int(round(total / dt)))
    assembly.advance(dt, steps, record_contacts=False)
    census = assembly.cluster_census()
    return ImpactResult(
        diameter=float(census.diameter_per_sphere * n_side),
        speed=float(speed),
        energy=0.5 * mass * speed**2,
        fragments=int(len(census)),
        largest=float(census.diameter.max()),
        bonds_broken=int(np.sum(~assembly.bond_alive)),
        mean_fragment=float(census.diameter.mean()),
        seed=int(seed),
        fragment_diameters=[float(v) for v in census.diameter],
    )


@dataclass
class BreakageKernel:
    """Fitted ``B(D, E) = k D^alpha E^beta`` with its validation set."""

    k_break: float
    alpha: float
    beta: float
    n_fit: int
    fit_points: list[dict]
    validation_points: list[dict]
    fit_r2: float

    def predict(self, diameter, energy) -> np.ndarray:
        d = np.asarray(diameter, dtype=np.float64)
        e = np.asarray(energy, dtype=np.float64)
        return self.k_break * d**self.alpha * e**self.beta

    @property
    def validation_error(self) -> float:
        """Aggregate relative error over the held-out grid (criterion C2)."""
        if not self.validation_points:
            return float("nan")
        observed = np.array([p["fragments"] for p in self.validation_points], dtype=np.float64)
        predicted = np.array(
            [float(self.predict(p["diameter"], p["energy"])) for p in self.validation_points]
        )
        return float(np.abs(predicted - observed).sum() / observed.sum())

    @property
    def validation_error_mean(self) -> float:
        if not self.validation_points:
            return float("nan")
        observed = np.array([p["fragments"] for p in self.validation_points], dtype=np.float64)
        predicted = np.array(
            [float(self.predict(p["diameter"], p["energy"])) for p in self.validation_points]
        )
        return float(np.mean(np.abs(predicted - observed) / observed))

    def as_dict(self) -> dict:
        return dict(
            k_break=float(self.k_break),
            alpha=float(self.alpha),
            beta=float(self.beta),
            n_fit=int(self.n_fit),
            fit_r2=float(self.fit_r2),
            form="B(D,E) = k_break * D**alpha * E**beta",
            validation_error=float(self.validation_error),
            validation_error_pointwise=float(self.validation_error_mean),
            fit_points=[dict(p) for p in self.fit_points],
            validation_points=[dict(p) for p in self.validation_points],
        )


def calibrate_breakage_kernel(
    results,
    *,
    hold_out,
) -> BreakageKernel:
    """Log-log least squares of ``B = k D^alpha E^beta`` with a held-out grid.

    ``hold_out`` is a predicate on the *record* of one impact; every record it
    accepts is excluded from the fit and used for the validation error, so the
    reported error is a genuine forecast of the power-law closure and not a fit
    residual.
    """
    fit_points, val_points = [], []
    for item in results:
        record = item.as_dict() if isinstance(item, ImpactResult) else dict(item)
        (val_points if hold_out(record) else fit_points).append(record)
    if len(fit_points) < 4:
        raise ValueError("need at least four calibration impacts")
    design = np.array(
        [
            [1.0, math.log(p["diameter"]), math.log(p["energy"])]
            for p in fit_points
        ]
    )
    target = np.array([math.log(p["fragments"]) for p in fit_points])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return BreakageKernel(
        k_break=float(math.exp(coefficients[0])),
        alpha=float(coefficients[1]),
        beta=float(coefficients[2]),
        n_fit=len(fit_points),
        fit_points=fit_points,
        validation_points=val_points,
        fit_r2=float(r2),
    )