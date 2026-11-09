"""Broken-ice floe field: the DEM experiment the population balance is tested on.

The field is an ensemble of cemented aggregates ("floes") in a periodic box that
is agitated by a *floe-level* granular temperature (the field's kinetic energy is
carried by floe centres of mass, see
:meth:`tensordem.floe_population.soft_sphere.SphereAssembly.cluster_thermostat`).
Floes collide, their bonds fail, and the fragment census is recorded on a series
of snapshots.  Three things are measured and nothing else:

``census``
    the floe diameters at every snapshot (the ``n(D, t)`` time series);
``collision rate``
    the births of *inter-floe* sphere contacts, counted per floe size class pair;
    this is the field's own collision kernel, from which the per-floe collision
    frequency ``nu(D)`` follows;
``structure``
    the contact value of the floe-centre pair distribution and the granular
    temperature of the floe ensemble.

The campaign uses the first half of the record as the calibration window and the
second half as the forecast window, so the population balance is never fitted to
the data it is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .soft_sphere import ContactModel, SphereAssembly, maxwellian_velocities, place_floes

__all__ = [
    "FieldConfig",
    "FieldHistory",
    "run_floe_field",
    "floe_kernel_from_events",
    "floe_class_edges",
]


@dataclass(frozen=True)
class FieldConfig:
    """Setup of the floe field (all lengths in units of the sphere diameter)."""

    specs: tuple[int, ...] = (3,) * 12 + (4,) * 12 + (5,) * 6
    temperature: float = 4.0
    sphere_fraction: float = 0.06
    radius: float = 0.5
    duration: float = 60.0
    n_snapshots: int = 25
    n_classes: int = 4
    breakage: bool = True
    seed: int = 0
    label_interval: int = 25

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("the field needs at least one floe")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.sphere_fraction < 0.5:
            raise ValueError("sphere_fraction must be in (0, 0.5)")
        if self.n_snapshots < 3:
            raise ValueError("need at least three snapshots")
        if self.n_classes < 2:
            raise ValueError("need at least two floe size classes")
        if min(self.specs) < 2:
            raise ValueError("a floe needs at least 2x2x2 spheres")


def floe_class_edges(radius: float, n_min: int, n_max: int, n_classes: int) -> np.ndarray:
    """Geometric class edges covering the resolved floe diameters (with a margin)."""
    edges = np.geomspace(
        2.0 * radius * n_min, 2.0 * radius * n_max, n_classes + 1
    )
    edges[0] *= 0.9
    edges[-1] *= 1.1
    return edges


@dataclass
class FieldHistory:
    """Everything the field run measured."""

    config: FieldConfig
    times: np.ndarray
    sizes: list[np.ndarray]
    temperature: float
    volume: float
    class_edges: np.ndarray
    class_diameters: np.ndarray
    contacts_calibration: np.ndarray
    contacts_forecast: np.ndarray
    calibration_span: float
    forecast_span: float
    g_floe: np.ndarray
    bond_breaks: int
    initial_sizes: np.ndarray

    @property
    def mean_size(self) -> np.ndarray:
        return np.array(
            [float(np.mean(s)) if s.size else np.nan for s in self.sizes], dtype=np.float64
        )

    @property
    def n_floes(self) -> np.ndarray:
        return np.array([int(s.size) for s in self.sizes])

    def class_counts(self, sizes) -> np.ndarray:
        index = np.clip(
            np.digitize(np.asarray(sizes), self.class_edges) - 1, 0, self.class_diameters.size - 1
        )
        return np.bincount(index, minlength=self.class_diameters.size).astype(np.float64)

    def as_dict(self) -> dict:
        return dict(
            times=[float(v) for v in self.times],
            mean_size=[float(v) for v in self.mean_size],
            n_floes=[int(v) for v in self.n_floes],
            sizes=[[float(x) for x in s] for s in self.sizes],
            temperature=float(self.temperature),
            volume=float(self.volume),
            class_edges=[float(v) for v in self.class_edges],
            class_diameters=[float(v) for v in self.class_diameters],
            contacts_calibration=[[int(v) for v in row] for row in self.contacts_calibration],
            contacts_forecast=[[int(v) for v in row] for row in self.contacts_forecast],
            calibration_span=float(self.calibration_span),
            forecast_span=float(self.forecast_span),
            g_floe=[[float(v) for v in row] for row in self.g_floe],
            bond_breaks=int(self.bond_breaks),
            initial_sizes=[float(v) for v in self.initial_sizes],
            breakage=bool(self.config.breakage),
            seed=int(self.config.seed),
            temperature_target=float(self.config.temperature),
            sphere_fraction=float(self.config.sphere_fraction),
            specs=[int(v) for v in self.config.specs],
        )


def _labels_of(assembly: SphereAssembly) -> np.ndarray:
    alive = (
        assembly.bonds[assembly.bond_alive]
        if assembly.bonds.size
        else np.zeros((0, 2), dtype=np.int64)
    )
    n = assembly.radius.size
    if not alive.size:
        return np.arange(n)
    graph = coo_matrix((np.ones(alive.shape[0]), (alive[:, 0], alive[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def _floe_centres(labels, position, box, mass, n_floes):
    """Centre of mass of every floe, unwrapped around its first sphere."""
    order = np.argsort(labels, kind="stable")
    starts = np.searchsorted(labels[order], np.arange(n_floes))
    reference = position[order[starts]]
    centres = np.zeros((n_floes, 3))
    total = np.bincount(labels, weights=mass, minlength=n_floes)
    for k in range(3):
        delta = position[:, k] - reference[labels, k]
        delta -= box[k] * np.round(delta / box[k])
        centres[:, k] = np.bincount(labels, weights=mass * delta, minlength=n_floes)
    return centres / total[:, None] + reference


def floe_contact_value(snapshots, *, n_classes: int, sphere_radius: float,
                       volume: float) -> np.ndarray:
    """Contact value of the floe-centre pair distribution, per class pair.

    Floes are treated as equivalent spheres of the census diameter, so
    ``sigma_ij = (D_i + D_j)/2`` and the shell half-width is one tenth of the
    smallest sphere diameter (the same estimator as the free-sphere gas).  The
    uncorrelated reference is the ideal-gas pair count ``N_a N_b / V`` times the
    shell volume, accumulated per pair because ``sigma`` depends on the pair.
    """
    counts = np.zeros((n_classes, n_classes))
    expected = np.zeros((n_classes, n_classes))
    samples = 0
    shell = 0.1 * (2.0 * sphere_radius)
    for centres, diameters, classes in snapshots:
        n = centres.shape[0]
        if n < 2:
            continue
        samples += 1
        for a in range(n):
            delta = centres[a + 1 :] - centres[a]
            dist = np.linalg.norm(delta, axis=1)
            sigma = 0.5 * (diameters[a] + diameters[a + 1 :])
            hit = np.abs(dist - sigma) < shell
            shell_volume = 4.0 / 3.0 * math.pi * ((sigma + shell) ** 3 - (sigma - shell) ** 3)
            ca = int(classes[a])
            for index in np.flatnonzero(hit):
                cb = int(classes[a + 1 + index])
                counts[min(ca, cb), max(ca, cb)] += 1.0
            for cb, sv in zip(classes[a + 1 :], shell_volume):
                expected[min(ca, int(cb)), max(ca, int(cb))] += sv / volume
    if samples == 0:
        return np.zeros((n_classes, n_classes))
    counts /= samples
    expected /= samples
    g = np.zeros((n_classes, n_classes))
    mask = expected > 0.0
    g[mask] = counts[mask] / expected[mask]
    return g


def run_floe_field(config: FieldConfig) -> FieldHistory:
    """Integrate the floe field and record its census and collision statistics."""
    rng = np.random.default_rng(config.seed)
    model = ContactModel()
    if not config.breakage:
        model = ContactModel(bond_tension=1e9, bond_compression=1e12)
    volume_spheres = float(
        sum(4.0 / 3.0 * math.pi * (config.radius**3) * n**3 for n in config.specs)
    )
    box_length = (volume_spheres / config.sphere_fraction) ** (1.0 / 3.0)
    box = np.array([box_length] * 3)
    radius, position, bonds, tag = place_floes(
        config.specs, box=box, rng=rng, radius=config.radius
    )
    n_floes = len(config.specs)
    mass = radius**3
    floe_mass = np.bincount(tag, weights=mass, minlength=n_floes)
    velocity = np.zeros((radius.size, 3))
    velocity += maxwellian_velocities(floe_mass, config.temperature, rng=rng)[tag]
    assembly = SphereAssembly(
        radius,
        position,
        box=box,
        model=model,
        velocity=velocity,
        bonds=bonds,
        tag=tag,
        rng=rng,
    )
    assembly.resolve_contacts(passes=3)
    dt = assembly.recommended_dt()
    total_steps = int(round(config.duration / dt))
    snapshot_every = max(1, total_steps // max(1, config.n_snapshots - 1))
    class_edges = floe_class_edges(
        config.radius, min(config.specs), max(config.specs), config.n_classes
    )
    class_diameters = np.sqrt(class_edges[:-1] * class_edges[1:])
    half_time = 0.5 * config.duration

    def census():
        labels = _labels_of(assembly)
        n_clusters = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        diameters = 2.0 * config.radius * counts ** (1.0 / 3.0)
        return labels, n_clusters, counts, diameters

    def snapshot():
        labels, n_clusters, counts, diameters = census()
        classes = np.clip(
            np.digitize(diameters, class_edges) - 1, 0, config.n_classes - 1
        )
        centres = _floe_centres(labels, assembly.position, assembly.box, mass, n_clusters)
        snapshots.append((centres, diameters, classes))
        times.append(assembly.time)
        sizes.append(diameters.copy())
        return labels, classes

    snapshots, times, sizes = [], [], []
    labels, classes = snapshot()
    sphere_class = classes[labels]
    initial_sizes = sizes[0].copy()

    contacts_calibration = np.zeros((config.n_classes, config.n_classes))
    contacts_forecast = np.zeros((config.n_classes, config.n_classes))
    assembly.contact_events.clear()
    state = {"processed": 0, "step": 0}

    def refresh_labels():
        nonlocal labels, classes, sphere_class
        labels_now = _labels_of(assembly)
        counts_now = np.bincount(labels_now).astype(np.float64)
        diameters_now = 2.0 * config.radius * counts_now ** (1.0 / 3.0)
        classes_now = np.clip(
            np.digitize(diameters_now, class_edges) - 1, 0, config.n_classes - 1
        )
        labels, classes = labels_now, classes_now
        sphere_class = classes[labels]

    def on_step(_assembly):
        nonlocal labels, classes, sphere_class
        state["step"] += 1
        if state["step"] % config.label_interval == 0:
            refresh_labels()
        events = assembly.contact_events
        processed = state["processed"]
        while processed < len(events):
            stamp, i, j, _v = events[processed]
            processed += 1
            a, b = int(sphere_class[i]), int(sphere_class[j])
            if a == b:
                continue                     # intra-floe contact, not a floe collision
            target = contacts_calibration if stamp < half_time else contacts_forecast
            target[min(a, b), max(a, b)] += 1.0
        state["processed"] = processed
        if state["step"] % snapshot_every == 0:
            snapshot()

    assembly.advance(
        dt,
        total_steps,
        cluster_temperature=config.temperature,
        thermostat_interval=5,
        record_contacts=True,
        callback=on_step,
        callback_interval=1,
    )
    spans = (half_time, max(assembly.time - half_time, 1e-12))
    g_floe = floe_contact_value(
        snapshots[: max(2, len(snapshots) // 2)],
        n_classes=config.n_classes,
        sphere_radius=config.radius,
        volume=float(np.prod(box)),
    )
    return FieldHistory(
        config=config,
        times=np.array(times),
        sizes=sizes,
        temperature=float(assembly.floe_temperature()),
        volume=float(np.prod(box)),
        class_edges=class_edges,
        class_diameters=class_diameters,
        contacts_calibration=contacts_calibration + (contacts_calibration.T - np.diag(np.diag(contacts_calibration))),
        contacts_forecast=contacts_forecast + (contacts_forecast.T - np.diag(np.diag(contacts_forecast))),
        calibration_span=float(spans[0]),
        forecast_span=float(spans[1]),
        g_floe=g_floe,
        bond_breaks=int(np.sum(~assembly.bond_alive)),
        initial_sizes=initial_sizes,
    )


def floe_kernel_from_events(
    contacts: np.ndarray,
    *,
    class_counts: np.ndarray,
    volume: float,
    span: float,
    mean_relative_speed,
) -> np.ndarray:
    """Floe collision kernel ``K_ij`` from the counted inter-floe events.

    ``class_counts`` are the mean numbers of floes per class during the window, so
    ``n_i = N_i / V`` and ``K_ij = rate_ij / (n_i n_j)`` (with the ``1/2`` for
    ``i == j``); ``mean_relative_speed`` is only used for the diagnostic
    comparison with the Enskog prediction and does not enter ``K``.
    """
    contacts = np.asarray(contacts, dtype=np.float64)
    density = np.asarray(class_counts, dtype=np.float64) / float(volume)
    rate = contacts / (float(span) * float(volume))
    kernel = np.zeros_like(contacts)
    for a in range(contacts.shape[0]):
        for b in range(contacts.shape[1]):
            denom = density[a] * density[b]
            if denom <= 0.0:
                continue
            kernel[a, b] = rate[a, b] / denom
            if a == b:
                kernel[a, b] *= 2.0
    return kernel