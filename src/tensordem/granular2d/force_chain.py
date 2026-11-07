"""Force-chain network extraction, load statistics and percolation on the contact graph.

Physics background
------------------
In a dense frictional granular packing the contact forces are strongly
heterogeneous: the probability density of the normal force decays exponentially
above the mean, and the load is carried by a sparse network of *force chains*.
The classic operational definition (Radjai et al. 1996; Peters et al. 2005) is a
threshold on the normal contact force,

.. math:: |f_c| > \\bar f + k\\,\\sigma_f ,\\qquad k = 1 ,

that separates the "strong" network from the weak, roughly isotropic background.
The strong network is then decomposed into connected components on the contact
graph; a component percolates when it spans the confining gap, which is the
mechanical definition of *jamming* used by this module: a load-bearing backbone
that connects the two channel banks while the bulk velocity has collapsed.

All functions are pure NumPy/SciPy and operate on the contact records produced by
:mod:`tensordem.granular2d.engine`.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "strong_force_mask",
    "coordination_number",
    "load_bearing_ratio",
    "fabric_tensor",
    "chain_components",
    "chain_length_distribution",
    "spanning_axis",
    "analyze_force_network",
]


def _disk_contacts(system) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contacts = system.contacts
    if "i" not in contacts or len(contacts["i"]) == 0:
        return (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros(0), np.zeros((0, 2)))
    kind = np.asarray(contacts["kind"])
    disk = kind == 0
    i = np.asarray(contacts["i"])[disk].astype(np.int64)
    j = np.asarray(contacts["j"])[disk].astype(np.int64)
    f = np.asarray(contacts["f_on_i"])[disk]
    n = np.asarray(contacts["n"])[disk]
    return i, j, np.linalg.norm(f, axis=1), n


def strong_force_mask(magnitudes, *, k: float = 1.0, reference: str = "mean") -> np.ndarray:
    """Boolean mask of contacts whose force exceeds ``mean + k * std``.

    ``reference='normal'`` uses the normal-force statistics instead of the total
    contact force magnitude; the default follows the |f| definition.
    """
    mags = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if len(mags) == 0:
        return np.zeros(0, dtype=bool)
    if not np.isfinite(mags).all():
        raise ValueError("magnitudes must be finite")
    if k < 0.0:
        raise ValueError("k must be nonnegative")
    if reference not in ("mean", "median"):
        raise ValueError("reference must be 'mean' or 'median'")
    centre = float(mags.mean()) if reference == "mean" else float(np.median(mags))
    threshold = centre + k * float(mags.std())
    return mags > threshold


def coordination_number(n_particles: int, i, j, mask=None) -> float:
    """Mean number of load-carrying contacts per particle ``Z = 2 M / N``."""
    i = np.asarray(i, dtype=np.int64).reshape(-1)
    j = np.asarray(j, dtype=np.int64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        i, j = i[m], j[m]
    return float(2.0 * len(i) / max(int(n_particles), 1))


def load_bearing_ratio(magnitudes, mask) -> float:
    """Share of the total contact force carried by the strong network."""
    mags = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    total = float(mags.sum())
    if total <= 0.0:
        return 0.0
    return float(mags[m].sum() / total)


def fabric_tensor(normals, mask=None) -> dict[str, float]:
    """Contact fabric tensor ``F_ab = <n_a n_b>`` and its anisotropy.

    ``anisotropy = (lambda_max - lambda_min)/(lambda_max + lambda_min)`` is zero
    for an isotropic contact network (both eigenvalues 1/2 in 2D) and tends to 1
    for a perfectly aligned one.
    """
    nrm = np.asarray(normals, dtype=np.float64)
    if nrm.ndim != 2 or nrm.shape[1] != 2:
        raise ValueError("normals must have shape (M, 2)")
    if mask is not None:
        nrm = nrm[np.asarray(mask, dtype=bool)]
    if len(nrm) == 0:
        return {"tensor": np.zeros((2, 2)), "anisotropy": 0.0,
                "eigenvalues": np.zeros(2), "n_contacts": 0}
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
    tensor = np.einsum("ma,mb->ab", nrm, nrm) / len(nrm)
    eig = np.linalg.eigvalsh(0.5 * (tensor + tensor.T))
    eig = np.sort(np.maximum(eig, 0.0))
    denom = float(eig.sum())
    aniso = float((eig[-1] - eig[0]) / denom) if denom > 0.0 else 0.0
    return {"tensor": tensor, "anisotropy": aniso, "eigenvalues": eig,
            "n_contacts": int(len(nrm))}


def chain_components(n_particles: int, i, j, mask=None):
    """Connected components of the (optionally thresholded) contact graph."""
    i = np.asarray(i, dtype=np.int64).reshape(-1)
    j = np.asarray(j, dtype=np.int64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        i, j = i[m], j[m]
    if len(i) == 0:
        return np.arange(int(n_particles)), np.ones(int(n_particles), dtype=np.int64)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(int(n_particles), int(n_particles)))
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    return labels, sizes


def chain_length_distribution(system, labels, sizes, *, min_size: int = 2) -> dict[str, np.ndarray | float]:
    """Chain length statistics of the components with ``size >= min_size``.

    ``length`` is measured both as particle count and as the geometric extent
    (maximum pairwise distance) of the component, which is the quantity that
    matters for gap spanning.
    """
    labels = np.asarray(labels, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    keep = np.flatnonzero(sizes >= min_size)
    counts = []
    extents = []
    for comp in keep:
        members = np.flatnonzero(labels == comp)
        if len(members) < 2:
            counts.append(len(members))
            extents.append(0.0)
            continue
        pts = system.pos[members]
        if system.periodic_x is not None:
            # minimum-image extent along the periodic axis
            x = pts[:, 0]
            span = system.periodic_x - (np.max(x) - np.min(x))
        else:
            span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        dy = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        counts.append(len(members))
        extents.append(float(np.hypot(span, dy)))
    counts = np.asarray(counts)
    extents = np.asarray(extents)
    if len(counts) == 0:
        return {"length": counts, "extent": extents, "p_length": counts,
                "mean_length": 0.0, "max_length": 0.0, "n_chains": 0}
    values, freq = np.unique(counts, return_counts=True)
    return {
        "length": counts,
        "extent": extents,
        "p_length_values": values,
        "p_length_prob": freq / freq.sum(),
        "mean_length": float(counts.mean()),
        "max_length": float(counts.max()),
        "mean_extent": float(extents.mean()),
        "n_chains": int(len(counts)),
    }


def spanning_axis(system, labels, *, axis: int = 1, band: float | None = None,
                  bounds=None) -> dict[str, object]:
    """Detect components that bridge both confining boundaries along ``axis``.

    ``band`` defaults to two mean radii, i.e. a chain counts as touching a bank
    when its outermost particle is within one particle diameter of that bank.
    Returns a mask over components plus the bridging component ids.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
    elif len(system.walls):
        pts = system.walls.reshape(-1, 2)
        lo, hi = float(pts[:, axis].min()), float(pts[:, axis].max())
    else:
        raise ValueError("bounds are required for a system without walls")
    if band is None:
        band = 2.0 * system.mean_radius
    coord = system.pos[:, axis]
    n_comp = int(labels.max()) + 1 if len(labels) else 0
    touching_lo = np.zeros(n_comp, dtype=bool)
    touching_hi = np.zeros(n_comp, dtype=bool)
    np.logical_or.at(touching_lo, labels, coord <= lo + band)
    np.logical_or.at(touching_hi, labels, coord >= hi - band)
    bridging = touching_lo & touching_hi
    return {"bridging_components": np.flatnonzero(bridging),
            "bridging": bool(bridging.any()),
            "lo": lo, "hi": hi, "band": band,
            "n_bridging": int(bridging.sum())}


def analyze_force_network(system, *, k: float = 1.0, axis: int = 1, band: float | None = None,
                          bounds=None) -> dict[str, object]:
    """Full force-chain audit of the current contact state of ``system``."""
    if not system.contacts:
        system.contact_forces()
    i, j, mags, normals = _disk_contacts(system)
    mask = strong_force_mask(mags, k=k)
    labels_strong, sizes_strong = chain_components(system.n, i, j, mask)
    labels_all, sizes_all = chain_components(system.n, i, j)
    dist = chain_length_distribution(system, labels_strong, sizes_strong, min_size=2)
    span = spanning_axis(system, labels_strong, axis=axis, band=band, bounds=bounds)
    span_all = spanning_axis(system, labels_all, axis=axis, band=band, bounds=bounds)
    fabric = fabric_tensor(normals, mask)
    fabric_all = fabric_tensor(normals)
    return {
        "n_particles": int(system.n),
        "n_contacts": int(len(i)),
        "n_strong": int(mask.sum()),
        "z_all": coordination_number(system.n, i, j),
        "z_strong": coordination_number(system.n, i, j, mask),
        "load_bearing_ratio": load_bearing_ratio(mags, mask),
        "fabric_anisotropy": fabric["anisotropy"],
        "fabric_anisotropy_all": fabric_all["anisotropy"],
        "fabric_tensor": fabric["tensor"].ravel(),
        "chain": dist,
        "spanning": span,
        "spanning_all": span_all,
        "labels_strong": labels_strong,
        "mask_strong": mask,
        "magnitudes": mags,
        "force_mean": float(mags.mean()) if len(mags) else 0.0,
        "force_std": float(mags.std()) if len(mags) else 0.0,
        "force_threshold": (float(mags.mean() + k * mags.std()) if len(mags) else 0.0),
    }