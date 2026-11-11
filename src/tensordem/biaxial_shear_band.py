"""Shear-band inclination in a plane-strain biaxial compression test.

Physics problem
---------------
A rectangular, freeze-bonded disc specimen is compressed in plane strain between
a displacement-driven top platen and a fixed bottom wall, while the two lateral
walls servo to a constant confining pressure ``sigma_3``.  A dense, bonded
frictional material localises its deformation into one (or a conjugate pair of)
narrow shear band(s) at/after the peak deviator stress.  The inclination of the
band with respect to the minor principal stress axis ``sigma_3`` is a classical
validation quantity because three competing relations predict it from only the
internal friction angle ``phi`` and the dilation angle ``psi``:

=============================  ============================================
Mohr-Coulomb (plasticity)      ``theta = 45 deg + phi/2``
Roscoe (1970, Geotechnique)    ``theta = 45 deg + psi/2``
Arthur et al. (1977, Geotech.) ``theta = 45 deg + (phi + psi)/4``
=============================  ============================================

``theta`` is measured from the ``sigma_3`` axis, which is horizontal in this
cell (``sigma_1`` is vertical).  Both ``phi`` and ``psi`` are *measured in the
same run* (secant friction angle at the deviator peak; post-peak volumetric
rate), so no parameter is borrowed from the theory that predicts the angle.

Measurement operator
--------------------
The band angle is measured from the particle displacement field, not read off
a picture:

1. a local deformation gradient is fitted by least squares to the ``k``
   nearest neighbours of every particle, giving the local maximum shear strain
   ``gamma = sqrt((e_xx - e_yy)**2 + (2 e_xy)**2)``;
2. the orientation of the ridge of ``gamma`` is found by a Radon projection:
   for each candidate inclination ``alpha`` the particles are projected onto
   the band normal ``n(alpha)``.  A planar band collapses into a single
   projection bin, so the fraction ``eta2`` of the weighted variance explained
   by the bin means peaks at the band orientation (the standard
   projection/radon localisation estimator; cf. Roscoe 1970 for the biaxial
   criterion, Desrues & Viggiani 2004 for the experimental methodology).  The
   operator is validated on synthetic bands of known angle in
   ``tests/test_biaxial_shear_band.py``;
3. the measurement is applied to the *incremental* displacement between
   successive strain snapshots, so the elastic + homogeneous compression
   background of the whole test is not carried into the orientation estimate;
4. conjugate bands (``alpha`` and ``180 - alpha``) are geometrically the same
   inclination to the vertical, so the reported angle is folded into
   ``[0, 90)`` degrees from the ``sigma_3`` axis.

Everything is SI.  The specimen is a jittered triangular disc lattice with the
same contact/bond laws as :mod:`tensordem.ridge_dem` (``Plane``, frozen bonds),
so this module adds a *test and a measurement*, not a second contact model.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import torch

from .ridge_dem import Plane, RidgeAssembly, RidgeMaterial, _advance_planes

__all__ = [
    "BandMeasurementError",
    "theory_angles_deg",
    "fold_angle_deg",
    "local_shear_strain",
    "radon_band_angle",
    "measure_band",
    "run_biaxial_shear_band",
]


class BandMeasurementError(ValueError):
    """Raised when a band angle cannot be measured from the supplied field."""


# ---------------------------------------------------------------------------
# theory
# ---------------------------------------------------------------------------
def theory_angles_deg(phi_deg: float, psi_deg: float) -> dict[str, float]:
    """Inclination (deg from the ``sigma_3`` axis) predicted by the three laws."""
    phi = float(phi_deg)
    psi = float(psi_deg)
    return {
        "mohr_coulomb": 45.0 + 0.5 * phi,
        "roscoe": 45.0 + 0.5 * psi,
        "arthur": 45.0 + 0.25 * (phi + psi),
    }


def fold_angle_deg(angle_deg: float) -> float:
    """Fold a band inclination into ``[0, 90]``; conjugates share the inclination.

    A band at ``alpha`` and its conjugate at ``180 - alpha`` are mirror images
    about the vertical and have the same angle to the vertical axis, hence the
    same physical inclination to ``sigma_3`` once expressed in ``[0, 90]``.
    """
    a = float(angle_deg) % 180.0
    return 180.0 - a if a > 90.0 else a


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def _affine_residual(x_ref: np.ndarray, x_now: np.ndarray) -> np.ndarray:
    """Displacement with the best-fit rigid-body + homogeneous part removed."""
    disp = x_now - x_ref
    centre = x_ref.mean(axis=0)
    d = x_ref - centre
    u = disp - disp.mean(axis=0)
    grad, *_ = np.linalg.lstsq(d, u, rcond=None)          # (2, 2): u = d @ grad
    return u - d @ grad


def local_shear_strain(x_ref: np.ndarray, x_now: np.ndarray, *, k: int = 8,
                       regularisation: float = 1e-9) -> np.ndarray:
    """Local maximum shear strain of the displacement field ``x_ref -> x_now``.

    A deformation gradient ``G`` (``u = G X``) is fitted by least squares to the
    ``k`` nearest neighbours of every particle; the returned value is the local
    maximum shear strain ``sqrt((e_xx - e_yy)**2 + (2 e_xy)**2)`` of the
    symmetric part ``e = (G + G^T)/2``.
    """
    ref = np.asarray(x_ref, dtype=np.float64)
    now = np.asarray(x_now, dtype=np.float64)
    if ref.shape != now.shape or ref.ndim != 2 or ref.shape[1] != 2:
        raise BandMeasurementError("x_ref and x_now must both be (N, 2)")
    count = ref.shape[0]
    if count < k + 1:
        raise BandMeasurementError(f"need at least {k + 1} particles, got {count}")
    disp = _affine_residual(ref, now)

    xt = torch.as_tensor(ref)
    dist = torch.cdist(xt, xt)
    kk = min(max(int(k), 3), count)
    idx = torch.topk(dist, kk, largest=False).indices          # (N, k)
    neighbours = ref[idx]                                      # (N, k, 2)
    dx = neighbours - neighbours.mean(axis=1, keepdims=True)
    du = disp[idx] - disp[idx].mean(axis=1, keepdims=True)
    a = np.einsum("nki,nkj->nij", dx, dx)
    b = np.einsum("nki,nkj->nij", du, dx)
    trace = np.trace(a, axis1=1, axis2=2)[:, None, None]
    a = a + regularisation * np.eye(2)[None, :, :] * np.where(trace > 0, trace, 1.0)
    grad = np.linalg.solve(a.transpose(0, 2, 1), b.transpose(0, 2, 1)).transpose(0, 2, 1)
    e_xx = grad[:, 0, 0]
    e_xy = 0.5 * (grad[:, 0, 1] + grad[:, 1, 0])
    e_yy = grad[:, 1, 1]
    return np.sqrt((e_xx - e_yy) ** 2 + 4.0 * e_xy ** 2)


def radon_band_angle(x: Sequence[float], y: Sequence[float],
                     weight: Sequence[float], *,
                     n_bins: int | None = None,
                     resolution_deg: float = 1.0) -> dict[str, Any]:
    """Band inclination (deg from +x) from a Radon/projection focus search.

    For every candidate inclination ``alpha`` the samples are projected on the
    band normal ``n = (-sin alpha, cos alpha)``; a band at ``alpha`` collapses
    into one bin, maximising the fraction ``eta2`` of the weighted variance
    explained by the bin means.  A conjugate peak at least 10 deg away is
    reported when it carries at least 60 % of the dominant score.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.float64).ravel()
    if not (xs.shape == ys.shape == w.shape):
        raise BandMeasurementError("x, y and weight must have the same length")
    if xs.size < 20:
        raise BandMeasurementError("at least 20 samples are needed")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(w).all()):
        raise BandMeasurementError("samples must be finite")
    w = np.clip(w, 0.0, None)
    if not w.sum() > 0.0:
        raise BandMeasurementError("weights sum to zero")
    w = w / w.sum()

    angles = np.arange(0.0, 180.0, float(resolution_deg))
    bins = int(n_bins) if n_bins else max(8, min(40, int(round(math.sqrt(xs.size)))))
    xs = (xs - xs.min()) / max(float(np.ptp(xs)), 1e-12)
    ys = (ys - ys.min()) / max(float(np.ptp(ys)), 1e-12)
    mu = float(w.mean())
    total_ss = float(np.sum((w - mu) ** 2))
    if not total_ss > 0.0:
        raise BandMeasurementError("degenerate (constant) weight field")

    eta = np.empty_like(angles)
    for i, alpha in enumerate(angles):
        rad = math.radians(alpha)
        s = xs * (-math.sin(rad)) + ys * math.cos(rad)
        s = (s - s.min()) / max(float(np.ptp(s)), 1e-12)
        index = np.minimum((s * bins).astype(np.int64), bins - 1)
        count = np.bincount(index, minlength=bins).astype(np.float64)
        wsum = np.bincount(index, weights=w, minlength=bins)
        occupied = count > 0
        bin_mean = np.zeros(bins)
        bin_mean[occupied] = wsum[occupied] / count[occupied]
        between = float(np.sum(count[occupied] * (bin_mean[occupied] - mu) ** 2))
        eta[i] = between / total_ss

    best = int(np.argmax(eta))
    peak_alpha = float(angles[best])
    peak_eta = float(eta[best])
    separation = np.abs(((angles - peak_alpha + 90.0) % 180.0) - 90.0)
    far = separation > 10.0
    conjugate = None
    if far.any():
        j = int(np.argmax(np.where(far, eta, -np.inf)))
        if eta[j] >= 0.6 * peak_eta:
            conjugate = {"angle_deg": float(angles[j]), "eta2": float(eta[j])}
    rest = eta[far]
    null_mean = float(np.mean(rest))
    null_std = float(np.std(rest))
    return {
        "angle_deg": peak_alpha,
        "folded_angle_deg": fold_angle_deg(peak_alpha),
        "eta2": peak_eta,
        "conjugate": conjugate,
        "null_eta2": null_mean,
        "z_score": (peak_eta - null_mean) / null_std if null_std > 0 else 0.0,
        "angles_deg": angles,
        "eta2_curve": eta,
        "bins": bins,
    }


def measure_band(x_ref: np.ndarray, x_now: np.ndarray, *,
                 bounds: tuple[float, float, float, float] | None = None,
                 k: int = 8) -> dict[str, Any]:
    """Localise and orient the shear band between two configurations.

    ``bounds`` = ``(xmin, xmax, ymin, ymax)`` of the measurement window; a
    boundary layer is normally excluded because the rough walls nucleate their
    own boundary bands that are not the material's shear band.
    """
    ref = np.asarray(x_ref, dtype=np.float64)
    now = np.asarray(x_now, dtype=np.float64)
    if bounds is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        inside = ((ref[:, 0] >= xmin) & (ref[:, 0] <= xmax)
                  & (ref[:, 1] >= ymin) & (ref[:, 1] <= ymax))
    else:
        inside = np.ones(ref.shape[0], dtype=bool)
    if int(inside.sum()) < 30:
        raise BandMeasurementError("too few particles inside the measurement window")
    gamma = local_shear_strain(ref[inside], now[inside], k=k)
    if not np.isfinite(gamma).all() or gamma.mean() <= 0.0:
        raise BandMeasurementError("local strain field is degenerate")
    out = radon_band_angle(ref[inside, 0], ref[inside, 1], gamma)
    out["localisation_index"] = float(gamma.std() / gamma.mean())
    out["particles"] = int(inside.sum())
    out["gamma_mean"] = float(gamma.mean())
    return out


# ---------------------------------------------------------------------------
# specimen
# ---------------------------------------------------------------------------
def _make_specimen(material: RidgeMaterial, *, columns: int, rows: int,
                   confining_stress: float, thickness: float = 1.0,
                   jitter: float = 0.03, relax_iterations: int = 60,
                   bond_cutoff: float = 1.2, servo_gain: float = 1.0e-4,
                   servo_velocity_limit: float = 2.0) -> tuple[RidgeAssembly, dict[str, Any]]:
    """Jittered triangular disc lattice in a four-plane servo biaxial cell.

    The lattice is dense (discs nominally touching) but the positions are
    jittered by ``jitter`` x spacing and overlaps are relaxed, so the specimen
    has no lattice-aligned failure planes -- a regular lattice would localise
    along its own 60 deg rows and bias the measured band angle.
    """
    radius = material.particle_radius
    spacing = 2.0 * radius
    row_spacing = spacing * math.sqrt(3.0) / 2.0
    width = (columns - 1) * spacing
    height = (rows - 1) * row_spacing
    generator = torch.Generator().manual_seed(material.seed + 101)
    points: list[tuple[float, float]] = []
    for row in range(rows):
        offset = 0.5 * spacing if row % 2 else 0.0
        for column in range(columns):
            x = -0.5 * width + column * spacing + offset
            if abs(x) > 0.5 * width + 1e-9:
                continue
            points.append((x, -0.5 * height + row * row_spacing))
    pos = torch.tensor(points, dtype=torch.float64)
    pos = pos + (jitter * spacing) * torch.randn(pos.shape, dtype=torch.float64,
                                                generator=generator)
    # relax residual overlaps so the contact forces start at zero
    count = int(pos.shape[0])
    pair_i, pair_j = torch.triu_indices(count, count, offset=1)
    for _ in range(int(relax_iterations)):
        delta = pos[pair_j] - pos[pair_i]
        distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        overlap = (spacing - distance).clamp_min(0.0)
        if not bool(overlap.any()):
            break
        unit = delta / distance[:, None]
        push = 0.5 * overlap[:, None] * unit
        update = torch.zeros_like(pos)
        update.index_add_(0, pair_i, -push)
        update.index_add_(0, pair_j, push)
        pos = pos + update
    delta = pos[pair_j] - pos[pair_i]
    distance = torch.linalg.vector_norm(delta, dim=1)
    keep = distance < bond_cutoff * spacing
    bonds = torch.stack((pair_i[keep], pair_j[keep]), dim=1)
    live_bond = torch.rand(bonds.shape[0], generator=generator) < material.freeze_bond_fraction
    bonds = bonds[live_bond]

    half_w = 0.5 * width + radius
    half_h = 0.5 * height + radius
    wall_stiffness = 4.0 * material.contact_stiffness
    planes = [
        Plane((1.0, 0.0), (-half_w, 0.0), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient,
              servo_stress=confining_stress, servo_width=height,
              servo_gain=servo_gain, servo_velocity_limit=servo_velocity_limit),
        Plane((-1.0, 0.0), (half_w, 0.0), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient,
              servo_stress=confining_stress, servo_width=height,
              servo_gain=servo_gain, servo_velocity_limit=servo_velocity_limit),
        Plane((0.0, 1.0), (0.0, -half_h), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient),
        Plane((0.0, -1.0), (0.0, half_h), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=material.friction_coefficient),
    ]
    assembly = RidgeAssembly(pos, torch.full((count,), radius, dtype=torch.float64),
                             torch.full((count,), material.particle_mass, dtype=torch.float64),
                             bonds, material, planes=planes, thickness=thickness)
    geometry = {"width": width, "height": height, "particles": count,
                "bonds": int(bonds.shape[0]), "confining_stress": float(confining_stress),
                "half_w": half_w, "half_h": half_h, "wall_stiffness": wall_stiffness}
    return assembly, geometry


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_biaxial_shear_band(*, confining_stress: float = 100.0e3,
                           columns: int = 20, rows: int = 40,
                           radius: float = 0.75,
                           strain_rate: float = 0.05,
                           strain_target: float = 0.15,
                           consolidation_steps: int = 4000,
                           snapshot_strain: float = 0.01,
                           jitter: float = 0.03,
                           wall_margin_diameters: float = 2.0,
                           k_local: int = 8,
                           servo_gain: float = 1.0e-4,
                           servo_velocity_limit: float = 2.0,
                           material_overrides: dict[str, Any] | None = None,
                           ) -> dict[str, Any]:
    """Run one constant-confining-pressure biaxial test and measure its band.

    ``strain_rate`` (1/s) is applied on the *current* height, so the axial
    strain rate is constant; the implied inertial number is reported so the
    quasi-static assumption is auditable.  Returns a JSON-serialisable dict.
    """
    overrides = dict(material_overrides or {})
    overrides.setdefault("particle_radius", float(radius))
    material = RidgeMaterial(**overrides)
    assembly, geometry = _make_specimen(
        material, columns=columns, rows=rows, confining_stress=confining_stress,
        jitter=jitter, servo_gain=servo_gain,
        servo_velocity_limit=servo_velocity_limit)
    dt = material.recommended_dt
    thickness = assembly.thickness

    for _ in range(int(consolidation_steps)):
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)

    left, right, bottom, top = assembly.planes
    width0 = float(right.point[0] - left.point[0])
    height0 = float(top.point[1] - bottom.point[1])
    prev = assembly.pos.detach().numpy().copy()
    reference = prev.copy()
    strain = 0.0
    history: list[dict[str, Any]] = []
    bands: list[dict[str, Any]] = []
    next_snapshot = snapshot_strain
    diameter = 2.0 * material.particle_radius
    max_steps = int(math.ceil(strain_target / (strain_rate * dt))) + 10
    for _step in range(max_steps):
        left, right, bottom, top = assembly.planes
        height = float(top.point[1] - bottom.point[1])
        assembly.planes[3] = replace(top, velocity=float(strain_rate * height))
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
        plane_forces = out["plane_forces"]
        left, right, bottom, top = assembly.planes
        width = float(right.point[0] - left.point[0])
        height = float(top.point[1] - bottom.point[1])
        sigma_v = -float(plane_forces[2][1]) / (width * thickness)
        sigma_h = 0.5 * (-float(plane_forces[0][0]) / (height * thickness)
                         + float(plane_forces[1][0]) / (height * thickness))
        strain = (height0 - height) / height0
        e_lat = (width0 - width) / width0
        history.append({"axial_strain": strain, "sigma_v": sigma_v,
                        "sigma_h": sigma_h, "deviator": sigma_v - sigma_h,
                        "volumetric_strain": strain + e_lat})
        if strain + 1e-12 >= next_snapshot:
            next_snapshot += snapshot_strain
            window = (float(left.point[0]) + wall_margin_diameters * diameter,
                      float(right.point[0]) - wall_margin_diameters * diameter,
                      float(bottom.point[1]) + wall_margin_diameters * diameter,
                      float(top.point[1]) - wall_margin_diameters * diameter)
            current = assembly.pos.detach().numpy().copy()
            try:
                band = measure_band(prev, current, bounds=window, k=k_local)
            except BandMeasurementError as exc:      # pragma: no cover - defensive
                band = {"angle_deg": float("nan"), "folded_angle_deg": float("nan"),
                        "eta2": float("nan"), "z_score": 0.0, "error": str(exc)}
            # the cumulative field (affine part removed) carries the whole
            # post-peak history and is the less noisy orientation estimator
            try:
                cumulative = measure_band(reference, current, bounds=window, k=k_local)
            except BandMeasurementError:              # pragma: no cover - defensive
                cumulative = {"folded_angle_deg": float("nan"), "eta2": float("nan"),
                              "z_score": 0.0}
            band["axial_strain"] = strain
            band["cumulative"] = {k: v for k, v in cumulative.items()
                                  if k not in ("angles_deg", "eta2_curve")}
            bands.append(band)
            prev = current
        if strain >= strain_target:
            break

    hist = {key: np.array([row[key] for row in history]) for key in history[0]}
    window = max(1, int(round(0.01 * hist["axial_strain"].size)))
    smooth = np.convolve(hist["deviator"], np.ones(window) / window, mode="same")
    peak_index = int(np.argmax(smooth))
    sigma1 = float(hist["sigma_v"][peak_index])
    sigma3 = float(hist["sigma_h"][peak_index])
    denom = sigma1 + sigma3
    sin_phi = max(min((sigma1 - sigma3) / denom, 0.95), 0.0) if denom else 0.0
    phi_deg = math.degrees(math.asin(sin_phi))

    # dilation angle: linear fit of eps_v against eps_a over the *late* post-peak
    # window (the first post-peak increments still contain the elastic unloading
    # and the bond-breaking burst and do not represent steady dilation)
    tail = hist["axial_strain"].size - peak_index
    start = peak_index + max(5, int(round(0.4 * tail)))
    eps_a = hist["axial_strain"][start:]
    eps_v = hist["volumetric_strain"][start:]
    slope = float(np.polyfit(eps_a, eps_v, 1)[0]) if eps_a.size >= 5 else 0.0
    psi_deg = math.degrees(math.asin(max(min(-slope, 0.95), -0.95)))

    theory = theory_angles_deg(phi_deg, psi_deg)
    # Representative band: the *late post-peak* window only.  Early snapshots
    # contain a high-eta2 horizontal compaction front driven by the servo
    # (z > 3 even though there is no shear band yet), so selecting the global
    # eta2 maximum would lock onto that transient instead of the band.  The
    # headline angle is the mean of the incremental-field estimates over the
    # last 60 % of the post-peak record and is only declared *resolved* when
    # those snapshots agree (spread < 10 deg over at least three snapshots).
    peak_strain = float(hist["axial_strain"][peak_index])
    final_strain = float(hist["axial_strain"][-1])
    tail_start = peak_strain + 0.4 * max(final_strain - peak_strain, 0.0)
    late = [b for b in bands if b["axial_strain"] >= tail_start
            and np.isfinite(b.get("folded_angle_deg", np.nan))]
    post_peak = [float(b["folded_angle_deg"]) for b in late]
    angle_mean = float(np.mean(post_peak)) if post_peak else float("nan")
    angle_std = float(np.std(post_peak)) if post_peak else float("nan")
    resolved = bool(len(post_peak) >= 3 and angle_std < 10.0)
    angle = angle_mean if resolved else float("nan")
    best = max(late, key=lambda b: b.get("cumulative", {}).get("z_score", -np.inf)) \
        if late else None
    clean = lambda b: {k: v for k, v in b.items() if k not in ("angles_deg", "eta2_curve")}
    inertial = strain_rate * diameter * math.sqrt(
        material.ice_density / max(confining_stress, 1.0))
    return {
        "confining_stress": float(confining_stress),
        "specimen": {"columns": int(columns), "rows": int(rows),
                     "particles": int(assembly.pos.shape[0]),
                     "bonds": int(geometry["bonds"]),
                     "radius": float(material.particle_radius),
                     "jitter": float(jitter),
                     "width0": width0, "height0": height0, "dt": float(dt)},
        "loading": {"axial_strain_rate": float(strain_rate),
                    "final_axial_strain": float(strain),
                    "steps": int(len(hist["axial_strain"])),
                    "inertial_number": float(inertial)},
        "peak": {"axial_strain": float(hist["axial_strain"][peak_index]),
                 "sigma_v": sigma1, "sigma_h": sigma3,
                 "deviator": float(sigma1 - sigma3),
                 "friction_angle_deg": phi_deg,
                 "dilation_angle_deg": psi_deg,
                 "sin_phi": sin_phi,
                 "post_peak_d_eps_v_d_eps_a": slope},
        "band": {"angle_deg": angle, "angle_mean_deg": angle_mean,
                 "angle_std_deg": angle_std, "resolved": resolved,
                 "late_snapshots": len(post_peak),
                 "representative": clean(best) if best else None,
                 "history": [clean(b) for b in bands]},
        "theory": theory,
        "theory_relative_error": {k: (angle - v) / v for k, v in theory.items()},
        "broken_bonds": int(assembly.broken_bonds),
        "stress_path": hist,
        "note": ("plane-strain biaxial, constant confining pressure, "
                 "top platen displacement driven; gravity neglected"),
    }