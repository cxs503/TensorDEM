"""Diagnostics for the plane-strain biaxial shear-band angle (CASE-BIAXIAL-SHEAR-BAND).

The delivered biaxial case measures a band inclination of ``44.3 +- 0.8`` /
``44.7 +- 0.5`` deg from ``sigma_3``, i.e. **25-26 % below** the 60 deg dense-sand
reference of Desrues & Viggiani (2004), and below all three classical
predictions built from the run's *own* ``phi`` and ``psi``
(Mohr-Coulomb, Roscoe 1970, Arthur 1977).  That is an honest FAIL, but the
failure mechanism was left as two untested hypotheses in
``docs/BIAXIAL_SHEAR_BAND.md`` section (6).  This module supplies the two
measurements needed to decide between them:

``estimator_bias_sweep``  **falsifies hypothesis (a)** -- the suspicion that the
    global affine removal in :func:`tensordem.biaxial_shear_band.measure_band`
    rotates the residual ridge towards the 45 deg maximum-shear direction.  A
    synthetic band of *known* inclination is injected into the real lattice
    geometry, displaced exactly like a test snapshot (homogeneous biaxial
    background + a finite-width tangential slip band, optional band dilation)
    and pushed through the unmodified production pipeline
    (``measure_band`` = affine removal -> kNN strain -> Radon focus).  The
    recovery error is the estimator's bias.

``band_local_dilation``  replaces the **global** dilation angle of the original
    case by the dilation angle measured *inside the band*.  This matters
    because Roscoe's relation ``theta = 45 + psi/2`` uses the dilatancy of the
    band, while the original run fed it the global ``d eps_v / d eps_a`` of a
    specimen whose compaction/dilation is partly absorbed by the servo front --
    a category mismatch (the case admits ``psi`` up to 65 deg, above ``phi``,
    which is the signature of that contamination).  A band that localises
    shear but not volume change is *by construction* an isochoric shear zone,
    and its inclination must be 45 deg if Roscoe's kinematics hold: measuring
    ``psi_band`` is therefore the decisive test of "is the 44.5 deg a model
    defect or the correct answer for this material state?".

Everything is SI and pure ``numpy`` (no DEM stepping here); the lattice
geometry comes from :func:`tensordem.biaxial_shear_band._make_specimen` or any
``(N, 2)`` array of particle centres.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .biaxial_shear_band import (
    BandMeasurementError,
    _affine_residual,
    fold_angle_deg,
    local_shear_strain,
    measure_band,
)

__all__ = [
    "band_frame",
    "legacy_radon_band_angle",
    "aspect_ratio_distortion_deg",
    "synthetic_band_configuration",
    "estimator_bias_sweep",
    "local_strain_tensor",
    "local_volumetric_strain",
    "band_mask",
    "band_local_dilation",
    "resolution_control",
]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def band_frame(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Band tangent ``t`` and normal ``n`` for an inclination from ``+x``.

    ``angle_deg`` is the inclination measured from the ``sigma_3`` axis
    (horizontal), exactly the convention of
    :func:`tensordem.biaxial_shear_band.radon_band_angle`.
    """
    rad = math.radians(float(angle_deg))
    tangent = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    normal = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    return tangent, normal


def band_mask(pos: np.ndarray, *, angle_deg: float, half_width: float,
              centre: Sequence[float] | None = None) -> np.ndarray:
    """Particles whose distance to the band centre line is ``<= half_width``."""
    points = np.asarray(pos, dtype=np.float64)
    _, normal = band_frame(angle_deg)
    origin = (np.asarray(centre, dtype=np.float64) if centre is not None
              else points.mean(axis=0))
    return np.abs((points - origin) @ normal) <= float(half_width)


# ---------------------------------------------------------------------------
# reference implementation of the *pre-fix* estimator (baseline for the audit)
# ---------------------------------------------------------------------------
def legacy_radon_band_angle(x: Sequence[float], y: Sequence[float],
                            weight: Sequence[float], *,
                            n_bins: int | None = None,
                            resolution_deg: float = 1.0) -> dict[str, Any]:
    """The pre-R2 implementation, kept **only** as an audit baseline.

    It min-max rescales ``x`` and ``y`` *independently* before projecting.  That
    map is not conformal: a physical direction ``(cos a, sin a)`` becomes
    ``(cos a / W, sin a / H)``, so the reported angle obeys

        ``tan(theta_reported) = (W / H) * tan(theta_true)``

    and an elongated specimen reports every inclination squashed towards 45 deg.
    Production code must use :func:`tensordem.biaxial_shear_band.radon_band_angle`
    instead; this copy exists so the bias can be quantified before/after and so
    that archived results (``44.25`` / ``44.67`` deg in
    ``docs/BIAXIAL_SHEAR_BAND.md``) can be inverted to their true angles.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.float64).ravel()
    if not (xs.shape == ys.shape == w.shape):
        raise BandMeasurementError("x, y and weight must have the same length")
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
    peak = float(angles[int(np.argmax(eta))])
    return {"angle_deg": peak, "folded_angle_deg": fold_angle_deg(peak),
            "eta2": float(np.max(eta)), "binning": "legacy_equal_width_normalised"}


def aspect_ratio_distortion_deg(angle_true_deg: Sequence[float] | float, *,
                                width: float, height: float) -> np.ndarray:
    """Analytic prediction of the pre-fix bias: ``atan((W/H) tan a)``.

    ``width``/``height`` are the physical extents of the measured point set
    (the min-max ranges that the legacy estimator normalised away).
    """
    angles = np.atleast_1d(np.asarray(angle_true_deg, dtype=np.float64))
    ratio = float(width) / float(height)
    return np.degrees(np.arctan(ratio * np.tan(np.radians(angles))))


# ---------------------------------------------------------------------------
# (a) estimator-bias test on a synthetic band of known inclination
# ---------------------------------------------------------------------------
def synthetic_band_configuration(pos: np.ndarray, *, angle_deg: float,
                                 slip: float, band_width: float,
                                 volumetric_strain: float = 0.0,
                                 background_axial_strain: float = 2.0e-3,
                                 centre: Sequence[float] | None = None
                                 ) -> np.ndarray:
    """Displace a lattice with a *known* shear band of inclination ``angle_deg``.

    The displacement is the sum of

    * a homogeneous plane-strain biaxial compression (``eps_yy =
      background_axial_strain``, ``eps_xx = -0.5 eps_yy``, i.e. isochoric, with
      ``sigma_1`` vertical as in the cell), and
    * a finite-width band: across the band normal ``n`` the tangential
      displacement ramps from ``0`` to ``slip`` over ``tanh(s / band_width)``,
      where ``s = (x - centre) . n``.  The tangential ramp is volume preserving
      (``t . n = 0``); an optional normal ramp of total opening
      ``volumetric_strain * band_width`` adds band dilatancy.

    Returns the displaced positions ``x_now``.
    """
    points = np.asarray(pos, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise BandMeasurementError("pos must be (N, 2)")
    if not float(band_width) > 0.0:
        raise BandMeasurementError("band_width must be positive")
    origin = (np.asarray(centre, dtype=np.float64) if centre is not None
              else points.mean(axis=0))
    tangent, normal = band_frame(angle_deg)
    s = (points - origin) @ normal
    ramp = 0.5 * (1.0 + np.tanh(s / float(band_width)))

    disp = np.zeros_like(points)
    # homogeneous biaxial background (isochoric, sigma_1 vertical)
    rel = points - origin
    disp[:, 0] += -0.5 * float(background_axial_strain) * rel[:, 0]
    disp[:, 1] += float(background_axial_strain) * rel[:, 1]
    # tangential slip band + optional band dilation
    disp += np.outer(float(slip) * ramp, tangent)
    if volumetric_strain:
        opening = float(volumetric_strain) * float(band_width)
        disp += np.outer(opening * ramp, normal)
    return points + disp


def estimator_bias_sweep(pos: np.ndarray, *, angles_deg: Sequence[float],
                         slip: float | None = None,
                         band_width: float | None = None,
                         volumetric_strain: float = 0.0,
                         measurement_window: Sequence[float] | None = None,
                         k: int = 8) -> dict[str, Any]:
    """Feed synthetic bands of known angle through the production estimator.

    ``measurement_window`` = ``(xmin, xmax, ymin, ymax)`` as in the biaxial case
    (wall margin excluded).  Defaults: slip = 4 % of the window height, band
    width = 3 particle spacings.
    """
    points = np.asarray(pos, dtype=np.float64)
    if measurement_window is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in measurement_window)
        inside = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                  & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
        window_points = points[inside]
    else:
        window_points = points
    if window_points.shape[0] < 40:
        raise BandMeasurementError("too few particles in the measurement window")
    spacing = float(np.median(np.sort(np.linalg.norm(
        window_points[:, None, :] - window_points[None, :, :], axis=2), axis=1)[:, 1]))
    height = float(np.ptp(window_points[:, 1]))
    width = float(np.ptp(window_points[:, 0]))
    width_band = float(band_width) if band_width is not None else 3.0 * spacing
    slip_value = float(slip) if slip is not None else 0.04 * height

    predicted = aspect_ratio_distortion_deg([float(a) for a in angles_deg],
                                            width=width, height=height)
    rows = []
    for k_angle, angle in enumerate(angles_deg):
        moved = synthetic_band_configuration(
            points, angle_deg=float(angle), slip=slip_value, band_width=width_band,
            volumetric_strain=volumetric_strain)
        legacy = None
        try:
            out = measure_band(points, moved, bounds=measurement_window, k=k)
            recovered = float(out["folded_angle_deg"])
            eta2 = float(out["eta2"])
            z = float(out["z_score"])
            error = recovered - float(angle)
            # same strain field, pre-fix estimator (audit baseline)
            gamma = local_shear_strain(window_points, moved[inside], k=k) \
                if measurement_window is not None else \
                local_shear_strain(points, moved, k=k)
            legacy = float(legacy_radon_band_angle(
                window_points[:, 0], window_points[:, 1], gamma)["folded_angle_deg"])
        except BandMeasurementError as exc:                 # pragma: no cover - defensive
            recovered, eta2, z, error = math.nan, math.nan, math.nan, math.nan
            rows.append({"injected_deg": float(angle), "recovered_deg": recovered,
                         "error_deg": error, "abs_error_deg": math.nan,
                         "eta2": eta2, "z_score": z, "error": str(exc)})
            continue
        rows.append({"injected_deg": float(angle), "recovered_deg": recovered,
                     "error_deg": error, "abs_error_deg": abs(error),
                     "eta2": eta2, "z_score": z,
                     "legacy_recovered_deg": legacy,
                     "legacy_error_deg": (legacy - float(angle)
                                          if legacy is not None else math.nan),
                     "analytic_prediction_deg": float(predicted[k_angle]),
                     "conjugate": out.get("conjugate")})
    errors = [row["abs_error_deg"] for row in rows if np.isfinite(row["abs_error_deg"])]
    legacy_errors = [abs(row["legacy_error_deg"]) for row in rows
                     if np.isfinite(row.get("legacy_error_deg", math.nan))]
    return {
        "angles_deg": [float(a) for a in angles_deg],
        "rows": rows,
        "legacy_max_abs_error_deg": (float(max(legacy_errors)) if legacy_errors
                                     else math.nan),
        "legacy_mean_abs_error_deg": (float(np.mean(legacy_errors)) if legacy_errors
                                      else math.nan),
        "aspect_ratio": (float(np.ptp(window_points[:, 0]) / np.ptp(window_points[:, 1]))
                         if np.ptp(window_points[:, 1]) > 0 else math.nan),
        "slip": slip_value,
        "band_width": width_band,
        "particle_spacing": spacing,
        "window_width": width,
        "window_height": height,
        "volumetric_strain": float(volumetric_strain),
        "max_abs_error_deg": float(max(errors)) if errors else math.nan,
        "mean_abs_error_deg": float(np.mean(errors)) if errors else math.nan,
        "biased": bool(errors and max(errors) > 5.0),
    }


# ---------------------------------------------------------------------------
# (b) band-local dilation
# ---------------------------------------------------------------------------
def local_strain_tensor(x_ref: np.ndarray, x_now: np.ndarray, *, k: int = 8,
                        regularisation: float = 1e-9,
                        remove_global_affine: bool = False) -> np.ndarray:
    """Per-particle deformation gradient ``G`` (``u = G X``) from a kNN fit.

    ``remove_global_affine`` mirrors
    :func:`tensordem.biaxial_shear_band.local_shear_strain` (which always removes
    it) and is the right choice for *shear* ridges.  It is **not** the right
    choice for a dilation measurement: removing the best-fit homogeneous field
    subtracts the specimen-average volumetric rate, so the band value would be
    reported relative to the mean instead of absolutely.  Default False here.
    """
    ref = np.asarray(x_ref, dtype=np.float64)
    now = np.asarray(x_now, dtype=np.float64)
    if ref.shape != now.shape or ref.ndim != 2 or ref.shape[1] != 2:
        raise BandMeasurementError("x_ref and x_now must both be (N, 2)")
    count = ref.shape[0]
    if count < k + 1:
        raise BandMeasurementError(f"need at least {k + 1} particles, got {count}")
    disp = _affine_residual(ref, now) if remove_global_affine else (now - ref)
    xt = torch.as_tensor(ref)
    dist = torch.cdist(xt, xt)
    kk = min(max(int(k), 3), count)
    idx = torch.topk(dist, kk, largest=False).indices
    neighbours = ref[idx]
    dx = neighbours - neighbours.mean(axis=1, keepdims=True)
    du = disp[idx] - disp[idx].mean(axis=1, keepdims=True)
    a = np.einsum("nki,nkj->nij", dx, dx)
    b = np.einsum("nki,nkj->nij", du, dx)
    trace = np.trace(a, axis1=1, axis2=2)[:, None, None]
    a = a + regularisation * np.eye(2)[None, :, :] * np.where(trace > 0, trace, 1.0)
    grad = np.linalg.solve(a.transpose(0, 2, 1), b.transpose(0, 2, 1)).transpose(0, 2, 1)
    return grad


def local_volumetric_strain(x_ref: np.ndarray, x_now: np.ndarray, *, k: int = 8,
                            remove_global_affine: bool = False) -> np.ndarray:
    """Per-particle volumetric strain, **compression positive**.

    The biaxial cell accumulates ``volumetric_strain = eps_a + eps_lat`` with
    axial *compression* positive (the platen shortens the specimen), and derives
    ``psi = asin(-d eps_v / d eps_a)`` from it: dilation (= volume increase)
    therefore shows up as a *decrease* of ``eps_v``.  This function returns
    ``-tr(e)`` so that it is on the same convention and
    :func:`band_local_dilation` can use the case's formula unchanged.
    """
    grad = local_strain_tensor(x_ref, x_now, k=k,
                               remove_global_affine=remove_global_affine)
    return -(grad[:, 0, 0] + grad[:, 1, 1])


def band_local_dilation(snapshots: Sequence[dict[str, Any]], *,
                        angle_deg: float, half_width: float,
                        dx_strain_floor: float = 1.0e-6,
                        k: int = 8) -> dict[str, Any]:
    """Dilation angle measured *inside* the band, band frame aware.

    ``snapshots`` is an ordered sequence of ``{"strain": float, "positions":
    (N, 2)}`` (the ``record_snapshots`` payload of
    :func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`).  For every
    consecutive pair the band mask is built on the *reference* configuration
    and the mean ``eps_v`` of the masked particles is differenced against the
    axial strain increment, giving ``d eps_v^band / d eps_a`` by least squares.
    The same is done for the complementary (background) set, so a band-local
    dilation can be compared with the diffuse one.

    Returns ``psi_band`` and ``psi_background`` in degrees
    (``asin(-slope)``, clamped to +-72 deg), the raw slopes, and the mask
    statistics.  Positive ``psi`` = dilation.
    """
    pairs = [s for s in snapshots if s.get("positions") is not None]
    if len(pairs) < 3:
        raise BandMeasurementError("need at least 3 snapshots")
    strain, band_ev, back_ev, all_ev, band_n, back_n = [], [], [], [], [], []
    for ref_snap, now_snap in zip(pairs[:-1], pairs[1:]):
        ref = np.asarray(ref_snap["positions"], dtype=np.float64)
        now = np.asarray(now_snap["positions"], dtype=np.float64)
        mask = band_mask(ref, angle_deg=angle_deg, half_width=half_width)
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        eps_v = local_volumetric_strain(ref, now, k=k)
        strain.append(float(now_snap["strain"]) - float(ref_snap["strain"]))
        band_ev.append(float(np.nanmean(eps_v[mask])))
        back_ev.append(float(np.nanmean(eps_v[~mask])))
        all_ev.append(float(np.nanmean(eps_v)))
        band_n.append(int(mask.sum()))
        back_n.append(int((~mask).sum()))
    if len(strain) < 2:
        raise BandMeasurementError("not enough valid snapshot pairs")
    # regress *accumulated* volumetric strain against *accumulated* axial
    # strain: the snapshot increments are near constant, so regressing the raw
    # per-pair increments would give a zero slope even for a steadily dilating
    # band (fixed in R2 after the synthetic unit test exposed it)
    eps_a = np.asarray(strain, dtype=np.float64)
    if float(eps_a.sum()) <= dx_strain_floor:
        raise BandMeasurementError("no axial strain increment in the snapshot set")
    eps_a = np.cumsum(eps_a)

    band_ev = np.cumsum(np.asarray(band_ev, dtype=np.float64))
    back_ev = np.cumsum(np.asarray(back_ev, dtype=np.float64))
    all_ev = np.cumsum(np.asarray(all_ev, dtype=np.float64))

    def _fit(values: np.ndarray) -> tuple[float, float]:
        matrix = np.column_stack([eps_a, np.ones_like(eps_a)])
        slope, intercept = np.linalg.lstsq(matrix, values, rcond=None)[0]
        residual = values - matrix @ np.array([slope, intercept])
        total = float(np.sum((values - values.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else math.nan
        return float(slope), r2

    band_slope, band_r2 = _fit(np.asarray(band_ev))
    back_slope, back_r2 = _fit(np.asarray(back_ev))
    all_slope, all_r2 = _fit(np.asarray(all_ev))
    clamp = lambda value: math.degrees(math.asin(max(min(-value, 0.95), -0.95)))
    return {
        "angle_deg": float(angle_deg),
        "half_width": float(half_width),
        "pairs": len(strain),
        "delta_axial_strain": float(eps_a[-1]),
        "psi_band_deg": clamp(band_slope),
        "psi_background_deg": clamp(back_slope),
        "psi_specimen_deg": clamp(all_slope),
        "d_eps_v_band_d_eps_a": band_slope,
        "d_eps_v_background_d_eps_a": back_slope,
        "d_eps_v_specimen_d_eps_a": all_slope,
        "band_r2": band_r2,
        "background_r2": back_r2,
        "specimen_r2": all_r2,
        "band_particles_mean": float(np.mean(band_n)),
        "background_particles_mean": float(np.mean(back_n)),
        "convention": ("eps_v compression positive; psi = asin(-d eps_v / d eps_a), "
                       "positive psi = dilation"),
    }


def resolution_control(pos: np.ndarray, *, angles_deg: Sequence[float],
                       slip: float | None = None, band_width: float | None = None,
                       volumetric_strain: float = 0.0, k: int = 8) -> dict[str, Any]:
    """Control: does the estimator work *without* the measurement window?

    The biaxial case excludes a wall margin, so the window shrinks with the
    specimen deformation.  Running the same synthetic sweep on the full lattice
    isolates window truncation from estimator bias.
    """
    return estimator_bias_sweep(pos, angles_deg=angles_deg, slip=slip,
                                band_width=band_width,
                                volumetric_strain=volumetric_strain,
                                measurement_window=None, k=k)