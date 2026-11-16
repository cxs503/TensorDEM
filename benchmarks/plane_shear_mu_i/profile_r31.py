"""CASE-DEM-PLANE-SHEAR-MUIDI round 31: bin-resolved flow profile + stress split.

Why this module exists
----------------------
Round 30 (``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R30.md``) closed the dimensional
mismatch (2-D disks must be judged against the 2-D law of da Cruz et al. 2005,
Eqs. 13-14) and rewrote the failure signature: ``nu_max`` now agrees to +1.7 %
while the *rate sensitivity* is 3-4x too small (b = 0.279 +- 0.031 vs 1.0) and
the intercept is 29 % low.  It left exactly two open items, both of which need
a measurement this device did not have:

1. **is the abscissa right?**  The nominal shear rate is ``2 V / h_eff``; the
   inertial number is built on it.  R30 noticed that the kinetic energy implies
   a granular velocity of order 0.1 m/s while ``I = 0.19`` needs
   ``gammadot d ~ 0.39 m/s``, and that the two cannot both be true.  Resolving
   it requires the *measured* mean velocity profile ``u_x(y)``: wall slip or
   shear localisation makes the realised rate smaller than the nominal one, and
   then every ``I`` on the abscissa is wrong.
2. **how large is the rate-dependent (flow) stress really?**  R30 could only
   bound it with ``p_kin <= KE/A`` (<= 0.132 % of ``p_bulk``), i.e. an upper
   bound that *falsified* the "missing flow stress explains b" hypothesis but
   did not measure the term.

This module supplies both, as a pure accumulator that is fed once per sample by
``plane_shear_2d.simulate``.  Nothing in the contact physics changes, so the
archived rounds-24..30 numbers stay comparable.

Definitions (all 2-D, thickness = diameter, ``phi`` from disk areas)
--------------------------------------------------------------------
* binning: free-disk centres into ``n_bins`` equal slabs of
  ``[y_bottom + r, y_top - r]`` (``r`` = nominal radius), contacts into slabs by
  their **midpoint** y (with the branch vector taken through the periodic image,
  exactly as ``measure()`` does for the total virial);
* per-bin contact stress ``sigma_ab = sum(r_a f_b) / (W h_bin t)``;
* per-bin kinetic stress ``p_kin = rho phi (var_x + var_y)/2``,
  ``tau_kin = -rho phi cov_xy`` (the fluctuation is taken about the bin mean
  velocity, pooled over every accumulated sample);
* ``gammadot_measured`` = the least-squares slope of ``u_x`` over the **central**
  band (default 60 % of the slab range), which is insensitive to the wall layers,
  with ``r2`` as the linearity diagnostic: a shear band shows up as ``r2 -> 0``
  and as a slope much larger than ``2 V / h``;
* wall slip: ``u_x`` of the wall-adjacent bin minus the imposed wall velocity;
  the slip length is that offset divided by the measured slope.

The reduction functions (:func:`couette_fit`, :func:`bin_index`,
:func:`kinetic_stress_terms`, :func:`i_from_rate`) are pure and are covered by
``test_r31_profile.py`` with synthetic profiles of known slope/slip/localisation.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "PROFILE_BINS",
    "CENTER_FRAC",
    "bin_range",
    "bin_index",
    "couette_fit",
    "kinetic_stress_terms",
    "i_from_rate",
    "ProfileAccumulator",
]

PROFILE_BINS = 10
CENTER_FRAC = 0.6
#: a Couette profile is called linear/unlocalised above this r2
R2_LINEAR = 0.99


# --------------------------------------------------------------------------- #
# pure reductions
# --------------------------------------------------------------------------- #
def bin_range(y_bottom: float, y_top: float, radius: float) -> tuple[float, float]:
    """Centre-to-centre range spanned by the free disks (wall-grain rocks excluded)."""
    y0 = float(y_bottom) + float(radius)
    y1 = float(y_top) - float(radius)
    if not y1 > y0:
        raise ValueError(f"empty bin range: y0={y0} y1={y1}")
    return y0, y1


def bin_index(y, y0: float, y1: float, n_bins: int = PROFILE_BINS) -> np.ndarray:
    """Slab index of ``y``, clipped into ``[0, n_bins-1]``.

    Clipping (not dropping) is deliberate: a disk whose centre sits a fraction of
    a radius outside the nominal range is still sheared material, and silently
    dropping it would bias ``phi`` and the velocity profile downwards near the
    walls - exactly where the R30 question is decided.
    """
    x = (np.asarray(y, dtype=np.float64) - float(y0)) / (float(y1) - float(y0))
    return np.clip(np.floor(x * n_bins).astype(np.int64), 0, n_bins - 1)


def couette_fit(y, ux, *, center_frac: float = CENTER_FRAC,
                weights=None) -> dict:
    """Least-squares ``u_x = a + b y`` on the central band, with ``r2``.

    Parameters
    ----------
    y, ux : array_like
        Bin-centre heights and bin-mean streamwise velocities.  Non-finite
        entries (empty bins) are ignored.
    center_frac : float
        Width of the fitted band as a fraction of the occupied range.  1.0 uses
        every non-empty bin (then wall layers enter the fit and a slip layer
        biases the slope low); the default 0.6 is the stiff choice.
    weights : array_like, optional
        Optional weights (e.g. bin occupancy) for a weighted fit.

    Returns
    -------
    dict with ``slope``, ``intercept``, ``r2``, ``n_used``, ``y_lo``, ``y_hi``.
    A fit is only attempted with >= 3 usable bins; otherwise every number is
    ``nan`` and ``n_used`` says how many bins there were.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    ux = np.asarray(ux, dtype=np.float64).ravel()
    w = np.ones_like(y) if weights is None else np.asarray(weights, float).ravel()
    nan = float("nan")
    good = np.isfinite(y) & np.isfinite(ux) & np.isfinite(w) & (w > 0)
    if good.sum() < 3:
        return {"slope": nan, "intercept": nan, "r2": nan,
                "n_used": int(good.sum()), "y_lo": nan, "y_hi": nan}
    ylo, yhi = float(y[good].min()), float(y[good].max())
    ymid = 0.5 * (ylo + yhi)
    half = 0.5 * center_frac * (yhi - ylo)
    m = good & (np.abs(y - ymid) <= half)
    if m.sum() < 3:
        m = good
    yy, uu, ww = y[m], ux[m], w[m]
    sw = ww.sum()
    my = float((ww * yy).sum() / sw)
    mu = float((ww * uu).sum() / sw)
    syy = float((ww * (yy - my) ** 2).sum())
    if syy <= 0.0:
        return {"slope": nan, "intercept": nan, "r2": nan,
                "n_used": int(m.sum()), "y_lo": ylo, "y_hi": yhi}
    slope = float((ww * (yy - my) * (uu - mu)).sum() / syy)
    intercept = mu - slope * my
    resid = uu - (intercept + slope * yy)
    ss_res = float((ww * resid ** 2).sum())
    ss_tot = float((ww * (uu - mu) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else nan
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "n_used": int(m.sum()), "y_lo": ylo, "y_hi": yhi}


def kinetic_stress_terms(phi, var_x, var_y, cov_xy, density: float) -> dict:
    """Kinetic (streaming) part of the stress for a 2-D bin.

    ``p_kin = rho phi (var_x + var_y)/2`` and ``tau_kin = -rho phi cov_xy``
    (traction convention ``sigma_ab = rho phi <v'_a v'_b>``, so a positive
    ``cov_xy`` for ``du_x/dy < 0`` gives a positive shear traction).
    """
    rho = float(density)
    phi = np.asarray(phi, dtype=np.float64)
    p_kin = rho * phi * 0.5 * (np.asarray(var_x, float) + np.asarray(var_y, float))
    tau_kin = -rho * phi * np.asarray(cov_xy, float)
    return {"p_kin": p_kin, "tau_kin": tau_kin}


def i_from_rate(gammadot, diameter: float, pressure: float,
                density: float) -> np.ndarray | float:
    """``I = gammadot d / sqrt(p/rho)`` (the device's own definition, reused)."""
    return np.asarray(gammadot, float) * float(diameter) / math.sqrt(
        float(pressure) / float(density))


# --------------------------------------------------------------------------- #
# the accumulator
# --------------------------------------------------------------------------- #
class ProfileAccumulator:
    """Pooled per-bin velocity / stress statistics over the measurement window.

    One :meth:`add` call per sampled instant.  Pooling raw sums (not per-sample
    means) keeps the fluctuation variance honest: subtracting the pooled mean
    leaves both the in-bin shear gradient spread and the genuine granular
    temperature, which is exactly what ``p_kin`` should contain.
    """

    VIRIAL_KEYS = ("xx", "xy", "yx", "yy")

    def __init__(self, n_bins: int = PROFILE_BINS):
        self.n_bins = int(n_bins)
        nb = self.n_bins
        self.n = np.zeros(nb)
        self.s_vx = np.zeros(nb)
        self.s_vy = np.zeros(nb)
        self.s_vx2 = np.zeros(nb)
        self.s_vy2 = np.zeros(nb)
        self.s_vxvy = np.zeros(nb)
        self.s_area = np.zeros(nb)
        self.c_count = np.zeros(nb)
        self.virial = np.zeros((nb, 2, 2))
        self.samples = 0
        self.n_free = 0

    # -- feeding -----------------------------------------------------------
    def add(self, cell) -> None:
        """Bin the current state of a ``RoughWallShearCell``."""
        case = cell.case
        sysm = cell.system
        free = cell.free_idx
        y0, y1 = bin_range(cell.y_bottom, cell.y_top, case.radius)
        nb = self.n_bins
        pos = sysm.pos
        vel = sysm.vel
        k = bin_index(pos[free, 1], y0, y1, nb)
        vx = np.asarray(vel[free, 0], dtype=np.float64)
        vy = np.asarray(vel[free, 1], dtype=np.float64)
        np.add.at(self.n, k, 1.0)
        np.add.at(self.s_vx, k, vx)
        np.add.at(self.s_vy, k, vy)
        np.add.at(self.s_vx2, k, vx * vx)
        np.add.at(self.s_vy2, k, vy * vy)
        np.add.at(self.s_vxvy, k, vx * vy)
        np.add.at(self.s_area, k, math.pi * np.asarray(sysm.radii[free]) ** 2)
        self.n_free = int(len(free))

        contacts = sysm.contacts
        if contacts is not None and len(contacts.get("i", [])):
            i_idx = np.asarray(contacts["i"], dtype=np.int64)
            j_idx = np.asarray(contacts["j"], dtype=np.int64)
            f = np.asarray(contacts["f_on_i"], dtype=np.float64)
            branch = pos[j_idx] - pos[i_idx]
            branch[:, 0] -= case.width * np.round(branch[:, 0] / case.width)
            ymid = 0.5 * (pos[i_idx, 1] + pos[j_idx, 1])
            kb = bin_index(ymid, y0, y1, nb)
            np.add.at(self.c_count, kb, 1.0)
            # traction convention sigma_ab = sum r_a f_b (force on i from j)
            for a in range(2):
                for b in range(2):
                    np.add.at(self.virial[:, a, b], kb, branch[:, a] * f[:, b])
        self.samples += 1

    # -- reducing ----------------------------------------------------------
    def finalize(self, *, width: float, thickness: float, y_bottom: float,
                 y_top: float, radius: float, density: float, v_wall: float,
                 center_frac: float = CENTER_FRAC,
                 n_bins: int | None = None) -> dict:
        """Pooled profile, per-bin stress split and the derived scalars."""
        nb = self.n_bins if n_bins is None else int(n_bins)
        if nb != self.n_bins:
            raise ValueError("n_bins mismatch with the accumulated data")
        if self.samples == 0:
            raise ValueError("no samples accumulated")
        y0, y1 = bin_range(y_bottom, y_top, radius)
        h_bin = (y1 - y0) / nb
        # stresses carry the out-of-plane thickness (= diameter, the engine's 2D
        # convention, identical to ``measure()``); the *area* fraction must not:
        # ``phi_gap`` in ``measure()`` is area over W x gap with no thickness.
        vol_bin = float(width) * h_bin * float(thickness)
        area_bin = float(width) * h_bin
        y_c = y0 + (np.arange(nb) + 0.5) * h_bin

        with np.errstate(invalid="ignore", divide="ignore"):
            cnt = self.n / self.samples
            ux = self.s_vx / self.n
            uy = self.s_vy / self.n
            var_x = self.s_vx2 / self.n - ux ** 2
            var_y = self.s_vy2 / self.n - uy ** 2
            cov = self.s_vxvy / self.n - ux * uy
            phi = self.s_area / (area_bin * self.samples)
            sig = self.virial / (vol_bin * self.samples)
        var_x = np.clip(var_x, 0.0, None)
        var_y = np.clip(var_y, 0.0, None)
        # empty bins -> nan so that the fit ignores them instead of reading 0
        empty = self.n <= 0
        ux = np.where(empty, np.nan, ux)
        uy = np.where(empty, np.nan, uy)
        cov = np.where(empty, np.nan, cov)
        phi = np.where(empty, np.nan, phi)
        sig = np.where(empty[:, None, None], np.nan, sig)

        kin = kinetic_stress_terms(np.where(empty, 0.0, phi),
                                   np.where(empty, 0.0, var_x),
                                   np.where(empty, 0.0, var_y),
                                   np.where(empty, 0.0, cov), density)
        p_kin_bin = kin["p_kin"]
        tau_kin_bin = kin["tau_kin"]
        p_ctc_bin = -0.5 * (sig[:, 0, 0] + sig[:, 1, 1])
        tau_ctc_bin = 0.5 * (np.abs(sig[:, 0, 1]) + np.abs(sig[:, 1, 0]))

        fit = couette_fit(y_c, ux, center_frac=center_frac, weights=self.n)
        slope = fit["slope"]
        # du_x/dy = -2 v_wall/h for a perfect Couette cell -> rate = |slope|
        gamma_dot_meas = abs(slope) if math.isfinite(slope) else float("nan")

        # wall-adjacent layers: the first/last non-empty bin
        filled = np.flatnonzero(self.n > 0)
        ux_bot = float(ux[filled[0]]) if len(filled) else float("nan")
        ux_top = float(ux[filled[-1]]) if len(filled) else float("nan")
        slip_bot = ux_bot - float(v_wall)      # >0: material outruns the wall
        slip_top = ux_top + float(v_wall)      # <0: material outruns the wall
        slip_len_bot = (slip_bot / slope) if (math.isfinite(slope) and slope != 0) else float("nan")
        slip_len_top = (slip_top / slope) if (math.isfinite(slope) and slope != 0) else float("nan")

        tot_vol = vol_bin * nb
        p_ctc = float(np.nansum(np.where(empty, 0.0, p_ctc_bin) * vol_bin) / tot_vol)
        tau_ctc = float(np.nansum(np.where(empty, 0.0, tau_ctc_bin) * vol_bin) / tot_vol)
        p_kin = float(np.nansum(p_kin_bin * vol_bin) / tot_vol)
        tau_kin = float(np.nansum(tau_kin_bin * vol_bin) / tot_vol)
        p_tot = p_ctc + p_kin
        tau_tot = tau_ctc + tau_kin

        def _ratio(a, b):
            return float(a / b) if b and math.isfinite(b) and b > 0 else float("nan")

        # local stress ratio per bin (localisation detector); empty bins and
        # contact-free bins are nan by construction, not zero
        with np.errstate(invalid="ignore", divide="ignore"):
            mu_bin = np.where(empty, np.nan,
                              0.5 * (np.abs(sig[:, 0, 1]) + np.abs(sig[:, 1, 0]))
                              / p_ctc_bin)
        return {
            "n_bins": nb,
            "samples": int(self.samples),
            "n_free": int(self.n_free),
            "bin_height": h_bin,
            "y_centre": y_c.tolist(),
            "count_mean": cnt.tolist(),
            "phi": np.where(empty, np.nan, phi).tolist(),
            "ux": ux.tolist(),
            "uy": uy.tolist(),
            "var_x": np.where(empty, np.nan, var_x).tolist(),
            "var_y": np.where(empty, np.nan, var_y).tolist(),
            "cov_xy": cov.tolist(),
            "p_contact": np.where(empty, np.nan, p_ctc_bin).tolist(),
            "tau_contact": np.where(empty, np.nan, tau_ctc_bin).tolist(),
            "p_kinetic": np.where(empty, np.nan, p_kin_bin).tolist(),
            "tau_kinetic": np.where(empty, np.nan, tau_kin_bin).tolist(),
            "mu_local": mu_bin.tolist(),
            "n_contacts_mean": (self.c_count / self.samples).tolist(),
            "fit": fit,
            "gamma_dot_measured": gamma_dot_meas,
            "ux_wall_bottom": ux_bot,
            "ux_wall_top": ux_top,
            "v_wall": float(v_wall),
            "slip_bottom": slip_bot,
            "slip_top": slip_top,
            "slip_length_bottom": slip_len_bot,
            "slip_length_top": slip_len_top,
            "p_contact_total": p_ctc,
            "tau_contact_total": tau_ctc,
            "p_kinetic_total": p_kin,
            "tau_kinetic_total": tau_kin,
            "p_total": p_tot,
            "tau_total": tau_tot,
            "mu_contact": _ratio(tau_ctc, p_ctc),
            "mu_kinetic": _ratio(tau_kin, p_kin),
            "mu_total": _ratio(tau_tot, p_tot),
            "kinetic_share_p": _ratio(p_kin, p_tot),
            "kinetic_share_tau": _ratio(tau_kin, tau_tot),
            "locality": {
                "r2_linear": fit["r2"],
                "linear": bool(math.isfinite(fit["r2"]) and fit["r2"] >= R2_LINEAR),
            },
        }


def compare_to_nominal(profile: dict, *, gap_eff: float, diameter: float,
                       pressure: float, density: float) -> dict:
    """Nominal-vs-measured abscissa, and the re-judged ``I``.

    ``I_nominal`` is the device's own ``2 V / h_eff d / sqrt(p/rho)``; ``I_measured``
    uses the slope of the measured profile.  Both are reported side by side so a
    reader can see whether the FAIL is an abscissa error or a physics error.
    """
    g_nom = 2.0 * float(profile["v_wall"]) / float(gap_eff)
    g_meas = float(profile["gamma_dot_measured"])
    ratio = (g_meas / g_nom) if (math.isfinite(g_meas) and g_nom > 0) else float("nan")
    return {
        "gamma_dot_nominal": g_nom,
        "gamma_dot_measured": g_meas,
        "gamma_dot_ratio": float(ratio),
        "I_nominal": float(i_from_rate(g_nom, diameter, pressure, density)),
        "I_measured": float(i_from_rate(g_meas, diameter,
                                        profile["p_contact_total"] or pressure,
                                        density)),
        "slip_length_bottom_over_h": (float(profile["slip_length_bottom"]) / float(gap_eff)
                                      if math.isfinite(profile["slip_length_bottom"]) else float("nan")),
        "slip_length_top_over_h": (float(profile["slip_length_top"]) / float(gap_eff)
                                   if math.isfinite(profile["slip_length_top"]) else float("nan")),
    }