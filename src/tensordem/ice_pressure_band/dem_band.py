"""Long-domain DEM pressure band on a rigid structure face.

Purpose: supply a **native** (simulated, not literature-anchored) estimate of the
along-band pressure correlation length ``lam_x`` and the dimensionless
``lam_x / h`` (``h`` = ice thickness), long enough that the finite-window
correction calibrated in :mod:`.calibration` can be applied honestly.

The experiment is the repository's existing ice--structure crushing DEM
(:class:`tensordem.ice_contact.sim.IceWallCrush`, evidence grade REAL): a slab of
induced-angularity spheres is driven into a rigid plane at ``x = 0``; contact
normal forces are bilinearly deposited on the ``(z, y)`` face grid, giving the
pressure field ``p(z, y, t)``.  Here the slab is made **long along the face
direction** ``y`` (``B = width_over_thickness * h``), which is the record whose
correlation length is asked for.

Mapping of the campaign coordinate: in this 2-D slice the contact band runs along
``y``; ``lam_x`` is *defined* as the correlation length along the contact band,
so ``lam_x = lam_y`` of this geometry.  This is stated explicitly rather than
relabelled silently.

The estimate is the correlation length of the **time-averaged pressure profile**
``pbar(y)`` over a pre-registered quasi-steady window (the second half of the
stroke), consistent with the campaign practice of measuring the *spatial*
correlation of the (mean) pressure field.  Two cross-checks are reported:

* per-snapshot profile correlation lengths, averaged over time (a different
  definition: the instantaneous patchiness, not the mean field);
* a *shuffle control*: the ``pbar(y)`` profile randomly permuted in ``y``, whose
  correlation length must collapse to the grid scale.  If it does not, the
  estimator is measuring its own smoothing, not the field.

Uncertainty: moving-block bootstrap over the recorded snapshots (block length
``max(2, T//10)``), which preserves the temporal correlation of the pressure
field; the bias correction is applied inside the bootstrap so the reported
interval contains the estimator bias, not just the sampling scatter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .estimators import METHODS, diagnose_regime, estimate_all
from .calibration import BiasModel, correct_length

__all__ = [
    "BandConfig",
    "block_coarsen_profile",
    "resolution_ladder",
    "BandRun",
    "run_band",
    "snapshot_profiles",
    "time_averaged_profile",
    "estimate_profile_lambda",
    "bootstrap_lambda",
    "shuffle_control",
]


@dataclass
class BandConfig:
    """Geometry and numerics of one long-domain pressure-band run."""

    thickness: float = 0.20               # h [m]
    width_over_thickness: float = 60.0    # record length B / h [-]
    d50_ratio: float = 0.25               # fragment diameter / h [-]
    depth: float = 0.30                   # Lx [m] slab depth into the ice
    n_max: int = 7000                     # fragment budget of the assembly
    drive_speed: float = 0.5              # v [m/s]
    dt: float = 5.0e-5                    # [s]
    stroke_over_thickness: float = 0.5    # stroke / h [-]
    save_every: int = 10                  # snapshot interval [steps]
    cells_per_thickness: int = 16         # dy = h / cells_per_thickness
    z_cells: int = 8                      # nz over the thickness
    solid_fraction: float = 0.50
    seed: int = 0
    steady_window: tuple[float, float] = (0.5, 1.0)
    z_band: tuple[float, float] = (0.1, 0.9)
    preload: float = 0.02
    settle_steps: int = 200
    mass_scale: float = 1.0

    @property
    def width(self) -> float:
        return self.width_over_thickness * self.thickness

    @property
    def dy(self) -> float:
        return self.thickness / self.cells_per_thickness

    @property
    def dz(self) -> float:
        return self.thickness / self.z_cells

    @property
    def ny(self) -> int:
        return int(round(self.width / self.dy))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({"width": self.width, "dy": self.dy, "dz": self.dz, "ny": self.ny})
        return d

    def validate(self) -> None:
        if self.thickness <= 0 or self.width <= 0:
            raise ValueError("thickness and width must be positive")
        if not 0.05 <= self.d50_ratio <= 0.5:
            raise ValueError("d50_ratio must be in [0.05, 0.5] to resolve the band")
        if not 0.0 < self.solid_fraction < 0.85:
            raise ValueError("solid_fraction must be in (0, 0.85)")
        if self.cells_per_thickness < 8:
            raise ValueError("cells_per_thickness < 8 cannot resolve a 2h pressure patch")
        if not 0.0 <= self.steady_window[0] < self.steady_window[1] <= 1.0:
            raise ValueError("steady_window must be an increasing sub-interval of [0, 1]")


@dataclass
class BandRun:
    pressure: np.ndarray                  # (T, nz, ny) [Pa]
    time: np.ndarray                      # (T,) [s]
    config: BandConfig
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "summary": self.summary,
                "n_snapshots": int(self.pressure.shape[0]),
                "pressure_max": float(np.max(self.pressure)),
                "pressure_mean": float(np.mean(self.pressure))}


def run_band(cfg: BandConfig) -> BandRun:
    """Run the ice--structure crushing DEM and return the pressure history."""
    from ..ice_contact.shapes import sphere_shape
    from ..ice_contact.sim import CrushParams, IceWallCrush
    from ..ice_contact.study import build_assembly

    cfg.validate()
    # Jittered-lattice assembly of the repository's ice--structure study (random
    # sequential packing saturates near 0.3 solid fraction and cannot load a
    # wall); volumes are handed to the sphere proxy one-to-one.
    volumes, positions, d50 = build_assembly(
        cfg.seed, (cfg.depth, cfg.width, cfg.thickness),
        d50_ratio=cfg.d50_ratio, solid_fraction=cfg.solid_fraction, n_max=cfg.n_max,
    )
    n = len(volumes)
    shapes = [sphere_shape(v) for v in volumes]
    params = CrushParams(mass_scale=cfg.mass_scale)
    sim = IceWallCrush(
        shapes,
        box=(cfg.depth, cfg.width, cfg.thickness),
        drive_speed=cfg.drive_speed,
        params=params,
        seed=cfg.seed,
        grid=(cfg.dy, cfg.dz),
        positions=positions,
        stroke=cfg.stroke_over_thickness * cfg.thickness,
        dt=cfg.dt,
        save_every=cfg.save_every,
        settle_steps=cfg.settle_steps,
        preload=cfg.preload,
    )
    sim.run()
    summary = sim.summary()
    summary.update({
        "n_particles": n,
        "d50_m": float(d50),
        "solid_fraction_target": cfg.solid_fraction,
        "max_pressure_pa": float(np.max(np.asarray(sim.pressure_array))),
        "wall_contacted": bool(np.any(np.asarray(sim.pressure_array) > 0.0)),
    })
    return BandRun(pressure=np.asarray(sim.pressure_array), time=np.asarray(sim.time_array),
                   config=cfg, summary=summary)


def _band_slice(nz: int, z_band: tuple[float, float]) -> slice:
    lo = int(np.floor(z_band[0] * nz))
    hi = int(np.ceil(z_band[1] * nz))
    lo = max(0, min(lo, nz - 1))
    hi = max(lo + 1, min(hi, nz))
    return slice(lo, hi)


def snapshot_profiles(
    pressure: np.ndarray, cfg: BandConfig
) -> np.ndarray:
    """Per-snapshot, z-band-averaged pressure profiles ``p_t(y)`` [Pa]."""
    zs = _band_slice(pressure.shape[1], cfg.z_band)
    t0 = int(round(cfg.steady_window[0] * pressure.shape[0]))
    t1 = int(round(cfg.steady_window[1] * pressure.shape[0]))
    t1 = max(t1, t0 + 2)
    return pressure[t0:t1, zs, :].mean(axis=1)


def time_averaged_profile(pressure: np.ndarray, cfg: BandConfig) -> np.ndarray:
    """Time-averaged pressure profile ``pbar(y)`` over the steady window."""
    return snapshot_profiles(pressure, cfg).mean(axis=0)


def estimate_profile_lambda(
    profile: np.ndarray,
    dy: float,
    *,
    models: dict | None = None,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    """Correlation length of a pressure profile in metres (raw + corrected)."""
    p = np.asarray(profile, dtype=float)
    if p.size < 16:
        raise ValueError("profile too short for a correlation-length estimate")
    L = p.size * dy
    regime = diagnose_regime(p, dy)
    raw = estimate_all(p, dy, methods=methods)
    out = {"record_length_m": L, "dx_m": dy, "n_samples": int(p.size),
           "regime": regime.to_dict(), "raw": raw, "corrected": {},
           "correction_factor": {}}
    for m in methods:
        model = None if models is None else models.get(m)
        if isinstance(model, dict):
            if "params" in model:
                model = BiasModel(method=m, form=model.get("form", "inv_poly"),
                                  params=tuple(model["params"]))
            elif "a" in model and "b" in model:
                model = BiasModel(method=m, form="inv_poly",
                                  params=(float(model["a"]), float(model["b"])))
            else:
                model = None
        if model is None:
            out["corrected"][m] = float("nan")
            out["correction_factor"][m] = float("nan")
            continue
        val = correct_length(raw[m], L, model)
        out["corrected"][m] = float(val)
        out["correction_factor"][m] = float(val / raw[m]) if raw[m] else float("nan")
    vals = [v for v in out["corrected"].values() if np.isfinite(v) and v > 0.0]
    if vals:
        out["value_m"] = float(np.median(vals))
        out["method_spread"] = float((max(vals) - min(vals)) / np.median(vals))
    else:
        out["value_m"] = float("nan")
        out["method_spread"] = float("nan")
    out["L_over_lambda"] = float(L / out["value_m"]) if out["value_m"] > 0 else float("inf")
    return out


def bootstrap_lambda(
    snapshots: np.ndarray,
    dy: float,
    *,
    models: dict | None = None,
    method: str = "integral",
    n_boot: int = 400,
    block: int | None = None,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Moving-block bootstrap over snapshots of the corrected correlation length."""
    snaps = np.asarray(snapshots, dtype=float)
    T = snaps.shape[0]
    if T < 4:
        raise ValueError("need at least 4 snapshots for a block bootstrap")
    if block is None:
        block = max(2, T // 10)
    block = int(min(block, T))
    n_blocks = int(np.ceil(T / block))
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        prof = snaps[idx].mean(axis=0)
        est = estimate_profile_lambda(prof, dy, models=models, methods=(method,))
        vals[b] = est["corrected"][method]
    ok = np.isfinite(vals)
    alpha = 0.5 * (1.0 - ci)
    lo = float(np.nanpercentile(vals[ok], 100 * alpha)) if ok.sum() > 8 else float("nan")
    hi = float(np.nanpercentile(vals[ok], 100 * (1 - alpha))) if ok.sum() > 8 else float("nan")
    return {
        "method": method,
        "n_boot": n_boot,
        "block_snapshots": block,
        "mean_m": float(np.nanmean(vals[ok])) if ok.sum() else float("nan"),
        "sd_m": float(np.nanstd(vals[ok], ddof=1)) if ok.sum() > 2 else float("nan"),
        "ci_low_m": lo,
        "ci_high_m": hi,
        "ci": ci,
        "failure_rate": float(1.0 - ok.mean()),
        "samples": vals[ok].tolist()[:2000],
    }


def block_coarsen_profile(profile: np.ndarray, factor: int) -> np.ndarray:
    """Box-average a profile onto a coarser grid (``factor`` fine cells per block)."""
    p = np.asarray(profile, dtype=float)
    factor = int(max(factor, 1))
    n = (p.size // factor) * factor
    if n < 16:
        raise ValueError("profile too short for this coarsening factor")
    return p[:n].reshape(-1, factor).mean(axis=1)


def resolution_ladder(
    profile: np.ndarray,
    dy: float,
    *,
    factors: tuple[int, ...] = (1, 2, 4, 8),
    models: dict | None = None,
    methods: tuple[str, ...] = ("integral", "efold"),
) -> list[dict]:
    """Correlation length versus the coarse-graining scale of the pressure band.

    A discrete-particle wall pressure is not a continuum field below the contact
    patch size: if the measured length tracks the block size, the field is white
    at that resolution and the number is the deposit kernel, not a physical
    patchiness scale.  The ladder makes that distinction explicit instead of
    quoting the finest-grid number as ``lam_x``.
    """
    out = []
    for f in factors:
        try:
            coarse = block_coarsen_profile(profile, f)
            est = estimate_profile_lambda(coarse, dy * f, models=models, methods=methods)
        except ValueError as exc:
            out.append({"block_factor": int(f), "dx_m": dy * f, "error": str(exc)})
            continue
        vals = {m: est["corrected"][m] for m in methods}
        vals = {k: v for k, v in vals.items() if np.isfinite(v) and v > 0}
        lam = float(np.median(list(vals.values()))) if vals else float("nan")
        out.append({
            "block_factor": int(f),
            "dx_m": float(dy * f),
            "n_samples": int(coarse.size),
            "lambda_m": lam,
            "lambda_over_grid": float(lam / (dy * f)) if np.isfinite(lam) else float("nan"),
            "regime": est["regime"]["label"],
            "L_over_lambda": est["L_over_lambda"],
        })
    return out


def shuffle_control(
    profile: np.ndarray, dy: float, *, models: dict | None = None,
    n_perm: int = 50, seed: int = 0,
) -> dict:
    """Random permutation of the profile: the length must collapse to the grid."""
    p = np.asarray(profile, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        q = rng.permutation(p)
        est = estimate_profile_lambda(q, dy, models=models, methods=("integral",))
        vals.append(est["corrected"]["integral"])
    ok = np.asarray([v for v in vals if np.isfinite(v)])
    L = p.size * dy
    return {
        "n_perm": n_perm,
        "median_m": float(np.median(ok)) if ok.size else float("nan"),
        "p95_m": float(np.percentile(ok, 95)) if ok.size else float("nan"),
        "grid_dx_m": dy,
        "record_length_m": L,
        "ratio_to_grid": float(np.median(ok) / dy) if ok.size else float("nan"),
    }