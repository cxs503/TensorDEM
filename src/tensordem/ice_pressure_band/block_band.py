"""Block-scale (floe / ice-block) along-hull pressure band: a 2-D rigid-disc DEM.

Why a *second* DEM
------------------
:mod:`.dem_band` drives a slab of *grains* (``d50 = 0.125-0.4 h``) into a rigid
wall and measures the along-wall pressure correlation length.  That experiment
returns ``lam_x/h ~ 0.06`` -- two orders below the literature band -- and the
maturation node concluded (honestly) that the number it measured was the
*pressure deposition kernel*, not a continuum patch scale.  The physical reason
is structural: an unbroken monodisperse granular layer transmits load along
force chains spaced by one grain, so no ``O(1-3 h)`` structure can exist in it.

Real ice--structure contact does not look like that.  The ice arrives as
**discrete blocks / floes** whose in-plane dimension is of the same order as the
thickness (``D/h = 1-4`` is the classical range for broken-ice blocks and for
the spall size of a crushed layer).  The along-hull pressure correlation length
of such a field is therefore *expected* to be a block-scale quantity.  This
module tests that statement instead of assuming it: it simulates a bed of rigid
blocks of controlled size ``D`` confined between a rigid hull face and a moving
ice platen, and measures the along-face pressure correlation length with the
calibrated estimator of :mod:`.estimators` / :mod:`.calibration`.

Measurement chain (deliberately explicit, because the chain *is* the result)
---------------------------------------------------------------------------
1. Hull-wall contact normal forces are deposited bilinearly on a fine along-face
   grid with spacing ``dy = D / cells_per_block`` -- a pressure *sensor* of
   footprint ``dy``.
2. The field is then coarse-grained by a box window of width
   ``gauge = gauge_over_block * D`` (pre-registered), which is what a finite
   pressure gauge / spatial average does to a discrete contact field.
3. ``lam_x`` is the correlation length of that coarse-grained profile, using the
   same estimators and the same finite-window bias correction as the synthetic
   calibration.

Step 2 is not cosmetic: a contact field measured with a footprint much smaller
than the structural scale returns the *footprint*, which is exactly the failure
mode of the grain-scale experiment.  :func:`gauge_ladder` sweeps the gauge and
reports the saturation, so the report can distinguish "structure" from "gauge".

Verification (machine-checkable, see :class:`BlockBandRun.summary`)
-------------------------------------------------------------------
* **Global force balance**: the sum of hull-wall normal forces must equal the
  platen normal force (third law for the whole bed) to within a few percent in
  the sampled window.  A broken contact model or a wrong minimum image shows up
  here immediately.
* **Quasi-static ratio**: the mean block velocity must be a small fraction of the
  drive speed, so the measured field is not an inertia artefact.
* **Shuffle control**: permuting ``p(y)`` must collapse ``lam_x`` to the gauge
  scale.

Simplifications (stated, not hidden): discs in the plane (non-circular blocks
are approximated by discs plus an explicit rolling resistance), no breakage, two
constant-velocity rigid walls, dry contact with a viscous Coulomb tangential law
(no tangential elastic history), no buoyancy.  The reported quantity is the
correlation length of the time-averaged pressure profile, which is the
stationary-field counterpart of the instantaneous patchiness (the per-snapshot
median is reported alongside).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .calibration import BiasModel, correct_length
from .estimators import METHODS, diagnose_regime, estimate_all

__all__ = [
    "BlockBandConfig",
    "BlockBandRun",
    "PRIMARY_METHODS",
    "coarse_grain",
    "deposit_profile",
    "block_load_acf",
    "estimate_profile_lambda",
    "forward_model_lambda",
    "gauge_ladder",
    "reference_stress",
    "shuffle_profile",
    "shot_noise_lambda",
    "shot_noise_profile",
    "simulate_block_band",
    "snapshot_profiles",
    "time_mean_profile",
]

#: the three methods that are actually calibrated for finite windows
PRIMARY_METHODS = ("integral", "expfit", "crossing")


@dataclass
class BlockBandConfig:
    """Geometry and numerics of one block-scale along-face band run."""

    thickness: float = 0.20                 # h [m], the normalising length
    band_over_thickness: float = 80.0       # along-face record length B / h
    block_over_thickness: float = 2.0       # block dimension D / h
    layer_rows: float = 2.0                 # bed depth Lx = layer_rows * D
    polydispersity: float = 0.08            # uniform half-width around D, fraction
    target_traction: float = 2.0e6          # servo target wall stress [Pa]
    drive_speed: float = 0.20               # platen speed cap [m/s]
    settle_steps: int = 4000                # load build-up (not sampled)
    sample_steps: int = 8000                # sampled steady window
    n_snapshots: int = 24                   # sampled configurations
    cells_per_block: int = 20               # deposition grid: dy = D / 20
    gauge_over_block: float = 0.5           # sensor footprint / D
    contact_time: float = 2.0e-3            # contact period [s] -> stiffness, dt
    steps_per_contact_time: int = 240       # dt = contact_time / this
    damping: float = 0.20                   # normal damping ratio
    mu: float = 0.50                        # wall / block friction
    mu_rolling: float = 0.20                # rolling resistance / (R Fn)
    density: float = 917.0                  # ice [kg/m^3]
    seed: int = 0

    # ---- derived quantities -------------------------------------------------
    @property
    def block(self) -> float:
        return self.block_over_thickness * self.thickness

    @property
    def band(self) -> float:
        return self.band_over_thickness * self.thickness

    @property
    def depth(self) -> float:
        return self.layer_rows * self.block

    @property
    def dy(self) -> float:
        return self.block / self.cells_per_block

    @property
    def ny(self) -> int:
        return int(round(self.band / self.dy))

    @property
    def gauge(self) -> float:
        return self.gauge_over_block * self.block

    @property
    def dt(self) -> float:
        return self.contact_time / self.steps_per_contact_time

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({"block": self.block, "band": self.band, "depth": self.depth,
                  "dy": self.dy, "ny": self.ny, "gauge": self.gauge, "dt": self.dt})
        return d

    def validate(self) -> None:
        if self.thickness <= 0 or self.band <= 0 or self.block <= 0:
            raise ValueError("thickness, band and block must be positive")
        if not 0.5 <= self.block_over_thickness <= 6.0:
            raise ValueError("block_over_thickness must be in [0.5, 6]")
        if not 0.5 <= self.layer_rows <= 4.0:
            raise ValueError("layer_rows must be in [0.5, 4]")
        if not 0.0 <= self.polydispersity < 0.4:
            raise ValueError("polydispersity must be in [0, 0.4)")
        if self.cells_per_block < 8:
            raise ValueError("cells_per_block < 8 cannot resolve a block contact")
        if self.gauge_over_block <= 0.0:
            raise ValueError("gauge must be positive")
        if self.target_traction <= 0.0:
            raise ValueError("target_traction must be positive")
        if self.settle_steps < 10 or self.sample_steps < 10:
            raise ValueError("settle_steps / sample_steps must be >= 10")


@dataclass
class BlockBandRun:
    """Result of one block-scale band run."""

    pressure: np.ndarray              # (T, ny) deposited pressure field [Pa]
    time: np.ndarray                  # (T,) [s]
    config: BlockBandConfig
    summary: dict = field(default_factory=dict)
    block_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    block_wall_force: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "summary": self.summary,
                "n_snapshots": int(self.pressure.shape[0]),
                "n_blocks_loaded": int(np.count_nonzero(self.block_wall_force)),
                "pressure_max": float(np.max(self.pressure)),
                "pressure_mean": float(np.mean(self.pressure))}


# ---------------------------------------------------------------------------
# DEM core
# ---------------------------------------------------------------------------
def _build_bed(cfg: BlockBandConfig, rng: np.random.Generator):
    """Staggered block bed in ``[0, Lx] x [0, B)`` with ``y`` periodic."""
    D, Lx, B = cfg.block, cfg.depth, cfg.band
    n_rows = max(1, int(np.ceil(cfg.layer_rows)))
    row_pitch = cfg.depth / n_rows
    xs, ys, rs = [], [], []
    for j in range(n_rows):
        x0 = min(max((j + 0.5) * row_pitch, 0.56 * D), max(Lx - 0.56 * D, 0.56 * D))
        n_y = max(2, int(round(B / (1.06 * D))))
        offset = 0.5 * (B / n_y) * (j % 2)
        y0 = np.mod((np.arange(n_y) + 0.5) * (B / n_y) + offset, B)
        rad = 0.5 * D * (1.0 + cfg.polydispersity * rng.uniform(-1.0, 1.0, n_y))
        xs.append(np.full(n_y, x0))
        ys.append(y0)
        rs.append(rad)
    x = np.concatenate(xs).astype(float)
    y = np.concatenate(ys).astype(float)
    r = np.concatenate(rs).astype(float)
    x += rng.uniform(-0.03, 0.03, x.size) * D
    y = np.mod(y + rng.uniform(-0.03, 0.03, y.size) * D, B)
    x = np.clip(x, r * 1.01, np.maximum(Lx - r * 1.01, r * 1.02))
    return x, y, r


def deposit_profile(y: np.ndarray, force: np.ndarray, dy: float, ny: int,
                    band: float) -> np.ndarray:
    """Bilinear deposition of point normal forces on a periodic along-face grid."""
    out = np.zeros(ny + 1)
    if y.size:
        pos = np.mod(y, band) / dy
        i0 = np.floor(pos).astype(int)
        w = pos - i0
        i0 = np.clip(i0, 0, ny - 1)
        np.add.at(out, i0, force * (1.0 - w))
        np.add.at(out, i0 + 1, force * w)
    prof = out[:ny].copy()
    prof[0] += out[ny]                       # periodic wrap
    return prof


def reference_stress() -> float:
    """Reference crushing stress ``1.0 MPa`` used only as an order-of-magnitude
    yardstick for "is the bed loaded in the physical range"."""
    return 1.0e6


def inertial_number(cfg: BlockBandConfig, traction: float) -> float:
    """``I = v_drive / sqrt(sigma / rho)``; ``I << 1`` means quasi-static."""
    return float(cfg.drive_speed / np.sqrt(max(traction, 1e-30) / cfg.density))


def simulate_block_band(cfg: BlockBandConfig) -> BlockBandRun:
    """Run the load-controlled block bed and return along-face pressure snapshots.

    The platen is **servo-controlled to a target traction** rather than driven at
    a fixed speed.  Displacement control at a large imposed strain puts the bed
    at tens of MPa (grain overlaps of order 20 %), i.e. outside the physical ice
    crushing range; a servo holds ``p ~ 1-3 MPa`` (``target_traction``) where the
    contact pattern is a genuine force network rather than a jammed lattice.
    """
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    D, B, Lx, dy, ny = cfg.block, cfg.band, cfg.depth, cfg.dy, cfg.ny
    x, y, r = _build_bed(cfg, rng)
    n = x.size

    # block mass and inertia (out-of-plane thickness = h)
    m = cfg.density * np.pi * r**2 * cfg.thickness
    inertia = 0.5 * m * r**2

    omega_n = 2.0 * np.pi / cfg.contact_time
    k_body = 2.0 * m * omega_n**2               # self-contact stiffness
    c_body = 2.0 * cfg.damping * np.sqrt(k_body * m)

    i_idx, j_idx = np.triu_indices(n, 1)
    m_red = m[i_idx] * m[j_idx] / (m[i_idx] + m[j_idx])
    kn_ij = 2.0 * m_red * omega_n**2
    cn_ij = 2.0 * cfg.damping * np.sqrt(kn_ij * m_red)
    ct_ij = 0.35 * cn_ij

    vx = np.zeros(n)
    vy = np.zeros(n)
    om = np.zeros(n)
    dt = cfg.dt
    x_platen = float(Lx)
    f_target = cfg.target_traction * B * cfg.thickness
    v_max = cfg.drive_speed
    n_total = int(cfg.settle_steps + cfg.sample_steps)
    sample_index = np.unique(np.linspace(cfg.settle_steps, n_total - 1,
                                         cfg.n_snapshots).astype(int))

    pressure = np.zeros((cfg.n_snapshots, ny))
    times = np.zeros(cfg.n_snapshots)
    f_wall_hist = np.zeros(n_total)
    f_plat_hist = np.zeros(n_total)
    n_contact_hist = np.zeros(n_total, dtype=int)
    com_v_hist = np.zeros(n_total)
    v_cmd_hist = np.zeros(n_total)
    overlap_max = 0.0
    k = 0
    f_plat_prev = 0.0
    block_force_acc = np.zeros(n)
    block_y_acc = np.zeros(n)
    n_acc = 0
    v_cmd = 0.0
    # ram inertia/damping: heavy compared with the bed so the load is imposed
    # rather than the displacement
    k_bed = float(np.mean(k_body)) / max(4.0, np.sqrt(max(n, 1)))
    m_ram = 20.0 * float(np.sum(m))
    c_ram = 2.0 * 0.7 * float(np.sqrt(k_bed * m_ram))

    for step in range(n_total):
        fx = np.zeros(n)
        fy = np.zeros(n)
        tq = np.zeros(n)
        fn_sum = np.zeros(n)
        f_wall = np.zeros(n)
        f_plat = np.zeros(n)

        # ---- block--block (y periodic: minimum image) ----
        dxm = x[:, None] - x[None, :]
        dym = y[:, None] - y[None, :]
        dym -= B * np.round(dym / B)
        d2 = dxm * dxm + dym * dym
        rs = r[:, None] + r[None, :]
        live = d2[i_idx, j_idx] < rs[i_idx, j_idx] ** 2
        if live.any():
            a, b = i_idx[live], j_idx[live]
            dx_ab, dy_ab = dxm[a, b], dym[a, b]
            d = np.maximum(np.sqrt(dx_ab * dx_ab + dy_ab * dy_ab), 1e-12)
            nx, nyv = dx_ab / d, dy_ab / d
            delta = (r[a] + r[b]) - d
            overlap_max = max(overlap_max, float(np.max(delta / (0.5 * (r[a] + r[b])))))
            vn = (vx[a] - vx[b]) * nx + (vy[a] - vy[b]) * nyv
            fn = np.maximum(kn_ij[live] * delta - cn_ij[live] * vn, 0.0)
            tx, ty = -nyv, nx
            vt = (vx[a] - vx[b]) * tx + (vy[a] - vy[b]) * ty
            ft = -np.sign(vt) * np.minimum(cfg.mu * fn, ct_ij[live] * np.abs(vt))
            fxa, fya = fn * nx + ft * tx, fn * nyv + ft * ty
            np.add.at(fx, a, fxa)
            np.add.at(fy, a, fya)
            np.add.at(fx, b, -fxa)
            np.add.at(fy, b, -fya)
            np.add.at(fn_sum, a, fn)
            np.add.at(fn_sum, b, fn)
            np.add.at(tq, a, -r[a] * ft)
            np.add.at(tq, b, -r[b] * ft)

        # ---- hull wall at x = 0 (outward normal +x) ----
        pen = r - x
        wall = pen > 0.0
        if wall.any():
            f_wall[wall] = np.maximum(k_body[wall] * pen[wall] - c_body[wall] * vx[wall], 0.0)
            fx[wall] += f_wall[wall]
            vt = vy[wall]
            ft = -np.sign(vt) * np.minimum(cfg.mu * f_wall[wall],
                                          0.35 * c_body[wall] * np.abs(vt))
            fy[wall] += ft
            tq[wall] += -r[wall] * ft
            fn_sum[wall] += f_wall[wall]

        # ---- platen at x = x_platen (outward normal -x), servo-load controlled ----
        # load-controlled ram: M_ram * s'' = F_target - F_platen - c_ram * s'
        # (s = platen penetration); a bang-bang velocity servo limit-cycles on a
        # stiff bed, a damped ram settles to the target load smoothly.
        a_ram = (f_target - f_plat_prev - c_ram * v_cmd) / m_ram
        v_cmd += dt * a_ram
        v_cmd = float(np.clip(v_cmd, -v_max, v_max))
        pen_p = (x + r) - x_platen
        plat = pen_p > 0.0
        if plat.any():
            vn_p = -(vx[plat] + v_cmd)
            f_plat[plat] = np.maximum(k_body[plat] * pen_p[plat] - c_body[plat] * vn_p, 0.0)
            fx[plat] -= f_plat[plat]
            vt = vy[plat]
            ft = -np.sign(vt) * np.minimum(cfg.mu * f_plat[plat],
                                          0.35 * c_body[plat] * np.abs(vt))
            fy[plat] += ft
            tq[plat] += -r[plat] * ft
            fn_sum[plat] += f_plat[plat]
        f_plat_prev = float(f_plat.sum())

        # ---- rolling resistance (explicit, capped at mu_r R Fn) ----
        cap = cfg.mu_rolling * r * fn_sum
        tq -= np.clip(0.5 * om * cap, -cap, cap)

        # ---- semi-implicit Euler (dt << contact period) ----
        vx += dt * fx / m
        vy += dt * fy / m
        om += dt * tq / inertia
        x += dt * vx
        y = np.mod(y + dt * vy, B)
        x_platen -= dt * v_cmd

        f_wall_hist[step] = float(f_wall.sum())
        f_plat_hist[step] = float(f_plat.sum())
        n_contact_hist[step] = int(np.count_nonzero(wall))
        com_v_hist[step] = float(np.mean(np.abs(vx)))
        v_cmd_hist[step] = float(v_cmd)

        if k < cfg.n_snapshots and step >= sample_index[k]:
            pressure[k, :] = deposit_profile(y[wall], f_wall[wall], dy, ny, B) / (
                dy * cfg.thickness)
            times[k] = step * dt
            block_force_acc += f_wall
            block_y_acc += y
            n_acc += 1
            k += 1

    # ---- verification diagnostics over the sampled window ----
    lo = int(cfg.settle_steps)
    fw = f_wall_hist[lo:]
    fp = f_plat_hist[lo:]
    mean_wall = float(np.mean(fw))
    mean_plat = float(np.mean(fp))
    balance = abs(mean_wall - mean_plat) / max(mean_plat, 1e-30)
    traction = float(np.sum(pressure.mean(axis=0)) * dy) / B
    summary = {
        "n_blocks": int(n),
        "n_wall_contacts_median": int(np.median(n_contact_hist[lo:])),
        "n_wall_contacts_per_block": float(np.median(n_contact_hist[lo:]) * D / B),
        "mean_wall_force_per_depth": mean_wall,
        "mean_platen_force_per_depth": mean_plat,
        "global_force_balance_rel": float(balance),
        "field_traction_pa": traction,
        "servo_traction_pa": float(mean_plat / B / cfg.thickness),
        "target_traction_pa": float(cfg.target_traction),
        "servo_load_rel_error": float(abs(mean_plat / B / cfg.thickness - cfg.target_traction)
                                     / cfg.target_traction),
        "reference_stress_pa": reference_stress(),
        "traction_over_reference": float(traction / reference_stress()),
        "quasi_static_ratio": float(np.mean(np.abs(com_v_hist[lo:])) / (D / cfg.contact_time)),
        "mean_abs_platen_speed": float(np.mean(np.abs(v_cmd_hist[lo:]))),
        "platen_speed_over_block_per_contact_time": float(
            np.mean(np.abs(v_cmd_hist[lo:])) / (D / cfg.contact_time)),
        "ram_mass_over_bed_mass": float(m_ram / float(np.sum(m))),
        "inertial_number": inertial_number(cfg, traction),
        "overlap_rel_max": float(overlap_max),
        "frac_time_wall_loaded": float(np.mean(fw > 1e-9)),
    }
    block_force = block_force_acc / max(n_acc, 1)
    block_y = block_y_acc / max(n_acc, 1)
    loaded = block_force > 1e-9
    summary["n_blocks_loaded"] = int(np.count_nonzero(loaded))
    summary["load_share_top_decile"] = float(
        np.sum(np.sort(block_force)[-max(1, n // 10):]) / max(float(np.sum(block_force)), 1e-30))
    return BlockBandRun(pressure=pressure, time=times, config=cfg, summary=summary,
                        block_y=block_y[loaded], block_wall_force=block_force[loaded])


# ---------------------------------------------------------------------------
# measurement chain: gauge, profiles, estimators
# ---------------------------------------------------------------------------
def coarse_grain(profile: np.ndarray, cells: int) -> np.ndarray:
    """Periodic box moving average of width ``cells`` (the sensor footprint)."""
    p = np.asarray(profile, dtype=float)
    n = p.size
    k = int(cells)
    if k <= 1:
        return p.copy()
    if k > n:
        raise ValueError("gauge wider than the record")
    pad = k // 2
    ext = np.concatenate([p[-pad:], p, p[:k - pad - 1]]) if pad else p
    return np.convolve(ext, np.ones(k) / k, mode="valid")[:n]


def snapshot_profiles(pressure: np.ndarray, cfg: BlockBandConfig) -> np.ndarray:
    """Per-snapshot coarse-grained along-face pressure profiles ``(T, ny)``."""
    cells = max(1, int(round(cfg.gauge / cfg.dy)))
    return np.stack([coarse_grain(row, cells) for row in np.asarray(pressure, float)])


def time_mean_profile(pressure: np.ndarray) -> np.ndarray:
    """Time-averaged along-face pressure profile (no gauge applied)."""
    return np.asarray(pressure, dtype=float).mean(axis=0)


def shuffle_profile(profile: np.ndarray, seed: int = 0) -> np.ndarray:
    """Negative control: randomly permute the profile (destroys all structure)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(profile, dtype=float).copy()
    rng.shuffle(p)
    return p


@dataclass
class BlockEstimate:
    value_m: float
    value_over_thickness: float
    L_over_lambda: float
    raw: dict
    corrected: dict
    method_spread: float
    regime: dict
    record_length_m: float
    dx_m: float

    def to_dict(self) -> dict:
        return asdict(self)


def _model_from(model):
    """Accept a :class:`BiasModel`, a :class:`TwoStageCalibration` or a dict.

    The two-stage calibrator (:mod:`.calibration`) is the estimator actually
    delivered by the maturation node; the parametric ``BiasModel`` is kept for
    backward compatibility with the earlier campaign stage.
    """
    if isinstance(model, BiasModel):
        return model
    if hasattr(model, "correct"):
        return model
    if isinstance(model, dict):
        if model.get("form") in (None, "two_stage") or "rho_mean" in model:
            from .calibration import TwoStageCalibration
            return TwoStageCalibration.from_dict(model)
        return BiasModel(method=str(model.get("method", "?")),
                         form=str(model.get("form", "inv_poly")),
                         params=tuple(model.get("params", ())))
    return None


def estimate_profile_lambda(
    profile: np.ndarray,
    dx: float,
    thickness: float,
    *,
    models: dict | None = None,
    methods: tuple[str, ...] = METHODS,
    primary: tuple[str, ...] = PRIMARY_METHODS,
) -> BlockEstimate:
    """Corrected correlation length of one along-face pressure profile."""
    p = np.asarray(profile, dtype=float)
    L = p.size * dx
    regime = diagnose_regime(p, dx)
    raw = estimate_all(p, dx, methods=methods)
    corrected: dict[str, float] = {}
    for meth in methods:
        model = None if models is None else _model_from(models.get(meth))
        if model is None:
            corrected[meth] = float("nan")
            continue
        if isinstance(model, BiasModel):
            corrected[meth] = float(correct_length(float(raw[meth]), L, model))
        else:
            corrected[meth] = float(model.correct(float(raw[meth]), L))
    vals = [corrected[meth] for meth in primary
            if meth in corrected and np.isfinite(corrected[meth]) and corrected[meth] > 0.0]
    value = float(np.median(vals)) if vals else float("nan")
    spread = float((max(vals) - min(vals)) / value) if vals and value > 0 else float("nan")
    return BlockEstimate(
        value_m=value,
        value_over_thickness=float(value / thickness) if np.isfinite(value) else float("nan"),
        L_over_lambda=float(L / value) if vals and value > 0 else float("inf"),
        raw={k: float(v) for k, v in raw.items()},
        corrected={k: float(v) for k, v in corrected.items()},
        method_spread=spread,
        regime=regime.to_dict(),
        record_length_m=float(L),
        dx_m=float(dx),
    )


def gauge_ladder(
    pressure: np.ndarray,
    cfg: BlockBandConfig,
    gauge_over_block: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0),
    *,
    models: dict | None = None,
    method: str = "crossing",
) -> list[dict]:
    """Correlation length versus sensor footprint (the gauge law).

    A *physical* patchiness saturates once the footprint drops below the
    structural scale; a *kernel-limited* number keeps growing linearly with the
    footprint.  The two are distinguishable without any theory input, which is
    why the ladder is the primary mechanism diagnostic.
    """
    rows = []
    mean = time_mean_profile(pressure)
    for g in gauge_over_block:
        cells = max(1, int(round(g * cfg.block / cfg.dy)))
        prof = coarse_grain(mean, cells)
        est = estimate_profile_lambda(prof, cfg.dy, cfg.thickness, models=models,
                                      methods=(method,), primary=(method,))
        lam = float(est.corrected.get(method, float("nan")))
        rows.append({
            "gauge_over_block": float(g),
            "gauge_over_thickness": float(g * cfg.block_over_thickness),
            "gauge_m": float(g * cfg.block),
            "cells": int(cells),
            "lambda_m": lam,
            "lambda_over_gauge": float(lam / (g * cfg.block)) if g > 0 else float("nan"),
            "lambda_over_block": float(lam / cfg.block) if np.isfinite(lam) else float("nan"),
            "lambda_over_thickness": float(lam / cfg.thickness) if np.isfinite(lam) else float("nan"),
            "regime": est.regime.get("label"),
        })
    return rows


# ---------------------------------------------------------------------------
# shot-noise null: the analytic explanation of a kernel-limited measurement
# ---------------------------------------------------------------------------
def shot_noise_profile(
    cfg: BlockBandConfig,
    *,
    contacts_per_block: float,
    force_mean: float,
    force_cv: float = 1.0,
    n_snapshots: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Deposition of a *Poisson* contact process -- the no-physics null.

    Contacts are placed at random along the face with density
    ``contacts_per_block / D``; forces are lognormal with the given coefficient
    of variation; the same bilinear deposition + gauge chain is applied as for
    the DEM.  ``lam`` of this field is what the measurement chain returns when
    the underlying contact pattern carries **no** spatial structure, so it is the
    null against which a DEM number is judged.
    """
    rng = np.random.default_rng(seed)
    ny, dy, B, D = cfg.ny, cfg.dy, cfg.band, cfg.block
    cells = max(1, int(round(cfg.gauge / dy)))
    rate = contacts_per_block / D
    sigma = np.sqrt(np.log1p(max(force_cv, 1e-6) ** 2))
    mu = np.log(max(force_mean, 1e-30)) - 0.5 * sigma**2
    out = np.zeros((n_snapshots, ny))
    for t in range(n_snapshots):
        nc = rng.poisson(rate * B)
        yc = rng.uniform(0.0, B, nc)
        fc = rng.lognormal(mu, sigma, nc)
        fine = deposit_profile(yc, fc, dy, ny, B)
        out[t, :] = coarse_grain(fine, cells) / (dy * cfg.thickness)
    return out


def shot_noise_lambda(
    cfg: BlockBandConfig,
    *,
    contacts_per_block: float,
    force_mean: float,
    force_cv: float = 1.0,
    n_real: int = 40,
    method: str = "crossing",
    models: dict | None = None,
    seed: int = 0,
) -> dict:
    """Ensemble ``lam`` of the Poisson (structureless) null field."""
    pressure = shot_noise_profile(cfg, contacts_per_block=contacts_per_block,
                                  force_mean=force_mean, force_cv=force_cv,
                                  n_snapshots=n_real, seed=seed)
    mean = time_mean_profile(pressure)
    est = estimate_profile_lambda(mean, cfg.dy, cfg.thickness, models=models,
                                  methods=(method,), primary=(method,))
    per_real = []
    for row in pressure:
        e = estimate_profile_lambda(coarse_grain(row, max(1, int(round(
            cfg.gauge / cfg.dy)))), cfg.dy, cfg.thickness, models=models,
            methods=(method,), primary=(method,))
        per_real.append(float(e.raw.get(method, float("nan"))))
    arr = np.asarray(per_real, dtype=float)
    arr = arr[np.isfinite(arr)]
    lam = float(est.corrected.get(method, float("nan")))
    return {
        "method": method,
        "lambda_m": lam,
        "calibrated": False,
        "lambda_over_thickness": float(lam / cfg.thickness) if np.isfinite(lam) else float("nan"),
        "single_realisation_median_m": float(np.median(arr)) if arr.size else float("nan"),
        "single_realisation_p16_m": float(np.percentile(arr, 16)) if arr.size else float("nan"),
        "single_realisation_p84_m": float(np.percentile(arr, 84)) if arr.size else float("nan"),
        "n_real": int(arr.size),
        "contacts_per_block": float(contacts_per_block),
        "force_cv": float(force_cv),
    }

# ---------------------------------------------------------------------------
# kernel-free structural measurement: the block-load sequence
# ---------------------------------------------------------------------------
def block_load_acf(block_y: np.ndarray, block_force: np.ndarray, block: float,
                   *, max_lag: int = 6, periodic_length: float | None = None) -> dict:
    """Autocorrelation of the *per-block* wall load -- no deposition kernel.

    This is the only measurement in the module that is free of the pressure
    sensor footprint: it asks whether neighbouring contact units carry
    correlated load.  ``rho_k`` is normalised by the sequence variance and
    averaged over all pairs at that lag (``k`` in units of one block spacing);
    ``lambda_over_block = 1/2 + sum_{k>=1} rho_k`` is the integral scale of a
    unit-spacing sequence (the ``1/2`` is the zero-width self term, exactly the
    value of a white sequence).
    """
    y = np.asarray(block_y, dtype=float)
    f = np.asarray(block_force, dtype=float)
    order = np.argsort(y)
    y, f = y[order], f[order]
    n = f.size
    out = {"n_blocks_loaded": int(n), "block_spacing_m": float(block),
           "force_mean": float(np.mean(f)) if n else float("nan"),
           "force_cv": float(np.std(f) / np.mean(f)) if n and np.mean(f) > 0 else float("nan")}
    if n < 6:
        out.update({"rho": [], "lambda_over_block": float("nan"),
                    "note": "too few loaded blocks"})
        return out
    x = f - f.mean()
    var = float(np.mean(x * x))
    rho = []
    if var <= 0.0:
        out.update({"rho": [0.0] * max_lag, "lambda_over_block": 0.5})
        return out
    for k in range(1, max_lag + 1):
        if n - k < 3:
            rho.append(float("nan"))
            continue
        # pairs are (i, i+k) in y order; their separation is k block spacings
        rho.append(float(np.mean(x[:-k] * x[k:]) / var))
    # integral scale of the load sequence: sum the lags while the correlation is
    # still positive (first-zero truncation).  Summing past the first zero would
    # add the finite-sample noise floor of the ACF (E[rho_k] ~ -1/(n-1) for a white
    # sequence) and push a genuinely white sequence to a *negative* length.
    acc = 0.0
    for r in rho:
        if not np.isfinite(r) or r <= 0.0:
            break
        acc += float(r)
    lam = 0.5 + acc
    out.update({"rho": rho, "lambda_over_block": lam,
                "rho_lag1": rho[0] if rho else float("nan"),
                "separation_over_block": [float(k) for k in range(1, max_lag + 1)]})
    return out


def _acf_from_block_sequence(block_acf: list[float], block: float, dx: float,
                             n: int) -> np.ndarray:
    """ACF of a unit-spacing load sequence convolved with the deposition and
    gauge kernels (uniform block spacing approximation)."""
    r = np.arange(n) * dx
    rho = np.zeros(n)
    # linear interpolation of rho_k on the block grid, zero beyond the last lag
    lags = np.asarray(block_acf, dtype=float)
    good = np.isfinite(lags)
    if good.sum() >= 2:
        rho = np.interp(r / block, np.arange(1, lags.size + 1)[good], lags[good],
                        left=1.0, right=0.0)
    else:
        rho = np.where(r < 0.5 * block, 1.0, 0.0)
    return rho


def forward_model_lambda(block_acf: list[float], cfg: BlockBandConfig,
                         *, method: str = "crossing", n: int = 4096,
                         seed: int = 0) -> dict:
    """Predict ``lam`` from the block-load ACF *plus* the measurement chain.

    The prediction uses the measured per-block load correlation ``rho_k`` and the
    known deposition + gauge kernels, synthesises a Gaussian field with that ACF
    and applies the same estimator.  If the predicted value reproduces the
    measured ``lam`` the number is fully explained by
    ``(rho_k, block spacing, gauge)`` -- nothing else is hidden in it.
    """
    dy = cfg.dy
    r = np.arange(n) * dy
    rho = _acf_from_block_sequence(block_acf, cfg.block, dy, n)
    tri = np.clip(1.0 - r / dy, 0.0, 1.0)            # deposition (bilinear)
    g = max(cfg.gauge, dy)
    box = np.clip(1.0 - r / g, 0.0, 1.0)             # gauge (box average)
    acf = rho * tri * box
    sym = np.concatenate([acf, acf[-2:0:-1]])        # symmetric -> real PSD
    psd = np.real(np.fft.rfft(sym))
    amp = np.sqrt(np.maximum(psd, 0.0))
    rng = np.random.default_rng(seed)
    phases = np.exp(2j * np.pi * rng.uniform(0.0, 1.0, amp.size))
    field = np.fft.irfft(amp * phases, n=n)
    est = estimate_profile_lambda(field, dy, cfg.thickness, models=None,
                                  methods=(method,), primary=(method,))
    lam = float(est.raw.get(method, float("nan")))
    return {"method": method, "lambda_m": lam,
            "lambda_over_block": float(lam / cfg.block) if np.isfinite(lam) else float("nan"),
            "lambda_over_thickness": float(lam / cfg.thickness) if np.isfinite(lam) else float("nan"),
            "rho_used": [float(v) for v in block_acf],
            "gauge_over_block": float(cfg.gauge_over_block)}
