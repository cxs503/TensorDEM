"""Cross-validation case specification for inclined-plane (chute) granular flow.

Physical reference
------------------
Silbert, Ertas, Grest, Halsey, Levine & Plimpton, "Granular flow down an
inclined plane: Bagnold scaling and rheology", Phys. Rev. E **64**, 051302
(2001) -- the paper-reproduction line SILBERT2001 of this repository.  This
module owns the *one* case specification that is used three times:

* by :mod:`tensordem.chute_flow.engine` (our solver),
* by the LIGGGHTS input deck (``artifacts/xval_chute2001/oss/liggghts``),
* by the second open-source DEM deck.

The geometry (base roughness layer + initial particle coordinates) is written
once to CSV, hashed, and read by *every* tool, so the initial condition and the
wall roughness are bit-identical across codes -- the hard precondition for a
cross-tool comparison to be meaningful.  Nothing here depends on any solver.

Units are SI.  The particle diameter ``d = 1 mm`` is the length unit; every
reported quantity is non-dimensionalised with ``d``, ``sqrt(g d)`` (velocity)
and ``sqrt(d / g)`` (time), which is also how LIGGGHTS/``units si`` decks are
checked for consistency.

Contact law (identical in all three codes)
------------------------------------------
normal      ``F_n = max(k_n delta - gamma_n v_n, 0)``      (linear spring-dashpot)
tangential  Cundall-Strack incremental spring, saturated at ``mu F_n``
rolling     disabled (``mu_rolling = 0``), as in Silbert et al.
restitution ``e = 0.88``  ->  ``gamma_n = 2 zeta sqrt(k_n m / 2)``,
``zeta = |ln e| / sqrt(pi^2 + ln^2 e)``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

__all__ = [
    "ChuteCase",
    "DEFAULT_CASE",
    "make_base",
    "make_initial_particles",
    "write_case_spec",
    "load_case_spec",
]

DEFAULT_CASE: dict = {
    # particle / contact
    "d": 1.0e-3,          # particle diameter [m]
    "density": 2500.0,    # [kg/m3]
    "kn": 500.0,          # normal stiffness [N/m]
    "kt_ratio": 0.5,      # k_t / k_n
    "restitution": 0.88,  # e (-> normal dashpot)
    "mu": 0.5,            # grain-grain / grain-base friction
    "mu_rolling": 0.0,    # rolling resistance disabled (Silbert et al.)
    "dt": 1.0e-5,         # time step [s]
    "gravity": 9.81,      # [m/s2]
    # cell (periodic in the flow x and span y directions)
    "lx_d": 12.0,         # cell length in diameters
    "ly_d": 12.0,         # cell width in diameters
    "n_target": 1200,     # target particle number (fixed -> fixed thickness)
    "theta_deg": 24.0,    # inclination
    # rough base
    "base_pitch_d": 0.9,     # in-plane pitch of the fixed base spheres
    "base_center_z_d": 0.5,  # base sphere centre height above the datum plane
    # initial packing
    "init_gap": 1.02,     # in-plane lattice spacing / d
    "jitter": 0.02,       # lattice jitter (fraction of a diameter)
    "seed": 2001,
}


@dataclass(frozen=True)
class ChuteCase:
    """Immutable case specification (see :data:`DEFAULT_CASE`)."""

    d: float = 1.0e-3
    density: float = 2500.0
    kn: float = 500.0
    kt_ratio: float = 0.5
    restitution: float = 0.88
    mu: float = 0.5
    mu_rolling: float = 0.0
    dt: float = 1.0e-5
    gravity: float = 9.81
    lx_d: float = 12.0
    ly_d: float = 12.0
    n_target: int = 1200
    theta_deg: float = 24.0
    base_pitch_d: float = 0.9
    base_center_z_d: float = 0.5
    init_gap: float = 1.02
    jitter: float = 0.02
    seed: int = 2001
    case_id: str = ""

    # ---------------- derived quantities ----------------
    @property
    def radius(self) -> float:
        return 0.5 * self.d

    @property
    def mass(self) -> float:
        return self.density * math.pi / 6.0 * self.d ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def lx(self) -> float:
        return self.lx_d * self.d

    @property
    def ly(self) -> float:
        return self.ly_d * self.d

    @property
    def theta(self) -> float:
        return math.radians(self.theta_deg)

    @property
    def z_wall(self) -> float:
        """Height of the top of the fixed base spheres (the wall datum)."""
        return (self.base_center_z_d + 0.5) * self.d

    @property
    def t_ref(self) -> float:
        return math.sqrt(self.d / self.gravity)

    @property
    def v_ref(self) -> float:
        return math.sqrt(self.gravity * self.d)

    @property
    def gravity_vector(self) -> tuple[float, float, float]:
        return (self.gravity * math.sin(self.theta), 0.0,
                -self.gravity * math.cos(self.theta))

    @property
    def zeta(self) -> float:
        e = max(min(self.restitution, 1.0 - 1e-12), 1e-6)
        ln_e = math.log(e)
        return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)

    @property
    def gamma_n(self) -> float:
        """Normal dashpot coefficient of a pair (reduced mass m/2)."""
        return 2.0 * self.zeta * math.sqrt(self.kn * 0.5 * self.mass)

    @property
    def contact_period(self) -> float:
        return 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn)

    # ---------------- io ----------------
    @classmethod
    def from_dict(cls, payload: dict) -> "ChuteCase":
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in payload.items() if k in known}
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "ChuteCase":
        d = self.to_dict()
        d.update(kw)
        return ChuteCase.from_dict(d)

    def summary(self) -> dict:
        """Numbers a reviewer needs to check the hard preconditions."""
        return {
            "case_id": self.case_id,
            "d_m": self.d,
            "density_kg_m3": self.density,
            "kn_N_m": self.kn,
            "kt_N_m": self.kt_ratio * self.kn,
            "restitution": self.restitution,
            "mu": self.mu,
            "mu_rolling": self.mu_rolling,
            "gamma_n_N_s_m": self.gamma_n,
            "zeta": self.zeta,
            "dt_s": self.dt,
            "dt_over_contact_period": self.dt / self.contact_period,
            "lx_m": self.lx,
            "ly_m": self.ly,
            "lx_over_d": self.lx_d,
            "ly_over_d": self.ly_d,
            "theta_deg": self.theta_deg,
            "gravity_vector_m_s2": list(self.gravity_vector),
            "v_ref_m_s": self.v_ref,
            "t_ref_s": self.t_ref,
            "z_wall_m": self.z_wall,
            "n_target": self.n_target,
            "seed": self.seed,
        }


def make_base(case: ChuteCase) -> np.ndarray:
    """Fixed rough base: jittered square lattice of spheres, same radius as the
    grains, overlapping by 10 % (no gaps a grain could fall through).

    The base is periodic in x and y with the cell, so the same coordinates can
    be re-used in a periodic LIGGGHTS box.  Returns ``(m, 3)`` centres.
    """
    rng = np.random.default_rng(case.seed)
    nx = max(4, int(round(case.lx / (case.base_pitch_d * case.d))))
    ny = max(4, int(round(case.ly / (case.base_pitch_d * case.d))))
    xs = (np.arange(nx) + 0.5) * case.lx / nx
    ys = (np.arange(ny) + 0.5) * case.ly / ny
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    span = case.lx / nx
    grid = grid + rng.uniform(-0.22, 0.22, size=grid.shape) * span
    grid[:, 0] %= case.lx
    grid[:, 1] %= case.ly
    z = np.full(len(grid), case.base_center_z_d * case.d)
    return np.column_stack([grid, z])


def make_initial_particles(case: ChuteCase) -> np.ndarray:
    """Loose lattice slab of ``n_target`` particles above the rough base.

    The in-plane lattice is commensurate with the periodic cell (``lx / nx``),
    so the slab is periodic in x and y exactly.  The number of full layers is
    ``round(n_target / (nx ny))``; the *actual* particle number is written into
    the case spec and used by every tool.
    """
    gap = case.init_gap * case.d
    nx = max(2, int(round(case.lx / gap)))
    ny = max(2, int(round(case.ly / gap)))
    per_layer = nx * ny
    n_layers = max(1, int(round(case.n_target / per_layer)))
    rng = np.random.default_rng(case.seed + 17)
    xs = (np.arange(nx) + 0.5) * case.lx / nx
    ys = (np.arange(ny) + 0.5) * case.ly / ny
    slab = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    pts = []
    for k in range(n_layers):
        layer = slab.copy()
        layer += rng.uniform(-case.jitter, case.jitter, size=layer.shape) * case.d
        z = case.z_wall + 0.62 * case.d + k * gap
        pts.append(np.column_stack([layer, np.full(len(layer), z)]))
    pos = np.vstack(pts)
    pos[:, 0] %= case.lx
    pos[:, 1] %= case.ly
    return pos


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_case_spec(case: ChuteCase, outdir: str | Path) -> dict:
    """Write ``case.json`` + ``base_spheres.csv`` + ``particles_init.csv``.

    All three tools read these files; the returned manifest carries the hashes
    that end up in ``result.case_spec``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    base = make_base(case)
    init = make_initial_particles(case)
    np.savetxt(out / "base_spheres.csv", base, delimiter=",",
               header="x_m,y_m,z_m", comments="")
    np.savetxt(out / "particles_init.csv", init, delimiter=",",
               header="x_m,y_m,z_m", comments="")
    n_particles = int(len(init))
    h_vol = n_particles * (math.pi / 6.0) * case.d ** 3 / (case.lx * case.ly)
    manifest = {
        "case_spec": case.summary(),
        "n_particles": n_particles,
        "n_base_spheres": int(len(base)),
        "initial_volume_thickness_m": h_vol,
        "initial_volume_thickness_d": h_vol / case.d,
        "files": {
            "base_spheres.csv": {"sha256": _sha256(out / "base_spheres.csv"),
                                 "rows": int(len(base))},
            "particles_init.csv": {"sha256": _sha256(out / "particles_init.csv"),
                                   "rows": n_particles},
        },
        "sampling": {
            "z_binning": "bins of 0.5 d above z_wall, particle-centre sampling",
            "profile_average": "time average over the steady window, "
                               "arithmetic mean of the velocities in each bin",
            "thickness": "h = N * (pi d^3 / 6) / (lx * ly)  (volume-equivalent)",
            "bagnold_fit": "vx(z) = C [h^p - (h - z)^p], p fitted on "
                           "[0.6, 3.0], z <= 0.9 h",
        },
    }
    (out / "case.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_case_spec(path: str | Path) -> tuple[ChuteCase, dict]:
    payload = json.loads(Path(path).read_text())
    case = ChuteCase.from_dict(payload["case_spec"])
    return case, payload