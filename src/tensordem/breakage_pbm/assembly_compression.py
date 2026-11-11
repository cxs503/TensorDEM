"""Load-controlled uniaxial compression of a breakable sphere assembly (DEM side).

Why this module
---------------
The DEM -> PBM cross validation needs a *direct* DEM simulation of a particle
population whose size distribution evolves by crushing.  The repository's
integrators (:class:`tensordem.dem3d.IceDEM3D`, :class:`tensordem.dem.CrushingDEM`)
either cannot change the particle count or are serial Python loops, so the
10^4-particle oedometric test is implemented here as a vectorised soft-sphere
DEM with:

* sphere-sphere contacts found by a k-d tree (cell-list equivalent), normal
  spring-dashpot plus Coulomb friction limited by ``mu``;
* rigid walls: four fixed lateral walls and a fixed floor give uniaxial-strain
  (oedometric) conditions, the top platen is a **load-controlled body** whose
  applied load is the prescribed stress history;
* per-particle Weibull strength with the weakest-link size effect
  (``sigma_c(V) = sigma_0 (V_0/V)**(1/m)``), drawn once at creation, so a
  particle breaks when its nominal contact stress ``F_c/d**2`` exceeds its own
  strength (the same convention as the single-particle crushing test);
* breakage by volume-conserving replacement with a daughter rule
  (:mod:`tensordem.breakage_pbm.daughter_distribution`), with a mass ledger
  around every replacement;
* Cundall non-viscous local damping so that the loading stays quasi-static, and
  **mass scaling** (``mass_scale``) to enlarge the stable time step -- the
  standard DEM trick for long quasi-static runs; the loading rate is slow enough
  that the scaled inertia does not change the quasi-static response (checked in
  the study by comparing the stress history of two mass scales).

Outputs
-------
``run()`` returns the recorded PSD samples ``d_1, d_2, d_4, d_8`` (volume
weighted diameters), the measured platen stress history, the breakage event log
(with the mass ledger), and the **effective-stress ratio**
``kappa = sigma_eff / sigma_macro`` measured *before the first breakage event*,
which is the single micro-macro input the population balance needs
(``sigma_eff = (E[sigma_p**m])**(1/m)``, Weibull's stress integral).  Everything
after that window is a prediction.

All quantities are SI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial import cKDTree

from .daughter_distribution import UniformDaughter
from .weibull_breakage import WeibullStrength, effective_stress

__all__ = ["AssemblyParams", "BreakableAssembly", "measure_kappa"]


@dataclass(frozen=True)
class AssemblyParams:
    """Packing, material and loading parameters of the assembly test."""

    n_particles: int = 10_000
    solid_fraction: float = 0.52      # target initial packing fraction
    box_width: float = 0.0            # lateral size [m]; 0 -> derived from phi
    d_mean: float = 0.010             # mean particle diameter [m]
    d_cv: float = 0.10                # lognormal coefficient of variation
    rho: float = 2500.0               # particle density [kg/m^3]
    mass_scale: float = 1.0e4         # density multiplier (time-step scaling)
    kn: float = 1.0e6                 # contact normal stiffness [N/m]
    zeta: float = 0.8                 # contact damping ratio
    mu: float = 0.0                   # Coulomb friction coefficient
    local_damping: float = 0.7        # Cundall non-viscous local damping
    strength: WeibullStrength = field(
        default_factory=lambda: WeibullStrength(m=3.0, sigma_0=1.0e6, v_0=1e-9))
    daughter: object = field(default_factory=UniformDaughter)
    stress_max: float = 3.0e6         # peak applied platen stress [Pa]
    t_end: float = 8.0                # duration of the stress ramp [s]
    record_times: tuple = (1.0, 2.0, 4.0, 8.0)
    dt_factor: float = 0.25
    max_breaks_per_step: int = 5
    max_generations: int = 1      # 1 = primary breakage only (see docs)
    min_fragment_age: int = 60      # steps a new fragment must be loaded before breaking
    min_daughter_fraction: float = 0.08  # DEM resolution floor on daughter volume
    max_overlap_fraction: float = 0.2   # overlap cap (fragment-insertion stabiliser)
    max_speed: float = 5.0              # velocity clamp (fragment-insertion stabiliser)
    platen_mass_ratio: float = 0.1      # platen mass / bed mass (load-control inertia)
    seed: int = 0

    @property
    def box(self) -> float:
        """Cubic sample edge for the requested solid fraction."""
        if self.box_width:
            return float(self.box_width)
        v = self.n_particles * self.particle_volume_mean / float(self.solid_fraction)
        return float(v ** (1.0 / 3.0))

    @property
    def area(self) -> float:
        return self.box ** 2

    @property
    def particle_volume_mean(self) -> float:
        return math.pi * self.d_mean ** 3 / 6.0

    @property
    def particle_mass_mean(self) -> float:
        return self.rho * self.mass_scale * self.particle_volume_mean

    @property
    def dt(self) -> float:
        # the time step must resolve the *smallest* particle present, i.e. the
        # daughter-resolution floor, not the mean particle
        m_min = self.min_daughter_fraction * self.particle_mass_mean
        return float(self.dt_factor * math.sqrt(m_min / self.kn))

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end / self.dt))


def _lognormal_diameters(n: int, d_mean: float, cv: float,
                         rng: np.random.Generator) -> np.ndarray:
    sigma = math.sqrt(math.log1p(cv * cv))
    mu = math.log(d_mean) - 0.5 * sigma ** 2
    d = rng.lognormal(mu, sigma, size=n)
    return d * (d_mean / float(np.mean(d)))  # exact mean


class BreakableAssembly:
    """Vectorised oedometric DEM with breakable polydisperse spheres."""

    def __init__(self, params: AssemblyParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self._build_packing()
        self.t = 0.0
        self.step_count = 0
        self.events = []
        self.stress_history = []
        self.psd = {}
        self.mass_ledger = {"mass_0": float(np.sum(self.mass))}
        self.kappa_window = {"sigma_eff": [], "sigma_macro": []}
        self.first_break_time = None
        self.broken_total = 0

    # ------------------------------------------------------------------ setup
    def _build_packing(self) -> None:
        """Lattice start + radius growth to the target solid fraction."""
        p = self.p
        n = int(p.n_particles)
        L = float(p.box)
        n_side = max(2, int(round(n ** (1.0 / 3.0))))
        coords = (np.arange(n_side) + 0.5) * (L / n_side)
        grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 3)
        if grid.shape[0] > n:
            grid = grid[:n]
        while grid.shape[0] < n:
            extra = grid[self.rng.integers(0, grid.shape[0], n - grid.shape[0])]
            grid = np.vstack([grid, extra + self.rng.normal(0.0, 2e-3, size=extra.shape)])
        spacing = L / n_side
        jitter = 0.12 * spacing
        pos = grid + self.rng.normal(0.0, jitter, size=grid.shape)
        pos = np.clip(pos, 1e-4, L - 1e-4)
        d = _lognormal_diameters(n, p.d_mean, p.d_cv, self.rng)
        d = d * (0.55 * spacing / float(np.max(d)))   # start below touching

        self.pos, self.d = pos, d
        self.r = 0.5 * d
        self.volume = math.pi * d ** 3 / 6.0
        self.mass = p.rho * p.mass_scale * self.volume
        self.vel = np.zeros_like(pos)

        target = float(p.solid_fraction)
        for _ in range(60):
            phi = float(np.sum(self.volume)) / L ** 3
            if phi >= target:
                break
            factor = min(1.25, (target / max(phi, 1e-9)) ** (1.0 / 3.0))
            factor = max(factor, 1.01)
            self.r *= factor
            self.volume = 4.0 / 3.0 * math.pi * self.r ** 3
            self._relax(max_iter=60, tol=5e-3 * spacing)
        self.d = 2.0 * self.r
        self.volume = 4.0 / 3.0 * math.pi * self.r ** 3
        self.mass = p.rho * p.mass_scale * self.volume
        self.strength = p.strength.sample_strength(self.volume, self.rng)
        self.pos[:, 0] = np.clip(self.pos[:, 0], self.r, L - self.r)
        self.pos[:, 1] = np.clip(self.pos[:, 1], self.r, L - self.r)
        self.pos[:, 2] = np.clip(self.pos[:, 2], self.r, L - self.r)
        self.platen_z = float(np.max(self.pos[:, 2] + self.r))
        self.age = np.zeros(self.pos.shape[0], dtype=np.int64)
        self.generation = np.zeros(self.pos.shape[0], dtype=np.int64)
        self.platen_mass = float(p.platen_mass_ratio) * float(np.sum(self.mass))
        self.platen_vel = 0.0
        self.phi0 = float(np.sum(self.volume)) / L ** 3

    def _relax(self, max_iter: int = 40, tol: float = 1e-6) -> None:
        """Geometric overlap removal (mass weighted, box preserving)."""
        L = float(self.p.box)
        for _ in range(int(max_iter)):
            pairs = cKDTree(self.pos).query_pairs(
                2.0 * float(np.max(self.r)) * 1.02, output_type="ndarray")
            if pairs.size == 0:
                return
            i, j = pairs[:, 0], pairs[:, 1]
            rij = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(rij, axis=1)
            overlap = (self.r[i] + self.r[j]) - dist
            worst = float(np.max(overlap))
            if worst <= tol:
                return
            act = overlap > 0.0
            if not np.any(act):
                return
            i, j = i[act], j[act]
            n_hat = rij[act] / np.clip(dist[act], 1e-12, None)[:, None]
            push = 0.55 * overlap[act]
            shift = np.zeros_like(self.pos)
            np.add.at(shift, i, -push[:, None] * n_hat)
            np.add.at(shift, j, +push[:, None] * n_hat)
            self.pos = self.pos + shift
            self.pos[:, 0] = np.clip(self.pos[:, 0], self.r, L - self.r)
            self.pos[:, 1] = np.clip(self.pos[:, 1], self.r, L - self.r)
            self.pos[:, 2] = np.clip(self.pos[:, 2], self.r, L - self.r)

    # ------------------------------------------------------------------ step
    def _contacts(self):
        cut = 2.0 * float(np.max(self.r)) * 1.02
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(cut, output_type="ndarray")
        return pairs

    def step(self) -> None:
        p = self.p
        dt = p.dt
        n = self.pos.shape[0]
        force = np.zeros_like(self.pos)
        pairs = self._contacts()
        i, j = pairs[:, 0], pairs[:, 1]
        rij = self.pos[i] - self.pos[j]      # from j to i
        dist = np.linalg.norm(rij, axis=1)
        overlap = (self.r[i] + self.r[j]) - dist
        active = overlap > 0.0
        i, j, rij, dist, overlap = i[active], j[active], rij[active], dist[active], overlap[active]
        force_n = np.zeros_like(overlap)
        if overlap.size:
            cap = p.max_overlap_fraction * (self.r[i] + self.r[j])
            overlap = np.minimum(overlap, cap)
            safe = np.where(dist > 1e-9, dist, 0.0)
            n_hat = np.where(safe[:, None] > 0.0,
                             rij / np.where(safe > 0.0, safe, 1.0)[:, None], 0.0)
            v_rel = self.vel[i] - self.vel[j]
            v_n = np.sum(v_rel * n_hat, axis=1)
            m_ij = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
            c_n = 2.0 * p.zeta * np.sqrt(p.kn * m_ij)
            force_n = p.kn * overlap + c_n * v_n
            force_n = np.clip(force_n, 0.0, None)
            fvec = force_n[:, None] * n_hat
            if p.mu > 0.0:
                v_t = v_rel - v_n[:, None] * n_hat
                v_t_mag = np.linalg.norm(v_t, axis=1)
                f_t = -np.clip(p.mu * force_n, 0.0, None)[:, None] * v_t / np.clip(
                    v_t_mag, 1e-12, None)[:, None]
                fvec = fvec + f_t
            np.add.at(force, i, +fvec)
            np.add.at(force, j, -fvec)

        # particle stress used by the breakage criterion: largest contact force
        contact_force = np.zeros(n)
        if overlap.size:
            np.maximum.at(contact_force, i, force_n)
            np.maximum.at(contact_force, j, force_n)

        # walls (lateral + floor fixed, top platen load-controlled body)
        w = float(p.box)
        h = float(p.box)
        cap_pen = p.max_overlap_fraction * 2.0 * self.r
        for axis in (0, 1):
            pen_lo = self.r - (self.pos[:, axis])
            act = pen_lo > 0.0
            f = np.where(act, p.kn * np.clip(pen_lo, 0.0, cap_pen)
                         + np.where(self.vel[:, axis] < 0.0,
                                    2.0 * p.zeta * np.sqrt(p.kn * self.mass) * self.vel[:, axis], 0.0), 0.0)
            force[:, axis] += f
            pen_hi = self.r - (w - self.pos[:, axis])
            act = pen_hi > 0.0
            f = np.where(act, p.kn * np.clip(pen_hi, 0.0, cap_pen)
                         + np.where(self.vel[:, axis] > 0.0,
                                    -2.0 * p.zeta * np.sqrt(p.kn * self.mass) * self.vel[:, axis], 0.0), 0.0)
            force[:, axis] -= f
        pen_floor = self.r - self.pos[:, 2]
        act = pen_floor > 0.0
        f_floor = np.where(act, p.kn * np.clip(pen_floor, 0.0, cap_pen)
                           + np.where(self.vel[:, 2] < 0.0,
                                      2.0 * p.zeta * np.sqrt(p.kn * self.mass) * self.vel[:, 2], 0.0), 0.0)
        force[:, 2] += f_floor
        pen_top = (self.pos[:, 2] + self.r) - self.platen_z
        act = pen_top > 0.0
        closing = np.clip(self.vel[:, 2] - self.platen_vel, 0.0, None)
        f_top = np.where(act, p.kn * np.clip(pen_top, 0.0, cap_pen)
                         + 2.0 * p.zeta * np.sqrt(p.kn * self.mass) * closing, 0.0)
        force[:, 2] -= f_top

        applied_stress = p.stress_max * min(self.t / p.t_end, 1.0)
        platen_reaction = float(np.sum(f_top))

        # local damping (Cundall) keeps the loading quasi-static
        if p.local_damping > 0.0:
            force = force - p.local_damping * np.abs(force) * np.sign(self.vel)

        self.vel += dt * force / self.mass[:, None]
        speed = np.linalg.norm(self.vel, axis=1)
        too_fast = speed > p.max_speed
        if np.any(too_fast):
            self.vel[too_fast] *= (p.max_speed / speed[too_fast])[:, None]
        self.pos += dt * self.vel
        self.pos[:, 0] = np.clip(self.pos[:, 0], self.r, w - self.r)
        self.pos[:, 1] = np.clip(self.pos[:, 1], self.r, w - self.r)

        # load-controlled platen body
        acc = (platen_reaction - applied_stress * p.area) / self.platen_mass
        self.platen_vel = (self.platen_vel + dt * acc) * 0.95
        self.platen_z += dt * self.platen_vel

        self.t += dt
        self.step_count += 1
        self.age += 1
        self.stress_history.append((self.t, applied_stress,
                                    platen_reaction / p.area))
        sigma_p = contact_force / (self.d ** 2)
        if self.first_break_time is None and float(np.mean(sigma_p)) > 0.0:
            if np.all(np.array(self.kappa_window["sigma_macro"]) == 0.0) or True:
                self.kappa_window["sigma_eff"].append(effective_stress(sigma_p, p.strength.m))
                self.kappa_window["sigma_macro"].append(applied_stress)
        self._break(sigma_p)

    # --------------------------------------------------------------- damage
    def _break(self, sigma_p: np.ndarray) -> None:
        p = self.p
        over = ((sigma_p > self.strength) & (self.age >= int(p.min_fragment_age))
                & (self.generation < int(p.max_generations)))
        if not np.any(over):
            return
        idx = np.flatnonzero(over)[: int(p.max_breaks_per_step)]
        if self.first_break_time is None:
            self.first_break_time = self.t
        new_pos, new_vel, new_d, new_gen = [], [], [], []
        for k in idx:
            v0 = float(self.volume[k])
            daughters = np.asarray(p.daughter.sample(v0, self.rng), dtype=np.float64)
            daughters = daughters * (v0 / float(np.sum(daughters)))
            floor = float(p.min_daughter_fraction) * v0
            if np.any(daughters < floor):   # DEM resolution floor (see docs)
                daughters = np.clip(daughters, floor, None)
                daughters = daughters * (v0 / float(np.sum(daughters)))
            radii_d = [(3.0 * vd / (4.0 * math.pi)) ** (1.0 / 3.0) for vd in daughters]
            placed = []
            for rd in radii_d:
                best = None
                for _ in range(24):
                    u = self.rng.normal(0.0, 1.0, 3)
                    u = u / max(np.linalg.norm(u), 1e-12)
                    cand = self.pos[k] + u * max(self.r[k] - rd, 0.0) * self.rng.random()
                    if all(np.linalg.norm(cand - c) >= 0.85 * (rd + rr)
                           for c, rr in placed):
                        best = cand
                        break
                    best = cand if best is None else best
                placed.append((best, rd))
                new_pos.append(best)
                new_vel.append(self.vel[k])
                new_d.append(2.0 * rd)
            new_gen.extend([int(self.generation[k]) + 1] * len(radii_d))
            self.events.append({
                "t": self.t,
                "parent_volume": v0,
                "parent_diameter": float(self.d[k]),
                "daughter_volumes": [float(v) for v in daughters],
                "daughter_fractions": [float(v / v0) for v in daughters],
                "generation": int(self.generation[k]),
                "sigma_p": float(sigma_p[k]),
                "strength": float(self.strength[k]),
            })
        keep = np.ones(self.pos.shape[0], dtype=bool)
        keep[idx] = False
        self.pos = np.vstack([self.pos[keep], np.asarray(new_pos)])
        self.vel = np.vstack([self.vel[keep], np.asarray(new_vel)])
        self.d = np.concatenate([self.d[keep], np.asarray(new_d)])
        self.r = 0.5 * self.d
        self.volume = math.pi * self.d ** 3 / 6.0
        self.mass = p.rho * p.mass_scale * self.volume
        new_volume = self.volume[self.pos.shape[0] - len(new_d):]
        self.strength = np.concatenate(
            [self.strength[keep],
             p.strength.sample_strength(new_volume, self.rng)])
        self.age = np.concatenate([self.age[keep], np.zeros(len(new_d), dtype=np.int64)])
        self.generation = np.concatenate([self.generation[keep],
                                          np.asarray(new_gen, dtype=np.int64)])
        self.broken_total += int(idx.size)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        p = self.p
        record = sorted(float(t) for t in p.record_times)
        targets = list(record)
        n_steps = p.n_steps
        for _ in range(n_steps):
            self.step()
            while targets and self.t >= targets[0]:
                self._record_psd(targets[0])
                targets.pop(0)
        while targets:
            self._record_psd(targets[0], reuse=True)
            targets.pop(0)
        mass_end = float(np.sum(self.mass))
        events = self.events
        mass_error = abs(mass_end - self.mass_ledger["mass_0"]) / self.mass_ledger["mass_0"]
        sigma_macro = np.array([row[2] for row in self.stress_history])
        kappa = float("nan")
        if self.kappa_window["sigma_eff"] and self.first_break_time is not None:
            t_win = np.array([row[0] for row in self.stress_history])
            mask = t_win <= self.first_break_time
            if np.any(mask):
                sa = np.array(self.kappa_window["sigma_eff"])[: int(np.sum(mask))]
                sm = np.array(self.kappa_window["sigma_macro"])[: int(np.sum(mask))]
                good = sm > 0.0
                if np.any(good):
                    kappa = float(np.mean(sa[good] / sm[good]))
        return {
            "generation_histogram": np.bincount(self.generation).tolist(),
            "psd": self.psd,
            "record_times": record,
            "events": events,
            "n_events": len(events),
            "n_broken_particles": int(self.broken_total),
            "n_particles_end": int(self.pos.shape[0]),
            "mass_0": self.mass_ledger["mass_0"],
            "mass_end": mass_end,
            "mass_residual": float(mass_error),
            "first_break_time": self.first_break_time,
            "kappa": kappa,
            "stress_history": self.stress_history,
            "sigma_macro_end": float(sigma_macro[-1]) if sigma_macro.size else 0.0,
            "dt": p.dt,
            "n_steps": p.n_steps,
            "total_breakage_mass_defect": float(max(
                [0.0] + [abs(sum(e["daughter_volumes"]) - e["parent_volume"]) / e["parent_volume"]
                         for e in events])),
        }

    def _record_psd(self, t: float, reuse: bool = False) -> None:
        if not reuse:
            self.psd[float(t)] = {
                "diameters": self.d.copy(),
                "volumes": self.volume.copy(),
                "weights": self.volume.copy(),
                "n": int(self.d.size),
            }
        else:
            self.psd.setdefault(float(t), {
                "diameters": self.d.copy(),
                "volumes": self.volume.copy(),
                "weights": self.volume.copy(),
                "n": int(self.d.size),
            })


def measure_kappa(sigma_particle, sigma_macro: float, m: float) -> float:
    """Effective-stress ratio ``sigma_eff/sigma_macro`` (Weibull stress integral)."""
    if sigma_macro <= 0.0:
        return float("nan")
    return float(effective_stress(sigma_particle, m) / sigma_macro)