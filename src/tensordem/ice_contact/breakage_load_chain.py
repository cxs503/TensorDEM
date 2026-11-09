"""IN27-S3 ice–ship load chain: DEM floe field + breakage/clearance + hull loads.

The chain wires :mod:`tensordem.breakage_clearance` into the ice–ship load
problem and delivers what the campaign node asks for: **force statistics**
(hull-girder vertical / horizontal bending and torsional moments, their block
extrema and GEV parameters) *and* **floe kinematics**, on the same nine-case
matrix that IN27-S2 used for the pressure-field study
(``v x h_ice = {1,3,5} m/s x {0.5,1.0,1.5} m``).

Model (reduced-order floe DEM, stated explicitly)
-------------------------------------------------
* **Ice field** — the layer is discretised into floes on a jittered lattice of
  the same solid fraction (0.55), size spread and ``d50 = 0.18 h_ice`` as the
  IN27-S2 assembly (``ice_contact.study.build_assembly``).  Floes are
  equivalent spheres in translational motion only: no rotation / spin and no
  clump interlocking are resolved, which is the main simplification of this
  chain (IN27-S2's particle DEM remains the reference for shape effects).
* **Ship** — a rigid hull panel of width ``B`` advancing at speed ``v``.  The
  hull surface at transverse position ``y`` carries the bow normal
  ``(cos b cos t, -sign(y) sin b, sin t)`` with ``b`` the bow half-angle and
  ``t`` the rise angle.  The vertical component is transmitted to the hull as
  the ride-up load (used for VBM/TBM) and is *not* applied to the floe: the
  floe slides along the inclined hull surface (unilateral-contact constraint,
  doing no work).  The in-plane part ``(cos b cos t, -sign(y) sin b, 0)`` is
  applied to the floe.
* **Contacts** — soft-sphere penalty contact with a crushing cap,

      F_n = clip(k delta + c v_n_closing, 0, sigma_c A)

  for the hull (``A`` the floe's projected area) and for floe–floe pairs
  (``A`` = min of the two projected areas).  Coulomb friction ``mu F_n``
  opposes the tangential relative velocity.  Contact impulses are applied by
  the breakage engine, so the energy ledger sees every force.
* **Operator** — ``clearance_rule="size"`` + ``enable_breakage=True`` is the
  *new* rule; ``clearance_rule="legacy_delta"`` + ``enable_breakage=False``
  reproduces the *old* ``delta > 0.25 m`` crush-clearance threshold (delete any
  floe whose hull penetration exceeds 0.25 m, no fragmentation).
* **Hull-girder loads** — with ``zeta_i = L/2 - |y_i - y_c| / sin b`` the bow
  station of floe *i* seen from midship (``L`` hull length), and ``y_c``,
  ``z_c`` the panel centre:

      VBM = sum_i F_z,i zeta_i                       (vertical bending moment)
      HBM = sum_i F_y,i zeta_i                       (horizontal bending moment)
      TBM = sum_i [(y_i-y_c) F_z,i - (z_i-z_c) F_y,i] (torsion)

  These are the standard hull-girder integrals evaluated on the DEM contact
  force field; they are reported as *block extrema* over the load record and
  fitted with a GEV law.

Everything numeric in the report is produced by this module; the only literals
are the acceptance thresholds of the campaign node.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import genextreme

from ..breakage_clearance import BreakageConfig, BreakageEngine, Floe, floe_kinematics
from .study import build_assembly

__all__ = [
    "ChainParams",
    "CASE_MATRIX",
    "ACCEPTANCE",
    "run_case",
    "run_matrix",
    "convergence_study",
    "negative_controls",
    "fit_gev_block_maxima",
    "block_extremes",
    "build_report",
    "main",
]

#: IN27-S3 operating matrix (identical to the IN27-S2 condition matrix).
CASE_MATRIX = tuple((v, h) for v in (1.0, 3.0, 5.0) for h in (0.5, 1.0, 1.5))

#: Acceptance thresholds of the campaign node (never used as fitting targets).
ACCEPTANCE = {
    "C1_mass_drift": 1.0e-10,
    "C2_energy_residual": 0.02,
    "C3_observed_order": 1.0,
    "C3_gci": 0.10,
    "C4_vbm_diff": 0.25,
    "C4_net_lateral_force_N": 50.0e6,
}


@dataclass(frozen=True)
class ChainParams:
    """Parameters of one ice–ship load-chain case (SI units)."""

    speed: float = 1.0
    thickness: float = 1.0
    method: str = "new"                 # "new" | "legacy"
    advance: float = 1.0                # hull advance over the record [m]
    dt: float = 1.0e-4
    max_steps: int = 60_000
    panel_width: float = 2.0
    ice_length_factor: float = 0.45
    solid_fraction: float = 0.55
    n_max: int = 220
    d50_ratio: float = 0.18
    seed: int = 20261109
    rho: float = 917.0
    mass_scale: float = 500.0           # quasi-static DEM inertia scaling [-]
    sigma_c: float = 8.0e6
    k_n: float = 5.0e6
    k_pp: float = 5.0e6
    pair_cap_factor: float = 1.0        # multiplies sigma_c*A for floe-floe caps
    pair_plastic: bool = False          # elastic (False) or elastic-plastic (True) floes
    zeta: float = 0.30
    mu_wall: float = 0.15
    mu_pp: float = 0.35
    bow_half_angle_deg: float = 20.0
    hull_rise_deg: float = 10.0
    gamma: float = 0.5
    h_min: float = 0.05
    h_max_break: float = 0.50
    breakage_time_s: float = 0.02
    n_children_max: int = 8
    penalty_ceiling_J: float = 1.0e6
    max_floe_count: int = 20_000
    legacy_delta_crush: float = 0.25
    n_blocks: int = 24
    transient_fraction: float = 0.2
    hull_length: float = 100.0
    record_series: bool = True

    def validate(self) -> None:
        if self.speed <= 0 or self.thickness <= 0 or self.advance <= 0:
            raise ValueError("speed, thickness and advance must be positive")
        if self.method not in ("new", "legacy"):
            raise ValueError("method must be 'new' or 'legacy'")
        if self.dt <= 0 or self.max_steps < 10:
            raise ValueError("dt must be positive and max_steps >= 10")
        if not 0.0 < self.solid_fraction < 0.85:
            raise ValueError("solid_fraction must be in (0, 0.85)")
        if self.n_blocks < 8:
            raise ValueError("n_blocks must be >= 8 for a GEV fit")
        if not 0.0 <= self.transient_fraction < 0.8:
            raise ValueError("transient_fraction must be in [0, 0.8)")

    @property
    def box(self) -> tuple[float, float, float]:
        return (max(self.ice_length_factor * self.thickness, 0.35),
                float(self.panel_width), float(self.thickness))

    @property
    def duration(self) -> float:
        return self.advance / self.speed

    def breakage_config(self) -> BreakageConfig:
        return BreakageConfig(
            sigma_c=self.sigma_c, gamma=self.gamma, h_min=self.h_min,
            h_max_break=self.h_max_break, penalty_ceiling_J=self.penalty_ceiling_J,
            max_floe_count=self.max_floe_count, dt=self.dt, seed=self.seed,
            breakage_time_s=self.breakage_time_s, n_children_max=self.n_children_max,
            clearance_rule="size" if self.method == "new" else "legacy_delta",
            legacy_delta_crush=self.legacy_delta_crush, rho=self.rho,
            enable_breakage=(self.method == "new"),
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _assemble(params: ChainParams) -> tuple[list[Floe], float]:
    volumes, positions, d50 = build_assembly(
        params.seed, params.box, n_max=params.n_max,
        solid_fraction=params.solid_fraction, d50_ratio=params.d50_ratio,
    )
    floes = []
    for i, (vol, pos) in enumerate(zip(volumes, positions)):
        d = (6.0 * float(vol) / math.pi) ** (1.0 / 3.0)
        floes.append(Floe(
            fid=i, mass=params.rho * float(vol) * params.mass_scale,
            volume=float(vol), char_size=d, position=np.array(pos, dtype=float),
            velocity=np.zeros(3),
        ))
    return floes, float(d50)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(params: ChainParams, *, path: str | None = None) -> dict:
    """Run one ice–ship load-chain case and return its statistics.

    The contact law is an elastic–plastic (crushing-capped) spring with a
    plastic offset ``delta_p`` per contact, so the energy bookkeeping is exact:
    ``delta_e = delta - delta_p`` is clipped at ``delta_c = cap/k`` and the
    plastic increment ``cap * d(delta_p)`` is the crushing dissipation.  Without
    the plastic offset the stored spring energy ``0.5 k delta^2`` would grow
    without bound inside the cap and the ledger could not close.
    """
    params.validate()
    floes, d50 = _assemble(params)
    engine = BreakageEngine(params.breakage_config(), floes)
    initial_mass = engine.alive_mass()
    initial_count = len(engine.alive_floes())

    bow = math.radians(params.bow_half_angle_deg)
    rise = math.radians(params.hull_rise_deg)
    sin_b = math.sin(bow)
    sin_t = math.sin(rise)
    cos_b = math.cos(bow)
    cos_t = math.cos(rise)
    inv_sin_b = 1.0 / sin_b
    y_c = 0.5 * params.panel_width
    z_c = 0.5 * params.thickness
    station_ref = 0.5 * params.hull_length

    dt = params.dt
    n_steps = min(int(round(params.duration / dt)), params.max_steps)
    skin = 0.02
    v_eps = 0.02

    vbm = np.zeros(n_steps)
    hbm = np.zeros(n_steps)
    tbm = np.zeros(n_steps)
    forces_x = np.zeros(n_steps)
    forces_y = np.zeros(n_steps)
    n_contacts = np.zeros(n_steps, dtype=int)
    max_pen = np.zeros(n_steps)
    spring_energy = np.zeros(n_steps)
    floe_counts = np.zeros(n_steps, dtype=int)

    plastic_hull: dict[int, float] = {}
    prev_spring = 0.0
    acc = {"hull_diss": 0.0, "pair_diss": 0.0, "hull_plastic": 0.0,
           "pair_plastic": 0.0, "external": 0.0, "hull_friction_diss": 0.0}
    plastic_pair: dict[tuple[int, int], float] = {}
    used = 0
    for step in range(n_steps):
        t = step * dt
        hull_x = params.speed * t
        alive = engine.alive_floes()
        n = len(alive)
        if n == 0:
            break
        pos = np.array([f.position for f in alive], dtype=float)
        vel = np.array([f.velocity for f in alive], dtype=float)
        mass = np.array([f.mass for f in alive], dtype=float)
        radius = 0.5 * np.array([f.char_size for f in alive], dtype=float)
        area = math.pi * radius ** 2
        fids = [f.fid for f in alive]

        wrench = np.zeros((n, 3))

        # --- hull contact (elastic-plastic spring with crushing cap) --------
        # The hull is a V-bow: the local normal is n = (cos b cos t,
        # sign(y') sin b, 0) (normalised), y' = y - y_c, and the penetration is
        # measured along n, delta = (X_hull - x) . n - r, so a hull advance dX
        # changes it by (V n_x - v . n) dt exactly.
        y_rel = pos[:, 1] - y_c
        side = np.where(y_rel >= 0.0, 1.0, -1.0)
        lat = side * sin_b
        n_raw = np.stack([np.full(n, cos_b * cos_t), lat, np.zeros(n)], axis=1)
        n_dir = n_raw / np.linalg.norm(n_raw, axis=1)[:, None]
        n_x = n_dir[:, 0]
        n_y = n_dir[:, 1]
        delta = np.maximum(0.0, (hull_x - pos[:, 0]) * n_x - y_rel * n_y - radius)
        in_contact = delta > 0.0
        cap = params.sigma_c * area
        delta_c = cap / params.k_n
        p_prev = np.array([plastic_hull.get(fid, float(delta[k]))
                           for k, fid in enumerate(fids)])
        delta_e = np.maximum(delta - p_prev, 0.0)
        over = delta_e > delta_c
        p_new = np.where(over, delta - delta_c, p_prev)
        delta_e = np.where(over, delta_c, delta_e)
        # a lost contact starts a fresh surface: the plastic memory is dropped
        p_new = np.where(in_contact, p_new, 0.0)
        plastic_hull = {fid: float(p_new[k]) for k, fid in enumerate(fids)}
        c_n = 2.0 * params.zeta * np.sqrt(params.k_n * mass)
        v_n_floe = np.sum(vel * n_dir, axis=1)
        v_closing = np.maximum(params.speed * n_x - v_n_floe, 0.0)
        f_el = np.where(in_contact, params.k_n * delta_e, 0.0)
        # the crushing cap bounds the *total* pressure: the damper can only use
        # the pressure head that the elastic part leaves free
        f_damp = np.minimum(c_n * v_closing, np.maximum(cap - f_el, 0.0))
        f_n = np.where(in_contact, f_el + f_damp, 0.0)
        wrench += f_n[:, None] * n_dir
        elastic_by_fid = {fid: float(0.5 * params.k_n * delta_e[k] ** 2)
                          for k, fid in enumerate(fids)}
        hull_fx = f_n * n_x
        hull_fy = f_n * n_y
        hull_fz = f_n * sin_t
        hull_spring = float(np.sum(0.5 * params.k_n * delta_e[in_contact] ** 2))
        hull_plastic = float(np.sum(cap[in_contact]
                                    * np.maximum(p_new[in_contact] - p_prev[in_contact], 0.0)))

        # --- floe-floe contacts (normal part) -------------------------------
        # note: the floe-floe spring/dissipation are internal and therefore
        # not ledger terms (their work already sits in the kinetic energy)
        pair_diss = 0.0
        pair_plastic = 0.0
        pi = pj = None
        pair_normal = None
        pair_fn = None
        if n > 1:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(float(2.0 * radius.max() + skin), output_type="ndarray")
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                dvec = pos[i] - pos[j]
                dist = np.linalg.norm(dvec, axis=1)
                overlap = radius[i] + radius[j] - dist
                mask = (overlap > 0.0) & (dist > 1e-12)
                if np.any(mask):
                    i, j = i[mask], j[mask]
                    dvec = dvec[mask]
                    dist = dist[mask]
                    overlap = overlap[mask]
                    n_ij = dvec / dist[:, None]
                    v_rel = vel[i] - vel[j]
                    v_n = np.sum(v_rel * n_ij, axis=1)
                    c_pp = 2.0 * params.zeta * np.sqrt(
                        params.k_pp * np.minimum(mass[i], mass[j]))
                    cap_p = params.pair_cap_factor * params.sigma_c * np.minimum(area[i], area[j])
                    dc_p = cap_p / params.k_pp
                    keys = [(min(fids[a], fids[b]), max(fids[a], fids[b]))
                            for a, b in zip(i, j)]
                    if params.pair_plastic:
                        p_prev_p = np.array([plastic_pair.get(k, float(ov))
                                             for k, ov in zip(keys, overlap)])
                        e_p = np.maximum(overlap - p_prev_p, 0.0)
                        overp = e_p > dc_p
                        p_new_p = np.where(overp, overlap - dc_p, p_prev_p)
                        e_p = np.where(overp, dc_p, e_p)
                        active_keys = set(keys)
                        for key in [k for k in plastic_pair if k not in active_keys]:
                            del plastic_pair[key]
                        for key, val in zip(keys, p_new_p):
                            plastic_pair[key] = float(val)
                        pair_plastic = float(np.sum(cap_p * np.maximum(p_new_p - p_prev_p, 0.0)))
                    else:
                        # purely elastic interior contacts: in this chain the
                        # crushing cap is a structure-interface (hull) effect
                        e_p = overlap
                        cap_p = np.full_like(overlap, np.inf)
                    f_el_p = params.k_pp * e_p
                    closing = np.maximum(-v_n, 0.0)   # n_ij points j -> i
                    f_damp_p = np.minimum(c_pp * closing, np.maximum(cap_p - f_el_p, 0.0))
                    f_pair = f_el_p + f_damp_p
                    pair_diss = float(np.sum(f_damp_p * closing)) * dt
                    f_vec = f_pair[:, None] * n_ij
                    np.add.at(wrench, i, f_vec)
                    np.add.at(wrench, j, -f_vec)
                    pi, pj, pair_normal, pair_fn = i, j, n_ij, f_pair

        # --- Coulomb friction, evaluated at the average step velocity -------
        v_avg0 = vel + 0.5 * (wrench * dt) / mass[:, None]
        vt_hull = v_avg0 - np.sum(v_avg0 * n_dir, axis=1)[:, None] * n_dir
        vt_hull_norm = np.linalg.norm(vt_hull, axis=1)
        f_fric_mag = np.minimum(params.mu_wall * f_n * np.tanh(vt_hull_norm / v_eps),
                                mass * vt_hull_norm / dt)
        f_fric = -f_fric_mag[:, None] * vt_hull / np.maximum(vt_hull_norm, 1e-12)[:, None]
        wrench += f_fric
        hull_diss = float(np.sum(f_damp[in_contact]
                                 * np.maximum(params.speed * n_x[in_contact]
                                              - np.sum(v_avg0 * n_dir, axis=1)[in_contact],
                                              0.0))) * dt
        # the friction impulse itself changes the sliding velocity inside the
        # step: the dissipated work is f * (v0 + v1)/2 * dt with the *signed*
        # velocity projection along the friction direction (v1 may reverse)
        v1 = np.maximum(vt_hull_norm - f_fric_mag * dt / mass, -vt_hull_norm)
        # Friction acts directly on the floe, so its work is already inside the
        # impulse work of the ledger.  It is therefore *reported* separately and
        # not added to the sinks (doing both would double count).
        friction_diss = float(np.sum(f_fric_mag * 0.5 * (vt_hull_norm + v1))) * dt
        v_rel_avg = params.speed * n_x - np.sum(v_avg0 * n_dir, axis=1)
        hull_external_work = float(np.sum(f_n * v_rel_avg)) * dt
        pair_fric = None
        if pi is not None:
            vt = (v_avg0[pi] - v_avg0[pj])
            vt = vt - np.sum(vt * pair_normal, axis=1)[:, None] * pair_normal
            vt_norm = np.linalg.norm(vt, axis=1)
            moving = vt_norm > 1e-9
            if np.any(moving):
                m_red = (mass[pi] * mass[pj]) / (mass[pi] + mass[pj])
                f_t = np.minimum(params.mu_pp * pair_fn[moving], m_red[moving] * vt_norm[moving] / dt)
                direction = -vt[moving] / vt_norm[moving, None]
                pair_fric = np.zeros_like(vt)
                pair_fric[moving] = f_t[:, None] * direction
                np.add.at(wrench, pi, pair_fric)
                np.add.at(wrench, pj, -pair_fric)
                v1 = np.maximum(vt_norm[moving] - f_t * dt / m_red[moving],
                                -vt_norm[moving])
                pair_diss += float(np.sum(f_t * 0.5 * (vt_norm[moving] + v1))) * dt

        # --- element work terms (need the total step impulse) --------------
        v_avg = vel + 0.5 * (wrench * dt) / mass[:, None]
        v_n_avg = np.sum(v_avg * n_dir, axis=1)
        # signed relative normal velocity: the damper only acts when closing,
        # but the element work is signed (the hull absorbs energy while the
        # spring unloads), otherwise the ledger would lose the return stroke
        v_rel_avg = params.speed * n_x - v_n_avg
        hull_diss = float(np.sum(f_damp[in_contact]
                                 * np.maximum(v_rel_avg[in_contact], 0.0))) * dt
        hull_external_work = float(np.sum(f_n * v_rel_avg)) * dt

        # --- hand the impulses to the breakage engine ----------------------
        # Floe-floe impulses go through the engine as well (so the kinetic
        # energy change sees them), but their spring/dissipation terms are NOT
        # booked as ledger sinks: for internal forces the pair work equals
        # -(storage + dissipation) exactly, so including both would double
        # count.  Only the hull-driven surface work and the hull element sinks
        # enter the ledger.
        acc["hull_diss"] += hull_diss
        acc["pair_diss"] += pair_diss
        acc["hull_plastic"] += hull_plastic
        acc["pair_plastic"] += pair_plastic
        acc["external"] += hull_external_work
        acc["hull_friction_diss"] += friction_diss
        spring_now = hull_spring
        penalty_increment = spring_now - prev_spring
        prev_spring = spring_now
        penetration = {fid: float(delta[k]) for k, fid in enumerate(fids)}
        fids_before = set(fids)
        engine.step(
            impulse_vectors=wrench * dt,
            crushing_impulse=f_n * dt,
            penalty_energy_J=penalty_increment,
            dissipation_energy_J=hull_diss + hull_plastic,
            external_work_J=hull_external_work,
            penetration=penetration,
            time_s=t,
        )

        # The hull elastic energy stored on a parent floe is released when it
        # shatters; the fragments start from a fresh (fully crushed) surface, so
        # that strain energy leaves the model and must be booked as a sink.
        removed = fids_before - set(engine.alive_fids())
        if removed:
            released_elastic = float(sum(elastic_by_fid.get(fid, 0.0) for fid in removed))
            acc["fragment_release"] = acc.get("fragment_release", 0.0) + released_elastic
            engine.add_sink_energy(released_elastic)
        else:
            released_elastic = 0.0

        # --- integrate positions (average velocity, energy-consistent) -----
        v_before = {fid: vel[k].copy() for k, fid in enumerate(fids)}
        for f in engine.alive_floes():
            v0 = v_before.get(f.fid)
            if v0 is None:                       # a breakage child
                f.position = f.position + f.velocity * dt
            else:
                f.position = f.position + 0.5 * (v0 + f.velocity) * dt

        # --- hull-girder loads ---------------------------------------------
        station = np.maximum(station_ref - np.abs(pos[:, 1] - y_c) * inv_sin_b, 0.0)
        vbm[step] = float(np.sum(hull_fz * station))
        hbm[step] = float(np.sum(hull_fy * station))
        tbm[step] = float(np.sum((pos[:, 1] - y_c) * hull_fz
                                 - (pos[:, 2] - z_c) * hull_fy))
        forces_x[step] = float(np.sum(hull_fx))
        forces_y[step] = float(np.sum(hull_fy))
        n_contacts[step] = int(np.count_nonzero(f_n))
        max_pen[step] = float(delta.max())
        spring_energy[step] = spring_now
        floe_counts[step] = len(engine.alive_floes())
        used = step + 1

    used = max(used, 1)
    t_axis = np.arange(used) * dt
    kin = floe_kinematics(engine.alive_floes())
    kin["n_floes"] = len(engine.alive_floes())
    kin["n_floes_initial"] = initial_count
    kin["d50_initial"] = d50

    start = int(params.transient_fraction * used)
    loads = {"VBM": vbm[:used], "HBM": hbm[:used], "TBM": tbm[:used]}
    extremes = {name: block_extremes(np.abs(arr[start:]), params.n_blocks)
                for name, arr in loads.items()}
    gev = {name: fit_gev_block_maxima(np.asarray(ext[0], dtype=float))
           for name, ext in extremes.items()}

    out = {
        "case": {
            "speed": params.speed, "thickness": params.thickness, "method": params.method,
            "d50": d50, "n_floes_initial": initial_count, "dt": params.dt,
            "advance": params.advance, "mass_scale": params.mass_scale,
            "steps": used, "duration_s": float(used * dt),
            "seed": params.seed, "h_min": params.h_min,
            "sigma_c": params.sigma_c, "gamma": params.gamma,
            "breakage_time_s": params.breakage_time_s,
            "n_blocks": params.n_blocks, "transient_fraction": params.transient_fraction,
        },
        "block_extremes": {name: {"values": [float(v) for v in ext[0]],
                                  "mean": float(np.mean(ext[0])),
                                  "max": float(np.max(ext[0])),
                                  "block_duration_s": float(ext[1] * dt)}
                           for name, ext in extremes.items()},
        "gev": gev,
        "loads": {
            "mean_abs_VBM": float(np.mean(np.abs(vbm[start:used]))),
            "max_abs_VBM": float(np.max(np.abs(vbm[start:used]))),
            "mean_abs_HBM": float(np.mean(np.abs(hbm[start:used]))),
            "max_abs_HBM": float(np.max(np.abs(hbm[start:used]))),
            "mean_abs_TBM": float(np.mean(np.abs(tbm[start:used]))),
            "max_abs_TBM": float(np.max(np.abs(tbm[start:used]))),
            "mean_force_x_N": float(np.mean(forces_x[start:used])),
            "mean_force_y_N": float(np.mean(forces_y[start:used])),
            "net_lateral_force_N": float(abs(np.mean(forces_y[start:used]))),
            "max_lateral_force_N": float(np.max(np.abs(forces_y[start:used]))),
            "n_contacts_mean": float(np.mean(n_contacts[start:used])),
            "max_penetration_m": float(np.max(max_pen[:used])),
            "max_spring_energy_J": float(np.max(spring_energy[:used])),
            "penalty_ceiling_J": params.penalty_ceiling_J,
        },
        "mass": {
            "initial_mass_kg": float(initial_mass),
            "final_mass_kg": float(engine.alive_mass()),
            "drift": float(engine.mass_drift()),
            "clearance_mass_fraction": float(engine.clearance_mass_fraction()),
            "removed_kg": float(initial_mass - engine.alive_mass()),
        },
        "energy": {
            "residual": float(engine.energy_residual()),
            "input_work_J": float(engine._cumulative_input_work_J),
            "surface_energy_J": float(engine._cumulative_surface_energy_J),
            "dissipation_J": float(engine._cumulative_dissipation_J),
            "penalty_energy_J": float(engine._cumulative_penalty_J),
            "clearance_energy_J": float(engine._cumulative_clearance_energy_J),
            "penalty_pollution_fraction": float(engine.penalty_pollution_fraction()),
            "breakdown": {k: float(v) for k, v in acc.items()},
        },
        "kinematics": kin,
        "counts": {
            "n_breakage_events": len(engine.breakage_events()),
            "n_clearance_events": len(engine.clearance_events()),
            "n_floes_final": len(engine.alive_floes()),
            "n_floes_initial": initial_count,
        },
        "floe_samples": ({
            "speeds": [float(np.linalg.norm(f.velocity)) for f in engine.alive_floes()],
            "sizes": [float(f.char_size) for f in engine.alive_floes()],
            "masses": [float(f.mass) for f in engine.alive_floes()],
        } if params.record_series else {}),
        "series": ({
            "t": t_axis.tolist(), "VBM": vbm[:used].tolist(),
            "HBM": hbm[:used].tolist(), "TBM": tbm[:used].tolist(),
            "floe_count": floe_counts[:used].tolist(),
        } if params.record_series else {}),
    }
    if path:
        Path(path).write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Block extremes / GEV
# ---------------------------------------------------------------------------

def block_extremes(values: np.ndarray, n_blocks: int) -> tuple[list[float], float]:
    """Split ``values`` into ``n_blocks`` equal blocks and return the maxima."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [], 0.0
    edges = np.linspace(0, arr.size, n_blocks + 1).astype(int)
    maxima = []
    for k in range(n_blocks):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        maxima.append(float(np.max(arr[lo:hi])))
    return maxima, float(arr.size) / n_blocks


def fit_gev_block_maxima(samples: np.ndarray) -> dict:
    """GEV fit (scipy ``genextreme``: shape ``c`` = -xi) of block maxima."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: dict = {"n": n, "shape_c": None, "loc": None, "scale": None, "method": None}
    if n < 5 or np.allclose(arr, arr[0]):
        out["method"] = "insufficient_variation"
        if n:
            out["loc"] = float(arr.mean())
            out["scale"] = 0.0
        return out
    spread = float(arr.std())
    try:
        c, loc, scale = genextreme.fit(arr)
        if not (math.isfinite(c) and math.isfinite(loc) and math.isfinite(scale)):
            raise ValueError("non-finite MLE")
        out.update({"shape_c": float(c), "loc": float(loc), "scale": float(scale),
                    "method": "mle"})
    except Exception:  # pragma: no cover - defensive fallback
        out.update({"shape_c": 0.0, "loc": float(arr.mean()),
                    "scale": float(spread * math.sqrt(6.0) / math.pi),
                    "method": "gumbel_moments_fallback"})
    try:
        p = 1.0 - 1.0 / max(n, 2)
        out["return_level_p90"] = float(
            genextreme.ppf(p, out["shape_c"], out["loc"], out["scale"]))
    except Exception:  # pragma: no cover
        out["return_level_p90"] = float(arr.max())
    return out


# ---------------------------------------------------------------------------
# Matrix / comparison
# ---------------------------------------------------------------------------

def run_matrix(*, speeds=(1.0, 3.0, 5.0), thicknesses=(0.5, 1.0, 1.5),
               base: ChainParams | None = None, workers: int = 1,
               record_series: bool = False) -> dict:
    """Run the nine-case matrix with both methods and compare."""
    base = base or ChainParams()
    tasks = [(v, h, m) for m in ("legacy", "new")
             for v in speeds for h in thicknesses]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_case, replace(base, speed=v, thickness=h, method=m,
                                                  record_series=record_series))
                    for v, h, m in tasks]
            cases = [f.result() for f in futs]
    else:
        cases = [run_case(replace(base, speed=v, thickness=h, method=m,
                                  record_series=record_series))
                 for v, h, m in tasks]

    new = [c for c in cases if c["case"]["method"] == "new"]
    old = [c for c in cases if c["case"]["method"] == "legacy"]
    comparison = []
    for c_new, c_old in zip(new, old):
        entry = {"speed": c_new["case"]["speed"], "thickness": c_new["case"]["thickness"]}
        for comp in ("VBM", "HBM", "TBM"):
            m_new = c_new["block_extremes"][comp]["mean"]
            m_old = c_old["block_extremes"][comp]["mean"]
            entry[f"{comp}_block_mean_new"] = m_new
            entry[f"{comp}_block_mean_old"] = m_old
            entry[f"{comp}_block_max_new"] = c_new["block_extremes"][comp]["max"]
            entry[f"{comp}_block_max_old"] = c_old["block_extremes"][comp]["max"]
            entry[f"{comp}_rel_diff"] = (abs(m_new - m_old) / m_old) if m_old > 0 else None
            entry[f"{comp}_gev_shape_new"] = c_new["gev"][comp]["shape_c"]
            entry[f"{comp}_gev_shape_old"] = c_old["gev"][comp]["shape_c"]
            entry[f"{comp}_gev_loc_new"] = c_new["gev"][comp]["loc"]
            entry[f"{comp}_gev_loc_old"] = c_old["gev"][comp]["loc"]
            entry[f"{comp}_gev_scale_new"] = c_new["gev"][comp]["scale"]
            entry[f"{comp}_gev_scale_old"] = c_old["gev"][comp]["scale"]
        entry["net_lateral_force_N_new"] = c_new["loads"]["net_lateral_force_N"]
        entry["net_lateral_force_N_old"] = c_old["loads"]["net_lateral_force_N"]
        entry["mass_drift_new"] = c_new["mass"]["drift"]
        entry["energy_residual_new"] = c_new["energy"]["residual"]
        entry["clearance_mass_fraction_new"] = c_new["mass"]["clearance_mass_fraction"]
        entry["clearance_mass_fraction_old"] = c_old["mass"]["clearance_mass_fraction"]
        entry["n_breakage_new"] = c_new["counts"]["n_breakage_events"]
        entry["n_clearance_new"] = c_new["counts"]["n_clearance_events"]
        entry["n_clearance_old"] = c_old["counts"]["n_clearance_events"]
        entry["floe_speed_mean_new"] = c_new["kinematics"]["mean_speed"]
        entry["floe_speed_var_new"] = c_new["kinematics"]["variance_speed"]
        entry["floe_speed_mean_old"] = c_old["kinematics"]["mean_speed"]
        comparison.append(entry)

    diffs = [e["VBM_rel_diff"] for e in comparison if e["VBM_rel_diff"] is not None]
    criteria = {
        "C1_mass_drift_max": max(c["mass"]["drift"] for c in new),
        "C2_energy_residual_max": max(c["energy"]["residual"] for c in new),
        "C4_vbm_rel_diff_max": max(diffs) if diffs else None,
        "C4_net_lateral_force_max_N": max(c["loads"]["net_lateral_force_N"] for c in new),
        "case_count_new": len(new), "case_count_old": len(old),
    }
    return {
        "matrix": {"speeds": list(speeds), "thicknesses": list(thicknesses),
                   "n_cases": len(new)},
        "cases_new": new, "cases_legacy": old,
        "comparison": comparison, "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Convergence (dual axis) and negative controls
# ---------------------------------------------------------------------------

def _gci(f_coarse: float, f_medium: float, f_fine: float, ratio: float = 2.0,
         safety: float = 1.25) -> dict:
    """Roache GCI with the observed order from three successive refinements."""
    out: dict = {"f_coarse": float(f_coarse), "f_medium": float(f_medium),
                 "f_fine": float(f_fine)}
    e_coarse = abs(f_medium - f_coarse)
    e_fine = abs(f_fine - f_medium)
    if e_fine <= 0.0 or e_coarse <= 0.0 or f_medium == 0.0:
        out.update({"observed_order": None, "gci": None,
                    "note": "zero successive difference"})
        return out
    r = math.log(e_coarse / e_fine) / math.log(ratio)
    out["observed_order"] = float(r)
    if r <= 0.0:
        out["gci"] = None
        out["note"] = "non-positive observed order"
        return out
    out["gci"] = float(safety * abs(e_fine / f_medium) / (ratio ** r - 1.0))
    return out


def convergence_study(*, speed: float = 3.0, thickness: float = 1.0,
                      base: ChainParams | None = None,
                      h_values=(0.10, 0.05, 0.025),
                      dt_values=(2.0e-4, 1.0e-4, 5.0e-5)) -> dict:
    """Dual-axis refinement: h_min at fixed dt, then dt at fixed h_min."""
    base = base or ChainParams()
    metrics = ("VBM_block_mean", "HBM_block_mean", "TBM_block_mean", "floe_speed_mean")

    def _metric(res: dict) -> dict:
        return {
            "VBM_block_mean": res["block_extremes"]["VBM"]["mean"],
            "HBM_block_mean": res["block_extremes"]["HBM"]["mean"],
            "TBM_block_mean": res["block_extremes"]["TBM"]["mean"],
            "floe_speed_mean": res["kinematics"]["mass_weighted_mean_speed"],
        }

    h_runs = [{"h_min": float(h), "metrics": _metric(run_case(
        replace(base, speed=speed, thickness=thickness, h_min=float(h),
                method="new", record_series=False)))}
        for h in h_values]
    dt_runs = [{"dt": float(dtv), "metrics": _metric(run_case(
        replace(base, speed=speed, thickness=thickness, dt=float(dtv),
                method="new", record_series=False)))}
        for dtv in dt_values]

    out = {"case": {"speed": speed, "thickness": thickness},
           "h_axis": {"values": [float(h) for h in h_values], "runs": h_runs},
           "dt_axis": {"values": [float(v) for v in dt_values], "runs": dt_runs},
           "metrics": {}}
    for m in metrics:
        f_h = [r["metrics"][m] for r in h_runs]
        f_dt = [r["metrics"][m] for r in dt_runs]
        out["metrics"][m] = {"h_axis": _gci(*f_h), "dt_axis": _gci(*f_dt)}
    return out


def negative_controls(*, speed: float = 3.0, thickness: float = 1.0,
                      base: ChainParams | None = None) -> dict:
    """Two negative controls of the campaign node (C5)."""
    base = base or ChainParams()
    # (a) infinite strength: no breakage, pure penalty contact
    a = run_case(replace(base, speed=speed, thickness=thickness, sigma_c=1.0e15,
                         method="new", record_series=False))
    # (b) h_min = 0: no clearance; the entity-explosion guard must fire
    guard_fired = False
    guard_message = ""
    try:
        run_case(replace(base, speed=speed, thickness=thickness, h_min=0.0,
                         method="new", record_series=False, max_floe_count=400,
                         advance=0.5, max_steps=8000))
    except RuntimeError as exc:
        guard_fired = True
        guard_message = str(exc)
    return {
        "infinite_strength": {
            "n_breakage_events": a["counts"]["n_breakage_events"],
            "n_clearance_events": a["counts"]["n_clearance_events"],
            "mass_drift": a["mass"]["drift"],
            "energy_residual": a["energy"]["residual"],
            "max_penetration_m": a["loads"]["max_penetration_m"],
            "floe_speed_mean": a["kinematics"]["mean_speed"],
            "passed": bool(a["counts"]["n_breakage_events"] == 0
                           and a["mass"]["drift"] < ACCEPTANCE["C1_mass_drift"]),
        },
        "h_min_zero_guard": {
            "guard_fired": guard_fired,
            "message": guard_message,
            "passed": bool(guard_fired),
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pass(value, threshold, mode="lt") -> dict:
    ok = value is not None and (value < threshold if mode == "lt" else value > threshold)
    return {"value": value, "threshold": threshold, "mode": mode, "passed": bool(ok)}


def build_report(*, base: ChainParams | None = None, workers: int = 1,
                 run_convergence: bool = True) -> dict:
    """Full IN27-S3 acceptance report (numbers + artifact paths)."""
    base = base or ChainParams()
    matrix = run_matrix(base=base, workers=workers, record_series=True)
    criteria = {}

    c1 = matrix["criteria"]["C1_mass_drift_max"]
    c2 = matrix["criteria"]["C2_energy_residual_max"]
    c4_vbm = matrix["criteria"]["C4_vbm_rel_diff_max"]
    c4_net = matrix["criteria"]["C4_net_lateral_force_max_N"]
    criteria["C1_mass_drift"] = _pass(c1, ACCEPTANCE["C1_mass_drift"])
    criteria["C2_energy_residual"] = _pass(c2, ACCEPTANCE["C2_energy_residual"])
    criteria["C4_vbm_block_extreme_diff"] = _pass(c4_vbm, ACCEPTANCE["C4_vbm_diff"])
    criteria["C4_net_lateral_force"] = _pass(c4_net, ACCEPTANCE["C4_net_lateral_force_N"])
    criteria["C6_end_to_end"] = {
        "value": {"cases_new": matrix["criteria"]["case_count_new"],
                  "cases_legacy": matrix["criteria"]["case_count_old"]},
        "threshold": 9, "mode": "eq_count", "passed": bool(
            matrix["criteria"]["case_count_new"] == 9
            and matrix["criteria"]["case_count_old"] == 9),
    }

    conv = convergence_study(base=base) if run_convergence else None
    if conv:
        primary = conv["metrics"]["HBM_block_mean"]
        axis_ok = []
        orders, gcis = [], []
        for axis in ("h_axis", "dt_axis"):
            res = primary[axis]
            orders.append(res.get("observed_order"))
            gcis.append(res.get("gci"))
            axis_ok.append(
                res.get("observed_order") is not None and res["observed_order"] >= ACCEPTANCE["C3_observed_order"]
                and res.get("gci") is not None and res["gci"] < ACCEPTANCE["C3_gci"])
        criteria["C3_dual_axis_convergence"] = {
            "metric": "HBM_block_mean",
            "observed_order": orders, "gci": gcis,
            "threshold_order": ACCEPTANCE["C3_observed_order"],
            "threshold_gci": ACCEPTANCE["C3_gci"],
            "passed": bool(all(axis_ok)),
            "full": conv,
        }
    else:
        criteria["C3_dual_axis_convergence"] = {"skipped": True, "passed": False}

    neg = negative_controls(base=base)
    criteria["C5_negative_controls"] = {
        **neg, "passed": bool(neg["infinite_strength"]["passed"]
                              and neg["h_min_zero_guard"]["passed"])}

    all_passed = all(c.get("passed", False) for c in criteria.values())
    return {
        "node": "P103W2C-N02", "mission_id": "8244", "repo": "TensorDEM",
        "matrix": matrix["matrix"],
        "acceptance_thresholds": ACCEPTANCE,
        "criteria": criteria,
        "cases": {"new": matrix["cases_new"], "legacy": matrix["cases_legacy"]},
        "comparison": matrix["comparison"],
        "all_passed": bool(all_passed),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IN27-S3 ice-ship load chain")
    parser.add_argument("--out", default="artifacts/in27_s3_breakage_clearance")
    parser.add_argument("--speed", type=float, nargs="*", default=None)
    parser.add_argument("--thickness", type=float, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-convergence", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ChainParams()
    if args.speed and args.thickness:
        matrix = run_matrix(speeds=tuple(args.speed), thicknesses=tuple(args.thickness),
                            base=base, workers=args.workers, record_series=True)
        (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2))
        print(json.dumps(matrix["criteria"], indent=2))
        return 0

    report = build_report(base=base, workers=args.workers,
                          run_convergence=not args.no_convergence)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v.get("passed") for k, v in report["criteria"].items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())