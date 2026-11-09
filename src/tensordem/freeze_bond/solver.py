"""Quasi-static boundary-driven solver with a closed energy ledger.

Loading is displacement controlled on a rigid platen; every load step is brought
to equilibrium with a modified-Newton iteration (the tangent stiffness uses the
secant stiffness of the damage model and the current active contact set), and the
bond damage history is advanced by an outer fixed-point loop.  Because the
problem is quasi-static and friction is Cundall-Strack (path dependent), the
energy ledger is accumulated **incrementally**:

.. math::

    W_{ext} = \\Delta U_{bond} + \\Delta U_{contact} + \\Delta U_{wall}
              + W_{diss,bond} + W_{slip}

``W_ext`` is the trapezoidal work of the platen, ``W_diss,bond`` comes from
:meth:`CohesiveBondSet.commit` (linear-softening fracture energy, exact for the
piecewise-linear law) and ``W_slip`` is the Cundall-Strack slip work at disk
contacts and walls.  The residual is reported as a percentage of ``W_ext`` and is
never hidden; see ``tests/test_freeze_bond_energy.py``.

The solver has no time integrator and no inertia: it is a *quasi-static* DEM
protocol, so the results are rate independent by construction and are not a
statement about rate effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import (
    BondedSpecimen,
    FrictionState,
    bond_separations,
    commit_friction,
    contact_pairs,
    internal_forces,
)
from .cohesive import CohesiveBondSet

__all__ = [
    "StepRecord",
    "LoadingResult",
    "tangent_matrix",
    "newton_solve",
    "equilibrate",
    "load_steps",
]


# --------------------------------------------------------------- stiffness
def tangent_matrix(
    spec: BondedSpecimen,
    positions: np.ndarray,
    info: dict[str, Any],
) -> sp.csr_matrix:
    """Secant stiffness ``S = -dF/dx`` of the current network (2n x 2n, sparse).

    The assembled blocks are the *positive definiteness* matrices ``k * (d ⊗ d)`` of
    every live spring, so a Newton step solves ``S du = +residual`` (the residual
    is the internal force).  Both conventions (Jacobi ``dF/dx`` or stiffness
    ``-dF/dx``) are equivalent as long as the sign in :func:`newton_solve`
    matches; here the stiffness form is used because every block is then a sum of
    positive semidefinite dyadic products and the matrix is easy to read.
    """
    n = spec.n_particles
    ndof = 2 * n
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def add_blocks(idx_a: np.ndarray, idx_b: np.ndarray, blocks: np.ndarray) -> None:
        """Add element-wise 2x2 blocks ``blocks[m]`` between ``idx_a[m]`` and ``idx_b[m]``."""
        na = int(idx_a.size)
        if na == 0:
            return
        blocks = np.broadcast_to(blocks, (na, 2, 2))
        for r in range(2):
            for c in range(2):
                rows.append(2 * idx_a + r)
                cols.append(2 * idx_b + c)
                vals.append(blocks[:, r, c])

    # ------------------------------------------------------------ bonds
    topo = spec.topology
    k0 = spec.bond_stiffness_per_area
    c_bond = (1.0 - info["bond_trial"]["damage"]) * k0 * topo.area
    eye = np.eye(2)
    blocks_bond = c_bond.reshape(-1, 1, 1) * eye[None, :, :]
    add_blocks(topo.i, topo.i, blocks_bond)
    add_blocks(topo.j, topo.j, blocks_bond)
    add_blocks(topo.i, topo.j, -blocks_bond)
    add_blocks(topo.j, topo.i, -blocks_bond)

    # --------------------------------------------------------- contacts
    cfg = spec.config
    pairs = info["contact_pairs"]
    if pairs.size:
        pi, pj = pairs[:, 0], pairs[:, 1]
        d_vec = positions[pj] - positions[pi]
        dist = np.linalg.norm(d_vec, axis=1)
        overlap = spec.radii[pi] + spec.radii[pj] - dist
        active = overlap > 0.0
        pi, pj, d_vec, dist = pi[active], pj[active], d_vec[active], dist[active]
        if pi.size:
            normal = d_vec / dist[:, None]
            tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
            k_n = cfg.contact_stiffness_N_m
            k_t = cfg.contact_stiffness_t_N_m
            blocks = (k_n * normal[:, :, None] * normal[:, None, :]
                      + k_t * tangent[:, :, None] * tangent[:, None, :])
            add_blocks(pi, pi, blocks)
            add_blocks(pj, pj, blocks)
            add_blocks(pi, pj, -blocks)
            add_blocks(pj, pi, -blocks)

    # ------------------------------------------------------------ walls
    for wall in spec.walls:
        pen = spec.radii - (positions - wall.point) @ wall.normal
        idx = np.nonzero(pen > 0.0)[0]
        if idx.size == 0:
            continue
        blocks = (wall.stiffness_n * wall.normal[:, None] * wall.normal[None, :]
                  + wall.stiffness_t * wall.tangent[:, None] * wall.tangent[None, :])
        blocks = np.broadcast_to(blocks, (idx.size, 2, 2))
        add_blocks(idx, idx, blocks)

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    k = sp.coo_matrix((v, (r, c)), shape=(ndof, ndof)).tocsr()
    # Tiny Levenberg regularisation: removes rigid-body singularity when a
    # particle momentarily has no live bond and no wall contact.  It is added to
    # the tangent only, never to the force balance.
    scale = float(np.max(np.abs(k.diagonal()))) if k.nnz else 1.0
    k = k + sp.identity(ndof, format="csr") * (1e-10 * max(scale, 1.0))
    return k


# ------------------------------------------------------------------ newton
def newton_solve(
    spec: BondedSpecimen,
    positions: np.ndarray,
    friction: FrictionState,
    *,
    f_scale: float,
    tol_rel: float = 1e-5,
    max_iter: int = 40,
    max_step_m: float | None = None,
    update_damage: bool = True,
) -> tuple[np.ndarray, dict[str, Any], int, float]:
    """Bring the configuration to equilibrium (``max|F| < tol_rel * f_scale``).

    The damage history variable is advanced *inside* the iteration (it is
    monotone, so the coupled fixed point cannot oscillate): after every trial
    configuration the peak effective separation is committed to ``delta_max`` and
    the forces are re-evaluated with the updated damage.  Unloading never heals,
    because ``delta_max`` only grows.
    """
    pos = positions.copy()
    info: dict[str, Any] = {}
    f_max = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        info = internal_forces(spec, pos, friction)
        if update_damage:
            trial = info["bond_trial"]
            if np.any(trial["delta_max"] > spec.cohesive.delta_max):
                spec.cohesive.delta_max = trial["delta_max"]
                info = internal_forces(spec, pos, friction)
        residual = info["forces"]
        f_max = float(np.max(np.linalg.norm(residual, axis=1)))
        if f_max < tol_rel * f_scale:
            break
        k = tangent_matrix(spec, pos, info)
        du = spla.spsolve(k, residual.reshape(-1)).reshape(-1, 2)
        step = float(np.max(np.linalg.norm(du, axis=1)))
        limit = max_step_m if max_step_m is not None else 0.25 * spec.config.radius_m
        if step > limit:
            du *= limit / step
        # backtracking: accept the Newton direction only if it reduces |F|
        alpha, best = 1.0, None
        for _ in range(4):
            candidate = pos + alpha * du
            info_try = internal_forces(spec, candidate, friction)
            if update_damage:
                trial_try = info_try["bond_trial"]
                if np.any(trial_try["delta_max"] > spec.cohesive.delta_max):
                    spec.cohesive.delta_max = trial_try["delta_max"]
                    info_try = internal_forces(spec, candidate, friction)
            f_try = float(np.max(np.linalg.norm(info_try["forces"], axis=1)))
            if best is None or f_try < best[0]:
                best = (f_try, candidate, info_try)
            if f_try < f_max:
                break
            alpha *= 0.5
        f_max, pos, info = best
        if f_max < tol_rel * f_scale:
            break
    return pos, info, it, f_max


def equilibrate(
    spec: BondedSpecimen,
    friction: FrictionState,
    *,
    f_scale: float,
    tol_rel: float = 1e-6,
    max_iter: int = 80,
) -> tuple[np.ndarray, dict[str, Any], int, float]:
    """Equilibrate the as-built specimen (wall preload) before loading."""
    return newton_solve(spec, spec.positions, friction, f_scale=f_scale,
                        tol_rel=tol_rel, max_iter=max_iter)


# ---------------------------------------------------------------- results
@dataclass
class StepRecord:
    step: int
    platen_offset_m: float
    platen_displacement_m: float
    platen_force_N: float
    load_cell_force_N: float
    lateral_force_N: float
    bond_energy_J: float
    contact_energy_J: float
    wall_energy_J: float
    friction_energy_J: float
    external_work_J: float
    bond_dissipated_J: float
    slip_dissipated_J: float
    broken_bonds: int
    mean_damage: float
    newton_iterations: int
    residual_N: float
    positions: np.ndarray = field(repr=False, default=None)
    bond_damage: np.ndarray = field(repr=False, default=None)
    contact_forces: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class LoadingResult:
    steps: list[StepRecord]
    ledger: dict[str, float]
    specimen: BondedSpecimen = field(repr=False, default=None)
    friction: FrictionState = field(repr=False, default=None)
    meta: dict[str, Any] = field(default_factory=dict)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "step": np.array([s.step for s in self.steps]),
            "platen_displacement_m": np.array([s.platen_displacement_m for s in self.steps]),
            "platen_force_N": np.array([s.platen_force_N for s in self.steps]),
            "load_cell_force_N": np.array([s.load_cell_force_N for s in self.steps]),
            "lateral_force_N": np.array([s.lateral_force_N for s in self.steps]),
            "bond_energy_J": np.array([s.bond_energy_J for s in self.steps]),
            "contact_energy_J": np.array([s.contact_energy_J for s in self.steps]),
            "wall_energy_J": np.array([s.wall_energy_J for s in self.steps]),
            "friction_energy_J": np.array([s.friction_energy_J for s in self.steps]),
            "external_work_J": np.array([s.external_work_J for s in self.steps]),
            "bond_dissipated_J": np.array([s.bond_dissipated_J for s in self.steps]),
            "slip_dissipated_J": np.array([s.slip_dissipated_J for s in self.steps]),
            "broken_bonds": np.array([s.broken_bonds for s in self.steps]),
            "mean_damage": np.array([s.mean_damage for s in self.steps]),
            "residual_N": np.array([s.residual_N for s in self.steps]),
        }


def _bond_observables(spec: BondedSpecimen, positions: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    dn, dt = bond_separations(spec, positions)
    trial = spec.cohesive.trial(dn, dt)
    return trial, spec.bond_stored_energy(trial)


def _energy_snapshot(spec: BondedSpecimen, positions: np.ndarray,
                     friction: FrictionState) -> tuple[dict[str, Any], float]:
    info = internal_forces(spec, positions, friction)
    trial = info["bond_trial"]
    u_bond = spec.bond_stored_energy(trial)
    u_total = (u_bond + info["contact_energy"] + info["wall_energy"]
               + info["wall_tangential_energy"])
    return info, u_total


def load_steps(
    spec: BondedSpecimen,
    *,
    wall_name: str = "platen",
    total_displacement_m: float,
    n_steps: int = 40,
    f_scale: float | None = None,
    tol_rel: float = 1e-5,
    max_newton: int = 40,
    max_passes: int = 3,
    record_positions: bool = True,
) -> LoadingResult:
    """Displacement-controlled quasi-static loading with the energy ledger.

    The reference state is the equilibrated as-built specimen: the cohesive
    committed state and the ledger are zeroed there, so all reported energies are
    increments from the reference.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if total_displacement_m == 0.0:
        raise ValueError("total_displacement_m must be non-zero")
    wall = spec.wall_by_name(wall_name)
    if not wall.movable:
        raise ValueError(f"wall {wall_name!r} is not movable")

    friction = FrictionState(n_nodes=spec.n_particles,
                             n_walls=len(spec.walls))
    if f_scale is None:
        sig_t, _sigma_s, sig_c = spec.cohesive.strengths()
        bond_force = max(float(np.max(sig_t)), 1.0) * float(np.mean(spec.topology.area))
        f_scale = max(bond_force, 1.0)

    # ------------------------------------------------------- reference state
    positions, info, iters, residual = equilibrate(spec, friction, f_scale=f_scale)
    spec.positions = positions.copy()
    dn0, dt0 = bond_separations(spec, positions)
    spec.cohesive.reset_reference(dn0, dt0)
    _, u_ref = _energy_snapshot(spec, positions, friction)
    wall_ref = wall.offset()
    platen_force_ref = float(info["wall_normal_forces"][wall_name])

    du = total_displacement_m / n_steps
    steps: list[StepRecord] = []
    external_work = 0.0
    slip_work_total = 0.0
    platen_force_prev = platen_force_ref
    u_prev = 0.0
    bond_diss_prev = 0.0

    for step in range(1, n_steps + 1):
        wall.point = wall.point + du * wall.normal
        pos_old = spec.positions.copy()
        # Newton + damage outer loop
        # Two-pass solve: the damage history is advanced *between* Newton passes
        # (not inside them) so that every bond follows its own monotone envelope
        # during a pass.  That keeps the incremental work accounting of the
        # committed path close to the real path (see the ledger residuals in
        # docs/freeze_bond_arching.md).
        newton_iters = 0
        f_max = np.inf
        for _pass in range(max_passes):
            pos, info, iters, f_max = newton_solve(
                spec, pos_old, friction, f_scale=f_scale, tol_rel=tol_rel,
                max_iter=max_newton, update_damage=False)
            trial, u_bond = _bond_observables(spec, pos)
            if not np.any(trial["delta_max"] > spec.cohesive.delta_max):
                break
            spec.cohesive.delta_max = trial["delta_max"]
            pos_old = pos.copy()
        trial, u_bond = _bond_observables(spec, pos)
        commit = spec.cohesive.commit(trial)
        slip = commit_friction(spec, pos, friction)
        info, u_total = _energy_snapshot(spec, pos, friction)
        external_work += 0.5 * (platen_force_prev
                                + float(info["wall_normal_forces"][wall_name])) * du
        slip_work_total += slip["slip_work_J"]
        spec.positions = pos.copy()
        u_applied = wall.offset() - wall_ref
        steps.append(StepRecord(
            step=step,
            platen_offset_m=wall.offset(),
            platen_displacement_m=u_applied,
            platen_force_N=float(info["wall_normal_forces"][wall_name]),
            load_cell_force_N=float(info["wall_normal_forces"].get("load_cell", 0.0)),
            lateral_force_N=float(info["wall_normal_forces"].get("wall_top", 0.0)
                                  + info["wall_normal_forces"].get("wall_bottom", 0.0)
                                  + info["wall_normal_forces"].get("wall_left", 0.0)
                                  + info["wall_normal_forces"].get("wall_right", 0.0)),
            bond_energy_J=float(u_bond),
            contact_energy_J=float(info["contact_energy"]),
            wall_energy_J=float(info["wall_energy"] + info["wall_tangential_energy"]),
            friction_energy_J=float(info["contact_friction_energy"]),
            external_work_J=float(external_work),
            bond_dissipated_J=float(commit["total_dissipated_J"]),
            slip_dissipated_J=float(slip_work_total),
            broken_bonds=int(commit["broken_bonds"]),
            mean_damage=float(np.mean(trial["damage"])),
            newton_iterations=int(newton_iters),
            residual_N=float(f_max),
            positions=pos.copy() if record_positions else None,
            bond_damage=trial["damage"].copy(),
            contact_forces={
                "bond_pairs": np.stack([spec.topology.i, spec.topology.j], axis=1),
                "bond_magnitudes": trial["traction"] * spec.topology.area,
                "bond_damage": trial["damage"].copy(),
                "contact_pairs": np.asarray(info["contact_edges"]).copy(),
                "contact_magnitudes": np.asarray(info["contact_magnitudes"]).copy(),
                "positions": pos.copy(),
            },
        ))
        platen_force_prev = float(info["wall_normal_forces"][wall_name])
        u_prev = u_applied
        bond_diss_prev = float(commit["total_dissipated_J"])

    last = steps[-1]
    u_change = (last.bond_energy_J + last.contact_energy_J + last.wall_energy_J) - u_ref
    dissipated = last.bond_dissipated_J + last.slip_dissipated_J
    residual_energy = last.external_work_J - u_change - dissipated
    ledger = {
        "external_work_J": float(last.external_work_J),
        "stored_energy_change_J": float(u_change),
        "bond_dissipated_J": float(last.bond_dissipated_J),
        "slip_dissipated_J": float(last.slip_dissipated_J),
        "residual_J": float(residual_energy),
        "residual_fraction": float(residual_energy / last.external_work_J)
        if last.external_work_J != 0.0 else float("nan"),
        "max_energy_residual_fraction": float(max(
            abs(s.external_work_J
                - ((s.bond_energy_J + s.contact_energy_J + s.wall_energy_J) - u_ref)
                - s.bond_dissipated_J - s.slip_dissipated_J) / abs(s.external_work_J)
            for s in steps if s.external_work_J != 0.0) or 0.0),
        "reference_stored_energy_J": float(u_ref),
        "n_steps": len(steps),
        "du_m": float(du),
    }
    return LoadingResult(steps=steps, ledger=ledger, specimen=spec, friction=friction,
                         meta={"wall_name": wall_name, "f_scale": float(f_scale),
                               "tol_rel": tol_rel})