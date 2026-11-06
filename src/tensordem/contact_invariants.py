"""Metamorphic (invariance) audit of the DEM3D contact/bond force kernel.

Motivation
----------
Every quantitative DEM result in this repository rests on one routine:
``IceDEM3D.forces``.  Its constitutive terms are verified one at a time
(Hertz law, restitution, friction cone, rolling resistance), but the *bookkeeping*
around them -- cell-list neighbour detection, bond/contact mutual exclusion,
``index_add_`` accumulation, pair-order independence -- has no dedicated audit.
Bookkeeping bugs are silent: they produce finite, plausible, wrong forces.

This module verifies the kernel by *properties that must hold regardless of the
constitutive law*, and by an **independent brute-force recomputation** of the
pair forces that shares no code path with the solver's spatial hashing.

Invariants
----------
``momentum_closure``
    For a closed system (no tool/platen contact, ``drag == 0``, no external
    loads) the sum of all internal forces vanishes to machine precision.  This
    is Newton's third law integrated over the whole particle set; it fails
    whenever an interaction is applied to only one of its two partners or an
    equal-and-opposite couple is mis-signed.

``translation``
    Adding a constant offset to every position leaves every force unchanged.
    The kernel must not depend on the absolute position; a cell-list or
    minimum-image bug typically breaks this.

``rotation``
    Rotating the whole configuration (positions *and* velocities) rotates the
    resulting forces by the same orthogonal matrix: ``F(Rx, Rv) == R F(x, v)``.
    Verifies isotropic central forces and vector-covariant damping.

``brute_force``
    The engine's force vector equals an independent ``O(N^2)`` reference that
    (i) loops over *all* particle pairs, (ii) skips pairs that still carry a live
    bond (bond and contact must not double-count), and (iii) applies the
    documented pair law.  Agreement to machine precision means the cell list
    misses no neighbour and adds none, and that the bond/contact exclusion mask
    is applied to the same set of pairs.

``determinism``
    Two identical fresh states give bitwise identical forces, and repeated
    evaluation of the same state is bitwise identical: no hidden state may leak
    between evaluations.  This invariant is what caught the tangential-history
    defect fixed in ``IceDEM3D.forces(update_history=False)`` -- before the fix,
    a second evaluation of a frictional state silently added one more
    ``v_tangent * dt`` slip increment and changed the force by ~3e-2 N here, and
    by 1.5e-4 m of trajectory after 60 monitored steps.

``load_superposition``
    External loads enter additively: ``F(loads) - F(0) == loads`` exactly.

``reciprocity``
    For the purely elastic path (zero velocities, no friction, no rolling) the
    tangent stiffness ``K = dF/dx`` computed by central finite differences is
    symmetric, i.e. the force field is the gradient of a potential energy.  An
    asymmetric ``K`` would mean the kernel is not derivable from an energy and
    cannot conserve energy in the sense the energy audits assume.

Frozen topology
---------------
Every invariant is a statement about *forces* for a given interaction set, so
the audit freezes the bond topology before comparing.  A case with
``pre_fracture=True`` first commits a fracture pass (one ``forces()`` call with
``update_fracture=True`` on the randomised state) and then replays that exact
``alive`` mask into every freshly built instance.  Without this step a
perturbed but intact lattice has *no* unbonded contacts at all -- every
overlapping pair is still bonded, the contact law is masked to zero, and the
brute-force comparison would silently test only the bond term.  The fractured
cases are what make the neighbour-search and exclusion-mask audit meaningful.

Reference (brute-force) law
---------------------------
The reference re-implements only the *documented* pair law of ``DEM3DConfig``
in its elastic limit, with velocities zero so the dashpots contribute nothing::

    delta   = 2R - |x_j - x_i|                     (overlap, > 0 means contact)
    linear  : F = k_c * delta * n_hat
    hertz   : F = (4/3) E* sqrt(R*) delta^(3/2) * n_hat

with ``E* = E / (2 (1 - nu^2))`` and ``R* = R / 2`` for equal spheres.  It is a
deliberately naive double loop so that it cannot inherit the solver's indexing.

Scope and limitations
---------------------
* The audit covers particle-particle bond/contact terms.  Tool, platen and
  external-load branches are checked only through ``load_superposition`` and
  through the requirement that a distant tool contributes exactly zero reaction.
* The brute-force reference does not reproduce the tangential (Cundall-Strack)
  spring, which is history dependent.  Cases with ``friction_coefficient > 0``
  are audited by the closure/symmetry invariants only, not by the reference.
* ``reciprocity`` is meaningful only for a hyperelastic path; it is skipped for
  cases with friction or rolling resistance, both of which are dissipative.
* Randomised states audit the kernel, not the physics: agreement here says the
  bookkeeping is right, nothing about whether the constitutive law matches ice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import torch

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)

__all__ = [
    "INVARIANTS",
    "TOLERANCES",
    "InvariantResult",
    "CaseSpec",
    "random_state",
    "frozen_state",
    "evaluate_forces",
    "brute_force_pair_forces",
    "bond_forces_reference",
    "DEFAULT_OFFSET",
    "INVARIANT_CHECKS",
    "momentum_closure",
    "translation",
    "rotation",
    "brute_force",
    "determinism",
    "load_superposition",
    "translation_invariance",
    "rotation_equivariance",
    "brute_force_agreement",
    "determinism_repeatability",
    "tangent_reciprocity",
    "run_audit",
    "summarise",
    "violations",
    "monitor_purity",
]

INVARIANTS = (
    "momentum_closure",
    "translation",
    "rotation",
    "brute_force",
    "determinism",
    "load_superposition",
    "reciprocity",
)


# --------------------------------------------------------------------------
# Case specification and state construction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseSpec:
    """One audit case: a geometry plus the contact-model switches under test.

    ``perturbation`` is the half-width of the uniform random displacement added
    to every particle, in units of the particle radius.  Values below 0.5 keep
    the lattice recognisable while creating both stretched bonds (tension) and
    genuine overlaps (contacts); values above 1.0 pack the box hard and stress
    the neighbour search.

    ``pre_fracture`` runs one fracture pass before the audit so that the frozen
    topology contains broken bonds and therefore real, unmasked contact pairs.
    """

    name: str
    contact_model: str = "linear"
    friction_coefficient: float = 0.0
    rolling_resistance_coefficient: float = 0.0
    perturbation: float = 0.6
    pre_fracture: bool = False
    fracture_fraction: float = 1.0
    breaking_strain: float = 1.0e6
    shear_breaking_strain: float = 1.0e6
    nx: int = 4
    ny: int = 4
    nz: int = 3
    tool_gap: float = 5.0
    tool_radius: float = 0.02

    @property
    def is_elastic(self) -> bool:
        """True when the constitutive path is hyperelastic (no dissipation)."""
        return (
            self.friction_coefficient == 0.0
            and self.rolling_resistance_coefficient == 0.0
        )

    def config(self) -> DEM3DConfig:
        """Return the solver configuration used by this case.

        The configuration deliberately disables every branch that would make the
        system open or non-autonomous, so that the invariants are statements
        about the particle-particle kernel:

        * ``drag=0`` -- otherwise momentum is removed at a rate proportional to
          velocity and ``momentum_closure`` cannot hold;
        * ``fix_x_edges=False``, ``fix_bottom=False`` -- supports would hold
          particles against the total force;
        * ``tool_gap=5`` with a tiny ``tool_radius`` -- the indenter sits five
          metres above the specimen, so its branch contributes exactly zero;
        * ``breaking_strain``/``shear_breaking_strain`` -- very large by default
          so a randomised state never fractures during an invariance check;
          finite only for the ``pre_fracture`` cases.
        """
        return DEM3DConfig(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            drag=0.0,
            fix_x_edges=False,
            fix_bottom=False,
            tool_gap=self.tool_gap,
            tool_radius=self.tool_radius,
            breaking_strain=self.breaking_strain,
            shear_breaking_strain=self.shear_breaking_strain,
            friction_coefficient=self.friction_coefficient,
            rolling_resistance_coefficient=self.rolling_resistance_coefficient,
            contact_model=self.contact_model,
        )

    def fresh(self) -> IceDEM3D:
        """Construct a solver instance with the frozen configuration."""
        return IceDEM3D(self.config())


def random_state(
    sim: IceDEM3D,
    seed: int,
    perturbation: float = 0.6,
    velocity_scale: float = 0.05,
) -> None:
    """Overwrite ``sim`` with a reproducible perturbed position/velocity state.

    Displacements are uniform in ``[-perturbation*R, +perturbation*R]`` around
    the lattice sites and velocities are uniform in
    ``[-velocity_scale, +velocity_scale]`` m/s.  Both use a private
    ``torch.Generator``, so the draw is independent of global RNG state and of
    test ordering.
    """
    generator = torch.Generator().manual_seed(int(seed))
    n = len(sim.positions)
    amplitude = float(perturbation) * sim.config.radius
    displacement = (torch.rand((n, 3), generator=generator, dtype=sim.dtype) - 0.5) * 2.0 * amplitude
    sim.positions = sim.initial_positions + displacement
    sim.velocities = (
        torch.rand((n, 3), generator=generator, dtype=sim.dtype) - 0.5
    ) * 2.0 * float(velocity_scale)


@dataclass(frozen=True)
class FrozenState:
    """Positions, velocities and (optionally) a frozen ``alive`` bond mask."""

    positions: torch.Tensor
    velocities: torch.Tensor
    alive: torch.Tensor | None
    broken_bonds: int


def frozen_state(case: CaseSpec, seed: int) -> FrozenState:
    """Build the reproducible state of ``case``/``seed`` and freeze its topology.

    For ``pre_fracture`` cases one fracture pass is committed on the randomised
    configuration; the resulting ``alive`` mask is returned and replayed into
    every instance used by the comparison, so both sides of an invariant see the
    same interaction set.
    """
    sim = case.fresh()
    random_state(sim, seed, case.perturbation)
    alive: torch.Tensor | None = None
    broken = 0
    if case.pre_fracture:
        if case.fracture_fraction < 1.0:
            # Mixed-topology case: start from the intact bond set and kill a
            # random fraction of the TOTAL bonds, so the frozen state carries
            # BOTH live bonds and unbonded contact pairs on the same lattice
            # (the audit's documented "natural next increment").  This is
            # guaranteed mixed regardless of how many the natural fracture pass
            # would break.  fracture_fraction=1.0 keeps the original
            # fully-fractured behaviour bit-for-bit.
            alive = sim.alive.new_ones(len(sim.pairs))
            n_kill = int(round(len(sim.pairs) * (1.0 - case.fracture_fraction)))
            gen = torch.Generator().manual_seed(int(seed) + 1000)
            perm = torch.randperm(len(sim.pairs), generator=gen)[:n_kill]
            alive[perm] = False
            broken = n_kill
        else:
            sim.forces(update_fracture=True)
            alive = sim.alive.clone()
            broken = len(sim.pairs) - int(alive.sum())
    return FrozenState(sim.positions.clone(), sim.velocities.clone(), alive, broken)


def evaluate_forces(
    case: CaseSpec,
    state: FrozenState | None = None,
    seed: int = 0,
    positions: torch.Tensor | None = None,
    velocities: torch.Tensor | None = None,
    loads: torch.Tensor | None = None,
    alive: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a fresh state for ``case`` and return the net force vector.

    Building a fresh instance for every evaluation is deliberate: it removes any
    possibility that tangential spring history, accumulated rolling torque or
    fracture bookkeeping leaks between the two sides of a comparison.  It costs
    a few milliseconds per call on the small audit lattices.
    """
    frozen = state if state is not None else frozen_state(case, seed)
    sim = case.fresh()
    sim.positions = (positions if positions is not None else frozen.positions).clone()
    sim.velocities = (velocities if velocities is not None else frozen.velocities).clone()
    mask = alive if alive is not None else frozen.alive
    if mask is not None:
        sim.alive = mask.clone()
    force, _ = sim.forces(loads, update_fracture=False)
    return force


def build_instance(case: CaseSpec, state: FrozenState) -> IceDEM3D:
    """Fresh solver instance carrying ``state`` (frozen topology applied)."""
    sim = case.fresh()
    sim.positions = state.positions.clone()
    sim.velocities = state.velocities.clone()
    if state.alive is not None:
        sim.alive = state.alive.clone()
    return sim


# --------------------------------------------------------------------------
# Independent brute-force reference
# --------------------------------------------------------------------------
def brute_force_pair_forces(sim: IceDEM3D, positions: torch.Tensor) -> torch.Tensor:
    """Return the contact force vector from an ``O(N^2)`` double loop.

    Only the elastic part of the documented pair law is reproduced, so this
    function is meaningful for states with zero velocities (dashpots idle) and
    without tangential/rolling terms.  Pairs that still carry a live bond are
    skipped: the solver accumulates the bond force for them and masks their
    contact force to zero, so the reference contributes nothing there either.
    """
    cfg = sim.config
    n = len(positions)
    force = torch.zeros((n, 3), dtype=sim.dtype, device=positions.device)
    hertz = cfg.contact_model == "hertz"
    e_star = hertz_reduced_modulus(cfg.young_modulus, cfg.poisson_ratio)
    r_eff = hertz_effective_radius(cfg.radius)
    contact_diameter = 2.0 * cfg.radius
    bonded = {
        (int(i), int(j))
        for (i, j), alive in zip(sim.pairs.tolist(), sim.alive.tolist())
        if alive
    }
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            delta = positions[j] - positions[i]
            distance = float(torch.linalg.vector_norm(delta))
            overlap = contact_diameter - distance
            if overlap <= 0.0:
                continue
            normal = delta / distance
            if hertz:
                magnitude = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * overlap ** 1.5
            else:
                magnitude = cfg.contact_stiffness * overlap
            # normal points from i to j, so the repulsive force on i is minus it.
            pair = magnitude * normal
            force[i] -= pair
            force[j] += pair
    return force


def bond_forces_reference(sim: IceDEM3D) -> torch.Tensor:
    """Closed-form bond contribution ``F_i = +k_b (l - l0) n_hat`` on the i row.

    Written out independently of the solver's accumulation so that
    :func:`brute_force_agreement` compares the *whole* force vector.  ``n_hat``
    points from i to j, so an extended bond pulls i towards j.
    """
    cfg = sim.config
    n = len(sim.positions)
    force = torch.zeros((n, 3), dtype=sim.dtype, device=sim.device)
    if len(sim.pairs) == 0:
        return force
    delta = sim.positions[sim.pairs[:, 1]] - sim.positions[sim.pairs[:, 0]]
    distance = torch.linalg.vector_norm(delta, dim=1)
    normal = delta / distance.clamp_min(torch.finfo(sim.dtype).eps)[:, None]
    extension = distance - sim.rest_lengths
    pair_force = (cfg.bond_stiffness * extension)[:, None] * normal
    pair_force = torch.where(sim.alive[:, None], pair_force, torch.zeros_like(pair_force))
    force.index_add_(0, sim.pairs[:, 0], pair_force)
    force.index_add_(0, sim.pairs[:, 1], -pair_force)
    return force


# --------------------------------------------------------------------------
# Individual invariants
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InvariantResult:
    """Outcome of one invariant for one case/seed.

    ``residual`` is a dimensionless relative defect: a norm of the discrepancy
    divided by the magnitude scale of the quantity being compared.  Thresholds
    live in :data:`TOLERANCES`; this class only measures.
    """

    invariant: str
    case: str
    seed: int
    residual: float
    detail: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "invariant": self.invariant,
            "case": self.case,
            "seed": self.seed,
            "residual": self.residual,
        }
        row.update(self.detail)
        return row


def _relative_norm(defect: torch.Tensor, scale: torch.Tensor | float) -> float:
    """``||defect||_inf / max(||scale||_inf, 1 N)`` as a Python float."""
    numerator = float(defect.abs().max()) if defect.numel() else 0.0
    denominator = (
        float(scale.abs().max()) if isinstance(scale, torch.Tensor) and scale.numel() else float(scale)
    )
    return numerator / max(denominator, 1.0)


def momentum_closure(case: CaseSpec, seed: int) -> InvariantResult:
    """Closed system: total internal force and the tool reaction must vanish."""
    state = frozen_state(case, seed)
    sim = build_instance(case, state)
    force, reaction = sim.forces(update_fracture=False)
    total = force.sum(dim=0)
    scale = float(force.abs().max())
    return InvariantResult(
        "momentum_closure",
        case.name,
        seed,
        _relative_norm(total, scale),
        {
            "sum_force_inf": float(total.abs().max()),
            "reaction_inf": float(reaction.abs().max()),
            "force_scale": scale,
            "residual_reaction": _relative_norm(reaction, scale),
            "broken_bonds": float(state.broken_bonds),
        },
    )


def translation_invariance(case: CaseSpec, seed: int, offset: Sequence[float]) -> InvariantResult:
    """A rigid translation of every particle must not change any force."""
    state = frozen_state(case, seed)
    base = evaluate_forces(case, state)
    shift = torch.tensor(list(offset), dtype=state.positions.dtype)
    moved = evaluate_forces(case, state, positions=state.positions + shift)
    defect = (moved - base).abs().max()
    return InvariantResult(
        "translation",
        case.name,
        seed,
        _relative_norm(defect, base),
        {"offset_norm": float(shift.norm()), "broken_bonds": float(state.broken_bonds)},
    )


def random_rotation(seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return a random proper rotation matrix (QR of a Gaussian matrix)."""
    generator = torch.Generator().manual_seed(int(seed) + 7919)
    matrix = torch.randn((3, 3), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if float(torch.linalg.det(q)) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def rotation_equivariance(case: CaseSpec, seed: int) -> InvariantResult:
    """``F(R x, R v) == R F(x, v)`` for a random proper rotation ``R``."""
    state = frozen_state(case, seed)
    base = evaluate_forces(case, state)
    rotation = random_rotation(seed, state.positions.dtype)
    rotated = evaluate_forces(
        case,
        state,
        positions=state.positions @ rotation.T,
        velocities=state.velocities @ rotation.T,
    )
    defect = (rotated - base @ rotation.T).abs().max()
    return InvariantResult(
        "rotation",
        case.name,
        seed,
        _relative_norm(defect, base),
        {"det": float(torch.linalg.det(rotation)), "broken_bonds": float(state.broken_bonds)},
    )


def brute_force_agreement(case: CaseSpec, seed: int) -> InvariantResult:
    """Engine forces must equal the independent ``O(N^2)`` reference.

    Velocities are zeroed: with ``v == 0`` the pair dashpot contributes nothing,
    which is the only regime the reference reproduces.  The bond term is compared
    against its own closed form, so the comparison still covers the whole force
    vector.  ``unbonded_overlaps`` in the detail is the number of contact pairs
    the reference actually evaluated -- if it is zero the case proves nothing
    about the neighbour search, which is why the fractured cases exist.
    """
    state = frozen_state(case, seed)
    zeros = torch.zeros_like(state.velocities)
    engine = evaluate_forces(case, state, velocities=zeros)
    sim = build_instance(case, state)
    sim.velocities = zeros.clone()
    reference_contact = brute_force_pair_forces(sim, sim.positions)
    reference = reference_contact + bond_forces_reference(sim)
    defect = (engine - reference).abs().max()

    # Count the contact pairs the reference really evaluated (unbonded overlap).
    bonded = {
        (int(i), int(j))
        for (i, j), alive in zip(sim.pairs.tolist(), sim.alive.tolist())
        if alive
    }
    n = len(sim.positions)
    diameter = 2.0 * sim.config.radius
    unbonded_overlaps = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            distance = float(torch.linalg.vector_norm(sim.positions[j] - sim.positions[i]))
            if distance < diameter:
                unbonded_overlaps += 1
    return InvariantResult(
        "brute_force",
        case.name,
        seed,
        _relative_norm(defect, reference),
        {
            "reference_scale": float(reference.abs().max()),
            "contact_scale": float(reference_contact.abs().max()),
            "bonded_pairs": int(sim.alive.sum()),
            "broken_bonds": float(state.broken_bonds),
            "unbonded_overlaps": float(unbonded_overlaps),
        },
    )


def determinism_repeatability(case: CaseSpec, seed: int) -> InvariantResult:
    """Same state evaluated twice is bitwise equal, fresh *and* on a reused instance.

    The reused-instance check uses ``update_history=False``: it asks whether a
    *probe* changes the state.  With the step-owner semantics
    (``update_history=True``) a second call intentionally advances the
    Cundall-Strack slip by ``v_tangent * dt``, which is an integration step, not
    a reproducibility defect.  Before ``update_history`` existed, ``diagnostics``
    went through the step-owner path and violated this invariant by ~1.7e-4
    relative.
    """
    state = frozen_state(case, seed)
    first = evaluate_forces(case, state)
    second = evaluate_forces(case, state)
    sim = build_instance(case, state)
    once, _ = sim.forces(update_fracture=False, update_history=False)
    twice, _ = sim.forces(update_fracture=False, update_history=False)
    fresh_defect = float((first - second).abs().max())
    reuse_defect = float((once - twice).abs().max())
    return InvariantResult(
        "determinism",
        case.name,
        seed,
        _relative_norm(torch.tensor(max(fresh_defect, reuse_defect), dtype=torch.float64), first),
        {"fresh_defect": fresh_defect, "reuse_defect": reuse_defect},
    )


def load_superposition(case: CaseSpec, seed: int) -> InvariantResult:
    """External loads must enter additively and exactly once."""
    state = frozen_state(case, seed)
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    loads = torch.rand(state.positions.shape, generator=generator, dtype=state.positions.dtype) - 0.5
    base = evaluate_forces(case, state)
    loaded = evaluate_forces(case, state, loads=loads)
    defect = (loaded - base - loads).abs().max()
    # Normalise by the force scale, not the load scale: adding a 0.5 N load to a
    # 467 kN contact state is exact only up to the round-off of that sum.
    return InvariantResult(
        "load_superposition",
        case.name,
        seed,
        _relative_norm(defect, loaded),
        {"load_scale": float(loads.abs().max()), "force_scale": float(loaded.abs().max())},
    )


def tangent_reciprocity(case: CaseSpec, seed: int, step_fraction: float = 1.0e-6) -> InvariantResult:
    """``K = dF/dx`` must be symmetric for the elastic path (central differences).

    ``K[a, b] = dF_a / dx_b`` with ``a, b`` running over the ``3N`` displacement
    degrees of freedom.  Symmetry ``K == K^T`` is the discrete statement that the
    force field is a gradient field.  Damping is idle because the state has zero
    velocities, and the step is ``step_fraction * R`` so the difference quotient
    stays in the linear regime of the (piecewise linear) linear model.
    """
    state = frozen_state(case, seed)
    zeros = torch.zeros_like(state.velocities)
    base_positions = state.positions
    n = len(base_positions)
    dof = 3 * n
    step = float(step_fraction) * case.config().radius
    flat = base_positions.reshape(-1)
    stiffness = torch.zeros((dof, dof), dtype=base_positions.dtype)
    for column in range(dof):
        plus = flat.clone()
        plus[column] += step
        minus = flat.clone()
        minus[column] -= step
        force_plus = evaluate_forces(
            case, state, positions=plus.reshape(n, 3), velocities=zeros
        )
        force_minus = evaluate_forces(
            case, state, positions=minus.reshape(n, 3), velocities=zeros
        )
        stiffness[:, column] = (force_plus - force_minus).reshape(-1) / (2.0 * step)
    asymmetry = float((stiffness - stiffness.T).abs().max())
    scale = float(stiffness.abs().max())
    return InvariantResult(
        "reciprocity",
        case.name,
        seed,
        asymmetry / max(scale, 1.0),
        {
            "asymmetry_abs": asymmetry,
            "stiffness_scale": scale,
            "relative_asymmetry": asymmetry / max(scale, 1e-300),
            "broken_bonds": float(state.broken_bonds),
        },
    )


# --------------------------------------------------------------------------
# Monitor purity: probing a run must not change it
# --------------------------------------------------------------------------
def monitor_purity(case: CaseSpec, seed: int, steps: int = 60) -> dict[str, float]:
    """Compare a monitored run against an unmonitored twin.

    Both runs integrate ``steps`` fixed steps from the identical randomised
    state; the second one additionally calls ``diagnostics()`` after every step,
    which is what the CLI and every plotting script do.  A probe that mutates
    the friction history makes the two trajectories drift apart, so this is the
    end-to-end counterpart of the ``determinism`` invariant.

    Returns the maximum absolute position and velocity divergence at the end of
    the run, plus the probe-level force discrepancy evaluated at the initial
    state.
    """
    def build() -> IceDEM3D:
        sim = case.fresh()
        random_state(sim, seed, case.perturbation)
        return sim

    unmonitored = build()
    monitored = build()
    for _ in range(int(steps)):
        unmonitored.step()
    for _ in range(int(steps)):
        monitored.step()
        monitored.diagnostics()

    # Probe-level check: the same state evaluated with and without history
    # update must agree when nothing has moved in between.
    probe_sim = build()
    probe_sim.velocities = torch.zeros_like(probe_sim.velocities) if case.is_elastic else probe_sim.velocities
    reference, _ = probe_sim.forces(update_fracture=False, update_history=False)
    probe, _ = probe_sim.forces(update_fracture=False, update_history=False)
    return {
        "steps": float(steps),
        "position_divergence_m": float((unmonitored.positions - monitored.positions).abs().max()),
        "velocity_divergence_m_s": float(
            (unmonitored.velocities - monitored.velocities).abs().max()
        ),
        "probe_repeat_defect_N": float((probe - reference).abs().max()),
        "broken_bonds": float(unmonitored.broken_bonds),
    }


# --------------------------------------------------------------------------
# Invariant-named entry points
# --------------------------------------------------------------------------
# ``INVARIANTS`` above lists the *physical* names that appear in
# ``InvariantResult.invariant``, ``TOLERANCES``, the summary table and the audit
# report.  Some of those natural names differ from the descriptive function
# names (``translation`` vs ``translation_invariance``, ...), so the invariant
# name alone used to be an unusable handle: dispatching on it required private
# knowledge of the mapping.  The thin wrappers below close that gap -- every
# entry of ``INVARIANTS`` except ``reciprocity`` is callable as
# ``contact_invariants.<invariant>(case, seed)`` -- and ``INVARIANT_CHECKS``
# makes the mapping explicit instead of implicit knowledge in ``run_audit``.
DEFAULT_OFFSET: tuple[float, float, float] = (1.5, -7.25, 3.5)


def translation(
    case: CaseSpec, seed: int, offset: Sequence[float] = DEFAULT_OFFSET
) -> InvariantResult:
    """Invariant ``translation``: ``F(x + c) == F(x)`` (see ``invariance``)."""
    return translation_invariance(case, seed, offset)


def rotation(case: CaseSpec, seed: int) -> InvariantResult:
    """Invariant ``rotation``: ``F(Rx, Rv) == R F(x, v)``."""
    return rotation_equivariance(case, seed)


def brute_force(case: CaseSpec, seed: int) -> InvariantResult:
    """Invariant ``brute_force``: agreement with the ``O(N^2)`` reference."""
    return brute_force_agreement(case, seed)


def determinism(case: CaseSpec, seed: int) -> InvariantResult:
    """Invariant ``determinism``: repeated evaluation is bitwise identical."""
    return determinism_repeatability(case, seed)


INVARIANT_CHECKS: dict[str, Callable[[CaseSpec, int], InvariantResult]] = {
    "momentum_closure": momentum_closure,
    "translation": translation,
    "rotation": rotation,
    "brute_force": brute_force,
    "determinism": determinism,
    "load_superposition": load_superposition,
}


# --------------------------------------------------------------------------
# Audit driver
# --------------------------------------------------------------------------
DEFAULT_CASES: tuple[CaseSpec, ...] = (
    CaseSpec("linear_intact", contact_model="linear", perturbation=0.5),
    CaseSpec("linear_fractured", contact_model="linear", perturbation=1.2, pre_fracture=True,
             breaking_strain=0.015, shear_breaking_strain=0.03),
    CaseSpec("hertz_fractured", contact_model="hertz", perturbation=1.2, pre_fracture=True,
             breaking_strain=0.015, shear_breaking_strain=0.03),
    CaseSpec("linear_friction_fractured", contact_model="linear", perturbation=1.2,
             pre_fracture=True, breaking_strain=0.015, shear_breaking_strain=0.03,
             friction_coefficient=0.4),
    CaseSpec("linear_friction_rolling_fractured", contact_model="linear", perturbation=1.2,
             pre_fracture=True, breaking_strain=0.015, shear_breaking_strain=0.03,
             friction_coefficient=0.4, rolling_resistance_coefficient=0.2),
    # Mixed bonded/unbonded topology: half the surviving bonds are killed so the
    # kernel must handle live bonds and unbonded contact pairs simultaneously.
    CaseSpec("linear_mixed", contact_model="linear", perturbation=0.6, pre_fracture=True,
             fracture_fraction=0.5, breaking_strain=0.015, shear_breaking_strain=0.03),
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3)


def run_audit(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cases: Sequence[CaseSpec] | None = None,
    offset: Sequence[float] = DEFAULT_OFFSET,
    include_reciprocity: bool = True,
) -> list[InvariantResult]:
    """Run every applicable invariant for every case/seed pair.

    Returns a flat list of :class:`InvariantResult`; :func:`summarise` produces
    the per-(invariant, case) worst case that the tests and the audit report
    threshold.  ``reciprocity`` is the expensive check (``6N`` force
    evaluations per sample) and can be disabled for quick runs; it is only
    applied to cases whose constitutive path is hyperelastic.
    """
    case_list = tuple(cases) if cases is not None else DEFAULT_CASES
    results: list[InvariantResult] = []
    for case in case_list:
        for seed in seeds:
            results.append(momentum_closure(case, seed))
            results.append(translation_invariance(case, seed, offset))
            results.append(rotation_equivariance(case, seed))
            results.append(determinism_repeatability(case, seed))
            results.append(load_superposition(case, seed))
            results.append(brute_force_agreement(case, seed))
            if include_reciprocity and case.is_elastic:
                results.append(tangent_reciprocity(case, seed))
    return results


def summarise(results: Sequence[InvariantResult]) -> list[dict[str, Any]]:
    """Aggregate results into one row per (invariant, case) with the worst residual."""
    grouped: dict[tuple[str, str], list[InvariantResult]] = {}
    for result in results:
        grouped.setdefault((result.invariant, result.case), []).append(result)
    rows: list[dict[str, Any]] = []
    for (invariant, case_name), items in sorted(grouped.items()):
        worst = max(items, key=lambda item: item.residual)
        rows.append(
            {
                "invariant": invariant,
                "case": case_name,
                "samples": len(items),
                "max_residual": worst.residual,
                "tolerance": TOLERANCES[invariant],
                "passed": worst.residual <= TOLERANCES[invariant],
                "worst_seed": worst.seed,
                "detail": worst.detail,
            }
        )
    return rows


# Thresholds used by the test module and the audit report.  Everything except
# reciprocity is a floating-point identity; 1e-12 is several orders above the
# float64 round-off of these magnitudes and many orders below any physically
# meaningful discrepancy.  Reciprocity is a finite-difference quotient and gets a
# correspondingly wider band.
TOLERANCES: dict[str, float] = {
    "momentum_closure": 1.0e-12,
    "translation": 1.0e-12,
    "rotation": 1.0e-12,
    "brute_force": 1.0e-12,
    "determinism": 0.0,
    "load_superposition": 1.0e-12,
    "reciprocity": 1.0e-8,
}


def violations(results: Sequence[InvariantResult]) -> list[InvariantResult]:
    """Return the results that exceed their invariant threshold."""
    return [r for r in results if r.residual > TOLERANCES.get(r.invariant, 0.0)]