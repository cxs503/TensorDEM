"""Mass/energy-conserving breakage–clearance operator for DEM ice–ship calculations.

Replaces the legacy ``delta > 0.25 m`` crushing-clearance threshold with a
physics-based fragmentation rule that conserves mass, keeps an explicit energy
ledger, and exposes floe kinematics as a first-class output.

Physics
-------
1. **Breakage rule** — a floe fragments when the *crushing impulse* exceeds the
   strength threshold,

       |J_i| > sigma_c * A_i * tau_b

   ``A_i`` is the projected contact area of floe *i*, ``tau_b`` the damage time
   (the time over which a sustained crushing pressure delivers the critical
   impulse).  With a crushing-capped contact law this is equivalent to "the
   contact pressure has been at the cap for a crushing time", which is why the
   criterion is not automatically satisfied (the pressure alone is clipped at
   ``sigma_c`` and can never exceed it).
   The parent is split into ``n`` children of equal mass,
   ``sum(V_child) = V_parent``.  ``n`` is the *smallest* value ``>= 2`` for which
   the released energy pays for the new surface,

       gamma * dA(n) <= E_avail,     dA(n) = pi d^2 (n^(2/3) - 1)

   with ``dA`` the increment of surface area of a size ``d`` parent and
   ``E_avail`` the step's impact work plus the stored penalty energy share.
   Children inherit the parent momentum (velocity), and the surface energy is
   deducted from their translational kinetic energy.

2. **Clearance** — a floe may only be removed when it has become smaller than
   the numerical resolution floor, ``d < h_min``.  Its mass, momentum and
   kinetic energy are recorded in the ledger as a sink, so global conservation
   is closed by construction.

   ``clearance_rule="legacy_delta"`` reproduces the historical rule
   (delete any floe whose hull penetration ``delta`` exceeds
   ``legacy_delta_crush = 0.25 m``).  It is kept *only* as a baseline for the
   old-vs-new comparison and is flagged as non-conservative.

3. **Energy ledger** — every step records the contact impulse, the surface
   energy, the contact (viscous/damping) dissipation supplied by the caller, the
   stored penalty energy (clamped at its ceiling), the clearance sinks and the
   kinetic-energy change.  The invariant is

       W_in = E_surface + E_dissipation + E_penalty + dKE + E_clearance

   and ``energy_residual()`` is the relative imbalance.  Because every term on
   the right-hand side is computed independently (the dissipation is the
   caller's damping work, not a plug), the residual is a *genuine* check, not a
   tautology.

4. **Penalty-energy ceiling** — the stored penalty energy is clamped at
   ``penalty_ceiling_J``.  The clipped excess is reported per step as a
   numerical-pollution indicator: it is energy that the contact model injected
   but cannot store, so it must be either dissipated or shown to be negligible.

5. **Negative controls** —
   (a) ``sigma_c -> inf`` disables breakage and degenerates to pure penalty
       contact; mass and momentum are still conserved.
   (b) ``h_min = 0`` (no clearance) triggers the entity-count explosion guard.

Scope
-----
Pure numpy, no torch.  The engine owns the floe population (mass, size,
position, velocity) and *applies* the contact impulses it is given; the DEM
time-stepper computes the contact law (force/penalty/dissipation) and the
position update and calls :meth:`BreakageEngine.step` once per step.  See
``tensordem.ice_contact.breakage_load_chain`` for the IN27-S3 load chain that
wires it into the 9-case ice–ship matrix.

Run: ``python -m pytest tests/test_breakage_clearance.py -q``
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BreakageConfig",
    "BreakageLedger",
    "BreakageEngine",
    "Floe",
    "BreakageEvent",
    "ClearanceEvent",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Floe:
    """A single ice floe (rigid body) tracked by the breakage engine."""

    fid: int
    mass: float
    volume: float
    char_size: float
    position: np.ndarray
    velocity: np.ndarray
    alive: bool = True

    @property
    def projected_area(self) -> float:
        """Projected contact area of the equivalent sphere [m^2]."""
        return math.pi * (0.5 * float(self.char_size)) ** 2

    @property
    def momentum(self) -> np.ndarray:
        return self.mass * self.velocity

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(self.velocity @ self.velocity)


@dataclass
class BreakageEvent:
    parent_fid: int
    child_fids: list[int]
    surface_energy_J: float
    impulse_Ns: float
    timestamp: float


@dataclass
class ClearanceEvent:
    fid: int
    mass_kg: float
    momentum_Ns: np.ndarray
    kinetic_energy_J: float
    timestamp: float


@dataclass
class BreakageLedger:
    step: int
    time_s: float
    contact_impulse_Ns: float
    input_work_J: float
    breakage_surface_energy_J: float
    dissipation_J: float
    clearance_momentum_Ns: np.ndarray
    clearance_energy_J: float
    penalty_energy_J: float
    penalty_energy_ceiling_J: float
    penalty_exceeded_J: float
    kinetic_energy_J: float
    total_mass_kg: float
    n_floes: int
    n_breakage_events: int
    n_clearance_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "time_s": float(self.time_s),
            "contact_impulse_Ns": float(self.contact_impulse_Ns),
            "input_work_J": float(self.input_work_J),
            "breakage_surface_energy_J": float(self.breakage_surface_energy_J),
            "dissipation_J": float(self.dissipation_J),
            "clearance_momentum_Ns": self.clearance_momentum_Ns.tolist(),
            "clearance_energy_J": float(self.clearance_energy_J),
            "penalty_energy_J": float(self.penalty_energy_J),
            "penalty_energy_ceiling_J": float(self.penalty_energy_ceiling_J),
            "penalty_exceeded_J": float(self.penalty_exceeded_J),
            "kinetic_energy_J": float(self.kinetic_energy_J),
            "total_mass_kg": float(self.total_mass_kg),
            "n_floes": int(self.n_floes),
            "n_breakage_events": int(self.n_breakage_events),
            "n_clearance_events": int(self.n_clearance_events),
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakageConfig:
    """Breakage / clearance parameters (SI units)."""

    sigma_c: float = 8.0e6
    gamma: float = 0.5
    h_min: float = 0.05
    h_max_break: float = 0.50
    penalty_ceiling_J: float = 1.0e6
    max_floe_count: int = 50_000
    dt: float = 1.0e-4
    seed: int = 42
    breakage_time_s: float = 0.02
    n_children_max: int = 8
    clearance_rule: str = "size"          # "size" | "legacy_delta"
    legacy_delta_crush: float = 0.25      # legacy threshold [m]
    rho: float = 917.0
    enable_breakage: bool = True          # False = legacy threshold baseline

    def validate(self) -> None:
        for name in ("sigma_c", "gamma", "h_max_break", "penalty_ceiling_J", "dt",
                     "breakage_time_s", "rho"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_floe_count < 100:
            raise ValueError("max_floe_count must be >= 100")
        if not math.isfinite(self.h_min) or self.h_min < 0.0:
            raise ValueError("h_min must be non-negative and finite")
        if self.h_min > 0 and self.h_min >= self.h_max_break:
            raise ValueError("h_min must be < h_max_break")
        if self.n_children_max < 2:
            raise ValueError("n_children_max must be >= 2")
        if self.clearance_rule not in ("size", "legacy_delta"):
            raise ValueError("clearance_rule must be 'size' or 'legacy_delta'")
        if self.clearance_rule == "legacy_delta":
            if not math.isfinite(self.legacy_delta_crush) or self.legacy_delta_crush <= 0:
                raise ValueError("legacy_delta_crush must be positive and finite")


# ---------------------------------------------------------------------------
# Breakage engine
# ---------------------------------------------------------------------------

class BreakageEngine:
    """Conservative fragmentation + clearance engine for a floe population."""

    def __init__(self, config: BreakageConfig, floes: list[Floe]) -> None:
        config.validate()
        self.config = config
        self.floes: list[Floe] = list(floes)
        self._next_fid = max((f.fid for f in self.floes), default=0) + 1
        self._ledger: list[BreakageLedger] = []
        self._breakage_events: list[BreakageEvent] = []
        self._clearance_events: list[ClearanceEvent] = []
        self._rng = np.random.default_rng(config.seed)

        self._initial_mass_kg = sum(f.mass for f in self.floes if f.alive)
        self._cleared_mass_kg = 0.0
        self._initial_ke_J = sum(f.kinetic_energy for f in self.floes if f.alive)
        self._cumulative_surface_energy_J = 0.0
        self._cumulative_penalty_J = 0.0
        self._cumulative_penalty_exceeded_J = 0.0
        self._cumulative_dissipation_J = 0.0
        self._cumulative_fragment_release_J = 0.0
        self._cumulative_input_work_J = 0.0
        self._cumulative_clearance_energy_J = 0.0
        self._cumulative_clearance_momentum = np.zeros(3)
        self._total_breakage_count = 0
        # accumulated crushing impulse per floe (damage variable).  The crushing
        # cap alone can never exceed sigma_c, so breakage needs a sustained
        # impulse history rather than a single-step pressure.
        self._crush_impulse: dict[int, float] = {}

    # -- helpers ------------------------------------------------------------
    def add_sink_energy(self, energy_J: float) -> None:
        """Book energy that left the model outside a step.

        Used for the elastic strain energy stored on a floe that shatters: the
        fragments start from a fully crushed (stress-free) surface, so the
        parent's stored contact energy does not follow the fragments.
        """
        energy = float(energy_J)
        if energy < 0.0:
            raise ValueError("sink energy must be non-negative")
        self._cumulative_dissipation_J += energy
        self._cumulative_fragment_release_J += energy

    def alive_floes(self) -> list[Floe]:
        return [f for f in self.floes if f.alive]

    def alive_fids(self) -> list[int]:
        return [f.fid for f in self.floes if f.alive]

    def alive_mass(self) -> float:
        return float(sum(f.mass for f in self.floes if f.alive))

    def mass_drift(self) -> float:
        """Relative imbalance between alive mass and (initial - recorded sinks)."""
        if self._initial_mass_kg == 0.0:
            return 0.0
        expected = self._initial_mass_kg - self._cleared_mass_kg
        return abs(self.alive_mass() - expected) / self._initial_mass_kg

    def energy_residual(self) -> float:
        """Relative imbalance of the cumulative energy ledger."""
        w_in = self._cumulative_input_work_J
        if w_in <= 0.0:
            return 0.0
        ke_now = float(sum(f.kinetic_energy for f in self.floes if f.alive))
        delta_ke = ke_now - self._initial_ke_J
        rhs = (self._cumulative_surface_energy_J + self._cumulative_dissipation_J
               + self._cumulative_penalty_J + delta_ke
               + self._cumulative_clearance_energy_J)
        return abs(w_in - rhs) / w_in

    def clearance_mass_fraction(self) -> float:
        """Fraction of the initial mass that left through the clearance sink."""
        if self._initial_mass_kg == 0.0:
            return 0.0
        return self._cleared_mass_kg / self._initial_mass_kg

    def penalty_pollution_fraction(self) -> float:
        """Fraction of the stored penalty energy that exceeded the ceiling."""
        total = self._cumulative_penalty_J + self._cumulative_penalty_exceeded_J
        if total <= 0.0:
            return 0.0
        return self._cumulative_penalty_exceeded_J / total

    def floe_velocity_stats(self) -> dict[str, float]:
        stats = floe_kinematics(self.alive_floes())
        stats["n"] = len(self.alive_floes())
        return stats

    def ledger_summary(self) -> list[dict[str, Any]]:
        return [led.to_dict() for led in self._ledger]

    def breakage_events(self) -> list[dict[str, Any]]:
        return [{"parent_fid": e.parent_fid, "child_fids": list(e.child_fids),
                 "surface_energy_J": e.surface_energy_J, "impulse_Ns": e.impulse_Ns,
                 "timestamp": e.timestamp} for e in self._breakage_events]

    def clearance_events(self) -> list[dict[str, Any]]:
        return [{"fid": e.fid, "mass_kg": e.mass_kg,
                 "momentum_Ns": e.momentum_Ns.tolist(),
                 "kinetic_energy_J": e.kinetic_energy_J,
                 "timestamp": e.timestamp} for e in self._clearance_events]

    # -- core step ----------------------------------------------------------
    def step(
        self,
        contact_impulse_Ns: float | np.ndarray | None = None,
        penalty_energy_J: float = 0.0,
        *,
        impulse_vectors: np.ndarray | None = None,
        dissipation_energy_J: float = 0.0,
        external_work_J: float = 0.0,
        crushing_impulse: np.ndarray | None = None,
        penetration: dict[int, float] | None = None,
        time_s: float | None = None,
    ) -> BreakageLedger:
        """Advance the operator by one step.

        Parameters
        ----------
        contact_impulse_Ns
            Scalar (or per-floe vector of magnitudes) contact impulse applied in
            the ``+x`` direction.  Ignored when ``impulse_vectors`` is given.
        penalty_energy_J
            Stored penalty (contact) energy produced during this step.
        impulse_vectors
            ``(n_alive, 3)`` impulse vectors aligned with :meth:`alive_floes`.
            They are *applied* here (``v += J/m``).
        dissipation_energy_J
            Viscous/damping work dissipated by the contact model this step.
        crushing_impulse
            Optional ``(n_alive,)`` magnitudes of the *structure-contact*
            impulse per floe.  Only these drive the crushing-damage criterion;
            without it the total impulse (including floe-floe collisions) is
            used, which over-predicts breakage in a dense packing.
        external_work_J
            Work done by an externally driven contact surface (e.g. the moving
            hull) that is not the floe's own velocity work.  With a driven
            plane the element input is ``F * V_n`` while the floe receives
            ``F * v_n``; the difference must enter the ledger here.
        penetration
            ``{fid: delta}`` hull penetration per floe, used by the legacy
            clearance rule.
        time_s
            Physical time (defaults to ``step * dt``).
        """
        cfg = self.config
        step = len(self._ledger)
        t = float(time_s) if time_s is not None else step * cfg.dt

        alive = self.alive_floes()
        n_alive = len(alive)
        penalty_clamped = min(float(penalty_energy_J), cfg.penalty_ceiling_J)
        penalty_excess = max(float(penalty_energy_J) - cfg.penalty_ceiling_J, 0.0)

        # --- assemble the impulse matrix ----------------------------------
        if impulse_vectors is not None:
            jac = np.asarray(impulse_vectors, dtype=float)
            if jac.shape != (n_alive, 3):
                raise ValueError(
                    f"impulse_vectors must have shape ({n_alive}, 3), got {jac.shape}")
        elif n_alive == 0:
            jac = np.zeros((0, 3))
        else:
            areas = np.array([f.projected_area for f in alive])
            if contact_impulse_Ns is None:
                magnitudes = np.zeros(n_alive)
            else:
                arr = np.asarray(contact_impulse_Ns, dtype=float)
                if arr.ndim == 0:
                    magnitudes = np.full(n_alive, float(arr))
                else:
                    if arr.shape != (n_alive,):
                        raise ValueError(
                            f"contact_impulse_Ns must be scalar or ({n_alive},), got {arr.shape}")
                    magnitudes = arr
            w = areas / areas.sum() if areas.sum() > 0 else np.full(n_alive, 1.0 / n_alive)
            jac = np.zeros((n_alive, 3))
            jac[:, 0] = magnitudes * w

        # --- apply impulses (engine owns the velocity update) -------------
        v_before = np.array([f.velocity.copy() for f in alive]) if n_alive else np.zeros((0, 3))
        for i, f in enumerate(alive):
            if f.mass > 0.0:
                f.velocity = f.velocity + jac[i] / f.mass
        v_after = np.array([f.velocity.copy() for f in alive]) if n_alive else np.zeros((0, 3))
        impulse_mag = np.linalg.norm(jac, axis=1) if n_alive else np.zeros(0)
        input_work = float(np.sum(0.5 * jac * (v_before + v_after))) if n_alive else 0.0

        # --- breakage ------------------------------------------------------
        breakage_candidates: list[tuple[Floe, float, float]] = []
        areas = np.array([f.projected_area for f in alive]) if n_alive else np.zeros(0)
        area_total = float(areas.sum()) if n_alive else 0.0
        damage_impulse = impulse_mag if crushing_impulse is None \
            else np.asarray(crushing_impulse, dtype=float)
        for i, f in enumerate(alive):
            if not cfg.enable_breakage:
                break
            if not f.alive:
                continue
            if f.char_size > cfg.h_max_break:
                continue
            if cfg.h_min > 0.0 and f.char_size < cfg.h_min * 1.05:
                continue
            area = float(areas[i])
            damage = self._crush_impulse.get(f.fid, 0.0) + float(damage_impulse[i])
            self._crush_impulse[f.fid] = damage
            impulse_crit = cfg.sigma_c * area * cfg.breakage_time_s
            if impulse_crit <= 0.0:
                continue
            if damage <= impulse_crit:
                continue
            share = area / area_total if area_total > 0 else 1.0 / max(n_alive, 1)
            e_avail = 0.5 * impulse_mag[i] * float(np.linalg.norm(v_before[i])) \
                + penalty_clamped * share
            n_children = self._child_count(f, e_avail)
            if n_children < 2:
                continue
            breakage_candidates.append((f, float(impulse_mag[i]), e_avail))
            self._crush_impulse.pop(f.fid, None)

        surface_energy_step = 0.0
        n_break = 0
        for parent, impulse_parent, e_avail in breakage_candidates:
            children, se = self._break_floe(parent, e_avail, t)
            if not children:
                continue
            n_break += 1
            self._total_breakage_count += 1
            surface_energy_step += se
            self._cumulative_surface_energy_J += se
            self._breakage_events.append(BreakageEvent(
                parent_fid=parent.fid,
                child_fids=[c.fid for c in children],
                surface_energy_J=se,
                impulse_Ns=impulse_parent,
                timestamp=t,
            ))

        # --- clearance -----------------------------------------------------
        n_clear = 0
        clearance_momentum_step = np.zeros(3)
        clearance_energy_step = 0.0
        for f in self.floes:
            if not f.alive:
                continue
            if cfg.clearance_rule == "legacy_delta":
                delta = float((penetration or {}).get(f.fid, 0.0))
                clear = delta > cfg.legacy_delta_crush
            else:
                clear = f.char_size < cfg.h_min
            if not clear:
                continue
            ke_removed = f.kinetic_energy
            clearance_momentum_step += f.momentum
            clearance_energy_step += ke_removed
            self._cleared_mass_kg += f.mass
            self._crush_impulse.pop(f.fid, None)
            f.alive = False
            n_clear += 1
            self._clearance_events.append(ClearanceEvent(
                fid=f.fid, mass_kg=f.mass, momentum_Ns=f.momentum.copy(),
                kinetic_energy_J=ke_removed, timestamp=t,
            ))
        self._cumulative_clearance_momentum += clearance_momentum_step
        self._cumulative_clearance_energy_J += clearance_energy_step

        # --- guards --------------------------------------------------------
        alive_count = len(self.alive_floes())
        if cfg.h_min <= 0.0 and self._total_breakage_count > cfg.max_floe_count // 10:
            raise RuntimeError(
                "h_min=0 (no clearance) triggered entity explosion guard after "
                f"{self._total_breakage_count} breakage events, {alive_count} floes at step {step}")
        if alive_count > cfg.max_floe_count:
            raise RuntimeError(
                f"Floe count {alive_count} exceeds guard {cfg.max_floe_count} at step {step}")

        # --- ledger --------------------------------------------------------
        self._cumulative_input_work_J += input_work + float(external_work_J)
        self._cumulative_dissipation_J += float(dissipation_energy_J)
        self._cumulative_penalty_J += penalty_clamped
        self._cumulative_penalty_exceeded_J += penalty_excess

        ke_after = float(sum(f.kinetic_energy for f in self.floes if f.alive))
        ledger = BreakageLedger(
            step=step, time_s=t,
            contact_impulse_Ns=float(impulse_mag.sum()) if n_alive else 0.0,
            input_work_J=input_work,
            breakage_surface_energy_J=surface_energy_step,
            dissipation_J=float(dissipation_energy_J),
            clearance_momentum_Ns=clearance_momentum_step,
            clearance_energy_J=clearance_energy_step,
            penalty_energy_J=penalty_clamped,
            penalty_energy_ceiling_J=cfg.penalty_ceiling_J,
            penalty_exceeded_J=penalty_excess,
            kinetic_energy_J=ke_after,
            total_mass_kg=self.alive_mass(),
            n_floes=alive_count,
            n_breakage_events=n_break,
            n_clearance_events=n_clear,
        )
        self._ledger.append(ledger)
        return ledger

    # -- fragmentation ------------------------------------------------------
    def _child_count(self, parent: Floe, e_avail: float) -> int:
        """Smallest n >= 2 whose new surface is paid by ``e_avail`` (0 = no break)."""
        cfg = self.config
        d = float(parent.char_size)
        best = 0
        for n in range(2, cfg.n_children_max + 1):
            # Children below the resolution floor are allowed: they are the
            # debris that the clearance rule removes (with their mass, momentum
            # and kinetic energy booked in the ledger).  They are *not* hidden
            # inside the fragmentation step.
            delta_area = math.pi * d * d * (n ** (2.0 / 3.0) - 1.0)
            if cfg.gamma * delta_area <= e_avail:
                best = n
                break
        return best

    def _break_floe(
        self, parent: Floe, e_avail: float, t: float
    ) -> tuple[list[Floe], float]:
        cfg = self.config
        d = float(parent.char_size)
        n = self._child_count(parent, e_avail)
        if n < 2:
            return [], 0.0

        v_parent = parent.velocity.copy()
        child_vol = parent.volume / n
        child_mass = parent.mass / n
        child_size = d * n ** (-1.0 / 3.0)
        delta_area = math.pi * d * d * (n ** (2.0 / 3.0) - 1.0)
        se = cfg.gamma * delta_area

        # surface energy is paid from the children's translational KE
        parent_ke = 0.5 * parent.mass * float(v_parent @ v_parent)
        se_paid = min(se, max(parent_ke - 1e-30, 0.0))
        speed = float(np.linalg.norm(v_parent))
        if parent_ke > 0.0 and se_paid > 0.0:
            speed *= math.sqrt(max(parent_ke - se_paid, 0.0) / parent_ke)
        direction = v_parent / float(np.linalg.norm(v_parent)) if speed > 0 else np.zeros(3)
        child_velocity = direction * speed

        # Children are placed on a Fibonacci sphere so that they touch but do
        # not interpenetrate: otherwise the newly born overlaps would be crushed
        # instantly and inject spurious crushing energy into the ledger.
        cluster_radius = 0.4 * child_size * n ** (1.0 / 3.0)
        phi = math.pi * (3.0 - math.sqrt(5.0))
        children: list[Floe] = []
        for k in range(n):
            z = 1.0 - 2.0 * (k + 0.5) / n
            rho = math.sqrt(max(1.0 - z * z, 0.0))
            unit = np.array([rho * math.cos(phi * k), rho * math.sin(phi * k), z])
            offset = cluster_radius * unit
            child = Floe(
                fid=self._next_fid,
                mass=child_mass,
                volume=child_vol,
                char_size=child_size,
                position=parent.position + offset,
                velocity=child_velocity.copy(),
            )
            children.append(child)
            self._next_fid += 1

        parent.alive = False
        self.floes.extend(children)
        return children, se_paid


# ---------------------------------------------------------------------------
# Kinematics summary
# ---------------------------------------------------------------------------

def floe_kinematics(floes: list[Floe]) -> dict[str, float]:
    """Velocity, size and momentum summary of a floe population."""
    alive = [f for f in floes if f.alive]
    if not alive:
        return {
            "mean_speed": 0.0, "variance_speed": 0.0, "mass_weighted_mean_speed": 0.0,
            "d50": 0.0, "d_min": 0.0, "d_max": 0.0, "total_mass": 0.0,
            "momentum_x": 0.0, "momentum_y": 0.0, "momentum_z": 0.0,
        }
    speeds = np.array([float(np.linalg.norm(f.velocity)) for f in alive])
    masses = np.array([f.mass for f in alive])
    sizes = np.array([f.char_size for f in alive])
    momentum = np.sum([f.momentum for f in alive], axis=0)
    return {
        "mean_speed": float(speeds.mean()),
        "variance_speed": float(speeds.var()),
        "mass_weighted_mean_speed": float(np.sum(masses * speeds) / masses.sum()),
        "d50": float(np.median(sizes)),
        "d_min": float(sizes.min()),
        "d_max": float(sizes.max()),
        "total_mass": float(masses.sum()),
        "momentum_x": float(momentum[0]),
        "momentum_y": float(momentum[1]),
        "momentum_z": float(momentum[2]),
    }