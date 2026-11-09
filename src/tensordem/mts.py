"""Multi-rate (multiple-time-step, MTS) contact integration with a decidable
time-step / subcycle contract.

Why this module exists
----------------------
``tensordem.hertz_impact_time`` answers *how long a Hertzian contact lasts*
(``t_c``); ``tensordem.contact_timestep_ledger`` answers *where the explicit
integrator stops being bounded* (Cundall's ``2 sqrt(m/k)``).  Neither answers the
engineering question that decides whether a frictional DEM run is affordable:

    given a *stiffness hierarchy* inside one assembly, which force arms must be
    re-evaluated on the fine step and which may be held on a coarse step -- and
    what is the largest coarse step that does not change the answer?

This module supplies exactly that, in two layers.

1. **The contract** -- :class:`StabilityGate` with the closed-form constructors
   :func:`hertz_gate` / :func:`hooke_gate` / :func:`friction_gate` /
   :func:`cundall_gate`.  A single-rate step is admissible only while
   ``dt / dt_c <= margin`` (default ``0.05``: at least 20 samples per contact).
   Above the margin the gate *requires* contact sub-cycling
   (``substeps = ceil(ratio / margin)``); at ``dt / dt_c >= 0.5`` the step is
   declared **inadmissible** and :meth:`StabilityGate.check` raises
   :class:`InadmissibleTimeStep` instead of returning a silently wrong
   restitution.  ``dt / dt_c = 0.5`` is the tightness negative control of the
   acceptance suite: a gate that accepts it is not a gate.

2. **The integrator** -- :class:`MTSDisks`, a 2-D rigid-disk assembly with the
   *same* contact laws as :mod:`tensordem.granular2d.engine` (linear normal
   spring + viscous dashpot, Cundall-Strack tangential spring with the
   friction-cone return map, rolling resistance with ``k_r = 0.25 k_t r_eff^2``),
   but with the force arms split by rate:

   * **fast arm** -- normal contacts of the *stiff* skeleton, evaluated on the
     fine step ``h = dt_outer / n_sub`` (``n_sub`` from the gate);
   * **slow arm** -- soft normal contacts, the complete friction/rolling state
     (``xi``, ``roll``) advanced once per coarse step with the return map applied
     *inside* the step, and the external/dissipative terms.

   The slow force is held constant over the coarse step while the stiff contacts
   are sub-cycled (symplectic Euler in each rate).  Every contact force is still
   applied as an equal-and-opposite pair, so linear momentum is conserved to
   round-off; the position-dependent part stays symplectic, so the energy error is
   bounded rather than secular.

Cost model (why this pays, and by how much)
-------------------------------------------
A single-rate solver must take ``dt <= margin * dt_crit(stiffest contact)`` for
*every* arm, so the friction and rolling arms -- the slow variables -- are paid
``n_sub`` times per coarse step.  The multi-rate solver pays them once and pays
the stiff normal arm ``n_sub`` times.  With a stiff subset covering a fraction
``f`` of the contacts the work ratio is ``n_sub / (1 + n_sub * f)``.  The module
counts contact evaluations per arm (:attr:`MTSDisks.work`) *and* the caller can
wall-clock it, so the speed-up is measured, not asserted.

Conventions (identical to the shipped 2-D engine)
-------------------------------------------------
For a disk pair ``(i, j)`` the normal ``n`` points from ``j`` to ``i``::

    f_n = max(0, k_ij d - c_n v_n),   c_n = 2 zeta sqrt(k_ij m_eff)
    xi  <- xi + dt v_t,   |xi| <= mu f_n / k_t          (cone return map)
    f_t = -k_t xi
    roll <- roll + dt (omega_i - omega_j),  |roll| <= mu_r r_eff f_n / k_r
    M_r = -k_r roll,      k_r = 0.25 k_t r_eff^2

Walls are the axis-aligned planes of a square box (normal + tangential, no
rolling).  All quantities are SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .contact_timestep_ledger import WallImpact, cundall_critical_timestep
from .hertz_impact_time import hertz_contact_time, linear_contact_time

__all__ = [
    "DEFAULT_MARGIN",
    "TIGHT_RATIO",
    "InadmissibleTimeStep",
    "StabilityGate",
    "stability_gate",
    "hertz_gate",
    "hooke_gate",
    "friction_gate",
    "cundall_gate",
    "MTSConfig",
    "MTSDisks",
    "ImpactResult",
    "impact_restitution",
    "alpha_for_restitution",
    "calibrate_alpha",
]

#: Largest admissible ``dt / dt_c`` for a *single*-rate step (20 samples/contact).
DEFAULT_MARGIN = 0.05
#: At or above this ratio the step is inadmissible, whatever the sub-cycling.
TIGHT_RATIO = 0.5


class InadmissibleTimeStep(ValueError):
    """Raised when the requested step violates the stability contract."""


# ---------------------------------------------------------------------------
# 1. The contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StabilityGate:
    """Machine-decidable verdict on one requested time step.

    ``dt_c`` is the contact time scale of the arm under consideration:

    * ``kind="hertz"``   -- Hertz contact duration ``C (5 m/(4k))^{2/5} v^{-1/5}``
      from :func:`tensordem.hertz_impact_time.hertz_contact_time` (velocity
      dependent; the scale of an impact);
    * ``kind="hooke"``   -- linear-spring half period ``pi sqrt(m/k)``;
    * ``kind="cundall"`` -- Cundall & Strack bound ``2 sqrt(m/k)`` (velocity
      independent; the scale of a quasi-static packing).
    """

    dt: float
    dt_c: float
    margin: float = DEFAULT_MARGIN
    tight_ratio: float = TIGHT_RATIO
    kind: str = "hertz"

    def __post_init__(self) -> None:
        for name in ("dt", "dt_c"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < float(self.margin) < float(self.tight_ratio)):
            raise ValueError("margin must satisfy 0 < margin < tight_ratio")
        if self.kind not in ("hertz", "hooke", "cundall"):
            raise ValueError("kind must be hertz, hooke or cundall")

    @property
    def ratio(self) -> float:
        """``dt / dt_c`` -- the dimensionless step."""
        return self.dt / self.dt_c

    @property
    def substeps(self) -> int:
        """Contact sub-steps that bring the fine step inside the margin."""
        return max(1, int(math.ceil(self.ratio / self.margin - 1e-9)))

    @property
    def fine_step(self) -> float:
        return self.dt / self.substeps

    @property
    def admissible(self) -> bool:
        """``False`` when no amount of sub-cycling can rescue the coarse step.

        The coarse step still sets how the *slow* arms (friction, rolling,
        external) are resolved, so a ratio at or above ``tight_ratio`` is rejected
        even when the contact is sub-cycled: that is the difference between
        "sub-cycle the stiff contact" and "take a step the friction arm cannot
        take".
        """
        return self.ratio < self.tight_ratio

    @property
    def status(self) -> str:
        if not self.admissible:
            return "fail"
        # tolerance: a fine step derived as ``coarse / substeps`` sits exactly on
        # the margin, and round-off must not report it as needing sub-cycling
        return "ok" if self.ratio <= self.margin * (1.0 + 1e-9) else "subcycle"

    def check(self, *, allow_inadmissible: bool = False) -> "StabilityGate":
        """Raise unless the step is admissible (``allow_inadmissible``: controls)."""
        if not self.admissible and not allow_inadmissible:
            raise InadmissibleTimeStep(
                f"dt/dt_c = {self.ratio:.4g} >= {self.tight_ratio} ({self.kind}): "
                f"inadmissible step (margin {self.margin}); sub-cycle the contact "
                f"or reduce dt"
            )
        return self

    def as_dict(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "dt_c": self.dt_c,
            "kind": self.kind,
            "ratio": self.ratio,
            "margin": self.margin,
            "tight_ratio": self.tight_ratio,
            "substeps": self.substeps,
            "fine_step": self.fine_step,
            "status": self.status,
            "admissible": self.admissible,
        }


def stability_gate(dt: float, dt_c: float, *, margin: float = DEFAULT_MARGIN,
                   tight_ratio: float = TIGHT_RATIO,
                   kind: str = "cundall") -> StabilityGate:
    """Generic entry point -- see :class:`StabilityGate`."""
    return StabilityGate(dt=float(dt), dt_c=float(dt_c), margin=float(margin),
                         tight_ratio=float(tight_ratio), kind=kind)


def hertz_gate(dt: float, *, m_eff: float, k_hertz: float, v_impact: float,
               margin: float = DEFAULT_MARGIN,
               tight_ratio: float = TIGHT_RATIO) -> StabilityGate:
    """Gate for an impact: ``dt_c`` is the Johnson Hertz contact duration."""
    return stability_gate(dt, hertz_contact_time(v_impact, m_eff, k_hertz),
                          margin=margin, tight_ratio=tight_ratio, kind="hertz")


def hooke_gate(dt: float, *, m_eff: float, stiffness: float,
               margin: float = DEFAULT_MARGIN,
               tight_ratio: float = TIGHT_RATIO) -> StabilityGate:
    """Gate for a linear unilateral contact: ``dt_c = pi sqrt(m_eff/k)``."""
    return stability_gate(dt, linear_contact_time(m_eff, stiffness),
                          margin=margin, tight_ratio=tight_ratio, kind="hooke")


def friction_gate(dt: float, *, m_eff: float, k_t: float, margin: float = 0.1,
                  tight_ratio: float = TIGHT_RATIO) -> StabilityGate:
    """Gate for the tangential (friction) arm: ``dt_c = pi sqrt(m_eff/k_t)``."""
    return stability_gate(dt, linear_contact_time(m_eff, k_t), margin=margin,
                          tight_ratio=tight_ratio, kind="hooke")


def cundall_gate(dt: float, *, m_min: float, k_max: float, margin: float = 0.1,
                 tight_ratio: float = TIGHT_RATIO) -> StabilityGate:
    """Gate for a quasi-static assembly: ``dt_c = 2 sqrt(m_min/k_max)``."""
    return stability_gate(dt, cundall_critical_timestep(m_min, k_max),
                          margin=margin, tight_ratio=tight_ratio, kind="cundall")


# ---------------------------------------------------------------------------
# 2. Single-pair impact under the contract
# ---------------------------------------------------------------------------
def alpha_for_restitution(*, e_target: float, v_impact: float, mass: float,
                          stiffness: float) -> float:
    """Hunt-Crossley damping coefficient that gives ``e_target`` to first order.

    ``1 - e^2`` is linear in ``alpha`` (quasi-static dissipation balance), so one
    probe at ``alpha = 1`` fixes the slope and the coefficient follows in closed
    form.  The ``O(alpha^2)`` correction is why the acceptance suite scores the
    ladder against a *converged* numerical reference rather than against this
    first-order value.
    """
    if not (0.0 < e_target < 1.0):
        raise ValueError("e_target must lie in (0, 1)")
    alpha_probe = 1.0e-4
    probe = _hunt_crossley_first_order(v_impact, mass, stiffness, alpha_probe)
    loss = 1.0 - probe ** 2
    if not math.isfinite(loss) or loss <= 0.0:
        raise ValueError("probe impact dissipates no energy: check the parameters")
    return float(alpha_probe * (1.0 - e_target ** 2) / loss)


def calibrate_alpha(*, e_target: float, v_impact: float, mass: float,
                    stiffness: float, dt_ratio: float = 5.0e-4,
                    tol: float = 1.0e-5, max_iter: int = 60) -> dict[str, float]:
    """Bisect the Hunt-Crossley coefficient against the *converged* impact solve.

    The first-order energy balance (:func:`alpha_for_restitution`) ignores the
    distortion of the trajectory and is therefore low by tens of percent at
    ``e = 0.5``.  This routine instead inverts the measured ``e(alpha)`` of the
    fine-step impact, so the nominal restitution really is the restitution.
    """
    low = 0.0
    high = alpha_for_restitution(e_target=e_target, v_impact=v_impact, mass=mass,
                                 stiffness=stiffness) * 8.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        result = impact_restitution(mass=mass, stiffness=stiffness, v_impact=v_impact,
                                    alpha=mid, dt_ratio=dt_ratio)
        if result.e_measured > e_target:
            low = mid
        else:
            high = mid
        if high - low <= tol * max(mid, 1.0e-30):
            break
    alpha = 0.5 * (low + high)
    check = impact_restitution(mass=mass, stiffness=stiffness, v_impact=v_impact,
                               alpha=alpha, dt_ratio=dt_ratio)
    return {"alpha": float(alpha), "e_measured": float(check.e_measured),
            "e_target": float(e_target), "dt_ratio": float(dt_ratio),
            "converged_error": float(abs(check.e_measured / e_target - 1.0))}


@dataclass
class ImpactResult:
    """Outcome of one contract-controlled impact."""

    e_target: float
    e_measured: float
    dt_ratio_requested: float
    gate: dict[str, object]
    substeps: int
    steps: int
    energy_gain: float
    contact_time_measured: float | None
    alpha: float = 0.0
    e_semi_analytic: float = float("nan")

    @property
    def restitution_error(self) -> float:
        """``|e_measured / e_target - 1|`` (NaN when no target was supplied)."""
        if not (self.e_target and math.isfinite(self.e_target) and self.e_target > 0.0):
            return float("nan")
        return abs(self.e_measured / self.e_target - 1.0)

    def restitution_error_vs(self, reference: float) -> float:
        """``|e_measured / reference - 1|`` against an independent reference."""
        reference = float(reference)
        if not math.isfinite(reference) or reference <= 0.0:
            return float("nan")
        return abs(self.e_measured / reference - 1.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "e_target": self.e_target,
            "e_measured": self.e_measured,
            "e_semi_analytic": self.e_semi_analytic,
            "alpha": self.alpha,
            "restitution_error": self.restitution_error,
            "dt_ratio_requested": self.dt_ratio_requested,
            "substeps": self.substeps,
            "steps": self.steps,
            "energy_gain": self.energy_gain,
            "contact_time_measured": self.contact_time_measured,
            "gate": self.gate,
        }


def impact_restitution(*, mass: float, stiffness: float, v_impact: float,
                       alpha: float | None = None, e_target: float | None = None,
                       dt_ratio: float, margin: float = DEFAULT_MARGIN,
                       subcycle: bool = True, allow_inadmissible: bool = False,
                       max_time_factor: float = 40.0) -> ImpactResult:
    """Hertzian impact of one sphere on a wall, stepped by the contract.

    The contact kernel is :class:`tensordem.contact_timestep_ledger.WallImpact`
    (velocity-Verlet + Hunt-Crossley dashpot), i.e. the shipped single-contact
    rig; what is new here is only *how the step is chosen*.  ``dt_ratio`` is the
    requested ``dt / t_c``.  With ``subcycle=True`` the gate turns it into
    ``n_sub`` contact sub-steps of ``dt / n_sub``; with ``subcycle=False`` the
    coarse step is used as is -- that is the tightness negative control.
    """
    if dt_ratio <= 0.0:
        raise ValueError("dt_ratio must be positive")
    if alpha is None:
        if e_target is None:
            raise ValueError("give alpha or e_target")
        alpha = alpha_for_restitution(e_target=e_target, v_impact=v_impact,
                                      mass=mass, stiffness=stiffness)
    tc = hertz_contact_time(v_impact, mass, stiffness)
    gate = hertz_gate(dt_ratio * tc, m_eff=mass, k_hertz=stiffness,
                      v_impact=v_impact, margin=margin)
    if subcycle:
        gate.check(allow_inadmissible=allow_inadmissible)
        n_sub = gate.substeps
    else:
        n_sub = 1
    dt_outer = gate.dt
    dt_fine = dt_outer / n_sub

    impact = WallImpact(mass=mass, stiffness=stiffness, law="hertz", damping=alpha)
    impact.rate = float(v_impact)
    energy0 = impact.energy()

    t, steps = 0.0, 0
    t_entry = t_exit = None
    in_contact = False
    gain = 1.0
    max_time = max_time_factor * tc
    while t < max_time:
        if not in_contact:
            overlap_old = impact.overlap
            rate_old = impact.rate
            impact.step(dt_outer)
            steps += 1
            t += dt_outer
            if overlap_old <= 0.0 < impact.overlap:
                # contact-event location: a coarse step would enter the contact
                # up to dt_outer late (an O(dt) phase error, not an integrator
                # error).  Roll back and locate the entry with the fine step,
                # which is the standard treatment of an impact event in DEM.
                impact.overlap, impact.rate = overlap_old, rate_old
                t -= dt_outer
                for _ in range(n_sub):
                    impact.step(dt_fine)
                    steps += 1
                    t += dt_fine
                    if impact.overlap > 0.0:
                        break
                in_contact = True
                t_entry = t
        else:
            overlap_old = impact.overlap
            for _ in range(n_sub):
                impact.step(dt_fine)
                steps += 1
                t += dt_fine
            if impact.overlap <= 0.0 < overlap_old:
                t_exit = t
                break
            if energy0 > 0.0:
                gain = max(gain, impact.energy() / energy0)
    e_measured = abs(impact.rate) / float(v_impact)
    contact_time = None if (t_entry is None or t_exit is None) else (t_exit - t_entry)
    return ImpactResult(
        e_target=float(e_target) if e_target is not None else _hunt_crossley_first_order(
            v_impact, mass, stiffness, alpha),
        e_measured=float(e_measured),
        dt_ratio_requested=float(dt_ratio),
        gate=gate.as_dict(),
        substeps=int(n_sub),
        steps=int(steps),
        energy_gain=float(gain),
        contact_time_measured=None if contact_time is None else float(contact_time),
        alpha=float(alpha),
        e_semi_analytic=_hunt_crossley_first_order(v_impact, mass, stiffness, alpha),
    )


def _hunt_crossley_first_order(v_in: float, m_eff: float, k: float,
                               alpha: float) -> float:
    """First-order Hunt-Crossley restitution (independent of the integrator)."""
    from .contact_timestep_ledger import hunt_crossley_semi_analytic_restitution
    return hunt_crossley_semi_analytic_restitution(v_in, m_eff, k, alpha)

# ---------------------------------------------------------------------------
# 3. The multi-rate disk assembly
# ---------------------------------------------------------------------------
def _scatter(force: np.ndarray, torque: np.ndarray, i: np.ndarray, j: np.ndarray,
             total: np.ndarray, torque_i: np.ndarray, torque_j: np.ndarray) -> None:
    """Accumulate pairwise forces/torques with ``bincount`` (no ``np.add.at``)."""
    n = len(force)
    fi = np.bincount(i, weights=total[:, 0], minlength=n)
    fj = np.bincount(j, weights=total[:, 0], minlength=n)
    force[:, 0] += fi - fj
    fi = np.bincount(i, weights=total[:, 1], minlength=n)
    fj = np.bincount(j, weights=total[:, 1], minlength=n)
    force[:, 1] += fi - fj
    torque += np.bincount(i, weights=torque_i, minlength=n)
    torque += np.bincount(j, weights=torque_j, minlength=n)


def _perp(vec: np.ndarray) -> np.ndarray:
    return np.stack([-vec[:, 1], vec[:, 0]], axis=1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


@dataclass
class MTSConfig:
    """Contact and bulk parameters (SI); defaults follow the shipped engine."""

    kn: float = 3.0e4
    kt_ratio: float = 0.5
    zeta: float = 0.4
    mu: float = 0.5
    mu_rolling: float = 0.05
    density: float = 917.0
    thickness: float = 0.03
    damping_time: float = 0.0
    drag_time: float = 0.0
    shear_rate: float = 0.0
    gravity: tuple[float, float] = (0.0, 0.0)
    skin: float = 0.003
    stiff_ratio: float = 1.0
    stiff_fraction: float = 0.0
    stiff_seed: int = 20260809
    outer_margin: float = 0.1
    fast_margin: float = 0.1
    #: Ablation switch: hold the *soft* contact force on the coarse step and
    #: sub-cycle only the stiff skeleton (classic RESPA).  Faster, but the two
    #: contact classes do not separate in frequency inside the same pair, so the
    #: energy ledger breaks (see docs/P82W2_N09_mts_contact_integration.md).
    split_slow_contacts: bool = False
    #: Ablation switch: keep the tangential/rolling state on the *coarse* step
    #: (cheaper, but the slip budget per step is over-estimated: measurable
    #: ``phi_J`` bias, see the jamming ablation in the campaign output).
    state_at_coarse: bool = False

    def __post_init__(self) -> None:
        for name in ("kn", "kt_ratio", "density", "thickness", "stiff_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("zeta", "mu", "mu_rolling", "damping_time", "drag_time", "skin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.stiff_ratio) < 1.0:
            raise ValueError("stiff_ratio must be >= 1")
        if not (0.0 <= float(self.stiff_fraction) < 1.0):
            raise ValueError("stiff_fraction must lie in [0, 1)")


class MTSDisks:
    """2-D frictional disks integrated by a multiple-time-step scheme.

    Two rate classes share one assembly:

    * the **contact rate** (fast) resolves the normal springs of the *active*
      contact set at ``h = dt_outer / n_sub``;
    * the **internal-variable rate** (slow) advances the tangential/rolling
      internal variables (``xi``, ``roll``) once per ``dt_outer`` with the cone
      return map, rebuilds the candidate-pair geometry and renews the active set.

    The friction-cone projection is still applied in *every* sub-step (the force
    magnitude clip ``|f_t| <= mu f_n`` uses the sub-step ``f_n``); what is coarse
    is only the increment of ``xi``.  This is the split that makes the stiff
    skeleton affordable without freezing any elastic force.
    """

    def __init__(self, positions, radii, config: MTSConfig | None = None, *,
                 box: tuple[float, float] | None = None,
                 periodic: bool = False) -> None:
        cfg = config or MTSConfig()
        self.config = cfg
        pos = np.array(positions, dtype=np.float64)
        rad = np.array(radii, dtype=np.float64).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] != 2 or len(pos) != len(rad):
            raise ValueError("positions must be (N, 2) and match radii")
        if (rad <= 0.0).any() or not np.isfinite(pos).all():
            raise ValueError("radii must be positive and positions finite")
        self.pos = pos
        self.radii = rad
        self.vel = np.zeros_like(pos)
        self.theta = np.zeros(len(rad))
        self.omega = np.zeros(len(rad))
        self.mass = cfg.density * math.pi * rad ** 2 * cfg.thickness
        self.inertia = 0.5 * self.mass * rad ** 2
        self.periodic = bool(periodic)
        if box is None and self.periodic:
            raise ValueError("a periodic cell needs its box size")
        self.box = None if box is None else (float(box[0]), float(box[1]))
        if self.box is not None and min(self.box) <= 0.0:
            raise ValueError("box sides must be positive")

        n = len(rad)
        n_stiff = int(round(cfg.stiff_fraction * n))
        stiff = np.zeros(n, dtype=bool)
        if n_stiff > 0:
            rng = np.random.default_rng(cfg.stiff_seed)
            stiff[rng.permutation(n)[:n_stiff]] = True
        self.stiff = stiff

        self.time = 0.0
        self.steps = 0
        self.work = {"slow": 0, "fast": 0, "single": 0}
        self.dt = 0.0
        self.n_sub = 1
        self.mode = "mts"
        self.gate: dict[str, object] | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None
        self._cache_pos: np.ndarray | None = None
        self._xi = np.zeros((0, 2))
        self._roll = np.zeros(0)
        self._wxi = np.zeros((4, n, 2))
        self._wroll = np.zeros((4, n))
        self._active = np.zeros(0, dtype=bool)
        self._pair_stiff = np.zeros(0, dtype=bool)
        self._slow_idx = np.zeros(0, dtype=np.int64)
        self._fast_idx = np.zeros(0, dtype=np.int64)
        self._wall_sets = None
        self._last_stress: np.ndarray | None = None
        self._rebuild(force=True)

    # ---------------------------------------------------------- neighbour list
    def _minimum_image(self, d: np.ndarray) -> np.ndarray:
        if not self.periodic:
            return d
        lx, ly = self.box
        d = d.copy()
        d[:, 0] -= lx * np.round(d[:, 0] / lx)
        d[:, 1] -= ly * np.round(d[:, 1] / ly)
        return d

    def _rebuild(self, *, force: bool = False) -> None:
        reach = 2.0 * float(self.radii.max()) + self.config.skin
        if not force and self._cache is not None and self._cache_pos is not None:
            drift = float(np.max(np.linalg.norm(self.pos - self._cache_pos, axis=1)))
            if drift <= 0.5 * self.config.skin:
                return
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(reach, output_type="ndarray").reshape(-1, 2)
        extra = []
        if self.periodic:
            lx, ly = self.box
            near = {
                "x0": np.flatnonzero(self.pos[:, 0] < reach),
                "x1": np.flatnonzero(self.pos[:, 0] > lx - reach),
                "y0": np.flatnonzero(self.pos[:, 1] < reach),
                "y1": np.flatnonzero(self.pos[:, 1] > ly - reach),
            }
            combos = [((lx, 0.0), ("x0",), ("x1",)),
                      ((0.0, ly), ("y0",), ("y1",)),
                      ((lx, ly), ("x0", "y0"), ("x1", "y1")),
                      ((lx, -ly), ("x0", "y1"), ("x1", "y0"))]
            for shift, a_keys, b_keys in combos:
                a = near[a_keys[0]]
                for key in a_keys[1:]:
                    a = np.intersect1d(a, near[key])
                b = near[b_keys[0]]
                for key in b_keys[1:]:
                    b = np.intersect1d(b, near[key])
                if not len(a) or not len(b):
                    continue
                d = (self.pos[b][None, :, :]
                     - (self.pos[a][:, None, :] + np.asarray(shift))).reshape(-1, 2)
                d = self._minimum_image(d).reshape(len(a), len(b), 2)
                ii, jj = np.nonzero(np.linalg.norm(d, axis=2) < reach)
                if len(ii):
                    extra.append(np.stack([a[ii], b[jj]], axis=1))
        if extra:
            pairs = np.vstack([pairs] + extra)
        if len(pairs):
            pairs = np.unique(np.sort(pairs.astype(np.int64), axis=1), axis=0)
            i, j = pairs[:, 0], pairs[:, 1]
        else:
            i = j = np.zeros(0, dtype=np.int64)
        # carry the internal variables across the rebuild by contact key: the
        # tangential displacement and the rolling angle are *history* variables,
        # so a neighbour-list renewal must not wipe them (it would silently turn
        # the assembly frictionless, and -- worse -- a dt-dependent one).
        xi, roll = self._remap_state(i, j)
        self._cache = (i, j)
        self._cache_pos = self.pos.copy()
        self._xi = xi
        self._roll = roll
        self._active = np.zeros(len(i), dtype=bool)
        if len(i):
            kij = self._pair_stiffness(i, j)
            self._pair_stiff = kij > self.config.kn * (1.0 + 1e-12)
        else:
            self._pair_stiff = np.zeros(0, dtype=bool)

    def _remap_state(self, i: np.ndarray, j: np.ndarray):
        """Re-index ``xi`` / ``roll`` onto a new contact list by contact key."""
        n = len(self.pos)
        new_keys = i * n + j
        xi = np.zeros((len(i), 2))
        roll = np.zeros(len(i))
        if self._cache is None or not len(self._cache[0]) or not len(new_keys):
            return xi, roll
        old_i, old_j = self._cache
        old_keys = old_i * n + old_j
        order = np.argsort(old_keys)
        sorted_keys = old_keys[order]
        pos = np.clip(np.searchsorted(sorted_keys, new_keys), 0, len(sorted_keys) - 1)
        found = sorted_keys[pos] == new_keys
        if found.any():
            xi[found] = self._xi[order[pos[found]]]
            roll[found] = self._roll[order[pos[found]]]
        return xi, roll

    def _pair_geometry(self, idx: np.ndarray | None = None):
        """``(i, j, n_ij, overlap, dist)``; ``n_ij`` points from ``j`` to ``i``."""
        i, j = self._cache
        if idx is not None:
            i, j = i[idx], j[idx]
        if not len(i):
            return i, j, np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        l = self._minimum_image(self.pos[j] - self.pos[i])
        dist = np.linalg.norm(l, axis=1)
        n = -l / np.maximum(dist, 1e-30)[:, None]
        return i, j, n, self.radii[i] + self.radii[j] - dist, dist

    def pair_geometry(self):
        """Public ``(i, j, n_ij, overlap)`` over the cached candidate pairs."""
        i, j, n, overlap, _ = self._pair_geometry()
        return i, j, n, overlap

    def _pair_stiffness(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        cfg = self.config
        stiff = self.stiff[i] | self.stiff[j]
        return cfg.kn * np.where(stiff, cfg.stiff_ratio, 1.0)

    def _active_margin(self) -> float:
        """Neighbour-list skin actually consumed by one coarse step."""
        vmax = float(np.max(np.abs(self.vel))) if len(self.vel) else 0.0
        return 2.0 * vmax * max(self.dt, 0.0) + 0.25 * self.config.skin

    # ------------------------------------------------------------- kinematics
    def _pair_kinematics(self, i, j, n_ij, overlap):
        """Normal force, relative velocity and levers for a contact selection."""
        cfg = self.config
        k_ij = self._pair_stiffness(i, j)
        m_eff = (self.mass[i] * self.mass[j]) / (self.mass[i] + self.mass[j])
        r_eff = self.radii[i] * self.radii[j] / (self.radii[i] + self.radii[j])
        lever_i = -self.radii[i][:, None] * n_ij
        lever_j = self.radii[j][:, None] * n_ij
        v_i = self.vel[i] + _perp(self.omega[i][:, None] * lever_i)
        v_j = self.vel[j] + _perp(self.omega[j][:, None] * lever_j)
        v_rel = v_i - v_j
        v_n = np.sum(v_rel * n_ij, axis=1)
        c_n = 2.0 * cfg.zeta * np.sqrt(k_ij * m_eff)
        f_n = np.maximum(0.0, k_ij * overlap - c_n * v_n)
        return k_ij, r_eff, lever_i, lever_j, v_rel, v_n, f_n

    def _tangential_force(self, xi, f_n, k_ij):
        """Tangential force from ``xi`` with the friction-cone return map."""
        cfg = self.config
        k_t = cfg.kt_ratio * k_ij
        f_t = -k_t[:, None] * xi
        mag = np.linalg.norm(f_t, axis=1)
        cap = cfg.mu * f_n
        scale = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
        return f_t * scale[:, None], scale

    # ------------------------------------------------------------- slow pass
    def _slow_pass(self, dt: float):
        """Renew the active set, advance the internal variables, return slow forces.

        The slow force collects everything that is resolved on the coarse step:
        the soft normal contacts (held constant over the step), the box walls and
        the external terms.  The stiff contacts are *not* in here -- they are the
        fast arm.  ``xi`` / ``roll`` advance once per coarse step with the cone
        return map applied inside the multiple step.
        """
        cfg = self.config
        i, j, n_ij, overlap, _ = self._pair_geometry()
        margin = self._active_margin()
        active = overlap > -margin
        self._active = active
        stiff = self._pair_stiff if len(self._pair_stiff) == len(active) else np.zeros(
            len(active), dtype=bool)
        if cfg.split_slow_contacts:
            self._fast_idx = np.flatnonzero(active & stiff)
            self._slow_idx = np.flatnonzero(active & ~stiff)
        else:
            self._fast_idx = np.flatnonzero(active)
            self._slow_idx = np.zeros(0, dtype=np.int64)
        self.work["slow"] += 1
        if cfg.mu > 0.0 and cfg.kt_ratio > 0.0:
            if self._slow_idx.size:
                self._advance_state(self._slow_idx, dt)
        force, torque = self._pair_forces(self._slow_idx, friction=True)
        return force + self._external(), torque

    def _advance_state(self, idx: np.ndarray, dt: float) -> None:
        """Tangential/rolling internal variables of ``idx`` over ``dt``."""
        cfg = self.config
        i, j, n_ij, overlap, _ = self._pair_geometry(idx)
        k_ij, r_eff, _, _, v_rel, v_n, f_n = self._pair_kinematics(i, j, n_ij, overlap)
        v_t = v_rel - v_n[:, None] * n_ij
        xi = self._xi[idx] + v_t * dt
        xi = xi - np.sum(xi * n_ij, axis=1)[:, None] * n_ij
        _, scale = self._tangential_force(xi, f_n, k_ij)
        self._xi[idx] = xi * scale[:, None]
        if cfg.mu_rolling > 0.0:
            k_t = cfg.kt_ratio * k_ij
            k_r = 0.25 * k_t * r_eff ** 2
            roll = self._roll[idx] + (self.omega[i] - self.omega[j]) * dt
            m_r = k_r * roll
            cap = cfg.mu_rolling * r_eff * f_n
            scale_r = np.where(np.abs(m_r) > cap, cap / np.maximum(np.abs(m_r), 1e-30), 1.0)
            self._roll[idx] = roll * scale_r

    # ------------------------------------------------------------- fast pass
    def _pair_forces(self, idx: np.ndarray, *, friction: bool,
                     state_dt: float | None = None):
        """Contact forces over ``idx``.

        ``state_dt`` advances the tangential/rolling internal variables by that
        increment *inside* the pass (the fine step).  ``None`` keeps them frozen,
        which is what the coarse pass and the ablation variant need.
        """
        cfg = self.config
        force = np.zeros((len(self.pos), 2))
        torque = np.zeros(len(self.pos))
        if not idx.size:
            return force, torque
        i, j, n_ij, overlap, _ = self._pair_geometry(idx)
        keep = overlap > 0.0
        if not keep.any():
            return force, torque
        sel = idx[keep]
        i, j, n_ij, overlap = i[keep], j[keep], n_ij[keep], overlap[keep]
        k_ij, r_eff, lever_i, lever_j, v_rel, v_n, f_n = self._pair_kinematics(
            i, j, n_ij, overlap)
        total = f_n[:, None] * n_ij
        if friction and cfg.mu > 0.0 and cfg.kt_ratio > 0.0:
            xi = self._xi[sel]
            roll = self._roll[sel]
            if state_dt is not None:
                v_t = v_rel - v_n[:, None] * n_ij
                xi = xi + v_t * state_dt
                xi = xi - np.sum(xi * n_ij, axis=1)[:, None] * n_ij
                self._xi[sel] = xi
            f_t, scale = self._tangential_force(xi, f_n, k_ij)
            if state_dt is not None:
                self._xi[sel] = xi * scale[:, None]
            total = total + f_t
            if cfg.mu_rolling > 0.0:
                k_t = cfg.kt_ratio * k_ij
                k_r = 0.25 * k_t * r_eff ** 2
                if state_dt is not None:
                    roll = roll + (self.omega[i] - self.omega[j]) * state_dt
                m_r = k_r * roll
                cap_r = cfg.mu_rolling * r_eff * f_n
                scale_r = np.where(np.abs(m_r) > cap_r,
                                   cap_r / np.maximum(np.abs(m_r), 1e-30), 1.0)
                if state_dt is not None:
                    self._roll[sel] = roll * scale_r
                m_r = m_r * scale_r
                np.add.at(torque, i, -m_r)
                np.add.at(torque, j, m_r)
        _scatter(force, torque, i, j, total, _cross(lever_i, total),
                 _cross(lever_j, -total))
        return force, torque

    def _fast_pass(self, h: float | None = None):
        """Active contacts at the sub-step.

        The *forces* are always re-evaluated here; the internal variables follow
        the sub-step when ``h`` is given (default: the sub-step of this system).
        Freezing them for the whole coarse step biases the friction (a measurable
        ``phi_J`` shift), so the default keeps them on the fine rate -- the coarse
        rate is what buys the speed, and the friction arm is not what pays for it.
        """
        self.work["fast"] += 1
        state_dt = self.dt / self.n_sub if h is None else h
        if self.config.split_slow_contacts or self.config.state_at_coarse:
            state_dt = None  # ablation: the coarse pass owns the state
        force, torque = self._pair_forces(self._fast_idx, friction=True,
                                          state_dt=state_dt)
        fw, tw = self._wall_forces(state_dt)
        return force + fw, torque + tw

    def _wall_active_sets(self):
        """Index sets of the disks near each box plane (renewed on the slow step)."""
        if self.box is None or self.periodic:
            return []
        lx, ly = self.box
        margin = self._active_margin()
        out = []
        for p, (axis, coord, sign) in enumerate(_PLANES(lx, ly)):
            dist = np.abs(self.pos[:, axis] - coord)
            sel = (self.radii - dist) > -margin
            if sel.any():
                out.append((p, axis, coord, sign, np.flatnonzero(sel)))
        return out

    def _wall_forces(self, dt_state: float | None = None):
        """Wall normal + tangential forces on the cached plane selections."""
        cfg = self.config
        force = np.zeros((len(self.pos), 2))
        torque = np.zeros(len(self.pos))
        if self.box is None or self.periodic:
            return force, torque
        if self._wall_sets is None:
            self._wall_sets = self._wall_active_sets()
        for p, axis, coord, sign, idx in self._wall_sets:
            if not idx.size:
                continue
            ov = self.radii[idx] - np.abs(self.pos[idx, axis] - coord)
            nvec = np.zeros((len(idx), 2))
            nvec[:, axis] = sign
            lever = -self.radii[idx][:, None] * nvec
            v_i = self.vel[idx] + _perp(self.omega[idx][:, None] * lever)
            v_n = v_i[:, axis] * sign
            c_n = 2.0 * cfg.zeta * np.sqrt(cfg.kn * self.mass[idx])
            f_n = np.maximum(0.0, cfg.kn * ov - c_n * v_n)
            total = nvec * f_n[:, None]
            if cfg.mu > 0.0 and cfg.kt_ratio > 0.0:
                v_t = v_i - v_n[:, None] * nvec
                xi = self._wxi[p, idx]
                if dt_state is not None:
                    xi = xi + v_t * dt_state
                    xi = xi - np.sum(xi * nvec, axis=1)[:, None] * nvec
                f_t, scale = self._tangential_force(xi, f_n, np.full(len(idx), cfg.kn))
                if dt_state is not None:
                    self._wxi[p, idx] = xi * scale[:, None]
                total = total + f_t
            live = ov > 0.0
            if not live.any():
                continue
            np.add.at(force, idx[live], total[live])
            np.add.at(torque, idx[live], _cross(lever[live], total[live]))
        return force, torque

    def _external(self) -> np.ndarray:
        cfg = self.config
        force = np.zeros_like(self.pos)
        g = np.asarray(cfg.gravity, dtype=np.float64)
        if np.any(g != 0.0):
            force += self.mass[:, None] * g[None, :]
        if cfg.drag_time > 0.0:
            if cfg.shear_rate != 0.0 and self.box is not None:
                target = np.zeros_like(self.pos)
                target[:, 0] = cfg.shear_rate * (self.pos[:, 1] - 0.5 * self.box[1])
                force += self.mass[:, None] * (target - self.vel) / cfg.drag_time
            else:
                force += -self.mass[:, None] * self.vel / cfg.drag_time
        return force

    # ------------------------------------------------------- single-rate path
    def _evaluate_single(self, dt: float):
        """All arms and the internal variables at the (fine) step: the baseline."""
        cfg = self.config
        i, j, n_ij, overlap, _ = self._pair_geometry()
        force = np.zeros((len(self.pos), 2))
        torque = np.zeros(len(self.pos))
        keep = overlap > 0.0
        if keep.any():
            i, j, n_ij, overlap = i[keep], j[keep], n_ij[keep], overlap[keep]
            k_ij, r_eff, lever_i, lever_j, v_rel, v_n, f_n = self._pair_kinematics(
                i, j, n_ij, overlap)
            total = f_n[:, None] * n_ij
            if cfg.mu > 0.0 and cfg.kt_ratio > 0.0:
                xi = self._xi[keep] + (v_rel - v_n[:, None] * n_ij) * dt
                xi = xi - np.sum(xi * n_ij, axis=1)[:, None] * n_ij
                f_t, scale = self._tangential_force(xi, f_n, k_ij)
                self._xi[keep] = xi * scale[:, None]
                total = total + f_t
                if cfg.mu_rolling > 0.0:
                    k_t = cfg.kt_ratio * k_ij
                    k_r = 0.25 * k_t * r_eff ** 2
                    roll = self._roll[keep] + (self.omega[i] - self.omega[j]) * dt
                    m_r = k_r * roll
                    cap_r = cfg.mu_rolling * r_eff * f_n
                    scale_r = np.where(np.abs(m_r) > cap_r,
                                       cap_r / np.maximum(np.abs(m_r), 1e-30), 1.0)
                    self._roll[keep] = roll * scale_r
                    m_r = m_r * scale_r
                    np.add.at(torque, i, -m_r)
                    np.add.at(torque, j, m_r)
            np.add.at(force, i, total)
            np.add.at(force, j, -total)
            np.add.at(torque, i, _cross(lever_i, total))
            np.add.at(torque, j, _cross(lever_j, -total))
        self._wall_sets = self._wall_active_sets()
        fw, tw = self._wall_forces(dt)
        self.work["single"] += 1
        return force + fw + self._external(), torque + tw

    # --------------------------------------------------------------- time step
    def critical_scales(self) -> dict[str, float]:
        """Stiffness-based contact time scales of the assembly (SI)."""
        cfg = self.config
        m_min = float(self.mass.min())
        k_soft = float(cfg.kn)
        k_stiff = float(cfg.kn) * float(cfg.stiff_ratio)
        return {
            "m_min": m_min,
            "k_soft": k_soft,
            "k_stiff": k_stiff,
            "dt_c_soft": cundall_critical_timestep(m_min, k_soft),
            "dt_c_stiff": cundall_critical_timestep(m_min, k_stiff),
            "k_t": float(cfg.kt_ratio) * k_soft,
        }

    def setup_timestep(self, *, dt: float | None = None, mode: str = "mts",
                       outer_margin: float | None = None,
                       fast_margin: float | None = None,
                       friction_margin: float = 0.25,
                       check_friction: bool = True) -> dict[str, object]:
        """Wire the stability contract into ``self.dt`` / ``self.n_sub``.

        ``mode="single"`` takes one step fine enough for the *stiffest* contact
        (what a single-rate solver must do); ``mode="mts"`` takes the coarse step
        the friction arm and the soft majority allow, and sub-cycles the contact
        resolution down to the stiff limit.
        """
        if mode not in ("mts", "single"):
            raise ValueError("mode must be 'mts' or 'single'")
        cfg = self.config
        scales = self.critical_scales()
        outer = cfg.outer_margin if outer_margin is None else float(outer_margin)
        fast = cfg.fast_margin if fast_margin is None else float(fast_margin)
        dt_stiff_limit = fast * scales["dt_c_stiff"]
        dt_friction_limit = friction_margin * linear_contact_time(
            0.5 * scales["m_min"], scales["k_t"]) if cfg.mu > 0.0 else math.inf
        dt_soft_limit = min(outer * scales["dt_c_soft"], dt_friction_limit)

        if mode == "single":
            coarse = dt_stiff_limit if dt is None else float(dt)
            gate = cundall_gate(coarse, m_min=scales["m_min"], k_max=scales["k_stiff"],
                                margin=fast).check()
            n_sub = 1
            fine = coarse
        else:
            coarse = dt_soft_limit if dt is None else float(dt)
            gate = cundall_gate(coarse, m_min=scales["m_min"], k_max=scales["k_soft"],
                                margin=outer).check()
            n_sub = max(1, int(math.ceil(coarse / dt_stiff_limit - 1e-12)))
            fine = coarse / n_sub
        fine_gate = cundall_gate(fine, m_min=scales["m_min"], k_max=scales["k_stiff"],
                                 margin=fast)
        friction = None
        if check_friction and cfg.mu > 0.0 and cfg.kt_ratio > 0.0:
            friction = stability_gate(coarse,
                                      linear_contact_time(0.5 * scales["m_min"], scales["k_t"]),
                                      margin=friction_margin, kind="hooke")
            friction.check()
        self.dt = coarse
        self.n_sub = n_sub
        self.mode = mode
        self.gate = {
            "mode": mode,
            "dt_outer": coarse,
            "n_sub": n_sub,
            "dt_fine": fine,
            "outer_gate": gate.as_dict(),
            "fine_gate": fine_gate.as_dict(),
            "friction_gate": None if friction is None else friction.as_dict(),
            "critical": scales,
        }
        return self.gate

    def step(self, n: int = 1) -> None:
        """Advance ``n`` coarse steps."""
        if self.dt <= 0.0:
            raise RuntimeError("call setup_timestep() before step()")
        if n < 0:
            raise ValueError("n must be nonnegative")
        cfg = self.config
        dt = self.dt
        inv_m = 1.0 / self.mass[:, None]
        inv_i = 1.0 / self.inertia
        for _ in range(n):
            self._rebuild()
            if self.mode == "single" or self.n_sub == 1:
                force, torque = self._evaluate_single(dt)
                self.vel += dt * force * inv_m
                self.omega += dt * torque * inv_i
                self.pos += dt * self.vel
                self.theta += dt * self.omega
            else:
                # RESPA / velocity-Verlet with two rate classes: the slow force
                # (soft contacts + walls + external) enters as half kicks at both
                # ends of the coarse step, the stiff contacts are re-evaluated on
                # every sub-step.
                f_slow, t_slow = self._slow_pass(dt)
                h = dt / self.n_sub
                # half kick with the slow force, then ``n_sub`` leapfrog
                # drift-kick sub-steps whose drifts sum to exactly ``dt``
                self.vel += 0.5 * dt * f_slow * inv_m
                self.omega += 0.5 * dt * t_slow * inv_i
                for _ in range(self.n_sub):
                    self.pos += h * self.vel
                    self.theta += h * self.omega
                    force, torque = self._fast_pass()
                    self.vel += h * force * inv_m
                    self.omega += h * torque * inv_i
                self.vel += 0.5 * dt * f_slow * inv_m
                self.omega += 0.5 * dt * t_slow * inv_i
            if cfg.damping_time > 0.0:
                decay = math.exp(-dt / cfg.damping_time)
                self.vel *= decay
                self.omega *= decay
            if self.periodic:
                lx, ly = self.box
                self.pos[:, 0] = np.mod(self.pos[:, 0], lx)
                self.pos[:, 1] = np.mod(self.pos[:, 1], ly)
            self.time += dt
            self.steps += 1

    # ------------------------------------------------------------- diagnostics
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass * np.sum(self.vel ** 2, axis=1))
                     + 0.5 * np.sum(self.inertia * self.omega ** 2))

    def potential_energy(self) -> float:
        """Elastic energy of the overlapping contacts (normal springs)."""
        cfg = self.config
        i, j, _, overlap, _ = self._pair_geometry()
        energy = 0.0
        if len(i):
            keep = overlap > 0.0
            if keep.any():
                k_ij = self._pair_stiffness(i[keep], j[keep])
                energy += float(0.5 * np.sum(k_ij * overlap[keep] ** 2))
        if self.box is not None and not self.periodic:
            lx, ly = self.box
            for axis, coord, _ in _PLANES(lx, ly):
                ov = self.radii - np.abs(self.pos[:, axis] - coord)
                sel = ov > 0.0
                if sel.any():
                    energy += float(0.5 * cfg.kn * np.sum(ov[sel] ** 2))
        return energy

    def energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.vel, axis=0)

    def stress_tensor(self) -> np.ndarray:
        """Virial stress (read-only: the internal variables are not advanced)."""
        cfg = self.config
        i, j, n_ij, overlap, _ = self._pair_geometry()
        stress = np.zeros((2, 2))
        keep = overlap > 0.0
        if keep.any():
            i, j, n_ij, overlap = i[keep], j[keep], n_ij[keep], overlap[keep]
            k_ij, _, _, _, v_rel, v_n, f_n = self._pair_kinematics(i, j, n_ij, overlap)
            total = f_n[:, None] * n_ij
            if (cfg.mu > 0.0 and cfg.kt_ratio > 0.0
                    and len(self._xi) == len(keep)):
                f_t, _ = self._tangential_force(self._xi[keep], f_n, k_ij)
                total = total + f_t
            sep = self._minimum_image(self.pos[j] - self.pos[i])
            stress += np.einsum("na,nb->ab", total, sep)
        if self.box is not None and not self.periodic:
            lx, ly = self.box
            for axis, coord, sign in _PLANES(lx, ly):
                ov = self.radii - np.abs(self.pos[:, axis] - coord)
                sel = ov > 0.0
                if not sel.any():
                    continue
                idx = np.flatnonzero(sel)
                nvec = np.zeros((len(idx), 2))
                nvec[:, axis] = sign
                lever = -self.radii[idx][:, None] * nvec
                v_i = self.vel[idx] + _perp(self.omega[idx][:, None] * lever)
                v_n = np.sum(v_i * nvec, axis=1)
                c_n = 2.0 * cfg.zeta * np.sqrt(cfg.kn * self.mass[idx])
                f_n = np.maximum(0.0, cfg.kn * ov[idx] - c_n * v_n)
                stress += np.einsum("na,nb->ab", f_n[:, None] * nvec, lever)
        area = self.box[0] * self.box[1] if self.box is not None else 1.0
        self._last_stress = stress / area
        return self._last_stress

    def pressure(self) -> float:
        """2-D scalar pressure ``P = -tr(sigma)/2`` (N/m)."""
        sigma = self.stress_tensor()
        return float(-0.5 * (sigma[0, 0] + sigma[1, 1]))

    def shear_stress(self) -> float:
        return float(self.stress_tensor()[0, 1])

    def area_fraction(self) -> float:
        if self.box is None:
            raise ValueError("area fraction needs a box")
        return float(np.sum(math.pi * self.radii ** 2) / (self.box[0] * self.box[1]))

    def coordination(self, *, threshold: float = 0.0) -> float:
        _, _, _, overlap, _ = self._pair_geometry()
        return 2.0 * int(np.count_nonzero(overlap > threshold)) / len(self.pos)

    def scale_box(self, factor: float) -> None:
        """Scale positions and the box about the box centre (radii unchanged)."""
        if self.box is None:
            raise ValueError("scale_box needs a box")
        factor = float(factor)
        if factor <= 0.0 or not math.isfinite(factor):
            raise ValueError("factor must be finite and positive")
        lx, ly = self.box
        self.pos *= factor
        self.box = (lx * factor, ly * factor)
        if self._cache_pos is not None:
            self._cache_pos = self._cache_pos * factor
        self._rebuild()

    def servo_pressure(self, p_target: float, *, steps: int, eta: float = 200.0,
                       max_rate: float = 35.0, every: int = 20) -> dict[str, float]:
        """Drive the box so the contact pressure tracks ``p_target``.

        ``d ln L / dt = (P - P_target) / eta`` with the step clipped to
        ``max_step``; the pressure (a slow variable) is refreshed every ``every``
        steps.
        """
        pressure = self.pressure()
        for k in range(int(steps)):
            if k % int(every) == 0:
                pressure = self.pressure()
            rate = (pressure - float(p_target)) / float(eta)
            rate = float(min(max(rate, -max_rate), max_rate))
            delta = self.dt * rate
            self.scale_box(1.0 + delta)
            self.step(1)
        pressure = self.pressure()
        return {"phi": self.area_fraction(), "pressure": float(pressure),
                "side": float(self.box[0]), "kinetic_energy": self.kinetic_energy()}


def _PLANES(lx: float, ly: float):
    """The four axis-aligned box planes as ``(axis, coordinate, normal sign)``."""
    return ((0, 0.0, +1.0), (0, lx, -1.0), (1, 0.0, +1.0), (1, ly, -1.0))
