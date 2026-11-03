"""Periodic dissipative granular gas: Haff's-law (homogeneous cooling) verification.

Why this module exists
----------------------
TensorDEM verifies its normal contact law with *two-body* impacts
(:mod:`tensordem.restitution`, :mod:`tensordem.contact_impact`,
:mod:`tensordem.contact_restitution`) and verifies elastic wave/energy
behaviour with bonded lattices (:mod:`tensordem.lattice_waves`,
:mod:`tensordem.dem3d_energy`).  Neither checks what the *same* contact law does
when thousands of simultaneous, rapidly reshuffling collisions must dissipate
energy consistently: the statistical (kinetic-theory) level was an open gap.

This module supplies the missing benchmark: a monodisperse, frictionless,
gravity-free gas of equal discs in a 2-D periodic box, using the *identical*
normal contact convention as :class:`tensordem.dem3d.IceDEM3D` (linear spring
with restitution-parameterised dashpot, compressive-only clamp
``F = max(k*delta - gamma*v_n, 0)``), the identical cell-free pair treatment
(no bonds, no tangential or rolling branch, so contact damping is the only
dissipation channel), and the same damping inversion helper
(:func:`tensordem.contact_restitution.damping_for_target_restitution`).

Physics benchmark
-----------------
In the homogeneous cooling state (HCS) of a dilute inelastic gas, kinetic
theory predicts that the translational granular temperature decays as
Haff's law (Haff 1983, *J. Fluid Mech.* 134:401)::

    T(t) = T0 / (1 + t/tau)**2,        equivalently
    1/sqrt(T) = 1/sqrt(T0) + (A/2) * t,

i.e. ``dT/dt = -A * T**(3/2)``.  The ``-2`` exponent follows from the collision
frequency scaling as ``sqrt(T)`` combined with a fixed fractional energy loss
per collision; it is independent of dimensionality and of the contact
stiffness.  For 2-D equal discs the collision rate per unit area is
``(1/2) n^2 d <|v_rel|>`` and the energy lost per collision is
``(1-e^2) * mu * v_n^2 / 2`` with ``mu = m/2``; the average must be weighted by
the collision rate ``|v_rel|``, which for a 2-D Maxwellian gives
``<|v_rel| v_n^2> / <|v_rel|> = 3 T / m`` (not ``2 T / m``; rate weighting biases
collisions towards fast pairs).  With ``E = n T`` (two translational DOF) and
``<|v_rel|> = sqrt(pi*T/m)``::

    A_kt = (3/8) * (1 - e**2) * n * d_diam * sqrt(pi / m),

with ``n`` the number density and ``d_diam`` the disc diameter.  The prefactor
inherits the dilute-limit (Boltzmann) approximation, so a finite-density run is
expected to differ; the *exponent* and the *linearity* of ``1/sqrt(T)`` are the
robust assertions, the prefactor comparison is reported as a diagnostic.

Conventions
-----------
* SI units.  2-D discs of mass ``m = density * pi * radius**2`` (per unit
  out-of-plane thickness).
* Granular temperature from equipartition, ``T = m * sum(v**2) / (2*N)`` for two
  translational degrees of freedom, so ``<v_x**2> = T/m``.
* Time integration: symplectic Euler (kick then drift), double precision.

Limitations (documented, not hidden)
------------------------------------
* Frictionless and rotationless: validates the *normal* branch only.  A
  frictional gas would add rotational energy and tangential dissipation, which
  are verified separately at the two-body level.
* Dilute-Boltzmann prefactor ignores finite-density (Enskog) and velocity
  correlation corrections; deviations grow with the packing fraction.
* HCS is linearly unstable at long times (clustering / inelastic collapse);
  fits are therefore restricted to a temperature window and never extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Any

import torch

from .contact_restitution import damping_for_target_restitution

__all__ = [
    "GranularGasConfig",
    "GranularGas",
    "kinetic_theory_cooling_coefficient",
    "fit_haff_exponent",
    "fit_inverse_sqrt_temperature",
    "maxwell_2d_diagnostics",
]


@dataclass(frozen=True)
class GranularGasConfig:
    """Configuration of a 2-D periodic granular gas.

    The stiffness is *not* a free input: it is derived from
    ``target_overlap_fraction`` so that the hardest impacts produce a
    prescribed maximum overlap ``delta_max / diameter``.  This keeps the
    contact close to the hard-disc limit without paying for a stiff contact,
    and it makes the time step follow the contact duration.
    """

    n_particles: int = 400
    radius: float = 5.0e-3
    density: float = 1000.0
    packing_fraction: float = 0.15
    restitution: float = 0.8
    target_overlap_fraction: float = 0.03
    initial_rms_speed: float = 0.05
    steps_per_contact: int = 40
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("n_particles",):
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be >= 2")
        if not (0.0 < self.restitution <= 1.0):
            raise ValueError("restitution must be in (0, 1]")
        for name in ("radius", "density", "packing_fraction", "target_overlap_fraction",
                     "initial_rms_speed"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < self.packing_fraction < 0.6):
            raise ValueError("packing_fraction must be in (0, 0.6)")
        if not (0.0 < self.target_overlap_fraction < 0.2):
            raise ValueError("target_overlap_fraction must be in (0, 0.2)")
        if int(self.steps_per_contact) < 20:
            raise ValueError("steps_per_contact must be >= 20")

    # -- derived quantities -------------------------------------------------
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def mass(self) -> float:
        return self.density * math.pi * self.radius ** 2

    @property
    def box_length(self) -> float:
        area = self.n_particles * math.pi * self.radius ** 2 / self.packing_fraction
        return math.sqrt(area)

    @property
    def number_density(self) -> float:
        return self.n_particles / self.box_length ** 2

    @property
    def stiffness(self) -> float:
        """Spring constant giving ``delta/diameter = target_overlap_fraction``
        at the hardest (initial) impacts."""
        m_eff = 0.5 * self.mass
        delta = self.target_overlap_fraction * self.diameter
        return m_eff * self.initial_rms_speed ** 2 / delta ** 2

    @cached_property
    def damping(self) -> float:
        """Dashpot coefficient realising ``restitution`` for a single impact.

        Uses the repository's clamped-law inversion, so the *observed*
        two-body restitution matches ``restitution`` (the engine's closed-form
        ``zeta = |ln e| / sqrt(pi**2 + ln**2 e)`` mapping does not, see
        :mod:`tensordem.contact_restitution`).
        """
        if self.restitution >= 1.0:
            return 0.0
        return float(damping_for_target_restitution(
            self.restitution, self.stiffness, 0.5 * self.mass)["gamma_n"])

    @property
    def contact_time(self) -> float:
        return math.pi * math.sqrt(0.5 * self.mass / self.stiffness)

    @property
    def dt(self) -> float:
        return self.contact_time / float(self.steps_per_contact)

    @property
    def initial_temperature(self) -> float:
        """``T0 = m * <v**2> / 2`` for the requested rms speed (2-D)."""
        return 0.5 * self.mass * self.initial_rms_speed ** 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_particles": int(self.n_particles),
            "radius": self.radius,
            "diameter": self.diameter,
            "density": self.density,
            "mass": self.mass,
            "packing_fraction": self.packing_fraction,
            "box_length": self.box_length,
            "number_density": self.number_density,
            "restitution_target": self.restitution,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "contact_time": self.contact_time,
            "dt": self.dt,
            "initial_rms_speed": self.initial_rms_speed,
            "initial_temperature": self.initial_temperature,
            "seed": int(self.seed),
        }


class GranularGas:
    """2-D periodic monodisperse gas with the engine's normal contact law."""

    _DTYPE = torch.float64

    def __init__(self, config: GranularGasConfig = GranularGasConfig()) -> None:
        self.config = config
        self.dtype = self._DTYPE
        gen = torch.Generator(device="cpu").manual_seed(int(config.seed))
        self.positions = self._poisson_disc_positions(gen)
        # Isotropic 2-D Maxwellian: per-component variance T/m = v_rms^2/2.
        sigma = config.initial_rms_speed / math.sqrt(2.0)
        self.velocities = torch.randn((config.n_particles, 2), generator=gen,
                                      dtype=self.dtype) * sigma
        # Re-impose the exact requested temperature (removes the seed-dependent
        # sampling error so T0 is a controlled input, not a random variable).
        self.velocities *= math.sqrt(config.initial_temperature
                                     / self.kinetic_temperature())
        self.time = 0.0
        self.steps = 0
        self.damping_work = 0.0
        self.collision_onsets = 0
        self.collision_steps = 0
        n = config.n_particles
        upper = torch.triu_indices(n, n, offset=1)
        self._pair_i, self._pair_j = upper[0], upper[1]
        self._prev_active = torch.zeros(self._pair_i.numel(), dtype=torch.bool,
                                        device=self._pair_i.device)

    # -- construction helpers ----------------------------------------------
    def _poisson_disc_positions(self, gen: torch.Generator) -> torch.Tensor:
        cfg = self.config
        n, box, minimum = cfg.n_particles, cfg.box_length, cfg.diameter * 1.02
        positions = torch.zeros((n, 2), dtype=self.dtype)
        count = 0
        while count < n:
            candidate = torch.rand(2, generator=gen, dtype=self.dtype) * box
            if count:
                delta = positions[:count] - candidate
                delta -= box * torch.round(delta / box)
                if torch.linalg.vector_norm(delta, dim=1).min() < minimum:
                    continue
            positions[count] = candidate
            count += 1
        return positions

    # -- state ---------------------------------------------------------------
    def kinetic_temperature(self) -> float:
        """Granular temperature ``T = m * sum(v^2) / (2 N)`` (two 2-D DOF)."""
        return float(self.config.mass * self.velocities.square().sum()
                     / (2.0 * self.config.n_particles))

    def kinetic_energy(self) -> float:
        return float(0.5 * self.config.mass * self.velocities.square().sum())

    def pressure_tensor(self) -> torch.Tensor:
        """Configurational+kinetic pressure from the momentum flux (diagnostic)."""
        cfg = self.config
        kinetic = cfg.mass * (self.velocities.T @ self.velocities)
        return kinetic / (2.0 * cfg.box_length ** 2)

    # -- dynamics ------------------------------------------------------------
    def _pair_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        box = cfg.box_length
        # Normal points from i to j (repository convention, see IceDEM3D.forces).
        delta = self.positions[self._pair_j] - self.positions[self._pair_i]
        delta -= box * torch.round(delta / box)  # minimum image
        distance = torch.linalg.vector_norm(delta, dim=1)
        safe = distance.clamp_min(torch.finfo(self.dtype).eps)
        normal = delta / safe[:, None]
        overlap = (cfg.diameter - distance).clamp_min(0.0)
        return overlap, normal, distance

    def step(self) -> None:
        """One symplectic-Euler step with compressive-only linear contacts."""
        cfg = self.config
        dt = cfg.dt
        k, gamma = cfg.stiffness, cfg.damping
        overlap, normal, _ = self._pair_geometry()
        active = overlap > 0.0
        forces = torch.zeros_like(self.positions)
        if bool(active.any()):
            ci = self._pair_i[active][:, None]
            cj = self._pair_j[active][:, None]
            n_act = normal[active]
            relative_v = self.velocities[self._pair_j[active]] - self.velocities[self._pair_i[active]]
            vn = (relative_v * n_act).sum(dim=1)
            # Compressive-only clamped law: F = max(k*overlap - gamma*vn, 0),
            # with vn > 0 for separating pairs (i.e. the damper opposes approach).
            raw = k * overlap[active] - gamma * vn
            mag = raw.clamp_min(0.0)
            force = -mag[:, None] * n_act  # pushes i away from j
            forces.index_add_(0, self._pair_i[active], force)
            forces.index_add_(0, self._pair_j[active], -force)
            # Damping power is evaluated on compressive (unclamped) contacts only;
            # a clamped contact transmits no force and therefore does no work.
            transmitting = raw > 0.0
            if bool(transmitting.any()):
                self.damping_work += float(gamma * vn[transmitting].square().sum()) * dt
                self.collision_steps += int(transmitting.sum())
        onsets = int((active & ~self._prev_active).sum())
        self.collision_onsets += onsets
        self._prev_active = active.clone()
        # Symplectic Euler: kick, then drift (wrapped into the periodic cell).
        self.velocities = self.velocities + forces * (dt / cfg.mass)
        self.positions = torch.remainder(self.positions + self.velocities * dt,
                                         cfg.box_length)
        self.time += dt
        self.steps += 1

    def run(self, n_steps: int, *, sample_every: int = 1) -> dict[str, list[float]]:
        """Advance ``n_steps`` and return a sampled history."""
        history: dict[str, list[float]] = {
            "time": [], "temperature": [], "kinetic_energy": [],
            "damping_work": [], "collision_onsets": [],
            "variance_x": [], "variance_y": [],
        }
        for step in range(n_steps):
            self.step()
            if (step + 1) % sample_every == 0 or step == 0:
                history["time"].append(self.time)
                history["temperature"].append(self.kinetic_temperature())
                history["kinetic_energy"].append(self.kinetic_energy())
                history["damping_work"].append(self.damping_work)
                history["collision_onsets"].append(float(self.collision_onsets))
                var = self.velocities.var(dim=0, unbiased=False) * self.config.mass
                history["variance_x"].append(float(var[0]))
                history["variance_y"].append(float(var[1]))
        return history


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------
def kinetic_theory_cooling_coefficient(restitution: float, number_density: float,
                                       diameter: float, mass: float) -> float:
    """Dilute-limit coefficient of ``dT/dt = -A T**(3/2)`` for 2-D discs.

    ``A = (3/8) * (1 - e^2) * n * d * sqrt(pi / m)``; see the module docstring
    for the rate-weighted energy balance it comes from.  The prefactor inherits
    the Boltzmann (dilute) approximation and ignores the finite contact
    duration of soft spheres, so simulated gases show ``A/A_kt > 1`` with the
    excess growing in packing fraction -- reported, never fitted away.
    """
    if not (0.0 < restitution <= 1.0):
        raise ValueError("restitution must be in (0, 1]")
    for name, value in (("number_density", number_density), ("diameter", diameter),
                        ("mass", mass)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    return (3.0 / 8.0) * (1.0 - restitution ** 2) * number_density * diameter * math.sqrt(
        math.pi / mass)


def fit_inverse_sqrt_temperature(times: list[float] | torch.Tensor,
                                 temperatures: list[float] | torch.Tensor) -> dict[str, float]:
    """Least-squares fit of ``1/sqrt(T) = a + b t`` (Haff's law, direct form).

    Returns ``slope`` (``A/2``), ``intercept``, ``r_squared``, ``tau_haff``
    (``2/(A sqrt(T0))``) and ``A``.
    """
    t = torch.as_tensor(times, dtype=torch.float64)
    temp = torch.as_tensor(temperatures, dtype=torch.float64)
    if t.numel() < 3 or t.numel() != temp.numel():
        raise ValueError("need at least 3 matched samples")
    y = 1.0 / torch.sqrt(temp)
    design = torch.stack((torch.ones_like(t), t), dim=1)
    solution = torch.linalg.lstsq(design, y).solution
    intercept, slope = float(solution[0]), float(solution[1])
    residual = y - design @ solution
    ss_res = float(residual.square().sum())
    ss_tot = float((y - y.mean()).square().sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    A = 2.0 * slope
    # 1/sqrt(T) = 1/sqrt(T0) + (A/2) t  =>  tau = 2/(A sqrt(T0)) = 2*intercept/A.
    tau = 2.0 * intercept / A if A != 0.0 and intercept != 0.0 else float("inf")
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "A": A,
        "T0_fit": 1.0 / intercept ** 2 if intercept else float("inf"),
        "tau_haff": tau,
    }


def fit_haff_exponent(times: list[float] | torch.Tensor,
                      temperatures: list[float] | torch.Tensor,
                      *, tau: float | None = None) -> dict[str, float]:
    """Free-exponent fit ``T = T0 (1 + t/tau)**(-alpha)`` with ``tau`` pinned.

    A simultaneous free fit of ``(T0, tau, alpha)`` is ill-posed: over a finite
    window ``tau`` and ``alpha`` trade off almost exactly (an unconstrained
    grid search happily returns ``alpha > 100``).  ``tau`` is therefore taken
    from the two-parameter :func:`fit_inverse_sqrt_temperature` fit (which is
    the Haff form itself), and ``alpha`` is then obtained by a one-parameter
    least-squares fit in log space.  ``alpha = 2`` is Haff's prediction.
    """
    t = torch.as_tensor(times, dtype=torch.float64)
    temp = torch.as_tensor(temperatures, dtype=torch.float64)
    if t.numel() < 4 or t.numel() != temp.numel():
        raise ValueError("need at least 4 matched samples")
    if bool((temp <= 0).any()):
        raise ValueError("temperatures must be positive")
    if tau is None:
        tau = fit_inverse_sqrt_temperature(t, temp)["tau_haff"]
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be finite and positive")
    x = torch.log1p((t - t[0]) / tau)
    y = torch.log(temp)
    design = torch.stack((torch.ones_like(x), x), dim=1)
    solution = torch.linalg.lstsq(design, y).solution
    residual = y - design @ solution
    ss_res = float(residual.square().sum())
    ss_tot = float((y - y.mean()).square().sum())
    return {
        "alpha": -float(solution[1]),
        "T0_fit": math.exp(float(solution[0])),
        "tau": float(tau),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def maxwell_2d_diagnostics(velocities: torch.Tensor, mass: float) -> dict[str, float]:
    """Isotropy and 2-D Maxwellian shape checks for the instantaneous velocities.

    For a 2-D Maxwellian each Cartesian component is Gaussian, so the excess
    kurtosis is 0 (kurtosis 3) and ``<v_x**2> = <v_y**2> = T/m``.
    """
    v = torch.as_tensor(velocities, dtype=torch.float64)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("velocities must have shape (N, 2)")
    n = v.shape[0]
    var = v.var(dim=0, unbiased=False)
    temperature = float(mass * v.square().sum() / (2.0 * n))
    kurtosis = {}
    for axis, name in enumerate(("x", "y")):
        component = v[:, axis] - v[:, axis].mean()
        scale = component.square().mean()
        kurtosis[name] = float((component ** 4).mean() / scale ** 2) if scale > 0 else float("nan")
    return {
        "temperature": temperature,
        "variance_x": float(var[0]),
        "variance_y": float(var[1]),
        "anisotropy": float(abs(var[0] - var[1]) / max(float(var.mean()), 1e-300)),
        "kurtosis_x": kurtosis["x"],
        "kurtosis_y": kurtosis["y"],
    }