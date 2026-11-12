"""BPM micro-parameter three-target joint calibration: feasibility + identifiability.

Node P270W2C-N08 (task m9756).  Legacy context: ``bpm_calibration`` (m7272) built a
*statistical* calibration pipeline fed with externally supplied stress histories,
and ``bpm_identifiability`` (m9188) wired the bonded specimen to the 3-D engine and
audited the mass-scaling gate / Weibull size effect / friction regularisation one
target at a time.  Neither answered the question this module answers:

    can *one* micro-parameter vector
        theta = (k_b, eps_b, mu_s, mu_d, k_c, k_p)
    satisfy three macroscopic targets simultaneously
        UCS in [5, 15] MPa,  sigma_c/sigma_t in [2.5, 4],  Weibull m in band,
    and is theta *identifiable* from those three observables at all?

Contents
--------
``CalibrationVector``
    the six micro-parameters as a frozen dataclass with literature-plausible
    bounds, a dimensionless unit-cube coordinate ``u`` (log scaled for the
    stiffnesses) and the mapping onto ``MicroParameters`` / ``DEM3DConfig``.
    Mapping decisions (documented, not hidden): ``k_b`` = bond axial stiffness,
    ``eps_b`` = median tensile bond breaking strain, ``mu_s``/``mu_d`` = the
    static/dynamic pair of the engine's ``friction_regularization="dual"`` law,
    ``k_c`` = post-failure contact normal stiffness, ``k_p`` = particle-particle
    *tangential* contact stiffness (engine ``tangential_stiffness``).
``DiscGeometry`` / ``run_brazilian``
    a cylindrical bonded specimen (engine ``geometry="cylinder_xz"``) loaded
    diametrally by a kinematic line load -- a real Brazilian test: nominal
    ``sigma_t = 2P/(pi D t)`` *and*, independently, the discrete (Love-Weber)
    stress tensor averaged over the central region of the disc.
``virial_stress_tensor`` / ``region_stress``
    per-particle Love-Weber stress from bond forces plus post-failure contact
    normal forces; ``region_stress`` averages it over the central zone and
    returns the principal-stress ratio, which for the ideal Brazilian disc must
    approach Hondros' ``-3.0`` at the centre.
``evaluate_design_point``
    one theta -> (UCS, sigma_t, m): one uniaxial compression specimen, ``n_disc``
    Brazilian discs, and a five-thickness uniaxial tension family of which three
    thicknesses are fitted and two (``t = 0.05`` and ``t = 0.8 m``) are held out.
``response_jacobian`` / ``condition_diagnostics``
    numerical Jacobian ``d(UCS, sigma_t, m)/d(theta)`` in unit-cube coordinates,
    condition number, parameter correlation matrix, Belsley condition indices and
    the Brun collinearity index, plus the structural rank statement (3 observables
    vs 6 parameters can never exceed rank 3).
``nsga2``
    a dependency-free real-coded NSGA-II (SBX crossover, polynomial mutation,
    non-dominated sorting, crowding distance) used on the response-surface
    surrogate of the three objectives.
``feasibility_verdict`` / ``holdout_error`` / ``geometry_prior_check``
    the machine verdicts of the campaign acceptance criteria.

Units are SI (Pa, m, s, N) unless a name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .bpm_calibration import LITERATURE_UCS_RANGES
from .bpm_identifiability import LoadProtocol, MicroParameters, SpecimenGeometry
from .dem3d import DEM3DConfig, IceDEM3D
from .granular3d.stress import contact_outer

__all__ = [
    "COND_NUMBER_LIMIT",
    "CalibrationVector",
    "DiscGeometry",
    "DriveResult",
    "HOLDOUT_TOLERANCE",
    "HONDROS_CENTER_RATIO",
    "HONDROS_TOLERANCE",
    "PARAMETER_BOUNDS",
    "PARAMETER_FIELDS",
    "PARAMETER_LOG",
    "PARAMETER_NAMES",
    "QUASISTATIC_KE_LIMIT",
    "REFERENCE_VECTOR",
    "STRENGTH_RATIO_BAND",
    "UCS_BAND_MPA",
    "bootstrap_size_effect_m",
    "condition_diagnostics",
    "default_protocol",
    "evaluate_design_point",
    "feasibility_verdict",
    "geometry_prior_check",
    "holdout_error",
    "nsga2",
    "LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR",
    "pair_stress_state",
    "principal_stress_ratio",
    "protocol_prior_check",
    "region_stress",
    "response_jacobian",
    "run_axial",
    "run_brazilian",
    "size_effect_modulus",
    "specific_volume",
    "surrogate_quadratic",
    "virial_stress_tensor",
]

# ===========================================================================
# Target bands (literature anchors) and acceptance thresholds
# ===========================================================================

#: Uniaxial compressive strength band for freshwater ice at -10 C [MPa].
#: Provenance: ``bpm_calibration.LITERATURE_UCS_RANGES["freshwater_ice_-10C"]``
#: (Schulson & Duval 2009).
UCS_BAND_MPA: tuple[float, float] = LITERATURE_UCS_RANGES["freshwater_ice_-10C"]["ucsi_MPa"]

#: Compressive/tensile strength ratio band.  Provenance: Schulson & Duval (2009)
#: quote sigma_c/sigma_t = 3-5 for fresh columnar ice at -10 C, 1e-3 1/s;
#: Petrovic (2003) reviews 3-6.  The band carried by the campaign DAG (2.5-4) is
#: narrower than the literature range, so passing it is not free.
STRENGTH_RATIO_BAND: tuple[float, float] = (2.5, 4.0)

#: Hondros (1959) centre principal-stress ratio sigma_load/sigma_transverse of a
#: diametrally loaded disc under a line load: exactly -3.
HONDROS_CENTER_RATIO: float = -3.0

#: AC5 tolerance on the Hondros ratio (relative).
HONDROS_TOLERANCE: float = 0.15

#: AC2: a condition number above this makes the parameter/target combination
#: non-identifiable in the sense of Belsley (1980).
COND_NUMBER_LIMIT: float = 1.0e4

#: AC7: quasi-static gate on max(KE)/external work of the specimen runs.
QUASISTATIC_KE_LIMIT: float = 1.0e-3

#: AC4 tolerance on the held-out thickness extrapolation.
HOLDOUT_TOLERANCE: float = 0.15


# ===========================================================================
# The calibration vector
# ===========================================================================

PARAMETER_NAMES: tuple[str, ...] = ("k_b", "eps_b", "mu_s", "mu_d", "k_c", "k_p")
PARAMETER_FIELDS: tuple[str, ...] = (
    "bond_stiffness",
    "breaking_strain",
    "static_friction",
    "dynamic_friction",
    "contact_stiffness",
    "tangential_contact_stiffness",
)

#: (low, high) box of the calibration search: about one decade of bond strength
#: around the m9188 operating point while staying numerically stable at the
#: engine's conservative time step.
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "k_b": (5.0e4, 2.0e6),        # N/m, bond axial stiffness (1.6 decades)
    "eps_b": (2.0e-3, 5.0e-2),    # -, median tensile bond breaking strain (1.4 decades)
    "mu_s": (0.1, 0.9),           # -, static friction
    "mu_d": (0.05, 0.6),          # -, dynamic friction (<= mu_s)
    "k_c": (1.0e5, 3.0e6),        # N/m, post-failure contact normal stiffness
    "k_p": (1.0e5, 3.0e6),        # N/m, contact tangential stiffness
}

#: Which parameters are sampled on a log scale in the unit cube.
PARAMETER_LOG: tuple[bool, ...] = (True, False, False, False, True, True)


@dataclass(frozen=True)
class CalibrationVector:
    """One point of the six-dimensional BPM micro-parameter space."""

    bond_stiffness: float = 2.56e5      # k_b  [N/m]
    breaking_strain: float = 0.015      # eps_b [-]
    static_friction: float = 0.6        # mu_s [-]
    dynamic_friction: float = 0.4       # mu_d [-]
    contact_stiffness: float = 5.0e5    # k_c  [N/m]
    tangential_contact_stiffness: float = 5.0e5   # k_p [N/m]

    def __post_init__(self) -> None:
        for name, fname in zip(PARAMETER_NAMES, PARAMETER_FIELDS):
            value = float(getattr(self, fname))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if self.dynamic_friction > self.static_friction:
            raise ValueError("mu_d must not exceed mu_s (dual friction law)")

    # -- unit-cube coordinates --------------------------------------------
    def to_unit(self) -> np.ndarray:
        """Map to the unit cube ``[0, 1]^6`` (log for stiffness-type entries)."""
        out = np.empty(6, dtype=np.float64)
        for k, (name, fname, log) in enumerate(
                zip(PARAMETER_NAMES, PARAMETER_FIELDS, PARAMETER_LOG)):
            lo, hi = PARAMETER_BOUNDS[name]
            value = float(getattr(self, fname))
            if log:
                value, lo, hi = math.log(value), math.log(lo), math.log(hi)
            out[k] = (value - lo) / (hi - lo)
        return out

    @classmethod
    def from_unit(cls, u: Sequence[float]) -> "CalibrationVector":
        """Inverse of :meth:`to_unit`; ``mu_d`` is clipped at ``mu_s``."""
        u = np.asarray(u, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(u)):
            raise ValueError("unit coordinates must be finite")
        values: dict[str, float] = {}
        for k, (name, fname, log) in enumerate(
                zip(PARAMETER_NAMES, PARAMETER_FIELDS, PARAMETER_LOG)):
            lo, hi = PARAMETER_BOUNDS[name]
            if log:
                lo, hi = math.log(lo), math.log(hi)
            x = lo + float(min(max(u[k], 0.0), 1.0)) * (hi - lo)
            if log:
                x = math.exp(x)
            values[fname] = x
        mu_s, mu_d = values["static_friction"], values["dynamic_friction"]
        if mu_d > mu_s:
            # The dual friction law requires mu_d <= mu_s; the projection is the
            # documented clipping of the search box and is flagged.
            values["dynamic_friction"] = mu_s
        vector = cls(**values)
        object.__setattr__(vector, "was_projected", bool(mu_d > mu_s))
        return vector

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, fname))
                for name, fname in zip(PARAMETER_NAMES, PARAMETER_FIELDS)}

    # -- engine mapping ----------------------------------------------------
    def to_micro(self, *, bond_weibull_modulus: float = 6.0,
                 bond_seed: int = 0, density: float = 917.0) -> MicroParameters:
        """``MicroParameters`` carrying the parameters of the uniaxial path."""
        return MicroParameters(
            bond_stiffness=float(self.bond_stiffness),
            contact_stiffness=float(self.contact_stiffness),
            breaking_strain=float(self.breaking_strain),
            density=float(density),
            bond_weibull_modulus=float(bond_weibull_modulus),
            bond_seed=int(bond_seed),
            friction_coefficient=float(self.dynamic_friction),
        )

    def friction_kwargs(self) -> dict[str, Any]:
        """Engine kwargs of the dual static/dynamic friction law and ``k_p``."""
        return {
            "friction_regularization": "dual",
            "static_friction_coefficient": float(self.static_friction),
            "dynamic_friction_coefficient": float(self.dynamic_friction),
            "tangential_stiffness": float(self.tangential_contact_stiffness),
        }


#: Operating point inherited from the m9188 identifiability node -- the only
#: parameter set in this repository that passed the UCS literature cross-check.
REFERENCE_VECTOR: CalibrationVector = CalibrationVector()


# ===========================================================================
# Specimen geometry
# ===========================================================================

#: Lattice particle radius shared by every specimen of the campaign [m].
PARTICLE_RADIUS: float = 0.0125

#: Uniaxial specimen: D = 2R*cross_n = 0.1 m, gauge = 2*(n_layers-1)R = 0.2 m,
#: i.e. H/D = 2, the ASTM D7012 slenderness.
UCS_GEOMETRY: SpecimenGeometry = SpecimenGeometry(
    n_layers=9, cross_n=4, radius=PARTICLE_RADIUS)

#: Thickness family of the Weibull size-effect study [m]; the last two entries are
#: the held-out extrapolation targets of AC4 (never used in the fit).
SIZE_EFFECT_THICKNESSES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
FIT_THICKNESSES: tuple[float, ...] = (0.1, 0.2, 0.4)
HOLDOUT_THICKNESSES: tuple[float, ...] = (0.05, 0.8)


@dataclass(frozen=True)
class DiscGeometry:
    """Brazilian disc: a cylinder (engine ``cylinder_xz``) with a diameter of
    ``n_lattice`` lattice spacings and ``n_layers`` layers along the cylinder axis.

    ``nx = nz = n_lattice`` and ``ny = n_layers``; the engine keeps the lattice
    sites inside the circle, so the physical diameter is ``n_lattice * 2R`` and
    the physical thickness is ``n_layers * 2R``.
    """

    n_lattice: int = 7                    # 7 -> D = 0.175 m
    n_layers: int = 4                     # 4 -> t = 0.1 m
    radius: float = PARTICLE_RADIUS

    def __post_init__(self) -> None:
        if self.n_lattice < 5:
            raise ValueError("disc needs n_lattice >= 5 for a resolvable centre")
        if self.n_layers < 2:
            raise ValueError("disc needs n_layers >= 2")

    @property
    def diameter_m(self) -> float:
        return self.n_lattice * 2.0 * self.radius

    @property
    def thickness_m(self) -> float:
        return self.n_layers * 2.0 * self.radius

    @property
    def ligament_m2(self) -> float:
        """Loaded ligament area ``D * t`` of the Brazilian formula."""
        return self.diameter_m * self.thickness_m

    @property
    def stress_scale_m2_per_N(self) -> float:
        """``sigma_t = 2P / (pi D t)``."""
        return 2.0 / (math.pi * self.diameter_m * self.thickness_m)

    def config(self, vector: CalibrationVector, *, mass_scaling: float = 100.0,
               decay_steps: int = 15, bond_seed: int = 0,
               bond_weibull_modulus: float = 6.0,
               density: float = 917.0) -> DEM3DConfig:
        """``DEM3DConfig`` of the disc, including quasi-static local damping."""
        base = dict(
            nx=self.n_lattice, ny=self.n_layers, nz=self.n_lattice,
            radius=self.radius, density=density * float(mass_scaling),
            bond_stiffness=float(vector.bond_stiffness),
            contact_stiffness=float(vector.contact_stiffness),
            breaking_strain=float(vector.breaking_strain),
        )
        mass = (density * float(mass_scaling) * 4.0 / 3.0
                * math.pi * self.radius ** 3)
        # Two passes: the conservative time step depends on *every* stiffness in
        # the config (``k_p`` included), so it is read off the assembled config
        # rather than off a partial one -- otherwise a stiff tangential contact
        # would silently violate the stability limit.
        probe = DEM3DConfig(
            **base,
            shear_breaking_strain=2.0 * float(vector.breaking_strain),
            contact_damping=0.0,
            contact_damping_model="absolute",
            tool_gap=100.0,
            bond_strength_weibull_modulus=float(bond_weibull_modulus),
            bond_strength_seed=int(bond_seed),
            geometry="cylinder_xz",
            fix_x_edges=False,
            fix_bottom=False,
            friction_coefficient=float(vector.dynamic_friction),
            **vector.friction_kwargs(),
        )
        dt = probe.recommended_dt
        return replace(probe, dt=dt, drag=mass / (float(decay_steps) * dt))


def axial_config(vector: CalibrationVector, geometry: SpecimenGeometry, *,
                 mass_scaling: float = 100.0, decay_steps: int = 15,
                 bond_seed: int = 0, bond_weibull_modulus: float = 6.0,
                 density: float = 917.0) -> DEM3DConfig:
    """``DEM3DConfig`` of the uniaxial specimen with the campaign parameters."""
    micro = vector.to_micro(bond_weibull_modulus=bond_weibull_modulus,
                            bond_seed=bond_seed, density=density)
    cfg = replace(micro.engine_config(geometry, mass_scaling=mass_scaling,
                                      decay_steps=decay_steps),
                  **vector.friction_kwargs())
    # ``k_p`` (tangential contact stiffness) enters the conservative time step, so
    # the step and the quasi-static damping are re-derived from the final config.
    dt = cfg.recommended_dt
    mass = (density * float(mass_scaling) * 4.0 / 3.0
            * math.pi * geometry.radius ** 3)
    return replace(cfg, dt=dt, drag=mass / (float(decay_steps) * dt))


# ===========================================================================
# Discrete stress field (Love-Weber virial)
# ===========================================================================

#: Per-contact tensor convention of the shared stress instrument
#: (:mod:`tensordem.granular3d.stress`): ``l (x) f`` with ``l = x_j - x_i`` the
#: branch vector and ``f`` the force *on* ``i`` from ``j``.

#: Volume per particle of the model lattice: the simple-cubic cell ``(2R)^3``.
#: It is the historical normalisation of this instrument and it is *not* the
#: specific volume of a finite specimen, so it does not reproduce the applied
#: boundary stress (see :func:`specific_volume` and the m16734 audit).
LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR: float = 8.0


def specific_volume(volume: float, n_particles: int) -> float:
    """Volume per particle ``V / N`` [m^3] of a specimen of volume ``volume``.

    Use this as ``volume_per_particle`` when the absolute stress level matters.
    The gauge volume (``A * gauge``) is the one that makes the volume average of
    :func:`virial_stress_tensor` reproduce the independent plane-cut traction of
    a homogeneously strained lattice (verified in
    ``scripts/verify_bpm_virial_absolute_m16734.py``).
    """
    volume = float(volume)
    n_particles = int(n_particles)
    if not (math.isfinite(volume) and volume > 0.0):
        raise ValueError("volume must be finite and positive")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    return volume / float(n_particles)


def pair_stress_state(engine: IceDEM3D) -> dict[str, Any]:
    """Pair list, branch vectors and pair forces of the discrete stress instrument.

    The force law of the *stress* instrument is written down here and nowhere
    else:

    * **bonds** -- every *alive* bond carries the axial force ``k_b (L - L0)``,
      tension positive (a positive force pulls ``i`` towards ``j``);
    * **contacts** -- pairs that are *not* bonded (broken bond, or a pair that was
      never bonded) with a positive overlap ``2R - L`` carry the linear repulsive
      normal force ``k_c * overlap``.

    Tangential (Cundall-Strack) and rolling forces are **not** included -- the
    engine keeps that history private -- so the tensor built from this state is
    symmetric by construction and cannot show the friction asymmetry
    ``sigma_ab - sigma_ba = sum_c |l| f_t / A``.  See the module notes.

    Returns a dict with the numpy ``branch`` ``(m, 3)`` (``x_j - x_i``),
    ``force_on_i`` ``(m, 3)``, the pair indices and the two branch masks.
    """
    pos = engine.positions
    i, j = engine.pairs.T
    direction = pos[j] - pos[i]
    length = torch.linalg.vector_norm(direction, dim=1).clamp_min(
        torch.finfo(pos.dtype).eps)
    unit = direction / length[:, None]

    extension = length - engine.rest_lengths
    f_bond = float(engine.config.bond_stiffness) * extension
    f_bond = torch.where(engine.alive, f_bond, torch.zeros_like(f_bond))

    overlap = 2.0 * float(engine.config.radius) - length
    contact = (~engine.alive) & (overlap > 0.0)
    f_contact = torch.where(contact,
                            float(engine.config.contact_stiffness) * overlap,
                            torch.zeros_like(overlap))
    # bond: tension pulls i towards j (+unit); contact: repulsion pushes i away
    force_on_i = (f_bond - f_contact)[:, None] * unit
    return {
        "i": i.detach().cpu().numpy().astype(np.int64),
        "j": j.detach().cpu().numpy().astype(np.int64),
        "branch": direction.detach().cpu().numpy(),
        "force_on_i": force_on_i.detach().cpu().numpy(),
        "bonded": engine.alive.detach().cpu().numpy().astype(bool),
        "contact": contact.detach().cpu().numpy().astype(bool),
        "n_particles": int(pos.shape[0]),
    }


def virial_stress_tensor(engine: IceDEM3D, *,
                         volume_per_particle: float | None = None,
                         scatter: str = "half") -> torch.Tensor:
    """Per-particle Love-Weber stress tensor ``(N, 3, 3)`` [Pa].

    The per-contact tensor ``l (x) f`` is produced by the shared instrument
    :func:`tensordem.granular3d.stress.contact_outer` from
    :func:`pair_stress_state` -- this function owns no private pair algebra any
    more (m16734).

    ``scatter`` -- how a pair is distributed over its two particles:

    * ``"half"`` (default, correct): ``+/- 1/2 l (x) f`` to each end, so the
      volume-weighted sum over all particles is exactly the virial sum
      ``sum_c l (x) f`` (verified to the last bit in
      ``tests/test_bpm_virial_absolute_m16734.py``);
    * ``"legacy"``: the full ``l (x) f`` to *both* ends, i.e. every pair counted
      twice.  This was the historical behaviour; it makes the region average a
      factor **2** too large.  Kept only to reproduce pre-m16734 numbers.

    ``volume_per_particle`` -- the specific volume used to normalise [m^3].
    ``None`` means the lattice cell ``8 R^3``, the historical value; pass
    :func:`specific_volume` (gauge volume / N) when the absolute level is used,
    because ``8 R^3`` is not the specific volume of a finite specimen.

    Sign convention: tension positive.  Bond and contact normal forces only (see
    :func:`pair_stress_state`), so the tensor is symmetric.
    """
    if scatter not in ("half", "legacy"):
        raise ValueError("scatter must be 'half' or 'legacy'")
    state = pair_stress_state(engine)
    outer = contact_outer(state["branch"], state["force_on_i"])   # (m, 3, 3)
    weight = 0.5 if scatter == "half" else 1.0
    if volume_per_particle is None:
        volume = LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR * float(
            engine.config.radius) ** 3
        volume = float(volume)
    else:
        volume = float(volume_per_particle)
        if not (math.isfinite(volume) and volume > 0.0):
            raise ValueError("volume_per_particle must be finite and positive")

    dtype, device = engine.positions.dtype, engine.positions.device
    per_contact = torch.from_numpy(np.ascontiguousarray(outer * weight)).to(
        dtype=dtype, device=device)
    stress = torch.zeros((int(state["n_particles"]), 3, 3), dtype=dtype,
                         device=device)
    index = torch.from_numpy(state["i"]).to(device=device)
    stress.index_add_(0, index, per_contact)
    index = torch.from_numpy(state["j"]).to(device=device)
    stress.index_add_(0, index, per_contact)
    return stress / volume


def principal_stress_ratio(eigenvalues: Sequence[float] | torch.Tensor) -> float:
    """``sigma_min / sigma_max`` of a principal-stress set, ``nan`` if degenerate.

    For the Brazilian disc centre ``sigma_max`` is the tensile (transverse) and
    ``sigma_min`` the compressive (load-direction) principal stress, so Hondros'
    ``-3`` is this ratio.  A non-positive ``sigma_max`` (no tension at all) is
    returned as ``nan``: the configuration is then outside the Brazilian regime
    and a ratio would be meaningless.
    """
    values = [float(v) for v in eigenvalues]
    if len(values) < 2:
        return float("nan")
    sigma_max = max(values)
    sigma_min = min(values)
    if sigma_max <= 0.0:
        return float("nan")
    return sigma_min / sigma_max


def region_stress(engine: IceDEM3D, *, center: Sequence[float] = (0.0, 0.0, 0.0),
                  radius: float, axes: Sequence[int] = (0, 2),
                  volume_per_particle: float | None = None,
                  scatter: str = "half") -> dict[str, Any]:
    """Love-Weber stress averaged over the particles inside ``radius`` of ``center``.

    Returns the averaged tensor (full 3x3 and the in-plane 2x2 block on ``axes``,
    default the x-z plane of the disc), its in-plane principal stresses
    (descending), the principal-stress ratio ``sigma_min / sigma_max`` and the
    particle count.  Averaging over a *region* rather than one lattice site is the
    point of AC5: the lattice-level stress is noisy, the region average is the
    continuum quantity comparable with Hondros' solution.
    """
    stress = virial_stress_tensor(engine, volume_per_particle=volume_per_particle,
                                  scatter=scatter)
    centre = torch.tensor(list(center), dtype=engine.positions.dtype,
                          device=engine.device)
    delta = engine.positions - centre
    distance = torch.linalg.vector_norm(delta, dim=1)
    mask = distance <= float(radius)
    count = int(mask.sum())
    if count == 0:
        raise ValueError("empty stress-averaging region")
    block = stress[mask].mean(dim=0)
    axes = list(axes)
    in_plane = block[axes][:, axes]
    eigenvalues, eigenvectors = torch.linalg.eigh(in_plane)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    sigma_max = float(eigenvalues[0])
    sigma_min = float(eigenvalues[1])
    return {
        "n_particles": count,
        "radius_m": float(radius),
        "tensor_3x3": block.detach().cpu().numpy().tolist(),
        "tensor_in_plane": in_plane.detach().cpu().numpy().tolist(),
        "principal_in_plane": [sigma_max, sigma_min],
        "principal_vectors_in_plane": eigenvectors.detach().cpu().numpy().tolist(),
        "ratio_compressive_to_tensile": principal_stress_ratio(eigenvalues),
        "hydrostatic": float(torch.trace(block) / 3.0),
    }


# ===========================================================================
# The staged (load-relax) driver shared by the axial and the disc specimen
# ===========================================================================

@dataclass
class DriveResult:
    """Read-out history of one displacement-controlled staged run."""

    load: np.ndarray                  # [N], magnitude
    strain: np.ndarray                # [-]
    stress: np.ndarray                # [Pa], nominal
    ke_ratio: float                   # max(KE)/work over the read-outs
    ke_ratio_peak: float              # max over every step
    work_J: float
    n_steps: int
    broken_bonds: int
    n_bonds: int
    probes: list[dict[str, Any]] = field(default_factory=list)
    ke_readout: np.ndarray = field(default_factory=lambda: np.zeros(0))
    broken_history: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def peak_stress(self) -> float:
        return float(np.max(self.stress)) if len(self.stress) else float("nan")

    @property
    def peak_load(self) -> float:
        return float(np.max(self.load)) if len(self.load) else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_stress_Pa": self.peak_stress,
            "peak_load_N": self.peak_load,
            "ke_ratio": self.ke_ratio,
            "ke_ratio_peak": self.ke_ratio_peak,
            "work_J": self.work_J,
            "n_steps": self.n_steps,
            "broken_bonds": self.broken_bonds,
            "n_bonds": self.n_bonds,
            "n_readouts": int(len(self.stress)),
            "ke_readout_max": float(np.max(self.ke_readout)) if len(self.ke_readout) else float("nan"),
        }


def _run_drive(engine: IceDEM3D, drive: torch.Tensor, support: torch.Tensor, *,
               protocol: LoadProtocol, sign: float, stress_scale: float,
               strain_denominator: float,
               probe: Callable[[], dict[str, Any]] | None = None) -> DriveResult:
    """Displacement-controlled staged loading for an arbitrary specimen/grip pair.

    The driven set moves at ``v`` during the ramp phase and at zero during the hold
    phase; the load is read at the end of every hold, with the viscous drag of the
    driven particles subtracted analytically.  ``stress_scale`` converts the
    reaction to the nominal stress of the specimen (``1/A`` for the uniaxial
    specimen, ``2/(pi D t)`` for the disc).  Assumption inherited from
    ``bpm_identifiability.run_specimen``: every particle has the same mass
    ``engine.config.mass``, so the total kinetic energy is ``0.5 * m * sum(v^2)``.
    """
    mass = float(engine.config.mass)
    drag = float(engine.config.drag)
    dt = float(engine.dt)
    n_drive = int(drive.sum())
    initial_drive_z = float(engine.initial_positions[drive, 2].mean())

    ramp_steps = int(protocol.ramp_steps)
    eps_inc = float(protocol.target_strain) / int(protocol.increments)
    v_ramp = eps_inc * float(strain_denominator) / (ramp_steps * dt)
    v_ramp = v_ramp if sign > 0 else -v_ramp
    vel_ramp = torch.tensor([0.0, 0.0, v_ramp], dtype=engine.dtype, device=engine.device)
    vel_hold = torch.zeros(3, dtype=engine.dtype, device=engine.device)

    def _load(force: torch.Tensor, v: float) -> float:
        raw = sign * (-float(force[drive, 2].sum()))
        return sign * raw - drag * abs(v) * n_drive

    def _strain() -> float:
        return abs(float(engine.positions[drive, 2].mean()) - initial_drive_z) / float(
            strain_denominator)

    def _kinetic() -> float:
        return 0.5 * mass * float(engine.velocities.square().sum())

    for _ in range(int(protocol.settle_steps)):
        engine.step()

    loads: list[float] = []
    strains: list[float] = []
    stresses: list[float] = []
    probes: list[dict[str, Any]] = []
    ke_readouts: list[float] = []
    broken_history: list[int] = []
    work = 0.0
    n_steps = 0

    phases: list[tuple[torch.Tensor, float, bool, int]] = []
    for _ in range(int(protocol.increments)):
        phases.append((vel_ramp, v_ramp, False, ramp_steps))
        phases.append((vel_hold, 0.0, True, int(protocol.hold_steps)))

    stopped = False
    for vel_vec, v, read_out, n_phase in phases:
        for _ in range(n_phase):
            force, _ = engine.forces()
            load = _load(force, v)
            work += abs(load) * abs(v) * dt
            acc = force / mass
            acc[support] = 0.0
            engine.velocities += acc * dt
            engine.velocities[support] = 0.0
            engine.velocities[drive] = vel_vec
            engine.positions += engine.velocities * dt
            engine.positions[support] = engine.initial_positions[support]
            engine.time += dt
            engine.step_count += 1
            n_steps += 1
            if _strain() >= protocol.target_strain or n_steps >= protocol.max_steps:
                stopped = True
                break
            if not bool(torch.isfinite(engine.positions).all()):
                raise FloatingPointError("non-finite specimen state")
        if read_out:
            load_now = _load(engine.forces()[0], 0.0)
            loads.append(abs(load_now))
            strains.append(_strain())
            stresses.append(abs(load_now) * float(stress_scale))
            ke_readouts.append(_kinetic() / work if work > 0 else float("inf"))
            broken_history.append(int(engine.broken_bonds))
            if probe is not None:
                probes.append(probe())
        if stopped:
            break

    ke_readout = np.asarray(ke_readouts, dtype=np.float64)
    # The quasi-static gate is evaluated on the read-out instants *up to the peak
    # load*, after the drive start-up transient (the first 20 % of the read-outs):
    #  * during a constant-ramp phase the driven layer trivially carries KE, so a
    #    per-step maximum would measure the boundary condition, not the specimen;
    #  * after the peak the fragments are ballistic, so a post-failure maximum
    #    measures debris, not the loading path that produced the strength.
    stresses_array = np.asarray(stresses, dtype=np.float64)
    if stresses_array.size:
        peak = int(np.argmax(stresses_array))
    else:
        peak = ke_readout.size - 1
    skip = int(0.2 * (peak + 1))
    window = ke_readout[skip:peak + 1] if ke_readout.size else ke_readout
    measured = float(np.max(window)) if window.size else float("nan")
    return DriveResult(
        load=np.asarray(loads, dtype=np.float64),
        strain=np.asarray(strains, dtype=np.float64),
        stress=np.asarray(stresses, dtype=np.float64),
        ke_ratio=float(measured),
        ke_ratio_peak=float(np.max(ke_readout)) if ke_readout.size else float("nan"),
        work_J=float(work),
        n_steps=int(n_steps),
        broken_bonds=int(engine.broken_bonds),
        n_bonds=int(len(engine.pairs)),
        probes=probes,
        ke_readout=ke_readout,
        broken_history=np.asarray(broken_history, dtype=np.int64),
    )

# ===========================================================================
# Uniaxial specimen (UCS and the size-effect family)
# ===========================================================================

def build_axial(vector: CalibrationVector, geometry: SpecimenGeometry, *,
                mass_scaling: float = 100.0, decay_steps: int = 15,
                bond_seed: int = 0, bond_weibull_modulus: float = 6.0
                ) -> tuple[IceDEM3D, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Cuboid bonded specimen with the campaign parameters and its kinematic grips."""
    cfg = axial_config(vector, geometry, mass_scaling=mass_scaling,
                       decay_steps=decay_steps, bond_seed=bond_seed,
                       bond_weibull_modulus=bond_weibull_modulus)
    engine = IceDEM3D(cfg)
    z = engine.positions[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    bottom = z < zmin + 0.5 * 2.0 * geometry.radius
    top = z > zmax - 0.5 * 2.0 * geometry.radius
    engine.fixed = bottom.clone()
    meta = {
        "gauge_m": float(zmax - zmin),
        "area_m2": float(geometry.area),
        "diameter_m": float(2.0 * geometry.radius * geometry.cross_n),
        "n_particles": int(engine.positions.shape[0]),
        "n_bonds": int(len(engine.pairs)),
        "n_top": int(top.sum()),
        "n_bottom": int(bottom.sum()),
    }
    return engine, top, bottom, meta


def adapt_protocol(protocol: LoadProtocol, n_bond_layers: int, *,
                   breaking_strain: float, safety: float = 0.75) -> LoadProtocol:
    """Make the displacement increment safe for the specimen's bond layers.

    The driven layer is the top layer only, so one increment moves it by
    ``eps_inc * gauge`` before the rest of the specimen follows; the outermost bond
    then carries that increment as an extension.  The increment is admissible when
    that extension stays below ``safety * eps_b * 2R`` (the bond breaking extension),
    i.e. when

        increments >= target_strain * (n_layers - 1) / (safety * eps_b).

    Without this scaling a long specimen tears its top bond layer in the first ramp
    and reports a strength of exactly zero (measured on t = 0.4 and t = 0.8 m in the
    first campaign of this node).
    """
    gauge_layers = max(1, int(n_bond_layers))
    required = float(protocol.target_strain) * gauge_layers / (
        float(safety) * float(breaking_strain))
    increments = int(min(200, max(int(protocol.increments), math.ceil(required))))
    eps_inc = float(protocol.target_strain) / max(increments, 1)
    # A second, weaker requirement: the ramp must be slow enough that the outermost
    # bond relaxes between steps (the specimen relaxes over ``decay_steps`` steps).
    transient = (float(protocol.decay_steps) * eps_inc * gauge_layers
                 / float(protocol.ramp_steps))
    ramp_steps = int(protocol.ramp_steps)
    if transient > float(safety) * float(breaking_strain):
        ramp_steps = int(math.ceil(ramp_steps * transient
                                   / (float(safety) * float(breaking_strain))))
    if increments == int(protocol.increments) and ramp_steps == int(protocol.ramp_steps):
        return protocol
    return replace(protocol, increments=increments, ramp_steps=ramp_steps)


def run_axial(vector: CalibrationVector, geometry: SpecimenGeometry,
              protocol: LoadProtocol, *, bond_seed: int = 0) -> DriveResult:
    """One displacement-controlled uniaxial run.

    ``protocol.mode = 'compression'`` gives the UCS, ``'tension'`` the tensile
    strength of one specimen of the size-effect family.
    """
    protocol = adapt_protocol(protocol, int(geometry.n_layers) - 1,
                              breaking_strain=float(vector.breaking_strain))
    engine, top, bottom, meta = build_axial(
        vector, geometry, mass_scaling=protocol.mass_scaling,
        decay_steps=protocol.decay_steps, bond_seed=bond_seed)
    sign = 1.0 if protocol.mode == "tension" else -1.0
    return _run_drive(engine, top, bottom, protocol=protocol, sign=sign,
                      stress_scale=1.0 / meta["area_m2"],
                      strain_denominator=meta["gauge_m"])


# ===========================================================================
# Brazilian disc
# ===========================================================================

def build_disc(vector: CalibrationVector, geometry: DiscGeometry, *,
               mass_scaling: float = 100.0, decay_steps: int = 15,
               bond_seed: int = 0, bond_weibull_modulus: float = 6.0
               ) -> tuple[IceDEM3D, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build the disc and its kinematic grips.

    The grips are the *edge lines* of the cylinder: the particles within ``0.6``
    lattice spacings of the top / bottom extremity, across all ``y`` layers -- a
    line load along the cylinder axis, which is exactly the loading case of
    Hondros' solution (a point load in the plane, uniform across the thickness).
    With ``n_lattice = 7`` the loaded arc half-angle is `arcsin(0.6/3) = 11.5 deg`,
    i.e. ``2 alpha ~ 23 deg``; the report quantifies the effect of the finite arc.
    """
    cfg = geometry.config(vector, mass_scaling=mass_scaling,
                          decay_steps=decay_steps, bond_seed=bond_seed,
                          bond_weibull_modulus=bond_weibull_modulus)
    engine = IceDEM3D(cfg)
    spacing = 2.0 * geometry.radius
    x = engine.positions[:, 0]
    z = engine.positions[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    grip = 0.6 * spacing
    support = (z <= zmin + grip) & (x.abs() <= grip)
    drive = (z >= zmax - grip) & (x.abs() <= grip)
    if int(support.sum()) == 0 or int(drive.sum()) == 0:
        raise ValueError("disc grips are empty")
    engine.fixed = support.clone()
    meta: dict[str, Any] = {
        "diameter_m": geometry.diameter_m,
        "thickness_m": geometry.thickness_m,
        "ligament_m2": geometry.ligament_m2,
        "n_particles": int(engine.positions.shape[0]),
        "n_bonds": int(len(engine.pairs)),
        "n_support": int(support.sum()),
        "n_drive": int(drive.sum()),
        "centre_m": (0.0, 0.0, 0.5 * (zmin + zmax)),
        "stress_scale_m2_per_N": geometry.stress_scale_m2_per_N,
        "arc_half_angle_rad": float(math.asin(min(1.0, grip / (0.5 * geometry.diameter_m)))),
    }
    return engine, drive, support, meta


def run_brazilian(vector: CalibrationVector, geometry: DiscGeometry,
                  protocol: LoadProtocol, *, bond_seed: int = 0,
                  probe_region_ratio: float = 0.25,
                  probe: bool = True) -> dict[str, Any]:
    """Diametral compression of the disc: tensile strength + centre stress field.

    ``sigma_t = 2P/(pi D t)`` from the peak load, and -- when ``probe`` is true --
    the centrally averaged Love-Weber stress tensor at every read-out, whose
    principal-stress ratio is comparable with Hondros' ``-3``.
    """
    engine, drive, support, meta = build_disc(
        vector, geometry, mass_scaling=protocol.mass_scaling,
        decay_steps=protocol.decay_steps, bond_seed=bond_seed)
    centre = meta["centre_m"]
    region_radius = float(probe_region_ratio) * geometry.diameter_m
    # Specific volume of the *cylinder* lattice (m16734): the probe reports an
    # absolute stress level, so it must be normalised by the specimen's specific
    # volume rather than by the lattice cell ``8 R^3`` (the two differ by 1.33x
    # on this lattice).  The AC5 ratio is invariant under either choice.
    volume_disc = math.pi * (geometry.diameter_m ** 2) / 4.0 * geometry.thickness_m
    volume_per_particle = specific_volume(volume_disc, meta["n_particles"])
    meta["volume_per_particle_m3"] = volume_per_particle

    def _probe() -> dict[str, Any]:
        return region_stress(engine, center=centre, radius=region_radius,
                             volume_per_particle=volume_per_particle)

    result = _run_drive(engine, drive, support, protocol=protocol, sign=-1.0,
                        stress_scale=meta["stress_scale_m2_per_N"],
                        strain_denominator=geometry.diameter_m,
                        probe=_probe if probe else None)
    payload: dict[str, Any] = dict(meta)
    payload.update(result.as_dict())
    payload["sigma_t_Pa"] = result.peak_stress
    payload["peak_index"] = int(np.argmax(result.stress)) if len(result.stress) else -1
    payload["stress_history_Pa"] = result.stress.tolist()
    payload["strain_history"] = result.strain.tolist()
    payload["load_history_N"] = result.load.tolist()
    payload["broken_history"] = result.broken_history.tolist()
    payload["ke_readout_history"] = result.ke_readout.tolist()
    if probe and result.probes:
        peak_index = int(np.argmax(result.stress))
        payload["stress_field_at_peak"] = result.probes[peak_index]
        payload["stress_field_first"] = result.probes[0]
        payload["hoop_ratio_at_peak"] = float(
            result.probes[peak_index]["ratio_compressive_to_tensile"])
        lo = max(0, peak_index - 3)
        band = [float(p["ratio_compressive_to_tensile"]) for p in result.probes[lo:peak_index + 1]]
        band = [r for r in band if math.isfinite(r)]
        payload["hoop_ratio_band"] = [min(band), max(band)] if band else []
        payload["hoop_ratio_history"] = [
            float(p["ratio_compressive_to_tensile"]) for p in result.probes]
        payload["sigma_transverse_history_Pa"] = [
            float(p["principal_in_plane"][0]) for p in result.probes]
        payload["sigma_load_history_Pa"] = [
            float(p["principal_in_plane"][1]) for p in result.probes]
        payload["region_n_particles"] = [
            int(p["n_particles"]) for p in result.probes]
    return payload


# ===========================================================================
# Weibull size effect
# ===========================================================================

def size_effect_modulus(thicknesses: Sequence[float],
                        mean_strengths: Sequence[float]) -> dict[str, float]:
    """Fit ``sigma_t(t) = sigma_0 * t**(-1/m)`` and return ``m``.

    Volume scaling.  Every specimen of the family keeps its cross-section fixed and
    only its *length* (thickness) changes, so the loaded volume is ``V ~ t`` and
    Weibull weakest-link theory gives ``sigma_t ~ V**(-1/m) ~ t**(-1/m)``.  This is
    a deliberate, documented deviation from the legacy
    ``bpm_calibration``/``bpm_identifiability`` convention ``t**(-2/m)`` (report
    section "departures"): ``-2/m`` is the scaling of a *surface* flaw population or
    of a Brazilian-disc thickness gradient, and applying it to a volume-scaled
    family silently doubles the inferred modulus.
    """
    t = np.asarray(thicknesses, dtype=np.float64)
    s = np.asarray(mean_strengths, dtype=np.float64)
    if t.size != s.size or t.size < 2:
        raise ValueError("need at least two (thickness, strength) pairs")
    if not np.all(t > 0.0) or not np.all(s > 0.0):
        raise ValueError("thicknesses and strengths must be positive")
    slope, intercept = np.polyfit(np.log(t), np.log(s), 1)
    if not slope < 0.0:
        raise ValueError(
            "size effect is not decreasing in thickness; m is not identifiable")
    return {
        "m": float(-1.0 / slope),
        "exponent": float(slope),
        "sigma_ref_Pa": float(math.exp(intercept)),
        "n_points": int(t.size),
        "r2": float(np.corrcoef(np.log(t), np.log(s))[0, 1] ** 2),
    }


def bootstrap_size_effect_m(samples: dict[float, Sequence[float]], *,
                            n_bootstrap: int = 400, seed: int = 20260101,
                            confidence: float = 0.95) -> dict[str, Any]:
    """Bootstrap 95 % CI of the size-effect modulus by resampling specimens.

    Within each thickness the replicates are resampled with replacement and the
    power law is refitted on the resampled thickness means, so the CI carries the
    specimen-to-specimen scatter and the leverage of the extreme thicknesses.
    """
    arrays = {float(t): np.asarray(samples[t], dtype=np.float64)
              for t in samples if len(samples[t]) >= 2}
    if len(arrays) < 2:
        raise ValueError("need replicate data at >= 2 thicknesses")
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        t_fit: list[float] = []
        s_fit: list[float] = []
        for t, values in arrays.items():
            resample = values[rng.integers(0, len(values), len(values))]
            t_fit.append(t)
            s_fit.append(float(np.mean(resample)))
        try:
            draws.append(size_effect_modulus(t_fit, s_fit)["m"])
        except ValueError:
            continue
    if not draws:
        raise ValueError("bootstrap produced no identifiable resample")
    draws_arr = np.asarray(draws)
    alpha = 0.5 * (1.0 - float(confidence))
    point = size_effect_modulus(list(arrays), [float(np.mean(v)) for v in arrays.values()])
    return {
        "m": point["m"],
        "sigma_ref_Pa": point["sigma_ref_Pa"],
        "r2": point["r2"],
        "ci_low": float(np.quantile(draws_arr, alpha)),
        "ci_high": float(np.quantile(draws_arr, 1.0 - alpha)),
        "ci_width": float(np.quantile(draws_arr, 1.0 - alpha)
                          - np.quantile(draws_arr, alpha)),
        "n_bootstrap": int(len(draws)),
        "thicknesses_m": sorted(arrays),
        "strength_mean_Pa": {f"{t:.3f}": float(np.mean(arrays[t])) for t in sorted(arrays)},
        "strength_n": {f"{t:.3f}": int(len(arrays[t])) for t in sorted(arrays)},
    }


# ===========================================================================
# Response-surface point
# ===========================================================================

def default_protocol(*, mode: str) -> LoadProtocol:
    """Campaign load protocol (staged/load-relax, quasi-static at beta = 100)."""
    return LoadProtocol(
        mode=mode,
        protocol="staged",
        mass_scaling=100.0,
        target_strain=0.02,
        increments=8,
        ramp_steps=12,
        hold_steps=60,
        settle_steps=40,
        decay_steps=15,
        max_steps=20_000,
    )


def protocol_prior_check(protocol: LoadProtocol, *,
                         thicknesses: Sequence[float] = SIZE_EFFECT_THICKNESSES,
                         radius: float = PARTICLE_RADIUS,
                         breaking_strain: float = CalibrationVector().breaking_strain,
                         safety: float = 0.75) -> dict[str, Any]:
    """Kinematic-drive consistency of the load protocol (the trap of this node).

    The driven layer is the *top layer only*, so during a ramp it advances by
    ``eps_inc * gauge`` before the rest of the specimen follows.  If that transient
    exceeds the bond breaking extension, the top bond layer tears during the ramp
    and the read-out load collapses to zero -- the specimen reports a strength of
    exactly 0 instead of failing in the gauge.  The quasi-static relaxation length
    is ``decay_steps`` steps, so the transient bond strain is about

        eps_transient ~ decay_steps * eps_inc * (n_layers - 1)

    and the protocol is admissible when ``eps_transient < safety * eps_b`` for every
    specimen of the family (``safety = 1``: the transient may use the whole breaking
    strain of the outermost bond, but must not tear it -- a smaller ``safety`` is the
    conservative choice and the measured margin is reported by the campaign).  This is a *numerical* prior, not physics: it says the
    load can be read at all.
    """
    rows = []
    for thickness in thicknesses:
        layers = max(2, int(round(float(thickness) / (2.0 * radius))) + 1)
        gauge_layers = layers - 1
        adapted = adapt_protocol(protocol, gauge_layers,
                                 breaking_strain=breaking_strain, safety=safety)
        eps_inc = float(adapted.target_strain) / int(adapted.increments)
        # (i) increment extension of the outermost bond layer, in strain units
        increment_strain = eps_inc * gauge_layers
        # (ii) transient extra extension during one ramp (relaxation over decay_steps)
        transient = (float(adapted.decay_steps) * eps_inc * gauge_layers
                     / float(adapted.ramp_steps))
        limit = float(safety) * float(breaking_strain)
        rows.append({
            "thickness_m": float(thickness),
            "n_layers": layers,
            "increments": int(adapted.increments),
            "ramp_steps": int(adapted.ramp_steps),
            "increment_bond_strain": increment_strain,
            "transient_bond_strain": transient,
            "limit": limit,
            "ok": bool(increment_strain < limit and transient < limit),
        })
    return {"specimens": rows, "passed": bool(all(row["ok"] for row in rows)),
            "safety": float(safety), "breaking_strain": float(breaking_strain)}


def evaluate_design_point(vector: CalibrationVector, *,
                          disc_geometry: DiscGeometry | None = None,
                          n_disc: int = 0,
                          n_size_reps: int = 4,
                          n_holdout_reps: int = 3,
                          fit_thicknesses: Sequence[float] = FIT_THICKNESSES,
                          holdout_thicknesses: Sequence[float] = HOLDOUT_THICKNESSES,
                          bond_seed: int = 0,
                          probe_disc: bool = False) -> dict[str, Any]:
    """One theta -> the three macroscopic responses plus all raw bookkeeping.

    Runs (all displacement controlled, all with quasi-static local damping):
      * one uniaxial compression specimen (H/D = 2) -> UCS,
      * a uniaxial tension family on ``fit_thicknesses`` (``n_size_reps`` each)
        -> the tensile strength ``sigma_t`` (mean of the reference thickness
        ``fit_thicknesses[0]``) *and* the Weibull size-effect modulus m,
      * the held-out thicknesses (``n_holdout_reps`` each) -> AC4 extrapolation,
      * optionally ``n_disc`` Brazilian discs (D = 175 mm) -> the nominal
        ``2P/(pi D t)`` tensile strength and the centre stress field of AC5.

    ``sigma_t`` is taken from the **direct tension** specimen, not from the
    Brazilian disc: the disc route is kept as an *audited alternative* (AC5) and
    the audit shows that the discrete lattice does not reproduce Hondros'
    transverse tension (see the report), so the disc cannot be the strength
    oracle.  Literature ``sigma_c/sigma_t`` ratios are likewise direct-tension
    based, so this is also the like-for-like comparison.
    """
    geometry = disc_geometry or DiscGeometry()
    protocol_c = default_protocol(mode="compression")
    protocol_t = default_protocol(mode="tension")

    ucs_run = run_axial(vector, UCS_GEOMETRY, protocol_c, bond_seed=bond_seed)
    ucs = ucs_run.peak_stress

    disc_strengths: list[float] = []
    disc_runs: list[dict[str, Any]] = []
    hoop_ratios: list[float] = []
    for rep in range(int(n_disc)):  # optional: 0 by default (see docstring)
        run = run_brazilian(vector, geometry, protocol_c,
                            bond_seed=bond_seed + 1000 + rep,
                            probe=bool(probe_disc and rep == 0))
        disc_strengths.append(float(run["sigma_t_Pa"]))
        disc_runs.append({k: v for k, v in run.items()
                          if k not in ("stress_history_Pa", "strain_history",
                                       "hoop_ratio_history",
                                       "sigma_transverse_history_Pa",
                                       "sigma_load_history_Pa",
                                       "region_n_particles")})
        ratio = run.get("hoop_ratio_at_peak", float("nan"))
        if isinstance(ratio, float) and math.isfinite(ratio):
            hoop_ratios.append(ratio)
    sigma_t = float(np.mean(disc_strengths)) if disc_strengths else float("nan")

    family: dict[str, dict[str, Any]] = {}
    all_thicknesses = list(fit_thicknesses) + list(holdout_thicknesses)
    for t in all_thicknesses:
        reps = int(n_size_reps) if t in tuple(fit_thicknesses) else int(n_holdout_reps)
        spec_geometry = SpecimenGeometry.from_thickness(float(t), cross_n=4,
                                                       radius=PARTICLE_RADIUS)
        strengths: list[float] = []
        ke: list[float] = []
        for rep in range(reps):
            run = run_axial(vector, spec_geometry, protocol_t, bond_seed=bond_seed + 100 + rep)
            strengths.append(float(run.peak_stress))
            ke.append(float(run.ke_ratio))
        family[f"{t:.3f}"] = {
            "thickness_m": float(t),
            "gauge_m": float((spec_geometry.n_layers - 1) * 2.0 * PARTICLE_RADIUS),
            "cross_section_m2": float(spec_geometry.area),
            "n_layers": int(spec_geometry.n_layers),
            "n_particles": int(spec_geometry.n_layers * spec_geometry.cross_n ** 2),
            "strengths_Pa": strengths,
            "mean_Pa": float(np.mean(strengths)),
            "std_Pa": float(np.std(strengths, ddof=1)) if len(strengths) > 1 else 0.0,
            "ke_ratio": float(max(ke)) if ke else float("nan"),
            "held_out": bool(t in tuple(holdout_thicknesses)),
        }

    if not disc_strengths:
        # Direct-tension route: the tensile strength is the mean strength of the
        # reference (thinnest fitted) thickness of the size-effect family.
        sigma_t = float(family[f"{float(fit_thicknesses[0]):.3f}"]["mean_Pa"])
    fit_data = {float(t): family[f"{t:.3f}"]["strengths_Pa"] for t in fit_thicknesses}
    try:
        size_effect: dict[str, Any] = bootstrap_size_effect_m(fit_data, n_bootstrap=400)
    except ValueError as exc:
        size_effect = {"m": float("nan"), "ci_low": float("nan"),
                       "ci_high": float("nan"), "ci_width": float("nan"),
                       "error": str(exc), "thicknesses_m": list(fit_thicknesses)}

    return {
        "theta": vector.as_dict(),
        "unit": vector.to_unit().tolist(),
        "ucs_Pa": float(ucs),
        "ucs_MPa": float(ucs / 1e6),
        "sigma_t_Pa": sigma_t,
        "sigma_t_MPa": sigma_t / 1e6,
        "strength_ratio": float(ucs / sigma_t) if sigma_t > 0 else float("nan"),
        "weibull_m": float(size_effect.get("m", float("nan"))),
        "weibull_m_ci": [float(size_effect.get("ci_low", float("nan"))),
                         float(size_effect.get("ci_high", float("nan")))],
        "size_effect": size_effect,
        "family": family,
        "disc": {
            "geometry": {"n_lattice": geometry.n_lattice,
                         "n_layers": geometry.n_layers,
                         "diameter_m": geometry.diameter_m,
                         "thickness_m": geometry.thickness_m},
            "strengths_Pa": disc_strengths,
            "runs": disc_runs,
            "hoop_ratio_mean": float(np.mean(hoop_ratios)) if hoop_ratios else float("nan"),
        },
        "gates": {
            "ucs_ke_ratio": float(ucs_run.ke_ratio),
            "ucs_ke_ratio_peak": float(ucs_run.ke_ratio_peak),
            "max_family_ke_ratio": float(max(v["ke_ratio"] for v in family.values())),
            "max_disc_ke_ratio": float(max(r["ke_ratio"] for r in disc_runs)) if disc_runs else float("nan"),
            "quasistatic_ok": bool(
                ucs_run.ke_ratio <= QUASISTATIC_KE_LIMIT
                and all(v["ke_ratio"] <= QUASISTATIC_KE_LIMIT for v in family.values())),
        },
        "ucs_trace": {"strain": ucs_run.strain.tolist(),
                      "stress_Pa": ucs_run.stress.tolist(),
                      "broken_bonds": int(ucs_run.broken_bonds),
                      "n_bonds": int(ucs_run.n_bonds),
                      "n_steps": int(ucs_run.n_steps)},
    }


# ===========================================================================
# Geometry priors (AC1)
# ===========================================================================

def geometry_prior_check(*, ucs_geometry: SpecimenGeometry = UCS_GEOMETRY,
                         disc_geometry: DiscGeometry | None = None,
                         thicknesses: Sequence[float] = SIZE_EFFECT_THICKNESSES,
                         radius: float = PARTICLE_RADIUS) -> dict[str, Any]:
    """AC1: specimen aspect ratios and lattice-resolution self-check.

    * uniaxial UCS specimen: slenderness ``H/D`` in ``[1.5, 2.5]`` (ASTM D7012 band,
      so end effects do not dominate the peak);
    * Brazilian disc: ``D/t >= 1.5`` (a disc, not a rod) and at least 5 particles
      across the diameter (a coarser lattice cannot represent the Hondros field);
    * uniaxial tension family: slenderness ``H/D`` in ``[0.5, 8]`` (squatter
      specimens are grip dominated, slenderer ones bend in the gauge);
    * lattice density: the simple-cubic lattice has number density ``1/(2R)^3`` and
      packing fraction ``pi/6``; both are checked against the analytic values.

    Returns the per-check booleans, the numbers behind them and an overall
    ``passed`` flag.  Nothing here is a physics result: it is the prior that has to
    hold before any of the strengths may be believed.
    """
    disc = disc_geometry or DiscGeometry()
    checks: dict[str, Any] = {}

    ucs_h_over_d = float(ucs_geometry.thickness / (2.0 * radius * ucs_geometry.cross_n))
    checks["ucs_slenderness"] = {
        "H_over_D": float(ucs_h_over_d),
        "band": [1.5, 2.5],
        "ok": bool(1.5 <= ucs_h_over_d <= 2.5),
        "n_particles": int(ucs_geometry.n_layers * ucs_geometry.cross_n ** 2),
    }

    checks["disc"] = {
        "D_over_t": float(disc.diameter_m / disc.thickness_m),
        "band": [1.5, float("inf")],
        "particles_across_diameter": int(disc.n_lattice),
        "ok": bool(disc.diameter_m / disc.thickness_m >= 1.5 and disc.n_lattice >= 5),
    }

    family = []
    for t in thicknesses:
        spec = SpecimenGeometry.from_thickness(float(t), cross_n=4, radius=radius)
        h_over_d = (spec.n_layers - 1) * 2.0 * radius / (2.0 * radius * spec.cross_n)
        family.append({
            "thickness_m": float(t),
            "n_layers": int(spec.n_layers),
            "H_over_D": float(h_over_d),
            "ok": bool(0.5 <= h_over_d <= 8.0),
            "n_particles": int(spec.n_layers * spec.cross_n ** 2),
        })
    checks["tension_family"] = {
        "specimens": family,
        "band": [0.5, 8.0],
        "ok": bool(all(item["ok"] for item in family)),
    }

    spacing = 2.0 * radius
    number_density = 1.0 / spacing ** 3
    sphere_volume = 4.0 / 3.0 * math.pi * radius ** 3
    checks["lattice_density"] = {
        "number_density_m^-3": float(number_density),
        "packing_fraction": float(number_density * sphere_volume),
        "analytic_packing_fraction": float(math.pi / 6.0),
        "ok": bool(abs(number_density * sphere_volume - math.pi / 6.0) < 1e-12),
    }
    checks["passed"] = bool(all(item.get("ok", True) for item in (
        checks["ucs_slenderness"], checks["disc"], checks["tension_family"],
        checks["lattice_density"])))
    return checks


# ===========================================================================
# Numerical Jacobian and identifiability diagnostics (AC2)
# ===========================================================================

#: Responses of the calibration problem and the transform used to make them
#: comparable in the Jacobian.  All three are strict-positive physical quantities
#: spanning about one decade, so the log transform makes ``J`` the matrix of
#: *relative* sensitivities (a 1 % change of the response per unit change of the
#: unit-cube coordinate), which is the only basis on which a condition number is
#: meaningful across units.
RESPONSE_KEYS: tuple[str, ...] = ("ucs_MPa", "sigma_t_MPa", "weibull_m")
RESPONSE_NAMES: tuple[str, ...] = ("UCS", "sigma_t", "m")


def response_vector(point: dict[str, Any],
                    keys: Sequence[str] = RESPONSE_KEYS) -> np.ndarray:
    """Log10 of the three responses of one design point (``nan`` if undefined)."""
    out = np.empty(len(keys), dtype=np.float64)
    for k, key in enumerate(keys):
        value = float(point.get(key, float("nan")))
        out[k] = math.log10(value) if (math.isfinite(value) and value > 0.0) else float("nan")
    return out


def response_jacobian(center: dict[str, Any], plus: Sequence[dict[str, Any]],
                      minus: Sequence[dict[str, Any]], *,
                      step: float,
                      keys: Sequence[str] = RESPONSE_KEYS) -> dict[str, Any]:
    """Central-difference Jacobian ``d y / d u`` of the response surface.

    ``plus[k]`` / ``minus[k]`` are the design evaluations at ``u0 + step e_k`` /
    ``u0 - step e_k`` of the *reference* point ``center``; ``step`` is the offset in
    unit-cube coordinates.  Rows are the responses (log10), columns the parameters.
    """
    if not (len(plus) == len(minus) == len(PARAMETER_NAMES)):
        raise ValueError("need one plus and one minus sample per parameter")
    y0 = response_vector(center, keys)
    jac = np.full((len(keys), len(PARAMETER_NAMES)), np.nan, dtype=np.float64)
    for k, (p, m) in enumerate(zip(plus, minus)):
        jac[:, k] = (response_vector(p, keys) - response_vector(m, keys)) / (2.0 * float(step))
    return {
        "y_at_center": y0.tolist(),
        "center_responses": {name: float(center.get(key, float("nan")))
                             for name, key in zip(RESPONSE_NAMES, keys)},
        "jacobian": jac.tolist(),
        "step": float(step),
        "plus_units": [p.get("unit") for p in plus],
        "minus_units": [m.get("unit") for m in minus],
        "response_transform": "log10",
        "parameter_names": list(PARAMETER_NAMES),
        "response_names": list(RESPONSE_NAMES),
    }


def collinearity_index(jacobian: np.ndarray, subset: Sequence[int]) -> float:
    """Brun et al. (2001) collinearity index of a column subset.

    ``gamma_S = 1 / sqrt(lambda_min(J_S^T J_S))`` with the columns of ``J_S`` scaled
    to unit length.  ``gamma`` is the factor by which the norm of a perturbation of
    the subset is amplified in the response -- ``gamma > 10`` means the subset is
    practically unidentifiable.  Restricted to a subset of at least two columns.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    cols = list(subset)
    if len(cols) < 2:
        raise ValueError("collinearity index needs at least two columns")
    block = jac[:, cols]
    if not np.all(np.isfinite(block)):
        return float("nan")
    norms = np.linalg.norm(block, axis=0)
    if np.any(norms == 0.0):
        return float("inf")
    scaled = block / norms
    eigenvalues = np.linalg.eigvalsh(scaled.T @ scaled)
    lambda_min = float(eigenvalues.min())
    if lambda_min <= 0.0:
        return float("inf")
    return float(1.0 / math.sqrt(lambda_min))


def condition_diagnostics(jacobian: np.ndarray, *,
                          parameter_names: Sequence[str] = PARAMETER_NAMES,
                          response_names: Sequence[str] | None = None,
                          limit: float = COND_NUMBER_LIMIT) -> dict[str, Any]:
    """AC2: condition number, correlation matrix, collinearity, rank statement.

    Returns
    -------
    dict with

    ``cond``
        ``sigma_max / sigma_min`` of ``J`` (Belsley's condition number);
    ``singular_values``, ``right_singular_vectors``
        the SVD of ``J``;
    ``condition_indices`` + ``variance_decomposition``
        Belsley's per-direction indices and the proportion of the variance of each
        parameter that is carried by each near-null direction -- this is what turns
        "cond is large" into "``k_b`` and ``eps_b`` cannot be separated";
    ``correlation``
        the parameter correlation matrix implied by ``J`` (``J^T J`` scaled to unit
        diagonal) -- the numbers a calibration report has to publish;
    ``collinearity``
        Brun's index of the full set and of every pair, with the worst pair named;
    ``rank``, ``n_parameters``, ``n_responses``, ``null_space_dimension``
        the structural statement.  With three responses and six parameters the rank
        can never exceed three, so at least three directions are *structurally*
        unidentifiable whatever the parameter values -- the verdict says so
        explicitly instead of pretending that a finite condition number means
        identifiability;
    ``reduction``
        a machine-readable dimension-reduction proposal: groups of parameters whose
        pairwise ``|corr| > 0.99`` or ``gamma > 100`` and may therefore be merged /
        fixed.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    names = list(parameter_names)
    n_resp, n_par = jac.shape
    finite = np.all(np.isfinite(jac))
    out: dict[str, Any] = {
        "n_parameters": int(n_par),
        "n_responses": int(n_resp),
        "finite": bool(finite),
        "parameter_names": names,
        "response_names": list(response_names if response_names is not None
                               else RESPONSE_NAMES[:n_resp]),
    }
    if not finite:
        out.update(cond=float("inf"), rank=0, null_space_dimension=int(n_par),
                   verdict="non-identifiable",
                   reason="the Jacobian has non-finite entries (a response could not be evaluated)")
        return out

    u, s, vt = np.linalg.svd(jac, full_matrices=False)
    tol = max(jac.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    cond = float(s[0] / s[-1]) if (s.size and s[-1] > 0.0) else float("inf")
    condition_indices = [float(s[0] / si) if si > 0 else float("inf") for si in s]

    # Parameter correlation matrix implied by the Jacobian: J^T J scaled to unit
    # diagonal.  Two parameters with |corr| -> 1 move the responses identically and
    # cannot be separated by any amount of (noise-free) measurement.
    gram = jac.T @ jac
    scale = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    correlation = np.zeros_like(gram)
    for a in range(n_par):
        for b in range(n_par):
            denominator = scale[a] * scale[b]
            correlation[a, b] = gram[a, b] / denominator if denominator > 0 else float("nan")

    # Belsley variance decomposition: which parameters are carried by a direction.
    variance_proportion = np.zeros((n_par, len(s)), dtype=np.float64)
    for k in range(len(s)):
        vk = vt[k] ** 2 if k < vt.shape[0] else np.zeros(n_par)
        for j in range(n_par):
            numerator = vk[j] / (s[k] ** 2) if s[k] > 0 else float("inf")
            denominator = sum((vt[kk][j] ** 2) / (s[kk] ** 2) if s[kk] > 0 else float("inf")
                              for kk in range(len(s)))
            variance_proportion[j, k] = numerator / denominator if denominator not in (0.0, float("inf")) else float("nan")

    # Column norms first: a parameter whose column of J vanishes is *structurally
    # unobservable* (no amount of measurement of these responses can fix it), and
    # every pair containing it is trivially "collinear" -- so such pairs are kept
    # out of the pair ranking and reported once, below.
    column_norms = np.linalg.norm(jac, axis=0)
    scale_max = float(column_norms.max()) if column_norms.size else 0.0
    observable = [bool(scale_max > 0.0 and column_norms[j] > 1e-12 * scale_max)
                  for j in range(n_par)]
    full_collinearity = collinearity_index(jac, range(n_par)) if n_par >= 2 else float("nan")
    pairs: list[dict[str, Any]] = []
    for a in range(n_par):
        for b in range(a + 1, n_par):
            if not (observable[a] and observable[b]):
                continue
            pairs.append({
                "pair": [names[a], names[b]],
                "collinearity": collinearity_index(jac, (a, b)),
                "correlation": float(correlation[a, b]),
            })
    def _pair_key(item: dict) -> tuple[int, float]:
        value = item["collinearity"]
        if math.isnan(value):
            return (2, 0.0)          # undefined: never the "worst" pair
        if math.isinf(value):
            return (0, 0.0)          # perfectly collinear: always first
        return (1, -value)           # finite: largest first

    pairs.sort(key=_pair_key)

    # Parameters whose column of J vanishes are *structurally unobservable*: no
    # amount of measurement of these responses can fix them.  They are reported
    # separately from the merely collinear pairs, because the remedy differs (fix
    # them, or add an observable that sees them).
    unobservable = [
        {"parameter": names[j], "column_norm": float(column_norms[j]),
         "action": "structurally unobservable from these responses: fix it at a "
                   "literature value or add an observable that depends on it"}
        for j in range(n_par)
        if scale_max > 0.0 and column_norms[j] <= 1e-12 * scale_max
    ]
    reduction = [
        {"pair": item["pair"], "collinearity": item["collinearity"],
         "correlation": item["correlation"],
         "action": "merge into one effective parameter (keep the product / ratio)"}
        for item in pairs
        if math.isinf(item["collinearity"])
        or (math.isfinite(item["collinearity"]) and item["collinearity"] > 10.0)
        or (math.isfinite(item["correlation"]) and abs(item["correlation"]) > 0.95)
    ]

    structural = int(n_par - min(n_resp, rank))
    if cond > float(limit):
        verdict = "non-identifiable"
        reason = (f"cond(J) = {cond:.3g} > {float(limit):.3g}: the response combination "
                  "is not identifiable at this parameter point")
    elif structural > 0:
        verdict = "structurally-underdetermined"
        reason = (f"{n_par} parameters against {n_resp} observables: at most {n_resp} "
                  f"directions are identifiable, {structural} remain unconstrained "
                  "(null space of J)")
    else:
        verdict = "identifiable"
        reason = f"cond(J) = {cond:.3g} <= {float(limit):.3g} and no null space"

    out.update(
        cond=cond,
        singular_values=[float(v) for v in s],
        condition_indices=condition_indices,
        right_singular_vectors=vt.tolist(),
        variance_decomposition=variance_proportion.tolist(),
        rank=rank,
        null_space_dimension=int(n_par - rank),
        structural_null_space_dimension=structural,
        correlation=correlation.tolist(),
        observable_parameters=[names[j] for j in range(n_par) if observable[j]],
        collinearity={"all": full_collinearity, "pairs": pairs,
                      "worst_pair": pairs[0] if pairs else None},
        reduction=reduction,
        unobservable_parameters=unobservable,
        verdict=verdict,
        reason=reason,
        limit=float(limit),
    )
    return out


# ===========================================================================
# Response-surface surrogate (quadratic, ridge stabilised)
# ===========================================================================

def _quadratic_features(u: np.ndarray) -> np.ndarray:
    """``[1, u_i, u_i u_j (i <= j)]`` -- 1 + n + n(n+1)/2 columns."""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    n = u.shape[1]
    cols = [np.ones((u.shape[0], 1))]
    cols.append(u)
    cross = [u[:, a] * u[:, b] for a in range(n) for b in range(a, n)]
    cols.append(np.stack(cross, axis=1))
    return np.concatenate(cols, axis=1)


def surrogate_quadratic(unit_samples: Sequence[Sequence[float]],
                        responses: Sequence[float], *,
                        ridge: float | Sequence[float] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                        ) -> dict[str, Any]:
    """Ridge-stabilised quadratic response surface in the unit cube.

    ``ridge`` is dimensionless: it is applied to the *standardised* feature matrix
    so the regularisation does not depend on the feature scaling.  Returns the
    coefficient vector, the in-sample ``R^2``, the leave-one-out ``R^2``/RMSE and a
    ``predict`` callable -- the honest accuracy statement of the surrogate is what
    the NSGA-II front inherits.
    """
    u = np.asarray(unit_samples, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != y.size:
        raise ValueError("unit_samples must be (n, d) and match responses")
    if u.shape[0] < 4:
        raise ValueError("need at least 4 samples")
    if not (np.isscalar(ridge) or len(ridge) == 1):
        # Model selection on the *honest* metric: the ridge that maximises the
        # leave-one-out R^2.  A 33-point design cannot support 28 unregularised
        # coefficients (in-sample R^2 saturates at 1 while LOO collapses), so the
        # regularisation strength is chosen by LOO and reported.
        candidates = []
        for value in ridge:
            fitted = surrogate_quadratic(u, y, ridge=float(value))
            candidates.append((fitted["loo_r2"], float(value), fitted))
        candidates.sort(key=lambda item: (-item[0] if math.isfinite(item[0]) else -math.inf))
        best = candidates[0][2]
        best["ridge_grid"] = [{"ridge": item[1], "loo_r2": item[0]} for item in candidates]
        return best
    ridge = float(np.asarray(ridge).reshape(-1)[0])
    features = _quadratic_features(u)
    # standardise columns (guard the constant column)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    # the intercept column stays exactly at 1 (mean 0, scale 1) so it is the only
    # unpenalised column of the ridge system
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std > 0, std, 1.0)
    z = (features - mean) / std
    gram = z.T @ z
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, z.T @ y)
    prediction = z @ coefficients
    residual = y - prediction
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    def _fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        z_tr, y_tr = z[train_idx], y[train_idx]
        g = z_tr.T @ z_tr + penalty
        c = np.linalg.solve(g, z_tr.T @ y_tr)
        return z[test_idx] @ c

    loo = np.empty_like(y)
    indices = np.arange(y.size)
    for i in range(y.size):
        loo[i] = _fit_predict(indices != i, indices == i)[0]

    def predict(u_new: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        uu = np.atleast_2d(np.asarray(u_new, dtype=np.float64))
        zz = (_quadratic_features(uu) - mean) / std
        return zz @ coefficients

    return {
        "coefficients": coefficients.tolist(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "r2": float(1.0 - np.sum(residual ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "loo_r2": float(1.0 - np.sum((y - loo) ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "loo_rmse": float(np.sqrt(np.mean((y - loo) ** 2))),
        "n_samples": int(y.size),
        "n_features": int(features.shape[1]),
        "ridge": float(ridge),
        "predict": predict,
    }


# ===========================================================================
# NSGA-II (real coded, dependency free)
# ===========================================================================

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_nondominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Non-dominated sorting of minimisation objectives -> list of fronts."""
    f = np.asarray(objectives, dtype=np.float64)
    n = f.shape[0]
    dominated_count = np.zeros(n, dtype=int)
    dominates: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(f[i], f[j]):
                dominates[i].append(j)
                dominated_count[j] += 1
            elif _dominates(f[j], f[i]):
                dominates[j].append(i)
                dominated_count[i] += 1
    fronts: list[list[int]] = []
    current = [i for i in range(n) if dominated_count[i] == 0]
    while current:
        fronts.append(list(current))
        nxt: list[int] = []
        for i in current:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(objectives: np.ndarray, front: Sequence[int]) -> np.ndarray:
    """NSGA-II crowding distance of the members of one front."""
    f = np.asarray(objectives, dtype=np.float64)[list(front)]
    n, m = f.shape
    distance = np.zeros(n, dtype=np.float64)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(f[:, k])
        span = f[order[-1], k] - f[order[0], k]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if span <= 0:
            continue
        for idx in range(1, n - 1):
            distance[order[idx]] += (f[order[idx + 1], k] - f[order[idx - 1], k]) / span
    return distance


def _sbx(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
         eta: float) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return c1, c2


def _polynomial_mutation(x: np.ndarray, rng: np.random.Generator, eta: float,
                         probability: float) -> np.ndarray:
    out = x.copy()
    mask = rng.random(out.shape) < probability
    if not mask.any():
        return out
    u = rng.random(out.shape)
    delta = np.where(u < 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                     1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))
    out[mask] = np.clip(out[mask] + delta[mask], 0.0, 1.0)
    return out


def nsga2(objective: Callable[[np.ndarray], np.ndarray], n_variables: int, *,
          population: int = 60, generations: int = 60, seed: int = 20260101,
          crossover_probability: float = 0.9, mutation_probability: float | None = None,
          eta_crossover: float = 15.0, eta_mutation: float = 20.0,
          initial: Sequence[Sequence[float]] | None = None) -> dict[str, Any]:
    """Real-coded NSGA-II on ``objective(X) -> F`` (minimisation, ``X`` in [0, 1]^d).

    Returns the final population, its objectives, the first (Pareto) front and the
    per-generation front size.  ``initial`` seeds the population with known design
    points (the DEM samples of the campaign), which keeps the search anchored where
    the surrogate is actually constrained by data.
    """
    if population < 4:
        raise ValueError("population must be >= 4")
    rng = np.random.default_rng(seed)
    mutation_probability = (1.0 / n_variables if mutation_probability is None
                            else float(mutation_probability))
    X = rng.random((population, n_variables))
    if initial is not None:
        seeded = np.atleast_2d(np.asarray(initial, dtype=np.float64))
        if seeded.shape[1] != n_variables:
            raise ValueError("initial points must have n_variables columns")
        take = min(population, seeded.shape[0])
        X[:take] = np.clip(seeded[:take], 0.0, 1.0)
    F = np.asarray(objective(X), dtype=np.float64)
    history: list[int] = []

    def _rank_and_crowd(F_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fronts = fast_nondominated_sort(F_local)
        rank = np.empty(F_local.shape[0], dtype=int)
        crowd = np.empty(F_local.shape[0], dtype=np.float64)
        for r, front in enumerate(fronts):
            rank[front] = r
            crowd[front] = crowding_distance(F_local, front)
        return rank, crowd

    for _ in range(int(generations)):
        rank, crowd = _rank_and_crowd(F)
        history.append(int(np.sum(rank == 0)))
        # binary tournament on (rank, -crowd)
        parents = np.empty_like(X)
        for i in range(population):
            a, b = rng.integers(0, population, 2)
            if (rank[a] < rank[b]) or (rank[a] == rank[b] and crowd[a] >= crowd[b]):
                parents[i] = X[a]
            else:
                parents[i] = X[b]
        offspring = parents.copy()
        for i in range(0, population - 1, 2):
            if rng.random() < crossover_probability:
                offspring[i], offspring[i + 1] = _sbx(parents[i], parents[i + 1],
                                                      rng, eta_crossover)
        offspring = _polynomial_mutation(offspring, rng, eta_mutation,
                                         mutation_probability)
        F_offspring = np.asarray(objective(offspring), dtype=np.float64)
        pool_X = np.vstack([X, offspring])
        pool_F = np.vstack([F, F_offspring])
        fronts = fast_nondominated_sort(pool_F)
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= population:
                selected.extend(front)
                continue
            distance = crowding_distance(pool_F, front)
            order = np.argsort(-distance, kind="stable")
            remaining = population - len(selected)
            selected.extend([front[i] for i in order[:remaining]])
            break
        X = pool_X[selected]
        F = pool_F[selected]

    fronts = fast_nondominated_sort(F)
    pareto = fronts[0] if fronts else []
    return {
        "population": X.tolist(),
        "objectives": F.tolist(),
        "pareto_front": F[pareto].tolist() if pareto else [],
        "pareto_population": X[pareto].tolist() if pareto else [],
        "front_history": history,
        "n_generations": int(generations),
        "population_size": int(population),
    }


# ===========================================================================
# Acceptance: holdout extrapolation (AC4), feasibility (AC3)
# ===========================================================================

def holdout_error(size_effect: dict[str, Any], family: dict[str, Any], *,
                  fit_thicknesses: Sequence[float] = FIT_THICKNESSES,
                  holdout_thicknesses: Sequence[float] = HOLDOUT_THICKNESSES,
                  tolerance: float = HOLDOUT_TOLERANCE,
                  reference: dict[str, float] | None = None) -> dict[str, Any]:
    """AC4: extrapolate the fitted power law to the held-out thicknesses.

    For each held-out thickness ``t_h``:

    * ``predicted_Pa`` = ``sigma_ref * t_h**(-1/m)`` from the thin family fit only;
    * ``measured_Pa``  = the DEM mean strength at ``t_h`` (never used in the fit);
    * ``relative_error`` = ``(predicted - measured) / measured``;
    * ``local_m``     = ``-ln(sigma(t_h)/sigma(t_n)) / ln(t_h/t_n)`` against the
      nearest *fitted* thickness ``t_n`` -- the *direction* statement.  The two
      held-out ends are consistent when both local moduli are positive (strength
      keeps decreasing with size) and their deviation from the global ``m`` has the
      same sign, i.e. the power law does not bend in opposite directions at the two
      ends of the extrapolation.

    ``reference`` optionally carries an independent measurement at the held-out
    thickness (the reference-material DEM value) used as the "literature
    extrapolation" of the DAG: the tolerance is applied to both the self-consistency
    error and, when a reference is given, to the transferability error.
    """
    m = float(size_effect.get("m", float("nan")))
    sigma_ref = float(size_effect.get("sigma_ref_Pa", float("nan")))
    reference = reference or {}
    rows: list[dict[str, Any]] = []
    for t_h in holdout_thicknesses:
        key = f"{float(t_h):.3f}"
        if key not in family:
            continue
        measured = float(family[key]["mean_Pa"])
        predicted = (sigma_ref * float(t_h) ** (-1.0 / m)
                     if (math.isfinite(m) and m > 0.0 and math.isfinite(sigma_ref))
                     else float("nan"))
        nearest = min(fit_thicknesses, key=lambda t: abs(math.log(float(t) / float(t_h))))
        sigma_n = float(family.get(f"{float(nearest):.3f}", {}).get("mean_Pa", float("nan")))
        local_m = (math.log(measured / sigma_n) / math.log(float(nearest) / float(t_h))
                   if (math.isfinite(sigma_n) and sigma_n > 0.0 and measured > 0.0)
                   else float("nan"))
        ref_value = reference.get(key, float("nan")) if reference else float("nan")
        rows.append({
            "thickness_m": float(t_h),
            "predicted_Pa": predicted,
            "measured_Pa": measured,
            "relative_error": ((predicted - measured) / measured
                               if measured else float("nan")),
            "absolute_relative_error": (abs(predicted - measured) / measured
                                        if measured else float("nan")),
            "nearest_fit_thickness_m": float(nearest),
            "local_m": local_m,
            "reference_Pa": float(ref_value),
            "reference_relative_error": ((predicted - float(ref_value)) / float(ref_value)
                                         if (reference and math.isfinite(float(ref_value))
                                             and float(ref_value) != 0.0)
                                         else float("nan")),
            "within_tolerance": bool(
                measured > 0.0 and math.isfinite(predicted)
                and abs(predicted - measured) / measured <= float(tolerance)),
        })
    finite_local = [row["local_m"] for row in rows if math.isfinite(row["local_m"])]
    direction_consistent = bool(
        len(finite_local) >= 2
        and all(value > 0.0 for value in finite_local)
        and len({math.copysign(1.0, value - m) for value in finite_local}) == 1)
    return {
        "rows": rows,
        "m": m,
        "sigma_ref_Pa": sigma_ref,
        "direction_consistent": direction_consistent,
        "all_within_tolerance": bool(all(row["within_tolerance"] for row in rows)) if rows else False,
        "tolerance": float(tolerance),
        "passed": bool(rows and all(row["within_tolerance"] for row in rows)
                       and direction_consistent),
    }


def feasibility_verdict(point: dict[str, Any], *,
                        ucs_band: tuple[float, float] = UCS_BAND_MPA,
                        ratio_band: tuple[float, float] = STRENGTH_RATIO_BAND,
                        m_truth: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
    """AC3: three-branch feasibility verdict at one parameter point.

    Targets
    -------
    * ``UCS``: the nominal uniaxial compressive strength inside ``ucs_band`` [MPa];
    * ``ratio``: ``sigma_c/sigma_t`` inside ``ratio_band``;
    * ``m``: the Weibull-modulus 95 % CI covers the reference ("truth") value
      ``m_truth["m"]`` (a *pseudo-truth*: the m of the reference material measured by
      the same pipeline, see the report's limitations -- no experimental macro-m
      exists for this synthetic bond model).

    Verdict
    -------
    ``(a) feasible``           all three satisfied -> deliver the calibration point;
    ``(b) paired-feasible``    exactly two satisfied -> the conflicting pair and the
                               minimal violation (in normalised units) are reported;
    ``(c) infeasible``         fewer than two satisfied -> the point is hopeless; the
                               campaign-level ``(c)`` verdict additionally needs the
                               Pareto front to show no feasible point at all.
    """
    ucs = float(point.get("ucs_MPa", float("nan")))
    ratio = float(point.get("strength_ratio", float("nan")))
    ci_low, ci_high = (point.get("weibull_m_ci") or [float("nan"), float("nan")])[:2]

    def _band_violation(value: float, band: tuple[float, float]) -> float:
        lo, hi = band
        if not math.isfinite(value):
            return float("inf")
        if value < lo:
            return (lo - value) / (hi - lo)
        if value > hi:
            return (value - hi) / (hi - lo)
        return 0.0

    ucs_violation = _band_violation(ucs, ucs_band)
    ratio_violation = _band_violation(ratio, ratio_band)
    if m_truth is not None and math.isfinite(float(m_truth.get("m", float("nan")))):
        m_target = float(m_truth["m"])
        if math.isfinite(ci_low) and math.isfinite(ci_high):
            if ci_low <= m_target <= ci_high:
                m_violation = 0.0
            else:
                gap = (ci_low - m_target) if ci_low > m_target else (m_target - ci_high)
                m_violation = abs(gap) / max(abs(m_target), 1e-12)
        else:
            m_violation = float("inf")
        m_target_out: float | None = m_target
    else:
        m_violation = float("nan")
        m_target_out = None

    satisfied = {
        "UCS": bool(ucs_violation == 0.0),
        "strength_ratio": bool(ratio_violation == 0.0),
        "weibull_m": bool(m_violation == 0.0) if math.isfinite(m_violation) else None,
    }
    violations = {
        "UCS": ucs_violation,
        "strength_ratio": ratio_violation,
        "weibull_m": m_violation,
    }
    n_satisfied = sum(1 for value in satisfied.values() if value is True)
    evaluated = [name for name, value in satisfied.items() if value is not None]
    if n_satisfied == len(evaluated) and len(evaluated) == 3:
        branch = "a"
        label = "feasible: all three targets satisfied by this theta"
        conflicting: list[str] = []
        minimal_violation = 0.0
    elif n_satisfied == 2:
        branch = "b"
        conflicting = [name for name, value in satisfied.items() if value is False]
        label = ("paired-feasible: only two targets can be met simultaneously; the "
                 "conflicting pair and its minimal violation are reported")
        minimal_violation = float(min(violations[name] for name in conflicting)) if conflicting else float("nan")
    else:
        branch = "c"
        conflicting = [name for name, value in satisfied.items() if value is False]
        label = ("infeasible at this theta: fewer than two targets are met -- "
                 "the calibration problem cannot be closed from these observables")
        minimal_violation = float(min(violations[name] for name in conflicting)) if conflicting else float("nan")
    return {
        "branch": branch,
        "label": label,
        "satisfied": satisfied,
        "violations_normalised": violations,
        "n_satisfied": int(n_satisfied),
        "conflicting_targets": conflicting,
        "minimal_violation": minimal_violation,
        "ucs_MPa": ucs,
        "strength_ratio": ratio,
        "weibull_m": float(point.get("weibull_m", float("nan"))),
        "weibull_m_ci": [float(ci_low), float(ci_high)],
        "m_pseudo_truth": m_target_out,
        "ucs_band_MPa": list(ucs_band),
        "ratio_band": list(ratio_band),
    }
