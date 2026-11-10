"""Contact-parameter dimension ledger and similarity-scale gate (DEM).

Motivation
----------
DEM contact parameters are the single most common source of silent physics
errors in granular modelling: a rolling-friction coefficient typed in metres is
consumed as a dimensionless ratio, a contact stiffness is quoted for a unit
particle and then reused in a scaled-up (coarse-grained) case, a restitution
coefficient is nudged to 1.05 and silently clamped inside the dashpot helper.
None of these produce an exception; they only bias the macroscopic answer.

This module makes such mistakes machine-catchable.  It has three parts:

1. **Parameter registry** (:data:`PARAMETERS`).  Every contact/geometry
   parameter carries its SI base-dimension vector ``(M, L, T)``, its unit
   string, a validity range and the *convention* under which the value must be
   read when the same symbol is used with two different dimensions in the
   literature (the rolling-resistance coefficient is the classic case: a lever
   arm ``mu_r[R]`` with ``M_r = mu_r |F_n|`` versus the dimensionless ratio
   ``M_r = mu_r R_eff |F_n|``).

2. **Fleet extraction and scan** (:func:`scan_fleet`).  All ``src/tensordem``
   modules that declare at least one registered contact parameter are parsed
   with :mod:`ast` (no imports, no torch) and every dataclass-style field with
   a literal default is extracted.  Modules in scope are then checked
   *fail-closed*: a contact-like field that is not in the registry is a
   violation, so a new case cannot introduce an unlabelled dimension.

3. **Similarity transform** (:class:`SimilarityMode`, :func:`scale_case_json`).
   For a geometric scale ``lambda`` the *required* exponent of every parameter
   follows from its dimension vector and from the base-unit scale exponents of
   the similarity mode (Froude/gravity similarity for granular flow: lengths
   x lambda, density fixed, gravity fixed).  :func:`audit_variant` compares a
   declared scaled variant with the required law and reports per-parameter
   deviations, which is what the CI gate uses.

Similarity algebra (why the exponents below are not guesses)
------------------------------------------------------------
With ``L -> lam L``, ``rho`` and ``g`` fixed:

    mass exponent     3 * 1                     = 3
    time exponent     (1 - 0) / 2               = 0.5    (from g: L/T^2 invariant)
    exponent(q)       a*M_exp + b*L_exp + c*T_exp  for q with dim M^a L^b T^c

Consequences relevant to the two disputed parameters:

* linear spring ``kn`` [N/m] (M L^0 T^-2): ``3 - 2*0.5 = 2``  -> ``kn ~ lam^2``
* Hertz stiffness ``kn`` [N/m^(3/2)] (M L^-1/2 T^-2): ``3 - 0.5 - 1 = 1.5``
  -> ``kn ~ lam^1.5``.  The frequently quoted "Hertz sqrt(R) law" is the
  scaling of ``kn = (4/3) E* sqrt(R)`` at *fixed* ``E*``; it is a
  material-property rule, it does **not** preserve gravity similarity.
* rolling lever arm ``mu_r`` [m]: exponent 1 -> ``mu_r ~ lam``; in the
  dimensionless convention (``mu_r^dim = mu_r^L / R``) the exponent is 0.
  Doubling ``mu_r`` for ``R -> 2R`` therefore *is* the invariant transform when
  ``mu_r`` carries length, and *is* the error when it is read as a ratio.

All quantities are SI.  The module is dependency-free apart from the standard
library, so the ledger can be rebuilt in CI in milliseconds.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 1. Dimension algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimension:
    """SI base-dimension vector ``M^m L^l T^t``."""

    m: float = 0.0
    l: float = 0.0
    t: float = 0.0

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.m + other.m, self.l + other.l, self.t + other.t)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.m - other.m, self.l - other.l, self.t - other.t)

    def power(self, p: float) -> "Dimension":
        return Dimension(self.m * p, self.l * p, self.t * p)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.m, self.l, self.t)

    def __str__(self) -> str:  # e.g. "M^1 L^0 T^-2"
        def fmt(x: float) -> str:
            return f"{x:g}"

        return f"M^{fmt(self.m)} L^{fmt(self.l)} T^{fmt(self.t)}"


DIMENSIONLESS = Dimension()
LENGTH = Dimension(l=1.0)
MASS = Dimension(m=1.0)
TIME = Dimension(t=1.0)


@dataclass(frozen=True)
class Parameter:
    """One registered contact parameter.

    ``convention`` documents how the value must be read; ``alternative`` gives
    the other convention that exists in the literature for the *same symbol*
    (so a value can be classified as "read under the wrong convention").
    """

    name: str
    dimension: Dimension
    unit: str
    convention: str
    lower: float | None = None
    upper: float | None = None
    alternative: Dimension | None = None
    note: str = ""

    def required_exponent(self, mode: "SimilarityMode") -> float:
        return mode.exponent(self.dimension)

    def in_range(self, value: float) -> bool:
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


#: Rolling-torque conventions.  ``ratio``: ``M_r = mu_r R_eff |F_n|`` (mu_r is
#: a pure number, the convention implemented in :mod:`tensordem.rolling_resistance`
#: and :mod:`tensordem.granular3d_core`).  ``lever``: ``M_r = mu_r |F_n|`` with
#: ``mu_r`` in metres (Ai et al. 2011, Iwashita & Oda 1998 parameterisations).
ROLLING_CONVENTIONS = ("ratio", "lever")


def _build_registry() -> dict[str, Parameter]:
    reg: list[Parameter] = [
        # --- material / geometry -------------------------------------------
        Parameter("radius", LENGTH, "m", "particle radius", lower=0.0),
        Parameter("diameter", LENGTH, "m", "particle diameter", lower=0.0),
        Parameter("density", Dimension(m=1.0, l=-3.0), "kg/m^3",
                  "grain density", lower=0.0),
        Parameter("gravity", Dimension(l=1.0, t=-2.0), "m/s^2",
                  "gravitational acceleration", lower=0.0),
        Parameter("box_width", LENGTH, "m", "box extent (x)"),
        Parameter("box_depth", LENGTH, "m", "box extent (y)"),
        Parameter("box_height", LENGTH, "m", "box extent (z)"),
        Parameter("fill_height", LENGTH, "m", "fill height"),
        Parameter("cylinder_radius", LENGTH, "m", "container radius"),
        Parameter("plate_width", LENGTH, "m", "moving plate width"),
        Parameter("plate_thickness", LENGTH, "m", "plate / floe thickness"),
        Parameter("bed_depth", LENGTH, "m", "bed depth"),
        Parameter("drag_depth", LENGTH, "m", "intruder depth below surface"),
        Parameter("packing_gap", DIMENSIONLESS, "-", "lattice gap / radius"),
        Parameter("jitter", DIMENSIONLESS, "-", "position noise / radius"),
        # --- stiffness ------------------------------------------------------
        Parameter("kn", Dimension(m=1.0, t=-2.0), "N/m",
                  "linear normal stiffness (F = kn delta)", lower=0.0,
                  alternative=Dimension(m=1.0, l=-0.5, t=-2.0),
                  note="Hertz parameterisation of the same symbol is N/m^(3/2)"),
        Parameter("kt", Dimension(m=1.0, t=-2.0), "N/m",
                  "tangential stiffness", lower=0.0),
        Parameter("kt_ratio", DIMENSIONLESS, "-", "kt / kn", lower=0.0),
        Parameter("kr", Dimension(m=1.0, l=2.0, t=-2.0), "N*m",
                  "rolling stiffness (torque per rad)", lower=0.0),
        Parameter("youngs_modulus", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa",
                  "contact Young's modulus", lower=0.0),
        Parameter("poisson", DIMENSIONLESS, "-", "Poisson ratio", lower=0.0,
                  upper=0.5),
        Parameter("gamma", Dimension(m=1.0, t=-1.0), "N*s/m",
                  "normal dashpot coefficient", lower=0.0),
        Parameter("damping_ratio", DIMENSIONLESS, "-", "viscous damping ratio",
                  lower=0.0, upper=1.0),
        # --- friction / restitution ----------------------------------------
        Parameter("mu", DIMENSIONLESS, "-", "sliding friction coefficient",
                  lower=0.0),
        Parameter("mu_rolling", DIMENSIONLESS, "-",
                  "rolling resistance ratio (M_r = mu_r R_eff |F_n|)",
                  lower=0.0, alternative=LENGTH,
                  note="doubling with R is WRONG in this convention"),
        Parameter("mu_r", DIMENSIONLESS, "-",
                  "rolling resistance ratio (M_r = mu_r R_eff |F_n|)",
                  lower=0.0, alternative=LENGTH),
        Parameter("rolling_lever", LENGTH, "m",
                  "rolling resistance lever arm (M_r = mu_r |F_n|)",
                  lower=0.0, alternative=DIMENSIONLESS),
        Parameter("rolling_mu", DIMENSIONLESS, "-",
                  "rolling resistance ratio", lower=0.0, alternative=LENGTH),
        Parameter("rest", DIMENSIONLESS, "-", "coefficient of restitution",
                  lower=0.0, upper=1.0),
        Parameter("restitution", DIMENSIONLESS, "-",
                  "coefficient of restitution", lower=0.0, upper=1.0),
        Parameter("restitution_t", DIMENSIONLESS, "-",
                  "tangential restitution", lower=0.0, upper=1.0),
        Parameter("friction_angle", DIMENSIONLESS, "rad",
                  "interparticle friction angle", lower=0.0),
        Parameter("omega_tol", Dimension(t=-1.0), "rad/s",
                  "rolling-rate regularisation"),
        Parameter("omega_ref", Dimension(t=-1.0), "rad/s",
                  "viscous rolling reference rate", lower=0.0),
        # --- cohesion -------------------------------------------------------
        Parameter("adhesion_energy", Dimension(m=1.0, l=2.0, t=-2.0), "J",
                  "pair adhesion energy", lower=0.0),
        Parameter("surface_energy", Dimension(m=1.0, t=-2.0), "J/m^2",
                  "specific surface energy", lower=0.0),
        Parameter("cohesion", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa",
                  "Mohr-Coulomb cohesion", lower=0.0),
        # --- loads / kinematics --------------------------------------------
        Parameter("normal_stress", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa",
                  "applied normal stress"),
        Parameter("shear_velocity", Dimension(l=1.0, t=-1.0), "m/s",
                  "platen / intruder speed"),
        Parameter("velocity", Dimension(l=1.0, t=-1.0), "m/s", "imposed speed"),
        Parameter("drag_velocity", Dimension(l=1.0, t=-1.0), "m/s",
                  "intruder speed"),
        # --- protocol times / numerics -------------------------------------
        Parameter("dt", TIME, "s", "time step", lower=0.0),
        Parameter("timestep", TIME, "s", "time step", lower=0.0),
        Parameter("settle_time", TIME, "s", "settling time", lower=0.0),
        Parameter("hold_time", TIME, "s", "confinement hold time", lower=0.0),
        Parameter("shear_time", TIME, "s", "shearing time", lower=0.0),
        Parameter("run_time", TIME, "s", "total simulated time", lower=0.0),
        Parameter("shear_strain_target", DIMENSIONLESS, "-",
                  "target engineering shear strain", lower=0.0),
        Parameter("seed", DIMENSIONLESS, "-", "RNG seed"),
        Parameter("steps", DIMENSIONLESS, "-", "step count"),
    ]
    return {p.name: p for p in reg}


PARAMETERS: dict[str, Parameter] = _build_registry()


# ---------------------------------------------------------------------------
# 1b. Fleet dictionary: every alias in use, with the dimension it carries
# ---------------------------------------------------------------------------
# The registry above holds the *canonical* parameters.  The fleet, however,
# spells the same quantity in many ways (``k_n``, ``contact_stiffness``,
# ``stiffness``, ``bond_stiffness`` ...).  ``_ALIAS_TABLE`` maps those
# spellings onto a canonical parameter; ``_RULES`` cover whole families by
# token pattern; ``_AMBIGUOUS`` marks names that are used with *two* different
# dimensions in the fleet, which is exactly the failure the ledger exists to
# surface (the label cannot be attached, so the gate reports it instead of
# guessing).

_ALIAS_TABLE: dict[str, str] = {
    # -- particle geometry / material ------------------------------------
    "r_p": "radius", "particle_radius": "radius", "r_mean": "radius",
    "d_p": "diameter", "rho": "density", "rho_p": "density",
    "rho_particle": "density",
    # -- normal stiffness -------------------------------------------------
    "k_n": "kn", "k_normal": "kn", "normal_stiffness": "kn",
    "contact_stiffness": "kn", "contact_stiffness_n_m": "kn",
    "wall_stiffness_n_m": "kn", "wall_stiffness_t_n_m": "kn",
    "contact_stiffness_t_n_m": "kn", "platen_stiffness": "kn",
    "bond_stiffness": "kn", "reference_stiffness": "kn",
    "primary_stiffness": "kn", "k_bond": "kn", "stiffness": "kn",
    "punch_stiffness": "kn",
    # -- tangential / rolling stiffness ------------------------------------
    "k_t": "kt", "k_tangential": "kt", "tangential_stiffness": "kt",
    "bond_shear_stiffness": "kt", "k_r": "kr", "rolling_stiffness": "kr",
    "kr_ratio": "kt_ratio", "kt_ratio": "kt_ratio",
    # -- elastic constants -------------------------------------------------
    "young": "youngs_modulus", "youngs": "youngs_modulus",
    "young_modulus": "youngs_modulus", "contact_young_modulus": "youngs_modulus",
    "bond_modulus": "youngs_modulus", "bond_modulus_pa": "youngs_modulus",
    "ice_young_modulus_pa": "youngs_modulus",
    "poisson_ratio": "poisson", "contact_poisson_ratio": "poisson",
    "ice_poisson_ratio": "poisson",
    # -- friction ----------------------------------------------------------
    "friction": "mu", "friction_coeff": "mu", "friction_coefficient": "mu",
    "mu_particle": "mu", "mu_grain": "mu", "mu_pp": "mu", "mu_wall": "mu",
    "wall_mu": "mu", "wall_friction": "mu", "wall_friction_coefficient": "mu",
    "mu_floor": "mu", "floor_mu": "mu", "floor_friction": "mu",
    "mu_side": "mu", "mu_slide": "mu", "mu_static": "mu", "mu_structure": "mu",
    "contact_friction": "mu", "hull_friction": "mu",
    "hull_friction_coefficient": "mu", "platen_friction": "mu",
    "platen_friction_coefficient": "mu", "punch_friction": "mu",
    "soil_friction_coefficient": "mu", "pipe_friction_coefficient": "mu",
    "friction_mu": "mu", "dissipated_friction": "mu",
    "primary_friction": "mu", "drilling_friction": "mu",
    # -- restitution -------------------------------------------------------
    "rest": "restitution", "restitution_coefficient": "restitution",
    "en": "restitution", "e_n": "restitution",
    # -- rolling resistance -------------------------------------------------
    "rolling_friction": "mu_rolling", "rolling_mu": "mu_rolling",
    "mu_roll": "mu_rolling", "rolling_resistance": "mu_rolling",
    "rolling_resistance_coefficient": "mu_rolling",
    "c_rolling": "mu_rolling",
    "rolling_omega_ref": "omega_ref", "omega": "omega_tol",
    "omega_smooth": "omega_tol", "relax_omega": "omega_tol",
    # geometry
    "thickness": "plate_thickness", "thickness_m": "plate_thickness",
    "contact_width": "plate_width",
    "bed_height": "fill_height", "fill_level": "fill_height",
}

#: ``name -> (dimension, unit, confidence, note)`` for whole families.
_RULES: tuple[tuple[str, Dimension, str, str], ...] = (
    ("_fraction", DIMENSIONLESS, "-", "ratio of a reference value"),
    ("_ratio", DIMENSIONLESS, "-", "ratio"),
    ("_factor", DIMENSIONLESS, "-", "multiplicative factor"),
    ("_scale", DIMENSIONLESS, "-", "scale factor"),
    ("_gain", DIMENSIONLESS, "-", "gain"),
    ("_seed", DIMENSIONLESS, "-", "integer seed"),
    ("_cv", DIMENSIONLESS, "-", "coefficient of variation"),
    ("_number", DIMENSIONLESS, "-", "count"),
    ("_steps", DIMENSIONLESS, "-", "step count"),
    ("_periods", DIMENSIONLESS, "-", "period count"),
    ("_exponent", DIMENSIONLESS, "-", "empirical exponent"),
    ("_per_area", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa", "per unit area"),
    ("_modulus", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa", "elastic modulus"),
    ("_strength", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa", "strength"),
    ("_pa", Dimension(m=1.0, l=-1.0, t=-2.0), "Pa", "stress-like"),
    ("_energy", Dimension(m=1.0, l=2.0, t=-2.0), "J", "energy"),
    ("_j", Dimension(m=1.0, l=2.0, t=-2.0), "J", "energy"),
    ("_force", Dimension(m=1.0, l=1.0, t=-2.0), "N", "force"),
    ("_tension", Dimension(m=1.0, l=1.0, t=-2.0), "N", "force"),
    ("_time", TIME, "s", "time"),
    ("_rate", Dimension(t=-1.0), "1/s", "rate"),
    ("_velocity", Dimension(l=1.0, t=-1.0), "m/s", "velocity"),
)

#: Names that appear with two different dimensions in the fleet.  The gate
#: reports them instead of guessing (see :func:`label_parameter`).
_DASHPOT = Dimension(m=1.0, t=-1.0)
_AMBIENT = Dimension(m=1.0, l=2.0, t=-1.0)
_AMBIGUOUS: dict[str, tuple[Dimension, str, str]] = {
    # name: (nominal dimension, unit, why the label cannot be decided)
    "damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs damping ratio [-]"),
    "contact_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "normal_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "tangent_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "tangential_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "bond_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "bulk_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "global_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "velocity_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "viscous_damping": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "viscous_damp": (_DASHPOT, "N*s/m", "dashpot [N s/m] vs ratio [-]"),
    "rotational_damping": (_AMBIENT, "N*m*s", "rotational dashpot vs ratio [-]"),
    "linear_heave_damping": (_DASHPOT, "N*s/m", "dashpot vs ratio [-]"),
    "zeta": (DIMENSIONLESS, "-", "damping ratio [-]"),
    "stiffness": (Dimension(m=1.0, t=-2.0), "N/m",
                  "linear [N/m] vs Hertz [N/m^(3/2)] vs modulus [Pa]"),
    "punch_stiffness": (Dimension(m=1.0, t=-2.0), "N/m",
                        "linear [N/m] vs per-area [Pa]"),
    "bond_stiffness": (Dimension(m=1.0, t=-2.0), "N/m",
                       "linear [N/m] vs per-area [Pa]"),
    "contact_stiffness": (Dimension(m=1.0, t=-2.0), "N/m",
                          "linear [N/m] vs Hertz [N/m^(3/2)]"),
}

#: Fields whose ``*_mu`` suffix means "mean of the distribution", not friction.
_MEAN_FIELDS: dict[str, str] = {
    "sail_height_mu": "sail_height", "keel_depth_mu": "keel_depth",
    "porosity_mu": "porosity",
}


#: Ordered token rules: the first rule whose token set is hit wins.
_TOKEN_RULES: tuple[tuple[frozenset[str], Dimension, str, str], ...] = (
    (frozenset({"over", "every", "nnz", "seed", "steps", "periods", "number",
                "index", "cv", "deg", "coefficient", "fraction", "ratio",
                "factor", "scale", "gain"}), DIMENSIONLESS, "-",
     "dimensionless by token"),
    (frozenset({"m2"}), Dimension(m=1.0, t=-2.0), "J/m^2",
     "specific (per area) energy"),
    (frozenset({"energy"}), Dimension(m=1.0, l=2.0, t=-2.0), "J", "energy"),
    (frozenset({"modulus", "young", "youngs"}), Dimension(m=1.0, l=-1.0, t=-2.0),
     "Pa", "elastic modulus"),
    (frozenset({"strength", "stress", "pressure", "pa"}), 
     Dimension(m=1.0, l=-1.0, t=-2.0), "Pa", "stress-like"),
    (frozenset({"force", "tension", "f"}), Dimension(m=1.0, l=1.0, t=-2.0),
     "N", "force"),
    (frozenset({"stiffness", "stiff", "k"}), Dimension(m=1.0, t=-2.0),
     "N/m", "linear stiffness"),
    (frozenset({"damping", "damp", "gamma"}), Dimension(m=1.0, t=-1.0),
     "N*s/m", "dashpot"),
    (frozenset({"time"}), TIME, "s", "time"),
    (frozenset({"rate", "omega", "eps", "frequency"}), Dimension(t=-1.0),
     "1/s", "rate"),
    (frozenset({"velocity", "speed", "threshold"}), Dimension(l=1.0, t=-1.0),
     "m/s", "velocity"),
    (frozenset({"poisson", "zeta", "restitution", "friction", "mu"}),
     DIMENSIONLESS, "-", "dimensionless material ratio"),
    (frozenset({"width", "length", "depth", "height", "thickness", "radius",
                "gap", "offset", "size"}), LENGTH, "m", "length"),
    (frozenset({"angle"}), DIMENSIONLESS, "rad", "angle"),
)

#: Fields whose dimension is decided by inspection (no rule can infer it).
_MANUAL_LABELS: dict[str, tuple[Dimension, str, str]] = {
    "cohesion_softening": (DIMENSIONLESS, "-", "softening ratio"),
    "cohesion_range": (DIMENSIONLESS, "-", "cohesion jitter ratio"),
    "cohesion_transfer": (DIMENSIONLESS, "-", "cohesion transfer ratio"),
    "bond_compression": (DIMENSIONLESS, "-", "bond compression parameter"),
    "bond_delay": (DIMENSIONLESS, "-", "bond activation delay ratio"),
    "velocity_rest_threshold": (Dimension(l=1.0, t=-1.0), "m/s",
                                "rest velocity threshold"),
    "stiffness_nnz": (DIMENSIONLESS, "-", "nonzero count of stiffness matrix"),
    "energy_every": (DIMENSIONLESS, "-", "output interval"),
    "f_bond": (Dimension(m=1.0, l=1.0, t=-2.0), "N", "bond force"),
    "contact_length": (LENGTH, "m", "contact patch length"),
    "contact_length_m": (LENGTH, "m", "contact patch length"),
    "contact_width_m": (LENGTH, "m", "contact patch width"),
    "a_bond": (DIMENSIONLESS, "-", "bond area exponent (ambiguous: area or "
                                   "dimensionless constant)"),
}


def _rule_dimension(name: str) -> tuple[Dimension, str, str] | None:
    if name in _MANUAL_LABELS:
        return _MANUAL_LABELS[name]
    for suffix, dim, unit, note in _RULES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return dim, unit, note
    toks = set(tokens(name))
    for keys, dim, unit, note in _TOKEN_RULES:
        if toks & keys:
            return dim, unit, note
    return None


@dataclass(frozen=True)
class Label:
    """Result of resolving a fleet field name to a dimension."""

    canonical: str
    dimension: Dimension
    unit: str
    confidence: str          # "manual" | "rule" | "unknown"
    note: str
    ambiguous: str | None = None


def label_parameter(name: str) -> Label | None:
    """Attach a dimension label to a fleet field name, or return ``None``.

    Resolution order: canonical registry, distribution means, explicit alias
    table, ambiguity table, suffix rules, token rules.
    """
    key = normalise(name)
    if key in PARAMETERS:
        p = PARAMETERS[key]
        return Label(key, p.dimension, p.unit, "manual", p.convention, None)
    if key in _MEAN_FIELDS:
        target = _MEAN_FIELDS[key]
        length_like = ("height" in target) or ("depth" in target)
        dim = LENGTH if length_like else DIMENSIONLESS
        unit = "m" if length_like else "-"
        return Label(target, dim, unit, "manual",
                     "distribution mean of " + target)
    if key in _ALIAS_TABLE:
        p = PARAMETERS[_ALIAS_TABLE[key]]
        amb = _AMBIGUOUS.get(key) or _AMBIGUOUS.get(_ALIAS_TABLE[key])
        return Label(_ALIAS_TABLE[key], p.dimension, p.unit,
                     "unknown" if amb else "manual", p.convention, amb)
    if key in _AMBIGUOUS:
        dim, unit, why = _AMBIGUOUS[key]
        return Label(key, dim, unit, "unknown", "ambiguous label", why)
    rule = _rule_dimension(key)
    if rule is not None:
        dim, unit, note = rule
        return Label(key, dim, unit, "rule", note, None)
    toks = set(tokens(key))
    if toks & {"mu", "friction", "restitution", "poisson", "zeta", "seed",
               "steps", "periods", "number", "fraction", "ratio", "factor",
               "scale", "gain", "cv", "fraction"}:
        return Label(key, DIMENSIONLESS, "-", "rule",
                     "dimensionless by token " + ",".join(sorted(toks)))
    return None


#: Name fragments that make a field "contact bearing".  Anything matching one
#: of these inside an in-scope module must be registered (fail-closed).
CONTACT_HINT_TOKENS: frozenset[str] = frozenset({
    "mu", "mur", "friction", "restitution", "rest", "stiffness", "kn", "kt",
    "kr", "damping", "gamma", "zeta", "adhesion", "energy", "young", "youngs",
    "poisson", "rolling", "omega", "viscos", "viscous", "cohesion", "hertz",
    "spring", "contact", "bond", "stiff",
})

#: Names that are intentionally *not* contact parameters but match the hints.
NON_CONTACT_ALLOWLIST: frozenset[str] = frozenset({
    "mu_i", "mu_w", "mu_eff", "mu_ratio", "mu_s",  # rheology closures audited separately
    "gamma_dot", "gamma_max", "mu_0", "mu_inf", "mu_min", "mu_max",  # sweep bounds
    "k_none", "seed", "omega_tol_max",
})


def normalise(name: str) -> str:
    return name.strip().lower().lstrip("_")


def tokens(name: str) -> tuple[str, ...]:
    return tuple(t for t in normalise(name).split("_") if t)


def contact_like(name: str) -> bool:
    """Token-based contact-likeness test (avoids 'thickness' ~ 'k_n' noise)."""
    toks = tokens(name)
    if not toks:
        return False
    if set(toks) & CONTACT_HINT_TOKENS:
        return True
    # abbreviations such as "k_n", "mu_w", "kn" that tokenise to short symbols
    return any(len(t) <= 3 and t in {"kn", "kt", "kr", "mu", "en", "et"}
               for t in toks)


# ---------------------------------------------------------------------------
# 2. Similarity mode and required exponents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityMode:
    """Base-unit scale exponents for a similarity family.

    ``length``: exponent of the geometric scale lambda applied to the base unit
    of length.  ``density``/``gravity``: the exponent of the *base unit* of mass
    density and of the gravitational acceleration (0 keeps them fixed, which is
    the DEM coarse-graining convention).

    Derived unit exponents (``q`` with dimension ``M^a L^b T^c``):

        M_exp = density_exp + 3 * length_exp
        T_exp = (length_exp - gravity_exp) / 2      (from g = L/T^2)
        exp(q) = a*M_exp + b*length_exp + c*T_exp
    """

    name: str = "froude_gravity"
    length: float = 1.0
    density: float = 0.0
    gravity: float = 0.0

    # -- derived ---------------------------------------------------------
    @property
    def mass(self) -> float:
        return self.density + 3.0 * self.length

    @property
    def time(self) -> float:
        return 0.5 * (self.length - self.gravity)

    def exponent(self, dim: Dimension) -> float:
        return dim.m * self.mass + dim.l * self.length + dim.t * self.time

    def exponent_of(self, name: str) -> float:
        key = normalise(name)
        if key not in PARAMETERS:
            raise KeyError(f"unregistered parameter {name!r}: cannot infer exponent")
        return self.exponent(PARAMETERS[key].dimension)

    def describe(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "length_exp": self.length,
            "density_exp": self.density,
            "gravity_exp": self.gravity,
            "mass_exp": self.mass,
            "time_exp": self.time,
        }


FROUDE = SimilarityMode()


def exponent_table(mode: SimilarityMode = FROUDE) -> dict[str, dict[str, Any]]:
    """Required scale exponent of every registered parameter."""
    table: dict[str, dict[str, Any]] = {}
    for name, p in sorted(PARAMETERS.items()):
        table[name] = {
            "unit": p.unit,
            "dimension": str(p.dimension),
            "convention": p.convention,
            "exponent": p.required_exponent(mode),
            "alternative_dimension": (
                str(p.alternative) if p.alternative is not None else None),
            "alternative_exponent": (
                mode.exponent(p.alternative) if p.alternative is not None else None),
        }
    return table


# ---------------------------------------------------------------------------
# 3. Fleet extraction (AST, no imports)
# ---------------------------------------------------------------------------

_NUMERIC_NODES = (ast.Constant, ast.UnaryOp, ast.BinOp)


def _literal_value(node: ast.AST) -> float | None:
    """Best-effort numeric evaluation of a default expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        return -inner if inner is not None else None
    if isinstance(node, ast.BinOp):
        left = _literal_value(node.left)
        right = _literal_value(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div) and right != 0.0:
            return left / right
        if isinstance(node.op, ast.Pow):
            try:
                return left ** right
            except (OverflowError, ValueError):
                return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
    return None


@dataclass(frozen=True)
class FieldRecord:
    """One class field with a literal default."""

    module: str            # repo-relative path
    cls: str
    name: str              # field name as written
    canonical: str         # normalised name
    value: float
    line: int
    registered: bool
    dimension: str | None
    unit: str | None
    confidence: str | None = None
    ambiguous: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module, "class": self.cls, "field": self.name,
            "canonical": self.canonical, "value": self.value, "line": self.line,
            "registered": self.registered, "dimension": self.dimension,
            "unit": self.unit, "confidence": self.confidence,
            "ambiguous": self.ambiguous,
        }


@dataclass
class Violation:
    kind: str
    severity: str          # "error" | "warn" | "info"
    module: str
    location: str          # class.field or function
    line: int
    message: str
    value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "severity": self.severity, "module": self.module,
            "location": self.location, "line": self.line,
            "message": self.message, "value": self.value,
        }


def _module_fields(tree: ast.Module, rel: str) -> list[FieldRecord]:
    out: list[FieldRecord] = []
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for node in classes:
        for stmt in node.body:
            targets: list[tuple[str, ast.AST]] = []
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value is not None:
                    targets.append((stmt.target.id, stmt.value))
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        targets.append((tgt.id, stmt.value))
            for name, value_node in targets:
                if not isinstance(value_node, _NUMERIC_NODES):
                    continue
                value = _literal_value(value_node)
                if value is None:
                    continue
                canon = normalise(name)
                label = label_parameter(canon)
                out.append(FieldRecord(
                    module=rel, cls=node.name, name=name, canonical=canon,
                    value=value, line=getattr(stmt, "lineno", 0),
                    registered=label is not None,
                    dimension=str(label.dimension) if label else None,
                    unit=label.unit if label else None,
                    confidence=label.confidence if label else None,
                    ambiguous=label.ambiguous if label else None,
                ))
    return out


_SIZE_FACTORS = ("r", "radius", "r_eff", "reff", "diameter", "d", "rmean",
                 "r_mean", "r_eff_m")


def _rolling_usage_sites(tree: ast.Module,
                         names: Iterable[str]) -> list[tuple[int, bool, str]]:
    """Find multiplications that consume a rolling-resistance symbol.

    Returns ``(line, has_length_factor, expression_text)``.  ``has_length_factor``
    is True when the product chain contains a radius-like symbol, i.e. the
    consumption site uses the *dimensionless ratio* convention.
    """
    wanted = {normalise(n) for n in names}
    sites: list[tuple[int, bool, str]] = []

    def identifiers(node: ast.AST) -> set[str]:
        ids: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                ids.add(normalise(sub.id))
            elif isinstance(sub, ast.Attribute):
                ids.add(normalise(sub.attr))
        return ids

    def text(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:  # pragma: no cover - unparse is py>=3.9
            return "<expr>"

    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        ids = identifiers(node)
        if not (ids & wanted):
            continue
        has_len = bool(ids & set(_SIZE_FACTORS))
        sites.append((getattr(node, "lineno", 0), has_len, text(node)))
    return sites


def _silent_clamp_sites(tree: ast.Module, name: str) -> list[tuple[int, str]]:
    """Detect ``min(<name>, ...)`` style silent clamping of an out-of-range value."""
    out: list[tuple[int, str]] = []
    key = normalise(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("min", "max", "clip"):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Attribute) and normalise(sub.attr) == key:
                        out.append((getattr(node, "lineno", 0), ast.unparse(node)))
                    elif isinstance(sub, ast.Name) and normalise(sub.id) == key:
                        out.append((getattr(node, "lineno", 0), ast.unparse(node)))
    return out


@dataclass
class FleetScan:
    root: str
    n_modules_scanned: int
    in_scope: list[str]
    fields: list[FieldRecord]
    violations: list[Violation]
    n_unregistered_info: int = 0

    # -- views -----------------------------------------------------------
    def by_parameter(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        view: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for f in self.fields:
            if not f.registered:
                continue
            view.setdefault(f.canonical, {}).setdefault(f.module, []).append(
                {"class": f.cls, "field": f.name, "value": f.value,
                 "line": f.line})
        return view

    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for v in self.violations:
            kinds[v.kind] = kinds.get(v.kind, 0) + 1
        return {
            "root": self.root,
            "modules_scanned": self.n_modules_scanned,
            "modules_in_scope": len(self.in_scope),
            "fields_extracted": len(self.fields),
            "fields_registered": sum(1 for f in self.fields if f.registered),
            "violations_by_kind": kinds,
            "non_contact_fields_untagged": self.n_unregistered_info,
            "violations_total": len(self.violations),
            "errors_total": len(self.errors()),
            "rolled_up_parameters": len(self.by_parameter()),
        }

    def as_json(self, mode: SimilarityMode = FROUDE) -> dict[str, Any]:
        return {
            "schema": "tensordem.contact_param_ledger/1",
            "generated_by": "tensordem.contact_param_ledger.scan_fleet",
            "similarity_mode": mode.describe(),
            "summary": self.summary(),
            "in_scope_modules": sorted(self.in_scope),
            "parameters": {
                name: {
                    "unit": p.unit,
                    "dimension": str(p.dimension),
                    "convention": p.convention,
                    "required_exponent": p.required_exponent(mode),
                    "range": [p.lower, p.upper],
                    "alternative_dimension": (
                        str(p.alternative) if p.alternative is not None else None),
                    "sites": self.by_parameter().get(name, {}),
                }
                for name, p in sorted(PARAMETERS.items())
            },
            "violations": [v.as_dict() for v in self.violations],
        }


def scan_fleet(root: str | Path = ".",
               package: str = "src/tensordem") -> FleetScan:
    """Extract contact parameters from ``package`` and audit them."""
    root_path = Path(root)
    base = root_path / package
    if not base.is_dir():  # allow being called from inside the package tree
        base = root_path
    files = sorted(p for p in base.rglob("*.py"))
    in_scope: list[str] = []
    fields: list[FieldRecord] = []
    violations: list[Violation] = []
    n_unregistered_info = 0

    for path in files:
        rel = str(path.relative_to(root_path)) if path.is_relative_to(root_path) \
            else str(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        records = _module_fields(tree, rel)
        # scope rule: a module that declares at least one registered contact
        # parameter is a *case-bearing* module and is audited fail-closed.
        if not any(r.registered for r in records):
            continue
        in_scope.append(rel)
        fields.extend(records)

        registered_here = {r.canonical for r in records if r.registered}
        for r in records:
            if r.registered:
                continue
            canon = r.canonical
            if canon in NON_CONTACT_ALLOWLIST:
                continue
            if contact_like(canon):
                violations.append(Violation(
                    kind="UNREGISTERED_CONTACT_PARAMETER", severity="error",
                    module=rel, location=f"{r.cls}.{r.name}", line=r.line,
                    message=("contact-like field has no dimension label in "
                             "PARAMETERS; register it (fail-closed)"),
                    value=r.value))
            else:
                n_unregistered_info += 1

        # range / convention checks on registered fields
        for r in records:
            if not r.registered:
                continue
            param = PARAMETERS.get(r.canonical)
            if param is None:
                continue  # alias / rule label: no range contract declared
            if not param.in_range(r.value):
                violations.append(Violation(
                    kind="OUT_OF_RANGE", severity="error", module=rel,
                    location=f"{r.cls}.{r.name}", line=r.line,
                    message=(f"{r.canonical} = {r.value:g} {param.unit} outside "
                             f"valid range [{param.lower}, {param.upper}]"),
                    value=r.value))
        # silent clamping of the restitution (hides an out-of-range input)
        if "restitution" in source:
            for line, expr in _silent_clamp_sites(tree, "restitution"):
                violations.append(Violation(
                    kind="SILENT_CLAMP", severity="warn", module=rel,
                    location="restitution", line=line,
                    message=(f"restitution is silently clamped ({expr}); "
                             "e > 1 or e < 0 would be hidden"), value=None))
        # rolling-resistance convention consistency
        rolling_names = registered_here & {"mu_rolling", "mu_r", "rolling_mu",
                                           "rolling_lever"}
        if rolling_names:
            for name in sorted(rolling_names):
                param = PARAMETERS[name]
                wants_length = param.dimension.l == 1.0
                for line, has_len, expr in _rolling_usage_sites(tree, [name]):
                    if has_len and wants_length:
                        violations.append(Violation(
                            kind="CONVENTION_MISMATCH", severity="error",
                            module=rel, location=name, line=line,
                            message=(f"{name} is declared as a lever arm [m] but "
                                     f"is multiplied by a length in `{expr}` "
                                     "(double length scaling)"), value=None))
                    elif (not has_len) and (not wants_length):
                        violations.append(Violation(
                            kind="CONVENTION_MISMATCH", severity="warn",
                            module=rel, location=name, line=line,
                            message=(f"{name} is declared dimensionless but is "
                                     f"consumed without a radius factor in "
                                     f"`{expr}` - check the convention"), value=None))
    # fleet-level ambiguity roll-up: a label that cannot be attached is a
    # finding in itself, so it is reported once per parameter with evidence.
    ambig: dict[str, list[FieldRecord]] = {}
    for r in fields:
        if r.registered and r.ambiguous:
            ambig.setdefault(r.canonical, []).append(r)
    for name, recs in sorted(ambig.items()):
        why = recs[0].ambiguous
        why = why[2] if isinstance(why, tuple) else str(why)
        values = [r.value for r in recs]
        spread = (min(values) < 1.0 < max(values)) or (
            max(values) / max(min(values), 1e-12) > 100.0)
        modules = sorted({r.module for r in recs})
        violations.append(Violation(
            kind="DIMENSION_AMBIGUITY", severity="warn",
            module=modules[0] if len(modules) == 1 else f"{len(modules)} modules",
            location=name, line=recs[0].line,
            message=(f"{why}; {len(recs)} site(s) in "
                     f"{len(modules)} module(s); values {min(values):g}..{max(values):g}"
                     + (" [spans both conventions]" if spread else "")),
            value={"min": min(values), "max": max(values),
                   "modules": modules}))

    scan = FleetScan(root=str(root_path), n_modules_scanned=len(files),
                     in_scope=in_scope, fields=fields, violations=violations)
    scan.n_unregistered_info = n_unregistered_info
    return scan


# ---------------------------------------------------------------------------
# 4. Similarity transform and variant audit (the CI gate core)
# ---------------------------------------------------------------------------


def scale_value(name: str, value: float | None, lam: float,
                mode: SimilarityMode = FROUDE,
                override_exponent: float | None = None) -> float | None:
    """Scale one parameter by ``lam ** exponent`` (identity for ``None``)."""
    if value is None:
        return None
    exp = mode.exponent_of(name) if override_exponent is None else override_exponent
    return value * lam ** exp


def scale_case_json(params: Mapping[str, Any], lam: float,
                    mode: SimilarityMode = FROUDE) -> dict[str, float]:
    """Apply the similarity transform to a mapping of parameters.

    Raises ``KeyError`` for an unregistered parameter (fail-closed: an
    unlabelled dimension must not be silently left unscaled).
    """
    out: dict[str, float] = {}
    for name, value in params.items():
        if value is None or isinstance(value, (str, bool)):
            out[name] = value
            continue
        out[name] = scale_value(name, float(value), lam, mode)
    return out


def audit_variant(base: Mapping[str, float], variant: Mapping[str, float],
                  lam: float, mode: SimilarityMode = FROUDE,
                  rtol: float = 1e-9) -> list[dict[str, Any]]:
    """Check a declared scaled variant against the similarity law.

    Returns one row per parameter with the observed exponent
    ``log(variant/base) / log(lam)`` and the required one.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(set(base) | set(variant)):
        param = PARAMETERS.get(normalise(name))
        if param is None:
            rows.append({"parameter": name, "status": "UNREGISTERED",
                         "message": "no dimension label; transform cannot be audited"})
            continue
        if name not in base or name not in variant:
            rows.append({"parameter": name, "status": "MISSING",
                         "message": "parameter present on one side only"})
            continue
        b, v = base[name], variant[name]
        if b is None or v is None or isinstance(b, str) or isinstance(v, str):
            continue
        required = param.required_exponent(mode)
        if b == 0.0:
            observed = 0.0 if v == 0.0 else math.inf
        elif v <= 0.0:
            observed = -math.inf
        else:
            observed = math.log(v / b) / math.log(lam)
        ok = abs(observed - required) <= 1e-6 or (
            abs(b) > 0 and abs(v - b * lam ** required) <= rtol * abs(v))
        rows.append({
            "parameter": name,
            "unit": param.unit,
            "dimension": str(param.dimension),
            "base": b,
            "variant": v,
            "required_exponent": required,
            "observed_exponent": observed,
            "status": "OK" if ok else "VIOLATION",
        })
    return rows


def audit_manifest(manifest: Mapping[str, Any], mode: SimilarityMode = FROUDE,
                   require_all_ok: bool = True) -> dict[str, Any]:
    """Audit a JSON manifest ``{cases: [{name, lam, base, variants: {...}}]}``.

    ``variants`` maps a variant label to either a parameter mapping (expected to
    follow the similarity law) or ``{"expected":"violation", "params": {...}}``
    for a *negative control* that must break the law.
    """
    report: dict[str, Any] = {"cases": [], "status": "OK", "n_violations": 0}
    for case in manifest.get("cases", []):
        lam = float(case["lam"])
        entry: dict[str, Any] = {"case": case["name"], "lam": lam,
                                 "variants": []}
        for label, spec in case.get("variants", {}).items():
            expected = "OK"
            params = spec
            if isinstance(spec, Mapping) and "params" in spec:
                expected = str(spec.get("expected", "OK"))
                params = spec["params"]
            rows = audit_variant(case["base"], params, lam, mode)
            hard = [r for r in rows if r.get("status") == "VIOLATION"]
            unreg = [r for r in rows if r.get("status") == "UNREGISTERED"]
            if expected == "violation":
                # negative control: it must be caught
                caught = bool(hard or unreg)
                status = "CAUGHT" if caught else "MISSED"
            else:
                status = "OK" if not hard and not unreg else "VIOLATION"
            if status in ("VIOLATION", "MISSED") and require_all_ok:
                report["status"] = "FAIL"
                report["n_violations"] += 1
            entry["variants"].append({
                "variant": label, "expected": expected, "status": status,
                "deviating_parameters": [r["parameter"] for r in hard],
                "unregistered_parameters": [r["parameter"] for r in unreg],
                "rows": rows,
            })
        report["cases"].append(entry)
    return report


#: Field-name patterns copied from a case class into the ledger.  Used by the
#: M2/M3 driver so that the *simulation* cases are scaled through exactly the
#: same exponent table as the *scanned* fleet cases.
def scale_dataclass(case: Any, lam: float,
                    mode: SimilarityMode = FROUDE,
                    skip: Sequence[str] = ("seed", "steps")) -> Any:
    """Return a copy of a frozen dataclass with every field scaled by the law."""
    import dataclasses

    changes: dict[str, Any] = {}
    for f in dataclasses.fields(case):
        if f.name in skip:
            continue
        value = getattr(case, f.name)
        if value is None or isinstance(value, (str, bool)):
            continue
        if normalise(f.name) in PARAMETERS:
            changes[f.name] = scale_value(f.name, float(value), lam, mode)
    return dataclasses.replace(case, **changes)


def write_ledger(root: str | Path, out: str | Path,
                 mode: SimilarityMode = FROUDE) -> dict[str, Any]:
    scan = scan_fleet(root)
    payload = scan.as_json(mode)
    payload["exponent_table"] = exponent_table(mode)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False),
                        encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="artifacts/contact_scale_ledger/ledger.json")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any error-severity violation is found")
    args = ap.parse_args(argv)
    payload = write_ledger(args.root, args.out)
    summary = payload["summary"]
    print(json.dumps(summary, indent=2))
    print(f"ledger written to {args.out}")
    if args.check and summary["errors_total"] > 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Dimension", "Parameter", "PARAMETERS", "SimilarityMode", "FROUDE",
    "exponent_table", "scale_value", "scale_case_json", "scale_dataclass",
    "audit_variant", "audit_manifest", "scan_fleet", "write_ledger", "main",
    "FieldRecord", "Violation", "FleetScan", "ROLLING_CONVENTIONS",
]