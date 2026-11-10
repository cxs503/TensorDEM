"""Reference design case: ice-class double-hull side structure.

The scantlings below are a *documented design case* for the verification surface
(a PC5 double-hull tanker side shell struck by an ice floe / boulder); they are
scenario data injected into the generic mechanics of
:mod:`tensordem.hull_collision`, not physics baked into the library.  Any other
scantling set can be built the same way by the caller.

Values (SI):
    outer shell      t = 22 mm, AH36 (sigma_y = 355 MPa), frame spacing 2.4 m
    longitudinal     T-bar 400x11 web + 150x15 flange with 600x22 attached plate
                     -> Z = 1.924e-3 m^3 (computed, see `longitudinal_Z`)
    inner shell      t = 18 mm, sigma_y = 355 MPa, spacing 2.4 m
    double-hull void distance from outer surface to inner shell outer face 2.022 m
    contact patch    3.0 m (along ship) x 3.0 m (vertical) -> effective width 3.0 m
    ice-steel friction coefficient 0.15 (upper end of the 0.05-0.15 range used for
    ice-structure contact)
"""
from __future__ import annotations

from .plastic_hinge import section_plastic_modulus
from .resistance import DoubleHullSide, SideShellLayer

__all__ = [
    "SCRATCH_SIGMA_Y",
    "STRIKING_MASS",
    "ADDED_MASS_COEFFICIENT",
    "effective_mass",
    "FRAME_SPAN",
    "OUTER_THICKNESS",
    "INNER_THICKNESS",
    "DOUBLE_HULL_DEPTH",
    "CONTACT_LENGTH",
    "CONTACT_HEIGHT",
    "FRICTION_COEFFICIENT",
    "longitudinal_section_modulus",
    "design_case",
    "overload_case",
]

SCRATCH_SIGMA_Y = 355.0e6
FRAME_SPAN = 2.4
OUTER_THICKNESS = 0.022
INNER_THICKNESS = 0.018
DOUBLE_HULL_DEPTH = 2.022
CONTACT_LENGTH = 3.0
CONTACT_HEIGHT = 3.0
FRICTION_COEFFICIENT = 0.15
#: ice floe / boulder mass of the striking body, and the hydrodynamic added-mass
#: coefficient of the struck ship's side (0.1 is the usual order for a
#: transverse impact of a large hull)
STRIKING_MASS = 2.0e6
ADDED_MASS_COEFFICIENT = 0.1


def effective_mass(striking_mass: float = STRIKING_MASS) -> float:
    """``m_striking * (1 + added_mass_coefficient)`` [kg]."""
    return float(striking_mass) * (1.0 + ADDED_MASS_COEFFICIENT)

#: T-bar longitudinal: (y_lower, y_upper, width) with y = 0 at the outer surface
_LONGITUDINAL_RECTANGLES = (
    (0.0, OUTER_THICKNESS, 0.600),          # attached plating
    (OUTER_THICKNESS, 0.422, 0.011),        # web (400 mm)
    (0.422, 0.437, 0.150),                  # flange (150 x 15)
)


def longitudinal_section_modulus() -> float:
    """Plastic section modulus of one longitudinal with attached plating [m^3]."""
    return section_plastic_modulus(_LONGITUDINAL_RECTANGLES)


def _longitudinal_bare_area() -> float:
    """Bare steel area of the T-bar (no attached plate) [m^2]."""
    return 0.400 * 0.011 + 0.150 * 0.015


def design_case() -> DoubleHullSide:
    """Outer + inner shell of the design case, engaged in series."""
    z_long = longitudinal_section_modulus()
    outer = SideShellLayer(
        name="outer_shell",
        thickness=OUTER_THICKNESS,
        sigma_y=SCRATCH_SIGMA_Y,
        frame_span=FRAME_SPAN,
        effective_width=CONTACT_HEIGHT,
        n_longitudinals=5,
        longitudinal_Z=z_long,
        longitudinal_area=_longitudinal_bare_area(),
        crush_depth=DOUBLE_HULL_DEPTH,
    )
    inner = SideShellLayer(
        name="inner_shell",
        thickness=INNER_THICKNESS,
        sigma_y=SCRATCH_SIGMA_Y,
        frame_span=FRAME_SPAN,
        effective_width=2.5,
        n_longitudinals=4,
        longitudinal_Z=0.0016,
        longitudinal_area=0.005,
        crush_depth=1.0,
    )
    return DoubleHullSide(layers=(outer, inner), friction_coefficient=FRICTION_COEFFICIENT)
