"""Melt-water lubricated ice/steel friction on the DEM contact surface.

Per-contact internal variables ``{T_i, h}`` with the Bowden-Hughes interface
energy balance, Greenwood-Williamson asperity load sharing and Reynolds squeeze
drainage; friction is a solved output, ``mu = f(h) mu_asp + (1-f) mu_film``.

Public surface
--------------
``MaterialPair``      thermophysical + surface-texture inputs (nothing fitted)
``MeltFilmContact``   contact state machine: ``solve_steady`` / ``march``
``solve_steady``      convenience steady solve at (V, sigma, T_bulk)
``run_campaign_sweep``  the sigma x V x T acceptance sweep and criteria C1-C7
"""

from .contact import ContactResult, ContactState, MeltFilmContact, solve_steady
from .physics import MaterialPair
from .sweep import evaluate_criteria, run_campaign_sweep

__all__ = [
    "MaterialPair",
    "ContactState",
    "ContactResult",
    "MeltFilmContact",
    "solve_steady",
    "run_campaign_sweep",
    "evaluate_criteria",
]
