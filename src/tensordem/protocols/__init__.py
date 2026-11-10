"""Frozen experimental protocols that turn DEM contact laws into macro anchors.

Modules
-------
``sandpile``
    Standardised sandpile pour: constant mass flow rate, fixed drop height,
    edge-wall exclusion, linear free-surface fit, probe-volume packing fraction
    and an energy ledger.  Produces the angle of repose, the deposit packing
    fraction and the contact fabric that the campaign gates on.
"""
from . import sandpile  # noqa: F401

__all__ = ["sandpile"]