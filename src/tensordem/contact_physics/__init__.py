"""DEM contact physics: bond + lubrication + statistics behind one stable API.

This sub-package is the **maturation layer** for three campaign deliverable
nodes of the ice-navigation campaign (``P121W5``):

===================  ==================================================
node                 physics
===================  ==================================================
``P121W5-N02``       temperature-dependent freeze bond, module
                     :mod:`tensordem.contact_physics.bond`
``P121W5-N05``       ice-ridge morphology / contact-network statistics,
                     module :mod:`tensordem.contact_physics.statistics`
``P121W5-N08``       melt-lubrication <-> contact two-way coupling, module
                     :mod:`tensordem.contact_physics.lubrication`
===================  ==================================================

and :mod:`tensordem.contact_physics.march` runs all three on one operating
point (:class:`OperatingPoint`) in a single loop
(:class:`ContactPhysicsMarch`).

Design rules
------------
1. **No physics is re-implemented.**  Every number is produced by the verified
   upstream module named in the adapter's docstring; the adapters only fix names,
   keyword defaults and the record layout.
2. **No upstream signature changes.**  The layer is purely additive: the
   modules, functions and signatures of the three source modules are frozen and
   a test compares their file hashes and ``inspect.signature`` strings against a
   committed baseline.
3. **One operating point, one loop.**  The linkage is directed (lubrication
   temperature -> bond damage clock, bond forces -> statistics sampler) which is
   what keeps an integrated run bit-identical to the three standalone runs.

Quick start
-----------
>>> from tensordem.contact_physics import ContactPhysicsMarch, OperatingPoint
>>> march = ContactPhysicsMarch(OperatingPoint())
>>> summary = march.run(50)
>>> summary["steps"], summary["energy"]["ledger_error"]  # doctest: +SKIP
(50, 0.0)

See ``docs/contact_physics_module.md`` for the verification table, the
performance benchmark and the known limitations.
"""
from __future__ import annotations

from .bond import TemperatureDependentBond
from .lubrication import LubricationContactCoupling
from .march import ContactPhysicsMarch, OperatingPoint
from .statistics import RidgeContactStatistics, RidgeMorphologyCase

__all__ = [
    "TemperatureDependentBond",
    "LubricationContactCoupling",
    "RidgeContactStatistics",
    "RidgeMorphologyCase",
    "OperatingPoint",
    "ContactPhysicsMarch",
]

__version__ = "1.0.0"