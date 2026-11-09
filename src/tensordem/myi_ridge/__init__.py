"""Multi-year ice (MYI) pressure-ridge keel interaction chain.

Modules
-------
``geometry``  ridge keel statistics (keel depth, porosity, block size) and the
              plane-strain keel cross-section generator;
``crushing``  plane-strain bonded-rubble DEM of the hull ice belt crushing the
              ridge keel, with a closed energy ledger;
``scaling``   extreme-load scaling law, monotonicity and the load envelope;
``panel``     ice-belt plate-panel FEM check of the DEM-derived load.
"""
from . import crushing, geometry, panel, scaling  # noqa: F401

__all__ = ["geometry", "crushing", "scaling", "panel"]
