"""CLI entry point: ``python -m tensordem.fem_dem_fracture``.

Runs the FEM -> DEM fracture handoff end to end (static census, dynamic velocity
inheritance, DEM-side intake) and writes the artefact bundle plus a manifest with
the reproduction command and the SHA-256 of every artefact.
"""
from __future__ import annotations

from .study_dynamic import main

if __name__ == "__main__":
    raise SystemExit(main())