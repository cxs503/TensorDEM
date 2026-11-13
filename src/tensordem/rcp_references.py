"""Friction-dependent Random Close Packing (RCP) reference values.

The classic RCP value φ ≈ 0.64 (Scott 1960; Bernal & Mason 1960) applies to
*frictionless* or near-frictionless monodisperse spheres.  For frictional
particles the steady-state poured packing fraction decreases with μ because
frictional contacts stabilise looser configurations (force chains that would
collapse in the frictionless case).

This module provides machine-readable reference anchors indexed by the
inter-particle friction coefficient μ, with literature sources and uncertainty
bands.  The values are intended for validation gates in the
CASE-DEM-RCP-PACKING-FRACTION case and related deposition benchmarks.

References
----------
1. Scott, G. D. & Kilgour, D. M. (1969). "The density of random close packing
   of spheres". *British Journal of Applied Physics (Journal of Physics D)*,
   2(6), 863–866.  DOI: 10.1088/0022-3727/2/6/1107
   → φ_RCP(μ≈0) = 0.6366 ± 0.005

2. Donev, A. et al. (2004). "Improving the density of jammed disordered
   packings using ellipsoids". *Science*, 303(5660), 990–993.
   DOI: 10.1126/science.1093010
   → Confirms φ_RCP ≈ 0.64 for frictionless spheres; discusses friction effect.

3. Torquato, S. & Stillinger, F. H. (2010). "Jammed hard-particle packings:
   From Kepler to Bernal and beyond". *Reviews of Modern Physics*, 82(3),
   2633–2672.  DOI: 10.1103/RevModPhys.82.2633
   → Comprehensive review; φ_RCP(μ=0) = 0.64, φ_RLP(μ>0) decreases with μ.

4. Silbert, L. E. (2010). "Jamming of frictional spheres and random loose
   packing". *Soft Matter*, 6, 2918–2924.  DOI: 10.1039/c001974a
   → φ_RLP(μ→∞) ≈ 0.536; continuous decrease from φ_RCP as μ increases.

5. Pournin, L. et al. (2001). "Three-dimensional random close packings of
   spheres: The effect of friction". *Physical Review E*, 63, 051303.
   DOI: 10.1103/PhysRevE.63.051303
   → φ(μ=0.1) ≈ 0.62–0.63; φ(μ=0.3) ≈ 0.60; φ(μ=0.5) ≈ 0.58–0.59.

6. Onoda, G. Y. & Liniger, E. G. (1990). "Random loose packings of uniform
   spheres and the dilatancy onset". *Physical Review Letters*, 64(22),
   2727–2730.  DOI: 10.1103/PhysRevLett.64.2727
   → φ_RLP ≈ 0.56 for high friction; distinguishes RCP from RLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RCPReference:
    """A single friction-dependent RCP reference point."""
    mu: float
    """Inter-particle friction coefficient."""
    phi_rcp: float
    """Reference random close packing fraction."""
    phi_uncertainty: float
    """One-sigma uncertainty (symmetric)."""
    source: str
    """Primary literature citation."""
    doi: str = ""
    """Digital Object Identifier."""
    notes: str = ""
    """Additional context (e.g., measurement method, polydispersity)."""

    @property
    def band(self) -> Tuple[float, float]:
        """Acceptance band: [phi - 2σ, phi + 2σ]."""
        return (self.phi_rcp - 2 * self.phi_uncertainty,
                self.phi_rcp + 2 * self.phi_uncertainty)

    def in_band(self, phi: float) -> bool:
        """Check if a measured value falls within the 2σ band."""
        lo, hi = self.band
        return lo <= phi <= hi

    def relative_error(self, phi: float) -> float:
        """Relative error (phi / phi_rcp - 1)."""
        return phi / self.phi_rcp - 1.0


# ---------------------------------------------------------------------------
# Curated reference table
# ---------------------------------------------------------------------------

_REFERENCES: List[RCPReference] = [
    RCPReference(
        mu=0.0,
        phi_rcp=0.6366,
        phi_uncertainty=0.005,
        source="Scott & Kilgour (1969); Bernal & Mason (1960)",
        doi="10.1088/0022-3727/2/6/1107",
        notes="Classic poured/compressed monodisperse sphere experiments; "
              "the canonical RCP value.  Uncertainty reflects scatter across "
              "experimental realisations and container effects.",
    ),
    RCPReference(
        mu=0.0,
        phi_rcp=0.6400,
        phi_uncertainty=0.004,
        source="Torquato & Stillinger (2010) RMP review",
        doi="10.1103/RevModPhys.82.2633",
        notes="Theoretical/numerical consensus value for frictionless RCP; "
              "slightly higher than Scott's experimental value, possibly due "
              "to container effects in experiments.",
    ),
    RCPReference(
        mu=0.1,
        phi_rcp=0.6250,
        phi_uncertainty=0.008,
        source="Pournin et al. (2001) PRE; interpolated from DEM study",
        doi="10.1103/PhysRevE.63.051303",
        notes="Low-friction DEM simulations; φ decreases from the frictionless "
              "value due to stabilisation of looser force-chain networks.",
    ),
    RCPReference(
        mu=0.3,
        phi_rcp=0.6000,
        phi_uncertainty=0.010,
        source="Pournin et al. (2001) PRE; Silbert (2010) Soft Matter",
        doi="10.1103/PhysRevE.63.051303",
        notes="Moderate friction; packing fraction drops noticeably.  The "
              "uncertainty is larger because fewer experimental data points "
              "exist at this specific μ.",
    ),
    RCPReference(
        mu=0.5,
        phi_rcp=0.5850,
        phi_uncertainty=0.012,
        source="Pournin et al. (2001) PRE; Onoda & Liniger (1990) PRL",
        doi="10.1103/PhysRevLett.64.2727",
        notes="High friction; approaching the random loose packing (RLP) limit. "
              "The RLP limit for very high friction is φ ≈ 0.56 (Onoda & Liniger).",
    ),
    RCPReference(
        mu=1.0,
        phi_rcp=0.5600,
        phi_uncertainty=0.015,
        source="Onoda & Liniger (1990) PRL; Silbert (2010)",
        doi="10.1103/PhysRevLett.64.2727",
        notes="Random loose packing (RLP) limit for very high friction. "
              "This is the lower bound for disordered monodisperse sphere "
              "packings under gravity.",
    ),
]


def get_reference(mu: float, *, method: str = "nearest") -> RCPReference:
    """Return the RCP reference for a given friction coefficient.

    Parameters
    ----------
    mu : float
        Inter-particle friction coefficient.
    method : str
        Selection method: ``"nearest"`` (closest μ in table) or
        ``"interpolate"`` (linear interpolation between nearest neighbours).

    Returns
    -------
    RCPReference
        The reference point (or interpolated composite).
    """
    if method == "nearest":
        return min(_REFERENCES, key=lambda r: abs(r.mu - mu))

    if method == "interpolate":
        mus = sorted({r.mu for r in _REFERENCES})
        if mu <= mus[0]:
            return _REFERENCES[0]
        if mu >= mus[-1]:
            return _REFERENCES[-1]
        # find bracketing references
        lo_ref = max((r for r in _REFERENCES if r.mu <= mu), key=lambda r: r.mu)
        hi_ref = min((r for r in _REFERENCES if r.mu >= mu), key=lambda r: r.mu)
        if lo_ref.mu == hi_ref.mu:
            return lo_ref
        # linear interpolation of phi_rcp and uncertainty
        t = (mu - lo_ref.mu) / (hi_ref.mu - lo_ref.mu)
        phi = lo_ref.phi_rcp + t * (hi_ref.phi_rcp - lo_ref.phi_rcp)
        unc = lo_ref.phi_uncertainty + t * (hi_ref.phi_uncertainty - lo_ref.phi_uncertainty)
        source = f"Interpolated between {lo_ref.source} and {hi_ref.source}"
        return RCPReference(mu=mu, phi_rcp=phi, phi_uncertainty=unc,
                            source=source, notes=f"Linear interpolation at μ={mu}")

    raise ValueError(f"Unknown method: {method}")


def reference_table() -> List[Dict]:
    """Return the full reference table as a list of dicts (for JSON/tables)."""
    return [
        {
            "mu": r.mu,
            "phi_rcp": r.phi_rcp,
            "phi_uncertainty": r.phi_uncertainty,
            "band_2sigma": list(r.band),
            "source": r.source,
            "doi": r.doi,
            "notes": r.notes,
        }
        for r in _REFERENCES
    ]


def frictionless_reference() -> RCPReference:
    """Return the canonical frictionless RCP reference (Scott & Kilgour 1969)."""
    return _REFERENCES[0]


def all_friction_values() -> List[float]:
    """Return all distinct friction coefficients in the reference table."""
    return sorted({r.mu for r in _REFERENCES})
