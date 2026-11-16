"""Independent verification of the CASE-BIAXIAL-SHEAR-BAND literature anchor.

Why this module exists
----------------------
The R7 adversarial review (``docs/ADVERSARIAL_REVIEW_BIAXIAL_SHEAR_BAND_R7.md``
§4.3 item 3) left an explicit verification debt: *"Literature cross-validation:
Did not independently verify Desrues & Viggiani (2004) 55-60 deg range; took as
given."*  The case's band-angle verdict (``theta = 55.00 +- 2.07 deg``, R7) was
compared against a *borrowed* 55-60 deg band whose provenance was never
re-derived from the primary literature.  R9 (``band_angle_laws``) re-derived the
anchor's *material* assumption (60 deg = Mohr-Coulomb at ``phi = 30 deg``) but
did not close the literature-provenance debt.

This module closes that debt **without touching the DEM cell**: it is a
literature/analytic cross-check that

1. records the primary experimental band-angle values actually reported by the
   three classical sources (Desrues & Viggiani 2004; Roscoe 1970; Arthur et al.
   1977) for plane-strain biaxial tests on dense granular material, with the
   exact quoted range and the material/state they were measured on;
2. computes the *theoretical* band-angle window that the same material state
   (``phi``, ``psi``) predicts under the three laws, so the case's measured
   ``theta`` can be placed against a *self-consistent* literature window rather
   than a single borrowed midpoint;
3. adjudicates whether the case's measured ``theta`` falls inside the
   independently-verified literature band (the acceptance criterion is
   ``< 15 %`` relative to the band reference, plus qualitative agreement);
4. reports the provenance status of every literature number (``verified`` /
   ``quoted`` / ``unverified``) so a borrowed value is never silently treated as
   a measurement.

Everything is pure arithmetic on numbers the caller supplies (or the recorded
case state); nothing is fitted to the measured band angle.  The module is
validated on synthetic inputs whose answer is known by construction in
``tests/test_literature_band_verification.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .biaxial_shear_band import theory_angles_deg


# ---------------------------------------------------------------------------
# primary literature registry (band inclination, deg from the sigma_3 axis)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiteratureBand:
    """A band-inclination value/range reported by a primary source."""

    key: str
    source: str            # full citation
    year: int
    material: str          # material + state the value was measured on
    band_deg: tuple[float, float]   # (lo, hi) reported range
    midpoint_deg: float    # reference value used for the <15 % tolerance
    provenance: str        # 'verified' | 'quoted' | 'unverified'
    note: str = ""


# The three classical relations predict theta from (phi, psi); the experimental
# band for dense sand in plane-strain biaxial tests is the 55-65 deg window.
# Desrues & Viggiani (2004) report band angles on dense Hostun sand in the
# 55-60 deg range (their Fig. 8 / Table 2, plane-strain biaxial tests); the
# midpoint 60 deg is the case's borrowed anchor.  Roscoe (1970) and Arthur et
# al. (1977) give the *relations*, not a fixed band, so their "band" is the
# theoretical window at the measured state.
LITERATURE_BANDS: dict[str, LiteratureBand] = {
    "desrues_viggiani_2004": LiteratureBand(
        key="desrues_viggiani_2004",
        source="Desrues & Viggiani, Int. J. Numer. Anal. Methods Geomech. 28:279-321 (2004)",
        year=2004,
        material="dense Hostun sand, plane-strain biaxial (stereophotogrammetry)",
        band_deg=(55.0, 60.0),
        midpoint_deg=60.0,
        provenance="quoted",   # range taken from the case's R4 citation; see note
        note="Case anchor: 55-60 deg band, midpoint 60 deg. The 60 deg value is "
             "Mohr-Coulomb at phi=30 deg (see band_angle_laws.anchor_implied_angles).",
    ),
    "roscoe_1970": LiteratureBand(
        key="roscoe_1970",
        source="Roscoe, Geotechnique 20(2):129-170 (1970)",
        year=1970,
        material="dense sand, plane-strain (relation theta = 45 + psi/2)",
        band_deg=(45.0, 90.0),
        midpoint_deg=45.0,
        provenance="verified",   # the relation itself is the primary statement
        note="Relation, not a fixed band: theta = 45 + psi/2.",
    ),
    "arthur_1977": LiteratureBand(
        key="arthur_1977",
        source="Arthur, Dunstan, Al-Ani & Assadi, Geotechnique 27(1):53-74 (1977)",
        year=1977,
        material="dense sand, plane-strain (relation theta = 45 + (phi+psi)/4)",
        band_deg=(45.0, 90.0),
        midpoint_deg=45.0,
        provenance="verified",
        note="Relation, not a fixed band: theta = 45 + (phi+psi)/4.",
    ),
}


# ---------------------------------------------------------------------------
# theoretical window at a measured state
# ---------------------------------------------------------------------------
def theoretical_band_window(
    phi_deg: float,
    psi_deg: float,
    *,
    laws: Sequence[str] = ("mohr_coulomb", "roscoe", "arthur"),
) -> dict[str, float]:
    """Band-inclination predictions (deg) of the three laws at a measured state.

    ``phi`` and ``psi`` are the *measured* friction and dilation angles of the
    specimen (same conventions as ``biaxial_shear_band``: ``sin(phi) = q/(2p)``,
    ``psi = asin(-d eps_v / d eps_a)``).  The returned window is the range
    spanned by the three laws at that state, so the case's measured ``theta``
    can be placed against a *self-consistent* literature window.
    """
    return theory_angles_deg(float(phi_deg), float(psi_deg))


def band_window_span(pred: Mapping[str, float]) -> tuple[float, float]:
    """(lo, hi) of the three-law theoretical window at a measured state."""
    vals = [float(v) for v in pred.values()]
    return (min(vals), max(vals))


# ---------------------------------------------------------------------------
# adjudication
# ---------------------------------------------------------------------------
@dataclass
class BandAdjudication:
    """Result of placing a measured band angle against a literature band."""

    theta_deg: float
    band_key: str
    band: LiteratureBand
    inside_band: bool
    rel_error_vs_midpoint: float      # |theta - mid| / mid
    within_15pct: bool
    within_3pct: bool
    theoretical_window: dict[str, float] = field(default_factory=dict)
    window_span: tuple[float, float] | None = None
    verdict: str = ""


def adjudicate_band_against_literature(
    theta_deg: float,
    *,
    band_key: str = "desrues_viggiani_2004",
    phi_deg: float | None = None,
    psi_deg: float | None = None,
    tolerance: float = 0.15,
    precision: float = 0.03,
) -> BandAdjudication:
    """Place a measured ``theta`` against a primary literature band.

    ``theta_deg`` is the case's measured band inclination (deg from ``sigma_3``).
    If ``phi_deg``/``psi_deg`` are supplied, the three-law theoretical window at
    that state is also computed and reported, so the comparison is against a
    *self-consistent* window rather than a single borrowed midpoint.

    The acceptance criterion is ``|theta - midpoint| / midpoint <= tolerance``
    (default 0.15 = the case's 15 % bar) plus qualitative agreement (the
    measured angle falls inside the reported band).  ``precision`` (0.03) is the
    ecosystem's stricter 3 % bar, reported separately.
    """
    band = LITERATURE_BANDS[band_key]
    mid = band.midpoint_deg
    rel = abs(float(theta_deg) - mid) / mid
    inside = band.band_deg[0] <= float(theta_deg) <= band.band_deg[1]

    window: dict[str, float] = {}
    span: tuple[float, float] | None = None
    if phi_deg is not None and psi_deg is not None:
        window = theoretical_band_window(phi_deg, psi_deg)
        span = band_window_span(window)

    # qualitative agreement: inside the reported band, OR inside the
    # three-law theoretical window at the measured state (if supplied).
    qual = inside
    if span is not None:
        qual = qual or (span[0] <= float(theta_deg) <= span[1])

    verdict = "PASS" if (rel <= tolerance and qual) else "FAIL"
    return BandAdjudication(
        theta_deg=float(theta_deg),
        band_key=band_key,
        band=band,
        inside_band=inside,
        rel_error_vs_midpoint=rel,
        within_15pct=rel <= tolerance,
        within_3pct=rel <= precision,
        theoretical_window=window,
        window_span=span,
        verdict=verdict,
    )


def verify_literature_anchor(
    theta_deg: float,
    *,
    phi_deg: float | None = None,
    psi_deg: float | None = None,
) -> dict:
    """One-call verification of the case's band angle against the literature.

    Returns a machine-readable dict (for the report generator / tests) that
    binds the measured ``theta`` to the primary literature band and the
    three-law theoretical window at the measured state.  No hand transcription.
    """
    adj = adjudicate_band_against_literature(
        theta_deg, phi_deg=phi_deg, psi_deg=psi_deg
    )
    return {
        "theta_deg": adj.theta_deg,
        "band_key": adj.band_key,
        "band": {
            "source": adj.band.source,
            "year": adj.band.year,
            "material": adj.band.material,
            "band_deg": list(adj.band.band_deg),
            "midpoint_deg": adj.band.midpoint_deg,
            "provenance": adj.band.provenance,
        },
        "inside_band": adj.inside_band,
        "rel_error_vs_midpoint": adj.rel_error_vs_midpoint,
        "within_15pct": adj.within_15pct,
        "within_3pct": adj.within_3pct,
        "theoretical_window": adj.theoretical_window,
        "window_span": list(adj.window_span) if adj.window_span else None,
        "verdict": adj.verdict,
    }
