"""Dimension-matched (2-D disk) reference law for CASE-DEM-PLANE-SHEAR-MUIDI.

Why this module exists
----------------------
Rounds 1-29 judged a **2-D disk** plane-shear cell against the triple carried by
``reference_law.py`` / ``reference_provenance.py``, ``(mu_s, mu_2, I_0) =
(0.36, 0.84, 0.30)`` (and its in-repo rival ``(0.379, 0.641, 0.279)``).  Both of
those are **3-D glass-bead** readings, and both are labelled in their own module
as QUOTED-/ATTRIBUTED-NOT-ARCHIVED.  The ledger acceptance, however, cites
*da Cruz et al. 2005* - which is itself a **2-D disk** simulation at contact
friction ``mu = 0.4``, i.e. exactly the device's ``mu_p = 0.4``.  The round-22
provenance archive (``runs/provenance_archive/``) holds the paper's text, and
round 25/27 already used the 2-D *dilatancy* numbers from it
(``phi_provenance_audit.py``); the 2-D *friction* numbers were never used,
because they are not a Jop triple - they are a **linear** law.

The paper's own statements (Eqs. 13-14), as archived:

    mu*(I) ~ mu*_min + b I,   mu*_min ~ 0.22, b ~ 1.0   (for mu != 0),
                              saturating for I >= 0.2                    (14)
    nu(I)  ~ nu_max - a I,    nu_max ~ 0.81, a ~ 0.3    (for mu = 0.4)    (13)

Nothing in this module is a new measurement or a fitted knob: the four numbers
are read out of the archived excerpt and *sentence-bound* by
:func:`verify_provenance`, which refuses to load the law unless each constant
still sits in the sentence that states it.  A gate that only checked "the number
occurs somewhere in the file" would pass on ``0.3`` in ``0.30`` - see
``test_r30_2d_reference.py`` for the tampered-text cases.

Scope / caveats (honest)
------------------------
* ``mu*_min = 0.22`` is stated by the paper for *any* ``mu != 0``, while
  ``nu_max = 0.81, a = 0.3`` are stated for ``mu = 0.4``.  The device runs
  ``mu_p = 0.4``, so the comparison is admissible, but the friction intercept is
  a weaker attribution than the dilatancy pair - recorded, not hidden.
* The linear law is the paper's own approximation; it "tends to saturate for
  ``I >= 0.2``" (quoted), so points at ``I >= 0.2`` are flagged rather than
  silently extrapolated.
* This module does **not** replace the 3-D envelope in
  ``reference_provenance.py``: archived round verdicts stay reproducible.  It is
  an *additional*, dimension-consistent yardstick.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

__all__ = [
    "MU_MIN_2D", "B_2D", "NU_MAX_2D", "A_DILATANCY_2D", "MU_CONTACT_2D",
    "SATURATION_I", "ProvenanceError", "Provenance", "verify_provenance",
    "mu_2d", "nu_2d", "fit_linear", "ARCHIVE",
]

# --- the four numbers of Eqs. (13)-(14), as printed in the paper -------------
MU_MIN_2D: float = 0.22      # mu*_min  (Eq. 14, "for mu != 0")
B_2D: float = 1.0            # b        (Eq. 14)
NU_MAX_2D: float = 0.81      # nu_max   (Eq. 13, "for mu = 0.4")
A_DILATANCY_2D: float = 0.3  # a        (Eq. 13, "for mu = 0.4")
MU_CONTACT_2D: float = 0.4   # the paper's interparticle friction of Eq. (13)
SATURATION_I: float = 0.2    # quoted: "mu* tends to saturate for I >= 0.2"

ARCHIVE = (Path(__file__).resolve().parent / "runs" / "provenance_archive"
           / "dacruz2005_excerpts.txt")

#: Each constant must occur within ``WINDOW`` characters *after* its anchor.
WINDOW = 400


class ProvenanceError(RuntimeError):
    """Raised when the archived text no longer supports a constant."""


@dataclass(frozen=True)
class Provenance:
    """What ``verify_provenance`` returns: the sentence each number sits in."""

    source: Path
    mu_sentence: str
    nu_sentence: str
    digest: str

    def as_dict(self) -> Dict[str, str]:
        return {"source": str(self.source), "mu_sentence": self.mu_sentence,
                "nu_sentence": self.nu_sentence, "digest": self.digest}


def _sentence(text: str, anchor: str, window: int = WINDOW) -> str:
    """Text from ``anchor`` up to the first sentence end inside ``window``."""
    i = text.find(anchor)
    if i < 0:
        raise ProvenanceError(f"anchor not found in archive: {anchor!r}")
    tail = text[i:i + window]
    m = re.search(r"\.\s", tail)          # a period followed by whitespace
    return (tail[:m.end()].strip() if m else tail.strip())


def verify_provenance(path: Optional[Path] = None) -> Provenance:
    """Bind every constant to the sentence that states it.

    Raises :class:`ProvenanceError` if the archive is missing, or if any number
    is absent from its own sentence (a bare "number occurs in the file" test is
    not enough: ``0.3`` matches inside ``0.30``, ``1.0`` inside ``0.10``).
    """
    src = Path(path) if path is not None else ARCHIVE
    if not src.exists():
        raise ProvenanceError(f"provenance archive missing: {src}")
    text = src.read_text(encoding="utf-8", errors="replace")

    mu_sentence = _sentence(text, "µ∗ min :")
    nu_sentence = _sentence(text, "ν(I) ≃νmax")

    for need, where in ((("µ∗ min ≃0.22"), mu_sentence),
                        (("b ≃1.0"), mu_sentence),
                        (("νmax ≃0.81"), nu_sentence),
                        (("a ≃0.3"), nu_sentence),
                        (("for µ = 0.4"), nu_sentence)):
        if need not in where:
            raise ProvenanceError(
                f"constant {need!r} is not bound to its sentence: {where[:120]!r}")

    # the two numbers must not be smuggled in from the *other* sentence
    if "0.81" in mu_sentence or "0.22" in nu_sentence:
        raise ProvenanceError("sentences swapped: 0.81 in mu, 0.22 in nu")

    import hashlib
    return Provenance(source=src, mu_sentence=mu_sentence, nu_sentence=nu_sentence,
                      digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])


# --------------------------------------------------------------------------- #
# the laws
# --------------------------------------------------------------------------- #
def mu_2d(I, mu_min: float = MU_MIN_2D, b: float = B_2D):
    """``mu*(I) = mu*_min + b I`` (Eq. 14).  Scalar or array."""
    if isinstance(I, (int, float)):
        return mu_min + b * float(I)
    import numpy as np
    return mu_min + b * np.asarray(I, dtype=float)


def nu_2d(I, nu_max: float = NU_MAX_2D, a: float = A_DILATANCY_2D):
    """``nu(I) = nu_max - a I`` (Eq. 13).  Scalar or array; clipped at 0.5."""
    if isinstance(I, (int, float)):
        return max(nu_max - a * float(I), 0.5)
    import numpy as np
    return np.maximum(nu_max - a * np.asarray(I, dtype=float), 0.5)


def fit_linear(I: Sequence[float], y: Sequence[float],
               w: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Weighted OLS ``y = a0 + a1 I`` with 1-sigma errors on both coefficients.

    Returns ``{'intercept', 'slope', 'intercept_se', 'slope_se', 'n', 'chi2'}``.
    The closed forms are the textbook ones; a two-point series makes the errors
    infinite (``se = nan``), which is reported rather than hidden.
    """
    import numpy as np
    x = np.asarray(I, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least two points to fit")
    wt = np.ones(n) if w is None else 1.0 / np.asarray(w, dtype=float) ** 2
    sw = wt.sum()
    mx = (wt * x).sum() / sw
    my = (wt * y).sum() / sw
    sxx = (wt * (x - mx) ** 2).sum()
    sxy = (wt * (x - mx) * (y - my)).sum()
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = y - (intercept + slope * x)
    dof = n - 2
    if dof > 0:
        s2 = (wt * resid ** 2).sum() / dof
        se_slope = math.sqrt(s2 / sxx)
        se_int = math.sqrt(s2 * (1.0 / sw + mx ** 2 / sxx))
    else:
        se_slope = se_int = float("nan")
    return {"intercept": float(intercept), "slope": float(slope),
            "intercept_se": float(se_int), "slope_se": float(se_slope),
            "n": int(n), "chi2": float((wt * resid ** 2).sum())}