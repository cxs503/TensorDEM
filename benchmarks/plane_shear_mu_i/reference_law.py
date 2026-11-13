"""Single source of truth for the CASE-DEM-PLANE-SHEAR-MUIDI reference law.

Rounds 1-5 each carried their own hand-copied copy of the ``mu(I)`` / ``phi(I)``
reference.  The ``mu`` part is stable, but the ``phi`` part was written as

    phi(I) = PHI_C + (PHI_J - PHI_C) / (I0/I + 1)      with PHI_C=0.58 < PHI_J=0.64

i.e. **rising** with the inertial number.  Dilatancy is the opposite: a sheared
granular bed expands, so ``phi`` must *decrease* with ``I`` (da Cruz et al. 2005
PRE 72:021309, Fig. 3 - their ``phi`` falls monotonically over
``I = 1e-3 ... 0.3``; GDR MiDi 2004, Eur. Phys. J. E 14:341, same direction;
the in-repo device dilates as well, ``dilatancy > 0`` in every round).  The
rising form was therefore a sign error, and it is the reason round-2/round-3
reported the measured ``phi`` as "30 % low against the reference" whenever the
comparison was made at low ``I``.

Fix (m17026): keep the same two magnitudes but give them their physically
admissible roles - ``PHI_MAX = 0.64`` is the ``I -> 0`` (jamming) limit and
``PHI_MIN = 0.58`` the ``I -> inf`` limit:

    phi(I) = PHI_MAX - (PHI_MAX - PHI_MIN) / (I0/I + 1)

Provenance status: the *pair of numbers* (0.58, 0.64) still has **no in-repo
literature anchor** (grep of ``brash_ice_rheology.mu_i`` / ``granular_column``
returns the ``mu(I)`` trio only), and no reading of the two papers is archived
in this repository.  What is verified here is the *monotonicity* (dilatancy) and
the *limits*; the absolute ``phi`` band remains PROVENANCE-UNVERIFIED, so the
``phi`` acceptance of the ledger is reported as WITHHELD rather than PASS.
"""
from __future__ import annotations

# --- mu(I): da Cruz et al. 2005 / GDR MiDi 2004 (unchanged, quoted in repo) --
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30

# --- phi(I): magnitudes inherited from round 1, roles corrected (see header) -
PHI_MAX, PHI_MIN = 0.64, 0.58
# backwards-compatible aliases: rounds 1-5 called these PHI_C / PHI_J
PHI_C, PHI_J = PHI_MIN, PHI_MAX

MU_REF_STATUS = ("da Cruz 2005 / GDR MiDi 2004 fit as quoted in rounds 1-5: "
                 "mu_s=0.36, mu_2=0.84, I0=0.30")
PHI_REF_STATUS = (
    "PROVENANCE-UNVERIFIED magnitudes (0.58, 0.64 inherited from round 1, no "
    "in-repo anchor; da Cruz 2005 is a 2D-disk study with phi ~ 0.84 -> 0.79, so "
    "these are 3D glass-bead numbers) + VERIFIED monotonicity (decreasing with "
    "I, i.e. dilatancy) and limits phi(I->0)=0.64, phi(I->inf)=0.58")


def ref_mu(I):
    """``mu(I) = mu_s + (mu_2 - mu_s)/(I0/I + 1)`` (scalar or array)."""
    try:
        return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)
    except TypeError:                       # numpy array input
        import numpy as np
        I = np.asarray(I, dtype=float)
        return MU_S + (MU_2 - MU_S) / (I_0 / np.maximum(I, 1e-300) + 1.0)


def ref_phi(I):
    """``phi(I) = phi_max - (phi_max - phi_min)/(I0/I + 1)``, decreasing in I."""
    try:
        return PHI_MAX - (PHI_MAX - PHI_MIN) / (I_0 / I + 1.0)
    except TypeError:                       # numpy array input
        import numpy as np
        I = np.asarray(I, dtype=float)
        return PHI_MAX - (PHI_MAX - PHI_MIN) / (I_0 / np.maximum(I, 1e-300) + 1.0)


def reference() -> dict:
    """Machine-readable provenance block (copied into every artifact)."""
    return {
        "mu": {"law": "mu_s+(mu_2-mu_s)/(I0/I+1)",
               "constants": {"mu_s": MU_S, "mu_2": MU_2, "I_0": I_0},
               "status": MU_REF_STATUS},
        "phi": {"law": "phi_max-(phi_max-phi_min)/(I0/I+1)",
                "constants": {"phi_max": PHI_MAX, "phi_min": PHI_MIN,
                              "I_0": I_0},
                "status": PHI_REF_STATUS},
    }


__all__ = ["MU_S", "MU_2", "I_0", "PHI_MAX", "PHI_MIN", "PHI_C", "PHI_J",
           "MU_REF_STATUS", "PHI_REF_STATUS", "ref_mu", "ref_phi", "reference"]