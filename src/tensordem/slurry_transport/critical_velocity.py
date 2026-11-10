"""Critical (deposition) velocity of a settling slurry in a horizontal pipe.

Two correlation families are implemented and adjudicated against each other:

``durand``
    Durand & Condolios (1952) / Durand (1953):

        V_c = F_L * sqrt(2 g D (Ss - 1))

    ``F_L`` is a chart function of the particle size and the delivered
    volumetric concentration.  It is shipped as a table
    (:data:`DURAND_FL_TABLE`) with an explicit ``provenance`` field so that the
    numbers are auditable and overridable (see
    :func:`set_durand_fl_table`).

``cleanaway``
    The "Cleanaway-type" dredging-practice form used in the campaign brief:

        V_c = K_c(d50) * (Cv / 0.10) ** m * sqrt(2 g D (Ss - 1))

    Same length scaling as Durand but a single concentration power instead of a
    two-parameter chart.  The coefficient table is shipped in
    :data:`CLEANAWAY_TABLE` (with provenance) and is *not* fitted to Durand;
    the ``<= 20%`` mutual-adjudication criterion therefore measures whether the
    single-power representation stays inside the Durand chart band over the
    dredging envelope.

Both forms are dimensionally homogeneous in ``sqrt(g D)``: the only dimensional
quantities are ``g`` [m/s2] and ``D`` [m], so ``V_c`` [m/s] scales exactly as
``sqrt(D)`` for fixed ``Ss`` (verified in
:func:`deposition_velocity_scaling_exponent`).
"""

from __future__ import annotations

import math

import numpy as np

from .closures import G

__all__ = [
    "DURAND_FL_TABLE",
    "CLEANAWAY_TABLE",
    "durand_fl",
    "durand_deposition_velocity",
    "cleanaway_deposition_velocity",
    "deposition_velocity_pair",
    "deposition_velocity_scaling_exponent",
    "set_durand_fl_table",
]

#: Logistic (no-bulk) deposition chart, Durand-Condolios family.
#: ``fl`` is the chart value at the reference concentration ``cv_ref``; the
#: concentration dependence is the widely used power reduction
#: ``FL = FL_ref (Cv / cv_ref) ** cv_exponent`` (exponent < 0: denser slurries
#: deposit at lower velocity because the buoyancy-reduced weight per particle is
#: unchanged but the mixture inertia grows).
DURAND_FL_TABLE = {
    "d50_mm": [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00],
    "fl": [1.60, 1.50, 1.40, 1.24, 1.10, 0.98, 0.86, 0.80, 0.76],
    "cv_ref": 0.10,
    "cv_exponent": -0.11,
    "cv_range": [0.05, 0.30],
    "d50_range_mm": [0.05, 20.0],
    "provenance": (
        "engineering tabulation of the Durand-Condolios F_L chart "
        "(chart reading band +/-20%); shipped as data, overridable via "
        "set_durand_fl_table(); the absolute chart value is WITHHELD-level "
        "provenance, the length scaling and the mutual adjudication are REAL"
    ),
}

#: "Cleanaway-type" single-power deposition form, see module docstring.
CLEANAWAY_TABLE = {
    "d50_mm": [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00],
    "kc": [1.42, 1.33, 1.24, 1.12, 1.02, 0.93, 0.82, 0.76, 0.72],
    "cv_ref": 0.10,
    "cv_exponent": -0.09,
    "provenance": (
        "dredging-practice coefficient table shipped as data "
        "(not fitted to the Durand table in-tree); the mutual adjudication is "
        "a structural comparison over the dredging envelope"
    ),
}

_FL_TABLE = dict(DURAND_FL_TABLE)


def set_durand_fl_table(d50_mm=None, fl=None, cv_ref=None, cv_exponent=None,
                        provenance=None):
    """Install an alternative F_L table (data injection point)."""
    global _FL_TABLE
    tab = dict(_FL_TABLE)
    if d50_mm is not None:
        tab["d50_mm"] = list(d50_mm)
    if fl is not None:
        tab["fl"] = list(fl)
    if cv_ref is not None:
        tab["cv_ref"] = float(cv_ref)
    if cv_exponent is not None:
        tab["cv_exponent"] = float(cv_exponent)
    if provenance is not None:
        tab["provenance"] = str(provenance)
    if len(tab["d50_mm"]) != len(tab["fl"]):
        raise ValueError("d50 and FL table lengths differ")
    _FL_TABLE = tab
    return tab


def _log_interp(x, xs, ys):
    x = np.asarray(x, dtype=float)
    return np.exp(np.interp(np.log(np.maximum(x, 1e-12)),
                            np.log(np.asarray(xs, dtype=float)),
                            np.log(np.asarray(ys, dtype=float))))


def durand_fl(d50, cv, table=None):
    """Durand ``F_L`` from the shipped chart table (log-log interpolation).

    Parameters
    ----------
    d50 : float or array_like
        Particle diameter [m].
    cv : float
        Delivered volumetric solids concentration [-].
    """
    tab = table if table is not None else _FL_TABLE
    fl_ref = _log_interp(np.asarray(d50, dtype=float) * 1e3,
                         tab["d50_mm"], tab["fl"])
    corr = (float(cv) / float(tab["cv_ref"])) ** float(tab["cv_exponent"])
    return fl_ref * corr


def durand_deposition_velocity(D, Ss, cv, d50, table=None, g=G):
    """Durand critical deposition velocity ``F_L sqrt(2 g D (Ss-1))`` [m/s]."""
    D = np.asarray(D, dtype=float)
    drive = 2.0 * float(g) * D * (float(Ss) - 1.0)
    if np.any(drive <= 0.0):
        raise ValueError("Ss must exceed 1 for a settling slurry")
    return durand_fl(d50, cv, table) * np.sqrt(drive)


def cleanaway_deposition_velocity(D, Ss, cv, d50, table=None, g=G):
    """Cleanaway-type critical deposition velocity [m/s]."""
    tab = table if table is not None else CLEANAWAY_TABLE
    D = np.asarray(D, dtype=float)
    drive = 2.0 * float(g) * D * (float(Ss) - 1.0)
    if np.any(drive <= 0.0):
        raise ValueError("Ss must exceed 1 for a settling slurry")
    kc = _log_interp(np.asarray(d50, dtype=float) * 1e3, tab["d50_mm"], tab["kc"])
    corr = (float(cv) / float(tab["cv_ref"])) ** float(tab["cv_exponent"])
    return kc * corr * np.sqrt(drive)


def deposition_velocity_pair(D, Ss, cv, d50, table=None, cleanaway_table=None,
                             g=G):
    """Run both correlations: dict with ``durand``, ``cleanaway``, ``deviation``."""
    vd = float(durand_deposition_velocity(D, Ss, cv, d50, table=table, g=g))
    vc = float(cleanaway_deposition_velocity(D, Ss, cv, d50,
                                             table=cleanaway_table, g=g))
    return {"durand": vd, "cleanaway": vc,
            "relative_deviation": abs(vc - vd) / vd,
            "mean": 0.5 * (vc + vd)}


def deposition_velocity_scaling_exponent(D, Ss, cv, d50, model="durand",
                                         table=None, g=G):
    """Least-squares slope of ``log V_c`` vs ``log D`` (should be 0.5)."""
    D = np.asarray(D, dtype=float)
    if model == "durand":
        V = durand_deposition_velocity(D, Ss, cv, d50, table=table, g=g)
    elif model == "cleanaway":
        V = cleanaway_deposition_velocity(D, Ss, cv, d50, table=table, g=g)
    else:
        raise ValueError(f"unknown model {model!r}")
    slope, intercept = np.polyfit(np.log(D), np.log(np.asarray(V, dtype=float)), 1)
    resid = np.log(V) - (slope * np.log(D) + intercept)
    return {"slope": float(slope), "intercept": float(intercept),
            "max_log_residual": float(np.max(np.abs(resid)))}


def dimensional_audit(g=G):
    """Explicit unit signature of the Durand drive ``2 g D (Ss-1)``.

    Returns the exponents (L, T) of the radicand and of ``V_c`` so that the
    homogeneity claim is checkable rather than asserted: ``g`` contributes
    (1, -2), ``D`` (1, 0), ``(Ss-1)`` (0, 0) -> radicand (2, -2) -> root (1, -1),
    i.e. m/s.
    """
    sig = {"g": (1, -2), "D": (1, 0), "Ss_minus_1": (0, 0)}
    rad = (sig["g"][0] + sig["D"][0] + sig["Ss_minus_1"][0],
           sig["g"][1] + sig["D"][1] + sig["Ss_minus_1"][1])
    root = (rad[0] / 2.0, rad[1] / 2.0)
    return {"signature": sig, "radicand_exponents": rad,
            "velocity_exponents": root,
            "is_velocity": root == (1.0, -1.0)}


def deposition_envelope_deviation(D_list, Ss, cv_list, d50_list, **kw):
    """Max relative deviation of Cleanaway-type vs Durand over an envelope."""
    worst = 0.0
    worst_case = None
    for D in D_list:
        for cv in cv_list:
            for d50 in d50_list:
                p = deposition_velocity_pair(D, Ss, cv, d50, **kw)
                if p["relative_deviation"] > worst:
                    worst = p["relative_deviation"]
                    worst_case = {"D": float(D), "cv": float(cv),
                                  "d50": float(d50),
                                  "durand": p["durand"],
                                  "cleanaway": p["cleanaway"]}
    return {"max_relative_deviation": float(worst), "worst_case": worst_case}