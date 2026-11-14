"""Provenance registry + envelope arithmetic for the CASE-DEM-PLANE-SHEAR-MUIDI
``mu(I)`` reference law.

Why this module exists
----------------------
Round 12 left the ledger acceptance blocked on a single question: *which* set of
Jop-law constants is the right yardstick?  Two different triples live in the
repository and neither is anchored to an archived reading of the papers:

  * ``reference_law.py``  ->  (mu_s, mu_2, I_0) = (0.36, 0.84, 0.30)
    status: "as quoted in rounds 1-5" - i.e. a hand-copied quote with no
    in-repo anchor.
  * ``tests/test_mu_i_rheology.py`` -> (0.379, 0.641, 0.279)
    status: attributed to "Table 1 of da Cruz et al. 2005".

Both are labelled da Cruz 2005 / GDR MiDi 2004, yet they disagree by up to
~15 % in ``mu`` at I = 0.3.  Round 12 concluded that a provenance round was
needed *before the acceptance could be judged*.  This module makes that
judgement mechanical instead of narrative: rather than picking a winner, it
carries both readings (plus their interpolation corners) as an explicit
**envelope** and lets downstream code ask the only question that matters:

    does *any* admissible reading of the paper rescue the acceptance band?

That question is answered by ``adjudicate_provenance.py``.  The point of this
split is that the answer is a statement about the *simulation*, not about the
bibliography - if no corner of the envelope passes, the citation question is
moot for the acceptance verdict.

Nothing here is a new measurement: this module is pure arithmetic over the
constants, and the only physics input is the shape of the Jop law.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product


# --------------------------------------------------------------------------- #
# Provenance registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LawProvenance:
    """One candidate ``mu(I)`` triple, with an explicit evidence label."""

    name: str
    mu_s: float
    mu_2: float
    I_0: float
    status: str
    source: str

    def mu(self, I):
        """``mu(I) = mu_s + (mu_2 - mu_s)/(I_0/I + 1)`` (scalar or array)."""
        return jop_mu_I(I, self.mu_s, self.mu_2, self.I_0)

    @property
    def triple(self) -> tuple[float, float, float]:
        return (self.mu_s, self.mu_2, self.I_0)


#: The triple that ``reference_law.py`` uses today (the yardstick that the
#: benchmark modules and the cached round-8..12 verdicts were all computed
#: against).  Kept first so that ``PRIMARY`` preserves the historical verdicts.
PRIMARY_REPO = LawProvenance(
    name="PRIMARY_REPO",
    mu_s=0.36,
    mu_2=0.84,
    I_0=0.30,
    status="QUOTED-NOT-ARCHIVED",
    source="reference_law.py, 'as quoted in rounds 1-5' (no in-repo anchor)",
)

#: The triple carried by the pytest suite, attributed to Table 1 of da Cruz.
TEST_GDR_MIDI = LawProvenance(
    name="TEST_GDR_MIDI",
    mu_s=0.379,
    mu_2=0.641,
    I_0=0.279,
    status="ATTRIBUTED-NOT-ARCHIVED",
    source="tests/test_mu_i_rheology.py, 'Table 1 of da Cruz et al. 2005'",
)

#: Every candidate the repository currently asserts is "the" reference law.
CANDIDATES: tuple[LawProvenance, ...] = (PRIMARY_REPO, TEST_GDR_MIDI)

#: The interpolation corners.  A provenance dispute that is *only* about the
#: citation cannot exclude any corner of the box spanned by the two readings,
#: so the envelope - not the two endpoints - is the honest admissible set.
ENVELOPE: tuple[LawProvenance, ...] = tuple(
    LawProvenance(
        name=f"CORNER(ms={ms:g},m2={m2:g},I0={i0:g})",
        mu_s=ms,
        mu_2=m2,
        I_0=i0,
        status="ENVELOPE-CORNER",
        source="box spanned by the two in-repo candidate triples",
    )
    for ms, m2, i0 in product(
        sorted({c.mu_s for c in CANDIDATES}),
        sorted({c.mu_2 for c in CANDIDATES}),
        sorted({c.I_0 for c in CANDIDATES}),
    )
)


# --------------------------------------------------------------------------- #
# The law
# --------------------------------------------------------------------------- #
def jop_mu_I(I, mu_s: float = PRIMARY_REPO.mu_s,
             mu_2: float = PRIMARY_REPO.mu_2,
             I_0: float = PRIMARY_REPO.I_0):
    """Jop--Pouliquen--Forterre law, scalar or numpy-array ``I``.

    ``I <= 0`` returns ``mu_s`` (the quasi-static limit where the law is
    defined by continuity); this mirrors ``tensordem.mu_i_rheology.jop_mu_I``.
    """
    if isinstance(I, (int, float)) or (hasattr(I, "ndim") and getattr(I, "ndim", 1) == 0):
        try:
            I = float(I)
        except (TypeError, ValueError):
            I = None
        if I is not None:
            if not (I > 0.0) or not (I == I):
                return mu_s
            return mu_s + (mu_2 - mu_s) / (I_0 / I + 1.0)
    try:
        return mu_s + (mu_2 - mu_s) / (I_0 / I + 1.0)
    except TypeError:                       # numpy array input
        import numpy as np

        I = np.asarray(I, dtype=float)
        I = np.where(np.isfinite(I) & (I > 0.0), I, np.nan)
        out = mu_s + (mu_2 - mu_s) / (I_0 / I + 1.0)
        return np.where(np.isfinite(out), out, mu_s)


def envelope_bounds(I):
    """``(mu_min, mu_max)`` over the full admissible envelope at inertial ``I``.

    This is the quantitative content of "the citation is unresolved": any
    measured ``mu`` that lands inside these bounds *could* be correct for the
    right paper reading; anything outside is wrong for every reading.
    """
    import numpy as np

    stack = np.stack([np.asarray(c.mu(I), dtype=float) for c in ENVELOPE])
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    return (float(lo), float(hi)) if stack.ndim == 1 else (lo, hi)


def envelope_spread(I):
    """Relative envelope width ``(mu_max - mu_min)/mu_min`` at inertial ``I``.

    Used by the tests to keep the "the two laws disagree materially" claim
    honest: if this ever drops below the 10 % acceptance band the provenance
    question stops being blocking and the registry should be collapsed.
    """
    lo, hi = envelope_bounds(I)
    return (hi - lo) / lo


# --------------------------------------------------------------------------- #
# Pressure-normaliser consistency
# --------------------------------------------------------------------------- #
def consistent_pressure_pair(*, sigma_zz: float, pressure: float,
                             tau: float) -> dict:
    """Build the two *internally consistent* ``(mu, normaliser)`` readings.

    The mu(I) law is a relation between the **shear stress** and the
    **pressure**, with the inertial number built from *that same* pressure:

        mu = tau/p,   I = gamma_dot d sqrt(rho/p).

    A plane-shear cell with walls in z controls ``sigma_zz`` directly, so there
    are two admissible choices and they must not be mixed:

    ``zz``  ``mu = tau/|sigma_zz|``, ``p = |sigma_zz|``   (the servo-controlled
            stress - what the device actually holds)
    ``p``   ``mu = tau/p``, ``p = -tr(sigma)/3``          (the continuum
            pressure - what the published law is written in)

    Mixing them (mu by ``p``, I by ``sigma_zz``) inflates ``mu`` by
    ``|sigma_zz|/p`` without moving the abscissa; that ratio is 1.11-1.31 in
    the archived round-9/10 points, i.e. up to +31 % of spurious ordinate.
    This helper makes the two conventions explicit so callers cannot mix them.
    """
    if pressure <= 0.0 or abs(sigma_zz) <= 0.0:
        raise ValueError("pressures must be non-zero")
    return {
        "zz": {"mu": tau / abs(sigma_zz), "p": abs(sigma_zz),
               "anisotropy_p_over_zz": pressure / abs(sigma_zz)},
        "p": {"mu": tau / pressure, "p": pressure,
              "anisotropy_p_over_zz": pressure / abs(sigma_zz)},
        "mixing_inflation": abs(sigma_zz) / pressure,
    }


# --------------------------------------------------------------------------- #
# Reverse calibration
# --------------------------------------------------------------------------- #
def fit_jop_law(I_values, mu_values) -> dict:
    """Least-squares ``(mu_s, mu_2, I_0)`` for a measured ``mu(I)`` series.

    The Jop law is linear in ``(mu_s, mu_2)`` given ``I_0``, so ``I_0`` is
    solved by a 1-D scan over a log grid (robust, derivative-free) and
    ``(mu_s, mu_2)`` by exact linear least squares at each ``I_0``.

    Returns ``{mu_s, mu_2, I_0, rms, max_abs_resid, r2, n}``.  ``n < 3`` makes
    the problem under-determined and is reported as such (``identified=False``)
    rather than silently returning a fitted triple.
    """
    import numpy as np

    I = np.asarray(I_values, dtype=float)
    mu = np.asarray(mu_values, dtype=float)
    ok = np.isfinite(I) & np.isfinite(mu) & (I > 0.0)
    I, mu = I[ok], mu[ok]
    n = int(I.size)
    out = {"n": n, "identified": False}
    if n < 3:
        out["reason"] = "fewer than 3 points: (mu_s, mu_2, I_0) is not identified"
        return out

    best = None
    for I0 in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 6001)):
        # mu = mu_s + (mu_2 - mu_s)/(I0/I + 1)
        #    = mu_s * x + mu_2 * (1 - x),  x = 1 - 1/(I0/I + 1)
        w = 1.0 / (I0 / I + 1.0)
        x = 1.0 - w
        A = np.column_stack([x, w])
        sol, *_ = np.linalg.lstsq(A, mu, rcond=None)
        resid = mu - A @ sol
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(sol[0]), float(sol[1]), float(I0), resid)

    rms, mu_s, mu_2, I_0, resid = best
    ss_tot = float(np.sum((mu - mu.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    out.update({
        "identified": True,
        "mu_s": mu_s,
        "mu_2": mu_2,
        "I_0": I_0,
        "rms": rms,
        "max_abs_resid": float(np.max(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"),
        "residuals": [float(r) for r in resid],
        "I_0_at_scan_bound": bool(I_0 > 9.99),
        "degenerate": bool(I_0 > 9.99),
    })
    return out


def best_case_over_envelope(I_values, mu_values, *, tol: float = 0.10) -> dict:
    """Best achievable agreement of a measured series with *any* Jop law.

    This is the falsification oracle for the provenance blocker.  It scans a
    continuous box that strictly contains the disputed envelope
    (``mu_s in [0.34, 0.40]``, ``mu_2 in [0.55, 0.95]``, ``I_0 in [0.22, 0.38]``)
    and reports both

    * ``min_max_abs_rel_err`` - the smallest possible worst-point error, i.e. a
      *lower bound* on how badly the measured curve misses the whole family; and
    * ``n_param_sets_within_tol`` - how many parameter sets in that box put
      **every** point inside ``tol``.

    If ``min_max_abs_rel_err > tol`` then no admissible reading of the paper can
    bring the measurement into the acceptance band: the discrepancy is
    structural (a defect of the device or of the abscissa), not bibliographic.
    """
    import numpy as np

    I = np.asarray(I_values, dtype=float)
    mu = np.asarray(mu_values, dtype=float)
    ok = np.isfinite(I) & np.isfinite(mu) & (I > 0.0)
    I, mu = I[ok], mu[ok]
    if I.size == 0:
        raise ValueError("no usable (I, mu) points")

    grid = product(
        np.linspace(0.34, 0.40, 13),
        np.linspace(0.55, 0.95, 21),
        np.linspace(0.22, 0.38, 17),
    )
    best = None
    n_within = 0
    for mu_s, mu_2, I_0 in grid:
        rel = np.abs(jop_mu_I(I, mu_s, mu_2, I_0) / mu - 1.0)
        worst = float(np.max(rel))
        if worst <= tol:
            n_within += 1
        if best is None or worst < best[0]:
            best = (worst, mu_s, mu_2, I_0, rel)

    worst, mu_s, mu_2, I_0, rel = best
    # A fit that sits on the edge of the search box is not identified: it says
    # "the true parameters lie outside anything physically documented", not
    # "these are the parameters".  Flag it so the caller cannot over-read it.
    edges = {"mu_s": (0.34, 0.40), "mu_2": (0.55, 0.95), "I_0": (0.22, 0.38)}
    at = []
    for nm, val in (("mu_s", mu_s), ("mu_2", mu_2), ("I_0", I_0)):
        lo, hi = edges[nm]
        if abs(val - lo) < 1e-9 or abs(val - hi) < 1e-9:
            at.append(nm)
    return {
        "min_max_abs_rel_err": worst,
        "boundary_hit": at,
        "identified": not at,
        "argmin": {"mu_s": float(mu_s), "mu_2": float(mu_2), "I_0": float(I_0)},
        "per_point_rel_err": [float(r) for r in rel],
        "n_param_sets_within_tol": n_within,
        "tol": tol,
        "provenance_can_rescue": bool(worst <= tol and not at),
        "search_box": {"mu_s": [0.34, 0.40], "mu_2": [0.55, 0.95],
                       "I_0": [0.22, 0.38]},
    }


def verdict_table(I_values, mu_values, tol: float = 0.10) -> list[dict]:
    """Per-candidate pass/fail of a measured series against every candidate law."""
    rows = []
    for cand in CANDIDATES:
        rel = []
        for I, mu in zip(I_values, mu_values):
            rel.append(float(mu / cand.mu(I) - 1.0))
        rows.append({
            "candidate": cand.name,
            "status": cand.status,
            "triple": list(cand.triple),
            "rel_err": rel,
            "max_abs_rel_err": max((abs(r) for r in rel), default=float("nan")),
            "n_within_tol": int(sum(1 for r in rel if abs(r) <= tol)),
            "n_points": len(rel),
            "passes_all": bool(rel) and all(abs(r) <= tol for r in rel),
        })
    return rows