"""Cross-validation of the **contact-local** mu(I) proxy against the bulk mu(I) route.

Motivation (why this module exists)
-----------------------------------
The column-collapse campaign (m17111) closed the Lube et al. (2004) runout law
with a *contact-local* inertial number

.. math:: I_c = \\frac{|v_t|}{\\sqrt{f_n/(\\rho d^2)}},

i.e. the gauge a single contact can build from its own normal force ``f_n`` and
slip speed ``|v_t|`` (stress scale ``P ~ f_n/d^2``, rate scale
``dot gamma ~ |v_t|/d``).  That hypothesis was registered as a *negative*
result in the ecosystem OSS ledger (``oss_solutions`` id 421,
tag ``mu-i-contact-proxy-negative``): in the slow dense band the proxy pins
``mu(I_c)`` at ``mu_s``, so the flow cannot reach the Coulomb friction the
contact law actually carries.

This module turns that ledger entry into a *measurable* cross-validation.  On
one and the same steady-shear state of the repository's constant-stress servo
shear cell (:func:`tensordem.mu_i_rheology.simulate_servo_shear`) it compares

* **REF (bulk route)** -- the literature/Jop et al. (2006) abscissa
  ``I = dot gamma d / sqrt(P/rho)`` built from the measured shear-rate profile
  and the Love-Weber slab pressure, and the measured ``mu = tau/P``;
* **GOT (contact-proxy route)** -- the ``f_n``-weighted aggregate of the
  contact closure ``mu(I_c)`` over the live contacts of the same state.

Two predeclared metrics (see ``prereg/xval_mu_i_contact_proxy_m17500.md``):

* ``i_mapping_error = |I_c/I_bulk - 1|``           -- is the proxy *the same*
  dimensionless number?
* ``mu_error = |mu_proxy/mu_bulk - 1|``            -- does the proxy reproduce
  the friction the bed actually carries?

plus the *structural* metric that explains the ledger's negative finding:

* ``shortfall = (mu_contact - mu(I_c)) / mu_contact`` -- in the dense slow
  band ``mu(I_c) -> mu_s = 0.36`` while the DEM contact law carries
  ``mu_contact = 0.5``; the closure can therefore never represent a bed whose
  grains are rougher than ``mu_s``.

Everything here is pure post-processing: the closure is evaluated *diagnostically*
and never substituted into the contact law of the device.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .granular_column.inertial_friction import (
    JOP_2006,
    contact_inertial_number,
    mu_of_inertial_number,
)

__all__ = [
    "DEFAULT_TOL",
    "aggregate_proxy",
    "contact_proxy_stats",
    "jop_bulk_mu",
    "shortfall",
    "slip_speed_at_contacts",
    "xval_row",
    "xval_summary",
]

#: Owner accuracy standard: a cross-validation error above 15 % is unacceptable.
DEFAULT_TOL = 0.15


def slip_speed_at_contacts(vel, omega, iu, ju, branch, radius: float):
    """Tangential (slip) speed at each contact, including both body spins.

    Same convention as :func:`tensordem.granular3d_core.contact_forces`:

    .. math:: v_t = (v_j - v_i) - v_n \\hat n + (\\omega_j\\times(-r\\hat n)
                                                   - \\omega_i\\times(r\\hat n))

    ``iu``/``ju`` may index a ghost-triplicated copy of the particle arrays (as
    in the periodic shear cell); the caller passes ``branch = x_j - x_i`` from
    that same copy, so the two are consistent.
    """
    vel = np.asarray(vel, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    iu = np.asarray(iu, dtype=np.int64)
    ju = np.asarray(ju, dtype=np.int64)
    branch = np.asarray(branch, dtype=np.float64)
    if len(iu) == 0:
        return np.zeros(0, dtype=np.float64)
    dist = np.linalg.norm(branch, axis=1)
    nvec = branch / np.maximum(dist, 1e-30)[:, None]
    v_rel = vel[ju] - vel[iu]
    v_n = (v_rel * nvec).sum(axis=1)
    surf_i = np.cross(omega[iu], nvec * float(radius))
    surf_j = np.cross(omega[ju], -nvec * float(radius))
    v_t = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)
    return np.linalg.norm(v_t, axis=1)


def contact_proxy_stats(f_n, slip_speed, diameter: float, density: float,
                        mu_s: float = JOP_2006["mu_s"],
                        mu_2: float = JOP_2006["mu_2"],
                        i_0: float = JOP_2006["i_0"]) -> Dict[str, float]:
    """Aggregate ``I_c`` and ``mu(I_c)`` over the live contacts of one state.

    Only contacts that carry a positive normal force are "live" (a separating
    pair has no friction cap).  Two aggregates are returned because they answer
    different questions:

    * ``*_fn_weighted`` -- the friction *capacity* of the state,
      ``sum(mu(I_c) f_n) / sum(f_n)``, i.e. what the bed would mobilise if the
      closure replaced the Coulomb cap contact by contact.  This is the
      quantity to compare with the measured macroscopic ``tau/P``.
    * ``*_median`` -- the typical contact, insensitive to a few heavily loaded
      force-chain contacts.
    """
    f_n = np.asarray(f_n, dtype=np.float64)
    slip = np.asarray(slip_speed, dtype=np.float64)
    if f_n.shape != slip.shape:
        raise ValueError("f_n and slip_speed must have the same shape")
    live = f_n > 0.0
    n_live = int(np.count_nonzero(live))
    weights = f_n[live]
    w_sum = float(weights.sum())
    out = {"n_contacts": float(f_n.size), "n_live": float(n_live),
           "live_fraction": float(n_live / f_n.size) if f_n.size else 0.0,
           "fn_sum": w_sum, "i_median": math.nan, "i_mean": math.nan,
           "i_p90": math.nan, "i_fn_weighted": math.nan,
           "mu_median": math.nan, "mu_mean": math.nan,
           "mu_fn_weighted": math.nan}
    if n_live == 0 or w_sum <= 0.0:
        return out
    i_c = contact_inertial_number(slip[live], weights, diameter, density)
    mu = mu_of_inertial_number(i_c, mu_s, mu_2, i_0)
    mu = np.asarray(mu, dtype=np.float64)
    out.update({
        "i_median": float(np.median(i_c)),
        "i_mean": float(i_c.mean()),
        "i_p90": float(np.percentile(i_c, 90.0)),
        "i_fn_weighted": float((i_c * weights).sum() / w_sum),
        "mu_median": float(np.median(mu)),
        "mu_mean": float(mu.mean()),
        "mu_fn_weighted": float((mu * weights).sum() / w_sum),
        "fn_sum": w_sum,
    })
    return out


def aggregate_proxy(samples: Sequence[Dict[str, float]],
                    frac: float = 0.2) -> Dict[str, float]:
    """Pool the per-sample contact statistics of the final ``frac`` window.

    ``frac = 0.2`` matches the servo device's *final window*, the window on
    which its constant-stress validity verdict is taken; the tail average
    (60 %) mixes the transient when the servo is still recovering.
    """
    rows = [r for r in samples if r and r.get("n_live", 0) > 0]
    if not rows:
        return {"n_samples": 0.0, "i_fn_weighted": math.nan,
                "mu_fn_weighted": math.nan, "i_median": math.nan,
                "mu_median": math.nan}
    n_final = max(1, int(frac * len(rows)))
    win = rows[-n_final:]
    fn_sum = float(sum(r["fn_sum"] for r in win))
    if fn_sum <= 0.0:
        return {"n_samples": float(len(win)), "i_fn_weighted": math.nan,
                "mu_fn_weighted": math.nan, "i_median": math.nan,
                "mu_median": math.nan}
    def _wavg(key):
        return float(sum(r[key] * r["fn_sum"] for r in win) / fn_sum)
    med_i = float(np.median([r["i_median"] for r in win]))
    med_mu = float(np.median([r["mu_median"] for r in win]))
    return {"n_samples": float(len(win)),
            "i_fn_weighted": _wavg("i_fn_weighted"),
            "mu_fn_weighted": _wavg("mu_fn_weighted"),
            "i_median": med_i, "mu_median": med_mu}


def jop_bulk_mu(inertial_number: float, params: Dict[str, float] | None = None
                ) -> float:
    """Reference bulk ``mu(I)`` -- Jop et al. (2006) with the quoted constants."""
    p = params or JOP_2006
    return float(mu_of_inertial_number(inertial_number, p["mu_s"], p["mu_2"],
                                       p["i_0"]))


def shortfall(mu_contact: float, i_c: float,
              params: Dict[str, float] | None = None) -> Dict[str, float]:
    """Friction the contact closure *loses* relative to the DEM contact law.

    ``mu_proxy = mu(I_c)`` is the largest friction the proxy can mobilise at a
    contact whose local inertial number is ``i_c``.  When the DEM grains carry a
    Coulomb coefficient ``mu_contact > mu_s`` and the flow is slow/dense
    (``i_c << i_0``), ``mu_proxy < mu_contact`` and the bed is systematically
    more slippery than the contact law it is supposed to represent.
    """
    mu_proxy = jop_bulk_mu(i_c, params)
    if mu_contact <= 0.0:
        raise ValueError("mu_contact must be positive")
    return {"mu_proxy": mu_proxy, "mu_contact": float(mu_contact),
            "shortfall": float((mu_contact - mu_proxy) / mu_contact),
            "capped": bool(mu_proxy < mu_contact)}


def xval_row(*, label: str, inertial_number_bulk: float, mu_bulk: float,
             proxy: Dict[str, float], mu_contact: float,
             params: Dict[str, float] | None = None) -> Dict[str, object]:
    """One REF-vs-GOT comparison row of the cross-validation table."""
    i_bulk = float(inertial_number_bulk)
    mu_ref = jop_bulk_mu(i_bulk, params)
    i_c = float(proxy["i_fn_weighted"])
    mu_proxy = float(proxy["mu_fn_weighted"])
    i_err = abs(i_c / i_bulk - 1.0) if i_bulk > 0 else math.nan
    mu_err = abs(mu_proxy / mu_bulk - 1.0) if mu_bulk else math.nan
    ref_err = abs(mu_bulk / mu_ref - 1.0) if mu_ref else math.nan
    short = shortfall(mu_contact, i_c, params)
    return {
        "label": label,
        "i_bulk_measured": i_bulk,
        "mu_bulk_measured": float(mu_bulk),
        "mu_jop_at_i_bulk": mu_ref,
        "bulk_vs_jop_rel_error": ref_err,
        "i_proxy_fn_weighted": i_c,
        "i_proxy_median": float(proxy["i_median"]),
        "mu_proxy_fn_weighted": mu_proxy,
        "mu_proxy_median": float(proxy["mu_median"]),
        "i_mapping_error": i_err,
        "mu_error": mu_err,
        "shortfall": short["shortfall"],
        "proxy_capped": short["capped"],
        "n_contacts": float(proxy.get("n_live", math.nan)),
    }


def xval_summary(rows: Iterable[Dict[str, object]],
                 tol: float = DEFAULT_TOL) -> Dict[str, object]:
    """Verdicts of the cross-validation, thresholds frozen *a priori*.

    ``mapping`` -- PASS if ``|I_c/I_bulk - 1| < tol`` at **every** row: the
    contact gauge and the bulk gauge are the same dimensionless number.
    ``friction`` -- PASS if ``|mu_proxy/mu_bulk - 1| < tol`` at every row.
    ``reference`` -- PASS if the bulk route itself reproduces the Jop law within
    ``tol`` at every row (a failing reference voids the comparison).
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to summarise")
    def _worst(key):
        vals = [float(r[key]) for r in rows
                if isinstance(r.get(key), (int, float))
                and math.isfinite(float(r[key]))]
        return max(vals) if vals else math.nan
    worst_i = _worst("i_mapping_error")
    worst_mu = _worst("mu_error")
    worst_ref = _worst("bulk_vs_jop_rel_error")
    n_capped = sum(1 for r in rows if r.get("proxy_capped"))
    return {
        "n_rows": len(rows),
        "tolerance": float(tol),
        "worst_i_mapping_error": worst_i,
        "worst_mu_error": worst_mu,
        "worst_bulk_vs_jop_rel_error": worst_ref,
        "verdict_mapping": "PASS" if worst_i < tol else "FAIL",
        "verdict_friction": "PASS" if worst_mu < tol else "FAIL",
        "verdict_reference": "PASS" if worst_ref < tol else "FAIL",
        "rows_capped_by_proxy": n_capped,
        "worst_shortfall": _worst("shortfall"),
    }