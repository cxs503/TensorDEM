"""Case matrix for the p176w2-n08 rotating-drum segregation verification.

The matrix is declared *before* any acceptance run (see
``prereg/P176W2_N08_rotating_drum_segregation.json``); the selection rule for
the operating point is physical, not tuned to the answer:

* the drum must be in the **rolling** regime (not cataracting/slipping), i.e.
  ``Fr = omega^2 R / g`` below the cataracting threshold ``Fr ~ 0.3``.  For
  ``R = 0.20 m`` this gives ``omega = 2.5 rad/s`` -> ``Fr = 0.127`` (rolling).
  Only the omega sweep leaves that window, on purpose.
* the run must be long enough for the percolation mechanism to act: the pilot
  (8 revolutions) showed the size-position correlation still growing linearly
  (+0.011 per revolution at d_l/d_s = 2), so the acceptance matrix uses 16
  revolutions and one 28-revolution saturation case.
* the bed must hold enough particles for percolation statistics
  (``N >= 500`` at the reference point).

Groups
------
ratio    size-ratio sweep d_l/d_s = 1.2/1.5/2.0/3.0          -> C1, C2
mono     monodisperse (d_l = d_s) and near-mono 1.05/1.10    -> C4, C5a
omega    drum speed 0 / 0.5 / 1 / 2 rad/s (+ reference 2.5)   -> C5b
fill     fill fraction 0.15 / 0.30 / 0.50 / 0.65             -> C5c
phil     large-species area fraction 0.15 / 0.30 / 0.50      -> C2 (density of
                                                               the mixture)
density  rho_l/rho_s = 3 (heavy coarse) and 0.33 (light coarse) -> C3 (core mode)
numeric  dt/2 and n_bins doubled on the reference case       -> numerical
                                                               sensitivity
repeat   second seed on the reference case                   -> reproducibility
"""
from __future__ import annotations

import math

__all__ = ["BASE", "build_cases", "reference_point"]

#: reference operating point (rolling regime, N >= 500)
BASE = dict(
    radius=0.20,
    fill=0.30,
    d_small=0.008,
    d_large=0.016,
    large_area_fraction=0.30,
    thickness=0.01,
    rho_small=2500.0,
    rho_large=2500.0,
    kn=5.0e4,
    kt_ratio=0.5,
    restitution=0.30,
    mu=0.40,
    mu_rolling=0.10,
    omega=2.5,
    dt=8.0e-5,
    settle_steps=8000,
    revs=16.0,
    sample_revs=0.5,
    n_bins=8,
    seed=0,
)


def reference_point():
    return dict(BASE)


def build_cases(revs: float | None = None, d_small: float | None = None,
                radius: float | None = None, dt: float | None = None) -> list[dict]:
    """Return the declared case matrix as a list of dicts.

    Each entry: ``{name, group, seed, cfg}`` where ``cfg`` are ``DrumConfig``
    keyword arguments.  Optional overrides exist only for the pilot runs.
    """
    base = dict(BASE)
    if revs is not None:
        base["revs"] = revs
    if d_small is not None:
        base["d_small"] = d_small
        base["d_large"] = d_small * (BASE["d_large"] / BASE["d_small"])
    if radius is not None:
        base["radius"] = radius
    if dt is not None:
        base["dt"] = dt

    cases: list[dict] = []

    def add(name, group, **kw):
        cfg = dict(base)
        cfg.update(kw)
        cases.append({"name": name, "group": group, "seed": cfg["seed"], "cfg": cfg})

    d_s = base["d_small"]
    # --- size ratio sweep ---------------------------------------------------
    for ratio in (1.2, 1.5, 2.0, 3.0):
        add(f"ratio_{ratio:g}", "ratio", d_large=ratio * d_s)
    # --- monodisperse / near-monodisperse negative controls ------------------
    add("mono_1.00_s0", "mono", d_large=1.00 * d_s)
    add("mono_1.00_s1", "mono", d_large=1.00 * d_s, seed=1)
    add("mono_1.05", "mono", d_large=1.05 * d_s)
    add("mono_1.10", "mono", d_large=1.10 * d_s)
    # --- angular velocity sweep --------------------------------------------
    for om in (0.0, 0.5, 1.0, 2.0):
        add(f"omega_{om:g}", "omega", omega=om, d_large=2.0 * d_s)
    # --- fill fraction sweep ------------------------------------------------
    for fl in (0.15, 0.50, 0.65):
        add(f"fill_{fl:g}", "fill", fill=fl, d_large=2.0 * d_s)
    # --- mixture composition sweep -----------------------------------------
    for pl in (0.15, 0.50):
        add(f"phil_{pl:g}", "phil", large_area_fraction=pl, d_large=2.0 * d_s)
    # --- density-driven (core / reverse Brazil-nut) mode --------------------
    add("density_3.0", "density", rho_large=3.0 * BASE["rho_small"], d_large=2.0 * d_s)
    add("density_0.33", "density", rho_large=BASE["rho_small"] / 3.0, d_large=2.0 * d_s)
    # --- numerical sensitivity + repeatability ------------------------------
    add("numeric_dt_half", "numeric", dt=base["dt"] / 2.0, d_large=2.0 * d_s)
    add("numeric_bins12", "numeric", n_bins=12, d_large=2.0 * d_s)
    add("repeat_seed1", "repeat", d_large=2.0 * d_s, seed=1)
    # --- saturation check (longer run than the rest of the matrix) ----------
    add("sat_ratio2_rev28", "saturation", d_large=2.0 * d_s, revs=28.0)
    return cases


def reference_case_name(revs: float | None = None) -> str:
    return "ratio_2"