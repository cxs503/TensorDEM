"""3x3 design matrix (roundness x concentration) and the two-way ANOVA.

Criterion C5 of node p472w2-N05 asks for the **main effects and the interaction**
of roundness and concentration on the effective friction, so that the attrition
mechanism (rounding) and the packing mechanism (concentration) are not narrated
as one blended effect when they are not separable.

Design
------
* factor A ``roundness`` with three target levels (``R_target``) -- each replicate
  realises its own *regular polygon* family: the side count ``n`` is randomised
  over ``{3, 4, 5, 6}`` and the fillet radius is solved so that the realised
  Wadell roundness equals the target.  This is *shape disorder at fixed
  roundness*: it is what supplies the within-cell error term (a balanced design
  with ``m`` replicates per cell).
* factor B ``concentration`` with three levels inside the measured DEM bracket of
  :class:`~tensordem.floe_attrition.friction.SphereBaseline`.

The response is ``mu_eff`` from the closure of
:mod:`tensordem.floe_attrition.friction`.  The ANOVA is a textbook balanced
two-way model with interaction::

    y_ijk = mu + a_i + b_j + (ab)_ij + e_ijk,   e ~ N(0, sigma^2)

and the table reports SS, df, MS, F and p for A, B, A x B and the residual.
``eta^2`` (share of total variance) is reported next to each effect because the
F-test alone says nothing about engineering relevance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from .friction import (InterlockingFriction, SphereBaseline, load_sphere_baseline)
from .geometry import RoundedPolygon, fillet_radius_for_roundness

__all__ = [
    "DesignCell",
    "build_design_matrix",
    "two_way_anova",
    "annex_table",
    "usable_side_counts",
]

SIDE_CHOICES = (3, 4, 5, 6)


@dataclass(frozen=True)
class DesignCell:
    """One evaluated replicate of the design matrix."""

    roundness_level: float
    concentration: float
    n_sides: int
    fillet_radius: float
    realized_roundness: float
    interlocking_index: float
    mu_effective: float
    mu_sphere: float
    coordination: float

    def as_dict(self) -> dict:
        return {
            "roundness_level": self.roundness_level,
            "concentration": self.concentration,
            "n_sides": self.n_sides,
            "fillet_radius": self.fillet_radius,
            "realized_roundness": self.realized_roundness,
            "interlocking_index": self.interlocking_index,
            "mu_effective": self.mu_effective,
            "mu_sphere": self.mu_sphere,
            "coordination": self.coordination,
        }


def build_design_matrix(roundness_levels: Sequence[float],
                        concentrations: Sequence[float], *,
                        replicates: int = 8,
                        edge_length: float = 0.6,
                        baseline: SphereBaseline | None = None,
                        model: InterlockingFriction | None = None,
                        seed: int = 20261111) -> list[DesignCell]:
    """Evaluate the 3x3 design: ``(roundness level) x (concentration)`` x ``m``.

    Replicate disorder: ``n_sides`` is drawn without replacement-avoidance from
    :data:`SIDE_CHOICES` in a deterministic shuffled order, so the same cell set
    is reproducible from the seed.
    """
    if replicates < 2:
        raise ValueError("at least 2 replicates are required for an error term")
    baseline = baseline or load_sphere_baseline()
    model = model or InterlockingFriction()
    rng = np.random.default_rng(seed)
    cells: list[DesignCell] = []
    for r_target in roundness_levels:
        for c in concentrations:
            # a sharp regular n-gon has R_min(n) = 4 pi A_0 / P_0^2; filleting can
            # only raise R, so the side counts usable at this level are the ones
            # with R_min(n) <= target (nested sets, reported in the artifact).
            order = [n for n in SIDE_CHOICES
                     if RoundedPolygon(n, edge_length, 0.0).wadell_roundness <= r_target]
            if not order:
                raise ValueError(f"roundness level {r_target} is below every sharp "
                                 f"regular polygon in {SIDE_CHOICES}")
            rng.shuffle(order)
            for k in range(replicates):
                n = order[k % len(order)]
                base = RoundedPolygon(n, edge_length, 0.0)
                r = fillet_radius_for_roundness(base, r_target)
                poly = base.with_fillet(r)
                breakdown = model.breakdown(poly.interlocking_index, c, baseline)
                cells.append(DesignCell(
                    roundness_level=float(r_target),
                    concentration=float(c),
                    n_sides=int(n),
                    fillet_radius=float(r),
                    realized_roundness=float(poly.wadell_roundness),
                    interlocking_index=float(poly.interlocking_index),
                    mu_effective=float(breakdown["mu_effective"]),
                    mu_sphere=float(breakdown["mu_sphere"]),
                    coordination=float(breakdown["coordination"]),
                ))
    return cells


def two_way_anova(cells: Sequence[DesignCell]) -> dict:
    """Balanced two-way ANOVA with interaction on ``mu_effective``.

    Returns the full table plus ``eta_squared`` shares and the group means, in a
    JSON-serialisable form.  Raises on an unbalanced design (no silent pooling).
    """
    if not cells:
        raise ValueError("empty design")
    levels_a = sorted({c.roundness_level for c in cells})
    levels_b = sorted({c.concentration for c in cells})
    a, b = len(levels_a), len(levels_b)
    groups: Dict[Tuple[int, int], List[float]] = {}
    for cell in cells:
        key = (levels_a.index(cell.roundness_level), levels_b.index(cell.concentration))
        groups.setdefault(key, []).append(cell.mu_effective)
    counts = {k: len(v) for k, v in groups.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"unbalanced design, replicate counts {sorted(set(counts.values()))}")
    m = counts[(0, 0)]
    y = np.array([[groups[(i, j)] for j in range(b)] for i in range(a)], dtype=float)  # (a,b,m)
    grand = float(y.mean())
    means_ij = y.mean(axis=2)
    means_i = y.mean(axis=(1, 2))
    means_j = y.mean(axis=(0, 2))

    ss_a = b * m * float(np.sum((means_i - grand) ** 2))
    ss_b = a * m * float(np.sum((means_j - grand) ** 2))
    ss_ab = m * float(np.sum((means_ij - means_i[:, None] - means_j[None, :] + grand) ** 2))
    ss_e = float(np.sum((y - means_ij[:, :, None]) ** 2))
    ss_t = float(np.sum((y - grand) ** 2))

    df_a, df_b, df_ab, df_e = a - 1, b - 1, (a - 1) * (b - 1), a * b * (m - 1)
    ms_a, ms_b, ms_ab, ms_e = (ss_a / df_a, ss_b / df_b, ss_ab / df_ab,
                               ss_e / df_e if df_e else float("nan"))

    def _f_p(ms: float, df: int) -> Tuple[float, float]:
        if not (ms_e > 0.0):
            return (float("inf") if ms > 0.0 else 0.0, 0.0 if ms > 0.0 else 1.0)
        f = ms / ms_e
        return float(f), float(stats.f.sf(f, df, df_e))

    f_a, p_a = _f_p(ms_a, df_a)
    f_b, p_b = _f_p(ms_b, df_b)
    f_ab, p_ab = _f_p(ms_ab, df_ab)

    rows = [
        {"source": "roundness", "ss": ss_a, "df": df_a, "ms": ms_a, "F": f_a,
         "p": p_a, "significant_5pct": bool(p_a < 0.05),
         "eta_squared": ss_a / ss_t if ss_t > 0 else 0.0},
        {"source": "concentration", "ss": ss_b, "df": df_b, "ms": ms_b, "F": f_b,
         "p": p_b, "significant_5pct": bool(p_b < 0.05),
         "eta_squared": ss_b / ss_t if ss_t > 0 else 0.0},
        {"source": "roundness:concentration", "ss": ss_ab, "df": df_ab, "ms": ms_ab,
         "F": f_ab, "p": p_ab, "significant_5pct": bool(p_ab < 0.05),
         "eta_squared": ss_ab / ss_t if ss_t > 0 else 0.0},
        {"source": "residual", "ss": ss_e, "df": df_e, "ms": ms_e, "F": None,
         "p": None, "significant_5pct": None,
         "eta_squared": ss_e / ss_t if ss_t > 0 else 0.0},
    ]
    return {
        "response": "mu_effective",
        "levels_roundness": levels_a,
        "levels_concentration": levels_b,
        "replicates": m,
        "grand_mean": grand,
        "total_ss": ss_t,
        "table": rows,
        "cell_means": [[float(means_ij[i, j]) for j in range(b)] for i in range(a)],
        "main_effect_roundness": [float(v) for v in means_i],
        "main_effect_concentration": [float(v) for v in means_j],
        "interaction_significant": bool(p_ab < 0.05),
        "interpretation": (
            "interaction significant (p < 0.05): roundness and concentration are "
            "NOT separable - report the resistance decay per concentration, never "
            "as one blended roundness story"
            if p_ab < 0.05 else
            "interaction not significant at 5 %: the two main effects may be "
            "narrated separately at this resolution"
        ),
    }


def annex_table(cells: Sequence[DesignCell]) -> list:
    """Flat, JSON-ready replicate table (the report's appendix)."""
    return [c.as_dict() for c in cells]


def usable_side_counts(roundness_levels: Sequence[float],
                       edge_length: float = 0.6) -> dict:
    """Side counts available at each roundness level (the nested design sets)."""
    out = {}
    for level in roundness_levels:
        out[float(level)] = [n for n in SIDE_CHOICES
                             if RoundedPolygon(n, edge_length, 0.0).wadell_roundness <= level]
    return out