"""Anchor-closure kernel for topic ``ice-polygon-floe-field`` (W18 paper-exp, exp #1).

The research question is: *can the external quantitative anchor of ISO 19906:2019
(the pressure-area relation p(A) of Annex A and the design global ice action of
Clauses 14/15) be closed verbatim, and converted to the caliber of this case —
the unordered polygonal floe field of IN32-N01 — after which the domain result
can be compared with it?*

This module contains only the **operator layer**: the algebra that turns a
pressure-area power law into a floe-field global load, the dimension algebra
that proves the conversion homogeneous, and the internal-consistency identities
that can be checked *without* the standard's text.  Nothing here invents a
coefficient.  The registry layer (:data:`ANCHOR_SLOTS`, :func:`load_registry`,
:func:`evaluate_criteria`) decides whether the *absolute* layer may be claimed:
until a slot carries ``transcription_status == "VERIFIED"`` with a non-empty
verbatim quote plus a table/page reference, every absolute number stays
WITHHELD and any statement of compliance with ISO 19906 is forbidden.

Design of the caliber bridge (declared, not measured)
-----------------------------------------------------
Two calibers are carried side by side because they answer different questions
and the difference is the whole physics of a floe field:

``structure_face``
    The ISO-style wide-structure caliber: the pressure acts on the nominal
    contact patch ``A = W * h`` of the structure face (``W`` = face width,
    ``h`` = ice thickness).  Global action ``F = p(A) * A``.

``floe_patch``
    The DEM-side caliber of the polygon floe field: every floe transmits load
    across its own contact patch ``A_i = kappa * 2 * r_eq,i * h`` (chord width
    set by the equivalent radius), and the field aggregate is ``F = sum_i
    p(A_i) * A_i``.  This is the caliber in which a floe-size distribution
    enters the ice action at all.

Both are written as ``F = p0 * A_ref * S`` with ``S`` dimensionless, so the
*average-pressure exponent* and the *aggregate* are separated from the two
absolute constants (``p0``, ``A_ref``) that only the standard's text can supply.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ANCHOR_SLOTS",
    "DIM_AREA",
    "DIM_FORCE",
    "DIM_PRESSURE",
    "average_pressure",
    "contact_pressure",
    "dimensional_checks",
    "evaluate_criteria",
    "field_kernel",
    "field_load",
    "floe_patch_areas",
    "floe_force",
    "integral_identity_residual",
    "load_factor",
    "load_ratio",
    "log_log_slope",
    "load_registry",
    "registry_discipline",
    "scale_invariance_ratio",
    "consistency_checks",
    "kernel_log_sensitivity",
]

# --------------------------------------------------------------------------- #
# dimension algebra  (base dimensions kg, m, s with rational exponents)
# --------------------------------------------------------------------------- #
DIMLESS: Dict[str, Fraction] = {}
DIM_AREA: Dict[str, Fraction] = {m: Fraction(2) for m in ("m",)}
DIM_PRESSURE: Dict[str, Fraction] = {"kg": Fraction(1), "m": Fraction(-1),
                                     "s": Fraction(-2)}
DIM_FORCE: Dict[str, Fraction] = {"kg": Fraction(1), "m": Fraction(1),
                                  "s": Fraction(-2)}


def _clean(d: Mapping[str, Fraction]) -> Dict[str, Fraction]:
    return {k: Fraction(v) for k, v in d.items() if Fraction(v) != 0}


def dim_mul(a: Mapping[str, Fraction], b: Mapping[str, Fraction],
            power: int = 1) -> Dict[str, Fraction]:
    out = {k: Fraction(v) for k, v in a.items()}
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + Fraction(v) * power
    return _clean(out)


def dim_pow(a: Mapping[str, Fraction], e: Fraction) -> Dict[str, Fraction]:
    return _clean({k: Fraction(v) * Fraction(e) for k, v in a.items()})


def dim_of_pressure(n: float) -> Dict[str, Fraction]:
    """``[p0 * (A/A_ref)^-n]``: the area ratio is dimensionless for any real n."""
    return dim_mul(DIM_PRESSURE, dim_pow(DIM_AREA, Fraction(0)), 1)


def dim_of_field_kernel(n: float) -> Dict[str, Fraction]:
    """``[sum A_i^(1-n)] = m^(2(1-n))`` — not dimensionless unless n = 1."""
    return dim_pow(DIM_AREA, Fraction(1) - Fraction(n))


def dim_of_field_load(n: float) -> Dict[str, Fraction]:
    """``[p0 * A_ref^n * K(n)]`` — the A_ref exponent must cancel K's."""
    return dim_mul(dim_mul(DIM_PRESSURE, dim_pow(DIM_AREA, Fraction(n))),
                   dim_of_field_kernel(n))


# --------------------------------------------------------------------------- #
# the pressure-area operator
# --------------------------------------------------------------------------- #
def average_pressure(area, *, p0: float, a_ref: float, n: float):
    """``p(A) = p0 * (A / A_ref)^(-n)``: *average* pressure over contact area A."""
    area = np.asarray(area, dtype=np.float64)
    if a_ref <= 0.0:
        raise ValueError("a_ref must be positive")
    return p0 * (area / a_ref) ** (-n)


def floe_force(area, *, p0: float, a_ref: float, n: float):
    """``F(A) = p(A) * A``: load transmitted across contact area A."""
    area = np.asarray(area, dtype=np.float64)
    return average_pressure(area, p0=p0, a_ref=a_ref, n=n) * area


def contact_pressure(area, *, p0: float, a_ref: float, n: float):
    """Local contact pressure whose *moving average* reproduces ``p(A)``.

    ``p(A) = (1/A) \\int_0^A p_loc(a) da`` with
    ``p_loc(a) = p0 (1-n) (a/A_ref)^(-n)``, valid for the sub-linear branch
    ``0 < n < 1`` (``n >= 1`` makes the integral diverge: an average pressure
    law that does not decay slower than ``A^-1`` is not admissible).
    """
    if not (0.0 < n < 1.0):
        raise ValueError("contact-pressure form needs 0 < n < 1")
    area = np.asarray(area, dtype=np.float64)
    return p0 * (1.0 - n) * (area / a_ref) ** (-n)


def integral_identity_residual(area: float, *, p0: float, a_ref: float,
                               n: float, n_quad: int = 400) -> float:
    """Relative residual of ``p(A)*A == \\int_0^A p_loc(a) da`` (Gauss-Legendre).

    Substitution ``a = A u^(1/(1-n))`` removes the integrable singularity at
    ``a -> 0``, so a fixed-order Gauss-Legendre rule is spectrally accurate.
    """
    x, w = np.polynomial.legendre.leggauss(int(n_quad))
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    a = area * u ** (1.0 / (1.0 - n))
    da_du = area / (1.0 - n) * u ** (n / (1.0 - n))
    integrand = contact_pressure(a, p0=p0, a_ref=a_ref, n=n) * da_du
    quad = float(np.sum(integrand * wu))
    closed = float(floe_force(np.array([area]), p0=p0, a_ref=a_ref, n=n)[0])
    return abs(quad - closed) / abs(closed)


def field_kernel(areas: Sequence[float], n: float) -> float:
    """``K(n) = sum_i A_i^(1-n)`` — the anchor-free floe-field kernel [m^(2(1-n))]."""
    a = np.asarray(areas, dtype=np.float64)
    if np.any(a <= 0.0):
        raise ValueError("areas must be positive")
    return float(np.sum(a ** (1.0 - n)))


def field_load(areas: Sequence[float], *, p0: float, a_ref: float, n: float) -> float:
    """``F = sum_i p(A_i) A_i = p0 * A_ref^n * K(n)`` (both routes, one number)."""
    direct = float(np.sum(floe_force(np.asarray(areas), p0=p0, a_ref=a_ref, n=n)))
    via_kernel = p0 * a_ref ** n * field_kernel(areas, n)
    if abs(direct - via_kernel) > 1e-9 * max(1.0, abs(direct)):
        raise AssertionError(f"kernel/direct routes disagree: {direct} vs {via_kernel}")
    return direct


def floe_patch_areas(r_eq: Sequence[float], thickness: float, *,
                     kappa: float = 0.5) -> np.ndarray:
    """Contact-patch area of each floe: ``A_i = kappa * 2 r_eq,i * h`` [m^2]."""
    r = np.asarray(r_eq, dtype=np.float64)
    return 2.0 * float(kappa) * r * float(thickness)


def load_ratio(areas: Sequence[float], n: float, *, areas_ref: Sequence[float]) -> float:
    """Anchor-free load ratio between two floe fields: ``K_B(n) / K_A(n)``.

    ``p0`` and ``A_ref`` cancel because both configurations share the same ice
    and the same reference area — which is exactly why the *operator* layer can
    be verified while the *absolute* layer stays WITHHELD.
    """
    return field_kernel(areas, n) / field_kernel(areas_ref, n)


def load_factor(areas: Sequence[float], *, a_ref: float, n: float) -> float:
    """Dimensionless aggregate ``S = sum_i (A_i/A_ref)^(1-n)`` so ``F = p0 A_ref S``."""
    a = np.asarray(areas, dtype=np.float64)
    return float(np.sum((a / float(a_ref)) ** (1.0 - n)))


def kernel_log_sensitivity(areas: Sequence[float], n: float) -> float:
    """``d ln K / dn = -sum_i A_i^(1-n) ln A_i / K`` (how much the answer rides on n)."""
    a = np.asarray(areas, dtype=np.float64)
    w = a ** (1.0 - n)
    return float(-np.sum(w * np.log(a)) / np.sum(w))


def log_log_slope(areas: Sequence[float], pressures: Sequence[float]) -> Tuple[float, float]:
    """Least-squares slope of ``ln p`` vs ``ln A`` (and the max abs residual)."""
    la = np.log(np.asarray(areas, dtype=np.float64))
    lp = np.log(np.asarray(pressures, dtype=np.float64))
    slope, intercept = np.polyfit(la, lp, 1)
    resid = float(np.max(np.abs(lp - (slope * la + intercept))))
    return float(slope), resid


def scale_invariance_ratio(*, p0: float, a_ref: float, n: float, area: float,
                           lam: float) -> Tuple[float, float]:
    """``p(lam*A)/p(A)`` vs the closed form ``lam^-n``."""
    a = np.array([area])
    got = float(average_pressure(a * lam, p0=p0, a_ref=a_ref, n=n)[0]
                / average_pressure(a, p0=p0, a_ref=a_ref, n=n)[0])
    return got, float(lam) ** (-n)


# --------------------------------------------------------------------------- #
# machine checks (all anchor-free)
# --------------------------------------------------------------------------- #
def dimensional_checks(n: float, *, a_ref_exponent: Optional[float] = None
                       ) -> Dict[str, object]:
    """Homogeneity table of the conversion chain, with the ``A_ref`` trap shown."""
    rows: List[Dict[str, object]] = []

    def row(name: str, lhs: Mapping[str, Fraction], rhs: Mapping[str, Fraction],
            note: str = "") -> None:
        rows.append({
            "check": name,
            "lhs": {k: str(v) for k, v in sorted(lhs.items())},
            "rhs": {k: str(v) for k, v in sorted(rhs.items())},
            "pass": lhs == rhs,
            "note": note,
        })

    row("A/A_ref 无量纲（指数段不携带量纲）",
        dim_mul(DIM_AREA, DIM_AREA, power=-1), DIMLESS)
    row("p(A)=p0·(A/A_ref)^(-n) 量纲 = [Pa]", dim_of_pressure(n), DIM_PRESSURE)
    row("F=p(A)·A 量纲 = [N]",
        dim_mul(dim_of_pressure(n), DIM_AREA, power=1), DIM_FORCE)
    row("K(n)=ΣA_i^(1-n) 量纲 = m^(2(1-n))",
        dim_of_field_kernel(n), dim_pow(DIM_AREA, Fraction(1) - Fraction(n)),
        note="非无量纲：n≠1 时 K 带量纲，故不能单独当作物理力")
    row("F=p0·A_ref^n·K(n) 量纲 = [N]（A_ref 指数必须抵消 K）",
        dim_of_field_load(n), DIM_FORCE)
    # the trap: dropping A_ref (i.e. setting A_ref = 1 m) hides a unit
    trap = dim_of_field_load(n) if a_ref_exponent is None else dim_mul(
        dim_mul(DIM_PRESSURE, dim_pow(DIM_AREA, Fraction(a_ref_exponent))),
        dim_of_field_kernel(n))
    rows.append({
        "check": "A_ref 归一化敏感性（A_ref 取 1 m 只是单位选择）",
        "exponent_used": 0.0 if a_ref_exponent is None else a_ref_exponent,
        "resulting_dims": {k: str(v) for k, v in sorted(trap.items())},
        "equals_N": trap == DIM_FORCE,
        "pass": (trap == DIM_FORCE) if a_ref_exponent is None else True,
        "note": "A_ref=1 m 时 F 的量纲退化为 m^(2-2n)/…，数值不是牛顿",
    })
    return {"rows": rows, "pass": all(bool(r["pass"]) for r in rows)}


def consistency_checks(*, n: float, p0: float, a_ref: float,
                       areas: Sequence[float], threshold_residual: float = 1e-9
                       ) -> Dict[str, object]:
    """Internal consistency of the operator: monotonicity, exponent recovery, …"""
    grid = np.logspace(np.log10(1e-3), np.log10(1e4), 401)
    p = average_pressure(grid, p0=p0, a_ref=a_ref, n=n)
    f = floe_force(grid, p0=p0, a_ref=a_ref, n=n)
    rows: List[Dict[str, object]] = []

    rows.append({"check": "p(A) 严格递减（401 点零违例）",
                 "violations": int(np.sum(np.diff(p) >= 0.0)),
                 "pass": bool(np.all(np.diff(p) < 0.0))})
    rows.append({"check": "F(A)=p·A 递增且亚线性（0<n<1）",
                 "violations": int(np.sum(np.diff(f) <= 0.0)),
                 "pass": bool(np.all(np.diff(f) > 0.0) and 0.0 < n < 1.0)})
    slope, resid = log_log_slope(grid, p)
    rows.append({"check": "d ln p / d ln A = -n",
                 "fitted": slope, "expected": -n, "fit_residual": resid,
                 "pass": abs(slope + n) < 1e-10 and resid < 1e-10})
    got, exp = scale_invariance_ratio(p0=p0, a_ref=a_ref, n=n, area=1.0, lam=10.0)
    rows.append({"check": "尺度不变性 p(λA)/p(A) = λ^(-n)（λ=10）",
                 "ratio": got, "expected": exp,
                 "pass": abs(got - exp) <= 1e-12 * abs(exp)})
    ii = [integral_identity_residual(a, p0=p0, a_ref=a_ref, n=n)
          for a in (0.05, 1.0, 5.0, 88.8)]
    rows.append({"check": "局部-平均一致性 p(A)·A = ∫_0^A p_loc da（4 个面积点）",
                 "residual_max": float(max(ii)),
                 "pass": max(ii) < 1e-10,
                 "note": "把『面积平均压力』与『局部接触压力幂律』钉在一起；无锚位也能检"})
    # discrete floe aggregate vs the same aggregate written through the kernel
    k = field_kernel(areas, n)
    direct = float(np.sum(floe_force(np.asarray(areas, dtype=float),
                                     p0=p0, a_ref=a_ref, n=n)))
    via = p0 * a_ref ** n * k
    rows.append({"check": "聚合双路一致：Σ p(A_i)A_i vs p0·A_ref^n·K(n)",
                 "direct": direct, "via_kernel": via,
                 "rel_residual": abs(direct - via) / abs(direct),
                 "pass": abs(direct - via) / abs(direct) < threshold_residual})
    return {"rows": rows, "pass": all(bool(r["pass"]) for r in rows)}


# --------------------------------------------------------------------------- #
# source registry (the only place where an anchor value may enter)
# --------------------------------------------------------------------------- #
ANCHOR_SLOTS: List[Dict[str, object]] = [
    {
        "id": "iso19906-2019-AnnexA-pA",
        "standard": ("ISO 19906:2019 Petroleum and natural gas industries — "
                     "Arctic offshore structures"),
        "anchor_kind": "pressure-area law p(A) 的幂律指数 n 与参考压力 p0 / 参考面积 A_ref",
        "clause": "Annex A（压力-面积关系）",
        "quote_verbatim": None,
        "quote_lang": "en",
        "table_or_page": None,
        "value": None,
        "value_units": {"n": "1 (dimensionless)", "p0": "Pa", "A_ref": "m^2"},
        "transcription_status": "WITHHELD",
        "withheld_reason": ("沙箱内无 ISO 19906:2019 正本；该标准为付费出版物，"
                            "公开渠道只有范围/目录级预览，不含 Annex A 正文。"
                            "按溯源纪律『禁凭记忆写数』，本槽位留空。"),
        "value_must_come_from": "原文逐字抄录（quote_verbatim + table_or_page 非空 + value 原值）",
    },
    {
        "id": "iso19906-2019-Clause14-15-global",
        "standard": ("ISO 19906:2019 Petroleum and natural gas industries — "
                     "Arctic offshore structures"),
        "anchor_kind": "设计全球冰力（global ice action）条款式与条款值",
        "clause": "Clause 14 / Clause 15（全球冰作用 / 设计方法）",
        "quote_verbatim": None,
        "quote_lang": "en",
        "table_or_page": None,
        "value": None,
        "value_units": {"F": "N"},
        "transcription_status": "WITHHELD",
        "withheld_reason": ("同上：无正本副本，表号/页码不可得；不以二手转述或"
                           "记忆中的系数替代原文。"),
        "value_must_come_from": "原文逐字抄录（quote_verbatim + table_or_page 非空 + value 原值）",
    },
]

_REQUIRED_VERIFIED_FIELDS = ("quote_verbatim", "table_or_page", "value")


def load_registry(path: Optional[str] = None) -> List[Dict[str, object]]:
    """Return the anchor registry, optionally overlaid by an injection file.

    The overlay exists so a *real* verbatim transcription can be dropped in
    later without touching code; it is validated here so a half-filled file
    cannot silently promote a slot to VERIFIED.
    """
    registry = [dict(slot) for slot in ANCHOR_SLOTS]
    if path is None:
        return registry
    with open(path, "r", encoding="utf-8") as fh:
        overlay = json.load(fh)
    entries = overlay.get("sources", overlay) if isinstance(overlay, dict) else overlay
    by_id = {str(e["id"]): e for e in entries if "id" in e}
    for slot in registry:
        inc = by_id.get(slot["id"])
        if inc is None:
            continue
        for key in ("quote_verbatim", "table_or_page", "value", "value_units",
                    "clause", "notes"):
            if key in inc:
                slot[key] = inc[key]
        status = str(inc.get("transcription_status", "")).upper()
        missing = [k for k in _REQUIRED_VERIFIED_FIELDS
                   if slot.get(k) in (None, "", [], {})]
        if status == "VERIFIED" and missing:
            raise ValueError(
                f"slot {slot['id']}: VERIFIED needs non-empty {missing}; "
                "a partial transcription may not be promoted")
        if status:
            slot["transcription_status"] = status
    return registry


def registry_discipline(registry: Iterable[Mapping[str, object]],
                        criteria: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """No PASS may rest on an unverified anchor or on a comparison-only value."""
    verified = {str(s["id"]) for s in registry
                if str(s.get("transcription_status", "")).upper() == "VERIFIED"}
    violations: List[str] = []
    for c in criteria:
        if str(c.get("status", "")).split()[0] != "PASS":
            continue
        src = c.get("value_source_id")
        if c.get("requires_anchor_value") and (src is None or str(src) not in verified):
            violations.append(f"{c.get('id')}: PASS 但锚位 {src!r} 未核销")
        for bad in ("cand-", "unverified", "reference_only"):
            if c.get("measured") and bad in json.dumps(c.get("measured"),
                                                       ensure_ascii=False):
                violations.append(f"{c.get('id')}: measured 中出现对照专用标记 {bad!r}")
    return {
        "verified_slots": sorted(verified),
        "violations": violations,
        "pass": not violations,
        "rule": ("未核候选/二手值只能出现在 comparison 字段；PASS 判据的 measured "
                 "必须溯源到 VERIFIED 原文锚位"),
    }


def evaluate_criteria(registry: Sequence[Mapping[str, object]], *,
                      recompute_residual: float,
                      absolute_loads: Optional[Mapping[str, float]],
                      checks: Mapping[str, Mapping[str, object]],
                      threshold_residual: float = 1e-9) -> List[Dict[str, object]]:
    """Turn the registry state + machine checks into the four pre-registered criteria.

    Kept as a pure function so that (a) the honest-FAIL state and (b) the
    verified state are the *same code path*, differing only in registry content —
    the injection path can therefore be tested with a labelled fixture.
    """
    verified = [s for s in registry
                if str(s.get("transcription_status", "")).upper() == "VERIFIED"]
    quotes = [s for s in verified
              if str(s.get("quote_verbatim") or "").strip()
              and str(s.get("table_or_page") or "").strip()
              and s.get("value") is not None]
    c1_pass = len(quotes) >= 2
    operator_ok = recompute_residual < threshold_residual
    absolute_ok = c1_pass and absolute_loads is not None
    c2_pass = operator_ok and absolute_ok

    crit: List[Dict[str, object]] = []
    crit.append({
        "id": "C1",
        "statement": ("引句 + 表号/页码 + 数值逐字录入工件（值=来源原值，禁改写/禁四舍五入失真）"),
        "status": "PASS" if c1_pass else "FAIL",
        "measured": {
            "verified_anchors": len(quotes),
            "required": 2,
            "quotes_present": sum(1 for s in quotes
                                  if str(s.get("quote_verbatim") or "").strip()),
            "slots": [{"id": s["id"],
                       "transcription_status": s.get("transcription_status")}
                      for s in registry],
        },
        "value_source_id": (quotes[0]["id"] if c1_pass else None),
        "gap": (None if c1_pass else
                "0/2 锚位核销：ISO 19906:2019 正本不可得 → 引句/表号/页码/原值全空，"
                "状态 WITHHELD；不以记忆或二手转述填数"),
        "criterion_pre_registered": True,
        "requires_anchor_value": True,
    })
    crit.append({
        "id": "C2",
        "statement": ("同口径换算脚本落盘且可复算（复算残差 <1e-9），并换算到本算例口径"),
        "status": ("PASS" if c2_pass else
                   ("PASS(operator) / WITHHELD(absolute)" if operator_ok else "FAIL")),
        "measured": {
            "recompute_residual_max_rel": recompute_residual,
            "threshold": threshold_residual,
            "operator_layer": "PASS" if operator_ok else "FAIL",
            "absolute_layer": "PASS" if absolute_ok else "WITHHELD",
            "absolute_loads_N": dict(absolute_loads) if absolute_loads else None,
        },
        "value_source_id": (registry[0]["id"] if absolute_ok else None),
        "gap": (None if c2_pass else
                "算子层已复算达标（残差 <1e-9，脚本落盘可复跑）；绝对层需要 p0 与 A_ref "
                "（Annex A 原文），故『换到本算例口径』的牛顿数不可声明"),
        "criterion_pre_registered": True,
        "requires_anchor_value": absolute_ok is False,
    })
    crit.append({
        "id": "C3",
        "statement": "量纲齐次与内部一致性机检通过",
        "status": "PASS" if (checks["dimensional"]["pass"]
                             and checks["consistency"]["pass"]) else "FAIL",
        "measured": {
            "dimensional_pass": bool(checks["dimensional"]["pass"]),
            "dimensional_rows": len(checks["dimensional"]["rows"]),
            "consistency_pass": bool(checks["consistency"]["pass"]),
            "consistency_rows": len(checks["consistency"]["rows"]),
        },
        "value_source_id": None,
        "gap": None,
        "criterion_pre_registered": True,
        "requires_anchor_value": False,
    })
    disc = checks["registry_discipline"]
    crit.append({
        "id": "C4",
        "statement": "未核候选登记值不得直接引用（只能作为核对对照）",
        "status": "PASS" if disc["pass"] else "FAIL",
        "measured": {"violations": disc["violations"],
                     "verified_slots": disc["verified_slots"]},
        "value_source_id": None,
        "gap": None,
        "criterion_pre_registered": True,
        "requires_anchor_value": False,
    })
    return crit


def overall_verdict(criteria: Sequence[Mapping[str, object]]) -> Tuple[str, str]:
    """Verdict = the *answer to the research question*, not the sum of criteria."""
    by_id = {c["id"]: c for c in criteria}
    c1 = str(by_id["C1"]["status"]).split()[0]
    if c1 != "PASS":
        return ("FAIL",
                "外部锚未核销（C1 FAIL，状态 WITHHELD）→ 本课题与规范的差额无法计算；"
                "算子层/量纲/一致性已机检通过（C2 operator, C3, C4），"
                "按诚实纪律记 honest FAIL，不粉饰、不改判据")
    missing = [cid for cid, c in by_id.items() if str(c["status"]).split()[0] != "PASS"]
    if missing:
        return ("PARTIAL", f"锚位已核销，但判据 {missing} 未达标")
    return ("PASS", "两个外部锚逐字核销，换算闭环，机检通过")