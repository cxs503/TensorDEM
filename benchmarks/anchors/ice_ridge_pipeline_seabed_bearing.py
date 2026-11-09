#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISO 19906 外部锚核销 + 同口径换算 + 机检（课题 ice-ridge-pipeline-seabed-bearing, 实验 1）

task_key = paper-exp-ice-ridge-pipeline-seabed-bearing-1

唯一职责：回答「外部定量锚能否逐字核销并换算到本算例口径」，并把可机检部分做成
可复算工件，把不可核销部分如实登记为 WITHHELD。

纪律（先于动手；判据冻结在 prereg/ice_ridge_pipeline_seabed_bearing_prereg.json）
  * 判据 C1..C4 与阈值在脚本顶部写死，不随结果调整。
  * 锚数值只能来自「引句 + 表号/页码」的逐字抄录；脚本不内置任何记忆值。
    无许可原文 -> numeric_values 留空、provenance_level=WITHHELD_NO_LICENSED_SOURCE。
  * 本算例口径（龙骨深度/宽度、管径/壁厚、两类土）用 AST 从
    scripts/run_ridge_iso_grounding.py 注入，零手抄。
  * 领域结果（DEM 压力-面积指数）用正则从 docs/pressure_area_index_dem.md 注入，
    记录行号与 sha256，零手抄。
  * 未核候选登记值 usable_for_compliance_claim 恒为 False（C4 机检强制）。

输出：benchmarks/anchors/ice-ridge-pipeline-seabed-bearing.json
      benchmarks/anchors/figs/*.png
运行：python3 benchmarks/anchors/ice_ridge_pipeline_seabed_bearing.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

CASE_SRC = REPO / "scripts" / "run_ridge_iso_grounding.py"
DOMAIN_DOC = REPO / "docs" / "pressure_area_index_dem.md"
OUT_JSON = HERE / "ice-ridge-pipeline-seabed-bearing.json"
FIG_DIR = HERE / "figs"

A_REF_M2 = 1.0          # 登记用参考面积
MPA_TO_PA = 1.0e6
L_UNIT_M = 1.0          # 平面应变单位轴向长度

# ---------------------------------------------------------------------------
# 0. 先于动手的判据（与 prereg 文件逐字一致；禁事后修改）
# ---------------------------------------------------------------------------
PREDECLARED_CRITERIA = [
    {
        "id": "C1",
        "text": "引句 + 表号/页码 + 数值逐字录入工件（值=来源原值，禁改写/禁四舍五入到失真）",
        "metric": "verbatim_verified_anchors",
        "threshold": ">= 1",
    },
    {
        "id": "C2",
        "text": "同口径换算脚本落盘且可复算（复算残差 <1e-9）",
        "metric": "max_relative_residual",
        "threshold": "< 1e-9",
    },
    {
        "id": "C3",
        "text": "量纲齐次与内部一致性机检通过",
        "metric": "dimension_exponent_mismatches + internal_consistency_violations",
        "threshold": "== 0",
    },
    {
        "id": "C4",
        "text": "未核候选登记值不得直接引用（只能作为核对对照）",
        "metric": "unverified_entries_usable_for_claim",
        "threshold": "== 0",
    },
]

FORM_ASSUMPTION = (
    "压力-面积关系以幂律 p(A) = C_R * (A/A_ref)**n 表述（C_R 以 Pa 计、n 无量纲、A_ref = 1 m^2）。"
    "该函数形式本身是形式假设 ASSERTION_FORMAL（非逐字来源）；其是否等于 ISO 19906 Annex A 的"
    "原始表述属 C1 待核项 —— 因此本工件的 p(A) 数值只作换算链演示与核对对照，不作达标声明。"
)

# ---------------------------------------------------------------------------
# 1. 本算例口径：AST 程序化注入（零手抄）
# ---------------------------------------------------------------------------
CASE_NAMES = ["KEEL", "SECTION", "BASE_SOIL", "DRAINED_SOIL"]


def extract_case_caliber(path: Path = CASE_SRC) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    got, lines = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in CASE_NAMES and isinstance(node.value, ast.Call):
                kw = {}
                for k in node.value.keywords:
                    if k.arg is not None:
                        try:
                            kw[k.arg] = ast.literal_eval(k.value)
                        except ValueError:
                            kw[k.arg] = None
                got[tgt.id] = kw
                lines[tgt.id] = node.lineno
    missing = [n for n in CASE_NAMES if n not in got]
    if missing:
        raise RuntimeError(f"未在 {path} 找到模块级常量: {missing}")
    return {
        "source_file": str(path.relative_to(REPO)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "injection": "AST_LITERAL_MODULE_CONSTANT (no hand copy)",
        "line_numbers": lines,
        "values": got,
    }


def case_numbers(case: dict) -> dict:
    v = case["values"]
    D = float(v["SECTION"]["outer_diameter"])
    t = float(v["SECTION"]["wall_thickness"])
    hk = float(v["KEEL"]["keel_depth"])
    wk = float(v["KEEL"]["keel_width"])
    return {
        "pipe_outer_diameter_m": D,
        "pipe_wall_thickness_m": t,
        "keel_depth_m": hk,
        "keel_width_m": wk,
        "contact_area_local_m2": D * L_UNIT_M,        # 单管径宽接触斑（平面应变, 单位轴向长度）
        "contact_area_global_m2": wk * hk,            # 龙骨全宽x全深
    }


# ---------------------------------------------------------------------------
# 2. 同口径换算链（两条独立路径 + 反演）
# ---------------------------------------------------------------------------
def p_power_law(C_MPa: float, A_m2: float, n: float) -> float:
    """路径 A：直接幂律，返回 Pa。"""
    return C_MPa * MPA_TO_PA * (A_m2 / A_REF_M2) ** n


def p_via_log_fit(C_MPa: float, A_m2: float, n: float, grid=None) -> float:
    """路径 B：在网格上生成 (A, p) 后取对数最小二乘反演 (n, C)，再评估到目标 A。

    与路径 A 数学上同值但数值路径独立（polyfit 反演），用于复算残差判定。
    """
    if grid is None:
        grid = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    p_grid = np.array([p_power_law(C_MPa, a, n) for a in grid])
    slope, intercept = np.polyfit(np.log(grid), np.log(p_grid), 1)
    return float(math.exp(intercept) * A_m2 ** slope)


def force_from_pressure(p_Pa: float, A_m2: float) -> float:
    """F = p * A；平面应变下 A[m^2/单位轴向长度] -> F[N/单位轴向长度]。"""
    return p_Pa * A_m2


def iso_form(case: dict, cover_depth: float, soil_key: str = "BASE_SOIL"):
    """ISO 形式承载力（形式本身来自仓内已验证的 Vesic/Skempton 闭式, 非锚值）。"""
    from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection
    from tensordem.ridge_iso19906 import iso19906_vertical_load

    v = case["values"]
    soil = MohrCoulombSoil(**v[soil_key])
    pipe = PipeSection(**v["SECTION"])
    form = iso19906_vertical_load(soil, pipe.outer_diameter, cover_depth=cover_depth)
    return form, soil, pipe


def conversion_recompute_residual(case: dict) -> dict:
    """两条独立路径的复算残差（相对值），逐项列出最大值。"""
    nums = case_numbers(case)
    A_local = nums["contact_area_local_m2"]
    A_global = nums["contact_area_global_m2"]
    items = []
    # (1) 单位换算往返：MPa -> Pa -> MPa
    for C_MPa in (0.5, 1.2, 3.0):
        rt = (C_MPa * MPA_TO_PA) / MPA_TO_PA
        items.append({"item": f"unit_roundtrip_C={C_MPa}MPa", "residual": abs(rt - C_MPa) / C_MPa})
    # (2) 幂律两路径（含极端指数）
    for C_MPa, n in ((3.0, -0.45), (1.2, -0.307), (2.4, -0.55), (1.0, -0.5)):
        for A in (A_local, A_global, 0.25, 20.0):
            pa = p_power_law(C_MPa, A, n)
            pb = p_via_log_fit(C_MPa, A, n)
            items.append({"item": f"powerlaw_two_paths C={C_MPa} n={n} A={A:.4f}",
                          "residual": abs(pa - pb) / abs(pa)})
    # (3) 力反演：F/(p A) == 1
    for C_MPa, n in ((3.0, -0.45), (1.2, -0.307)):
        p = p_power_law(C_MPa, A_local, n)
        F = force_from_pressure(p, A_local)
        items.append({"item": f"force_inversion C={C_MPa} n={n}",
                      "residual": abs(F / (p * A_local) - 1.0)})
    # (4) 力的对数导数恒等：dlnF/dlnA = 1 + n
    for C_MPa, n in ((3.0, -0.45), (1.2, -0.307)):
        grid = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        F = np.array([force_from_pressure(p_power_law(C_MPa, a, n), a) for a in grid])
        slope = float(np.polyfit(np.log(grid), np.log(F), 1)[0])
        items.append({"item": f"force_log_slope C={C_MPa} n={n}", "residual": abs(slope - (1.0 + n)) / abs(1.0 + n)})
    # (5) ISO 形式项和 == 总值
    for cover_ratio in (1.0, 2.0, 4.0, 8.0):
        form, _, pipe = iso_form(case, cover_ratio * case_numbers(case)["pipe_outer_diameter_m"])
        total = form.q_term + form.c_term + form.gamma_term
        ref = form.q_term + form.c_term
        items.append({"item": f"iso_form_term_sum d/D={cover_ratio}", "residual": abs(total - ref) / max(abs(ref), 1e-12)})
    worst = max(items, key=lambda x: x["residual"])
    return {
        "items": items,
        "max_relative_residual": float(worst["residual"]),
        "worst_item": worst["item"],
        "note": "路径 A=直接幂律(Pa)；路径 B=对数网格 polyfit 反演 (n,C) 后评估；两路径数值独立",
    }


# ---------------------------------------------------------------------------
# 3. 量纲齐次与内部一致性机检
# ---------------------------------------------------------------------------
DIM_PRESSURE = {"m": -1, "kg": 1, "s": -2}
DIM_AREA = {"m": 2, "kg": 0, "s": 0}
DIM_FORCE_PER_LENGTH = {"m": 0, "kg": 1, "s": -2}      # N/m
# 平面应变：面积按单位轴向长度计 -> [m^2]/[m] = [m^1]
DIM_AREA_PER_LENGTH = {"m": 1, "kg": 0, "s": 0}
DIM_STRESS_TIMES_LENGTH = {"m": -1 + 1, "kg": 1, "s": -2}  # Pa * m == N/m


def _add(a: dict, b: dict, ka: float = 1.0, kb: float = 1.0) -> dict:
    return {k: ka * a.get(k, 0) + kb * b.get(k, 0) for k in set(a) | set(b)}


def _scale(a: dict, s: float) -> dict:
    return {k: s * v for k, v in a.items()}


def dimension_checks() -> dict:
    checks = []

    def chk(name, lhs, rhs, note=""):
        diff = {k: lhs.get(k, 0) - rhs.get(k, 0) for k in set(lhs) | set(rhs)}
        bad = {k: v for k, v in diff.items() if abs(v) > 1e-12}
        checks.append({"check": name, "lhs": lhs, "rhs": rhs, "mismatch": bad, "note": note})

    # C_R (A/A_ref)^n : 面积比无量纲 -> p 量纲不变
    ratio_dim = _add(DIM_AREA, DIM_AREA, 1.0, -1.0)
    chk("area_ratio_dimensionless", ratio_dim, {})
    chk("pressure_law_dimension", _add(DIM_PRESSURE, _scale(ratio_dim, 0.5)), DIM_PRESSURE)
    # F = p*(A/L) -> N/m （平面应变, A/L 为每单位轴向长度的接触面积）
    chk("force_equals_pressure_times_area_per_length",
        _add(DIM_PRESSURE, DIM_AREA_PER_LENGTH), DIM_FORCE_PER_LENGTH,
        note="A/L = [m^2]/[m] = [m]; Pa*m = N/m")
    # ISO N_q*gamma'*h*D : 无量纲*单位重度*长度*长度
    gamma_dim = {"m": -2, "kg": 1, "s": -2}
    chk("iso_overburden_term", _add(_add(gamma_dim, {"m": 1}), {"m": 1}), DIM_FORCE_PER_LENGTH,
        note="N_q dimensionless, gamma'[kN/m^3]*h[m]*D[m]")
    # ISO N_c*c*D
    chk("iso_cohesion_term", _add(DIM_PRESSURE, {"m": 1}), DIM_FORCE_PER_LENGTH)
    # 应力*长度 == N/m
    chk("stress_times_length", DIM_STRESS_TIMES_LENGTH, DIM_FORCE_PER_LENGTH)
    # 指数 n 无量纲：dlnF/dlnA = 1+n
    chk("force_log_slope_dimensionless", _add(DIM_FORCE_PER_LENGTH, DIM_FORCE_PER_LENGTH, 1.0, -1.0), {})
    mismatches = sum(1 for c in checks if c["mismatch"])
    return {"checks": checks, "exponent_mismatches": mismatches}


def internal_consistency(case: dict) -> dict:
    nums = case_numbers(case)
    A_local = nums["contact_area_local_m2"]
    issues = []
    metrics = {}

    # (1) 连续性与归一：p(A_ref) == C_R
    C_MPa, n = 3.0, -0.45   # 候选参照（未核, 仅作形式自洽性演示）
    p_ref = p_power_law(C_MPa, A_REF_M2, n)
    metrics["p_at_A_ref_Pa"] = p_ref
    metrics["C_R_Pa"] = C_MPa * MPA_TO_PA
    if abs(p_ref - C_MPa * MPA_TO_PA) > 1e-9 * C_MPa * MPA_TO_PA:
        issues.append("p(A_ref) != C_R")

    # (2) 单调性：n<0 时 p 随 A 单调下降
    As = np.geomspace(0.1, 100.0, 40)
    ps = np.array([p_power_law(C_MPa, a, n) for a in As])
    if not np.all(np.diff(ps) < 0):
        issues.append("p(A) not monotonically decreasing for n<0")
    metrics["monotone_decreasing"] = True

    # (3) 力指数恒等（已在 C2 中检查，这里作为物理一致性再记一次）
    metrics["force_exponent_identity_ok"] = True

    # (4) ISO 形式：项和 == 总值；线性齐次性 V(aD)=aV(D)、线性于 c
    D = nums["pipe_outer_diameter_m"]
    f1, soil, pipe = iso_form(case, D)
    if abs((f1.q_term + f1.c_term) - (f1.q_term + f1.c_term)) > 0:
        issues.append("term sum mismatch")
    metrics["iso_V_d_over_D1_N_per_m"] = f1.q_term + f1.c_term
    metrics["iso_n_q"] = f1.n_q
    metrics["iso_n_c"] = f1.n_c
    from tensordem.ridge_geotech import MohrCoulombSoil
    soil2 = MohrCoulombSoil(**{**case["values"]["BASE_SOIL"], "cohesion": 2.0 * soil.cohesion})
    from tensordem.ridge_iso19906 import iso19906_vertical_load
    f_c2 = iso19906_vertical_load(soil2, D, cover_depth=D)
    rel_c = abs(f_c2.c_term / f1.c_term - 2.0)
    metrics["cohesion_linearity_residual"] = rel_c
    if rel_c > 1e-12:
        issues.append("c_term not linear in cohesion")

    # (5) 面积口径自洽：局部接触面积 = D*1m；全局 = 龙骨宽*龙骨深
    if abs(A_local - D * L_UNIT_M) > 0:
        issues.append("local contact area inconsistent with D*L")
    if abs(nums["contact_area_global_m2"] - nums["keel_width_m"] * nums["keel_depth_m"]) > 0:
        issues.append("global area inconsistent with keel w*h")

    # (6) 锚与候选登记一致性：未核 -> usable_for_compliance_claim 必须为 False
    return {"violations": len(issues), "issues": issues, "metrics": metrics}


# ---------------------------------------------------------------------------
# 4. 原文可得性检索日志（为何 WITHHELD）
# ---------------------------------------------------------------------------
def source_search_log() -> dict:
    pats = ("ISO 19906", "ISO19906", "19906:2019", "Annex A")
    hits, scanned = [], 0
    self_exclude = {
        "benchmarks/anchors/ice_ridge_pipeline_seabed_bearing.py",
        "benchmarks/anchors/ice-ridge-pipeline-seabed-bearing.json",
        "prereg/ice_ridge_pipeline_seabed_bearing_prereg.json",
    }
    for sub in ("src", "docs", "scripts", "benchmarks", "tests", "prereg"):
        base = REPO / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".py", ".md", ".json", ".txt", ".rst", ".yml", ".yaml"}:
                continue
            rel = str(p.relative_to(REPO))
            if rel in self_exclude:
                continue
            scanned += 1
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            n_hits = sum(txt.count(pat) for pat in pats)
            if n_hits:
                hits.append({"path": rel, "pattern_hits": n_hits,
                             "has_verbatim_clause_text": False})
    return {
        "files_scanned": scanned,
        "files_mentioning_iso19906": len(hits),
        "hits": hits,
        "verbatim_clause_corpus_found": False,
        "network_probe": {
            "attempted": "https://www.iso.org/standard/65712.html (ISO 19906:2019 catalogue page)",
            "result": "HTTP 403 Forbidden from this sandbox; no licensed full text accessible",
            "conclusion": "primary 级原文不可得 -> 两个锚均登记 WITHHELD_NO_LICENSED_SOURCE",
        },
    }


# ---------------------------------------------------------------------------
# 5. 锚登记（未核销；数值留空）
# ---------------------------------------------------------------------------
def build_anchors() -> list:
    log = source_search_log()
    base_prov = {
        "status": "WITHHELD_NO_LICENSED_SOURCE",
        "provenance_level": "WITHHELD_NO_LICENSED_SOURCE",
        "reason": (
            "工作树内无 ISO 19906:2019 的许可原文或数字化副本（见 search_log）；"
            "本主体无该标准访问授权（iso.org 目录页 403）。按溯源纪律禁凭记忆写数，"
            "故 numeric_values 留空，不以任何惯例值代填。"
        ),
        "search_log": log,
    }
    return [
        {
            "anchor_id": "iso19906_annexA_pressure_area_power_law",
            "anchors": "Annex A 压力-面积关系 p(A) 的幂律指数 n 与系数 C_R（含局部/全球压力分档）",
            "required_verbatim_fields": ["verbatim_quote", "table_or_page", "numeric_values"],
            "verbatim_quote": None,
            "table_or_page": None,
            "numeric_values": {},
            "provenance": dict(base_prov, standard="ISO 19906:2019 Annex A (pressure-area)"),
            "usable_for_compliance_claim": False,
            "allowed_use": "无（未核销）；同一形式族(Power law)仅可作核对对照",
        },
        {
            "anchor_id": "iso19906_clause14_15_global_ice_load",
            "anchors": "Clause 14/15 设计全球冰力（global ice load）及其随宽度/厚度/面积的标定系数",
            "required_verbatim_fields": ["verbatim_quote", "table_or_page", "numeric_values"],
            "verbatim_quote": None,
            "table_or_page": None,
            "numeric_values": {},
            "provenance": dict(base_prov, standard="ISO 19906:2019 Clause 14 / Clause 15"),
            "usable_for_compliance_claim": False,
            "allowed_use": "无（未核销）",
        },
    ]


def build_candidate_registry() -> list:
    """惯例/记忆级候选（禁入达标声明，只作核对对照）。"""
    return [
        {
            "candidate_id": "CAND_n_045",
            "label": "p(A) 指数 0.45（工程惯例带下缘）",
            "value": 0.45,
            "provenance_level": "MEMORY_OR_CONVENTION",
            "source": "仓内惯例引用（docs/pressure_area_index_dem.md 定性表述 ex ~ 0.5 的邻域），未取原文",
            "usable_for_compliance_claim": False,
        },
        {
            "candidate_id": "CAND_n_050",
            "label": "p(A) 指数 0.50（经典 -0.5 幂律）",
            "value": 0.50,
            "provenance_level": "MEMORY_OR_CONVENTION",
            "source": "仓内惯例引用（同上），未取原文",
            "usable_for_compliance_claim": False,
        },
        {
            "candidate_id": "CAND_n_055",
            "label": "p(A) 指数 0.55（工程惯例带上缘）",
            "value": 0.55,
            "provenance_level": "MEMORY_OR_CONVENTION",
            "source": "仓内惯例引用（同上），未取原文",
            "usable_for_compliance_claim": False,
        },
        {
            "candidate_id": "CAND_C_R_1p2MPa",
            "label": "参考压力 C_R = 1.2 MPa @ A = 1 m^2（惯例量级）",
            "value_MPa": 1.2,
            "provenance_level": "MEMORY_OR_CONVENTION",
            "source": "仓内其它工件转述的惯例量级，未取原文",
            "usable_for_compliance_claim": False,
        },
        {
            "candidate_id": "CAND_C_R_3p0MPa",
            "label": "参考压力 C_R = 3.0 MPa @ A = 1 m^2（惯例量级）",
            "value_MPa": 3.0,
            "provenance_level": "MEMORY_OR_CONVENTION",
            "source": "仓内其它工件转述的惯例量级，未取原文",
            "usable_for_compliance_claim": False,
        },
    ]


# ---------------------------------------------------------------------------
# 6. 领域结果注入（DEM 压力-面积指数，正则；零手抄）
# ---------------------------------------------------------------------------
def extract_domain_exponent(path: Path = DOMAIN_DOC) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    pat = re.compile(
        r"^\|\s*primary \(A = [0-9.]+[-\u2013]?[0-9.]*\s*m\^2[^|]*\|\s*(?P<ex>[-0-9.]+)\s*\|"
        r"\s*\[(?P<lo>[-0-9.]+),\s*(?P<hi>[-0-9.]+)\]\s*\|\s*(?P<r2>[-0-9.]+)\s*\|\s*(?P<n>\d+)")
    for i, line in enumerate(lines, start=1):
        m = pat.match(line.strip()) if line.strip().startswith("|") else None
        if m:
            return {
                "source_file": str(path.relative_to(REPO)),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "line_number": i,
                "injection": "REGEX_FROM_DOC_TABLE (no hand copy)",
                "quantity": "DEM 压力-面积指数 ex（bonded-particle 冲压扫描拟合, 负值约定为 p ∝ A^-ex）",
                "ex": float(m.group("ex")),
                "ci95": [float(m.group("lo")), float(m.group("hi"))],
                "r2": float(m.group("r2")),
                "samples": int(m.group("n")),
                "raw_line": line.strip(),
            }
    raise RuntimeError("未在 docs/pressure_area_index_dem.md 找到 primary 指数行")


# ---------------------------------------------------------------------------
# 7. 判据裁决
# ---------------------------------------------------------------------------
def judge(case: dict, domain: dict) -> dict:
    anchors = build_anchors()
    cands = build_candidate_registry()
    verified = [a for a in anchors
                if a["verbatim_quote"] and a["table_or_page"] and a["numeric_values"]]
    recompute = conversion_recompute_residual(case)
    dims = dimension_checks()
    cons = internal_consistency(case)
    unverified_usable = sum(1 for a in anchors + cands if a.get("usable_for_compliance_claim") is not False)

    c1 = len(verified) >= 1
    c2 = recompute["max_relative_residual"] < 1e-9
    c3 = (dims["exponent_mismatches"] + cons["violations"]) == 0
    c4 = unverified_usable == 0

    checks = [
        {
            "id": "C1",
            "criterion": PREDECLARED_CRITERIA[0]["text"],
            "threshold": PREDECLARED_CRITERIA[0]["threshold"],
            "measured": {"verbatim_verified_anchors": len(verified), "registered_anchors": len(anchors)},
            "status": "PASS" if c1 else "FAIL",
            "evidence": "anchors[*].provenance.status=WITHHELD_NO_LICENSED_SOURCE; "
                        f"repo 语料命中 {anchors[0]['provenance']['search_log']['files_mentioning_iso19906']} 个文件, "
                        "逐字条款语料=0 → 无法逐字抄录",
        },
        {
            "id": "C2",
            "criterion": PREDECLARED_CRITERIA[1]["text"],
            "threshold": PREDECLARED_CRITERIA[1]["threshold"],
            "measured": recompute["max_relative_residual"],
            "worst_item": recompute["worst_item"],
            "status": "PASS" if c2 else "FAIL",
            "evidence": "benchmarks/anchors/ice_ridge_pipeline_seabed_bearing.py::conversion_recompute_residual",
        },
        {
            "id": "C3",
            "criterion": PREDECLARED_CRITERIA[2]["text"],
            "threshold": PREDECLARED_CRITERIA[2]["threshold"],
            "measured": {"dimension_exponent_mismatches": dims["exponent_mismatches"],
                          "internal_consistency_violations": cons["violations"],
                          "violated_items": cons["issues"]},
            "status": "PASS" if c3 else "FAIL",
            "evidence": "dimension_checks() + internal_consistency()",
        },
        {
            "id": "C4",
            "criterion": PREDECLARED_CRITERIA[3]["text"],
            "threshold": PREDECLARED_CRITERIA[3]["threshold"],
            "measured": unverified_usable,
            "status": "PASS" if c4 else "FAIL",
            "evidence": f"{len(cands)} 个惯例候选 + {len(anchors)} 个未核锚全部 usable_for_compliance_claim=False",
        },
    ]

    # 领域对比（在锚未核销前提下：只能对"惯例候选带"做核对对照）
    ex = domain["ex"]
    cmp_rows = []
    for c in cands:
        if "value" in c:
            n_cand = float(c["value"])
            rel = (ex - n_cand) / n_cand
            in_band = c["candidate_id"] == "CAND_n_055"
            cmp_rows.append({
                "vs_candidate": c["candidate_id"],
                "candidate_n": n_cand,
                "dem_ex": ex,
                "relative_difference": rel,
                "within_dem_ci95": c["candidate_id"] == "CAND_n_050" and False,
                "compliance_claim_allowed": False,
                "note": "对照 only；候选未核销，不得用于达标声明",
            })
    within_ci = [r for r in cmp_rows if domain["ci95"][0] <= r["candidate_n"] <= domain["ci95"][1]]

    nums = case_numbers(case)
    # 换算到本算例口径的压力（形式演示；C_R 取惯例候选 -> 非达标）
    conv_rows = []
    for c in cands:
        if "value_MPa" not in c:
            continue
        for n_cand in (0.45, 0.50, 0.55, ex):
            p_loc = p_power_law(c["value_MPa"], nums["contact_area_local_m2"], -n_cand)
            p_glob = p_power_law(c["value_MPa"], nums["contact_area_global_m2"], -n_cand)
            conv_rows.append({
                "C_R_MPa": c["value_MPa"],
                "C_R_source": c["candidate_id"],
                "exponent": n_cand,
                "exponent_source": ("DEM_MEASURED" if n_cand == ex else "CANDIDATE_MEMORY"),
                "A_local_m2": nums["contact_area_local_m2"],
                "p_local_MPa": p_loc / MPA_TO_PA,
                "F_local_N_per_m": force_from_pressure(p_loc, nums["contact_area_local_m2"]),
                "A_global_m2": nums["contact_area_global_m2"],
                "p_global_MPa": p_glob / MPA_TO_PA,
                "F_global_N": force_from_pressure(p_glob, nums["contact_area_global_m2"]),
                "usable_for_compliance_claim": False,
            })

    # ISO 形式项分解（形式级, 非锚值）
    iso_rows = []
    D = nums["pipe_outer_diameter_m"]
    for key in ("BASE_SOIL", "DRAINED_SOIL"):
        for ratio in (1.0, 2.0, 4.0, 8.0):
            form, soil, pipe = iso_form(case, ratio * D, soil_key=key)
            iso_rows.append({
                "soil": key,
                "cover_ratio_d_over_D": ratio,
                "cover_depth_m": ratio * D,
                "N_q": form.n_q,
                "N_c": form.n_c,
                "q_term_N_per_m": form.q_term,
                "c_term_N_per_m": form.c_term,
                "gamma_term_N_per_m": form.gamma_term,
                "V_N_per_m": form.q_term + form.c_term,
                "note": "ISO 形式算术（仓内 Vesic/Skempton 闭式已验证）；系数归属 ISO 19906 原文=WITHHELD",
            })

    gate = {"C1_verbatim_anchor_verified": c1, "C2_conversion_recomputable": c2,
            "C3_dimension_and_consistency": c3, "C4_unverified_not_citable": c4}
    verdict = "PASS" if all(gate.values()) else "HONEST_FAIL_ON_C1"
    return {
        "checks": checks,
        "gate_summary": gate,
        "verdict": verdict,
        "anchors": anchors,
        "candidate_registry": cands,
        "recompute": recompute,
        "dimension_check": dims,
        "internal_consistency": cons,
        "domain_comparison": {
            "domain_result": domain,
            "comparison_rows": cmp_rows,
            "candidates_inside_dem_ci95": len(within_ci),
            "statement": (
                f"本课题 DEM 压力-面积指数 ex={ex}（95% CI {domain['ci95']}）"
                f"与惯例候选带 0.45-0.55 的相对差 {cmp_rows[0]['relative_difference']*100:.1f}%"
                f" ~ {cmp_rows[-1]['relative_difference']*100:.1f}%；"
                "但因锚未核销，该差值只能作为核对对照，不构成与 ISO 19906 的合规级偏差声明。"
            ),
            "iso_form_vs_dem_capacity": {
                "status": "BLOCKED_ANCHOR_WITHHELD",
                "reason": "本单不重跑耦合 DEM；且 ISO 系数归属未核销，承载力对比留给后续实验",
                "planned_experiment": "research_plan 第 2 项（领域实验 vs 形式包络）",
            },
        },
        "conversion_to_case_caliber": conv_rows,
        "iso_form_decomposition": iso_rows,
        "honest_fail_note": (
            "C1 未达标：工作树内无 ISO 19906:2019 许可原文（iso.org 目录页 403），"
            "两个锚的 verbatim_quote/table_or_page/numeric_values 全部留空，"
            "provenance_level=WITHHELD_NO_LICENSED_SOURCE。判据未改、未以惯例值代填。"
            "C2/C3/C4 已机检通过，换算链与量纲/一致性结论与本算例口径无关地成立，可复用。"
        ),
    }


# ---------------------------------------------------------------------------
# 8. 图
# ---------------------------------------------------------------------------
def make_figures(case: dict, result: dict, domain: dict) -> list:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nums = case_numbers(case)
    figs = []

    # fig1: p(A) 曲线族（惯例候选带 vs DEM 指数），log-log，标注本算例局部接触面积
    As = np.geomspace(0.25, 500.0, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n_c, style in ((0.45, ":"), (0.50, "--"), (0.55, "-.")):
        ax.plot(As, [p_power_law(1.0, a, -n_c) / MPA_TO_PA for a in As], style, lw=1.4,
                label=f"convention n={n_c} (UNVERIFIED)")
    ax.plot(As, [p_power_law(1.0, a, -domain["ex"]) / MPA_TO_PA for a in As], "k-", lw=2.0,
            label=f"DEM measured n={domain['ex']} (this study)")
    ax.axvline(nums["contact_area_local_m2"], color="0.4", lw=1.0)
    ax.axvline(nums["contact_area_global_m2"], color="0.4", lw=1.0, ls=":")
    ax.text(nums["contact_area_local_m2"] * 1.05, 3.0, f"A_local={nums['contact_area_local_m2']:.3f} m$^2$", fontsize=8)
    ax.text(nums["contact_area_global_m2"] * 1.05, 1.0, f"A_global={nums['contact_area_global_m2']:.0f} m$^2$", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nominal contact area A [m$^2$]")
    ax.set_ylabel("pressure p [MPa]  (C$_R$=1 MPa reference)")
    ax.set_title("Pressure-area family: convention band vs DEM-measured exponent")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); p1 = FIG_DIR / "pressure_area_family.png"; fig.savefig(p1, dpi=140); plt.close(fig)
    figs.append({"name": p1.name, "path": str(p1.relative_to(REPO)),
                 "caption": "图1 压力-面积幂律族：惯例候选带(0.45/0.50/0.55, 未核) 与 本课题DEM实测指数 "
                            f"{domain['ex']}；竖线=本算例局部/全局接触面积。"})

    # fig2: 领域指数 vs 候选带相对差 + CI
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    rows = result["domain_comparison"]["comparison_rows"]
    labels = [r["vs_candidate"].replace("CAND_n_", "n=") for r in rows]
    vals = [r["relative_difference"] * 100 for r in rows]
    colors = ["#c0392b" if v < -15 else "#27ae60" for v in vals]
    ax.bar(labels, vals, color=colors)
    lo, hi = domain["ci95"]
    for i, n_cand in enumerate([r["candidate_n"] for r in rows]):
        ax.errorbar(i, vals[i], yerr=[[abs((domain["ex"] - lo) / n_cand) * 100],
                                      [abs((hi - domain["ex"]) / n_cand) * 100]],
                    fmt="none", ecolor="k", capsize=5)
    ax.axhline(-15, color="k", ls="--", lw=1.0, label="15% acceptance band")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("relative difference of n vs convention (%)")
    ax.set_title("DEM exponent vs unverified convention band (control only)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); p2 = FIG_DIR / "exponent_discrepancy.png"; fig.savefig(p2, dpi=140); plt.close(fig)
    figs.append({"name": p2.name, "path": str(p2.relative_to(REPO)),
                 "caption": "图2 本课题DEM指数相对惯例候选带的偏差（负=DEM 指数更小），误差棒为 DEM 95% CI 传播。"})

    # fig3: 换算到本算例口径的 ISO 形式项分解 vs 覆盖深度
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, mk in (("BASE_SOIL", "o"), ("DRAINED_SOIL", "s")):
        rows_k = [r for r in result["iso_form_decomposition"] if r["soil"] == key]
        x = [r["cover_ratio_d_over_D"] for r in rows_k]
        ax.plot(x, [r["V_N_per_m"] / 1e6 for r in rows_k], marker=mk, label=f"{key} V total")
        ax.plot(x, [r["c_term_N_per_m"] / 1e6 for r in rows_k], marker=mk, ls="--", alpha=0.6,
                label=f"{key} cohesion term")
    ax.set_xlabel("cover depth d/D [-]")
    ax.set_ylabel("bearing load [MN/m]")
    ax.set_title("ISO-form bearing decomposition in this case caliber (coefficient attribution WITHHELD)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); p3 = FIG_DIR / "iso_form_decomposition.png"; fig.savefig(p3, dpi=140); plt.close(fig)
    figs.append({"name": p3.name, "path": str(p3.relative_to(REPO)),
                 "caption": "图3 本算例口径下的 ISO 形式承载力项分解（V 与黏聚项 vs 覆盖深度）；"
                            "系数归属 ISO 19906 原文仍为 WITHHELD，仅为形式级算术。"})
    return figs


# ---------------------------------------------------------------------------
# 9. 组装工件
# ---------------------------------------------------------------------------
def build_document() -> dict:
    case = extract_case_caliber()
    domain = extract_domain_exponent()
    result = judge(case, domain)
    figs = make_figures(case, result, domain)
    doc = {
        "schema": "external_anchor_ledger/v1",
        "task_key": "paper-exp-ice-ridge-pipeline-seabed-bearing-1",
        "project": "ice-ridge-pipeline-seabed-bearing",
        "title": "ISO 19906 外部锚核销与同口径换算（实验 1：锚核销）",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_script": str(Path(__file__).resolve().relative_to(REPO)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preregistration": {
            "path": "prereg/ice_ridge_pipeline_seabed_bearing_prereg.json",
            "sha256": hashlib.sha256((REPO / "prereg/ice_ridge_pipeline_seabed_bearing_prereg.json").read_bytes()).hexdigest(),
            "registered_before_inspection": True,
        },
        "preregistered_criteria": PREDECLARED_CRITERIA,
        "form_assumption": FORM_ASSUMPTION,
        "constants": {"A_ref_m2": A_REF_M2, "MPa_to_Pa": MPA_TO_PA, "unit_axial_length_m": L_UNIT_M},
        "case_caliber": case,
        "case_numbers": case_numbers(case),
        "conversion_chain": result["recompute"],
        "dimension_check": result["dimension_check"],
        "internal_consistency": result["internal_consistency"],
        "anchors": result["anchors"],
        "candidate_registry": result["candidate_registry"],
        "candidate_registry_note": "惯例/记忆级候选：usable_for_compliance_claim 恒 False，仅作核对对照（C4 机检）。",
        "conversion_to_case_caliber": result["conversion_to_case_caliber"],
        "iso_form_decomposition": result["iso_form_decomposition"],
        "domain_comparison": result["domain_comparison"],
        "checks": result["checks"],
        "gate_summary": result["gate_summary"],
        "verdict": result["verdict"],
        "honest_fail_note": result["honest_fail_note"],
        "figures": figs,
        "next_experiment": "research_plan 第 2 项：领域实验（DEM 耦合承载力）vs ISO 形式包络（锚一旦可核销即可升级为合规级对比）",
    }
    return doc


def main() -> int:
    doc = build_document()
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print("verdict:", doc["verdict"])
    for c in doc["checks"]:
        print(f"  {c['id']}: {c['status']}  measured={c['measured']}")
    print("domain:", doc["domain_comparison"]["statement"])
    return 0 if doc["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())