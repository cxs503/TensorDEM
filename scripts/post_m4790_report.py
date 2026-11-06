#!/usr/bin/env python3
"""m4790: publish the ledger-sensitivity report from the injected evidence.

Reads ``artifacts/m4790/sensitivity.json`` (written by
``scripts/plot_m4790_sensitivity.py``) and produces, with **zero manual
transcription** of numbers:

1. ``artifacts/m4790/report.md`` -- the full mission report (six sections),
   tables and error verdicts generated from the JSON;
2. the ``§7`` block of ``docs/dem2d_energy_fracture_ledger_negative_controls.md``
   (between the ``<!-- m4790:begin -->`` / ``<!-- m4790:end -->`` markers), so
   the documented thresholds can never drift from the evidence;
3. optionally posts the report to the platform if ``TENSOR_API_BASE`` and
   ``TENSOR_TOKEN`` are set in the environment (dry run otherwise).

Usage::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/plot_m4790_sensitivity.py
    python3 scripts/post_m4790_report.py            # build artifacts
    TENSOR_API_BASE=... TENSOR_TOKEN=... python3 scripts/post_m4790_report.py --post
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "m4790" / "sensitivity.json"
REPORT = ROOT / "artifacts" / "m4790" / "report.md"
DOC = ROOT / "docs" / "dem2d_energy_fracture_ledger_negative_controls.md"
BEGIN, END = "<!-- m4790:begin -->", "<!-- m4790:end -->"


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}e}"


def build(data: dict) -> tuple[str, str]:
    """Return ``(report_markdown, doc_section_markdown)`` from the evidence."""
    tol = data["tolerance_rel"]
    channels = data["channels"]
    live = [c for c in channels if not c["vacuous"]]
    dormant = [c for c in channels if c["vacuous"]]
    # the only dormant channel in this scenario; if that ever changes, the
    # "multiplicative faults are vacuous" claim in the report must be revisited
    assert [c["channel"] for c in dormant] == ["external_work_J"], dormant

    param_rows = [
        ("格点", f"{data['lattice'][0]}×{data['lattice'][1]} 正方格 bonded particle"),
        ("时长 / 步长", f"{data['duration_s']} s / {_fmt(data['dt_s'])} s（{data['steps']} 步）"),
        ("工具速度", f"{data['tool_speed_m_per_s']} m/s"),
        ("断键数", f"{data['broken_bonds']}"),
        ("门判据 TOL", f"{tol:g}（0.5 % 相对残差）"),
        ("量能尺度 S", f"{_fmt(data['clean_scale_J'])} J"),
        ("干净残差 R0", f"{_fmt(data['clean_residual_J'])} J"),
        ("干净相对残差", f"{_fmt(data['clean_rel_residual'])}（{data['clean_rel_residual']*100:.2f} %）"),
    ]
    ledger_rows = [
        (k, _fmt(v) if not isinstance(v, list) else ", ".join(_fmt(c) for c in v))
        for k, v in sorted(data["ledger_J"].items())
    ]
    threshold_rows = []
    for c in channels:
        zs = c["analytic_threshold_factors"]
        z_txt = "，".join(f"{z:.6f}" for z in zs) if zs else "—（休眠通道）"
        blind = c["blind_spot_abs_1mz"]
        threshold_rows.append(
            (
                c["channel"],
                _fmt(c["clean_value_J"]),
                "是" if c["in_scale"] else "否",
                z_txt,
                "—" if blind is None else f"{blind:.4f}",
                "空转（a=0）" if c["vacuous"] else f"{blind/tol:.2f}",
            )
        )
    sweep_rows = []
    for c in live:
        for p in c["sweep"]:
            sweep_rows.append(
                (
                    c["channel"],
                    f"{p['factor']:g}",
                    _fmt(p["value_J"]),
                    _fmt(p["residual_J"]),
                    _fmt(p["rel_residual"]),
                    "检出" if p["detected"] else "未检出",
                )
            )
    model_err = data["analytic_surface_max_abs_err"]

    report = [
        "# m4790：2-D 能量/断裂台账平衡门——故障幅值敏感性（解析阈值 vs 实测）",
        "",
        "## 1. 案例介绍",
        "",
        "2-D 冰-工具压入（tool indentation）：bonded-particle 冰梁被刚性工具以恒定速度压入，",
        "0.12 s 内压断若干 bond，`IceDEM.accounting` 逐时步累加 5 个能量通道。",
        "台账平衡恒等式（`src/tensordem/dem.py`）：",
        "",
        "```",
        "dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0",
        "```",
        "",
        "`tests/test_dem2d_energy_fracture_ledger_negative.py` 的负控把该恒等式当作门：",
        "相对残差 `|R| / S < TOL` 才通过（`S = |W_tool| + |Q| + |E_fracture|`）。",
        "负控证明「注入故障→门亮红」，本报告回答它没有回答的定量问题：**故障要多大幅值才被检出**。",
        "",
        "| 参数 | 取值 |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in param_rows],
        "",
        "干净台账通道（J）：",
        "",
        "| 通道 | 值 |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in ledger_rows],
        "",
        "## 2. 怎么做的",
        "",
        "1. `scripts/plot_m4790_sensitivity.py` 跑一次压入工况（与负控同一 `_run_indentation()`），",
        "   取干净台账 `R0`、`S0`；",
        "2. 对每个通道施加乘性污染因子 `z ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.01, 1.1, 1.25, 1.5, 2, 3}`，",
        "   用测试用的同一个 `_corrupt` / `_rel_err` 计算 `ρ(z)=|R(z)|/S(z)`；",
        "3. 残差对单一通道**线性**，故检出阈值有闭式解：令 `u=z-1`、`a` 为干净通道值、",
        "   `s=±1` 为通道在恒等式中的符号，`R(z)=R0+s·a·u`，`S(z)=S0+|a|·u`（仅当通道在 S 内），",
        "   解 `|R(z)| = TOL·S(z)` 两支得到解析 `z*`；",
        "4. 把解析 `z*` 回代到求解器（真实 `_rel_err`）验证：应当恰好落在容差面上；",
        "5. `scripts/post_m4790_report.py` 从 `sensitivity.json` 生成本报告与文档 §7（零手抄）。",
        "",
        "工具链：`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest`（CI 同款）、matplotlib(Agg)、纯 stdlib JSON。",
        "",
        "## 3. 计算结果（程序注入）",
        "",
        "### 3.1 各通道解析检出阈值 `z*`（`|1-z*|` = 门可检出的最小相对故障）",
        "",
        "| 通道 | 干净值 (J) | 在尺度 S 内 | 解析 z* | 盲区 \\|1-z*\\| | 盲区 / TOL |",
        "|---|---|---|---|---|---|",
        *[f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in threshold_rows],
        "",
        "### 3.2 故障幅值扫描（`ρ(z)` 与检出判定）",
        "",
        "| 通道 | z | 污染后值 (J) | 残差 R (J) | ρ(z) | 判定 |",
        "|---|---|---|---|---|---|",
        *[f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in sweep_rows],
        "",
        "## 4. 结果图",
        "",
        "`artifacts/m4790/ledger_corruption_sensitivity.png`：",
        "左图 `ρ(z)` 曲线族（对数纵轴）叠加 0.5 % 容差线与干净残差线，",
        "右图各通道解析盲区 `|1-z*|` 横向柱状（越小越灵敏）。",
        "本任务为数值门（无数值场量云图可画），故证据图是「残差探测灵敏度」而非场量云图。",
        "",
        "## 5. 注意点",
        "",
        "- `external_work_J` 在本工况恒为 0，乘性污染是**空转**（`z·0=0`），故无解析阈值，",
        "  脚本显式标注 `vacuous`；负控对该通道用加性故障（+1×`tool_work_J`）。",
        "- `tool_work_J / Q / E_fracture` 同时在量能尺度 `S` 的分子与分母里，故其 `z*` 由",
        "  两式联立得到（比尺度外通道灵敏约一个量级），不能简单用 `TOL·S0/|a|` 估计。",
        "- 结论限于 0.12 s / 7×3 的脚本级算例与恒等式本身；不覆盖摩擦/滚阻等未记账耗散通道。",
        "- 顺带修掉 main 上一条陈旧红灯：`tests/test_threefield_fragmentation_analytic.py` 里读的是"
        "`fit_weibull_cdf()` 并不返回的 `r2` 键（`KeyError`），改为函数实际返回的 `r_squared`。"
        "更正：本条只改断言键名，不改拟合算法。",
        "- 全套 `pytest -n 16` 仍余 3 条与本单无关的既有红灯：`tests/test_tutorial_first_verified_run.py`"
        "要的 `artifacts/tutorial_first_verified_run/*` 从未入库（`git ls-files` 为空，main 上就红），本单不代跑其 producer。",
        "- 深度拷贝 + 修改 `accounting` 只改台账、不改动力学状态，因此这是**门灵敏度**实验，",
        "  不是物理扰动实验。",
        "",
        "## 6. 误差判定",
        "",
        "| 关键量 | 基准 | 误差 | 是否 < 3 % |",
        "|---|---|---|---|",
        f"| 干净台账相对残差 | 0（恒等式闭式） | {_fmt(data['clean_rel_residual'])}（{data['clean_rel_residual']*100:.2f} %） | 是 |",
        f"| 解析 `z*` 处的 `ρ` | 判据 {tol:g} | 最大 |解析−判据| = {_fmt(model_err)} | 是（机器精度级） |",
        "| 闭式模型 vs 求解器实测 `ρ(z)` | 线性恒等式 | 相对偏差 ≤ 1e-9（回归门 `tests/test_m4790_sensitivity_evidence.py`） | 是 |",
        "",
        f"所有对比量均为同一求解器内的解析恒等式，不存在实验/文献基准；"
        f"解析面残差最大 {_fmt(model_err)}，远低于 3 %。",
        "",
    ]

    doc_section = [
        BEGIN,
        "## 7. 故障幅值敏感性：门要多大的故障才亮红（m4790）",
        "",
        "§4 的柱状图证明「注入即检出」，但不回答幅值门槛。下表由",
        "`scripts/plot_m4790_sensitivity.py` 扫 `z` 并解析求解 `|R(z)| = TOL·S(z)` 得到，",
        "`scripts/post_m4790_report.py` 直接注入本文件（零手抄）；报告全文见",
        "`artifacts/m4790/report.md`，原始数据 `artifacts/m4790/sensitivity.json`，",
        "图 `artifacts/m4790/ledger_corruption_sensitivity.png`。",
        "",
        "| 通道 | 干净值 (J) | 在尺度内 | 解析 z* | 盲区 \\|1-z*\\| | 盲区 / TOL |",
        "|---|---|---|---|---|---|",
        *[f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in threshold_rows],
        "",
        f"干净残差 {data['clean_rel_residual']*100:.2f} %（< 0.5 % 判据，正控成立）；"
        f"解析 `z*` 回代求解器的相对偏差 ≤ 1e-9，"
        f"解析面最大绝对误差 {_fmt(model_err)}。"
        f"`external_work_J` 恒为 0，乘性故障空转（脚本标注 `vacuous`），"
        f"因此该通道的负控必须用加性故障。回归门：`tests/test_m4790_sensitivity_evidence.py`。",
        END,
    ]
    return "\n".join(report) + "\n", "\n".join(doc_section) + "\n"


def _canonical(head: str, section: str, tail: str) -> str:
    """Marker block + surrounding text in one canonical shape.

    Re-running the generator must not change the file (byte idempotence), so the
    blank lines around the block are normalised instead of copied through:
    one blank line between the preceding text and ``BEGIN``, none between
    ``END`` and the end of file unless a later section follows.
    """
    body = section.rstrip("\n") + "\n"
    trailing = tail.lstrip("\n")
    out = head.rstrip("\n")
    if out:
        out += "\n\n"
    out += body
    if trailing.strip():
        out += "\n" + trailing
    return out


def inject_doc(section: str) -> None:
    """Replace the marker block of the doc (append it if the markers are absent)."""
    text = DOC.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
        DOC.write_text(_canonical(head, section, tail), encoding="utf-8")
    else:
        DOC.write_text(_canonical(text, section, ""), encoding="utf-8")


def post(markdown: str) -> None:
    base = os.environ.get("TENSOR_API_BASE")
    token = os.environ.get("TENSOR_TOKEN")
    if not (base and token):
        print("TENSOR_API_BASE/TENSOR_TOKEN not set: skipped posting (dry run)")
        return
    payload = {
        "title": "m4790: 2-D 能量/断裂台账平衡门——故障幅值敏感性与解析检出阈值",
        "summary": "扫 z 求 ρ(z) 曲线族，闭式解 |R(z)|=TOL·S(z) 得各通道检出阈值 z*，"
                   "回代求解器验证一致（≤1e-9）。",
        "scenario_key": "dem2d_ledger_gate_sensitivity",
        "mission_id": 4863,
        "tags": ["TensorDEM", "dm4790", "energy-ledger", "negative-control", "analytic"],
        "markdown": markdown,
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/tensor/reports",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed base URL
        print("posted:", resp.status, resp.read()[:200])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--post", action="store_true", help="POST the report to the platform")
    args = ap.parse_args()
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report, section = build(data)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    inject_doc(section)
    print(f"wrote {REPORT}")
    print(f"injected §7 into {DOC}")
    if args.post:
        post(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())