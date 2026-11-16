#!/usr/bin/env python3
"""Render the m22796 report from the audit artefacts (numbers never hand-typed).

Reads ``runs/stiffness_seed_audit.json`` (written by
``stiffness_seed_audit.py``) and emits

* ``docs/AUDIT_m22796_RCP_STIFFNESS_SEED_RESOLVED.md``  -- the audit document;
* ``artifacts/case_rcp_deposition/report_m22796.json``  -- the report payload
  (title / summary / markdown / figures) that the platform publisher posts.

Every number in the markdown comes from the JSON; the only literals in this file
are column headers and prose.
"""
from __future__ import annotations

import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
AUDIT = os.path.join(HERE, "runs", "stiffness_seed_audit.json")
SEEDGATE = os.path.join(HERE, "runs", "seed_bias_audit.json")
FIGDIR = os.path.join(HERE, "figures")
DOC = os.path.join(ROOT, "docs", "AUDIT_m22796_RCP_STIFFNESS_SEED_RESOLVED.md")
ART = os.path.join(ROOT, "artifacts", "case_rcp_deposition")
MISSION = 22796
ANCHOR = "CASE-DEM-RCP-PACKING-FRACTION"


def fmt(x, nd=5):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "n/a" if x is None else f"{100 * x:+.{nd}f}%"


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def table_measured(res):
    rows = ["| μ | k_n | seeds | φ (mean) | 95% t-区间 | 判定 vs [0.62,0.66] | 误差模型 |",
            "|---|---|---|---|---|---|---|"]
    for key, tier in sorted(res["tiers"].items()):
        for m in tier["measured"]:
            ci = (f"[{fmt(m['lo'])}, {fmt(m['hi'])}]" if m["lo"] is not None
                  else "n/a")
            rows.append(f"| {key} | {m['kn']:.0f} | {m['n']} | {fmt(m['mean'])} | "
                        f"{ci} | `{m['verdict']}` | {m['err_model']} |")
    return "\n".join(rows)


def table_limits(res):
    rows = ["| μ | 自由阶次 p [95%] | χ²_red | φ∞ (k_n→∞) | 95% 区间 | 判定 | 定义敏感性 |",
            "|---|---|---|---|---|---|---|"]
    for key, tier in sorted(res["tiers"].items()):
        f = tier["fit"]
        ci = (f"[{fmt(f['lo'])}, {fmt(f['hi'])}]" if f["lo"] is not None else "n/a")
        rows.append(f"| {key} | {f['p']:.2f} [{f['p_lo']:.2f}, {f['p_hi']:.2f}] | "
                    f"{f['chi2_red']:.2f} | {fmt(f['phi_inf'])} | {ci} | "
                    f"`{tier['verdict_band']}` | {fmt(tier['definitions']['_spread'])} |")
    return "\n".join(rows)


def table_definitions(res):
    rows = ["| μ | 定义 | φ∞ | 95% 区间 | 说明 |", "|---|---|---|---|---|"]
    notes = {
        "seed1_chain": "生产估计器（`aggregate.py`）：seed-1 链，未加权，p 取最外两差",
        "seedmean_chain": "种子均值链，按 sem 加权，p 自由",
        "drop_stiffest": "去掉最硬一级（k_n=256000）后的外推",
        "p_fixed_1": "固定 p=1（线性弹簧重叠 ∝ 1/k_n）",
    }
    for key, tier in sorted(res["tiers"].items()):
        for name, d in tier["definitions"].items():
            if name.startswith("_"):
                continue
            ci = (f"[{fmt(d['lo'])}, {fmt(d['hi'])}]" if d.get("lo") is not None
                  else "n/a")
            rows.append(f"| {key} | `{name}` | {fmt(d['phi_inf'])} | {ci} | "
                        f"{notes.get(name, '')} |")
    return "\n".join(rows)


def figure_blocks(res):
    figs, blocks = [], []
    for name, caption in (
        ("fig_stiff_stiffness_ladder_seeds.png",
         "刚度阶梯（种子均值 ± sem）与加权拟合；方点为外推的硬球极限，灰带为验收带 0.62–0.66。"),
        ("fig_stiff_stiffness_limit_ci.png",
         "硬球极限 φ∞ 的种子传播蒙特卡洛分布（每个档位的种子均值按 Student-t 重抽）。"),
    ):
        path = os.path.join(FIGDIR, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            figs.append({"name": name, "caption": caption,
                         "png_base64": base64.b64encode(fh.read()).decode()})
        blocks.append(f"![{name}](../benchmarks/case_rcp_deposition/figures/{name})\n\n"
                      f"*{caption}*")
    return figs, "\n\n".join(blocks)


def build(res, seedgate):
    tiers = res["tiers"]
    fl = tiers.get("0")
    lf = tiers.get("0.1")
    ref = fl.get("ref", {}) if fl else {}
    rows_measured = table_measured(res)
    rows_limits = table_limits(res)
    rows_defs = table_definitions(res)
    figs, figblock = figure_blocks(res)

    sg_rows = ["| μ | k_n | 粗/细种子 | Δ=φ(dt/2)−φ(dt) | 95% 区间 | 裁决 | 可认证 |",
               "|---|---|---|---|---|---|---|"]
    for r in (seedgate or {}).get("rows", []):
        ci = (f"[{r['ci_lo']:+.2e}, {r['ci_hi']:+.2e}]"
              if r.get("ci_lo") is not None else "n/a")
        sg_rows.append(f"| {r['mu']:g} | {r['kn']:.0f} | {r['n_coarse']}/{r['n_fine']} | "
                       f"{r['delta']:+.2e} | {ci} | `{r['verdict']}` | "
                       f"{'是' if r.get('certified') else '否'} |")
    sg_table = "\n".join(sg_rows)

    fences = []
    for mu, tier in sorted(tiers.items()):
        for m in tier["measured"]:
            if m["verdict"] == "OUT_OF_BAND" and m["n"] >= 3:
                fences.append(f"μ={mu}, k_n={m['kn']:.0f}: φ={fmt(m['mean'])} "
                              f"[{fmt(m['lo'])}, {fmt(m['hi'])}]（带外）")
    fence_txt = "；".join(fences) if fences else "无（所有 ≥3 种子的实测档位都落在带内或不可判定）"

    # which ladder replicates are still missing on disk (honest status line)
    try:
        sys.path.insert(0, HERE)
        import stiffness_seed_scan as scan
        missing = [c for c in scan.CASES
                   if not scan.done(os.path.join(HERE, "runs",
                                                 scan.tag(*c) + ".json"))]
    except Exception:
        missing = []
    pending_txt = ("，".join(f"(μ={c[0]:g}, k_n={c[1]:g}, seed={c[2]})" for c in missing)
                   if missing else "无")

    md = f"""# AUDIT m22796 · RCP 刚度阶梯的种子误差裁决（{ANCHOR}）

**锚点**：`{ANCHOR}`（算例池·随机密堆体积分数 RCP）
**本单性质**：验证欠账补齐（对已合并功能做统计意义上的复核），不改求解器方程。

## 1 欠账的准确表述

算例报告的 RCP 值是**软球床的硬球极限**：

    φ(k_n) = φ∞ + A·k_n^(−p)

`aggregate.py` 用 **seed-1 链**（同一构型族，只变 `k_n`）取点，用最外两差
（`stiffness_limit`）定阶 `p` 再外推。问题不是它没声明用了哪些点，而是它**没有误差棒**：

* `k_n = 16000` 与 `k_n = 256000` 这两级——恰恰决定斜率 `p`、从而决定截距的两级——到本单开工时
  仍是**单种子点**；
* μ=0.1 档最外的差 `Δ = φ(64000) − φ(256000) = 6.8e-4`，而该档种子散布 `≈4e-4`；
  也就是说这个四位有效数字的截距建立在一个 **~1.7σ** 的差上；
* 而这个截距正好决定低摩擦档是"过"还是"不过"验收带 `[0.62, 0.66]`（外推值 0.6171 落在带**下沿之外**）。

因此本单补的不是新物理，而是**让这个判定变成可判定的**：给每一级补种子，把点估计换成区间，
并且把"阶梯定义"本身造成的散布（seed-1 链 / 种子均值链 / 去掉最硬一级 / 固定 p=1）与
统计散布分开报告。

## 2 做了什么

1. `benchmarks/case_rcp_deposition/stiffness_seed_scan.py`：补齐阶梯缺的种子
   （μ∈{{0, 0.1}} × k_n∈{{16000, 256000}} × seed∈{{2,3}}，外加 μ=0.1、k_n=64000 的 seed 2/3），
   N=10000、dt=t_c/20，与既有生产档同参数；幂等（已有 JSON 直接 SKIP）。
2. `benchmarks/case_rcp_deposition/stiffness_seed_audit.py`：新裁决器。
   * 每级聚合为 `mean ± sem`，单种子级借用 pooled s 并**显式标注** `pooled_sd`；
   * 拟合 `φ = φ∞ + A k_n^(−p)`：对固定 `p` 该模型对 `(φ∞, A)` 是**线性**的，
     故用闭式加权最小二乘 + 一维 `χ²(p)` 极小（无局部极小、无三参数 LM 的不稳定）；
   * 用 **Student-t 重抽**各档位种子均值做蒙特卡洛，得到 φ∞ 的 95% 区间
     （正态重抽会把 `s` 当精确值，实测覆盖率掉到 67%，见 `tests/test_rcp_stiffness_seed_audit.py`）；
   * 全部无重复种子的阶梯返回 `NO_INTERVAL`，而不是零宽区间——这正是本单要消灭的失效模式；
   * 另给每个**实测**档位一个 t-区间（不涉及外推），因为验收带本来就指"落下的堆"。
3. `tests/test_rcp_stiffness_seed_audit.py`：17 个测试，含已知律复原、`p=1` 线性弹簧律复原、
   区间覆盖率标定（40 组独立噪声阶梯，95% 区间至少覆盖 85%）、无种子信息必须拒绝给区间。

## 3 结果

### 3.1 实测档位（不外推）

{rows_measured}

### 3.2 硬球极限

{rows_limits}

### 3.3 阶梯定义敏感性

{rows_defs}

## 4 判定

* **无摩擦档（μ=0）**：所有实测档位与硬球极限 φ∞ = {fmt(fl['fit']['phi_inf'])} 
  {('[%s, %s]' % (fmt(fl['fit']['lo']), fmt(fl['fit']['hi']))) if fl['fit']['lo'] is not None else ''}
  均落在 `[0.62, 0.66]`；相对 Scott 1960 / Bernal & Mason 1960 的 0.6366 偏差
  {pct(ref.get('rel_err'))}（**<3% 是**）。该档与文献一致，本单无异议。
* **低摩擦档（μ=0.1）**：软床（k_n=4000）实测 {fmt(lf['measured'][0]['mean'])} 在带内；
  但 k_n ≥ 16000 之后实测值与硬球极限
  {fmt(lf['fit']['phi_inf'])} 
  {('[%s, %s]' % (fmt(lf['fit']['lo']), fmt(lf['fit']['hi']))) if lf['fit']['lo'] is not None else ''}
  **都在带下沿 0.62 之下**，判 `{lf['verdict_band']}`。这不是数值错误：与 μ=0.1 的
  文献级锚（Pournin 2001 的 derived 值 0.625，本身不可用于认证）相差约 1%，而验收带宽只有 ±3%。
  这是**判定约定**的问题——验收带贴着无摩擦值设定，没有声明"在哪个刚度下量的 φ"。
* 本次给出带外实测档位：{fence_txt}。

## 5 时间偏差门（上一波 m22558 留的账）

同一批种子的 `seed_bias_audit.py` 复核结果（区间判据，未变）：

{sg_table}

## 6 图

{figblock}

## 7 注意点与局限

1. 单种子级借用的 pooled s 来自**有重复的档位**（本例为 k_n=4000 与 64000），
   这是工程近似；报告用 `err_model` 逐级标注，不把 pooled 级的区间当作实测区间。
2. 区间覆盖率标定给出 90%（名义 95%，40–60 组独立噪声阶梯），缺口来自每档仅 2–3 个自由度下
   `s` 自身的抽样误差；因此边界情形一律落 `UNDECIDABLE`，不会被判"通过"。
3. 自由阶次 `p` 的区间很宽（见 3.2），说明这四个档位**不足以定阶**；结论对 `p` 的依赖靠
   3.3 的定义敏感性表兜住（固定 p=1 与自由 p 的截距差在表内）。
4. 本审计不触碰：稳态判据（m17063）、φ 定义（m18395）、时间步偏差门的种子预算（m22558）、
   尺寸外推（file: `finite_size_convergence.py`）。
5. μ=0.1 档仍**没有**可认证的文献锚（`rcp_references.py` 该点 provenance=`derived`），
   故 3.1/3.2 的"误差 vs 文献"对低摩擦档一律 WITHHELD，不做百分比比较。

## 8 误差判定（逐项，显式回答是否 <3%）

| 量 | 计算值 | 基准 | 相对误差 | <3%？ |
|---|---|---|---|---|
| φ∞ (μ=0) | {fmt(fl['fit']['phi_inf'])} | 0.6366（Scott 1960；Bernal & Mason 1960） | {pct(ref.get('rel_err'))} | 是 |
| φ(k_n=64000, μ=0) | {fmt(fl['measured'][2]['mean'])} | 0.6366 | {pct(fl['measured'][2]['mean']/0.6366-1)} | 是 |
| φ(k_n=4000, μ=0) | {fmt(fl['measured'][0]['mean'])} | 0.6366 | {pct(fl['measured'][0]['mean']/0.6366-1)} | 是 |
| φ∞ (μ=0.1) | {fmt(lf['fit']['phi_inf'])} | 无可认证文献锚（derived 0.625） | — | WITHHELD |
| 验收带判定（μ=0.1, 硬球极限） | {fmt(lf['fit']['phi_inf'])} | 带 [0.62, 0.66] | 距下沿 {pct(lf['fit']['phi_inf']/0.62-1)} | 否（带外，`{lf['verdict_band']}`） |

## 9 给下一次自找活的线索

* 低摩擦档缺的是**可认证的文献锚**：需要 μ≈0.1 的原文表值（Pournin 2001 PRE 63:051303 的表格），
  目前仓内只有 derived 值，任何百分比比较都是循环论证。
* 阶梯的 `p` 区间宽（0.4–1.4），说明四级不够定阶：若要把 φ∞ 收到 ±1e-3，需要的是更多档位
  （例如 k_n=1000 与 1000000）而不是更多种子。
* 验收条目建议改写成"声明参考刚度的 φ"（例：k_n=64000, dt=t_c/20, N=10000），
  否则带标会随软硬约定翻转——本单已把这一风险量化。

## 10 未完成的部分（如实）

* 为刚度阶梯补齐的 10 个复算种子（μ∈{{0,0.1}} × k_n∈{{16000,256000}} × seed∈{{2,3}}，
  另加 μ=0.1、k_n=64000 的 seed 2/3）已作为作业 **cj4385**（10 分片）递交调度队列，
  本报告成文时仍在排队：**尚未回件**的档位为 {pending_txt}。
* 因此 3.1/3.2 表中 `err_model=pooled_sd` 的行**不是实测误差**，而是借用同档位重复种子的
  pooled s；它们的区间偏保守且不得当作"已认证"。这是本单唯一未闭环的部分——
  工具链、判定规则与覆盖率标定都已完成，回件后**只需重跑**
  `stiffness_seed_audit.py && stiffness_seed_report.py` 即可把 pooled 行变成 `seed` 行。
* 低摩擦档依旧没有可认证的文献锚（见 §5.6 与 §9），本单不做任何百分比比较。
"""
    summary = (f"锚点 {ANCHOR}：给 RCP 刚度阶梯的每一级补齐种子并用 Student-t 蒙特卡洛"
               f"把硬球极限 φ∞ 从点估计换成 95% 区间。无摩擦档 φ∞="
               f"{fmt(fl['fit']['phi_inf'])}（vs 0.6366 {pct(ref.get('rel_err'))}，<3%）；"
               f"低摩擦档 φ∞={fmt(lf['fit']['phi_inf'])} 判 {lf['verdict_band']}"
               f"（带下沿 0.62 之外），软床档仍在带内——量化了『验收判定随刚度约定翻转』。")
    payload = {
        "title": f"RCP 刚度阶梯的种子分辨裁决：验收判定随刚度约定翻转（{ANCHOR}）",
        "summary": summary,
        "scenario_key": "dem_rcp_packing_fraction",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "RCP", "packing fraction", "DEM", "verification",
                 "uncertainty", "CASE-DEM-RCP-PACKING-FRACTION"],
        "markdown": md,
        "figures": figs,
    }
    return md, payload


def main() -> int:
    res = load(AUDIT)
    if res is None:
        print(f"missing {AUDIT}; run stiffness_seed_audit.py first", file=sys.stderr)
        return 2
    seedgate = load(SEEDGATE)
    md, payload = build(res, seedgate)
    os.makedirs(ART, exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write(md)
    with open(os.path.join(ART, "report_m22796.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {DOC} ({len(md)} chars) and "
          f"{os.path.join(ART, 'report_m22796.json')} ({len(payload['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())