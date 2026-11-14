# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律（m21033，kn=1e6 复验）

**算例**: 平底圆形料仓自由卸料，稳态质量流率 `W` 对 Beverloo (1961) 律。
**判定**: 见 §④（主算例 D5，容差 < 10 %，总账 `CASE-DEM-SILO-BEVERLOO`）。

> **状态（截至本 commit）**：6 算例矩阵已递交集群作业 **cj3141**（n330，stage_dir
> 直达节点）。loong64 节点单核速度显著低于本机（本机 D5 实测 ~4 min；节点 30+ min
> 仍未产出首个算例），本单调用预算耗尽时结果未回。**数值判定待 cj3141 终态后由
> 后续单补全**（fetch → `benchmarks/m21033_beverloo/analyse.py` → 更新 §③/§④ →
> 重发报告）。honest 声明：本 commit 不含任何流率数值，不预填任何误差。

## ① 问题与参考出处

平底圆形料仓底板圆心圆孔，颗粒在重力下自由流出；Beverloo 等人 1961 年给出经验律

```
W = C · rho_b · sqrt(g) · (D − k·d)^(5/2)
```

出处：Beverloo, W. A.; Leniger, H. A.; van de Velde, J., *The flow of granular
solids through orifices*, Chemical Engineering Science **15**(3-4), 260-269 (1961),
DOI 10.1016/0009-2509(61)80059-4；球状颗粒系数对 `C = 0.58`（0.55–0.60）、
`k = 1.4`（1.2–1.6），Eq. (4), p. 265。机器可读参考
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`（零手抄；provenance 标注
cited, not re-digitized——1961 原文不在仓库、环境无外网）。

**前序欠账**（本单要还的账）：

| 单 | 几何 | kn [N/m] | μ | 结果 |
|---|---|---|---|---|
| m20055 | 平底料仓 | 1e5 | 0.30 | D4 **+25.0 %** FAIL |
| m15259 | 锥形 hopper 30° | 1e6 | 0.15 | D/d=5 **−0.4 %** PASS |
| m20585 | 诊断 | — | — | 根因=接触刚度；建议 kn→1e6 |

## ② 方法与实现

复用既有 REAL 3-D DEM `tensordem.silo_discharge`（线性弹簧-阻尼法向 +
Cundall–Strack 切向 + 滚阻，`granular3d_core` 内核），协议沿用 m20055 的
settled+plug（松散晶格净距 0.30 r 堵孔压实 0.60 s → 开孔；稳态段=卸料流质量
分位 0.2–0.8，切 5 子窗口报 CV）。相对 m20055 只改两处（预注册
`prereg/m21033_beverloo.md`）：

1. **kn 1e5 → 1e6 N/m**：m20585 根因诊断——软接触在孔口变形放大有效孔径，
   流率系统性高估；hopper 内核 kn=1e6 已证 −0.4 %；
2. **μ 0.30 → 0.15**：玻璃珠低摩擦（Beverloo 实验条件，m15259 经验）。

矩阵 6 算例：D4/D5/D6（律校验）、D5_mu30（摩擦敏感性）、D5_kn1e5（kn 归因臂）、
D5_dthalf（dt/2 收敛）。集群作业 **cj3141**（stage_dir 直达节点，单核×6 串行）。

## ③ 参数表（含无量纲数）

| case | D/d | μ_p | kn [N/m] | n_bed | H/D | silo_D/D | ρ_b [kg/m³] | φ | 卸料占比 | W [kg/s] | W_ref [kg/s] | 误差 | C_rec | CV | <10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _待注入_ | | | | | | | | | | | | | | | |

无量纲完整性：`D/d`、`silo_D/D`、`H/D`、`rho_b/rho_p`（=φ）、`dt/T_contact`、
`μ_p`、`μ_wall`、`e`、Beverloo 前置因子 `C = W/(rho_b √g (D−kd)^{5/2})` 全部
由 `analyse.py` 从原始遥测程序化生成（`runs/m21033_beverloo/report_tables.md`）。

## ④ 对照表：ref vs got + 误差判定

_待注入（`runs/m21033_beverloo/summary.json`）_

## ⑤ 收敛性与参数敏感性

_待注入：dt/2 收敛、μ 敏感性、kn 归因臂_

## ⑥ 失败模式、工件与复现

**工件**

* 算例脚本 `benchmarks/m21033_beverloo/run_case.py`；分析 `analyse.py`；
  报告发布 `post_report_m21033.py`
* 逐算例原始数据 `runs/m21033_beverloo/*.json`（含 git sha、host、dt、窗口序列、
  卸料事件序列）；汇总 `summary.json` / `report_tables.md`
* 图 `docs/figures/m21033_beverloo_flow_law.png`、
  `docs/figures/m21033_beverloo_discharge_stream.png`
* 回归测试 `tests/test_m21033_beverloo_campaign.py`（10 项：ref 完整性、
  预注册一致性、公式、稳态窗口不变量、判定算术）

**复现命令**

```bash
PYTHONPATH=src python benchmarks/m21033_beverloo/run_case.py --case D5
PYTHONPATH=src python benchmarks/m21033_beverloo/analyse.py
PYTHONPATH=src python -m pytest tests/test_m21033_beverloo_campaign.py -q
```
