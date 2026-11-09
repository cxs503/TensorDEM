# 冰脊龙骨触底 / 海底管道承载力：外部锚（ISO 19906）核销记录

课题 `ice-ridge-pipeline-seabed-bearing`，research_plan 第 1 项实验：
**外部锚可核销性 + 同口径换算 + 一致性机检**。

- 预登记（冻结判据）：`prereg/ice_ridge_pipeline_seabed_bearing_prereg.json`
- 工件（判据逐条 + 实测值 + 来源/脚本路径）：`benchmarks/anchors/ice-ridge-pipeline-seabed-bearing.json`
- 生成脚本（判据数字全部由脚本注入，零手抄）：
  `python3 benchmarks/anchors/ice_ridge_pipeline_seabed_bearing.py`
- 机检测试：`python3 -m pytest tests/test_ridge_seabed_external_anchor.py -q`
- 图：`benchmarks/anchors/figs/{pressure_area_family,exponent_discrepancy,iso_form_decomposition}.png`

## 1. 判据与裁决（先于动手，阈值未改）

| 判据 | 阈值 | 实测 | 裁决 |
|---|---|---|---|
| C1 引句 + 表号/页码 + 数值逐字录入（值=来源原值） | ≥ 1 锚 | 逐字核销锚 **0 / 2** | **FAIL** |
| C2 同口径换算可复算 | 残差 < 1e-9 | 最大相对残差 **1.28e-14** | PASS |
| C3 量纲齐次 + 内部一致性 | 违约数 == 0 | 量纲不匹配 **0**，一致性违约 **0** | PASS |
| C4 未核候选值不得直接引用 | 可引用未核值数 == 0 | **0**（5 个惯例候选全标 `usable_for_compliance_claim=False`） | PASS |

**总结论：`HONEST_FAIL_ON_C1`。** 两个外部锚（ISO 19906:2019 Annex A 压力-面积关系；
Clause 14/15 设计全球冰力）都无法在本主体可达范围内取得许可原文：

- 工作树内 `ISO 19906` / `19906:2019` / `Annex A` 的命中全部是**转述或定性引用**
  （`docs/pressure_area_index_dem.md`、`docs/IC_CONTACT_*`、`src/tensordem/ridge_iso19906.py`
  等），无一条是可逐字抄录的条款文本（`verbatim_clause_corpus_found = false`）；
- `https://www.iso.org/standard/65712.html` 目录页在本沙箱返回 **HTTP 403**，无全文。

按溯源纪律，`verbatim_quote` / `table_or_page` / `numeric_values` 全部留空，**不以惯例值代填**；
两个锚登记 `provenance_level = WITHHELD_NO_LICENSED_SOURCE`，`usable_for_compliance_claim = false`。

## 2. 已经机检成立的部分（与锚是否核销无关，可复用）

1. **本算例口径零手抄**：龙骨深度/宽度、管径/壁厚、两类土参数由 AST 从
   `scripts/run_ridge_iso_grounding.py` 的模块级常量注入（带源文件 sha256 与行号）。
   本算例局部接触面积 `A = D × 1 m`，全局接触面积 `A = w_keel × h_keel`，两者均由脚本算出。
2. **换算链两条独立路径**：路径 A 直接幂律 `p = C_R (A/A_ref)^n`；
   路径 B 在对数网格上 `polyfit` 反演 `(n, C_R)` 再评估；再加单位往返、力反演、
   力的对数斜率恒等 `d lnF/d lnA = 1 + n`、ISO 形式项和 = 总值。
   最大相对残差 **1.28e-14**（C2 判据 < 1e-9）。
3. **量纲齐次**：压力 `kg·m⁻¹·s⁻²`、平面应变面积 `[m²]/[m]=[m]`、
   `Pa·m = N/m`、`γ'·h·D = N/m`、`c·D = N/m` 全部匹配（不匹配 0 项）。
4. **内部一致性**：`p(A_ref) = C_R`、`n < 0` 时单调下降、`c` 项线性于黏聚力、
   面积口径与几何自洽，违约 0 项。

## 3. 领域对比（在锚未核销前提下只能作核对对照）

本课题领域实验结果（bonded-particle DEM 冲压扫描，`docs/pressure_area_index_dem.md` §3.1
primary 行，正则注入 + sha256）给出压力-面积指数 **ex = 0.307（95 % CI [0.253, 0.358]）**。

与惯例候选带（0.45 / 0.50 / 0.55，全部 `MEMORY_OR_CONVENTION`）的相对差
**−31.8 % / −38.6 % / −44.2 %**，均超出 15 % 接受带，且 DEM 的 95 % CI 不覆盖该带。

**这不能写成「DEM 与 ISO 19906 相差 40 %」**：候选带本身未核销，只是惯例量级；
只有在取得许可原文并逐字录入（C1 转 PASS）后，同一脚本产出的差值才具备合规级含义。

耦合 DEM 承载力 vs ISO 形式的对比（本单不做）：`BLOCKED_ANCHOR_WITHHELD`，
留给 research_plan 第 2 项实验。

## 4. 注意点 / 局限

- **C1 是本课题的硬门**：没有许可原文，任何「对标 ISO 19906 误差 x %」的表述都是越线的。
  取得原文后只需把 `verbatim_quote` / `table_or_page` / `numeric_values` 填入
  `build_anchors()`，脚本与判据无需任何改动即可重判——这是本工件刻意保留的升级通道。
- 压力-面积幂律 `p(A) = C_R (A/A_ref)^n` 的**函数形式本身是形式假设**（`ASSERTION_FORMAL`），
  不是逐字来源；工件中所有 p(A) 数值因此都标 `usable_for_compliance_claim = false`。
- 二维平面应变、单位轴向长度：可迁移的是指数，绝对压力不可直接外推三维。
- ISO 形式承载力项分解用的是仓内已验证的 Vesic/Skempton 闭式（形式级算术 REAL），
  其系数对 ISO 19906 的**归属**同样 WITHHELD。

## 5. 结论一句话

锚核销在 C1 上诚实失败（无许可原文，未以记忆值代填）；换算链、量纲齐次与一致性机检全部通过
（残差 1.28e-14、违约 0）；领域指数与惯例候选带的差值（−32 % ~ −44 %）登记为**仅作对照**，
升级为合规级声明的前置条件是把 ISO 19906 原文逐字录入工件。