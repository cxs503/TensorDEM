# ice-polygon-floe-field — 外部锚核销（W18 paper-exp 实验 #1）

**Repo**: TensorDEM · **branch/commit**: `agent/m7514` / `88131d287184` · **task_key**: `paper-exp-ice-polygon-floe-field-1` · **generated**: 2026-11-08T15:55:46+00:00 (UTC)

**Artifact**: `benchmarks/anchors/ice-polygon-floe-field.json` — 全部数字由 `benchmarks/anchors/build_ice_polygon_floe_field.py` 注入，无手抄

**Verdict**: `FAIL` — 外部锚未核销（C1 FAIL，状态 WITHHELD）→ 本课题与规范的差额无法计算；算子层/量纲/一致性已机检通过（C2 operator, C3, C4），按诚实纪律记 honest FAIL，不粉饰、不改判据

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

本算例＝IN32-N01 的无序多边形碎冰场（航道内碎冰盖）：floe 为 Laguerre prescribed-area 镶嵌生成的多边形，接触为凸多边形 SAT 接触（带力臂与力矩）。外部锚要对标 **ISO 19906:2019** 的 p(A) 面积-压力关系（Annex A）与全球冰作用（Clause 14/15）。

| 参数 | 取值 / 来源 |
|---|---|
| floe 数 | N = 400 |
| FSD 族 | BimodalFSD(), DeltaFSD(5.0), LogNormalFSD(0.5), PowerLawFSD(2.5) |
| 覆盖率 φ | 0.60, 0.85, 0.90 |
| 冰厚 h | 0.500003 m（=0.1×median r_eq，代码给出） |
| 场盒 | 338.13 × 211.33 m |
| 接触斑口径 | A_i = κ·2·r_eq,i·h，κ = 0.5 |
| seed | [0, 1, 2, 3] |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **来源通道探测**（每通道一次，判完即止）：tcp 443、http GET 正本页、文件系统清单扫描 1214 个文本文件，primary text 命中 0。

2. **锚位登记表**（`ANCHOR_SLOTS`：Annex A p(A) / Clause 14-15 全球冰力）：只能由 `--anchor-file` 注入的逐字抄录填值；VERIFIED 必须同时具备 quote_verbatim + table_or_page + value，半填文件直接 raise（不得静默升级）。

3. **算子层换算链**：p(A)=p₀(A/A_ref)^(-n) → F(A)=p(A)·A → 场聚合 F_field=Σᵢp(A_i)A_i = p₀·A_ref^n·K(n)，K(n)=ΣᵢA_i^(1-n)。绝对层缺 p₀、A_ref 即 WITHHELD；相对层 p₀、A_ref 约掉，故锚未核销也能先做可复算的同口径换算。

4. **机检**：量纲齐次（分数指数代数，含 A_ref 归一化陷阱行）、局部-平均压力恒等式（Gauss–Legendre，4 个面积点）、log-log 指数反演、尺度不变性、聚合双路一致、口径退化（同族 FSD 不同 φ 的接触斑集合）。

5. **复算**：重载工件内 (A, p) 采样点反演 n，与工件内 n 及 F=p·A 比对。

6. 工具链：numpy + 本仓 `tensordem.polygon_floe_field` / `tensordem.fsd_packing`；matplotlib 出图；pytest 30 例。


## 3. 计算结果
### 3.1 判据逐条

| 判据 | 状态 | 实测（节选） | 差距 |
|---|---|---|---|
| C1 引句 + 表号/页码 + 数值逐字录入工件（值=来源原值，禁… | `FAIL` | {"verified_anchors": 0, "required": 2, "quotes_present": 0, "slots": [{"id": "iso19906-2019-AnnexA-pA", "transcription_status": "WITHHELD"}, {"id": "i | 0/2 锚位核销：ISO 19906:2019 正本不可得 → 引句/表号/页码/原值全空，状态 WITHHELD；不以记忆或二手转述填数 |
| C2 同口径换算脚本落盘且可复算（复算残差 <1e-9），并换算到… | `PASS(operator) / WITHHELD(absolute)` | {"recompute_residual_max_rel": 0.0, "threshold": 1e-09, "operator_layer": "PASS", "absolute_layer": "WITHHELD", "absolute_loads_N": null} | 算子层已复算达标（残差 <1e-9，脚本落盘可复跑）；绝对层需要 p0 与 A_ref （Annex A 原文），故『换到本算例口径』的牛顿数不可声明 |
| C3 量纲齐次与内部一致性机检通过… | `PASS` | {"dimensional_pass": true, "dimensional_rows": 6, "consistency_pass": true, "consistency_rows": 7} | — |
| C4 未核候选登记值不得直接引用（只能作为核对对照）… | `PASS` | {"violations": [], "verified_slots": []} | — |

### 3.2 算子层（单位归一 p₀=1 Pa、A_ref=1 m²；只复算算子，不是标准原值）

| A [m²] | p/p₀ | F/(p₀·A_ref) |
|---|---|---|
| 0.05 | 3.608787004 | 0.1804393502 |
| 0.1 | 2.681637067 | 0.2681637067 |
| 0.5 | 1.345740275 | 0.6728701376 |
| 1 | 1 | 1 |
| 5 | 0.5018353497 | 2.509176749 |
| 10 | 0.37290654 | 3.7290654 |
| 88.8 | 0.1463187174 | 12.9931021 |

log-log 斜率 = -0.4284（应 = -n = -0.4284），拟合残差 2.22e-16。

### 3.3 本算例口径的场聚合核与锚无关系数

| 配置 | K(n)=ΣA_i^(1-n) [m^(2(1-n))] | dlnK/dn | K 在 n∈[0.20,0.70] 带宽 |
|---|---|---|---|
| BimodalFSD()|phi=0.60 | 639.045635 | -0.928528 | 797.015 … 503.357 |
| BimodalFSD()|phi=0.85 | 639.045635 | -0.928528 | 797.015 … 503.357 |
| BimodalFSD()|phi=0.90 | 639.045635 | -0.928528 | 797.015 … 503.357 |
| DeltaFSD(5.0)|phi=0.60 | 675.3394413 | -0.916289 | 832.552 … 526.553 |
| DeltaFSD(5.0)|phi=0.85 | 675.3394413 | -0.916289 | 832.552 … 526.553 |
| DeltaFSD(5.0)|phi=0.90 | 675.3394413 | -0.916289 | 832.552 … 526.553 |
| LogNormalFSD(0.5)|phi=0.60 | 700.3217525 | -1.04334 | 893.926 … 531.849 |
| LogNormalFSD(0.5)|phi=0.85 | 700.3217525 | -1.04334 | 893.926 … 531.849 |
| LogNormalFSD(0.5)|phi=0.90 | 700.3217525 | -1.04334 | 893.926 … 531.849 |
| PowerLawFSD(2.5)|phi=0.60 | 709.4003571 | -1.02905 | 899.614 … 538.301 |
| PowerLawFSD(2.5)|phi=0.85 | 709.4003571 | -1.02905 | 899.614 … 538.301 |
| PowerLawFSD(2.5)|phi=0.90 | 709.4003571 | -1.02905 | 899.614 … 538.301 |

| 比值（锚无关） | 候选 n 下 | n∈[0.20,0.70] 带宽 |
|---|---|---|
| BimodalFSD()|phi=0.60 / PowerLawFSD(2.5)|phi=0.85 | 0.9008250821 | 0.885952 … 0.935083 |
| BimodalFSD()|phi=0.85 / PowerLawFSD(2.5)|phi=0.85 | 0.9008250821 | 0.885952 … 0.935083 |
| BimodalFSD()|phi=0.90 / PowerLawFSD(2.5)|phi=0.85 | 0.9008250821 | 0.885952 … 0.935083 |
| DeltaFSD(5.0)|phi=0.60 / PowerLawFSD(2.5)|phi=0.85 | 0.9519863283 | 0.925455 … 0.978174 |
| DeltaFSD(5.0)|phi=0.85 / PowerLawFSD(2.5)|phi=0.85 | 0.9519863283 | 0.925455 … 0.978174 |
| DeltaFSD(5.0)|phi=0.90 / PowerLawFSD(2.5)|phi=0.85 | 0.9519863283 | 0.925455 … 0.978174 |
| LogNormalFSD(0.5)|phi=0.60 / PowerLawFSD(2.5)|phi=0.85 | 0.9872024245 | 0.987096 … 0.993677 |
| LogNormalFSD(0.5)|phi=0.85 / PowerLawFSD(2.5)|phi=0.85 | 0.9872024245 | 0.987096 … 0.993677 |
| LogNormalFSD(0.5)|phi=0.90 / PowerLawFSD(2.5)|phi=0.85 | 0.9872024245 | 0.987096 … 0.993677 |
| PowerLawFSD(2.5)|phi=0.60 / PowerLawFSD(2.5)|phi=0.85 | 1 | 1 … 1 |
| PowerLawFSD(2.5)|phi=0.90 / PowerLawFSD(2.5)|phi=0.85 | 1 | 1 … 1 |

结构面口径（A=W·h）：A_face = 88.7763 m²，半宽/全宽载荷比 = 0.6728701376。

### 3.4 机检与复算
| 检查 | 结果 |
|---|---|
| [dime] A/A_ref 无量纲（指数段不携带量纲） | PASS |
| [dime] p(A)=p0·(A/A_ref)^(-n) 量纲 = [Pa] | PASS |
| [dime] F=p(A)·A 量纲 = [N] | PASS |
| [dime] K(n)=ΣA_i^(1-n) 量纲 = m^(2(1-n)) | PASS |
| [dime] F=p0·A_ref^n·K(n) 量纲 = [N]（A_ref 指数必须抵消 K） | PASS |
| [dime] A_ref 归一化敏感性（A_ref 取 1 m 只是单位选择） | PASS |
| [cons] p(A) 严格递减（401 点零违例） | PASS |
| [cons] F(A)=p·A 递增且亚线性（0<n<1） | PASS |
| [cons] d ln p / d ln A = -n | PASS |
| [cons] 尺度不变性 p(λA)/p(A) = λ^(-n)（λ=10） | PASS |
| [cons] 局部-平均一致性 p(A)·A = ∫_0^A p_loc da（4 个面积点） | PASS |
| [cons] 聚合双路一致：Σ p(A_i)A_i vs p0·A_ref^n·K(n) | PASS |
| [cons] 口径退化机检：同一 FSD 不同 φ 的 floe 接触斑集合一致 | PASS |
| 复算 | 残差 0 < 1e-09；采样点反演 n = 0.4284 |
| 登记纪律 | 违例 0；已核销槽位 [] |

## 4. 结果图

- `artifacts/anchors/ice_polygon_floe_field/fig1_operator.png` — p(A) 与 F=p·A 算子，含 n 扫描带（0.20 / 候选 / 0.70）。
- `fig2_field_kernels.png` — 左：φ=0.85 四族 FSD 的 floe 接触斑面积分布；右：锚无关系数随 n 的漂移（缺锚的代价＝带宽）。
- `fig3_checks.png` — 量纲/一致性机检逐条红绿。

## 5. 注意点（坑与局限）

- **正本不可得**：ISO 19906:2019 为付费标准，文件系统 0 命中、公开页 403；引句/表号/页码/p₀/A_ref 全空，故『与规范一致 X%』在本状态下不可检验。

- **K(n) 不是力**：量纲为 m^(2(1-n))；把 A_ref 归一为 1 m 得到的数字不是牛顿（量纲表专门列了这条陷阱行）。

- **口径退化**：floe_patch 口径下同族 FSD 的接触斑集合与堆积浓度无关（残差 ~1e-15），故本工件不能给出『载荷随冰密集度变化』的结论——那只能由力链/堵塞 DEM 实验回答。

- **候选指数只能当对照**：平台内部先例 n 仅用于敏感性带宽，登记 comparison_only，未进入任何 PASS 判据的 measured（登记纪律机检）。

- 文献级对照留待 C1 核销后进行；本工件不引用任何未经逐字抄录的规范数值。


## 6. 误差判定（对基准逐项）

| 关键量 | 基准 | 本工件误差 | 是否 <3% |
|---|---|---|---|
| 复算残差（采样点反演 n、p、F、K） | 机检阈值 1e-9 | 0 | 是 |
| log-log 指数反演 | -n | 0 | 是 |
| 局部-平均压力恒等式 / 聚合双路 | 闭式解 | 2.87e-14 | 是 |
| 场载荷数值 vs ISO 规范值 | ISO 19906 正本 | **未计算**（锚位 WITHHELD） | **不可判定** — 缺 p₀ 与 A_ref，无基准可比；不声明任何百分比 |

**honest FAIL**：C1 FAIL（0/2 锚位核销）→ `verdict = FAIL`；C2 算子层 PASS / 绝对层 WITHHELD；C3、C4 PASS。未完成部分＝绝对层与领域-规范差额（理由见 §5/§6）。

