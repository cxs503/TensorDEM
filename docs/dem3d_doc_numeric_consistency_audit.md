# 文档数值主张一致性审计（证据链内容层，m3558）

状态：工具 + 台账 + 棘轮闸门 + 负控测试落地；审计语料 25 篇文档 136 条验收行；
发现并修正 1 处自相矛盾的验收结论。代码与数据：`scripts/audit_doc_numeric_claims.py`、
`docs/doc_numeric_claims_ledger.json`、`scripts/plot_doc_numeric_claims.py`、
`tests/test_doc_numeric_claims.py`、图 `docs/figures/doc_numeric_claims_*.png`。

## 1. 案例介绍（审计对象、语料规模、为什么这是欠账）

TensorDEM 的论文级结论几乎都以表格形式出现：「关键量 → 基准 → 实测 → 相对误差 →
是否 < 3 %」。上一轮 `docs/dem3d_doc_evidence_pointer_audit.md`（m2985）解决了**指针层**：
文档引用的每个路径都可解析（ORPHAN = 0）。它在 §6 明确留下未做的一半：

> **指针≠正确性**：本次只保证「文档→文件」可达，不保证文件内容与文档数字一致。
> 内容一致性是另一层（需逐文档重跑 benchmark 复现数字），本次未做，属明确的后续增量。

本单就做这一层的**可机器判定部分**：不重跑任何仿真，只审「文档自身印出来的数字能否支撑它自己
印出来的结论」。这是证据链上最便宜、也最容易腐烂的一环——数字被改一半、结论栏没跟着改，
文档看上去毫无异常，但读者按文档复算会得到相反的结论。

审计语料（参数表）：

| 项 | 值 |
|---|---|
| 扫描范围 | `docs/**/*.md` |
| 语料 Markdown 表格行 | 2120 |
| 带验收结论（`<3%?` / 达标）的表 | 42 张 |
| 进入判定的验收行 | 136 |
| 贡献验收行的文档 | 25 |

## 2. 怎么做的（方法、判据、工具链）

`scripts/audit_doc_numeric_claims.py`（纯标准库，单次运行 ≈ 0.18 s，2600 行/秒级别）：

1. **解析**：逐文档切 Markdown 表格；表头按语义定位四列——验收结论列
   (`< 3 %?` / `达标` / `是否通过`)、相对误差列、基准列、实测列。分隔行与多行表体都处理。
2. **判据 A（重算）**：若一行同时给出基准 `r`、实测 `m`、相对误差 `e`，则
   `|m-r|/|r|` 必须落在**实测值印刷精度**决定的区间内。基准按精确值处理（它们是闭式解或
   设计名义值），实测按印刷半 ULP 处理。只有区间窄到能分辨该误差量级时才判定
   （区间宽度 ≤ 区间上界的 50 %），否则记 `printed resolution too coarse to discriminate`
   ——宁可弃权，不制造噪声。
3. **判据 B（结论逻辑）**：回答「是」的行，必须存在**单一单位读法**（整行按分数读，或整行
   按百分数读）使该行所有误差都 < 3 %；回答「否」的行必须使至少一个误差 ≥ 3 %。
   写成 `%` 的字面量锁死百分数读法（`0.4 %` 不允许被重新解释成 40 %）。
4. **弃权分类**：误差单元格是散文（`0.00 deg (bracket 0.1 deg = 0.3 % of 11.3 deg)`）、
   表格根本没有误差列、基准与实测共用一个复合列（`measured vs reference`）、
   单元格不是单个数——一律记 `UNCHECKABLE` 并写原因，**绝不静默通过**。
5. **棘轮**：`--write` 落台账；`--check` 与台账比对，出现新的 FAIL 行、或原本可判定的行
   退化成 `UNCHECKABLE`（例如把数字改写成散文），返回 1。数值结论只能变好，不能悄悄变坏。

判据的**负控**写进 `tests/test_doc_numeric_claims.py`（23 个测试）：合成文档里预置已知坏结论
（8 % 误差答「是」、0.4 % 误差答「否」、0.099918 vs 0.1 却写 8.2 %、散文单元格、
复合表头、棘轮新增 FAIL、可判定行退化为散文），每一种都必须被抓到；另有一组「干净文档
不得误报」的正控。没有这组控制，"0 处不一致"与"检查器什么也没做"无法区分——这与把求解器
换成 stub 仍报告通过是同一类错误。

## 3. 计算结果（全部由 `--emit-tables` 从台账程序化生成，零手抄）

rows_with_verdict=136 checkA=3 checkB=102 status={'PASS': 102, 'UNCHECKABLE': 34}
| metric | value |
|---|---|
| markdown table rows in `docs/**/*.md` | 2120 |
| rows carrying an error verdict | 136 |
| documents contributing verdict rows | 25 |
| check B (verdict logic) evaluated | 102 |
| check A (recompute) evaluated | 3 |
| status PASS | 102 |
| status FAIL | 0 |
| status UNCHECKABLE | 34 |

| why a row could not be recomputed | rows |
|---|---|
| table has no reference/measured columns | 86 |
| reference and measured share one composite column | 8 |
| reference/measured cells are not single numbers | 3 |
| printed resolution too coarse to discriminate | 2 |
| (recomputed) | 3 |

| document | rows | PASS | FAIL | UNCHECKABLE |
|---|---|---|---|---|
| `docs/dem3d_energy_ledger_analytic_verification.md` | 15 | 13 | 0 | 2 |
| `docs/dem3d_fracture_gen_audit.md` | 11 | 9 | 0 | 2 |
| `docs/buoyancy_free_float_verification.md` | 8 | 8 | 0 | 0 |
| `docs/dem2d_hull_polyline_geometry_audit.md` | 8 | 6 | 0 | 2 |
| `docs/dem3d_bond_fracture_energy.md` | 7 | 6 | 0 | 1 |
| `docs/dem3d_tangential_friction_verification.md` | 7 | 5 | 0 | 2 |
| `docs/ice_floe_physics_validation.md` | 7 | 6 | 0 | 1 |
| `docs/ridge_capacity_skempton_audit.md` | 7 | 7 | 0 | 0 |
| `docs/dem3d_buoyancy_float.md` | 6 | 0 | 0 | 6 |
| `docs/dem3d_contact_impact_analytic.md` | 6 | 6 | 0 | 0 |
| `docs/dem3d_lattice_elasticity.md` | 6 | 6 | 0 | 0 |
| `docs/dem3d_rolling_resistance_verification.md` | 6 | 4 | 0 | 2 |
| `docs/dem2d_lattice_elasticity.md` | 5 | 2 | 0 | 3 |
| `docs/dem3d-rolling-resistance/report.md` | 5 | 3 | 0 | 2 |
| `docs/TENSORDEM_UPSTREAM_DIFF_M1630.md` | 4 | 4 | 0 | 0 |
| `docs/dem3d_brazilian_hondros_closure.md` | 4 | 4 | 0 | 0 |
| `docs/dem3d_integration_order_verification.md` | 4 | 4 | 0 | 0 |
| `docs/dem3d_shear_static_reference.md` | 4 | 1 | 0 | 3 |
| `docs/dem3d_rolling_resistance.md` | 3 | 0 | 0 | 3 |
| `docs/dem_analytic_verification_debt_restoration.md` | 3 | 3 | 0 | 0 |
| `docs/fracture_gen_mass_footprint_audit.md` | 3 | 1 | 0 | 2 |
| `docs/threefield_lbm_transport_verification.md` | 3 | 1 | 0 | 2 |
| `docs/dem3d_scale_benchmark.md` | 2 | 2 | 0 | 0 |
| `docs/dem3d_doc_evidence_pointer_audit.md` | 1 | 1 | 0 | 0 |
| `docs/dem3d_shear_cross_section_convergence.md` | 1 | 0 | 0 | 1 |

| document | reference | measured | stated error | admissible interval | recomputed? |
|---|---|---|---|---|---|
| `docs/dem3d_tangential_friction_verification.md` | 0.1 | 0.099918 | 0.0822 % | [0.000815, 0.000825] | yes |
| `docs/dem3d_tangential_friction_verification.md` | 0.5 | 0.499969 | 0.0062 % | [6.1e-05, 6.3e-05] | yes |
| `docs/dem3d_tangential_friction_verification.md` | 20.4 | 20.399988 | 6e-05 % | [5.64e-07, 6.13e-07] | yes |

## 4. 结果图

图 1 `docs/figures/doc_numeric_claims_status.png`：25 篇文档的验收行按审计结论堆叠
（PASS / FAIL / UNCHECKABLE），可见验收行在文档间的分布极不均匀，少数几篇长报告
（力学标定、剪切、滚阻系列）贡献了大部分行。

图 2 `docs/figures/doc_numeric_claims_recomputability.png`：内容层的真实覆盖面——
136 条验收行里只有 3 条能由文档自身给出的两个数字重算，86 条（63 %）所在的表格
**根本没有基准/实测列**。这不是算术错误，而是**可复核性缺口**：这些行只留下了误差，
没有留下产生误差的两个数。

两张图都由 `scripts/plot_doc_numeric_claims.py` 直接读台账绘制，报告中的数字与图同源。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

* **复合表头会制造假失败**：`| measured vs reference |` 一列同时命中我的「基准」与「实测」
  关键字，若允许两列同指一列，`r == m` 使重算误差恒为 0，于是把两行正确的记录判成 FAIL。
  修法是显式要求基准列 ≠ 实测列并记 `reference and measured share one composite column`。
  这个假失败一直到加负控测试才暴露——再次说明控制组不是形式主义。
* **单位歧义是真实风险**：`0.0822` 既可能是 8.22 % 也可能是 0.0822 %。早期版本两种读法都试，
  结果「0.4 % 答否」被解释成「40 % 答否」而放行。现在 `%` 锁死读法，歧义只允许出现在
  没写单位数的行上。
* **棘轮的键不能用整行文本**：最初用 `doc + 整行` 作键，一行只要改数字（哪怕改对了）就被
  判成「退化」。现在用 `doc + 首列标签` 作键，并区分三种回归：新增 FAIL、PASS→FAIL、
  可判定→UNCHECKABLE。
* **判据 A 的覆盖面很窄（3 行）**，这是语料形状决定的，不是实现限制：绝大多数验收表只印
  误差，不印基准与实测（86 行），另外还有单元格是散文（28 行含在 34 条 UNCHECKABLE 内）、
  复合列（8 行）、多数字单元格（3 行）、精度不足（2 行）。因此本单的结论强度主要来自
  判据 B（102 行）——它能证明「结论栏没有与自己的误差数字矛盾」，但**不能**证明误差本身
  是真算出来的。后者需要重跑产生误差的脚本，属下一层增量。
* **只审 `docs/**/*.md`**：根 `README.md`、`AGENTS.md` 不在范围内（与 m2985 范围一致）。
* **不做语义判断**：文本里写「否则必须不一致」的解释性说明会被当作结论栏读，这正是本单
  唯一一处真实发现的成因（见下）。审计不理解物理意图，只做算术一致性。

**本单唯一一处真实发现（已修）**：`docs/dem3d_integration_order_verification.md` 第 117 行
在 `< 3 % ?` 列里答 `no`，而同行误差是 0.93 %（< 3 %）。读上下文可知该行比较的是连续解
`at²/2` 而非离散闭式解，`no` 回答的其实是「是否应当一致」这个不同的问题。已把该单元格
改为 `n/a — an O(dt) truncation offset is *expected* here ...`，使列头与单元格语义一致；
物理结论未改。修正后该行退出判定集合（136 行，原 137 行），台账已同步。

## 6. 误差判定（逐项，显式回答「是否 < 3 %」）

本单度量的量是**离散整数计数**与**印刷精度的算术恒等**，判据是精确包含（容差 0），
不涉及任何物理基准，因此「< 3 %」这一列对机械量不适用，如实声明如下：

| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 结论栏自相矛盾的验收行 | 0 | 0（修正前 1，见 §5） | 精确 0 | **是**（整数恒等） |
| 台账与当前树一致性 | 逐条相等 | 相等（`--check` 返回 0） | 精确 0 | **是** |
| 可重算行的重算一致性 | 落在印刷精度区间内 | 3/3 落在区间内 | 精确 0 | **是** |
| 负控测试通过率 | 23/23 | 23/23 | 精确 0 | **是** |
| 内容层可复核覆盖率 | 100 %（理想） | 3/136 = 2.2 % 可重算 | 不适用 | **否——如实报告**（缺口成因见 §5：86 行表格无基准/实测列） |
| 物理精度 | 无基准 | 本单不产生物理量 | 不适用 | **不适用**（声明原因：审计对象是文档自身的算术一致性，非力学量） |

**诚实结论**：本单把「文档数值主张」这一层做成了可重复执行的闸门，并证明它确实会失败
（负控 23 项）；但在当前语料上它**没有**给出「误差数字本身可信」的证据——只有 2.2 % 的行
能就地重算。要闭合那一层，需要逐文档重跑 benchmark 复现数字（m2985 §6 指出的同一方向），
本单未做，不作任何粉饰。
