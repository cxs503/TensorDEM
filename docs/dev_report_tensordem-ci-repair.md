# TensorDEM CI 基线修复报告（m2419）

**任务**：main 上 `Run complete pytest suite` 作业红；近 30 提交中 14 个带「N 个文件被门禁滤除」标记。
本单**本地全量 pytest 从 44 failed + 5 collection errors → 全绿**，根因全部定位为「测试落地、实现被门禁滤除」的孤儿交付，
并把被滤实现从各 `agent/m<单号>` 分支打捞并集回 main。

- 仓库：TensorDEM，分支 `agent/m2419`（自 `origin/main` = `abcbb39` 切出）
- 判据：`python -m pytest tests/` 全绿（本地全量，非子集）
- 证据分级：本次修复的每条实现均为 **REAL**（可被测试与解析解复核），无 MOCK/STUB 残留

---

## 1. 失败清单与逐个归因

修复前（`pytest tests/ --continue-on-collection-errors -p no:randomly`）：
**44 failed, 859 passed, 2 skipped, 5 errors**（348 s）。

| 模块 | 失败/错误数 | 归因类别 | 根因 | 打捞来源 |
|---|---|---|---|---|
| `test_dem3d_hertz_contact.py` | 1 ERROR | 孤儿 import | `dem3d.hertz_contact_potential` 未落地（Hertz 法向律整体被滤） | m2247 |
| `test_dem3d_restitution.py` | 1 ERROR + 4 FAILED | 孤儿 import + 真实判据缺陷 | 恢复系数闭式解 `clamped_contact_restitution` / 平台探针缺失；打捞后暴露探针配置与闭式解前提不一致 | m2269 |
| `test_fracture_gen_mass_footprint.py` | 1 ERROR | 孤儿 import | `fracture_gen.MASS_RADIUS_FACTOR` / `face_footprint_radius` / `footprint_aware` 未落地 | m2245 |
| `test_friction_cone_verification.py` | 1 ERROR | 孤儿 import | `contact_tangential.advance_tangential_contact` 等库仑锥状态投影未落地 | m2298 |
| `test_tangential_friction_verification.py` | 1 ERROR | 孤儿 import | `contact_tangential.tangential_cycle_dissipated_work` + `tensordem.tangential_verification` 未落地 | m2220 |
| `test_dem3d_contact_impact.py` | 5 FAILED | 真实失败（回归） | 已落地的 SIR-PC2\|F05 重构把法向阻尼**整段替换**为「由 `restitution_coefficient` 派生」，默认 `e = 1.0` 时 `gamma_n = 0`，`contact_damping` 被静默丢弃 → 力律残差 0.82（见 §3.1） | m2140（模块）+ 本单回归修复 |
| `test_dem3d_platen_contact_mask.py` | 3 FAILED | 孤儿实现 | 压板接触掩膜（只在真正相交时受力）未落地 | m2133 |
| `test_dem3d_restitution_verification.py` | 4 FAILED | 孤儿实现 | `tensordem.contact_restitution` 整模块未落地 | m2254 |
| `test_fracture_gen_audit.py` | 14 FAILED | 孤儿实现 | `fracture_gen` 的 `mass_mapping`/`face_extent`/`volume_fraction`/`mass_accounting` 审计 API 未落地 | m2287 |
| `test_ice_floe_physics.py` | 16 FAILED | 孤儿实现 | `benchmarks/ice_floe_physics.py` 四锚点验证框架未落地（main 上是旧版两端锚点） | m2203 |
| `test_restitution.py` | 2 FAILED | 真实失败（回归） | 同上 §3.1 的法向阻尼回归：引擎级闭合 ±3% 判据直接崩 | m2124（审计）+ 本单回归修复 |

**归因结论**：44 个失败中 39 个（5 errors + 34 FAILED）是「测试落地、实现被滤」的孤儿交付；5 个是
SIR-PC2|F05 落地重构引入的**真实物理回归**（法向阻尼被丢弃），并非测试问题。

---

## 2. 打捞清单（11 个被滤交付）

| 单号 | 过滤数 | 打捞回 main 的产物 | 提交时刻 |
|---|---|---|---|
| m2124 | 3 | （`src/restitution.py` 主体已在 main）引擎级审计闭环随回归修复恢复 | 11-03 00:24 |
| m2133 | 4 | `src/tensordem/dem3d.py` 压板接触掩膜、`benchmarks/dem3d/brazilian_cases.json`、`scripts/benchmark_dem3d_brazilian.py` | 00:25 |
| m2140 | 8 | `src/tensordem/contact_impact.py`（夹紧恢复系数闭式解 + RK4 + 成对撞击探针） | 00:27 |
| m2203 | 4 | `benchmarks/ice_floe_physics.py`（四锚点验证：静平衡/阿基米德力/垂荡周期/摩擦起动） | 01:37 |
| m2220 | 5 | `src/tensordem/contact_tangential.py`（`tangential_cycle_dissipated_work`）、`src/tensordem/tangential_verification.py` | 02:37 |
| m2245 | 5 | `src/tensordem/fracture_gen.py`（`MASS_RADIUS_FACTOR`、`mass_radius`、`face_footprint_radius`、`footprint_aware`、`on_plane_tolerance`） | 02:47 |
| m2247 | 6 | `src/tensordem/dem3d.py`（可选 Hertz 法向律 + 稳定步长）、`scripts/verify_dem3d_hertz_contact.py`、`dem3d_cli.py` | 02:43 |
| m2254 | 4 | `src/tensordem/contact_restitution.py`、`scripts/verify_dem3d_restitution_closure.py`（原同名脚本改名，见 §2.1） | 03:43 |
| m2269 | 3 | `src/tensordem/restitution.py`（夹紧闭式解 + 标定反演 + 压板探针） | 04:00 |
| m2287 | 6 | `src/tensordem/fracture_gen.py`（`mass_mapping`/`face_extent`/`volume_fraction`/`mass_accounting`）、`scripts/plot_fracture_gen_audit.py` | 03:36 |
| m2298 | 3 | `src/tensordem/contact_tangential.py`（`advance_tangential_contact` 库仑锥状态投影、`saturation_displacement`） | 04:44 |

### 2.1 同名双实现去重（AST 扫描结论）

三处冲突用 AST 逐文件比对函数名与函数体（脚本见 `artifacts/salvage/merge_fracture_gen.py` 与 §5）：

1. **`restitution.damping_for_restitution`** — 两份实现，**签名语义完全不同**：
   - m2124：`(e_target, stiffness, m_eff, ...)`，对**夹紧**律做单调二分反演（引擎标定口径）；
   - m2269：`(stiffness, restitution, mass)`，对**非夹紧**律做解析反演。
   保留 m2124 版为 `damping_for_restitution`（`tests/test_restitution.py` 按该口径调用），m2269 版改名
   `damping_for_restitution_analytic` 并同步其测试导入（纯改名，判据未动）。
2. **`contact_tangential.compute_tangential_force`** — 第二返回值语义冲突：m2220 要求「投影后的位移态
   `F_t/k_t`」，m2298 写「未饱和弹簧力」。统一为**投影位移态**（m2220 的 3 条断言直接检验它；
   m2298 的消费者全部丢弃该返回值，`advance_tangential_contact` 自带投影），并在文档串里写清口径。
3. **`fracture_gen` 质量映射** — 三套并行 API（`mass_mode ∈ {cell, sphere}`、`mass_mapping ∈
   {equal_volume_spheres, cell_volume}`、`lattice_cell`）。合并为一个解析器
   `resolved_mass_mapping`：`mass_mapping` 优先 → `mass_mode` 别名（cell→cell_volume、sphere→
   equal_volume_spheres）→ 默认 `cell_volume`（体积一致映射，冻结 prereg 意图）。
   `mass_accounting()["relative_error"]` 取**带符号**相对误差，新增 `absolute_relative_error`
   与 `within_15pct`；`test_fracture_gen_audit.py` 中相应 1 处断言加 `abs()`（不改变判据强度）。

另有 **同名脚本** `scripts/verify_dem3d_restitution.py`（m2124 与 m2254 各写一份）：保留 m2124 版
（`tests/test_restitution.py` 通过 `sys.path` 直接 import 它的 `measure_pair`/`measure_platen`），
m2254 版以 `scripts/verify_dem3d_restitution_closure.py` 落盘，两个证据链都保留。

### 2.2 未被滤、但被打捞代码带出来的真实缺陷

- **法向阻尼回归（SIR-PC2\|F05）**：`dem3d.py` 的接触法向力改为「若 `restitution_coefficient < 1` 则用
  恢复系数派生阻尼，否则 `gamma_n = 0`」，**把 `contact_damping` 从力律里删掉了**。修复：抽出
  `IceDEM3D._normal_damping`，`e == 1.0`（默认哨兵）回落到 `contact_damping`。这一条同时修好
  `test_restitution.py::EngineClosureTests`、`test_dem3d_contact_impact.py::SolverImpactTests` 共 7 个失败。
- **压板恢复系数探针不是单自由度**：m2269 的 `measure_platen_restitution` 用 2×2×2 粘结块，只有底层 4 个
  粒子真正接触，键的柔度与逐层接触使「块」不是闭式解描述的 1-DOF 质量块；在 m2133 的物理接触掩膜
  落地后（接触力只在真正相交时存在），块配置相对夹紧闭式解偏 **+11 %…+20 %**（旧数值只是靠「未接触
  也施加阻尼」的非物理前置阻尼凑出来的）。修复：把探针改为**单自由粒子**（其余粒子停靠并固定、键全部
  断开），偏差降到 **≤ 0.1 %**（§3.2 图 B）。

---

## 3. 结果

### 3.1 法向力律回归（修复前 → 修复后）

| 量 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| `pair_impact(e=0.5).force_law_residual` | 0.8239 | 0.0（浮点精确） | < 1e-12 |
| `pair_impact(e=0.5)` 实测 e vs 夹紧闭式 | 判据崩 | 0.55432 vs 0.55028（+0.73 %） | < 3 % |
| `test_restitution` 引擎闭合误差（c = 5, 20 N·s/m） | 崩（>100 %） | < 3 % | < 3 % |
| `contact_impact` 5 条子判据 | 全红 | 全绿 | — |

### 3.2 引擎-解析解对照（图 B 数据，程序化产出、零手抄）

`IceDEM3D` 压板探针 vs 两条闭式解（`k_n = 5 kN/m`，单自由度探针）：

| c_n (N·s/m) | 实测 e | 夹紧闭式 e | 非夹紧闭式 e | 误差(夹紧) | 误差(非夹紧) |
|---|---|---|---|---|---|
| 0.5 | 0.956137 | 0.956068 | 0.955670 | +0.007 % | +0.049 % |
| 1.0 | 0.914927 | 0.914802 | 0.913278 | +0.014 % | +0.181 % |
| 2.0 | 0.839691 | 0.839476 | 0.833888 | +0.026 % | +0.696 % |
| 3.0 | 0.772892 | 0.772607 | 0.761050 | +0.037 % | +1.556 % |
| 5.0 | 0.660086 | 0.659722 | 0.632434 | +0.055 % | +4.372 % |
| 8.0 | 0.530354 | 0.529953 | 0.474468 | +0.076 % | +11.779 % |
| 10.0 | 0.463161 | 0.462764 | 0.387874 | +0.086 % | +19.410 % |
| 12.0 | 0.407460 | 0.407080 | 0.313512 | +0.093 % | +29.966 % |

**误差判定**：关键量「引擎实测恢复系数 vs 夹紧闭式解」逐项 **≤ 0.093 %，是否 < 3 %：是**；
同一探针 vs 非夹紧教科书映射最大 **29.97 %**（证明该映射不能作引擎标定口径）。

同时用一个**独立 RK4**（本单全新编写，非调用仓库闭式解）积分同一条 `F = max(k·p + c·ṗ, 0)` 律，
与仓库三份闭式实现（m2124/m2140/m2269）在 c ∈ {1,2,5,10} 上**逐位一致到 5 位小数**
（0.91480 / 0.83948 / 0.65972 / 0.46276），即"解析锚点"本身不含实现偏差。

### 3.3 修复后全量回归

见 §6 验证输出。

---

## 4. 注意点 / 局限

1. **默认质量映射变更是有意的**：仓库默认从「冻结 legacy 内接球映射」改为 `cell_volume`（= main 落地
   的 `mass_mode="cell"` 语义，也是唯一能在整数格点上复现 FEM 连续体质量的映射）。m2245 的审计脚本与
   2 条测试因此**显式指定** `mass_mapping="equal_volume_spheres"` 才审计冻结映射——-47.64 % 的亏损判据
   一字未改。
2. **`restitution_coefficient` 与 `contact_damping` 的优先级**：`e < 1` 时以 `e` 为准（派生阻尼），
   `e == 1.0` 视为"未指定"回落 `contact_damping`。这是本次修复定的口径，已在代码注释与文档串写明；
   若后续要求"两者都给时报警"，属新判据，不在本单范围。
3. **压板探针的口径变化**：`measure_platen_restitution` 现测单自由粒子（`m_eff = m`），与闭式解假设严格
   对齐；不再覆盖"粘结块撞击"场景。块场景的偏差（+11…+20 %）是**真实的键柔度/逐层接触动力学**，不是
   实现错误，但需要 2-DOF 参考解才能作为判据，未在本单展开。
4. **孤儿打捞用三路合并而非直接覆盖**：main 上同名文件已被其他交付改动（`dem3d.py` 258 行、`restitution.py`
   297 行、`fracture_gen.py` 85 行插入），直接 checkout 会回退他人成果，因此逐 hunk `git apply --3way` +
   手工并集，冲突处均保留双方能力。
5. **未做的事**：未逐个重跑被打捞交付自带的 `scripts/*.py` 审计脚本全量产出（仅跑被测试 import 的路径）；
   `outputs/`、`artifacts/` 下的历史 JSON 未重生成（不属 pytest 判据）。

---

## 5. 复现

```bash
cd /root/tp-agent/work/m2419
python -m pytest tests/ -q --continue-on-collection-errors -p no:randomly   # 判据
python artifacts/salvage/merge_fracture_gen.py                              # 三套质量映射并集生成器
```

---

## 6. 验证输出（本地全量）

图：`artifacts/salvage/ci_repair_evidence.png`
（A 面板＝修复前各模块失败数；B 面板＝修复后引擎压板探针 vs 两条闭式解，误差标注为程序化注入）

```
$ python -m pytest tests/ -q --continue-on-collection-errors -p no:randomly
.......................................................................... [ 96%]
..
978 passed, 2 skipped, 2 warnings, 114 subtests passed in 408.97s (0:06:48)
```