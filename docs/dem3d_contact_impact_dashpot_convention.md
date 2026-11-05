# DEM3D 碰撞时长/恢复系数解析基准复核：绝对阻尼约定缺失（真实验证欠账清偿）

范围：`src/tensordem/contact_impact.py`（成对碰撞验证 harness）、
`tests/test_dem3d_contact_impact.py`（已合并的解析基准套件）、新增
`tests/test_contact_impact_dashpot_convention.py` 与
`scripts/verify_contact_impact_durations.py`。本增量不改物理模型，只修
**验证 harness 的装置约定**，并把它对解析基准的偏差如实量化。

## 1. 欠账是什么（问题陈述）

已合并的解析验证套件 `tests/test_dem3d_contact_impact.py` 在本分支 `HEAD` 上是红的：

```
FAILED tests/test_dem3d_contact_impact.py::SolverImpactTests::test_contact_durations
E  AssertionError: 0.12167254282818885 not less than 0.03
```

进一步复算（两球对心碰撞探针，装置与 `src/tensordem/contact_restitution.py` 的 `measure_dem3d_impact` 同构；见 §3 的 before 列）显示问题不是"差 12%"
这么小：harness 在**任何**目标恢复系数下测得的 `e` 都 ≈ 1.0（`e_target=0.5` →
`e_measured=1.00021`），说明对心碰撞里**阻尼根本没起作用**，测出来的"接触时长"是
纯弹性碰撞的时长。因此该套件中 `test_restitution_matches_clamped_analytic`、
`test_known_bias_is_visible_in_solver`、`test_corrected_mapping_realises_target` 等
断言全部一起失效（`-x` 只在第一个失败处停下，掩盖了这一点）。

根因（一行配置）：`DEM3DConfig.contact_damping_model` 默认 `"restitution"`，
该分支**忽略** `contact_damping`，改用 `restitution_coefficient` 重建 `gamma_n`
（`src/tensordem/dem3d.py:665`）。而 harness 传入的 `c_n = restitution_to_damping(e, k_n, m_eff)`
是 **N·s/m 绝对阻尼**（正是被复核的映射），于是：

| 传入 | 引擎实际使用 | 后果 |
|---|---|---|
| `contact_damping = c_n` [N·s/m] | `gamma_n` 由 `restitution_coefficient`（默认 1.0）导出 | `gamma_n ≈ 0` → 完全弹性，`e≈1.0` |

同一坑在 `scripts/verify_dem3d_restitution.py:93` 已被显式规避
（`contact_damping_model="absolute"`），但 `contact_impact.py` 的 harness 漏了这一行。
换言之：**已合并的"接触时长/恢复系数解析验证"结论在装置层面是不可复现的**
（`WITHHELD` 级证据，而不是文档声称的 REAL）。

## 2. 修法（最小改动）

`_build_pair()` 的 `DEM3DConfig` 显式声明装置约定：

```python
contact_damping_model="absolute",   # gamma_n = contact_damping [N s/m]
```

一行改动，不动求解器默认值（`"linear"`/`"restitution"` 仍是引擎默认，既有回归
artefact 不受影响）。同时新增
`tests/test_contact_impact_dashpot_convention.py`（5 项）把该约定钉死：
测得 `e` 必须落在耗散侧（`<0.95`）、必须落在钳制闭式解 3% 内、两个接触时长基准
必须 3% 内闭合、"力钳零后仍重叠"的时长排序必须成立、以及 `dt` 细化必须单调收小残差
（证明残差属离散而非巧合）。

## 3. 计算结果（程序化生成，`scripts/verify_contact_impact_durations.py`）

算例：等质量对心碰撞，`k_n = 5.0e4 N/m`、`r = 0.05 m`、`ρ = 900 kg/m³`（`m_eff = 0.2356 kg`）、
`v0 = 0.5 m/s`，初间隙 1 µm，其余格点固定。参考解 = 力钳制（`max(0, k·δ − c·δ̇)`）
弹簧-阻尼闭式解：力持续时长 `θ*/ω_d`，几何重叠时长 `θ*/ω_d + δ(t*)/(e·v0)`。

修复前（默认 `"restitution"` 约定，同一 harness）：

| `e_target` | `dt` [s] | `e_measured` | 力时长误差 | 重叠时长误差 |
|---|---|---|---|---|
| 0.7 | 1.19e-4 | 1.00032 | +6.400 % | −1.197 % |
| 0.5 | 1.01e-4 | 1.00021 | **+12.167 %** | −2.921 % |
| 0.3 | 6.07e-5 | 1.00008 | **+21.282 %** | −5.052 % |

修复后（`contact_damping_model="absolute"`，全 dt 扫描）：

| `e_target` | `dt_scale` | `e_measured` | `e` 误差 vs 钳制闭式 | 力时长误差 | 重叠时长误差 | 接触步数 |
|---|---|---|---|---|---|---|
| 0.7 | 1.0 | 0.72210 | +0.543 % | −1.067 % | −1.197 % | 57 |
| 0.7 | 0.5 | 0.71993 | +0.241 % | −1.067 % | −0.330 % | 115 |
| 0.7 | 0.25 | 0.71900 | +0.111 % | −0.133 % | −0.330 % | 230 |
| 0.5 | 1.0 | 0.55432 | +0.733 % | −1.226 % | −1.473 % | 68 |
| 0.5 | 0.5 | 0.55212 | +0.334 % | −0.389 % | −0.748 % | 137 |
| 0.5 | 0.25 | 0.55109 | +0.146 % | −0.389 % | −0.386 % | 275 |
| 0.3 | 1.0 | 0.39938 | +0.564 % | −1.458 % | −0.813 % | 117 |
| 0.3 | 0.5 | 0.39812 | +0.246 % | −0.375 % | −0.389 % | 235 |
| 0.3 | 0.25 | 0.39749 | +0.089 % | −0.375 % | −0.177 % | 471 |

最差值：`|Δe| = 0.733 %`、`|Δ t_force| = 1.458 %`、`|Δ t_overlap| = 1.473 %`（全部在默认
`dt` 下即已 <3%），且随 `dt` 减半单调收小（`e`: 0.733→0.334→0.146 %，
比率 2.19/2.29 ≈ 一阶）。力时长残差在细网格上量化平台化（步数取整，`dt/T_f ≈ 1 %` 量级），
这正是"步数×dt"度量的一阶特性，已在文档与测试中显式声明，而不是靠加宽判据掩盖。

产物：`artifacts/contact_impact_dashpot/impact_duration_convergence.{json,csv,png}`
（9 行全表 + 三联图：恢复系数误差、时长误差 vs `dt`、绝对时长 vs 解析值）。

## 4. 结果图

`artifacts/contact_impact_dashpot/impact_duration_convergence.png`

* (a) 三个 `e_target` 的实测恢复系数相对钳制闭式解的误差 vs `dt`，全部落在 ±3% 带内并随 `dt` 收小；
* (b) 两个接触时长基准的误差 vs `dt`（力钳制时长系统性偏短 −0.1…−1.5%，重叠时长同量级）；
* (c) 绝对时长：模拟重叠时长严格大于力钳制时长（物理正确性：力归零后仍重叠），且两者分别贴住解析曲线。

## 5. 注意点（陷阱与适用范围）

1. **引擎默认约定 ≠ 验证 harness 要验证的约定。** `contact_damping_model="restitution"`
   存在的原因见 `dem3d.py:143-149` 的注释（工具/压板路径用同一 `gamma_n`）；但任何
   注入"绝对阻尼系数"的外部 harness 都必须显式切换，否则静默退化为弹性碰撞。
   **症状识别**：`e_measured ≈ 1.0002`（比 1 略大是离散导致的超调），力时长误差随
   `e_target` 减小而暴涨（6.4% → 12.2% → 21.3%），因为解析接触时长随阻尼变长而模拟时长不变。
2. **不能用"加宽判据"过关**：把 3% 放宽到 15% 会让 `test_known_bias_is_visible_in_solver`
   反过来失去意义（该测试正是靠 3% 带证明未钳制映射有 >3% 偏差）。
3. 实测 `e` 比目标大（`e=0.5` → 0.554）**不是新错误**，而是已审计的"未钳制映射偏差"
   （`docs/dem3d_contact_impact_analytic.md`）：`restitution_to_damping` 反演的是未钳制
   恒等式，钳制力截断使实际回弹更快。修复后该偏差在求解器中重新可见（+10.9%），
   与闭式预测一致；`damping="clamped_mapping"` 分支则把目标恢复到 <3%。
4. 适用范围：线性弹簧-阻尼 + 零力钳制 + 自由对心碰撞、`ζ < 1`。Hertz 非线性接触
   （`contact_model="hertz"`）的时长基准是另一条标度律，不在本增量覆盖内。

## 7. 跨模块同源复核（2026-11，CI 清红）

同一个默认值坑在本仓另一处验证装置上复发：`contact_restitution.measure_dem3d_impact`
构造 `DEM3DConfig(contact_damping=gamma_n)` 时同样没有声明
`contact_damping_model="absolute"`，因此 §1 的"完全弹性"症状原样出现——该系统在
`gamma_n ∈ {0, 1, 3, 5, 10} N·s/m` 下测得的 `e` 全为 `1.00007`，与解析钳制参考值
（`e_reference = 0.9148 / 0.7726 / 0.6597 / 0.4628`）完全无关。

修法同上，一行配置（`measure_dem3d_impact` 内声明装置约定）。修复后该装置与
`clamped_restitution_ratio` 的偏差为 **0.01 %…0.39 %**（`gamma_n = 0…10`，
含 `both_free=True` 的折合质量 `m/2` 情形），即 §2 的结论由 `WITHHELD` 升为 REAL。
回归门禁：`tests/test_dem3d_restitution_verification.py`（9 项）。

## 6. 复现

```
python -m pytest tests/test_contact_impact_dashpot_convention.py tests/test_dem3d_contact_impact.py -q
python scripts/verify_contact_impact_durations.py
```

本分支实测：`tests/test_dem3d_contact_impact.py` +
`test_contact_impact_dashpot_convention.py` + `test_restitution.py` +
`test_dem3d_restitution.py` + `test_contact_tangential.py` +
`test_friction_cone_verification.py` + `test_tangential_friction_verification.py`
= **93 passed（修复前该子集 1 failed）**；新增模块单独 5 passed / 22 passed（含原套件）。