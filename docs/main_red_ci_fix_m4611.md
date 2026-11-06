# main CI 红修复记录（m4611）

基线：`main @ 1a5a9c99`，`unit-tests` job 红。全量本地复跑（`python -m pytest tests -q`，
torch 2.14.1 / CPU）得到 **22 failed, 1478 passed, 10 skipped**，分布在 5 个测试文件。

本单据「完成标准 = main CI 绿」逐项定位根因并修复，全部改动均在 worktree 内：

| # | 红测 | 数量 | 根因 | 处置 |
|---|---|---|---|---|
| 1 | `test_dem3d_contact_invariants.py::test_mixed_topology_all_invariants_pass` | 8 subtest | 测试按**不变量名**派发 `getattr(contact_invariants, invariant)(case, seed)`，但模块只提供描述性函数名（`translation_invariance` …），`translation/rotation/brute_force/determinism` 四个句柄根本不存在 → `AttributeError` | 在生产模块补齐不变量名入口 + `INVARIANT_CHECKS` 注册表（`DEFAULT_OFFSET` 统一平移量），`__all__` 导出 |
| 2 | `test_dem3d_energy_friction_rolling.py` | 11 | (a) **真实内核缺陷**：`_alive_bond_for_pairs` 在无键体系（`pairs` 非空、`_bond_keys` 为空）时对空张量做 `searchsorted`+索引 → `IndexError: index is out of bounds for dimension with size 0`，整个能量账本无法初始化。(b) 修好崩溃后暴露的 7 条锚点失配：Case R 闭式解忽略了「滚阻只能阻碍滑移」→ 越过 `t_stop` 后不会反向加速；Case F 子系统法向通道开着，绝对 residual 不可能是固定焦耳地板 | (a) `dem3d.py` 加空键集短路（全部视作 contact）；(b) 锚点改为物理正确闭式：滚阻功 `-μ_r F_n v0²/(2a)`（停止距离）、`|v_T| < 0.02 v0`、零 μ_r 负控、dt 细化不劣化；摩擦侧 residual 改按机械能标度评分（2%）并写明法向通道开放的局限 |
| 3 | `test_dem3d_hertz_stability.py` | 6 | 断言与仓内**已修正的判据**矛盾：无阻尼半隐式 Euler 在稳定域内 `det A = 1` ⇒ `ρ ≡ 1`（谱半径在单位圆上），绝不是 0.32–0.82；`d = dt_rec·ω ≈ 0.02–0.05`，则「0.3 R 附近失稳」也不成立（需要 39 倍步长，因 `k_sec ∝ √δ`，临界重叠是 `O(10⁶ R)`）。另外 `bond_stiffness=0/contact_stiffness=0` 被 `DEM3DConfig` 校验明确拒绝（零刚度会让所有键对静默无力） | 按 `tensordem.stability` 的正确闭式重写：设计重叠处 `d ≤ 0.5` 且 `ρ = 1±1e-12`；失稳阈值用 `dt = 2/ω` 双侧穿越验证；设计点步长余量 `dt_crit/dt_rec − 1 > 10`；新增「`recommended_dt` 由 Hertz 割线刚度支配、与亚 Hertz 线性刚度无关」用例 |
| 4 | `test_doc_evidence_pointers.py` | 4 | 4 个指针指向**未入库**文件（二维晶格弹性文档、三支 CI 取日志/修复合集脚本，均系门禁滤除的历史产物，本树无此文件） | 指针改指同主题且在库文档 / 用 `gh` CLI 等价流程表述，然后 仓内证据审计脚本 重生成 证据账本 JSON（unresolved 0 orphan） |
| 5 | `test_ridge_dem_core_physics.py::test_recommended_dt_elastic_limit` | 1 | 测试只断言弹性极限，而 `RidgeMaterial.recommended_dt` 返回 `min(elastic, viscous)`；该组参数（13×5e4 + 2e3 阻尼）下粘性极限才是紧的（2.209e-4 s vs 弹性 3.329e-4 s） | 断言改为 `min()` 并显式记录「本组参数由粘性极限支配」 |

## 验证

```
$ python -m pytest tests -q
```
修复前 22 failed / 1478 passed → 修复后见 `artifacts/m4611/` 内的日志摘要。

受影响文件单跑：`test_dem3d_contact_invariants.py` 15 passed（56 subtests）、
`test_dem3d_hertz_stability.py` + `test_dem3d_energy_friction_rolling.py` +
`test_ridge_dem_core_physics.py` 37 passed、`test_doc_evidence_pointers.py` 14 passed。

## 局限

* 本单只是恢复 main 的合并健康度：修复的 4 个锚点用例改的是**断言**（因为旧断言与仓内
  正确闭式/接口契约矛盾），代码侧只改了 `_alive_bond_for_pairs` 的空键集缺陷与
  `contact_invariants` 的公开入口；没有新增物理能力。
* Case F 的法向通道仍为开放子系统，residual 只能按机械能标度评分（2%），没有做到绝对焦耳
  地板；真要闭合需在脚本层加法向跟踪力并重新推导可回复功恒等式。
* 计算集群（外协方 loong64）上的 torch 因缺 `libflexiblas.so.3` 无法导入，全量回归只能在
  控制面 CPU 上跑；集群侧属环境欠账，未在本单处理。