# 2-D 引擎断裂账本通道的闭式审计（fracture_release_J / pair_contact_acquired_J）

状态：验证欠账清偿（2026-11-04，selfstart m3721）。报告 3817（m3635）闭合了
2-D 能量账本的工具功/耗散/残差，并点名剩余欠账：`fracture_release_J` 与
`pair_contact_acquired_J` 只有恒等式覆盖——恒等式对**通道数值**不敏感（某通道
整体乘错常数，对面通道吸收误差，等式照样闭合）。本单把这两个通道钉到独立编码
的闭式解上。

## 1. 案例与参数（SI，程序化注入）

| 项 | 值 |
|---|---|
| 粒子半径 r / 静长 L₀ | 0.025 m / 0.05 m |
| 键刚度 k / 接触刚度 k_c | 2000 N/m / 2000 N/m |
| 断裂应变 ε_b / 剪切 ε_s | 0.015 / 0.03 |
| Case A 预应变 ξ₀ | L₀(ε_b+1e-4) = 7.55e-4 m（单键受控拉断） |
| Case B 扫描 | k ∈ {2000, 4000} N/m × ξ₀ ∈ {8e-4, 1.6e-3} m |
| Case C 网格 / 刀具速度 / 时长 | 7×3 / 0.2 m/s / 0.12 s（生产路径） |
| Case C 步长 dt | 3.479e-4 s（recommended_dt），345 步 |
| Case D 负控 | ξ₀ = 0.99·ε_b·L₀，50 步 |

## 2. 方法

- **A 受控拉断**：2×2 网格只留一条轴向键，预置伸长 ξ₀（>阈值），零速，单步。
  闭式：`released = ½kξ₀²`；拉断无重叠 → `acquired ≡ 0`。
- **B 通道敏感性（非恒等式判据）**：账本必须按 `k·ξ²` 缩放；硬编码或仅恒等式
  的通道过不了。
- **C 生产路径逐事件复核**：求解器在**步前位置**评估断裂，故每步快照步前几何，
  对每个新断键独立重算 `½k(d−L₀)²` 与 `½k_c(2r−d)⁺²`，与账本累计值比对。
- **D 负控**：亚阈值应变 50 步不断键、账本恒零。
- 引擎约定（本单确认）：断裂步内已断键不再施力，释放能直接耗散、不转动能，
  故单步总能量残差 R = Δ(U+KE)+released−acquired 应为机器精度零。

工具链：Python 3.11 + torch float64，`scripts/verify_dem2d_fracture_ledger.py`
生成 `docs/artifacts/dem2d_fracture_ledger/`（JSON/CSV/PNG），
`tests/test_dem2d_fracture_ledger.py` 5 项回归。运行须 `PYTHONPATH=$PWD/src`
（provenance 测试防兄弟工作树 sys.path 泄漏，同 3817 §5.3 的坑）。

## 3. 结果（程序化注入）

| case | 量 | 引擎 | 闭式 | 相对误差 | <3% ? |
|---|---|---|---|---|---|
| A | fracture_release_J | 5.700250e-04 | 5.700250e-04 | 3.4e-15 | 是 |
| A | pair_contact_acquired_J | 0 | 0 | — | 是 |
| A | 记录伸长 vs 预置 ξ₀ | 7.550000e-04 | 7.55e-04 | 1.7e-12 | 是 |
| A | 单步总能量残差 | 0.0 J | 0 | 0 | 是 |
| B | 4 组 k·ξ² 缩放 | 全部命中 | k·ξ² | <1e-12 | 是 |
| C | released（21 事件） | 1.329703e-02 | 1.329703e-02 | 3.9e-16 | 是 |
| C | acquired（5 个重叠事件） | 4.798520e-03 | 4.798520e-03 | 3.6e-16 | 是 |
| D | 断键数 / 账本 | 0 / 0 | 0 / 0 | 0 | 是 |

Case C 的 21 个断裂事件中 5 个发生在 d<2r（重叠）状态，acquired 通道非空，
审计非空转。

## 4. 结果图

`docs/artifacts/dem2d_fracture_ledger/fracture_ledger_closure.png`：
左=A/B 账本 vs ½kξ²（四点全落 1:1 线）；中=C 逐事件 recorded vs 闭式；
右=C 事件审计相对误差随 dt 细化（机器精度平台，无离散化漂移）。

## 5. 注意点

1. **断裂步无力的约定**：`forces()` 先判断、后算力，断键当步不施力——释放能
   全部按耗散记账，不产生动能。这是模型约定而非缺陷，但意味着「断裂能→动能」
   的转化路径在本引擎不存在；若未来加断裂能→动能分配，需重审此约定。
2. **范围**：闭式锚只证明**实现**忠于 `½kξ²` / `½k_c·overlap²` 公式，不证明
   `bond_stiffness`/`breaking_strain` 对应任何实测冰模量/强度（材料标定是
   另一条欠账线，见 `docs/dem3d_bond_fracture_energy.md` 的 3-D 侧）。
3. **负载纪律**：全部案例 <2 万粒子步（CPU），load15≈122/128 核下完成。

## 6. 误差判定

关键量 vs 闭式基准逐项：A released 3.4e-15、A 残差 0、B 缩放 <1e-12、
C released 3.9e-16、C acquired 3.6e-16、D 恒零——**全部 <3%，判定 PASS**。
无 FAIL 项；无基准豁免项。
