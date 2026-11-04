# DEM3D 剪切判据秩亏缺修复：单键/链状试件不再被虚假剪切应变误断

## 1. 案例介绍（为什么这是验证欠账，不是新功能）

`docs/dem3d_energy_ledger_analytic_verification.md` 的 **Finding 4** 定谳了一个
**模型缺陷**：`IceDEM3D._objective_shear_strain`（`src/tensordem/dem3d.py`）用
最小二乘局部变形梯度 `F = cross @ pinv(cov)` 重建 Green-Lagrange 应变，再取
偏量不变量 `sqrt(2/3 dev(E):dev(E))` 作为剪切断裂判据。当某颗粒的**参考协方差
`cov = Σ_b r_b r_bᵀ` 秩亏**（少于 3 个非共面键，如单键/链状试件）时，`pinv` 退化为
投影 `e eᵀ`，于是**刚性未变形**的单键也报告 `E = 0.5(e eᵀ - I)` 的偏量范数 `1/3`。

默认剪切阈值 `shear_breaking_strain = 0.03`，故**每个一颗粒宽（链状）试件的键在
第一步就被虚假剪切判据打断**。该缺陷正是闭式验证用例（单键子系统）的构造方式，
因此它阻塞了所有薄/线性试件与单键单元设置——这是能量账本解析验证战役明确
**"本战役未修复"** 的遗留欠账（`docs/dem3d_energy_ledger_analytic_verification.md`
§Recommended fix）。

| 参数 | 值 |
|---|---|
| 键刚度 k | 2000 N/m |
| 颗粒半径 R | 0.025 m |
| 晶格间距 a = 2R | 0.05 m |
| 默认剪切阈值 | 0.03 |
| 单键参考协方差 | `r rᵀ`，秩 1 |
| 虚假不变量（修复前） | 0.333333（=1/3） |

## 2. 怎么做的（方法/流程/关键设置）

**判据先行。** 秩亏邻域（<3 个非共面键）无法定义完整 3-D 变形梯度，最小二乘拟合
退化，故该处的剪切不变量**不可信**。文档给出的"行为保持"替代方案是：**当参考
协方差秩 < 3 时跳过剪切判据**——它不改变任何满秩邻域（所有已交付的立方晶格
Z=26 键，秩=3），因此**无需重标定** UCS/弯曲基准。

**实现（最小改动）。** 在 `_objective_shear_strain` 中，对每个颗粒的 `cov`（n,3,3）
用 `torch.linalg.eigvalsh` 求特征值（升序），以相对容差 `1e-8 × max_eig` 计秩，
`full_rank = (rank ≥ 3)`，最后把不变量乘上 `full_rank` 掩码（秩亏颗粒置 0）：

```python
eig = torch.linalg.eigvalsh(cov)
max_eig = eig[:, -1].clamp_min(1e-30)
full_rank = (eig > 1e-8 * max_eig[:, None]).sum(dim=1) >= 3
...
shear = torch.sqrt((2.0 / 3.0) * dev.square().sum(dim=(1, 2)).clamp_min(0.0))
return shear * full_rank.to(shear.dtype)
```

**工具链。** 复用 `scripts/verify_dem3d_energy_ledger_analytic.py` 的
`probe_shear_invariant()`（P1 单键 / P2 满晶格单轴 / P3 满晶格刚体旋转三锚点）；
测试 `tests/test_dem3d_energy_ledger_analytic.py::ShearInvariantProbeTests` 把
原"缺陷钉死"的两条用例改为"修复钉死"（单键不变量 <1e-10、刚性单键保持存活）。

## 3. 计算结果（程序化注入，零手抄）

`probe_shear_invariant()` 修复前后对比（同一探针，`_probe()` 缓存）：

| 锚点 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| P1 单键未变形不变量 | 0.333333 | **0.0** | <1e-10 |
| P1 单键退化标记 | True | **False** | — |
| P2 满晶格未变形不变量 | 4.62e-16 | 4.62e-16 | <1e-10（不变）|
| P2 满晶格单轴不变量 | 0.006700 | 0.006700 | 与解析 1.6e-12%（不变）|
| P3 满晶格刚体旋转不变量 | 6.34e-16 | 6.34e-16 | <1e-10（不变）|

**行为保持验证**：满秩（立方晶格）三锚点全部**逐位不变**——只有秩亏单键被修正，
故无需重标定任何已交付基准。

## 4. 结果图

本单为**纯代码/测试修复**（无场量云图、无新物理量），故不生成 matplotlib 图。
证据是测试套件绿（见 §5）与探针数值表（§3），理由：修复对象是数值判据的
秩亏退化，不是场量分布；图对"单键不变量从 1/3 变 0"无额外信息量。

## 5. 注意点（踩过的坑/局限/适用范围）

- **容差选择**：`1e-8 × max_eig` 远高于机器精度（~1e-16）又远低于满秩邻域最小
  真实特征值（~0.1×max），故既正确识别秩 1/2 退化，又不误伤各向异性满秩配置。
  已交付立方晶格特征值各向同性（~0.02），无此风险。
- **行为保持边界**：本修复只影响**秩 < 3** 的邻域。若未来出现"近退化"满秩配置
  （最小特征值比 <1e-8），需重审容差——但当前所有基准无此形态。
- **2-D 版本**：`src/tensordem/dem.py::_local_objective_shear_strain` 用 2×2 协方差
  与不同剪切度量（`sqrt((E_xx-E_yy)²+4E_xy²)`），存在**同类**秩亏退化（单 2-D 键
  协方差秩 1）。本单未改（保持改动最小、聚焦 3-D 主求解器），列为后续欠账。
- **不改变断裂物理**：本修复不引入任何新断裂准则，只消除秩亏邻域的**虚假**剪切
  触发；真实剪切断裂（满秩邻域）行为不变。

## 6. 误差判定

| 关键量 | 基准 | 实测 | 相对误差 | <3% ? |
|---|---|---|---|---|
| 单键未变形不变量 | 0（解析）| 0.0 | 0（对零基准）| 是 |
| 满晶格单轴不变量 | 0.006700（解析）| 0.006700 | 1.6e-12% | 是 |
| 满晶格未变形/刚体旋转 | 0（解析）| 4.6e-16 / 6.3e-16 | ~0 | 是 |

**测试**：`tests/test_dem3d_energy_ledger_analytic.py` 17 项全绿；含剪切/断裂/能量/
力学验证的 49 项相关测试全绿（无回归）。修复前单键报告 0.3333（对零基准为
**无限大**相对误差），修复后为 0——这是对"零基准"的显式修正，非数值误差。

## 复跑

```bash
PYTHONPATH=src python -m pytest -q tests/test_dem3d_energy_ledger_analytic.py
PYTHONPATH=src python -m pytest -q \
  tests/test_dem3d.py tests/test_dem3d_shear_specimen.py \
  tests/test_dem3d_fracture_diagnostics.py tests/test_dem3d_fracture_objectivity.py \
  tests/test_dem3d_energy_audit.py tests/test_dem3d_mechanics_validation.py \
  tests/test_dem3d_mechanics_benchmarks.py
```
