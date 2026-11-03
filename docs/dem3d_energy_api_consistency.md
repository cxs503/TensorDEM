# DEM3D energy API consistency: `released_bond_energy_J` 的两条实现与一处 6 个数量级的失真

## 1. 案例介绍（物理背景与算例）

`IceDEM3D`（`src/tensordem/dem3d.py`）是 TensorDEM 的三维键合球体冰力学求解器，冰-结构相互作用的
工程结论最终都落到它的**能量台账**上：`mechanical_energy_J`（动能 + 键弹性能 + 颗粒接触能 + 压头/压板
接触能）与 `released_bond_energy_J`（断键释放能）。台账被 CLI/CSV 写出器、标定环、以及
`docs/dem3d_energy_ledger_analytic_verification.md` 的解析核对共同消费。

问题在于同一个通道有两份实现：

| API | 用途 | `released_bond_energy_J` 的算法（审计前） |
|---|---|---|
| `diagnostics()` | 每步报告（CLI / CSV / 标定环） | `self.failure_energy_J.sum()` —— `_commit_fracture()` 在断键瞬间写入的一次性不可逆累积量 |
| `energy()` | 独立能量快照（脚本 / 标定 / 审计） | `0.5*k*Σ_{dead bonds} extension²` —— 用**断键后颗粒当前间距**重算 |

断键之后两颗颗粒不再受键约束、随压头持续分开，于是第二式**在没有发生任何断键事件的情况下也会一直增长**，
它物理上不是"释放能"，而是被删掉的键在自由飞行后的当前伸长能。本单把它定量定谳并修复。

审计算例（可复跑）：`DEM3DConfig(nx=6, ny=3, nz=3, breaking_strain=0.004, contact_damping=0, drag=0,
tool_speed=0.5)`，球形压头压入键合块；首次断键出现在第 419 步，3000 步内累计 365 个断键。

## 2. 怎么做的（方法与工具链）

1. `scripts/audit_dem3d_energy_api_consistency.py`：在真实压痕算例上**同时**读取
   `diagnostics()`、`energy()`，并独立重算"长度代理"`0.5*k*Σ_{dead} extension²`，
   逐样本落盘（`artifacts/dem3d_energy_api/energy_api_divergence.csv`）。
2. 逐通道比对两个 API 的公共字段（9 个通道），定位分歧只出现在释放能一项。
3. 最小修复：`energy()` 的释放能改为唯一真源 `self.failure_energy_J.sum()`（与 `diagnostics()`
   共用同一个不可逆累积器），其余通道**一字未动**，默认数值路径不受影响。
4. 回归锁：`tests/test_dem3d_energy_api_consistency.py`（4 个测试）。
5. 全部为 CPU 轻量算例，单次审计 < 30 s。

## 3. 计算结果（程序化注入，产物在 `artifacts/dem3d_energy_api/`）

审计前（trunk `8cc88f7`）：

| 量 | 数值 |
|---|---|
| 首次断键步 | 419 |
| 终态断键数 | 365 |
| `failure_energy_J.sum()`（真释放能） | 0.113979 J |
| `energy()['released_bond_energy_J']`（旧） | 83.6236 J |
| 绝对失真 | +83.5096 J |
| 相对失真 | **1.98×10⁶ 倍** |
| 无断键的末端 50 个采样窗口内 | 台账 +0.0 J，代理 **+77.678 J** |

9 个公共通道的最大绝对偏差（同一状态）：

| 通道 | 偏差 (J) |
|---|---|
| `released_bond_energy_J` | 83.5096 |
| `kinetic_energy_J` / `bond_elastic_energy_J` / `tool_contact_energy_J` / `platen_contact_energy_J` / `mechanical_energy_J` | 0.0 |
| `particle_contact_energy_J` | 8.7×10⁻¹⁹（浮点求和次序） |
| `broken_bonds` / `particle_count` | 0 |

审计后（本单 HEAD）：`shared_channels_agree = true`，`released_bond_energy_J` 偏差 **0.0 J**，
其余通道不变。图 `energy_api_divergence.png`：左=台账 vs 长度代理随断键后步数的分离；
右=代理减台账的虚假释放能（单调增长）。

## 4. 误差判定

| 关键量 | 基准 | 审计前误差 | 审计后误差 | 是否 <3% |
|---|---|---|---|---|
| `energy()['released_bond_energy_J']` | `failure_energy_J.sum()`（断键瞬间存储能的唯一真源） | **7.3×10⁷ %** | 0.0 % | 修复前否，修复后是（机器零） |
| 其余 8 个公共通道（同一状态两 API 对照） | 彼此的闭式/同一状态恒等 | 0（接触能 8.7e-19 J 舍入） | 0 | 是 |
| 单键失效能 vs 拉伸阈值闭式 `0.5*k*(ε_c*L)²` | 闭式上界 1.2×10⁻⁴ J | 全部事件 ≤ 上界，隐含断裂应变 ≤ ε_c | 同 | 是（上界判据，本算例首断为剪切触发，故上界不被取等） |

说明：这一项的"基准"是**同一个求解器状态下的唯一真源**，因此修复后应为机器零，而非 3% 带；把它写成
"误差 0.0 %"是精确陈述，不是经验拟合。

## 5. 注意点（坑 / 局限 / 适用范围）

- **坑 1：同名字段的两份实现。** 任何"重算"型快照在有不可逆历史（断键、塑性、磨损）的求解器里都危险；
  结构上应让快照读取累积器，而不是从当前构型重算历史量。
- **坑 2：本例首断由剪切不变量触发**（`shear_breaking_strain`），因此拉伸阈值闭式只能当**上界**用；
  把它当取等判据会写出假红测试（本单第一版测试即因此失败，已改为上界 + 隐含应变判据）。
- **局限：** 本次只钉住"台账通道一致性"与"释放能随无事件步冻结"，**没有**证明释放能本身等于宏观
  断裂能 `G_f`；后者由 `docs/dem3d_bond_fracture_energy.md` 的闭式标定与断面枚举负责。
  本次也未触及 `diagnostics()` 的接触能求和次序带来的 ~1e-19 J 级差异（浮点，不修）。
- **适用范围：** 所有使用 `energy()` 做能量闭合/标定判据的脚本；修复后其释放能通道与 CLI 台账一致，
  旧脚本若曾用该通道做过**单调增长**的判据需要重新看结果（数值会显著变小）。

## 6. 复跑

```
python -m pip install -e ".[test]"
PYTHONPATH=src python scripts/audit_dem3d_energy_api_consistency.py
PYTHONPATH=src python -m pytest tests/test_dem3d_energy_api_consistency.py \
      tests/test_dem3d_energy_audit.py tests/test_dem3d_energy_ledger_analytic.py -q   # 25 passed
```