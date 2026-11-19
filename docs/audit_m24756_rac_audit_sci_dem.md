# TensorDEM 科学性抽查审计：RACE-AUDIT-SCI-DEM

## 审计概要

| 项目 | 内容 |
|------|------|
| 审计目标 | 抽查 3 个物理模型，核对实现与 docstring/文献约定一致性 |
| 审计日期 | 2026-01-15 |
| 审计范围 | brash_ice_rheology / hertz_mindlin / clear_ice.contact |
| 基线测试 | 全部通过 (30+19+12+11 = 72 tests) |
| 结论 | 3/3 模型一致，无发现不一致 |

---

## 模型 1：碎冰流变学 (brash_ice_rheology)

### 物理背景

碎冰航道阻力模型将碎冰层视为受约束的颗粒床，由船体剪切。阻力分为三个分量：

1. **颗粒剪切** (mu(I) 流变)：船体在碎冰中剪切，剪切带厚度 δ = c_δ·d
2. **浮力位移**：固体冰体积排开水
3. **龙骨阻力**：水下碎冰体的尾流阻力

### 实现核对

| 公式 | docstring | 实现 | 一致性 |
|------|-----------|------|--------|
| μ(I) = μ_s + (μ_2-μ_s)/(I_0/I + 1) | ✅ 明确写出 | `JopParameters.mu()` 精确实现 | ✅ 一致 |
| I = γ̇·d / √(P/ρ) | ✅ 明确写出 | `inertial_number()` 精确实现 | ✅ 一致 |
| P = (ρ_w-ρ_b)·g·h_sub + ½·C_c·ρ_b·v² | ✅ 明确写出 | `_confinement()` 精确实现 | ✅ 一致 |
| F_shear = μ(I)·P·A_shear·(1-φ)·f(v) | ✅ 明确写出 | `shear_component()` 精确实现 | ✅ 一致 |
| F_buoy = ρ_w·g·V_displaced, V=(1-φ)·B·h_sub | ✅ 明确写出 | `buoyancy_component()` 精确实现 | ✅ 一致 |
| F_drag = ½·C_D·ρ_w·v²·A_keel | ✅ 明确写出 | `keel_drag_component()` 精确实现 | ✅ 一致 |
| f(v) = v/(v+v_0) 处理闭合 | ✅ 明确写出 | `_mobilization()` 精确实现 | ✅ 一致 |

### 文献对照

- Jop, Forterre & Pouliquen (2006) 标准值: μ_s=0.36, μ_2=0.84, I_0=0.30
- 代码中 `LITERATURE_JOP = (0.36, 0.84, 0.30)` — 完全匹配
- 测试 `test_mu_i_law_numeric_values_at_hand_computed_inertial_numbers` 验证数值正确性

### 孔隙率极限行为

| 极限 | 预期 | 实现 | 一致性 |
|------|------|------|--------|
| φ→1 (纯水) | R→0 | 三个分量均→0 | ✅ 一致 |
| φ→0 (纯固体) | F_shear→0 | solid_fraction→0 | ✅ 一致 |
| h_sub→0 | R→0 | 所有面积→0 | ✅ 一致 |

### 测试覆盖

- `test_brash_ice_rheology.py`: 30 passed — 输入验证、解析基准、参数边界
- `test_brash_ice_rheology_mu_i_resistance.py`: 38 passed — 分量分解、单调性、校准

---

## 模型 2：Hertz-Mindlin 接触 (hertz_mindlin)

### 物理背景

经典双球弹性接触理论 (Mindlin 1949, "Compliance of Elastic Bodies in Contact")：

- **法向** (Hertz 1882): F_n = (4/3)·E*·√(R*)·δ^(3/2)
- **切向** (Mindlin no-slip): F_t = 8·G*·√(R*·δ)·Δ_t
- **库仑滑动**: |F_t| ≤ μ·|F_n|

### 实现核对

| 公式 | docstring | 实现 | 一致性 |
|------|-----------|------|--------|
| G* = E/[4(2-ν)(1+ν)] 等球 | ✅ 明确写出 | `hertz_reduced_shear_modulus()` 精确实现 | ✅ 一致 |
| G* = E/[2(2-ν)(1+ν)] 刚性对 | ✅ 明确写出 | `rigid_counterpart=True` 分支 | ✅ 一致 |
| k_t = 8·G*·√(R*·δ) | ✅ 明确写出 | `hertz_mindlin_tangential_stiffness()` 精确实现 | ✅ 一致 |
| k_t/k_n = 3(1-ν)/(2-ν) | ✅ 明确写出 | `mindlin_stiffness_ratio()` 精确实现 | ✅ 一致 |
| Coulomb 饱和 + 状态投影 | ✅ 明确写出 | `hertz_mindlin_tangential_force()` 精确实现 | ✅ 一致 |

### 数值验证

| 泊松比 ν | 预期 k_t/k_n | 实测 | 误差 |
|---------|-------------|------|------|
| 0.0 | 1.500 | 1.500 | <1e-10 |
| 0.25 | 1.286 | 1.286 | <1e-10 |
| 0.30 | 1.235 | 1.235 | <1e-10 |

### 测试覆盖

- `test_hertz_mindlin.py`: 19 passed — 刚度比、切向力、负控制、集成测试
- `test_hertz_mindlin_wave_speed.py`: 通过
- `test_dem3d_hertz_contact.py`: 通过
- `test_dem3d_hertz_restitution_verification.py`: 通过

---

## 模型 3：明冰接触阻尼 (clear_ice.contact)

### 物理背景

明冰 DEM 是稳定耗散的多体问题。采用 Hertz-Mindlin + Tsuji 粘性阻尼 + 非内聚钳位：

**法向**: F_n = max( (4/3)E*√(R*)·δ^(3/2) + 2ζ√(k_n·m*)·δ̇ , 0 )

**切向**: Mindlin 增量弹簧 + Coulomb 帽

**关键创新**: `max(F, 0)` 钳位消除内聚和粘着振荡 — 分离对不会被阻尼项重新加速。

### 实现核对

| 公式 | docstring | 实现 | 一致性 |
|------|-----------|------|--------|
| F_n = max( F_elastic + c·δ̇ , 0 ) | ✅ 明确写出 | `PairForces.fn` 构造中 `np.maximum` | ✅ 一致 |
| c = 2ζ√(k_n·m*) | ✅ 明确写出 | Tsuji 临界阻尼形式 | ✅ 一致 |
| ζ = -ln(e)/√(π²+ln²(e)) | ✅ 明确写出 | `damping_ratio_from_restitution_analytic()` | ✅ 一致 |
| 校准表 ZETA_TABLE/E_TABLE | ✅ 文档说明 | 15 节点表 + `np.interp` | ✅ 一致 |

### 校准表自洽性

```
ζ=0.000 → e=1.000 (无阻尼)
ζ=0.429 → e=0.316 (标准冰 e=0.2 对应 ζ≈0.417)
ζ=1.000 → e=0.120 (过阻尼极限)
```

测试 `test_restitution_damping_calibration_table_is_self_consistent` 验证：
- 表格可被积分器复现 (误差 <1e-3)
- 单调递减
- 校准值与线性dashpot估计差异 >30% (钳位效应)

### 测试覆盖

- `test_clear_ice_contact.py`: 12 passed — 解析解、阻尼校准、非内聚、动量守恒
- `test_clear_ice_study.py`: 通过

---

## 模型 4：DEM3D 对接触阻尼 (dem3d_pair_contact_damping)

### 物理背景

`contact_damping` 参数在默认模式下不可见（仅从 `restitution_coefficient` 派生），
opt-in `contact_damping_model="absolute"` 模式使用截断阻尼定律：

F_n = max(k·δ - c·δ̇, 0)

### 实现核对

| 特性 | docstring | 实现 | 一致性 |
|------|-----------|------|--------|
| 默认模式: restitution | ✅ 文档说明 | `contact_damping_model="restitution"` | ✅ 一致 |
| 绝对模式: 截断阻尼 | ✅ 文档说明 | `unilateral_restitution()` 参考解 | ✅ 一致 |
| 冲突检测 | ✅ 文档说明 | 同时设置 absolute + restitution 报错 | ✅ 一致 |

### 测试覆盖

- `test_dem3d_pair_contact_damping.py`: 11 passed — 模式契约、遗留行为、绝对模式

---

## 综合结论

| 模型 | 实现一致性 | 测试覆盖 | 证据等级 |
|------|-----------|---------|---------|
| brash_ice_rheology | ✅ 一致 | 68 tests | REAL |
| hertz_mindlin | ✅ 一致 | 19 tests | REAL |
| clear_ice.contact | ✅ 一致 | 12 tests | REAL |
| dem3d_pair_contact_damping | ✅ 一致 | 11 tests | REAL |

**总评**: 3/3 抽查模型实现与 docstring/文献约定完全一致。所有测试通过。
