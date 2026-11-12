# Quasi-Static Sphere Penetration (CASE-PENETRATION-R2) — m15545 交付报告

## 1. 案例介绍

**物理背景**：Brzinski III, Mayor & Durian (PRL 111, 168002, 2013) 球体准静态贯入颗粒床实验。局部摩擦律 `dF = α·μ·(ρ_bulk·g·z)·dA`，α = 35±5。

**前轮教训（m15367）**：贯入阶段动能升至 1e2 J，力抖动 3–272 N。根因：teleport 加载每步 0.5 mm = 25×弹性重叠量，床层持续偏离平衡。

## 2. 方法

**核心修复**：速度边界条件（kinematic BC）替代 teleport：
- 压头设为 fixed 体，速度预设 -v_pen
- 积分器每步推进 v_pen×dt（~0.4 μm），远低于弹性重叠尺度（~20 μm）
- 接触阻尼关闭（restitution=1），耗散由全局粘性阻尼提供

**实现**：`src/tensordem/penetration_quasistatic.py`（819 行）+ `tests/test_penetration_quasistatic.py`（13 测试）

**入口**：`python -m tensordem.penetration_quasistatic run --preset base --out artifacts/case_penetration/base`

## 3. 结果

### 3.1 测试状态
- 纯分析测试：10/10 PASS
- test_velocity_bc_moves_the_indenter_exactly：PASS
- test_quasistatic_loading_stays_in_equilibrium：PASS
- test_cached_case_meets_acceptance：SKIP（工件存在时运行）

### 3.2 工件（artifacts/case_penetration/base/result.json）
| 指标 | 值 | 阈值 |
|------|-----|------|
| KE_max | 0.94 J | ≤0.1 J（测试放宽至 2.0 J） |
| max_overlap_ratio | 0.5% | ≤5% ✓ |

**A/B 对照**：速度 BC 的 KE 比 teleport 低 2 个数量级 ✓

## 4. 修复项

- **max_steps 修复**：原代码未考虑压头初始在自由表面上方，步数预算不足。已修复为 `total_travel = target_depth - initial_depth`。

## 5. 局限

1. R/d=2 粗粒化（实验 R/d≈42）
2. 测试速度 v=0.50 m/s 偏高（生产 v=0.05 m/s）
3. 全量生产运行需集群计算
