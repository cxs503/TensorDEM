# P73W2-N11: 堆积床有效热导率

## 概述

本模块实现随机堆积床（单分散球）的有效热导率 k_eff(φ, k_s/k_f) 计算，
用于散货自热（coal/grain self-heating）的上游缺口。

## 物理模型

### 接触热导网络
- Hertz 弹性接触半径：a_c = √(R_eff · δ)
- Maxwell 收缩热阻：R_contact = 1/(2 k_s a_c)
- 扩展热阻：R_spread = 1/(π k_s R_particle)
- 键热导：g_bond = 1/(R_contact + R_spread)

### FVM 场求解
- 周期边界 + 线性背景场：T = T' + G·z
- CG 迭代求解 Kirchhoff 方程
- 体素化：固体/流体相标记，调和平均面热导

### 双通道分解
- 固体骨架通道：图网络（接触对稀疏线性系统）
- 间隙流体通道：FVM（固体近似绝缘）
- 并联叠加：k_eff = k_solid_channel + k_fluid_channel

## 解析模型

### Wiener 严格变分界（硬门）
- 串联界：k_series = 1/((1-φ)/k_s + φ/k_f)
- 并联界：k_parallel = (1-φ)k_s + φk_f
- **不可伪造的最强判据**：k_series ≤ k_num ≤ k_parallel

### Maxwell-Eucken ME1/ME2
- ME1（流体连续）：k_eff = k_f(k_s+2k_f+2(1-φ)(k_s-k_f))/(k_s+2k_f-(1-φ)(k_s-k_f))
- ME2（固体连续）：k_eff = k_s(k_f+2k_s+2φ(k_f-k_s))/(k_f+2k_s-φ(k_f-k_s))
- 极限：φ→0 → k_s，φ→1 → k_f（精确）
- **注意**：ME 假设完美接触，预测值高于含接触电阻的数值解

### Zehner-Schlünder (1970)
- 单元胞模型：流体圆柱 + 轴上球
- 径向级数积分，近轴路径受 B/λ 限制
- 自身散度 ~10%

### Krupiczka (1967)
- 经验关联：k_eff/k_f = λ^n
- n = 0.280 - 0.757 log₁₀(φ) - 0.057 log₁₀(λ)

## 验证判据 (G1-G8)

| 判据 | 描述 | 阈值 |
|------|------|------|
| G1 | 热流守恒（各 z 面通量相等） | ≤1e-10 |
| G2 | Wiener 硬门（所有组合） | 全部满足 |
| G2b | ME 锚点 + 极限 | k_num ≤ ME2，极限 <2% |
| G3 | Z-S 对比 | 因子 2 内 |
| G4 | 退化极限 | k_s=k_f 误差 ≤1e-9 |
| G5 | 种子散度 | CV ≤3% |
| G6 | 均质 1D 线性剖面 | T'≈0，斜率=1 |
| G7 | Kozeny-Carman/Archie | m ∈ [1.5, 3.0] |
| G8 | 辐射排除 | T > 300°C 显式排除 |

## 已知局限

1. **网格依赖性**：体素化 FVM 接触热导分布导致网格依赖（12→32 变化 ~40%）
2. **ME 模型**：假设完美接触，不含接触电阻
3. **Z-S 散度**：高对比度下与 FVM 偏差较大
4. **辐射**：T > 300°C 时辐射不可忽略，模型显式排除

## 使用示例

```python
from tensordem.thermal import packed_bed as pb

# 生成堆积床
bed = pb.generate_static_bed(porosity=0.40, n_spheres=100, seed=42)

# 计算有效热导率
result = pb.effective_conductivity(bed, k_solid=1.4, k_fluid=0.026, grid=24)
print(f"k_eff = {result['k_eff']:.4f} W/(m·K)")

# Maxwell-Eucken 模型
me1 = pb.maxwell_eucken_1(0.4, 50.0, 0.6)  # 流体连续
me2 = pb.maxwell_eucken_2(0.4, 50.0, 0.6)  # 固体连续
```

## 文件

- `src/tensordem/thermal/packed_bed.py`：主模块
- `tests/test_packed_bed_conductivity.py`：原有测试（19 项）
- `tests/test_packed_bed_p73w2n11.py`：G1-G8 验证测试（29 项）
- `scripts/packed_bed_p73w2n11_validation.py`：完整验证脚本
- `benchmarks/packed_bed_p73w2n11/result.json`：机器可读结果
- `benchmarks/packed_bed_p73w2n11/figures/`：结果图

## 参考

- Zehner, P., Schlünder, E.U. (1970) Chemie Ingenieur Technik 42, 933-941
- Krupiczka, R. (1967) Int. J. Heat Mass Transfer 10, 567-574
- Maxwell, J.C. (1873) A Treatise on Electricity and Magnetism
