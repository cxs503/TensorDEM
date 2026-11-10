# P493-N03: TensorDEM 颗粒姿态层从零到一验证面

## 概述

本验证面为 TensorDEM 颗粒旋转动力学建立从零到一的锚点，以 Poinsot 无外力矩刚体运动
Jacobi 椭圆函数解析解为核心基准，验证凸多面体惯性张量计算、四元数姿态积分、角动量守恒的完整链路。

## 验收判据与结果

| 判据 | 描述 | 阈值 | 实测最差值 | 状态 |
|------|------|------|-----------|------|
| C1 | 凸多面体惯性张量闭式 vs 解析解 | < 1e-10 | 7.36e-16 | ✅ PASS |
| C2 | Poinsot 姿态角误差 (10 周期) | < 0.1° | 0.076° | ✅ PASS |
| C3 | 四元数归一化漂移 (1000 步) | < 1e-12 | 2.22e-16 | ✅ PASS |
| C4 | 姿态矩阵正交性 ||R^T R - I||_F | < 1e-12 | 2.36e-15 | ✅ PASS |
| C5 | 角动量守恒 \|L\| 漂移 | < 1e-10 | 8.36e-12 | ✅ PASS |
| C6 | 动能守恒漂移 | < 1e-10 | 1.82e-11 | ✅ PASS |

## 文件清单

- `src/tensordem/attitude_verification.py` — 核心验证模块（已有，本次修正正四面体惯量公式引用）
- `src/tensordem/attitude_matrix.py` — 姿态矩阵工具（新增：四元数→旋转矩阵、正交性检验）
- `tests/test_p493_n03_attitude_verification.py` — pytest 验证用例（新增）
- `verify_p493_n03.py` — 独立验证脚本（含报告生成与可视化）
- `artifacts/p493_n03/verification_report.json` — 机器可读验证报告
- `artifacts/p493_n03/*.png` — 5 张验证图

## 运行方式

```bash
# 独立验证脚本（含可视化）
python verify_p493_n03.py

# pytest 用例
python -m pytest tests/test_p493_n03_attitude_verification.py -v
```

## 物理背景

### Poinsot 运动

无外力矩刚体的 Euler 方程：
```
I₁ dω₁/dt = (I₂ - I₃) ω₂ ω₃
I₂ dω₂/dt = (I₃ - I₁) ω₃ ω₁
I₃ dω₃/dt = (I₁ - I₂) ω₁ ω₂
```

其解析解由 Jacobi 椭圆函数给出（设 I₁ < I₂ < I₃）：
- 绕最大惯量轴（轴3）稳定旋转：ω₁ ∝ cn(τ,k), ω₂ ∝ sn(τ,k), ω₃ ∝ dn(τ,k)
- 绕最小惯量轴（轴1）稳定旋转：类似形式
- 绕中间惯量轴（轴2）不稳定（Dzhanibekov 效应）

### 凸多面体惯性张量

基于 Mirtich (1996) 的闭式公式，通过散度定理将体积分转化为三角形面与原点构成的
四面体之和。符号体积保证原点在多面体外时仍正确。

## 参考文献

- Landau & Lifshitz, *Mechanics*, §33-37
- Goldstein, Poole & Safko, *Classical Mechanics*, 3rd ed., Ch. 5
- Poinsot, *La théorie nouvelle de la rotation des corps*, 1834
- Mirtich, "Fast and Accurate Computation of Polyhedral Mass Properties", *J. Graphics Tools*, 1(2), 1996
