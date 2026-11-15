# CASE-DEM-BIAXIAL-FRICTION: 双轴剪切摩擦角算例

## 概述

本算例测量平面应变双轴压缩 DEM 试样的峰值 Mohr-Coulomb 摩擦角，并与经典临界状态参考值比较。

## 物理背景

平面应变双轴压缩试验是土力学中测量摩擦性材料强度的标准方法。试样在恒定侧向围压下轴向加载，通过峰值偏应力计算 Mohr-Coulomb 摩擦角：

$$\sin\phi = \frac{\sigma_1 - \sigma_3}{\sigma_1 + \sigma_3}$$

其中 $\sigma_1$ 为轴向（大主）应力，$\sigma_3$ 为侧向（小主）应力，凝聚力 $c = 0$。

## 参考基准

Bolton (1986) "The strength and dilatancy of sands", *Géotechnique* **36**(1): 65-78.

- **峰值摩擦角**：Table 2, p.72：密实 Hostun 砂平面应变峰值摩擦角 $\phi_{ps} = 38°$（$D_r = 80\%$, $\sigma_3 = 100$ kPa）
- **临界状态摩擦角**：Table 1, p.67：$\phi_{crit} \approx 33°$（棱角石英砂）

## 验收标准

| 量 | 基准 | 容差 |
|---|---|---|
| 峰值摩擦角 $\phi_{peak}$ | 38.0° (Bolton 1986) | 相对误差 < 8% |

临界状态分析为信息性交叉校验，不作为验收门。

## 算例配置

### 锚点配置（正式结果）

| 参数 | 值 | 说明 |
|---|---|---|
| 试样尺寸 | 24×48 列×行 | 1152 圆盘 |
| 颗粒半径 | 0.75 m | |
| 围压 $\sigma_3$ | 100 kPa | |
| 壁摩擦 $\mu_{wall}$ | 0.0 | 无摩擦侧壁（OSS 惯例） |
| 固结步数 | 4000 | |
| 应变率 | 0.05 s⁻¹ | |
| 应变目标 | 0.30 | 延伸至 30% 以提取临界状态 |
| 伺服增益 | 3×10⁻⁴ | |
| 冻结合金比例 | 0.35 | |

### 快速配置（冒烟测试）

8×16 = 128 圆盘，应变目标 5%，固结 300 步。仅验证管线可运行，不作为结果。

## 关键设计决策

### 为什么 wall_friction = 0？

侧向伺服壁粗糙时承载竖向剪力，偏置顶板测量的轴向应力。OSS 三轴/双轴惯例使用无摩擦侧壁，使边界牵引为纯法向。

### 为什么 strain_target = 0.30？

临界状态检测器要求轴向应变 ≥ 0.20 才能开始窗口分析。30% 应变目标确保后峰分支足够长以提取临界状态摩擦角。峰值摩擦角在偏应力最大值处测量（通常 $\varepsilon_a \approx 0.05$–$0.10$），更长目标不影响峰值测量。

## 应力定义

双应力定义独立可审计：

1. **壁面牵引**：$\sigma_v = -F_{top}/(W \cdot t)$, $\sigma_h = 0.5(-F_l + F_r)/(H \cdot t)$
2. **Love-Weber 内部 virial 应力**：含边界层接触力

## 临界状态分析

从应力路径尾部提取临界状态摩擦角 $\phi_{cv}$，并与 Rowe (1962) 应力剪胀预测交叉校验：

$$\frac{\sigma_1}{\sigma_3} = \left(\tan^2\left(45° + \frac{\phi_\mu}{2}\right)\right) \cdot \left(1 - \frac{d\varepsilon_v}{d\varepsilon_1}\right)$$

其中 $\phi_\mu = \arctan(\mu)$ 为颗粒间摩擦角（$\mu = 0.55$ → $\phi_\mu \approx 28.8°$）。

## 运行方式

```bash
# 锚点算例（正式结果）
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py

# 快速冒烟测试
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py --quick
```

## 产物

- `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` — 完整结果记录
- `artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_friction.png` — 三面板图
- `artifacts/CASE-DEM-BIAXIAL-FRICTION/sens_200kPa/run.json` — 围压敏感性（200 kPa）

## 测试覆盖

`tests/test_case_dem_biaxial_friction.py` 钉扎：

- 预声明配置/参考/容差不可静默变异
- `friction_angle_deg` 解析值校验
- `judge_friction` 验收算术边界校验
- 快速管线端到端可运行
- 临界状态分析集成
- 提交工件自洽性（如存在）
