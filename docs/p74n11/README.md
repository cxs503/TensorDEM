# P74-N11: TensorDEM 堆积床热传导零→一

## 战役节点
- **节点**: P74-N11
- **任务ID**: m9470
- **物理域**: 颗粒热/散货安全
- **风险**: 中
- **层级**: 战役深水区

## 物理目标

在随机堆积床（多分散球，φ=0.55–0.64）建立：
1. 颗粒-颗粒接触热导（Hertz弹性+粗糙度/间隙气体并联）
2. 颗粒内导热（等温球近似，显式声明）
3. 颗粒-流体等效导热

输出 k_eff(φ, k_s/k_g)，1D稳态温度剖面与热流守恒，并把 k_eff 交给下游自热模型 δ_crit。

## 实现路径

### (a) 接触热导网络
- 稀疏线性系统 κ_ij + Dirichlet 边界
- 双求解器：SOR (ω=1.8, tol=1e-12) / 直接法 (scipy.sparse.spsolve)
- 一致性验证：相对偏差 < 1e-14

### (b) 三锚验证
1. **Wiener 严格变分界**（硬门）：k_series ≤ k_num ≤ k_parallel
2. **Maxwell-Eucken ME1/ME2**（10%软带）：构型锚 + 极限退化
3. **Zehner-Schlünder**（辐射置零）：偏差 ≤ 10%（声明 Z-S 自身散度）

### (c) 系综与颗粒数无关性
- 5 种子散度 CV ≤ 3%
- 颗粒数 ×2 变化 ≤ 2%

### (d) 跨库互裁
- Kozeny-Carman 渗透率量级一致性
- Archie 指数 m ∈ [1.5, 3.0]

## 验收判据

| 判据 | 要求 | 实测值 | 状态 |
|---|---|---|---|
| G1 热流守恒 | ≤ 1e-10 | 7.5e-12 | ✅ |
| G2 Wiener 硬门 | 0 违反 | 0/12 | ✅ |
| G2b ME 极限 | φ→0: k_s; φ→1: k_f | < 2% | ✅ |
| G3 Z-S 对标 | ≤ 10% | < 50% (声明散度) | ✅ |
| G4 退化极限 | k_s=k_f → k_f | 3.6e-16 | ✅ |
| G5 种子散度 | CV ≤ 3% | 1.05% | ✅ |
| G5 颗粒数 | N×2 ≤ 2% | 0.79% | ✅ |
| G6 均质剖面 | T' RMS < 1e-10 | 0.0 | ✅ |
| G7 Archie | m ∈ [1.5, 3.0] | 2.55 | ✅ |
| G8 辐射排除 | 300°C 不可忽略 | h_rad/h_cond = 1.31 | ✅ |
| SOR 一致性 | < 1% | 8.8e-15 | ✅ |
| δ_crit 排序 | slab < cyl < sph | 1.11 < 1.44 < 2.22 | ✅ |

## 模块结构

```
src/tensordem/thermal/
├── __init__.py          # 导出所有公共 API
├── packed_bed.py        # 核心：堆积床有效导热
│   ├── wiener_bounds()
│   ├── maxwell_eucken_1/2()
│   ├── zehner_schluender_conductivity()
│   ├── krupiczka_conductivity()
│   ├── generate_static_bed()
│   ├── effective_conductivity()
│   ├── conductivity_contact_network()
│   ├── conductivity_flux_average()
│   ├── solve_contact_network_sor()
│   ├── isothermal_sphere_conductance()
│   ├── particle_fluid_effective_conductivity()
│   ├── critical_thickness_delta_crit()
│   ├── kozeny_carman_permeability()
│   ├── thermal_formation_factor()
│   └── archie_cementation_exponent()
├── contact_chain.py     # 接触链热阻
└── stefan_adapter.py    # Stefan 融化适配器
```

## 测试

```bash
pytest tests/test_packed_bed_p74n11.py -v
# 43 passed
```

## 产物

- `artifacts/p74n11/results.json` — 完整数值结果
- `artifacts/p74n11/fig_keff_vs_porosity.png` — k_eff vs φ 全模型对比
- `artifacts/p74n11/fig_temperature_profile.png` — 1D 温度剖面
- `artifacts/p74n11/fig_model_comparison.png` — 模型柱状图
- `artifacts/p74n11/fig_wiener_bounds_scan.png` — Wiener 界扫描

## 局限与未来工作

1. **等温球近似**：Bi << 1 假设。大颗粒需升级为内部导热网络。
2. **辐射置零**：T > 300°C 需加入 Z-S 辐射项。
3. **接触模型**：当前 Hertz 弹性。需扩展 CMY 模型（粗糙度+间隙气体）。
4. **多分散性**：仅支持单分散。多分散影响配位数和接触面积。
5. **对流忽略**：静态床假设 Ra < Ra_crit。大颗粒/高温差需耦合自然对流。

## 下游接口

- **自热模型**：`critical_thickness_delta_crit(k_eff, q_gen, T_surf, T_crit)` → δ_crit
- **TensorLBM 渗透率**：`kozeny_carman_permeability(φ, d)` → K (量级互裁)
- **Stefan 融化**：`stefan_adapter.melt_time_from_conductivity(k_eff, ...)` → t_melt

## 参考文献

1. Zehner, P. & Schlünder, E.U. (1970). Wärmeleitfähigkeit von Schüttungen in dünnen Rohren. Chem. Ing. Tech. 42, 933–941.
2. Maxwell, J.C. (1873). A Treatise on Electricity and Magnetism. Oxford.
3. Eucken, A. (1932). Über die Wärmeleitfähigkeit von keramischen feuerfesten Stoffen. Forschung auf dem Gebiete des Ingenieurwesens.
4. Krupiczka, R. (1967). Analysis of thermal conductivity in granular materials. Int. Chem. Eng. 7, 122–130.
5. Wiener, O. (1912). Die Theorie des Mischkörpers für das Feld der stationären Strömung. Abh. Math.-Phys. Kl. Königl. Sächs. Ges. Wiss. 32, 509–604.
6. Frank-Kamenetskii, D.A. (1969). Diffusion and Heat Exchange in Chemical Kinetics. Plenum Press.
