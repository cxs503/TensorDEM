# CASE-SILO-JANSSEN - 筒仓壁压分布 vs Janssen 解析（TensorDEM）

任务 m15265 · 算例池第 1 轮 · 目标容差 <15% · 全部数字由 `benchmarks/silo_janssen_wall_pressure.py` 程序化注入（零手抄）

## 1. 问题与 ref 出处

深筒仓静置颗粒床的壁面压力并不随深度线性增长，而是在特征深度 `z0 = R/(2 mu_w K)` 后趋于饱和 `P_inf = rho_b g R/(2 mu_w K)`：

```
sigma_h(z) = K sigma_v(z),  sigma_v(z) = P_inf (1 - exp(-z/z0))
```

ref 出处：H. A. Janssen, *Versuche über Getreidedruck in Silozellen*, Zeitschrift des Vereines deutscher Ingenieure **39** (1895) 1045-1049; English translation/commentary: M. Sperl, *Experiments on corn pressure in silo cells - translation and comment of Janssen's paper from 1895*, Granular Matter **8** (2006) 59-65, DOI 10.1007/s10035-005-0224-z.

所有 ref 数值由代码按上式在**声明参数 + 实测堆积密度 rho_b** 上求值得到；Janssen 1895 原始实测表（Sperl 2006 全文）在本沙箱不可得（Springer paywall），该表数据登记为 **WITHHELD**，不以二手数字替代。

参数无关的自证恒等式：`P_inf / z0 = rho_b g`（比重量 N/m^3；注意不是 `rho_b g R`）。本单顺带修正了 `silo_flow/janssen.py` 中把该恒等式写成 `P_inf/z0 = rho_b g R` 的量纲错误（旧口径导致 identity 误差恒为 ~1/R 倍）。

## 2. 方法与实现

复用 TensorDEM 现有能力 `tensordem.silo_flow`（Cundall-Strack 线性弹簧-阻尼 + 壁面摩擦 + 每步能量账本），未另起炉灶：`JanssenCase` 装填 -> 静置 -> 两层独立测量（壁面法向力/面积 slab、切片力平衡 sigma_v）-> `scipy.optimize.curve_fit` 拟合 Janssen 形式 -> 与两条 ref 路线对照：

* `self_consistent`：用**同一次运行实测**的 `K_meas` 构造 Janssen（检查形状与动量平衡，不检验 K 本身）；
* `apriori`：用 Jaky 估计 `K_J = 1 - sin(atan(mu))`（唯一绝对压力预测路线，报告但不进入验收门）。

仿真入口：`python3 benchmarks/silo_janssen_wall_pressure.py --case <name> --out runs/m15265/sim/<name>.json`；后处理：`--analyze runs/m15265`。

## 3. 参数表（无量纲数完整）

| 参数 | 符号 | 基准值 | 无量纲 | 敏感性取值 |
|---|---|---|---|---|
| 筒仓半径 | R | 80 mm | H/R = 3.75 | deep: 45 mm 填高 -> H/R = 5.63 |
| 填高 | H | 300 mm | R/d = 17.8 | coarse: d=6 mm -> R/d = 13.3 |
| 颗粒直径 | d | 4.5 mm | d/R = 0.056 | |
| 出料口半径 | r_o | 10 mm | r_o/R = 0.125 | |
| 壁摩擦 | mu_w | 0.40 | - | 0.25 / 0.60 |
| 颗粒摩擦 | mu | 0.40 | phi = 21.8 deg | 0.25 |
| 法向刚度 | kn | 1e4 N/m | kn/(rho g d^3) - 见表 | |
| 恢复系数 | e | 0.3 | - | |
| 密度 | rho_p | 2500 kg/m^3 | - | |

各算例实测体积分数/堆积密度与颗粒数见第 4 节表（N, rho_b 列）。

## 4. 对照表（ref vs got，误差打勾/叉，判据 <15%）

| case | N | rho_b | K_meas | P_inf ref [Pa] | P_inf got [Pa] | err % | z0 ref [mm] | z0 got [mm] | err % | identity err % | base err % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coarse | 31142 | 1605 | 1.435 | 1097 | 5.344e+07 | 4.869e+04 | 0.06969 | 3035 | 4.355e+04 | 0.1181 | 2.453 | False |

（`pass`=True 记为 ✅，False 记为 ❌；err 列单位为 1，表中按原值显示，换算百分数见 comparison.json。）

## 5. 收敛性/参数敏感性

* 壁摩擦 mu_w：0.25/0.40/0.60 三档，Janssen 预言 z0 与 P_inf 均 ∝ 1/mu_w；实测 z0 列： m。
* 分辨率 R/d：base(17.8) vs coarse(13.3)，看 N 与误差是否同向漂移。
* 深度 H/z0：base(3.75) vs deep(5.63)，检验饱和段是否出现平台。
* 随机种子：base vs seed7 给出指标散度（重复性）。
* 层数/拟合窗口：见 per-case `profile` 里 above-floor 子集与最差 slab。

## 6. 失败模式、工件与生态摩擦

* 近表面 slab 的预测应力 ~0，无 floor 的相对误差会被放大（m7751 曾报 worst slab 28.7%）；本单预注册了 `0.05 P_inf` 绝对下限，同时如实保留未截断的最差 slab 数。
* 软刚度 kn=1e4 N/m 下底板反力含重叠偏置 -> base pressure 单独一条判据。
* 生态摩擦：平台 knowledge_hub 引擎 8021 未起（`/tensor/knowledge/search` 返 engine_offline），开工三查的 capability 检索只能退化为仓内 grep；psycopg2 在 sandbox 为 stub、`psql` 直连 knowledge_base 被拒（permission denied）；集群队列 250+ 深度排队，本单因此启用本地 setsid 兜底并在报告中标注。


## 7. 判定（honest FAIL）与缺陷诊断（本单实际跑出的唯一完成算例：coarse）

**判据：<15% → 结论：本波 FAIL（honest-FAIL，不粉饰）。**

| 量 | ref（Janssen, K_meas=1.435） | got（DEM 拟合/实测） | 误差 | <15%? |
|---|---|---|---|---|
| 壁压饱和幅值 P_inf | 1097.5 Pa | 5.34e+07 Pa | 4.87e+04× | ✗ |
| 特征深度 z0 | 69.7 mm | 3.03e+06 mm | 4.35e+04× | ✗ |
| 参数无关恒等式 P_inf/z0 vs rho_b g | 15748 N/m^3 | 17607 N/m^3 | 11.8% | ✓ |
| 底板反力 vs P_inf | 1097.5 Pa | 3790.0 Pa | 245% | ✗ |
| 拟合质量 R^2 | 1 | 0.321 | - | ✗ |

**三个根因（按证据强度排序）**

1. **拟合退化（机制级）**：Janssen 的 P_inf 与 z0 只在 `z >> z0` 段可辨识；本算例
   `z0_ref=70 mm` 与填高 H=300 mm 同量级（H/z0≈4.3），
   壁应力 slab 序列近线性（sigma_w 从 0 单调升到 `15280.9` Pa 未见平台），
   `scipy.optimize.curve_fit` 在无上界约束下把 (P_inf, z0) 推到 (5.34e+07 Pa, 3.03e+03 m)
   的平坦极小值，R^2=0.32。**这是 `silo_flow/janssen.py` 现有验收口径的缺陷：
   没有 `H/z0 >= 3` 前置条件检查，也没有对退化拟合的拒绝判据。**
2. **K 归约缺陷（归约级）**：`k_measured = mean(sigma_w/sigma_v)` 直接对全部层求平均，
   近表面层 sigma_v->0 时比值发散，本算例给出 K_meas=1.435（物理上应为 0.4~0.6），
   使 self_consistent 参考系本身失真。
3. **量纲缺陷（已修复）**：库内恒等式原写作 `P_inf/z0 = rho_b g R`，正确为比重量
   `P_inf/z0 = rho_b g`（差 1/R≈12.5 倍）。本单已修 `src/tensordem/silo_flow/janssen.py`
   并在本次运行上验证：修正后恒等式误差 11.8%（旧的错误口径会报 1298%）。

**未完成的部分（如实申报）**
* `base`（锚点）/`seed7`/`muw025`/`muw060`/`deep`/`smooth` 六例：已递交集群（cj1029/cj1038–cj1044），
  队列深度 250+，35 min 内未派发；本地 setsid 兜底跑 base/muw060 至收尾时仍未完成（base 需 ~7.6e4 颗粒，
  本地 load≈283 下 ETA >40 min）。因此 base 锚点判据本轮 **NOT_RUN**，不是 PASS。
* 敏感性轴（mu_w/深度/分辨率/种子）本轮只有 coarse(R/d=13.3) 一个点，**不足以支撑敏感性结论**（WITHHELD）。
* `repro_cmd` 见 `runs/m15265/repro.sh`；`ref` 的 Janssen 1895 原始实测表 **WITHHELD**（paywall）。

**下一轮（接棒）应做的最小修复**：在 `run_janssen` 增加 (a) 退化拒绝（R^2<0.9 或 z0>H/2 时不出 P_inf）、
(b) K_meas 用深部层（sigma_v > 0.2 P_inf）加稳健中位数、(c) 前置断言 `H/z0_expected >= 3`
（即 H >= 3R/(2 mu_w K)），并把 base 算例填高抬到 H=0.6 m（N≈1.5e5，上集群分片）。
