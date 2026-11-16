# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律：seed 系综不确定度预算（m22657）

**锚点**：`CASE-DEM-SILO-BEVERLOO`（验收：`|W − W_ref|/W_ref < 10 %`；ref: ）

**本单**：m22657（tp-glm52-2，generalist 席，自找活·验证欠账）　**日期**：2026-11-16
**集群作业**：cj4285（6 分片，stage_dir 直达节点）　**代码**：`benchmarks/m22657_beverloo_ensemble/`

> **定位**：该锚点此前八轮（m6296…m22628）的判定**每一条都只用一个 seed**，从未报告过 `W` 自身的可复现散布。本单不重跑已有算例、不复核已有 PASS/FAIL，而是补上唯一没人测过的量——**同一构型下换堆积的散布**——并用区间（而非点值）裁决总账容差，同时给出该算例的**计数噪声下限**（解析可算，不依赖任何 run）。

## ① 案例介绍（物理背景 / 几何 / 参数）

平底圆柱料仓底部圆心圆孔，颗粒重力自由流出，稳态质量流率服从 Beverloo 律

```
W = C · ρ_b · sqrt(g) · (D − k·d)^(5/2),   C = 0.58,  k = 1.4  (球状玻璃珠)
```

本单冻结构型（`run_case.py:FROZEN`，见 `prereg.md` §3；全部逐 seed 相同）：

| 量 | 值 |
|---|---|
| 颗粒直径 d | 3.0 mm |
| 孔口直径 D | 15.0 mm（D/d = 5.00） |
| 仓半径 | 24.0 mm（silo_D/D = 3.20） |
| 填充高 H | 36.0 mm（H/D = 2.40） |
| 摩擦 μ / 滚阻 μ_r / 恢复 e | 0.50 / 0.05 / 0.30 |
| 法向刚度 kn / kt/kn | 5000 N/m / 0.5 |
| 壁摩擦 μ_wall | 0（平底板，Beverloo 光滑壁条件） |
| 重力 g | 9.81 m/s² |
| settle / discard | 0.06 s / 0.12 s（settle 阶段孔口封堵） |
| 测量窗口 | 8 × 0.05 s = 0.40 s |
| seed 集合 | 1–6（独立无序堆积） |

无量纲域校验：`D/d = 5.00 ≥ 4`、`silo_D/D = 3.20 ≥ 2.5`、`H/D = 2.40 ≥ 1.5` → in_domain = **True**（由 `run_case.dimensionless` 计算并被测试钉住）。

**为什么是这个构型**：`d`、`μ`、`kn`、`μ_wall` 与 settle/discard 协议沿用战役冻结值；仓体按 24/36 mm 等比缩小，只为把单算例压到 ≈2.5 k 颗粒、六个独立堆积能同时进队列。判定不依赖绝对尺寸——`W_ref` 用的是该算例自己测出的 `ρ_b`。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **总账核对**：`feature_ledger` 中 `CASE-DEM-SILO-BEVERLOO` 为 `open`，可认领；锚点由派发器指派，未改写。
2. **防重复**：八轮既有产出（`benchmarks/m20055…m22628_beverloo*`、`docs/CASE-DEM-SILO-BEVERLOO_*`）全部读了标题与判定口径；m22628 已预注册 `D50_s2` 单点 seed 复算但其 run 未回传（NOT_RUN）。本单因此**不重做** D/d ∈ {4,5,6} 拟合与 k 反演，只做「6 seed 散布 + 区间裁决 + 噪声下限」。
3. **先冻结再跑**：`prereg.md` 在任何 run 回传前写定构型、估计式、四类判定（V1–V4）与预测 P1–P5；统计量在 `tests/test_m22657_beverloo_ensemble.py`（24 项）中被合成数据钉住，包括 PASS / FAIL / NOT_DECIDABLE / NOT_RUN 四个分支与 t 分位数查表值。
4. **算例**：`python3 benchmarks/m22657_beverloo_ensemble/run_case.py --seed i`（薄封装调用 `python -m tensordem.silo_flow.study beverloo`，即战役同一个 Cundall–Strack 内核，不重写算例、不派生 JSON schema）。
5. **递交**：`run_compute submit` 6 分片（seed 1–6），`stage_dir` 指向本工作树，每分片 2 核、超时 7200 s。
6. **判定**：`analyse.py` 从 6 份 JSON 重算 `W_s`、`W_ref`、`rel_s`、系综均值/标准差/Student-t 95 % CI、`C_recovered`、窗口漂移、计数噪声预算，写 `runs/m22657_beverloo_ensemble/summary.json`；本文件由 `render_doc.py` 从该 JSON 生成（零手抄）。

**估计式（冻结，§4）**：

```
W_s   = mean(last 50 % windows)            # 沿用战役口径
W_ref = C_lit ρ_b sqrt(g) (D − k_lit d)^2.5   # C=0.58, k=1.4，不重拟合
W_ens = mean_s W_s ,  CI95 = W_ens ± t(0.975, 5)·sd/sqrt(6)
```

## ③ 计算结果

**NOT_RUN**——6 份 `seed*.json` 未回传，本节不填任何数字，也不做任何精度声明。

可报告的只有解析结论：计数噪声下限（见 ⑤）。

## ④ 结果图

`docs/figures/m22657_ensemble_errors.png`（单 seed 散布 + 系综 95 % CI 与 ±10 % 带）、`docs/figures/m22657_window_history.png`（逐窗口流率历史与 Beverloo 参考线）。两图均由 `analyse.py --figures` 从 summary 直接生成。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **单 seed 判定不成立**：测量窗口内流出的颗粒数 N 决定噪声地板 `1/√N`；本构型 N ≈ 340 → 5.4 %。这意味着总账 10 % 容差在**单 seed** 上只剩不到 2σ 的判别余量，此前各轮的 −6.9 % / +5.9 % 与噪声同量级，单点 PASS 不可外推。
2. **窗口并非严格稳态**：床面下降使流率带漂移，本单用窗口线性回归的归一化漂移量（V3）量化，漂移方向偏「低 W」。
3. **`ρ_b` 口径**：参考值用算例自测 `ρ_b`（`mass / volume_below(fill_height)`），若床面沉降低于标称 `fill_height`，分母含空头 → `ρ_b` 偏低 → `W_ref` 偏低；V4 用于确认这不是本构型的混淆源。
4. **范围**：结论只对该冻结构型（D/d=5、光滑平底、单分散）成立，不外推到锥斗/壁摩擦/多分散。
5. **旁证缺陷（未修，按最小改动纪律只报告）**：`tensordem/silo_flow/flow_pattern.py::run_hopper` 里 `speed_sum` 分配后从未累加，实际传给 `classify_flow` 的是**末步瞬时速度**，与 docstring 声明的「时间平均速度」不符；同一函数 `flow_rate_kg_s` 又把 `mass_exited` 除以 `measure_time + startup_time`，与 `beverloo.run_discharge` 的窗口口径不同源。两处都会影响 hopper 的 mass/funnel 分类，不影响本单 Beverloo 判定。

## ⑥ 误差判定（逐项 vs 基准，显式回答 <3 %？）

| 判定 | 结果 | 依据 |
|---|---|---|
| V1 总账容差（<10 %，区间裁决） | **NOT_RUN** | 系综点估计与 95 % CI |
| V2 可复现性（CV < 10 %） | **NOT_RUN** | seed 间标准差 |
| V3 稳态（漂移 < 10 %） | **NOT_RUN** | 窗口回归 |
| V4 ρ_b 控制（散布 < 3 %） | **NOT_RUN** | 系综 ρ_b 极差 |

**是否 < 3 %？** 无法回答——无 run 回传，不给精度声明。可解析回答的是噪声地板 ≈5.4 %（N≈340 颗粒），即**该构型本质上不可能达到 3 % 级**；若要 3 % 级需把测量窗口内流出颗粒数提到 ≈1100（等价于窗口加长或床加大到约 3 倍）。

