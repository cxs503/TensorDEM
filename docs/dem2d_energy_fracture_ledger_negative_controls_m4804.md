# 2-D 能量/断裂台账负控的检出裕度（m4804）

日期：2026-11-06 · 分支：`agent/m4875` · 范围：`tests/`、`scripts/`、`docs/figures/`、
`artifacts/m4804_negative_control_margins/`

本页回答 `docs/dem2d_energy_fracture_ledger_negative_controls.md` 遗留的定量问题：
那 7 个「必须变红」的负控，**究竟能看到多大的故障**？负控只证明「门在某个给定的
注入幅度下亮红」，不证明这个幅度接近门的物理分辨极限。若注入幅度取到通道值的
数量级，即使门只有 ~10 % 的灵敏度也会通过，负控就退化成一个仪式。m4804 用**闭式
分辨率**把这件事变成可判定的量，并用独立二分法复算同一个边界。

## 1. 案例与物理背景

工况与正/负控完全一致：2-D 冰-工具压入（tool indentation），7×3 的
bonded-particle 冰梁，工具以恒定速度压入，运行到冰梁内多条 bond 断裂，所以
`fracture_release_J` 与 `pair_contact_acquired_J` 两个通道都非平凡。
`src/tensordem/dem.py` 的 `IceDEM.accounting` 逐时步累加 5 个能量通道，
平衡恒等式为

```
dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0
```

其中 `external_work_J` 在本工况**休眠**（不加外驱动，恒为 0），这正是它只能用
加性故障做负控的原因（乘性故障对 0 是空转）。船型几何量（Cb/L/B）在本单不出现：
这是求解器内核的台账身份与门禁灵敏度问题，不涉及船舶总体几何，故不触发几何前置
校验。工况参数（网格、步长、步数、断键数）与各通道清洁值全部由脚本从运行时
对象导出并写入下表，不在正文手抄。

## 2. 方法：把「门能看多小」写成闭式

门（`balance_check`）是台账通道的**线性**判据：

```
|r0 + s_ch * (x - x0)| <= tol * S0
```

- `r0`：清洁台账残差（恒等式的闭式零点附近）；
- `s_ch` ∈ {−1, +1}：该通道在恒等式里的符号；
- `x0`：该通道的清洁值；`S0`：清洁数据的参考能量尺度；
- `tol`：相对判据（与正控同一值）。

对 `x` 解这对线性不等式得到**接受带** `[x_lo, x_hi]`（`acceptance_band`），
分辨率就是 `x0` 到带最近边缘的距离：

```
加性故障（单位 J）      d* = tol*S0 - |r0|                     （故障即偏移量）
乘性故障（单位 |α-1|）  d* = (tol*S0 - |r0|) / |x0|             （故障即缩放因子）
```

无拟合参数、无自由常数；`|r0| -> 0` 时 `d* -> tol*S0`，即门的分辨率不能优于判据
预算的绝对份额。两个退化情形被显式建模而不是被忽略：

1. **休眠通道**（`x0 = 0`）：`min_detectable_fault(..., "multiplicative")` 返回
   `reachable=False`，因为「把 0 乘成别的数」在数学上不是一次注入——这正是历史
   缺陷（×−1 空转）的根源。
2. **清洁数据已在带外**（`tol*S0 < |r0|`）：台账在任何注入之前就不闭合，分辨率
   记为 0，`reason` 说明「已越界」，避免把这种情况误报成一个有限分辨率。

**交叉验证**：每条闭式边界都被 `bisect_min_detectable` 用纯二分法在门自己的
布尔输出上重新定位（二分不读闭式公式，只用闭式结果决定往哪一侧找），两者一致
到 1e-9 相对才算通过。此外在边界上取 `1 - 1e-9` 倍必须**通过**、`1 + 1e-6` 倍
必须**变红**，钉住「刀锋」位置。

## 3. 结果

下表由 `scripts/negative_controls_m4804.py` 注入（同一次运行写
`artifacts/m4804_negative_control_margins/margins.json`），正文无手抄数字；
重复运行字节稳定。

<!-- BEGIN:m4804-injected-table -->
| 通道 | 清洁值 $x_0$ (J) | 加性分辨率 $d^\*$ (J) | $d^\*/x_0$ | 乘性分辨率 $\lvert\alpha^\*-1\rvert$ | 二分复算 $d^\*$ (J) | 相对偏差 |
|---|---|---|---|---|---|---|
| `tool_work_J` | 1.031e-01 | 2.845e-04 | 2.76e-03 | 2.758e-03 | 2.845e-04 | 2.4e-14 |
| `external_work_J` | 0.000e+00 | 2.845e-04 | —（休眠） | —（休眠通道无乘性控制） | 2.845e-04 | 1.1e-15 |
| `damping_dissipation_J` | 1.266e-02 | 2.845e-04 | 2.25e-02 | 2.248e-02 | 2.845e-04 | 3.0e-15 |
| `fracture_release_J` | 1.330e-02 | 2.845e-04 | 2.14e-02 | 2.139e-02 | 2.845e-04 | 3.0e-15 |
| `pair_contact_acquired_J` | 4.799e-03 | 2.845e-04 | 5.93e-02 | 5.928e-02 | 2.845e-04 | 1.5e-15 |

判据裕度 $d^\* = \mathrm{tol}\cdot S_0 - \lvert r_0\rvert$ = 2.845e-04 J（tol = 0.005，$S_0$ = 1.291e-01 J，$r_0$ = 3.610e-04 J）。

| 已装负控故障 | 通道 | 模式 | 故障幅度 | 分辨率 | 裕度 $M$ | 检出 |
|---|---|---|---|---|---|---|
| 零化 | `fracture_release_J` | multiplicative | $\alpha$ = 0（$\Delta\alpha$ = 1） | 2.139e-02 | 46.7 | 是 |
| 三倍 | `fracture_release_J` | multiplicative | $\alpha$ = 3（$\Delta\alpha$ = 2） | 2.139e-02 | 93.5 | 是 |
| 零化 | `pair_contact_acquired_J` | multiplicative | $\alpha$ = 0（$\Delta\alpha$ = 1） | 5.928e-02 | 16.9 | 是 |
| 三倍 | `pair_contact_acquired_J` | multiplicative | $\alpha$ = 3（$\Delta\alpha$ = 2） | 5.928e-02 | 33.7 | 是 |
| 零化 | `tool_work_J` | multiplicative | $\alpha$ = 0（$\Delta\alpha$ = 1） | 2.758e-03 | 363 | 是 |
| 零化 | `damping_dissipation_J` | multiplicative | $\alpha$ = 0（$\Delta\alpha$ = 1） | 2.248e-02 | 44.5 | 是 |
| 加性 | `external_work_J` | additive | $+$1.031e-01 J | 2.845e-04 | 363 | 是 |
<!-- END:m4804-injected-table -->

读法：加性分辨率是「门能看见的最小偏移能量」；它相对通道值的比例说明该通道
本身的量级离门的分辨极限有多远。已装负控的裕度 `M = 故障幅度 / d*`，`M >= 1`
即负控确实超过了门的分辨极限；`M` 越大，该负控越不可能是「靠幅度取胜」的仪式。
休眠通道的加性负控用 `tool_work_J` 的量级作故障，因此它的 `M` 直接反映「以工具
做功量级注入到外驱动通道」的可检性。

## 4. 图

`docs/figures/m4804_negative_control_margins.png`
（由 `scripts/plot_m4804_ledger_negative_margins.py` 从上面的 JSON 生成）：

- **左panel**：7 个已装负控各自的裕度 `M`（对数横轴），竖线为 `M = 1`
  的分辨极限；柱长即「这个负控比门的最小可检故障强多少倍」。
- **右panel**：逐通道的加性分辨率（J，对数轴）与通道自身清洁值对比，用来判断
  哪个通道的门禁最"钝"（分辨率相对通道值比例最大），以及休眠通道的加性分辨率
  与活跃通道是否在同一量级。

## 5. 注意点与局限

- 分辨率只刻画**单通道、恒定幅度**的注入。相关故障（如两个通道同时按比例漂移，
  互相抵消）不在本页判据内；门的线性结构决定了这类故障可能被抵消，需要成对
  负控才能覆盖。
- 参考尺度 `S0` 固定为**清洁**数据（闭式解的前提）。已装门用的是污染数据的
  `S`，两者差一个 `O(故障/tol/S0)` 的小量；`test_scale_convention_gap_is_negligible`
  与 JSON 的 `max_data_scale_boundary_shift_rel` 量化了这个偏移，并把它限制在
  1 % 以内，故解析裕度对已装门仍是保守的（不会高估可检性）。
- 工况是脚本级规模（7×3、0.12 s 量级），非工程尺度；结论限于**台账身份与门禁
  灵敏度**，不外推到工程结构的断裂预报精度。
- 负控与分辨率的"绿"只证明门对**注入的通道故障**敏感；不证明台账物理完备
  （未计摩擦/滚阻/接触阻尼以外的其它耗散通道）。

## 6. 复现

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
    tests/test_dem2d_energy_fracture_ledger_negative.py -q
# 29 passed

python3 scripts/negative_controls_m4804.py        # JSON + 日志 + 本文表格注入
python3 scripts/plot_m4804_ledger_negative_margins.py   # 图
```

## 7. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3 % |
|---|---|---|---|
| 清洁台账相对残差 | 0（恒等式闭式） | 见上表 `r0/S0`（正控同一判据 tol 下通过） | 是 |
| 闭式加性分辨率 vs 二分复算 | 二分边界（独立算法） | `max_bisection_gap_rel`（机器精度级） | 是 |
| 闭式乘性分辨率 vs `(tol*S0-\|r0\|)/\|x0\|` | 代数恒等式 | 1e-9 相对（测试断言） | 是 |
| 已装负控裕度 | 1（门的分辨极限） | 见上表 `M`，全部 > 1 | 是（最小裕度见 JSON `min_control_margin`） |
| 尺度约定（清洁 vs 污染 `S`）对边界的影响 | 无偏移 | `max_data_scale_boundary_shift_rel` < 1 % | 是 |

上表的数值以 `artifacts/m4804_negative_control_margins/margins.json` 为唯一来源；
本文不重复抄写任何由脚本导出的数字。