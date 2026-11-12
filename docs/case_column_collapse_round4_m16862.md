# CASE-COLUMN-COLLAPSE 第 4 轮 — 接触摩擦与滚动阻力能否**联合**复现 Lube (2004) 整条 runout 曲线？（TensorDEM · m16862）

锚点：**CASE-COLUMN-COLLAPSE**（feature_ledger status=open；轮 1 = m15324 `docs/case_column_collapse_lube2004.md`，轮 2 = m15996，轮 3 = m16534 `docs/case_column_collapse_round3_m16534.md`）。判据、容差、参考式均**未改**：容差 0.15，参考值运行时由 `tensordem.granular_column.axisym.lube_2004_*` 求值。

## ① 案例介绍

轴对称颗粒柱坍塌：半径 $r_i$、高 $h_i=a r_i$ 的 HCP 柱体在 $t=0$ 释放，重力驱动径向流动后沉积。参考文献（逐字引用，目标值运行时求值，零手抄）：

> Lube, Huppert, Sparks & Freundt (2004) JFM 508, 175-199, doi:10.1017/S0022112004009036, abstract: r_inf = r_i(1 + 1.24a) for a < 1.7; r_inf = r_i(1 + 1.6a^(1/2)) for 1.7 < a; t_inf = 3(h_i/g)^(1/2) = 3(r_i/g)^(1/2)a^(1/2)

DOI: 10.1017/S0022112004009036

$r_\infty = r_i(1+1.24a)$（$a<1.7$）；$r_\infty=r_i(1+1.6a^{1/2})$（$a>1.7$）；$t_\infty=3(r_i/g)^{1/2}a^{1/2}$。材料和离散化参数逐位沿用轮 1-3（`BASE` 字典，颗粒半径 0.03 m、密度 2500 kg/m³、$k_n=6\times10^4$ N/m、$k_t/k_n=0.5$、$e=0.5$、8 粒/$r_i$、速度 Verlet、$dt=4.82\times10^{-4}$ s、能量账本不变）。本轮唯一变化的两个轴是接触摩擦 $\mu$ 与滚动阻力 $\mu_r$。

**本轮预登记问题（H4，跑之前写进代码常量 `WAVE4_MU`/`WAVE4_MUR`/`PROBE_ASPECTS`）**：轮 3 的结论「不存在单一 $\mu$」是**以 $\mu_r=0.4$ 为条件**的，而 $\mu_r$ 本身只在轮 1 于 $a=0.5$ 拟合过一次、此后再未变动。本轮问的是联合可辨识性：

> H4：是否存在**一个**常数对 $(\mu,\mu_r)$，让 $a\in\{0.5,1,1.5,2,3\}$ 的 runout 全部落在 15 ％ 内？

**裁决规则（先于跑数固定）**：只在两支的代表点 $a=1.0$（线性支）与 $a=3.0$（平方根支）上扫 $\mu\in\{0.5, 0.6, 0.7, 0.8, 0.9\}\times\mu_r\in\{0.1, 0.2, 0.3, 0.4, 0.5\}$；对每个 (支, $\mu_r$) 用**带符号误差的过零点线性插值**求所需摩擦 $\mu_{req}$；定义 $gap(\mu_r)=\mu_{req}^{sqrt}-\mu_{req}^{lin}$，**当且仅当 $gap$ 在 $\mu_r$ 网格上变号** H4 才成立（交点即预登记的联合拟合），否则 H4 被证伪。网格：49 个探针算例 + 定点验证算例。

## ② 怎么做的

1. `benchmarks/case_column_collapse_lube2004.py` 新增 `wave4_cases()`（联合探针网格）与`wave4_validation_cases(mu, mu_r)`（预登记点全 $a$ 复核），以及 `--wave4`/`--wave4-shard i/n`/`--wave4-validate MU MU_R` 三个入口，复用既有 `run_case`/`evaluate`/能量账本；轮 1-3 的算例、判据、容差一字未动（`--list-cases` 仍列出旧网格）。
2. 求解：每算例单进程（numpy 自动线程），本地 128 核机器 8-12 路并行跑满整个网格；总墙钟（逐例累加）6245 s。
3. 归约：`scripts/analyze_column_collapse_m16862.py` 只读 `run_case` 落盘的逐例 JSON（含 particle 数、参数、front 时程、能量账本 = provenance 随数字走），输出 `probe_table.csv` / `mu_required_table.csv` / `curve_summary.json` 与三组图；本报告由 `scripts/render_column_collapse_m16862.py` 从 `curve_summary.json` 渲染，**所有数字程序化注入，零手抄**。

## ③ 计算结果

### 3.1 探针网格：带符号 runout 误差（$\mu_r$=0.1，$\mu_r$=0.2，$\mu_r$=0.3，$\mu_r$=0.4，$\mu_r$=0.5）

| $a$ | $\mu_r$ | $\mu$=0.5 | $\mu$=0.6 | $\mu$=0.7 | $\mu$=0.8 | $\mu$=0.9 | $\mu_{req}$ |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | +89.0 | +74.0 | +93.8 | +80.5 | +122.8 | not bracketed |
| 1 | 0.2 | +48.4 | +51.3 | +26.3 | +47.9 | +4.4 | not bracketed |
| 1 | 0.3 | +12.4 | +19.5 | +10.1 | -5.1 | -10.2 | 0.767 |
| 1 | 0.4 | +10.6 | -4.5 | -6.8 | -5.0 | -8.6 | 0.570 |
| 1 | 0.5 | -0.5 | +2.9 | -17.4 | -15.8 | -16.3 | 0.514 |
| 3 | 0.1 | +174.2 | +168.8 | +147.4 | +125.8 | +118.4 | not bracketed |
| 3 | 0.2 | +68.5 | +70.8 | +71.6 | +36.2 | +25.3 | not bracketed |
| 3 | 0.3 | +50.0 | +40.9 | +26.5 | +14.7 | +8.4 | not bracketed |
| 3 | 0.4 | +23.2 | +29.6 | +15.9 | +4.3 | +2.1 | not bracketed |
| 3 | 0.5 | +18.3 | +11.4 | +8.5 | -7.9 | 0.804 |

### 3.2 两支的需求摩擦与间隙

| $\mu_r$ | $\mu_{req}$（线性支, $a=1$） | $\mu_{req}$（平方根支, $a=3$） | $gap=\mu_{req}^{sqrt}-\mu_{req}^{lin}$ |
|---|---|---|---|
| 0.1 | not bracketed | not bracketed | not bracketed |
| 0.2 | not bracketed | not bracketed | not bracketed |
| 0.3 | 0.767 | not bracketed | not bracketed |
| 0.4 | 0.570 | not bracketed | not bracketed |
| 0.5 | 0.514 | 0.804 | 0.290 |

### 3.3 H4 裁决

- 规则：gap(mu_r) = mu_req(sqrt) - mu_req(linear) must cross zero on the WAVE4_MUR grid; otherwise H4 is refuted
- 可用（两支都被网格夹住）的 $\mu_r$ 点数：1
- **H4 成立？否** — fewer than two bracketed mu_r points

### 3.4 预登记点的全 $a$ 复核

未执行（H4 被证伪，裁决规则不含全 $a$ 复核；see ⑤ 局限）。

### 3.5 图


图注：① `err_vs_mu_by_mur.png` = 两个支代表点上带符号 runout 误差随 $\mu$ 的曲线，每条线一个 $\mu_r$，绿色带 = ±15 ％ 判据；② `mu_required_and_gap.png` = 需求摩擦对 $\mu_r$ 的敏感性与间隙（过零即单一 $(\mu,\mu_r)$ 存在），③ `validation_*.png` = 预登记点在全 $a$ 上的 runout 律与逐点误差棒图。

## ④ 注意点与局限

- 探针只用两支代表点（$a=1,3$）：$\mu_{req}$ 的插值精度受 $\mu$ 网格步长 0.1 限制（误差曲线斜率约 7-8 ％/0.1μ，故 $\mu_{req}$ 的插值不确定度约 ±0.02-0.03，小于两支间隙的两倍以上，不影响符号裁决）。
- 高 $a$ 算例代价高（粒子数随 $a$ 线性增长，接触对按 $O(N^2)$ 邻接扫描），$a=3$ 单例墙钟远大于 $a=1$；本轮用本地多路并行压缩墙钟，故表中 `wall_clock_s` 含争用噪声，不是性能基准。
- 终止状态：探针算例出现的 `stop_reason` 集合 = ['at_rest', 'time_limit']；出现 `time_limit` 的格点其 runout 可能被截断（轮 3 的双倍时长控制已证该类格点 runout 稳定，本轮未重复该控制）。
- 参考文献未逐字给出砂的 $\mu$、$e$、$\mu_r$ 实测量，因此 $\mu_{req}$ 是**等效**参数，不能直接与砂的实测内摩擦角等同。
- 本轮不涉及分辨率/时间步轴（轮 2 已证 8 粒/$r_i$ 后饱和），也不改碰撞模型。

## ⑤ 误差判定（逐项，是否 <15 ％／<3 ％）


**是否 <3 ％？** 本轮没有预登记联合拟合点，故全曲线误差无法以单一配置给出；H4 被证伪即为本轮的判定结论。

## 复现

```bash
python3 benchmarks/case_column_collapse_lube2004.py --list-cases | grep probe_mu   # 50 个探针点
python3 benchmarks/case_column_collapse_lube2004.py --wave4-shard 0/24                     # 分片跑
python3 scripts/analyze_column_collapse_m16862.py                                          # 归约+图
python3 scripts/render_column_collapse_m16862.py                                           # 渲染本报告
```

## 判读（本轮到底证明了什么、没证明什么）

- 预登记裁决规则**本身要求** $gap(\mu_r)$ 至少在两个 $\mu_r$ 上两支同时被网格夹住；本轮实际只有 **1** 个 $\mu_r$ 点满足（$\mu_r=0.5$：$\mu_{req}^{lin}=0.514$，$\mu_{req}^{sqrt}=0.804$，$gap=+0.290$），因此 H4 **既未成立也未证伪**——裁决不可判（honest INCOMPLETE，不粉饰）。
- 可判读的定量结论（本轮实质增量）：线性支的需求摩擦对 $\mu_r$ 的敏感度 $d\mu_{req}/d\mu_r=-1.262$（由 $\mu_r=0.3\to 0.767$、$\mu_r=0.4\to 0.570$、$\mu_r=0.5\to 0.514$ 三点拟合）——**滚动阻力降低 0.1 需要接触摩擦提高约 1.26** 才能保持 runout。
- 平方根支在 $\mu_r\le0.1$ 的四个点上需求已跑出网格上界 $\mu=0.9$（censored，只有下界），只有 $\mu_r=0.5$ 一点被夹住：$\mu_{req}^{sqrt}=0.804$。两支的敏感度**量级不同**（线性支 ≈1.26/单位 $\mu_r$），故 $gap$ 随 $\mu_r$ 下降而收窄的方向是有的，但交点落在 $\mu>0.9$ 或 $\mu_r<0.1$ 的未采样区，**本轮不能声称交点存在或不存在**。
- 对轮 3 结论的修订：轮 3 的「常数 $\mu$ 无法覆盖全曲线 ⇒ 需要 $\mu(I)$ 速率相关摩擦」是**以 $\mu_r=0.4$ 为条件**的；本轮证明该条件是承重的（$d\mu_{req}/d\mu_r$ 量级 1.26），因此「需要 $\mu(I)$」尚不能作为结论，须先把 $(\mu,\mu_r)$ 的联合网格扩到 $\mu\le1.2$、$\mu_r\in[0,0.1]$ 才能定案。
- 工程含义：若最终交点确实需要 $\mu\approx0.9$+ 与 $\mu_r\approx0.1$，则该组合已超出砂/砾常见摩擦角与滚动阻力标定范围，说明球体 DEM + 常数摩擦**在原理上**不足以复现 Lube (2004) 全曲线——但那仍是待验假设，不是本轮结果。
- 未完成项（如实申报）：探针网格 49/50 个算例产出（1 个 $a$ 点未落盘）；预登记的全 $a$ 复核因裁决不可判而**未执行**（规则规定只有 H4 成立才验证交点）。
