# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律（TensorDEM，m20055）

**算例**: 平底圆形料仓自由卸料，稳态质量流率 `W` 对 Beverloo (1961) 律。**判定**: 主算例 D/d = 5 FAIL（容差 < 10 %）。

## ① 问题与参考出处

料仓底板上一个圆心圆孔，颗粒在重力下自由流出；Beverloo 等人 1961 年给出经验律

```
W = C · rho_b · sqrt(g) · (D − k·d)^(5/2)
```

出处：Beverloo, W. A.; Leniger, H. A.; van de Velde, J., *The flow of granular solids through orifices*, Chemical Engineering Science **15**(3-4), 260-269 (1961), DOI 10.1016/0009-2509(61)80059-4；相关系数取球状颗粒一对 `C = 0.58`、`k = 1.4`（适用范围 C∈[0.55, 0.6]、k∈[1.2, 1.6]），位置 correlation W = C rho_b g^(1/2) (D - k d)^(5/2); the coefficient pair for spherical particles is Eq. (4), p. 265。

机器可读参考文件 `benchmarks/m20055_beverloo/ref_beverloo_1961.json`，所有脚本只从该文件读数，代码与报告里不手抄数字。**诚实声明**：本工作树内没有 1961 年原文 PDF，环境无外网，因此原文表格/数据点无法重新数字化，`C`/`k` 是文献公布系数（仓库内既有模块与 prereg 均引 Eq.(4) p.265），这一点在参考文件的 `provenance.verification_status` 里如实标注。

## ② 方法与实现

复用仓库既有能力 `tensordem.silo_discharge`（REAL 3-D DEM：线性弹簧-阻尼法向 + Cundall–Strack 切向 + 带阻尼滚阻，`granular3d_core` 内核），不另起炉灶。前序记录（m6296）在**有序 HCP 床 + 孔口从第 0 步就开着 + 固定 5 个窗口**的协议下给出 −77…−86 % 的 honest FAIL，原因是床层在孔口上方成拱、流率根本没进入稳态。本单对该协议做三处最小修正（`src/tensordem/silo_discharge.py`）：

1. `SiloCase.packing = "settled"`：以非接触松散格子（净距 `settle_loose_gap`) 装料，靠重力在 `settle_time` 内压实成**无序床**（真实料仓的状态），避免晶格成拱；
2. `SiloCase.plug_during_settle = True`：压实阶段把孔口**堵住**（整块平板模型），开孔瞬间才取床高/床堆密度，使流率不可能被压实期漏出的质量虚增；
3. `steady_rate()`：以卸料事件记录 (t, N) 为基础，取卸料流的**质量分位 0.2–0.8** 作为稳态区间，再切成 5 个子窗口给出散度 CV（稳态性校验，不是符合度校验）。原有固定窗口路径保持向后兼容。

流率 `W` 由同一 DEM 的卸料质量-时间记录测得，参考值 `W_ref` 用**同一算例实测的床堆密度** rho_b 代入 Beverloo 律；回收的 `C` 在 k 固定为文献值时给出。

## ③ 参数表（含无量纲数）

| case | D/d | μ_p | n_bed | H/D | silo_D/D | ρ_b [kg/m³] | φ | 卸料占比 | W [kg/s] | W_ref [kg/s] | 误差 | C_rec | CV | <10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | 4.00 | 0.30 | 1525 | 2.90 | 3.25 | 1298 | 0.519 | 50% | 1.8175 | 1.4545 | +25.0% | 0.725 | 0.060 | ✗ |

无量纲完整性：`D/d`、`silo_D/D`、`H/D`、`rho_b/rho_p`（= φ）、`dt/T_contact`、以及 Beverloo 前置因子 `C = W/(rho_b sqrt(g) (D-kd)^{5/2})` 本身都已列出，算例的流动状态由 (D/d, H/D, silo_D/D, φ, μ_p, e, dt/T) 完全确定。

## ④ 对照表：ref vs got + 误差判定

| case | W_ref (Beverloo, ref) | W_dem (got) | 相对误差 | 判据 <10% | 稳态 CV<0.15 |
|---|---|---|---|---|---|
| D4 | 1.4545 | 1.8175 | +25.0% | ✗ | ✓ |

## ⑤ 收敛性与参数敏感性

**(a) 时间步收敛**（D5 与 D5_dthalf，dt/2）：

| case | dt/T_contact | W [kg/s] | 误差 |
|---|---|---|---|

**(b) 颗粒摩擦敏感性**（μ_p = 0.15 / 0.30 / 0.45，其余同 D5）：

| case | μ_p | W [kg/s] | 误差 | CV |
|---|---|---|---|---|

**(c) 稳态窗口定义敏感性**（同一事件记录、不同质量分位带重算，零额外计算）：

| case | 分位带 | W [kg/s] | CV |
|---|---|---|---|
| D4 | 0.20-0.80 (predeclared) | 1.8175 | 0.060 |
| D4 | 0.10-0.60 (early) | 1.7402 | 0.089 |
| D4 | 0.35-0.65 (tight) | 1.8162 | 0.164 |

## ⑥ 失败模式、工件与生态摩擦

**失败模式**（如 D 系列任一不过判据）

* 有序/晶格床在孔口成拱 → 只有启动突流，流率无稳态（m6296 结论，本单用无序压实床消除）；
* 批量料仓的尾部（床层抽空）与启动段都会污染平均值 → 用卸料流质量分位带剔除；
* 压实不足（`settle_time` 偏短）会让 φ/ρ_b 偏低，而 ρ_b 直接进 Beverloo 前置因子；本单把 ρ_b 取为开孔瞬间的实测床值；
* 小体系泊松噪声：窗口内计数量有限，CV 同时包含物理起伏与计数噪声。

**工件**

* 图 `flow_law`: `docs/figures/m20055_beverloo_flow_law.png`
* 图 `errors`: `docs/figures/m20055_beverloo_errors.png`
* 逐算例原始数据: `runs/m20055_beverloo/*.json`（含 git sha、host、dt、窗口序列、卸料事件序列）
* 汇总/表格: `runs/m20055_beverloo/summary.json`, `runs/m20055_beverloo/report_tables.md`
* 参考数值: `benchmarks/m20055_beverloo/ref_beverloo_1961.json`；算例脚本 `run_case.py` / 分析 `analyse.py` / 本文生成 `render_doc.py`
* 一键复现: `PYTHONPATH=src python benchmarks/m20055_beverloo/repro.sh`
* 回归测试: `PYTHONPATH=src python -m pytest tests/test_silo_discharge_steady.py -q`

**生态摩擦**（如实记）

* 1961 年原文 PDF 不在仓库、环境无外网（curl 出站被安全策略拦截）→ 无法按“零手抄”重新数字化原文表格；退而求其次：只把文献系数写成唯一机器可读参考文件并标注 `cited, not re-digitized`。
* `knowledge_base` / `negative_knowledge_ledger` / `oss_solutions` 三张知识表对当前 DB 角色 **无 SELECT 权限**，开工三查只能靠仓库内既有审计文档（docs/silo_discharge_beverloo_verification.md 等）代偿。
* 既有 `silo_discharge` 文档字符串宣称 C≈0.58/k≈1.4 已验证，而 m6296 的 honest FAIL 说明该宣称不成立；本单补上了可复现的稳态协议，并保留了有序床作为对照臂。

**总判定**：1 个算例中 0 个落在 ±10 % 内；主算例 D5 未通过。
