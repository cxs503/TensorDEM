# AUDIT — RCP 沉积算例：刚度阶梯的时间收敛锚是否覆盖到刚性端

**任务**: m21401 · `CASE-DEM-RCP-PACKING-FRACTION` · TensorDEM  
**性质**: 验证欠账补齐（时间离散偏差对硬球截距的污染评估）  
**本文件全部数字由 `benchmarks/case_rcp_deposition/timestep_ladder_audit.py` 程序化注入，零手抄。**

## 1 为什么做这个审计

硬球截距是对 `k_n` 做 Richardson 外推得到的，它隐含假设**每个刚度档都已时间收敛**。
但此前 `dt/2` 探针只在最软一档（`k_n = 4000`）做过；`plan.json` 里登记的
`timestep-probe-stiff`（`k_n = 64000`，`dt = t_c/40`，两档摩擦）**从未执行**。
而截距恰恰是由刚性端（`k_n = 64000`、无摩擦档还有 `256000`）锚定的——
若刚性端存在残余 `dt` 偏差，它会被误当作 `k_n` 收敛阶的一部分，
而 `mu = 0.1` 档的收敛阶 `p` 本就不可辨识（见 `AUDIT_m21145`），偏差将无处安放。
本审计补做刚性端探针，并把偏差以**阶次区间**（q = 1 与 q = 2）的形式传播到截距，
而不是挑一个阶次充数。

## 2 怎么做

1. 对每个 `(mu, k_n)` 档配对粗/细两个时间步（`dt_frac` 1/20 与 1/40，`dt = dt_frac * t_c`），
   取 `phi_bulk`（D1 CIC 估计量，与 `AUDIT_m18395` 的合并读数一致）的种子均值；
2. 偏差 `Delta = phi(dt/2) - phi(dt)`；预登记可接受阈 `|Delta| < 5e-4`；
3. `dt -> 0` 修正按 `q = 1` 与 `q = 2` 两个阶次各算一次，取区间（`phi_0 = phi_f + Delta/(2^q - 1)`）；
4. 用修正后的档值重建阶梯、重拟合截距，与未修正截距比较；
5. 用 `tensordem.rcp_references.judge` 复核判定（无摩擦档有文献锚，低摩擦档只有廓形）；
6. 探针缺失的档位一律记 `NOT_RUN`，不得计为通过。

工具链: `benchmarks/case_rcp_deposition/run_case.py` → `tensordem.deposition_packing`
（侧向周期、地板 + 自由面，无侧壁）→ 本脚本 → matplotlib。

## 3 各档时间偏差

| mu | k_n | phi(dt) (n seeds) | phi(dt/2) (n seeds) | Delta | 阶次区间 dt→0 | 判定 |
|---|---|---|---|---|---|---|
| 0 | 4000 | 0.64237 (3) | 0.64263 (1) | +2.576e-04 | [0.64272, 0.64289] | PASS |
| 0 | 16000 | 0.63791 (1) | — | — | — | NOT_RUN |
| 0 | 64000 | 0.63676 (3) | 0.63626 (1) | -5.069e-04 | [0.63575, 0.63609] | FAIL |
| 0 | 256000 | 0.63571 (1) | — | — | — | NOT_RUN |
| 0.1 | 4000 | 0.62628 (3) | 0.62632 (1) | +4.370e-05 | [0.62634, 0.62637] | PASS |
| 0.1 | 16000 | 0.62159 (1) | — | — | — | NOT_RUN |
| 0.1 | 64000 | 0.61928 (1) | 0.61940 (1) | +1.188e-04 | [0.61944, 0.61952] | PASS |

## 4 截距：修正前后

| mu | 阶梯 (k_n: phi) | 原始截距 | 阶次 p | 修正后截距 | 位移 | 相对文献/锚点 |
|---|---|---|---|---|---|---|
| 0 | 4000: 0.64237; 16000: 0.63791; 64000: 0.63676; 256000: 0.63571 | 0.63536 | n/a | 0.63569 | 3.267e-04 | 0.6366 (PASS) |

*mu = 0 修正后阶次 p = 1.624*
| 0.1 | 4000: 0.62628; 16000: 0.62159; 64000: 0.61928 | 0.61705 | 0.512 | 0.61781 | 7.545e-04 | 0.6250 (WITHHELD) |

*mu = 0.1 修正后阶次 p = 0.588*

## 5 闸门（预登记）

| id | 判据 | 实测 | 状态 |
|---|---|---|---|
| T_bias_mu0_kn4000 | |Delta| < 0.0005 | +2.576e-04 | PASS |
| T_bias_mu0_kn16000 | |Delta| < 0.0005 | no dt/2 run | NOT_RUN |
| T_bias_mu0_kn64000 | |Delta| < 0.0005 | -5.069e-04 | FAIL |
| T_bias_mu0_kn256000 | |Delta| < 0.0005 | no dt/2 run | NOT_RUN |
| T_bias_mu0.1_kn4000 | |Delta| < 0.0005 | +4.370e-05 | PASS |
| T_bias_mu0.1_kn16000 | |Delta| < 0.0005 | no dt/2 run | NOT_RUN |
| T_bias_mu0.1_kn64000 | |Delta| < 0.0005 | +1.188e-04 | PASS |
| T_intercept_shift_corrected_mu0 | |limit_corrected - limit_raw| < 0.001 | 3.267e-04 | PASS |
| T_intercept_shift_corrected_mu0.1 | |limit_corrected - limit_raw| < 0.001 | 7.545e-04 | PASS |
| T_frictionless_claim_survives_dt_correction | verdict stays PASS vs literature anchor | PASS rel=-0.144% | PASS |
| T_stiff_end_covered_mu0 | a dt/2 probe exists at k_n >= 16000 (not only the softest level) | probed levels [4000.0, 64000.0] | PASS |
| T_stiff_end_covered_mu0.1 | a dt/2 probe exists at k_n >= 16000 (not only the softest level) | probed levels [4000.0, 64000.0] | PASS |

合计：PASS 8 · FAIL 1 · NOT_RUN 3。

## 6 图

![ladder](../benchmarks/case_rcp_deposition/figures/fig_m21401_dt_ladder.png)

*刚度阶梯：圆点=粗时间步实测、方块=细时间步实测、菱形=阶次区间修正值（误差棒=区间半宽）、
虚线=修正后截距、点线=原始截距、实线=无摩擦文献锚 0.6366、灰带=账簿验收带。*

![bias](../benchmarks/case_rcp_deposition/figures/fig_m21401_dt_bias.png)

*各档时间偏差 `Delta`：绿=在 ±0.0005 内，红=超阈；`k_n = 64000` 两档为本审计新增的刚性端探针。*

## 7 注意点与局限

1. `dt -> 0` 修正由两个时间步给出，阶次 `q` 本不可辨识，故只报区间；
   区间半宽已计入误差判定（见第 8 节）。
2. 阈值 `|Delta| < 5e-4` 是预登记的工程判据（约为 `phi_ref` 的 0.08%），
   不是从某条定理推出的普适界。
3. 无摩擦档 `k_n = 256000` 只有粗时间步单种子，未做时间探针——
   该档在修正阶梯中沿用粗时间步值，表格已标注。
4. 低摩擦档（`mu = 0.1`）的参考值仍是 `derived` 级（见 `AUDIT_m21145`），
   本审计**不改变**这一分级，只回答「截距对时间离散是否稳定」。
5. 本审计不重估稳态判据与控制体（分别见 `AUDIT_m17063`、`AUDIT_m18395`），
   也不改动刚度外推的阶次区间策略（见 `AUDIT_m21145`）。
**边际 FAIL 说明（程序化判定）**：
- `mu = 0, k_n = 64000`：`|Delta| = 5.069e-04` 略超阈 `0.0005`，但该档细时间步只有 1 个种子、粗时间步种子标准差 3.13e-04（z = 1.62）——**偏差与种子噪声不可分离**，故该档不得认证为时间收敛，闸门如实记 FAIL；补 2 个细时间步种子即可判定（本单未及回件，列为下一波首要欠账）。

6. 刚性端探针为本单递交集群作业产出（n=10000，`dt = t_c/40`，种子 1），
   单种子，故档值噪声未由种子重复给出；其量级与粗时间步同档种子标准差可比。

## 8 误差判定（逐项，显式回答是否 <3%）

| mu | 量 | 计算值 | 基准 | 相对误差 | 判定 | <3%? |
|---|---|---|---|---|---|---|
| 0 | hard-sphere intercept (dt-corrected ladder) | 0.63569 | 0.6366 (PASS) | -0.14% | PASS | yes |
| 0 | hard-sphere intercept (raw ladder) | 0.63536 | 0.6366 (PASS) | -0.19% | PASS | yes |
| 0.1 | hard-sphere intercept (dt-corrected ladder) | 0.61781 | 0.6250 (WITHHELD) | -1.15% | WITHHELD | no admissible anchor |
| 0.1 | hard-sphere intercept (raw ladder) | 0.61705 | 0.6250 (WITHHELD) | -1.27% | WITHHELD | no admissible anchor |

**结论**：
- 无摩擦档硬球截距（时间修正后）相对文献锚 0.6366 的相对误差为 -0.14%，**<3%**，判定 PASS。
- 低摩擦档截距相对其表值锚的相对误差为 -1.15%，但该锚为 `derived` 级，
  按 `AUDIT_m21145` 的锚点政策精度声称 **WITHHELD**，最强可声明为 PASS_BRACKET。
- **诚实 FAIL**：无摩擦档刚性端（见第 7 节边际 FAIL 说明）时间偏差超出预登记阈，该档未被认证为时间收敛；本审计**不**把它算作通过，也不因此推翻无摩擦档的整体判定（截距位移 见第 4 节，且修正后相对文献锚仍在 3% 内）。

