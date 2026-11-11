# 交差审计 — CASE-SILO-JANSSEN（m15265，TensorDEM）

审计单：m15321（board-audit） · 被审单：m15265（succeeded，auto dev-acceptance）
被审提交：`TensorDEM` branch `agent/m15265` @ `284e1efc`（work tree `/root/tp-agent/work/m15265`）
审计日期：2026-11-11 · 结论：**L2（独立功能实现）· disposition = approve with notes（非返修，但基准未通过）**

## 0. 独立性声明与限制

* 本审计**不是**重做：未重跑 DEM 算例（单例 414 s，锚点 base 需 ≈1.5e5 颗粒）。已把锚点复现命令递交集群
  （`cj1061`，`PYTHONPATH=src python3 benchmarks/silo_janssen_wall_pressure.py --case base --out runs/m15265/sim/base.json`），
  递交时队列 272 在前，审计关闭时 `queued` 未派发 → 「锚点可复现」判为 **not_run（未验证）**，不得读作通过。
* 审计方对被审代码无修改权；下文所有数字均来自被审工件或审计方自己的只读重算。

## 1. 基准裁判（唯一裁判）核查

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SILO-JANSSEN';
→ 0 rows（2026-11-11 22:05 实查；表共 14 行，全部 pending，无一行为本锚点）
```

**F1（blocking，机制级）：锚点未登记进基准裁判表。** `CASE-SILO-JANSSEN` 在 `tensor_benchmark_cases` 中 0 行，
因此容差不是由裁判给出，而是取自任务书/预注册的 `TOL = 0.15`（`refs/benchmarks/silo_janssen_wall_pressure.py:87`，
`refs/prereg/m15265_silo_janssen.md` §Metrics）。→ 本单的 PASS/FAIL 目前**无法被机器复核**，只能人工核。
改进：算例池入池时应同时 INSERT 该行（case_name=P_inf/z0/base_pressure、metric、tolerance='<15%'），否则「算例池铺开」不产生可裁定的基准。

## 2. 三件套是否真存在（逐项）

| 件 | 指针 | 审计核验 |
|---|---|---|
| run_artifacts | `refs/runs/m15265/sim/coarse.json`（465 行，sha256 `f43115f1…`）；`refs/runs/m15265/comparison.json`；`refs/runs/m15265/figures/fig2_error_bars.png`（41 KB）；`refs/runs/m15265/log_coarse.log` | **真**。coarse 是 31142 颗粒、4552 步、`wall=414 s` 的真 DEM 运行（log 行 + 工件内含 `profiles`/`energy`/`wall_clock_s=413.73`）。`smoke/base.json` 是 quick 模式（R=0.03, H=0.10, 4560 颗粒）冒烟件，**不是**锚点 base（该件正确落在 `smoke/` 而非 `sim/`）。 |
| benchmark ref vs got 数值 | `refs/runs/m15265/comparison.json::rows[0]` | **真数值，但结果是硬 FAIL**：`P_inf_ref=1097.47 Pa` vs `P_inf_got=5.3436e7 Pa` → `p_inf_err=4.8690e4`（即 4,868,959%）；`z0_ref=0.06969 m` vs `z0_got=3034.9 m` → `z0_err=4.3547e4`；`base_err=2.453`；`r2=0.321`；`pass=false`。判据 <15% → **超出容差 5 个数量级**。 |
| repro_cmd 可执行 | `refs/runs/m15265/repro.sh`（7 例串行 + analyze + pytest） | **部分可执行（已实跑非 DEM 部分）**：`--list` 正常；`--analyze` 在副本上重跑产出与提交件**逐值一致**（`rows identical: True`，`p_inf_err=48689.58832055903` 两侧同位，报告 §4 表 diff 为空）；`pytest tests/test_silo_janssen_benchmark.py -q` → **5 passed in 10.07 s**。**未执行**：`repro.sh` 的 7 例 DEM 循环（本地 ≈50 min 起、且 repro.sh 内 `log_*.log` 三个为 0 字节说明该脚本从未整体跑过；`log_base.log`/`log_muw060.log`/`probe.log` 均 0 B）。 |

## 3. 六段报告核查（程序化注入 vs 手抄）

`refs/runs/m15265/CASE_SILO_JANSSEN_report.md` 与平台报告 id 14377（`mission_id=15265`，1 图 `fig2_error_bars.png`）六段齐全
（①问题+ref 出处 ②方法 ③参数表 ④ref vs got 对照 ⑤敏感性 ⑥失败模式+摩擦）+ ⑦判定/根因，且**逐量显式答「<15%?」**（✗/✓ 列）：
判断为**六要素实质满足**，且 honest-FAIL 未粉饰（原文明写「本波 FAIL」「base 锚点本轮 NOT_RUN，不是 PASS」）。

但报告有两处**完整性缺陷**（非致命，见 F4/F5）：
* §5 敏感性一节是**空承诺**：「实测 z0 列： m」后没有任何数字（只有 coarse 一个点），却在文中列出 5 条敏感性轴 →
  形式上像做了敏感性分析，实际 WITHHELD。报告 §7 末尾已如实说明「不足以支撑敏感性结论」，**但 §5 段落本身未标 WITHHELD**。
* §4 表头单位与单元格不符：表头 `z0 ref [mm]` / `err %`，单元格给出的是 **m** 与**比值**（`0.06969`、`4.869e+04`）。
  脚注只解释了 err 列单位（「单位为 1」），未解释 z0 列 → 读者按表头读会把 z0 误差读错 1000×。

## 4. 判级（L0–L5）：**L2**

| 级 | 判据（机器口径 SCHEME） | 实质核验 | 判 |
|---|---|---|---|
| L1 | tests 非空 | 5 条归约完整性测试，审计实跑 5 passed；且 `pytest -k "silo or janssen"` **31 passed / 3 skipped / 9355 deselected in 85.31 s** → janssen.py 量纲修**无回归** | ✅ |
| L2 | impl_files 非占位 + entry + run_artifacts | `benchmarks/silo_janssen_wall_pressure.py`（539 行，真 argparse 入口 `--case/--analyze/--list`）、`src/tensordem/silo_flow/janssen.py`（量纲修正，真实现）、31142 颗粒 414 s 真运行工件 | ✅ |
| L3 | 另需 integration_callers（主求解器/平台侧真调用） | 本交付（benchmark harness）**无任何仓内调用方**：`grep -rn "silo_janssen_wall_pressure" .` 仅命中自身与 tests；仓内 `src/tensordem/verification/granular/bulk_flow/campaign.py:38` 调用的是**既有的** `run_janssen`，早于本单，不能算本交付的集成。且分支**未合入** `/root/tensor-ecosystem/TensorDEM` main（该 main 无 `benchmarks/silo_janssen_wall_pressure.py`、无 `runs/m15265`） | ❌ |
| L4 | 另需 benchmark 数值对照 + repro_cmd | 数值对照存在但**方向是反的**：got 比 ref 大 4.87e4 倍，超出 15% 容差 5 个数量级；锚点 base **NOT_RUN**；7 例 repro 从未整体执行 | ❌ |
| L5 | 另需 external_anchor | Janssen 1895 原始实测表 WITHHELD（Sperl 2006 paywall），只有闭式解析，无外部实验锚 | ❌ |

**级别 = 功能实质，不由报告篇幅改变**：交付的是「真代码 + 真 DEM 工件 + 可重放归约」的**独立功能实现**（L2）；
基准验证这一层**没有通过**（不是「没做」而是「做了且不合格」），因此不得记 L4。
（对照：同仓 m15235 CASE-TRIAXIAL-CD 亦判 L2，口径一致。）

## 5. 分级 findings

**F1 blocking（机制）** 基准裁判 0 行 —— 见 §1。
**F2 high（正确性）** 壁面路线归约在 slab 层面对比时**漏乘 K**：`analyse_case()` 用 `sig_ref = janssen_profile(depth,p_inf,z0)`（= σ_v）
与实测 `sigma_w`（按文件头定义应为 `K·σ_v`）直接比。证据：`comparison.json::cases.coarse.self_consistent.profile`
`rel_err_mean=1.629`；而按 `K_meas=1.435` 校正后应≈0.13 量级。后果：次要指标 `worst_slab=13.23`、`rel_err_rms=2.94` 系统性虚高。
**F3 high（测试盲区）** 5 条测试只覆盖**归约在合成 Janssen 床上为 0 误差**，而合成件把 `wall_profile.sigma_w` 直接设成 `sigma_v`
（`refs/tests/test_silo_janssen_benchmark.py:60` 前后 `"wall_profile":{"sigma_w": sigma.tolist()}`）→ **测试把 F2 的错误约定写进了夹具**，
所以真实运行中 `r2=0.32` 的拟合退化（P_inf→5.3e7）与 F2 都**不可能被测试捕获**。测试断言的是不变量，但这些不变量对齐的是错误约定。
**F4 medium（可复现性）** `repro.sh` 从未整体执行（3 个 0 B 日志）；锚点 base 未跑（auditor 递交 cj1061 仍 queued）。
**F5 medium（预注册漂移）** 预注册 M3 写的是 `|P_inf/z0 − rho_b g R|/(rho_b g R)`（`refs/prereg/…md` §Metrics），
而代码在本单把定义改成 `rho_b g`（正确，`P_inf/z0=rho_b g` 无量纲自洽）→ 同一个算例 M3 由 **1298%（FAIL）变 11.8%（PASS）**。
修正本身数学正确、报告 §1/§7 已披露，但**预注册文件未同步 amend**，且变更发生在看到 coarse 结果之后（提交 `284e1efc` 把「跑算例」与「改恒等式」混在一个 commit）→ 事后改判据的风险不可排除。
**F6 low（工件一致性）** `refs/runs/m15265/sim/coarse.json` 内 `identity_error=12.976`（旧 `rho_b g R` 口径），
而 `comparison.json` 顶层携带同一 stale 字段、`self_consistent.identity_rel_err=0.118` 是**事后重算**值 → 同一工件里两个恒等式误差并存，
下游若读 case 文件会得到 1298%。建议只保留一个字段并标注口径。
**F7 low（交付）** 分支 `agent/m15265` 未合入生态仓 main（`/root/tensor-ecosystem/TensorDEM` @ `ad75394d` 无本单文件）→ L3 不可能成立。

## 6. 已执行的检查 / 未执行

已执行：`--list`；`pytest tests/test_silo_janssen_benchmark.py -q`（5 passed, 10.07 s）；`pytest -k "silo or janssen"`（31 passed/3 skipped, 85.31 s）；
`--analyze` 副本重放比对（rows 逐值一致）；`tensor_benchmark_cases` / `tensor_maturity_verdicts` / `tensor_dev_missions` 查询；平台报告 14377 读取；
仓库 grep 调用方；生态仓 main 检出核对；工件 sha256。
未执行（not_run，不得读作通过）：锚点 `base` 及另 5 例 DEM 重跑（集群 cj1061 queued）；`repro.sh` 端到端；Janssen 1895 原表核对（WITHHELD）。

## 7. 处置（disposition）

**approve with notes / 不返修，但基准判为未通过。** L2 记 confirmed（真代码 + 真工件 + 归约可重放），
**不得**据此认为「筒仓壁压 vs Janssen 已验证」：唯一跑完的 coarse 例误差 4.87e4×，锚点 base NOT_RUN。
下一轮最小修复（按性价比）：①补 `tensor_benchmark_cases` 行（F1）；②slab 对比乘 K（F2）+ 夹具改 `sigma_w=K·sigma_v`（F3）；
③拟合加退化拒绝（R²<0.9 或 z0>H/2 不出 P_inf）+ `H/z0 ≥ 3` 前置断言；④把 K_meas 改成深部层稳健中位数；
⑤anchor 抬到 H=0.6 m 后上集群分片跑 7 例并整体跑通 `repro.sh`（F4）；⑥预注册 amend 记录 M3 定义变更（F5）。