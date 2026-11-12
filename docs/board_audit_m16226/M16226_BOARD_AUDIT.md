# M16226 交差核验单 — m16147 · CASE-DEM-MINDLIN-TANGENTIAL（TensorDEM）

审计性质：**交差核验（verification, not re-do）**。被审对象=m16147（已 succeeded，Merge #1351 / main `00025434`）。
本审计只读交付字节 + 独立重算 + 独立复跑，不重做算例、不改被审仓代码。

| 项 | 值 |
|---|---|
| anchor | `CASE-DEM-MINDLIN-TANGENTIAL`（切向 Mindlin 微滑移→完全滑移过渡） |
| 被审单 | m16147（返修轮 1，原单 m16065） |
| 被审 commit | `00025434` "Merge #1351 (m16147) via merge-master"（7 files, +1678/−14） |
| 被审工作树 | `/root/tp-agent/work/m16147` |
| 审计工作树 | `/root/tp-agent/work/m16226`（分支 `agent/m16226`） |
| 上一裁定 | `tensor_maturity_verdicts.id=425`（m16065，声明 L4，**gate_level=1**，gate_state=`reopened`）——本次核验的核心问题：返修后 `run_artifacts/entry/integration_callers` 是否真的到位 |

---

## 0. 结论（disposition: **approve with notes → L4**）

交付**真实存在且可复算**：源码非占位、入口可跑、工件字节存在且 sha256 自洽、独立复跑复现同一数字、
单元/集成测试 10 passed、判据 9/9 PASS。**判 L4（基准验证）成立**；不给 L5（定律是 1953 教科书闭式，
无独立实验/第三方数值锚，且求解器侧只接单调加载支）。

返修备注（不阻塞消账，见 §5）：F1 裁判未登记 / F2 报告样本数与工件不一致 / F3 求解器只接单调支 /
F4 误差口径命名残留 / F5 平台门 grep 目标仓陈旧。

---

## 1. 三件套逐项取证

### 1.1 run_artifacts（真实运行工件）

交付工作树 `/root/tp-agent/work/m16147/artifacts/mindlin_tangential/`：

| 文件 | 字节 | 说明 |
|---|---|---|
| `results.json` | 5231 | 参考链 + 全指标 + 收敛/敏感性 + 环境 |
| `curves.csv` | 697895 | 6438 行求解器曲线 + 1200 点解析网格 |
| `checksums.json` | 679 | 6 个工件 sha256 |
| `run.log` | 3028 | 完整控制台日志（含 8 核/loongarch64/python 3.11.6/torch 2.7.1） |
| `figures/fig1..fig4.png` | — | F_t–δ 曲线 / 逐点误差 / Δt+μ+Fn 敏感性 / 粘着圆 c/a |

**审计独立复核**：把上述 6 个文件按字节复制进本仓 `docs/board_audit_m16226/evidence/`，
用**本审计自己**的 sha256 与交付的 `checksums.json` 逐项比对 → **6/6 全 match**（results.json `c819c3dc…`、
curves.csv `c8e3a7f1…`、fig1 `562fd50a…`、fig2 `f7918174…`、fig3 `a2096c83…`、fig4 `45fe63f3…`）。

> **为什么复制入库**：TensorDEM 的 `.gitignore` 排除 `artifacts/`，所以交付工件**不在 git 里**；
> 上一轮 gate 正是以 `run_artifacts(工件缺失)` 把 L4 降到 L1。本审计把被审字节落到**受版本管理的**
> `docs/board_audit_m16226/evidence/`（不改一个字节），使工件对平台门与后续复算都可达。

### 1.2 benchmark：ref vs got 逐项（**唯一裁判**）

注册裁判（`tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-MINDLIN-TANGENTIAL'`）：
**0 行**；全表 29 行、26 个 anchor，`LIKE '%MINDLIN%'` 亦 0 行。→ 机器裁判**未登记**（F1，如实记录，本审计不代填，
与 id=425 的处置一致）。实际在force的判据=使命 goal 文本 `<5%` + `run.py/validate.py` 预登记阈值（两者一致，无回溯调参）。

| # | 指标 | ref（解析） | got（求解器） | 误差 | 判据 | 判定 |
|---|---|---|---|---|---|---|
| B1 | 微滑移段 max \|F_t−F_MD\|/(μF_n) | `μF_n[1−(1−δ/δ_s)^{3/2}]` | 同 | **3.7007e-16** | <5% | PASS（余量 14 个数量级） |
| B2 | 过渡段 \|δ−δ_s\|≤0.1δ_s | 同上 | 同 | **1.8504e-16** | <5% | PASS |
| B3 | 完全滑移平台 | μF_n=0.3 N | 0.3000000 N | **1.8504e-16** | <1% | PASS |
| B4 | 初始斜率 k_t | 8G\*a=465959.97 N/m | 461694.63 N/m（割线估计） | **9.1539e-3** | <2% | PASS（割线偏差，非实现误差） |
| B5 | 拟合 δ_s | 9.657482e-07 m | 9.657482e-07 m | **4.3854e-16** | <5% | PASS |
| B6 | 判别力：遗留线性 Cundall–Strack 偏离 | — | **0.192441** | 必须 >5% | >5% | PASS（算例不可被旧模型通过） |

**审计独立重算（不读交付代码，只读工件字节）**：

* 从 `curves.csv` 6438 个求解器样点重算 `|F_t−F_MD|/(μF_n)`：微滑移 **2.243e-13**、过渡 **1.817e-13**、
  平台 **0.0**（与交付 3.70e-16 的差异是 CSV 15 位十进制序列化精度地板，仍低于判据 11 个数量级）。
* 从 Hertz 一阶原理独立重算参考链：a=2.574429e-04 m、k_t=465959.9669 N/m、δ_s=9.657482e-07 m，与 `results.json`
  相对差 2.1e-16 / 1.3e-16 / 2.2e-16；恒等式 `k_t/k_n=3(1−ν)/(2−ν)=1.235294` 复算一致。
* 独立重算线性律最大偏离：**0.192445 at δ/δ_s=0.6667**（δ=0.5δ_s 处 0.103553）→ 与交付 19.244%/10.355% 一致。

> **口径警示（重要，非缺陷但必须写清）**：B1/B2/B3/B5 的 1e-16 级一致性是**闭式与闭式**的一致
> ——`run.py` 用独立实现的 Mindlin 闭式作参考，求解器实现同一闭式，所以它验证的是**实现与接线正确**，
> 不是定律本身被独立证实。真正不可循环的量是 **B4（k_t 只差 0.915%）**、**δ_s ∝ μ^1.000000**、
> **δ_s ∝ F_n^0.666667**（theory 2/3）与 **B6 判别力**：这三者要跑通
> "法向力→Hertz 半径→G\*→k_t→δ_s" 整条链才成立，不是回填曲线能做到的。

### 1.3 repro_cmd（可执行 + 已复跑）

```bash
PYTHONPATH=src python benchmarks/case_mindlin_tangential/run.py   --out artifacts/mindlin_tangential
PYTHONPATH=src python benchmarks/case_mindlin_tangential/validate.py --out artifacts/mindlin_tangential
PYTHONPATH=src python -m pytest tests/test_mindlin_partial_slip.py -q
```

* **审计复跑（本机 loongarch64, py3.11.6/torch2.7.1）**，临时副本 `/tmp/repro_m16147`（去 .git）：
  * 全量 `run.py`：日志 `docs/board_audit_m16226/evidence/repro_full_run.log`，
    复现 micro **3.700e-16**、gross **1.850e-16**、plateau **0.300000 N**、k_t rel **9.154e-03**、
    fitted δ_s **4.385e-16**、legacy **19.244%**、analytic linear **19.231%/10.355%**——与交付数字逐项一致。
  * `--quick`（50.6 s）同样复现 6/6 criteria pass。
  * `validate.py` 在交付树原样执行：**9/9 PASS → ACCEPT**（含 checksum manifest + sha256 逐项 match）。
  * `pytest tests/test_mindlin_partial_slip.py -q`：**10 passed in 12.22 s**（与交付自报 10 passed/13.45 s 一致）。

---

## 2. 判级（L0–L5）

| 级 | 要求 | 本单证据 | 判 |
|---|---|---|---|
| L1 数学/单元验证 | 闭式锚点 | 测试 10 项：0/δ_s 端点、初始斜率、反演往返 ≤1e-12δ_s、c/a=(1/2)^{1/3}、反向支端点/斜率/环面积∝Δ² | ✅ |
| L2 独立功能实现 | 输入→计算→输出 | `src/tensordem/mindlin_partial_slip.py`（317 行，闭式/反演/粘着半径/反向支/环面积）+ `dem3d.py` opt-in 分支（+86 行） | ✅ |
| L3 系统集成 | 接入主求解器 | `DEM3DConfig.tangential_model="mindlin_partial_slip"` 在**主求解器** `dem3d.py` 两处分支生效：platen 支 :1191、颗粒-颗粒支 :1436；默认 `cundall_strack` 路径不变（回归 60 passed/2 skipped 自报，本审计抽验 `test_mindlin_partial_slip.py` 覆盖 legacy 不变性） | ✅ |
| L4 基准验证 | 完整算例+误差+可复现 | 6 项判据全 PASS + Δt×{0.5,1,2} + μ×3 + F_n×3 敏感性 + 判别力 + 干净副本复跑逐项复现 | ✅ |
| L5 科研工程 | 独立外部锚/新物理 | 定律=1953 教科书闭式（Johnson 1985 §4.3 eq 4.32）；无实验数据、无第三方数值解、无网络核对原表；求解器侧仅接单调加载支 | ❌（不给） |

**裁定：L4**。百页报告不改变级别——级别按功能实质：真代码 + 主求解器集成 + 完整算例 + 逐项误差 + 可复现。

---

## 3. 主张 vs 证据（overstated claims 检查）

| 交付主张 | 核验 | 结论 |
|---|---|---|
| "run.py 从不 import `tensordem.mindlin_partial_slip`" | `grep ^import/^from`：只 import `tensordem.dem3d`（:57） | ✅ 属实 |
| "参考值零手抄，闭式在 run.py 内由 Hertz 一阶原理评估" | 参考链独立重算差 2e-16 | ✅ 属实 |
| "无 rate dependence（代数支）" | Δt×{0.5,1,2} 误差 {7.4e-16,3.7e-16,3.7e-16} | ✅ 属实 |
| "参考原表未核对（付费墙/无网）" | 如实披露，未伪称引用原表数字 | ✅ 诚实 |
| "微滑移段 n=5766 / 完全滑移 n=672" | 从交付 `curves.csv` 独立统计：δ≤0.9δ_s **5602**、\|δ−δ_s\|≤0.1δ_s **438**、δ≥1.1δ_s **398**（总数 6438 与 `n_samples` 一致） | ❌ **不可溯源**（F2）：三个分段样本数与工件不符，且 `results.json` 不含 `n_micro/n_gross` 字段；不影响判据（误差量级相同），但违反零手抄/可溯源纪律 |
| "初始斜率误差 9.15e-3 是割线估计偏差" | 独立重算 k_t_secant 461694.6 与 theory 465960 之比 | ✅ 属实（且方向为偏低，符合割线估计） |

---

## 4. 探针与失败模式（adversarial-review）

| 探针 | 结果 |
|---|---|
| 输入边界：δ=0 / δ=δ_s / δ→∞ | 测试断言 F(0)=0 精确、F(δ_s)=μF_n 精确、单调非减、大 δ 平台饱和（`partial_slip_magnitude` clamp） |
| 反演一致性 | `inverse_displacement` 往返 ≤1e-12 δ_s |
| 向量化/非激活接触 | `partial_slip_force` 返回沿 δ 方向的力、方向单位向量 (0.6,0.8) 精确、miss 接触力=0 |
| 跨实现一致 | 与 `contact_tangential.mindlin_backbone` 差 ≤1e-14 μF_n（两条独立代码路径） |
| 遗留路径不变性 | `tangential_model="cundall_strack"` 默认路径位级不变（有回归测试） |
| 配置合法性 | 非法组合（非 hertz 接触 / 开摩擦正则化）→ `ValueError`，防止静默降级语义 |
| 复现性 | 干净副本全量复跑，数字逐项一致；sha256 6/6 match |
| 循环论证风险 | 见 §1.2 口径警示：已量化"哪些量是循环的、哪些不是"，未把 1e-16 包装成独立精度主张 |

---

## 5. 发现（severity-ranked）

* **F1 [medium · registry]** `tensor_benchmark_cases` 该 anchor **0 行**（全表 29 行），机器裁判缺席；判据实际来自
  使命 goal `<5%` + 预登记阈值。→ 池侧补登（本审计不代填，保持独立性）。影响：maturity_gate 无法用注册裁判复核。
* **F2 [medium · doc traceability]** README §4 分段样本数 5766/672 与工件不符（独立重算 5602/438/398），且
  `results.json` 未落 `n_micro/n_gross` → 报告数字不可 SQL/工件溯源。判据不受影响。建议返修：把分段计数写入 results.json 并改 README。
* **F3 [low · scope]** 求解器侧只接**单调加载支**：`dem3d.py` 在 `mindlin_partial_slip` 下 `tang_disp_next = tang_disp`
  （不投影、不设运动锚点），反向/循环路径未接 `reversal_branch`/`reversal_loop_area`（模块内有、也有循环状态机
  `contact_tangential.mindlin_tangential_step`）。README §6 已如实申报为"单调骨干准确"。本算例为单调加载，不受影响；
  但由此**模型在循环加载下不产生滞回耗散**（力是 |δ| 的无历史函数），属能力边界，不得外推到循环/地震类工况。
* **F4 [low · naming]** `run.py:256-272` 的 `err_rel_muN` 及 `results.json` 的 `max_*_err` 是**对 μF_n 归一化的绝对偏差**，
  不是相对误差（上一轮 F2 的残留命名）；README/validate 的注释已写明"fraction of mu\*Fn"，量级上不构成误报（因 ref≈μF_n）。
* **F5 [info · platform]** `maturity_gate.callers_verified()` 在 `/root/tensor-ecosystem/TensorDEM` 就地 grep，
  而该克隆的 local main 落后 origin/main **277 个 commit**（`ad75394d` vs `8133d44e`），新代码对它不可见
  → 上一轮 id=425 的 `integration_callers(grep 无引用)` 属**环境性误判**，非交付缺陷。
  同理 `.gitignore` 排除 `artifacts/`，`run_artifacts` 永远"缺失"。本审计以 `docs/board_audit_m16226/evidence/`（受版本管理）规避后者。

---

## 6. 未做的检查（unknown / not_run，不得当作 passed）

* **未**核对 Mindlin & Deresiewicz 1953 原表（付费墙 + 沙箱无网）；未做与实验数据/第三方 FEM-BEM 的对照 → 这也是不给 L5 的原因。
* **未**做完整回归套件复跑（交付自报 60 passed/2 skipped/18 subtests；本审计只复跑 `tests/test_mindlin_partial_slip.py` 10 passed）。
* **未**做循环加载下的滞回环数值验证（F3 属能力边界，未量化误差）。
* **未**在集群节点（loong64 双节点）完成全量复跑：`cj1279` 递交后长时间排队（提交时 ahead 448），本审计改在 99 本机
  loongarch64 干净副本复跑（同架构），未取得集群侧第二机器独立佐证。
* **未**审 TensorDEM 其余摩擦/切向实现（只审本 anchor 交付面）。

---

## 7. 处置

**approve with notes → L4 消账**。返修子项（不阻塞）：F1 池侧补登裁判、F2 报告分段计数落 results.json 并订正、
F3 循环支接线或明确写入能力边界文档、F4 字段命名。平台侧观察：F5 门 grep 目标仓陈旧。

审计工件：
`docs/board_audit_m16226/evidence/`（被审 6 工件字节 + 复跑日志）、
`scripts/board_audit_m16226_verdict.py`（证据装配 + gate 预判 + 落表）、本文档。