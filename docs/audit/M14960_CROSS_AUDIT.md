# m14960 交差核验（GAP-DEM-FRICTION · TensorDEM）— 独立对抗审计

**审计对象**：mission **m14960**（第 2 轮·调研结合，状态 succeeded）· 提交 **`3fcf006d`**（branch `agent/m14960`，2026-11-11 13:27:39）
**审计单**：m14978（任务榜交差核验，非重做）· 审计方 `board-audit` · **判级 L3**（L4 不成立，见 §6）
**审计基线**：从 `git archive 3fcf006d` 的**干净检出**出发（不含任何未跟踪残留），所有数字由 `scripts/audit/m14960_cross_audit.py` 机器复算 → `artifacts/audit_m14960/audit.json`。

---

## 0. 结论先行

| 项 | 裁定 |
|---|---|
| 三件套（run_artifacts / benchmark ref vs got / repro_cmd） | **run_artifacts 部分成立；repro_cmd 成立（比特级复现）；benchmark 口径成立但**该算例被本仓自己的稳态门在 0.15 容差下判 REJECT |
| 交付等级 | **L3**（真代码+已集成装置+可复现跑；基准验证 L4 未达成） |
| 交差裁定 | **reopened（返修）**：headline「−4.6%，首次满足 <10%」不能承重 |
| 诚实面 | **优**：报告自陈「不宣布锚点关闭」「文献锚未在线核验」「N=1392 未跑完」；下文缺陷均属**表述/工件一致性**，非伪造数据 |

---

## 1. 独立性声明与验证范围（做了什么、没做什么）

**独立性**：审计方不引用被审报告的任何数字；每个承重数字都由（a）提交内工件、（b）本审计方独立重跑、（c）仓内自己的 `_validity_gate` 三路中至少两路独立得出。审计在 m14960 的工作树之外、以 `git archive` 快照进行。

**已跑（可复现）**：

```bash
git archive 3fcf006d -o /tmp/m14960_snapshot.tar
mkdir -p /tmp/audit_m14960 && tar xf /tmp/m14960_snapshot.tar -C /tmp/audit_m14960
cd /tmp/audit_m14960 && PYTHONPATH=src python scripts/m14960/run_friction_matrix.py \
    --tag audit_size_medium --geom medium --mu 0.5 --gamma 0.6 --sigmas 4000   # 219 s
python -m pytest tests/test_shear_cell_friction.py -q                        # 10 passed, 84 s
PYTHONPATH=src python <TensorDEM>/scripts/audit/m14960_cross_audit.py --snapshot /tmp/audit_m14960
```

**未跑（unknown，不记为 pass）**：`mu030/mu050/mu070` 三组日志的独立重跑（单组 ≈ 35 s 制床 + 2×(290–355 s)；六次 ≈ 30 min，本单预算外）；`size_large`（N=1392）；`oss_inventory` SQL 复核（审计侧重跑通路不含该库）；文献锚在线核验（沙箱无出站 HTTP）。

---

## 2. 三件套逐项取证

### 2.1 run_artifacts —— 部分成立

| 指针 | 存在 | 内容判定 |
|---|---|---|
| `artifacts/shear_cell_m14960/size_medium.json`（109 450 B） | ✅ | 真实：含 `shear_strain_history`/`shear_stress_history`/`normal_stress_history` 共 1511 点时间序列、`total_steps=48246`、`secs=189` |
| `artifacts/shear_cell_m14960/size_small.json`（68 794 B） | ✅ | 真实：945 点时间序列、`total_steps=38881` |
| `artifacts/shear_cell_m14960/summary.json` | ✅ | 由 `analyse_friction_matrix.py` 生成，可复现 |
| `artifacts/shear_cell_m14960/log_levels.json`（7 行） | ⚠️ | 7 行数据无对应 per-tag JSON；`runs/*.log` 存在（每份 3 行）但**无提交脚本可再生成该 JSON**（C6 FAIL） |
| `docs/SHEAR_CELL_M14960_tables.md` 表 2 / 表 3 | ❌ | 表 2 三行全标 **(未完成)**；表 3 **正文 0 行**（仅表头），而报告 R2 引用「表 3 第 1–2 行」（C4 FAIL） |

### 2.2 benchmark ref vs got

| 量 | ref | got | 差 | 判定 |
|---|---|---|---|---|
| μ_macro（N=390, μ_p=0.5, σ_n=4000 Pa, γ=0.6） | 文献准静态 0.38（带 0.36–0.40） | **0.3627** | **−4.6%** | 数字本身**真实且可复现**；但窗口非稳态（§3），且锚未与材料标定（C7 WITHHELD） |
| μ_macro（N=78 同条件） | 同上 | 0.3037 | −20.1% | 真实，但同被稳态门拒绝（drift −0.57） |

数值三路一致：提交工件 τ=1447.633 552 m、本审计重跑 τ=1447.633 552 m、`mu=0.362 657 900 96` 完全相同（**Δ=0.0**，C1 PASS）；伺服保持 σ_n=3991.733（−0.21%）亦复现。

### 2.3 repro_cmd —— 成立

`PYTHONPATH=src python scripts/m14960/run_friction_matrix.py --tag … --geom medium --mu 0.5 --gamma 0.6 --sigmas 4000` 在干净检出中 **219 s 完成**，9 个承重量（σ_n/τ/μ/I/drift/r²/steps/N/shear_rate）与提交工件**逐位相同**（`artifacts/audit_m14960/rerun_size_medium.json`、`rerun_size_medium.log`）。这是本单最硬的正面结论：**TensorDEM 的直剪盒装置是确定性的、可复现的**（seed=6107）。


### 2.4 平台自身 L4 门的独立印证（机读，非本审计自造）

`GET /api/v1/tensor/bench/l4-verdict/14960` →

```json
{"status": "CLAIMED_NOT_PROVEN", "l4_benchmark_pass": false, "rows": 0,
 "failed_checks": ["execution_success","source_cluster_job","cluster_job_bound",
                   "ref_got_error_present","within_tolerance"],
 "reason": "no tensorbench_results row for this mission: the L4 benchmark claim is self-declared, not measured",
 "rule": "L4 requires >=1 platform-recomputed benchmark row (ref/got/error)"}
```

即：平台侧**没有**任何被重算过的 benchmark 行（ref/got/error 三元组未入 `tensorbench_results`）。这与本审计 §3/§6 的 L4 不成立**相互独立地**一致。

---

## 3. 阻断级发现（S1）：headline 数字取自本仓判定为非稳态的窗口

`ShearCellCase.stationarity_tol` 默认 `None`（关）；m14960 的 `size_*` 两次运行均以默认值提交，JSON 里 `stationarity_tol: null`。而报告的 `mu030/mu050/mu070` 行用的是 **0.15** 容差并据此 REJECT 了两行。把**同一条 0.15 判据**施加到 headline 两行（`src/tensordem/shear_cell.py:612 _validity_gate`，本审计直接调用）：

| run | 记录 drift_rel | drift R² | gate 关 | **gate @0.15** |
|---|---|---|---|---|
| size_medium（headline） | **+0.240** | 0.083 | valid=True | **REJECT**：`relative drift 0.240 > tolerance 0.150` |
| size_small | −0.566 | 0.211 | valid=True | **REJECT**：`−0.566 > 0.150` |

更进一步，同一段 τ 历史取不同窗口（C3）：

| 窗口 | size_medium τ [Pa] | μ | vs 0.38 | 是否 <10% |
|---|---|---|---|---|
| 后 50%（=提交值） | 1447.6 | 0.3627 | −4.6% | 是 |
| 后 40% | 1436.3 | 0.3598 | −5.3% | 是 |
| 后 25% | 1505.1 | 0.3771 | −0.8% | 是 |
| 后 10% | 1888.1 | 0.4730 | **+24.5%** | **否** |

τ 逐段单调上升（前 1/10 = 1267.3 Pa → 后 1/10 = 1888.7 Pa），R²=0.083 说明漂移几乎不可用直线解释。**同一算例的 μ 在 0.360–0.473 之间（相对极差 28.8%）**：(i) 提交值只是「后 50% 窗口均值」，不是稳态摩擦系数；(ii) 「−4.6% <10%」是窗口选择的产物，换成后 10% 立刻变成 +24.5% 不达标。图 `artifacts/audit_m14960/window_sensitivity.png`（a：τ–γ 仍在漂移，阴影=平均窗口；b：μ 随窗口长度口径漂移，红线=±10% 门）。

`size_small` 更极端（τ 从 926 Pa 掉到 733 Pa，μ 0.304→0.185），说明 3 层盒在剪切中**退化**——因此报告的 R1「+19.4% 是有限尺寸效应」把两个都非稳态的点做了差，方向可信但**幅度不可引用**。

**后果**：验收判据「摩擦系数 vs 实验 <10%」在本单**未被成立地证明**；L4（基准验证）不成立。

---

## 4. 次级发现（S2/S3）

- **S2 工件与结论不一致**：报告/任务 R2 引「表 3 第 1–2 行（2 级命中）」，但提交的 `docs/SHEAR_CELL_M14960_tables.md` 表 3 正文**空**、表 2 三行**全为 (未完成)**。7 行数据只以文本日志 + 汇总 JSON 存在，缺 per-tag 机读工件，无法被 gate 机器复核。
- **S2「程序化注入零手抄」不成立**：`docs/SHEAR_CELL_M14960.md` §5 误差判定表残留未渲染模板 `{100*er(0.3627):+.1f}%`、`{100*j['size_medium']['levels'][0]['sigma_rel_err']:+.2f}%`（引用未定义变量 `j`），5 行中 2 行**没有任何数字**；§3.1 表头也留 `%+.1f%%` 模板。任务要求的「误差逐项列出」实际有 2/5 缺项（C5 FAIL）。
- **S3 日志行不可再生成**：`scripts/m14960/analyse_friction_matrix.py` 全文没有读取 `runs/*.log` 的代码；文档 §6 复现段给出的是一段**不在仓内**的 heredoc。故 `log_levels.json` 的 7 行只能靠人工/会话内片段重建（C6 FAIL）。
- **S3 锚不是「与被测材料配套的实验」**：ref=0.38（带 0.36–0.40）按模型知识引用、未在线核验（报告已自陈），且 DEM 的 μ_p=0.5 **未与锚对应材料标定**；同一算例对带边界 0.36 读 +0.7%、对 0.40 读 −9.3%——「<10%」在带边界处结论翻转，不构成预测能力（C7 → WITHHELD）。
- **正面核验**：`mu030/mu050` 首级出现 τ **精确等于** μ_p·σ_n（600.00/2000=0.3000；1000.00/2000=0.5000，drift −0.00）。审计独立复核认为这与「压板与顶层全程滑动、切向力被 Coulomb 极限夹住」的物理一致，**不是伪造痕迹**；报告把它判为「稳态门 PASS 恰是退化信号」的 R2 结论成立（但表 3 空行仍需补齐）。

---

## 5. 未跑的检查（记 unknown，不记 pass）

`mu030/mu050/mu070` 重跑 · `size_large`（N=1392） · `oss_inventory` 空集 SQL 复核 · 文献锚在线核验（无出站） · 2D 周期 SLLOD / 三轴交叉校验（报告自列为后续）。

---

## 6. 判级推导（L0–L5）

| 级 | 门槛 | 本单 |
|---|---|---|
| L1 数学/单元验证 | — | 超过 |
| L2 独立功能实现 | 真代码 输入→计算→输出 | **达成**：`scripts/m14960/run_friction_matrix.py`（133 行，单点驱动，输出机读 JSON）+ `analyse_friction_matrix.py`（146 行，表/图程序化生成） |
| L3 系统集成 | 接入主求解器/装置/平台 | **达成**：驱动直调 `src/tensordem/shear_cell.py`（`prepare_bed`/`simulate_shear_cell`/`measure_macro_friction`，L674），装置回归 `tests/test_shear_cell_friction.py` **10 passed（本审计实跑，84 s）**；数字与 `granular3d_core` 接触核一致 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | **未达成**：可复现（C1 比特级 PASS）✅；完整算例 ✅（48246 步、1511 点时序）；误差分析 ❌——算例被本仓同一稳态门 @0.15 判 REJECT，μ 窗口口径漂移 28.8%，且外部锚未材料标定（C7 WITHHELD） |
| L5 科研工程成果 | — | 不适用 |

**L3 结论**：真实、已集成、确定可复现，但 benchmark 的承重结论被自家判据否掉 → 返修而非消账。

---

## 7. 返修清单（按性价比排序）

1. **关稳态门重跑或延长剪切**：`--tol 0.15` 下把 γ 推到 τ 平台（m14840 记录 γ=0.30 时 drift 达 +0.8，本单 γ=0.6 仍 +0.24），直到 `|drift_rel|≤0.15` 且 R² 尚可，再报 μ；在此之前 headline 应写作「窗口均值，非稳态」（一句话即可消掉 S1 的表述风险）。
2. **补 per-tag 机读工件**：把 `runs/*.log` 的解析写进 `scripts/m14960/`（提交），重生成表 2/表 3；空行要么补数要么明写「未跑」。
3. **修 §5 注入**：误差判定表用 `analyse_friction_matrix.py` 直接 `format` 生成，杜绝未渲染模板；`<3%` 列改为机器判定。
4. **锚的材料配套**：给 μ_s 锚绑定实验材料参数（粒径分布/形状/μ_p 量级），或改用「μ_p 标定 → 独立工况验证」协议（报告 §2.3 已提出，需落地成算例）。
5. **退化排除项**：把「μ/μ_p∈[0.98,1.02] ⇒ 界面滑移」写进 `_validity_gate`（报告 §4.1 已识别，仍未实现）——这会给 R2 一个机器判据而不是人工rhetoric。

---

## 8. 归档指针

| 项 | 路径 |
|---|---|
| 机器复算 | `artifacts/audit_m14960/audit.json`（C1–C7） |
| 独立重跑 | `artifacts/audit_m14960/rerun_size_medium.json`、`rerun_size_medium.log` |
| 装置回归 | `artifacts/audit_m14960/pytest_shear_cell_friction.log`（10 passed） |
| 审计图 | `artifacts/audit_m14960/window_sensitivity.png` |
| 复算脚本 | `scripts/audit/m14960_cross_audit.py`、`scripts/audit/m14960_audit_figure.py` |
| 判级落表 | `tensor_maturity_verdicts`：repo=TensorDEM, scope='m14960 GAP-DEM-FRICTION', mission_id='14978', level=3, verdict_by='board-audit' |