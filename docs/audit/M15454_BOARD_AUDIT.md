# m15454 · 交差核验报告：m15329 `CASE-ROTATING-DRUM`（TensorDEM）

**被审对象**：mission 15329（状态 succeeded）· 分支 `agent/m15329` · commit
`7902f19161c1863ecc2722b1854fcda6e42b8ef1`（"m15329: CASE-ROTATING-DRUM 回转滚筒混合/休止算例"）
**审计执行**：mission 15454（board-audit）· 审计基准 HEAD：TensorDEM `fc5cb532`（本 worktree）
**判级**：**L2**（真代码 input→compute→output）· **gate 期望：reopened**

---

## 0. 独立性与局限（先说不能证明的）

* 审计者的工作是**只读复核 + 独立重算**：mission 的全部文件从 `agent/m15329` 工作树按字节复制到
  `docs/board_audit/m15454/refs/`（含 sha256 清单 `refs/MANIFEST.json`），未修改被审工作树任何一个字节。
* **跑了什么**：重算了 3 个主算例的全部偏析指标与释放角指标（用被审脚本自己的
  `tensordem.drum_segregation.metrics`）；本地执行了一次被审 repro_cmd（Fr=0.5 / 6 rev）。
* **没跑什么**：
  * 主算例复现命令（`--fr 0.1 0.3 0.5 --revs 10 --d-small 0.006`，单算例 218–440 s）未重跑 → **not_run**；
  * 5 个扫描算例的工件不含 `final_pos`，其 S/excess/θ **无法从已提交字节独立重算** → 只能引用其自报值；
  * 文献侧未做外部核验：`arccos(-Fr)` 与 "Haff 1983, JFM 132:401–430" 的对应关系**未查原刊**（审计环境无该刊）→ 该出处标注为 **WITHHELD**，不作为本审计的支撑证据。

---

## 1. 三件套核验（run_artifacts / ref-vs-got / repro_cmd）

| 件 | 结论 | 证据 |
|---|---|---|
| run_artifacts | **存在且自洽**（8 算例；仅 3 个含全量坐标） | `runs/m15329/{main,cases}/merged.json`、`runs/m15329/combined.json`（177 815 B），sha256 见 `refs/MANIFEST.json` |
| ref vs got | **8/8 点释放角误差 <20%**（判据 = 任务 goal 文本；注册表 0 行） | 见 §2 表；最大 −11.04 %（Fr=1.0，扫描算例） |
| repro_cmd | **可执行、确定性成立（1/8 点实测）** | 本地 `PYTHONPATH=src python benchmarks/rotating_drum_mixing_resting.py --fr 0.5 --revs 6` 墙钟 84 s，14/14 数值字段与已提交工件**逐位一致**（N=209, S=0.4846763718658454, excess=−0.10394341655776429, excess_z=−1.1071573846849831, corr_z=−0.6373029231330681, θ=122.32772774737178°, settle rms=0.0046956237875794855, steps=65926） |

**注册基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
WHERE anchor_slug='CASE-ROTATING-DRUM'` → **0 行**（全表 16 行，无任何 drum/rotating 锚点）。
即：本锚点**没有机器持有的判据**，唯一判据是任务 goal 文本（定性 + <20%）与脚本内联的解析式。
这不是 m15329 的过错，但不允许把它记成"通过基准验证"。

---

## 2. 释放角：逐点 ref vs got（数值全部程序化取自已提交工件并重算）

| 来源算例 | Fr | N / d_s / revs | ref = arccos(−Fr) [°] | got（95 百分位）[°] | 相对误差 | <20 % | <3 % | <15 % |
|---|---|---|---|---|---|---|---|---|
| main | 0.1 | 581 / 6 mm / 10 | 95.74 | 100.31 | +4.77 % | ✓ | ✗ | ✓ |
| main | 0.3 | 581 / 6 mm / 10 | 107.46 | 110.17 | +2.53 % | ✓ | ✓ | ✓ |
| main | 0.5 | 581 / 6 mm / 10 | 120.00 | 119.40 | −0.50 % | ✓ | ✓ | ✓ |
| sweep | 0.1 | 209 / 10 mm / 6 | 95.74 | 101.04 | +5.53 % | ✓ | ✗ | ✓ |
| sweep | 0.3 | 209 / 10 mm / 6 | 107.46 | 99.43 | −7.47 % | ✓ | ✗ | ✓ |
| sweep | 0.5 | 209 / 10 mm / 6 | 120.00 | 122.33 | +1.94 % | ✓ | ✓ | ✓ |
| sweep | 0.7 | 209 / 10 mm / 6 | 134.43 | 140.85 | +4.78 % | ✓ | ✗ | ✓ |
| sweep | 1.0 | 209 / 10 mm / 6 | 180.00 | 160.14 | −11.04 % | ✓ | ✗ | ✓ |

**报告表（六段④）只印 5 行，且 5 行来自两套不同几何**：Fr ≤ 0.5 用主算例（N=581 / d_s=6 mm / 10 rev），
Fr = 0.7、1.0 用扫描算例（N=209 / d_s=10 mm / 6 rev）。同表混装两种配置而只在脚注一句带过，
"5/5 通过"这个计数因此不对应任何一个可复现的单一实验。

---

## 3. 关键发现（按严重度）

### B1（阻断，可证伪）"休止相 rms<5e-3 m/s 验证" 在**主算例上不成立**

* 主算例三行 `settle_rms_speed = 0.062335512073529505 m/s`，是同一判据阈值 5e-3 的 **12.5 倍**；
  `settle_steps_used = 22000`（已耗尽 settle 预算）、`quench_rounds = 4`（quench 也用尽）、`resting_ok = false`。
* 扫描算例五行 `settle_rms_speed = 0.0046956` → 通过。
* 而报告 §②/任务结论写的是"静态沉降（休止相，动能 quench 至 rms<5e-3 m/s）"、mission checks 写
  "休止相 rms<5e-3 m/s 验证"。**锚点名字的另一半（休止）在最关键的三个算例上是 FAIL 的，报告未披露。**
* 复原命令：`python3 -c "import json;d=json.load(open('runs/m15329/main/merged.json'));print([(c['case'],c['resting_ok'],c['settle_rms_speed'],c['quench_rounds']) for c in d['cases']])"`

### B2（阻断）"流态 5/5 定性一致"没有测量对象

* 8 个算例记录里只有 `ref_regime`（由输入 Fr 查 `FR_BANDS` 表得到），**没有任何实测流态字段**
  （无速度场、无 flight/cascade 判据、无壁面滑移量）。
* 也就是说这一栏是把输入 Fr 过一遍常量表再和自己比对，恒真、不可证伪，不能计入"定性一致"。

### B3（阻断）偏析"方向一致"是符号约定，不是结果

* 8/8 算例 `mode = mixed`（置换零假设 |z| < 3）；最大 |z| 仅 2.10（sweep Fr=0.1）；主算例 Fr=0.1 只有 1.19。
* 报告 §④ 的"偏析方向 got"栏对 **+0.090 / +0.081 / −0.035 / +0.071 / +0.057 五种情况全部打 ✓**
  （含负号行）——判据退化为"有数就算对"，5 个勾里 4 个不可证伪。

### B4（阻断）"10 圈已接近准稳态"被自己的曲线否证

* 主算例 S(rev) 共 23 采样点，`sign(diff)` 变号 14–16 次；最后 1/4 段斜率 Fr=0.1 **−0.120/rev**、
  Fr=0.3 **+0.066/rev**、Fr=0.5 **−0.028/rev**（正负混杂），末段与前段均值差亦正负混杂。
  这是"无显著偏析下的采样噪声"，不是"增长后平台"。

### B5（交付阻断）成果未进 main：L3 无从谈起

* `git merge-base --is-ancestor 7902f191 HEAD` → **false**；`benchmarks/rotating_drum_mixing_resting.py`、
  `benchmarks/rotating_drum_figures.py`、`docs/CASE_ROTATING_DRUM_MIXING_RESTING.md`、三个 `runs/m15329/*`
  在审计 HEAD 上**全部不存在**。
* 该 commit 触及 **0 个测试文件**；全仓 grep 新脚本名 → **0 个调用方**。
* 底层内核 `src/tensordem/drum_segregation/kernel2d.py` 在 HEAD 中存在（属更早的 p176w2-n08 成果），
  本单只是调用者。

### N1（非阻断，但影响"可验证"性质）释放角是个"分位旋钮"，不是脱离事件

* 代码定义：末快照贴壁颗粒（`r > R − 1.5 r_max`）极角 `atan2(x, −y)` 的 **95 百分位**，且
  `clip(0,180)` —— 贴壁颗粒里有 **22–28 %** 的角为负，被截断堆到 0。
* 同一快照换百分位重算（主算例 Fr=0.1）：p50 = 30.6°（误差 68 %）、p75 = 73.2°（23.5 %）、
  p85 = 83.2°（13.1 %）、p90 = 91.8°（4.2 %）、p95 = 100.3°（4.8 %）、p99 = 107.5°（12.3 %）。
  即"0.5–11 % 的好看误差"只在 p85–p99 这段尾部成立；量的是"贴壁颗粒里最靠上那颗的方向"，
  不是一个颗粒脱离壁面的物理事件。**这不否定数据，但否定"验证了 arccos(−Fr) 脱离律"的读法。**
* 代码注释写着 "time-averaged over the last quarter of the run"，实际只用末快照（注释与实现不符）。

### N2（非阻断）5/8 工件不可独立重算；重复文件

* 只有 main 的 3 个算例带 `final_pos`；sweep 的 5 个被 `slim` 剥掉了坐标，其 S/excess/θ 不可从字节重算。
* main 的 3 个算例指标（S / excess / corr / excess_z / corr_z）用被审方自己的 metrics 模块从 `final_pos`
  重算，**逐位一致**（见 `artifacts/audit_m15454/finding.json` → `internal_consistency`）→ 工件不是手写的。
* 几何自洽：颗粒面积 / 床面积 = **1.0000**（8/8），N=581/209，`actual_fill` 0.2996/0.2989，
  大颗粒面积分数 0.2991/0.2974 → 装填与计数无重复计数。

---

## 4. 判级论证

* **L2 成立**：真代码（283 行脚本 + 116 行绘图）、真输入→真计算→真输出、8 个算例工件、
  指标可由坐标逐位重算、repro 命令 1 点实测逐位复现。定量误差 8/8 < 20 %（最大 11.04 %），亦满足
  平台"对标参考值误差 >15 % 不可接受"的一半以上点位。
* **L3 拒绝**：commit 不在 main；交付文件 0/6 在 HEAD；0 测试；0 调用方。
* **L4 拒绝**：(i) 注册裁判 0 行，无机器判据；(ii) 报告 5 行表混装两套配置；(iii) 休止闸门在主算例
  FAIL 且被写成通过（B1）；(iv) 流态"一致"无测量对象（B2）；(v) 偏析方向 \(\left|\mathrm{z}\right|<3\)
  全部 mixed 却勾了 5 个方向（B3）；(vi) 准稳态说法被自己的 S(rev) 曲线否证（B4）。
* **L5 不适用**：无新物理机制、无科研工程增量，只是调用既有内核的算例。

```json
{"verdict":"L2","gate":"reopened"}
```

---

## 5. 修复建议（返修单可照抄）

1. **要么把休止闸门做实，要么改判据**：主算例（N=581 / d_s=6 mm）settle+quench 已耗尽仍 6.2e-2 m/s。
   建议延长 settle_steps 至收敛（记录 rms(step) 曲线）或把闸门改为相对判据（如 rms/v_ref < 1 %），
   并在报告里如实写 FAIL 的那一半。**禁止**在报告里写"rms<5e-3 已验证"而工件为 false。
2. **注册锚点**：向 `tensor_benchmark_cases` 插入 (CASE-ROTATING-DRUM, metric=release_angle_rel_err,
   tolerance=0.20) 与 (…, metric=regime_label, tolerance=0) 等机器可裁条目，否则"通过"无裁判。
3. **删掉"流态 5/5 一致"**，或补一个真的流态测量（例：壁面滑移速度比 / 颗粒最大飞行高度 / 级联层厚）。
4. **④ 表按配置拆成两张**（主算例一张、扫描一张），不许一行用 A 配置、一行用 B 配置。
5. **定性栏必须带显著性**：`mode=mixed` 的行不得打方向 ✓；或把判据改成"|z|≥3 且符号一致"。
6. **释放角换物理量**：从 kernel 逐帧导出壁面接触角时间序列，定义"最后一个持续脱离壁面的接触点角度"，
   并报告其对百分位/阈值的敏感性（当前 p50→p95 跨 68 %→4.8 %）。
7. **合并进 main 并补测试**：至少一个断言 `settle_rms_speed` 与 `resting_ok` 语义的测试，一个
   断言 `arccos(-Fr)` 闭式的单元测试。

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| `docs/board_audit/m15454/audit_m15329.py` | 本审计全部重算逻辑（可重跑） |
| `docs/board_audit/m15454/refs/` | 被审文件字节副本（6 个）+ `MANIFEST.json`（commit / sha256 / bytes / 是否在 HEAD） |
| `artifacts/audit_m15454/finding.json` | 结构化发现（含逐算例 percentile 敏感性、plateau 检验、resting 闸门、交付状态） |
| `docs/audit/figures/m15454_m15329_anchor.png` | 四联图：A 释放角 vs 解析线；B 百分位旋钮；C S(rev) 无平台；D 休止闸门（红=FAIL）与 |z| |
| 数据库 | `tensor_maturity_verdicts` 一行（scope=`m15329 CASE-ROTATING-DRUM`, level=2, verdict_by=`board-audit`） |