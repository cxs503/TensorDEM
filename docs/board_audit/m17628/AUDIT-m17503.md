# 任务榜交差审计 — m17503 `XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE`（TensorDEM）

**审计单** m17628（board-audit，非重做）｜**被审单** m17503（status=succeeded，commit `ad072d30`，分支 `agent/m17503`）
**审计对象**：TensorDEM `src/tensordem/mts_contact.py` 的 Lees–Edwards / 周期 SLLOD 交叉对标交付
**基准裁判**：`tensor_benchmark_cases`（唯一裁判）→ **本锚点 0 行**（SQL 见 `verification.json` C5）
**裁定级别**：**L3（系统集成）**，不是 L4 —— 理由见 §5

---

## 1. 审计范围与独立性

| 项 | 内容 |
|---|---|
| 审计独立面 | 只读被审工作树 `/root/tp-agent/work/m17503`（审计期间 `git status` 干净），另在 `/tmp/audit_m17503` 的**副本**里复跑，未改被审工作树内容（复跑后已 `git checkout` 还原唯一被改的 `wall_clock_total` 字段） |
| 复跑内容 | ① `pytest tests/test_lees_edwards_sllod.py`（5 passed, 1 xfailed, 5.98 s）② `pytest tests/test_lees_edwards_sllod.py tests/test_mts_contact.py`（**30 passed, 1 xfailed**, 64.93 s）③ `xval_matrix.py --only-invariants`（不变量逐位一致）④ `xval_matrix.py --gammas 10 --gamma-end 0.2 --no-sens`（legacy/LE 两臂 `tau/P`、`P` 逐位一致）⑤ 独立探针 `refs/probe/probe_overlap.py`（重叠量级） |
| 未跑（如实列，不记为 passed） | ① 全仓 `pytest`（仅跑与改动相关的两个文件）② 完整 3 剪率矩阵（只复跑 γ=10，另两率按同代码同种子推定可复现，未逐一复跑）③ 时间步敏感性（`sensitivity` 工件为空，见 F3，无法复跑）④ OSS（LAMMPS/ESPResSo/HOOMD）源码级逐行对标（被审单自记 WITHHELD，本审计同样未做）⑤ 与 `main` 的逐位回归基线对照（见 F5） |
| 环境 | loongarch64 本机，Python 3.11；被审单 `head_commit` 为 null、`git_status` 空，身份只由工作树 HEAD=`ad072d30` 绑定 |

## 2. 三件套核验（每项都指向可机检物）

| 三件套 | 结论 | 证据 |
|---|---|---|
| **run_artifacts** | **PASS** | `refs/artifacts/results.json`（3561 B）、`tables.json`（1200 B）、`xval_periodic_sllod.png`（93739 B，PNG magic 校验通过）；sha256 见 `verification.json#C1` |
| **benchmark ref vs got** | **FAIL（裁判缺位 + 判据未达）** | ① `SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE'` = **0** → 无 ref/容差判据行；② 被审单自定判据 <15 %，实测两臂 `tau/P` 相对偏差 **361.0 % / 162.7 % / 654.8 %**（γ=10/20/40），全部超差；③ 外部锚点=WITHHELD（其报告 §1 自述 OSS 源码未能在本环境核验） |
| **repro_cmd** | **PASS（有保留）** | 审计独立复跑复现：legacy `tau/P=+0.1082, P=3.315e+00`，LE `tau/P=-0.0415, P=1.045e+03`，`coord 5.90/5.76`，`strain_end 0.000/0.201` —— 与 `results.json` 一致；不变量块（`n_contacts 335/338`、`frame_invariance 1.363e-02/3.568e-01`、`rate_error 9.3432/8.8240`）逐位一致。**保留**：报告 §6 印出的复现序列缺 `--only-invariants` 一步，照抄执行会把 `invariants` 写成 `{}`、`sensitivity` 也留空，见 F3 |

**程序化核对（零手抄）**：`tables.json` 的 cross 表由 `results.json` 的 `cases` 独立重算，三行差值全对（361.0/162.7/654.8，误差 <0.05 pp）；不变量表的 straddle fraction 与 `Ly/|<l_y>|` 公式亦独立重算一致（`verification.json#C2/C3`，全 `match=true`）。而 `tables.json` 本身确由 `make_report_artifacts.py` 从 JSON 渲染（`refs/impl/make_report_artifacts.py`），**报告数字非手抄**这一条成立。

**报告六段**：①案例介绍+参数表 ②方法与实现 ③计算结果 ④收敛性/敏感性 ⑤误差判定（逐项显式回答 <3 %）⑥失败模式与工件 —— 六段齐全（`verification.json#C4`，六段标题全命中；`<3 %` 显式作答存在）。附加分：报告开头即 honest-FAIL 结论表（R1–R4）并给出失败模式 F1–F3，未粉饰。

## 3. 严重度排序的发现

| 级别 | 发现 | 证据 |
|---|---|---|
| **blocking（对 L4，不影响 L3）** | **F1 判据未过且无裁判行**：锚点自定 <15 %，实测 163–655 %；`tensor_benchmark_cases` 无本锚行，故 L4 的「基准数值对照」既缺裁判也未被满足 | `verification.json#C2/C5/C6`；`results.json` |
| **blocking（能力可用性）** | **F2 LE 臂输出被近似奇异重叠支配**：审计独立探针（同参未加松弛、909 步、n=64、φ=0.80、γ=10）显示 legacy 臂最大重叠 **0.02 % r**（0 对 >10 % r），LE 臂最大重叠 **14.65 % r**、**5 对** >10 % r；LE 臂 `P=1604 Pa / τ=-258.8 Pa`（τ/P=-0.161，负摩擦）几乎全部由这几对深穿透接触贡献。即：被审单报告的 LE 臂宏观量**不能充当参考真值**，与报告自述 F1（像选择仍取共动系，帧不变性残差 0.357）一致，但被审单自身的工件并未量化这一点（其 `invariants` 只在静态 jammed 胞上测，测不到动力学中的深穿透） | `refs/probe/probe_overlap_out.txt`；`results.json#invariants`；图 (c) |
| **medium（provenance）** | **F3 工件与复现命令不自洽**：`results.json.wall_clock_total = 0.0249 s`，而 6 算例矩阵实跑约 95 s（审计测 γ=10 单率 33.8 s/2 臂）→ 该字段被**增量写**覆盖成最后一次调用的计时；`sensitivity = []`，报告 §4 引用的时间步敏感性（τ/P 0.2381→0.2439, 2.4 %）**无工件、不可复跑**（报告自述「未随本次矩阵留档」）；§6 复现序列漏 `--only-invariants` | `results.json`；报告 §4/§6；本审计 §2 表 |
| **low（表述夸大）** | **F4「934 % 是解析精确值、非统计」**：该数是解析式 `Ly/⟨|l_y|⟩` 落在**实测**跨边界接触集合上的值（legacy 9.3432 / LE 8.8240，两臂不同即证其依赖构型），确定性可复现但不是与构型无关的解析常数；措辞宜改为「解析式的确定性求值」 | `refs/impl/xval_matrix.py:123-128`；`invariants` |
| **low（计数）** | **F5 回归计数**：被审单称两测试文件「31 passed」，实为 **31 collected = 30 passed + 1 xfailed**（xfail 即 F1 的记录位） | 本审计复跑输出 |
| **low（集成面）** | **F6 无生产调用方**：`lees_edwards` 仅被 `scripts/m17503/*` 与 `tests/*` 置真；仓内唯一 `shear_cell_run` 调用方 `benchmarks/mts_contact_gate.py:142` **未传该开关** → LE 路径属「主求解器内的一等公共 API，但产品路径未行使」 | `grep -rn lees_edwards src/benchmarks/`；`refs/impl/mts_contact.py:1422,1556` |

**被审单位点属实、经审计独立复核为真**（不是走过场）：
* I1 倾斜晶格最小像：`test_minimum_image_uses_tilted_lattice` 与 9×9 暴力格点搜索逐元素 `atol=1e-15` 对齐（审计复跑通过）；
* I2 跨 y 边界仿射速度补正：`test_affine_velocity_continuity_across_the_y_boundary` 通过，单接触级精确；
* I3 帧不变性：legacy 臂被断言必须破（1.363e-02），LE 臂 `xfail` 诚实登记未修好的像选择缺陷（0.357）；
* 默认关闭、零剪切路径逐位不变（`test_zero_shear_rate_is_identical_with_and_without_lees_edwards`），既有 25 项 `test_mts_contact.py` 无回归。

## 4. 判级（L0–L5）

**L3 = 系统集成。** 逐级判据：

| 级 | 判据 | 结论 |
|---|---|---|
| L1 | 数学/单元验证 | ✅ 6 项不变量测试，5 pass + 1 xfail（缺陷登记）；I1 与暴力搜索对拍 |
| L2 | 独立功能实现（输入→计算→输出） | ✅ `src/tensordem/mts_contact.py` 内真实现（triclinic `_minimum_image` 5 像扫描、`_pairs_disk` 共动系映射、`_forces` 仿射速度补正、virial 用真 branch、`strain` 双推进点），入口 `scripts/m17503/xval_matrix.py`，工件齐、可复跑 |
| L3 | 系统集成（接入主求解器/平台） | ✅ 改动落在**主求解器模块本身**，`MTSSystem(lees_edwards=)`/`shear_cell_run(lees_edwards=)` 为 `__all__` 导出的公共 API，`shear_cell_run` 被仓内 benchmark 门调用（但不传该开关，F6）；默认关闭、既有路径逐位不变 |
| L4 | 基准验证（完整算例+误差分析+可复现） | ❌ **不达**：`tensor_benchmark_cases` 本锚 0 行（无裁判）；自定 <15 % 判据实测 163–655 % FAIL；外部锚点 WITHHELD；且 LE 臂被深穿透接触支配（F2）→ 没有可用的参考真值，"crosscheck"只证明了**两实现不一致且差异可解释**，未产生**通过验收的基准对照** |
| L5 | 科研工程成果 | ❌ 不达（无外部锚点、无通过判据的基准） |

**级别的实质**：本交付的真实价值不在「对标通过」，而在**把两条此前无人证明的周期 SLLOD 缺陷钉成可机检不变量并修好其中一条**（I2 单接触级精确，I1 与暴力搜索对拍通过），同时诚实登记了未修好的 I3（像选择）。这与「锚点名 `...-CROSSCHECK-AVAILABLE`」相符（交叉对标**可用**），但与「判据 <15 %」（**未达**）必须分开记账：判据不达 → 不升 L4；功能实质（in-solver 公共 API + 不变量级验证 + 可复跑工件）→ 已越过 L2，落 L3。

## 5. 处置与返修建议

**Disposition：approve with notes（L3 confirmed），并建议后继单**：
1. **F2（主因）**：把像选择的 argmin 搬到真实倾斜晶格（`_minimum_image` 已有该逻辑），同时把切向历史键 `(i, j, image slot)` 改为「物理接触键」，修掉 I3 的 0.357 → 才有资格谈 L4；
2. **F3**：`wall_clock_total` 改为累加式（增量写不能覆盖总计时），把时间步敏感性跑一遍并落 `sensitivity` 工件，复现块补 `--only-invariants`；
3. **F2 的量化**：把「最大重叠 / 深穿透接触数」做成与 `tau/P` 并列的输出字段（本次审计的探针即最小可用版本），否则读者无法判断 LE 臂宏观量是否可信；
4. 要做 L4，需先向 `tensor_benchmark_cases` 登记一行（case_name/metric/tolerance/reference），否则任何 XVAL 都只能停在 L3。

## 6. 工件与复现

```
docs/board_audit/m17628/verify_m17503.py           # 本审计的独立核验脚本（C1-C6）
docs/board_audit/m17628/verification.json          # 核验结果（含 sha256、SQL、逐行重算）
docs/board_audit/m17628/refs/impl/mts_contact.py   # 被审实现（m17503 版，1763 行）
docs/board_audit/m17628/refs/impl/xval_matrix.py   # 被审对照矩阵入口
docs/board_audit/m17628/refs/tests/test_lees_edwards_sllod.py
docs/board_audit/m17628/refs/artifacts/{results.json,tables.json,xval_periodic_sllod.png}
docs/board_audit/m17628/refs/probe/probe_overlap.py + probe_overlap_out.txt   # 独立重叠探针
docs/board_audit/m17628/fig_m17628.png            # 图：(a) tau/P 两臂 (b) 判据偏差 (c) 重叠探针
```
复现：`python3 docs/board_audit/m17628/verify_m17503.py /root/tp-agent/work/m17503`；
被审件复跑见 §1 复跑内容（`PYTHONPATH=src python scripts/m17503/xval_matrix.py --gammas 10 --gamma-end 0.2 --no-sens`）。