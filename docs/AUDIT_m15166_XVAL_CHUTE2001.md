# 交差审计 · m15166 / XVAL-DEM-CHUTE2001（斜面颗粒流三方横向对比）

**审计单** m15245 · **对象单** m15166（succeeded）· **仓** TensorDEM · **被审 commit** `452cae2a`
（PR #1272 merged）· **审计方式** 独立复核（证据指针 + 独立重跑 + 受控 A/B 探针）
**裁定** 级别 **L2**（基础设施真代码）· 锚点判据 **0/3 未达** · 处置 **request changes / reopen**

---

## ① 审计对象与判据出处

| 项 | 内容 |
|---|---|
| 锚点 | `XVAL-DEM-CHUTE2001`（feature_ledger: open；acceptance = metrics 逐项：工具间差 + vs ref） |
| 案例 | 斜面周期槽道，d=1 mm，ρ=2500 kg/m³，k_n=500 N/m，e=0.88，μ=0.5，dt=1e-5 s（dt/T_c=0.044） |
| 交付物 | `src/tensordem/chute_flow/{spec,engine,analysis,runner}.py`、`tools/xval_chute2001/harness.py`、`tests/test_chute_flow.py`、`artifacts/xval_chute2001/result.json`、`runs/xval_chute2001/`（11 case 输入 + 4 个 ours result）、`docs/XVAL_CHUTE2001.md` |
| 判据（唯一裁判）| `tensor_xval_cases#5`（`XVAL-CHUTE`, status=**active**）：①稳态速度剖面工具间差 <15% ②Bagnold 指数各工具 ±20% 内一致 ③h_stop 两倾角点 端点 ±2° |
| **判据表缺口** | 审计单指定的 `tensor_benchmark_cases` 对本锚点 **0 行**（全表 14 行、无此锚点）→ 即使有数值也没有机器容差门可触发（F6） |

## ② 怎么做的（审计流程）

1. 读 m15166 工作树交付物 + `artifacts/xval_chute2001/result.json` + `docs/XVAL_CHUTE2001.md`，逐项核三件套（run_artifacts / ref vs got / repro_cmd）。
2. 查库：`tensor_xval_cases#5` 三项判据、`tensor_dev_missions#15166.result`（自报 checks）——自报对 WITHHELD 是诚实的（无粉饰）。
3. **独立重跑**（均在我方 m15245 工作树 / 本机单核，未用集群）：
   * `pytest tests/test_chute_flow.py -q` → **7 passed, 8.11 s**（L1 证据复现）；
   * `harness.py build --outdir /tmp/xvbuild` → rc=0，生成 11 个 case（入口可执行）；
   * `harness.py run --tool ours --cases arrest_t18_n300` → rc=0，但**数值不可复现**（见 F2）。
4. **受控 A/B 探针**（`artifacts/audit_m15166/probe_geometry.py`）：同一份共享几何 CSV，A=交付态（`case.json` 经 `ChuteEngine` 装载）、B=盒子修正态（lx=ly=10 d，即 case.json 记录值），各 3000 步、300 颗粒，比较是否漏底。
5. 读代码定位：`spec.py:161 from_dict` / `spec.py:63 DEFAULT_CASE` / `runner.py:_load` / `harness.py:DECK_TEMPLATE` / `analysis.py:bagnold_fit,h_stop_from_scan`。
6. 按 skill（adversarial-review）输出：分级 findings、复现命令、未跑项、夸大表述、处置。

## ③ 计算结果

### 3.1 三判据逐项（基准裁判）

| 判据 | tol | delivered got | 判定 |
|---|---|---|---|
| 稳态 vx(z) 工具间差 | <15% | 无稳态（`steady_A` `steady_detected_at_steps=null`，跑到 20000/60000 步）、无 OSS 数值 | **NOT_MET** |
| Bagnold 指数 p 各工具 | ±20% | `analysis.bagnold_fit` 有实现且单测回收 p=1.5，但**无任何 run 产出 bagnold_p**、无 OSS 数值 | **NOT_MET** |
| h_stop 两倾角点 | ±2° | arrest 扫描只 8.2 t_ref（settle=8000），θ=18/20° 均为 flow、无停机观测、无 OSS 数值 | **NOT_MET** |

**0/3 达成**，报告也如实标 WITHHELD——这一点上交付方**没有虚标**。

### 3.2 交付数值 vs 独立重跑（`arrest_t18_n300`）

| 量 | 交付（m15166） | 独立重跑（m15245） | 说明 |
|---|---|---|---|
| steps_total | 14 000 | 80 000 | 交付 `run_spec.json` settle=8000/window=6000；`harness.py:63` case_matrix 是 60000/20000 |
| state | flow | flow | 同 |
| v_rms/v_ref | 5.89 | **64.98** | 时间越长越大 → 自由落体特征，非剪切流 |
| vx/v_ref | 1.42 | **19.22** | 同上 |
| h_over_d | 1.0908 | 1.0908 | 两者都用了 **12 d 盒子**（错） |

### 3.3 受控 A/B 探针（3000 步，300 颗粒，同一份共享几何）

| 量 | A 交付态（盒子 12 d） | B 盒子修正（10 d） |
|---|---|---|
| 基床覆盖长度 | 9.43 d | 9.43 d |
| 盒子−基床覆盖（缝隙） | **2.574e-3 m ≈ 2.57 d** | 0.574 d（0.9 d 间距正常边缘） |
| z_min/d | **−3.54**（漏到基床下且持续下落） | **1.23**（稳稳坐在基床上） |
| 低于 基床顶−1d 的颗粒 | **37/300 = 12.3%** | **0/300** |
| v_rms/v_ref | 0.593 | 0.450 |

→ **交付态把 12.3% 的颗粒从周期基床的缝里漏出去了**；修正盒子后逐项恢复物理合理。

## ④ 图

![seam leak](../artifacts/audit_m15166/fig_seam_leak.png)

`artifacts/audit_m15166/fig_seam_leak.png`：同一份几何、同一初值、3000 步后 z/d 分布。
灰色（交付态，盒子 12 d）有 12.3% 颗粒落到 z<0（基床顶以下、且继续下落）；蓝色（盒子修正 10 d）无颗粒漏出。
**结论：交付数值的奇高 v_rms 来自"漏底自由落体"，不是槽道剪切流。**

## ⑤ 注意点（坑与局限）

1. **静默丢键**（F1，阻断级）：`case.json` 存的是 *summary* 键（`d_m/lx_over_d/kn_N_m`），`ChuteCase.from_dict` 只认 dataclass 字段名（`d/lx_d/kn`），于是整份 spec 被丢光、回落 `DEFAULT_CASE(lx_d=12)`——而共享几何 CSV 是 10 d 盒生成的。**"三方同一份几何文件"在自家这条腿上就不成立**。
2. **产物与代码漂移**（F2）：交付 `run_spec.json`（8000/6000）不是提交代码 `case_matrix`（60000/20000）的产物 → `repro_cmd` 能跑但复现不出交付数字。
3. **自家 deck 造成的 OSS 阻断**（F3）：`harness.py` 的 `DECK_TEMPLATE` 无条件写 `comm_modify vel yes`，而文档化的 3.8 serial 构建恰好 `Unknown command`（committed `log.txt` 就是这条）；文档引用的 `pair_gran.cpp:245` 上下文**没留在任何提交工件里**（3/3 个 `run.log` 都是 1 字节）。且从未尝试 MPI 构建（ghost 速度本就是 comm 属性）。
4. **文档自相矛盾**（F4）：①表 N=300 → h=1.57 d，④表同 case 写 1.09 d；真值 `300·(π/6)/100 = 1.571 d`。
5. **原始 dump 未入库**（F5）：steady 剖面 20 个 bin 的数值没有对应 `dump.csv`，无法从工件复算。
6. **局限**：本次未重建 LIGGGHTS（`build/` 被 .gitignore）、未装第二 DEM、未跑满 60000 步收敛；全部重跑在控制面单核（300 颗粒级），运行时间不代表生产能力。

## ⑥ 误差判定（逐项、显式回答"是否 <3%"）

| 关键量 | 基准/参考 | 交付值 | 误差 | <3%? |
|---|---|---|---|---|
| 层厚 h（N=300，10 d 盒） | 1.571 d（`300·(π/6)/10²`，可由颗粒数守恒闭式算出） | 1.0908 d | **−30.6%**（相对基准） | **否** |
| 盒子尺寸 Lx | 10 d（`case.json`、`chute.data` 均为 10 d） | 12 d | **+20%** | **否** |
| 颗粒漏底比例 | 0（基床应无洞） | 12.3% | 绝对 12.3 个百分点 | **否** |
| 判据①稳态剖面工具间差 | 无（无稳态、无 OSS） | — | 无法计算 | **无法判定** |
| 判据②Bagnold p | 3/2（Bagnold 标度）/ 无 OSS | 无 run 产出 | — | **无法判定** |
| 判据③h_stop | 无 OSS 端点 | 无观测 | — | **无法判定** |

> honest-FAIL 声明：三项基准判据一个数值也没取得，故本审计**不产出任何"工具间一致"结论**；
> 且已取得的"ours"数值被 F1 判定为**非该案例的物理量**（漏底自由落体），不能进任何对照表。

## 判级与处置

* **L1** 单元验证：`tests/test_chute_flow.py` 7 passed（我独立重跑复现）✔
* **L2** 独立功能实现：engine/spec/analysis/runner 真代码、无 placeholder、入口跑通、产出工件 ✔（本次裁定级别）
* **L3** 系统集成：✘ 唯一调用方是审计 harness（`tools/xval_chute2001/harness.py:241`），未接入 TensorDEM CLI/平台
* **L4** 基准验证：✘ 无 ref vs got、无误差分析、`repro_cmd` 不复现
* **L5** 科研工程：✘ 无外部锚点数值

**处置 = request changes（返修轮）**，返修项按 F1→F6 顺序：
① `case.json` 写 `asdict(case)`（或 `from_dict` 加 summary 别名）+ 往返回归测试；② 用代码重生成 case spec；③ 去掉无条件 `comm_modify`／改 MPI 构建跑 LIGGGHTS；④ 提交原始 dump；⑤ 注册 `tensor_benchmark_cases` 行；⑥ 再跑 steady A/B 与 arrest 扫描至 ≥60 t_ref 后逐项对判据。