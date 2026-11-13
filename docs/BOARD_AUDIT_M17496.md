# 交差核验 · m17455 = XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT（TensorDEM）

- 审计单：m17496（board-audit，非重做）；被审单：m17455（succeeded，TensorDEM，worktree `/root/tp-agent/work/m17455`，branch `agent/m17455`）
- 唯一裁判：`tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT'` → **0 行**
- 落表：`tensor_maturity_verdicts` id=**572**，`repo=TensorDEM`，`scope=17455 XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT`，`level=2`，`verdict_by=board-audit`，`gate_state=pending`（本域不自行改门状态，交 maturity_gate）
- 结论一句话：**机制是真的、数字是真的、可复现是真的；但没有一行提交、没有一个消费方、没有登记裁判行 → L2，不是 L3/L4。**

---

## ① 案例介绍（被审对象是什么）

m17455 的任务是 XVAL（交叉对标）：把 TensorPhysics 的 `library-binding-frozen-contract` 做法移植到 TensorDEM 的**供应方**一侧——即 TensorDEM 对外暴露的求解器绑定面（时钟、载荷形状/单位、瞬时非累加载荷语义、整数子步规则、重启入口、负向断言）从"散落在 docstring 里的隐含假设"变成**声明式 + sha256 钉扎 + 内省校验 + 行为探针**的冻结契约。

| 项 | 值 |
|---|---|
| 锚点 | `XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT` |
| 契约 schema / 版本 | `tensordem.binding-contract/1` / `1.0.0` |
| 冻结指纹 | `sha256:0cbfed8c1172f8412dedfc013e87c79f56f0567ca22d782e0387818f0ee4105d` |
| 环境 | Python 3.11.6 / torch 2.7.1 / linux loongarch64 / 单核，脚本墙钟 1.68 s（无集群作业） |
| 判据来源 | 本仓 `prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT.json`（8 条，含 1e-12 阈值 + 停止规则）；**平台裁判表无本锚点行** |

## ② 怎么做的（本次核验的方法）

审计脚本 `docs/board_audit/m17496/audit_m17455.py`，全部从已落盘字节出发，唯一执行的命令是被审单自己登记的复现命令（输出重定向到 `/tmp/m17496_repro`，不碰 mission 文件）：

1. **三件套取证**：11 个交付路径是否存在 → 逐文件 sha256 + 字节数，落 `docs/board_audit/m17496/refs/`（byte-exact 拷贝 + `MANIFEST.json`）。
2. **交付链**：`git status --porcelain`、`git cat-file -e HEAD:<path>`、`git log --all --grep=17455`、以及本仓 HEAD 是否含该文件。
3. **登记裁判查询**：`tensor_benchmark_cases` 按锚点查（结果 0 行，全库共 54 个登记算例）。
4. **独立复现**：重跑 `PYTHONPATH=src python3 scripts/xval_library_binding_contract.py --outdir /tmp/m17496_repro`，对 `result.json` 做**递归深比对**（float 容差 1e-12，仅排除 `wall_s` 计时字段）。
5. **测试完整性**：读 `tests/test_binding_contract.py` 判其断言的是不变量还是仅"跑一遍"，并重跑。
6. **裁判器对抗探针**（审计方自造，不改仓）：把契约里 `external_forces.units` 从 `N` 改成 `kN` 后调 `verify(contract=...)`/`assert_conforms()`，看指纹失配是否被状态位拦截；另造一个被 `del positions` 的活体实例，看 live 漂移是否被抓到。
7. **OSS ref 零手抄核验**：`TensorPhysics` 侧被引文件的**行数**与负向断言是否与报告引用一致。
8. **集成性**：`grep -rln binding_contract src/ benchmarks/ scripts/ tests/`，找 solver/平台侧消费方。

## ③ 计算结果（核验数字表，程序化产出）

**表 3.1 三件套与交付链**

| 检查 | 结果 | 证据 |
|---|---|---|
| run_artifacts 存在 | ✅ 11/11 路径落盘（result.json 19254 B sha256 `abe0efb1…`，4 张 png，run.json 776 B） | `refs/MANIFEST.json` |
| ref vs got 数值 | ✅ 8 条判据表 + 冲量扫描 n=1..128 + 7 行变异矩阵 + 19 条探针明细 | `refs/…__result.json` |
| repro_cmd 可执行且可复现 | ✅ exit 0，**0 处数值差异**（仅 wall_s 不同） | `artifacts/audit_m17496/finding.json: repro_diff` |
| 已提交到分支/HEAD | ❌ 7 个交付路径全为 `??`（untracked）；`git log --all --grep=17455` 无提交；HEAD 无 `binding_contract.py` | `git_state` |
| 登记裁判行 | ❌ 0 行（54 个登记算例中无本锚点；XVAL 类仅有 `XVAL-FEM-TPB1976`） | 表查询 |

**表 3.2 被审单自报判据 vs 审计复核**

| 判据（预登记） | ref | got（审计复跑） | 阈值 | 判定 |
|---|---|---|---|---|
| 冲量-动量相对误差（2-D，n=1..128 max） | 解析 n·dt·F | 2.66e-16 | 1e-12 | ✅ 过（6 个数量级余量） |
| 冲量-动量相对误差（3-D） | 解析 n·dt·F | 3.56e-16 | 1e-12 | ✅ |
| 诊断台账 vs 解析冲量 | 解析 n·dt·F | 1.77e-16 | 1e-12 | ✅ |
| 原生时钟 vs n·dt | 解析 n·dt | 1.91e-16 | 1e-12 | ✅ |
| 重启位置漂移 | 0 m（bitwise） | 0.0 | 0.0 | ✅ |
| 7 类漂移检测率（契约检测器） | 7/7 | 1.000（7/7） | 1.0 | ✅ |
| 19 条语义探针通过率 | 全过 | 19/19 | 1.0 | ✅ |
| 内省检查符合率 | 全部冻结项 | 42/42 | 1.0 | ✅ |
| 对照基线（仅名字清点） | — | 1/7 = 0.143 | — | 契约 1.0 vs 基线 0.143，Δ=0.857 |

**表 3.3 对抗探针（审计方独立自造）**

| 探针 | 期望 | 实测 | 判读 |
|---|---|---|---|
| 契约 units `N`→`kN` 后 `verify(contract=…)` | 应报漂移 | `status=CONFORMS`，`breaking=0`，`frozen_fingerprint_match=False` | ⚠ 见 F2：指纹失配只挂在辅助字段，不进入状态判定 |
| 同上 `assert_conforms()` | 应抛 | 不抛（用默认契约时 fp_match=True） | ⚠ 同上 |
| 活体实例 `del positions` 后 `verify(instances=…)` | 应抓 | `status=DRIFT`，`breaking=2` | ✅ 活体漂移真被抓 |
| OSS ref 行数（报告claim 113 / 43） | 一致 | 113 / 43，且 `test_adapter_inventory_meta.py` 确有 `assert not hasattr(Solver,'restore')` | ✅ 零手抄可核 |
| solver/平台侧消费方 | 至少 1 | 0（仅 `tests/` + `scripts/`） | ❌ L3 不成立 |

## ④ 结果图

`docs/audit/figures/m17496_m17455_audit.png`（三栏，全部程序化读自 `result.json` 与审计复核值）：

- 左：7 类受控变异下「契约检测器 vs 仅名字基线」逐类命中（基线只在 `state_attribute_removed` 命中 1 次）；
- 中：冲量-动量相对误差 vs 子步数 n（重跑值），叠画预登记 1e-12 与 owner 15 % 线——误差与 n 无关，是浮点舍入量级；
- 右：本次核验的 yes/no 门（复现=yes、已提交=no、消费方=no、登记裁判=no）——这张图就是判级为 L2 的依据。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **"可复现"≠"已交付"**：本单最硬的问题不是数字，而是**交付链断了**——7 个路径全 untracked，`head_commit=null`，自动推送无提交可推。证据链只活在一个 worktree 里，回收即失。这是 F1（阻断级）。
2. **指纹钉扎没有进入状态判定**（F2）：`verify()` 里 `frozen_fingerprint_match` 只是报告字段，`status` 仅由 `breaking` 决定；审计方把 unit 常量改掉后仍得 `CONFORMS`。绑定方若只看 `status`，声明面被悄悄改（如 N→kN）不会报警。建议：`frozen_fingerprint_match=False` 时 `status` 至少降为 `SUSPECT` 或让 `assert_conforms()` 抛。
3. **检测率 1.0 是上界不是现场率**（F4）：7 类变异由同一个模块作者定义，检测器与变异集同源（循环论证），因此 1.0 不能外推为"现场漂移都能抓"。真正的外部锚只能来自真实的接口变更史。
4. **冲量恒等式近乎平凡**（F5）：自由飞行线性积分器上 `m·Σdv = n·dt·F` 本就精确到舍入，误差与 n 无关（1e-16 平线）。它验证的是**时钟/载荷记账/非累加语义**，不是接触物理。被审单自己在 §6.2 与 ⑥ 写明了这点，属如实申报而非夸大。
5. **3-D 冻结面只覆盖 40%**（F3，8/20）：已有披露；未冻结的 3-D 物理诊断被绑定方依赖时仍可静默漂移。
6. **本锚点没有物理量**：owner 的"误差>15% 不可接受"在本锚点上无物理主体（对标的是机制/做法），因此 15 % 只能套在记账恒等式上（实测 1e-16），不能宣称"物理精度达标"。
7. **裁判表缺行**：`tensor_benchmark_cases` 无本锚点 → L4 无法机器裁定。这属平台侧供给缺口（锚点未登记判据行），审计不代填。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 冲量-动量（2-D，max over n=1..128） | 解析 n·dt·F | 2.66e-16 | ✅ 是（远超要求） |
| 冲量-动量（3-D） | 解析 n·dt·F | 3.56e-16 | ✅ 是 |
| 台账 vs 解析冲量 | 解析 n·dt·F | 1.77e-16 | ✅ 是 |
| 原生时钟 vs n·dt | 解析 n·dt | 1.91e-16 | ✅ 是 |
| 重启漂移 | 0 m | 0.0 m（bitwise） | ✅ 是 |
| 漂移检测率 | 7/7 期望 | 0 漏检，基线漏 6 | ✅ 是（但见 ⑤.3 的适用边界） |
| 审计复现值 vs 归档值 | 归档 result.json | 0 处差异（仅 wall_s） | ✅ 是 |

**判级（L0–L5）**：`L2`。

- L1 ✅ 数学/单元验证（解析恒等式 + 不变量测试）；
- L2 ✅ 独立功能实现（真代码：活体类内省 → 漂移分类 → 输出报表；7 类变异真造真检；结果 bitwise 可复现）；
- L3 ❌ 系统集成缺失（0 消费方、0 提交、无 CI/门消费 `assert_conforms()`）；
- L4 ❌ 基准验证缺失（登记裁判 0 行；数值基准是自洽恒等式，非独立外部基准）；
- L5 ❌ 不适用。

## ⑦ 处置建议

- 被审单侧（返修即可升 L3）：把 7 个交付路径 **commit 到 agent/m17455**（自动推送才有东西可推）；补 1 个真实消费方（例如在 `IceDEM`/`IceDEM3D` 的 `__init__` 或测试夹具入口调用 `assert_conforms()` 做开箱自检，或在 CI 里跑指纹钉扎）；`verify()` 把 `frozen_fingerprint_match=False` 纳入状态（F2）。
- 平台侧：为本锚点登记 `tensor_benchmark_cases` 判据行（如 `impulse_rel_err ≤ 1e-12`、`detection_rate = 1.0`），否则 L4 永远无法机器裁定，交差核验只能停在"自登记判据"。
- 审计侧局限：本次未重做物理对标（锚点无物理量）；未审 TensorPhysics 侧适配器自身的正确性；对抗探针为 2 项（指纹/活体），未穷举全部契约字段；审计方已用平台库账号补查 `tensor_xval_cases`（7 行，锚点为 CHUTE2001/COOK/TPB1976/SOD1978/CAVITY/TAYLORGREEN/DAMBREAK）——**同样没有本锚点行**；`tensor_benchmark_cases` + `tensor_xval_cases` 两表合查仍为空，故"裁判为空"结论成立。

---

### 附：本次审计产物清单

| 路径 | 内容 |
|---|---|
| `docs/board_audit/m17496/audit_m17455.py` | 核验脚本（拷贝/MANIFEST/复现/深比对/对抗探针/图） |
| `docs/board_audit/m17496/refs/**` | 11 个被审交付件的 byte-exact 拷贝 + `MANIFEST.json`（sha256/bytes/in_audited_head） |
| `artifacts/audit_m17496/finding.json` | 机器可读核验结论（git_state / referee / callers / oss_refs / repro_diff / checks） |
| `docs/board_audit/m17496/land_verdict.py` | verdict 落表（id=572） |
| `docs/audit/figures/m17496_m17455_audit.png` | ④ 图 |