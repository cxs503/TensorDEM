# XVAL · DEM library binding frozen contract（对标 TensorPhysics 的 library-binding-frozen-contract 做法）

**锚点** `XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT`（第 1 轮 · 开源对标）· **仓** TensorDEM
**判据** owner 精度标准：对标误差 >15% 不可接受；本单另有预登记阈值见 `prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT.json`
**工件** `artifacts/xval_library_binding_contract/result.json`（所有数字程序化注入，本文件不手抄计算）
**复现** `python scripts/xval_library_binding_contract.py` → 1.96 s（单核 CPU），`pytest tests/test_binding_contract.py -q` → 20 passed / 7.1 s
**第 2 轮** 消费者侧覆盖（契约是否仍罩住绑定方真正调用的名字）见 `docs/XVAL_DEM_LIBRARY_BINDING_CONTRACT_R2.md`（mission 18458）

**结论** 8/8 预登记判据达标（含 4 项数值判据相对误差 ≤3.6e-16，远优于 15%）；漂移检测 7/7（对照基线 1/7）

---

## ① 问题与 ref 出处

### 1.1 问题

TensorDEM 的状态求解器（`IceDEM` 2-D / `IceDEM3D` 3-D）是被**跨仓当作 Python 库绑定**使用的：TensorPhysics 的
`TensorDEMAdapter` 把它适配成 `Solver` 协议，席位驱动器和船舶平台也直接调它的 `step()`。绑定方的正确性依赖一批
**没有机器门禁的隐含事实**：native `dt` 与全局步长的整数倍关系、时钟同步容差、载荷数组的形状/单位/符号、
"载荷瞬时施加、跨调用不累加、在 [t, t+dt] 内保持" 的语义、重启入口的形态。

这些事实一旦被静默改动（改名参数、把可选参数变必填、去掉 `load_time` 时钟护栏、新增 `restore` 让 checkpoint
冒充 restart），绑定方不会报错，只会算错。问题是：**供应方（TensorDEM）如何把这条边界冻结成可机器校验的契约**。

### 1.2 ref 出处（零手抄：均读自 OSS 源码/文档）

| 出处 | 读到的关键做法 |
|---|---|
| `TensorPhysics/src/tensorphysics/protocols.py` | `Solver` 协议五项 `name/initialize/advance/finalize/snapshot`；`SolverContext`(step/time/dt/metadata)；**明确不含 `restore`**（checkpoint ≠ restart） |
| `TensorPhysics/src/tensorphysics/adapters/tensordem.py`（113 行） | 构造期校验 `step` 可调用 + `dt`/`time` 存在；`time_tolerance=1e-10` 绝对容差；`advance` 要求 `global dt / native dt` 为整数（`isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9)`）；每个 substep 起点取载荷并**原样**传给 `step(external_forces=..., load_time=...)`，"不静默插值/缩放/重映射"；结束时钟校验，容差按 `max(1,|t|)` 缩放；失败不回滚已推进的 substep；`snapshot()` 返回 `restartable=False` |
| `TensorPhysics/docs/adapter-contracts.md` | 逐仓公开接口清点；载荷语义（SI 牛顿、形状 `(N,2)`、瞬时、不累加）；"专属适配器必须自己验证时钟、形状、checkpoint 与受力契约"；集成门 7 条（统一物理时钟、字段形状/坐标系/单位/几何版本、合力合力矩、作用反作用符号、质量动量/能量守恒、零载/无冰/刚冰/敞水对照、记录版本与容差） |
| `TensorPhysics/docs/adapter-inventory.md` | 能力矩阵 ✅/⚠️/❌（"存在且有测试 / 存在但偏离协议 / 明确缺失"），逐项显式登记**缺什么**（诚实标注 `snapshot` 返回 `dict` 而非 `SolverState`） |
| `TensorPhysics/tests/test_adapter_inventory_meta.py`（43 行） | 把清点结论写成**元测试**：断言协议成员集合，并断言 **`assert not hasattr(Solver, "restore")`**（负向断言冻结"不存在"） |

ref 的共性：**契约=显式清点 + 硬前置校验 + 负向断言 + 元测试兜底**，而不是散落在散文里的假设。

### 1.3 本仓现状（改动前）

`grep` 全仓：无 binding/contract 模块，无对外接口指纹。跨仓绑定靠的隐含事实只存在于
`IceDEM`/`IceDEM3D` 的 docstring 与调用方代码里；`tests/` 无任何"接口被静默改动"的元测试。
即：供应方**没有把边界立起来**。

---

## ② 方法与实现

### 2.1 新增模块 `src/tensordem/binding_contract.py`（供应方冻结契约）

| 组成 | 作用 |
|---|---|
| `BINDING_CONTRACT`（纯数据） | 两个冻结条目 `IceDEM` / `IceDEM3D`：`state`（10 / 8 个属性的类型·形状·单位）、`properties`、`methods`（参数 **kind/requiredness/annotation 标记** + 语义说明）、`clock`（native dt 来源、整数倍规则、`time_tolerance`、`load_time` 相对容差 1e-12）、`capabilities`（restart/exchange/substep load stamp/fracture events/fragments）、`explicitly_absent`（负向断言） |
| `fingerprint()` / `FROZEN_FINGERPRINT` | 对契约**规范化 JSON 的 sha256** 做钉扎。任何对冻结面的编辑都改指纹 → `tests/test_binding_contract.py::test_frozen_fingerprint_is_pinned` 立即红，必须"同 commit 显式重钉"。本单指纹 `sha256:0cbfed8c…4105d` |
| `verify(instances)` | 内省比对：属性存在性、property 形态、方法可调用性/classmethod 绑定、**参数名·kind·是否必填·注解标记**、非法参数新增、负向断言。漂移按 `breaking / narrowing / additive / doc` 分级；`assert_conforms()` 在 breaking 时抛 `BindingContractError`（附结构化 findings） |
| `probe_semantics()` | 19 个行为探针（语义而非名字）：形状/类型/有限性护栏、`load_time` 陈旧护栏、**冲量-动量恒等式**（2-D/3-D）、**载荷不累加**、内部 ledger 交叉路径、时钟 `n·dt`、**重启往返逐位一致 + 继续推进逐位一致**、`forces()` 与 `step()` 同载一致 |
| `mutation_matrix()` | 7 个受控变异体（每个漂移类一个），验证检测器不是"摆设" |
| `substep_rule_check()` | 复刻 ref 的整数倍子步规则，在容差两侧取样给出接受/拒绝边界 |

模块顶层**不 import torch**（纯 declarative），绑定方可在没装求解器的环境里读清单。

### 2.2 试验设计（experimental-design skill 落地）

| 项 | 内容 |
|---|---|
| Decision | 是否把"供应方冻结契约 + 漂移检测"落为本仓的跨仓绑定机制（而非维持散文+调用方各自假设） |
| Alternatives | A 维持现状（散文+调用方冒烟）；B 仅名字级清点；C 全量声明式契约 + 指纹 + 行为探针（本单） |
| Hypothesis | H1：契约检测器对 7 类真实漂移全部可分类，且行为语义（瞬时/不累加/时钟/重启）可被数值恒等式证伪 |
| Metrics | 漂移检测率（7 类）、语义探针通过率、冲量-动量相对误差、ledger 相对误差、时钟相对误差、重启漂移 |
| Baseline / control | 名字级清点（pre-contract 基线，由同一批变异体喂入） |
| Thresholds（预登记） | 检测 7/7=100%（基线预期≈1/7）；探针 19/19；冲量/ledger/时钟相对误差 <1e-12（owner 线 15%）；重启漂移 = 0 |
| Confounders | 2-D 与 3-D 的约束不同（固定粒子、阻尼、drag 会吃掉动量）→ 探针用"无支撑/无阻尼/无 drag/tool 抬离"的隔离样本；子步数扫掠把"数值误差"与"结构恒等"分开 |
| 判别性 | H0（绑定方把载荷当累加量再发一次）预测冲量比 `n(n+1)/2 / n = (n+1)/2`；n=2/4/8/16 时 H0 比值 1.5/2.5/4.5/8.5，与 H1 的 1.0 分离 ≥50%（≫15%） |

---

## ③ 参数表（含无量纲数）

隔离探针几何：2-D 键合圆盘格 `nx=6, ny=3`（18 粒子），`radius=0.025 m`，`thickness=0.2 m`，`density=917 kg/m³`，
`bond_stiffness=contact_stiffness=2000 N/m`，`contact_damping=0, drag=0`，`fix_edges=False, tool_gap=10 m`（工具脱离）；
3-D：`nx=ny=nz=4`（64 球），`contact_damping=0, drag=0, fix_bottom=False, fix_x_edges=False, tool_gap=10 m`。

| 量 | 值 | 无量纲化 |
|---|---|---|
| 粒子质量 m | `917·π·0.025²·0.2 = 0.36011 kg` | — |
| native dt | `4.53624e-4 s`（= `0.2·√(m/K_tot)`，`K_tot=70000 N/m`） | `dt/T_c = 1.076e-2`（`T_c=π√(m/k)=4.2158e-2 s`） |
| 每步工具行程 | `1.0 m/s × 4.536e-4 s = 4.536e-4 m` | `Δx/r = 1.815e-2` |
| 载荷/粒子 | `F=(0.25, -0.1) N` | `F/(m·g) = 0.0708`（g=9.81） |
| 单步单粒子冲量 | `1.134e-4 N·s` | `F·dt/(m·√(k/m))` = 1.008 |
| 子步数扫掠 | n = 1,2,4,8,16,32,64,128 | `t_total/T_c = 1.08e-2 … 1.377` |
| 子步接受边界 | `rel_tol = abs_tol = 1e-9` | 比值 4+1e-6 拒 / 4+1e-10 收 / 4.5 拒 |
| 重启检查步数 | 10 → 快照 → 再 1 步 | 漂移 0.0 m（逐位） |

---

## ④ 对照表（ref vs got，误差打勾/叉）

| # | 判据 | ref（解析/OSS 期望） | got | 相对误差 | <15% | 判定 |
|---|---|---|---|---|---|---|
| 1 | 冲量-动量恒等式（2-D，n=1…128） | `n·dt·F` | `m·ΣΔv` | ≤ **2.7e-16** | ✅（超 15 档 14 个量级） | MET |
| 2 | 冲量-动量恒等式（3-D，n=8） | `n·dt·F` | `m·ΣΔv` | **3.56e-16** | ✅ | MET |
| 3 | 内部 ledger 交叉路径（`accounting["external_impulse_Ns"]`） | `n·dt·F` | 独立账本 | **1.77e-16** | ✅ | MET |
| 4 | 时钟 `time == step_count·dt`（10 步） | `n·dt` | `model.time` | **1.91e-16** | ✅ | MET |
| 5 | 重启往返 + 继续推进 | 逐位一致 | 逐位一致 | **0.0 m** | ✅（0 容差） | MET |
| 6 | 契约内省一致性 | 42/42 冻结检查 | 42/42 | 0 | ✅ | CONFORMS |
| 7 | 语义探针 | 19/19 | 19/19 PASS | 0 | ✅ | MET |
| 8 | 漂移检测率（7 类受控变异） | 7/7 | **7/7 = 1.000**，基线 1/7 = 0.143 | Δ = **+85.7 pp** | ✅ | MET |

**逐项回答「是否 <3%」**：①–⑤ 的相对误差在 0 … 3.6e-16，**远小于 3%**（机器精度量级，因为恒等式是结构性的，
不是数值逼近）；⑥⑦ 为布尔/计数一致（0 偏差）；⑧ 为检测覆盖率（1.000，Δ +85.7 pp）。无一项超过 15%。

漂移分类明细（`mutation_matrix`，全部按预登记级别分类）：

| 变异类（受控） | 期望变更 | 期望级别 | 观测 | 检测 | 报告状态 |
|---|---|---|---|---|---|
| `state_attribute_removed`（删 `positions`） | attribute_removed | breaking | 同名 | ✅ | DRIFT |
| `method_removed`（`exchange_state=None`） | method_removed | breaking | 同名 | ✅ | DRIFT |
| `param_renamed`（`external_forces→loads`） | param_removed | breaking | 同名 | ✅ | DRIFT |
| `param_became_required` | param_became_required | breaking | 同名 | ✅ | DRIFT |
| `param_kind_widened`（`load_time` 去 kw-only） | param_kind_widened | additive | 同名 | ✅ | CONFORMS（诚实：对关键字调用者源兼容） |
| `param_added_required`（加 `external_torque`） | param_added | breaking | 同名 | ✅ | DRIFT |
| `negative_assertion_violated`（加 `restore`） | negative_assertion_violated | breaking | 同名 | ✅ | DRIFT |

---

## ⑤ 收敛性 / 参数敏感性

**(a) 子步数扫掠（n = 1…128，图 `fig_impulse_error.png`）**：相对误差在 `0 … 2.66e-16` 之间**不随 n 增长**
（n=16 恰为 0，n=128 为 2.1e-16，纯舍入噪声）。判读：冲量-动量是**结构性恒等式**（每步 `v += dt·F/m` 的累积与
`n·dt·F` 代数相等），因此该量不是"可能收敛不上"的数值格式，而是绑定语义是否被正确消费的直接探针；
误差不随载荷步数漂移，说明不存在隐藏的载荷累积/重置。

**(b) 子步接受边界（容差敏感性，图 `fig_substep_boundary.png`）**：比值 1/2/4/16 全部接受并给出正确子步数；
`4+1e-6`（相对偏差 2.5e-7 > 1e-9）**拒绝**；`4+1e-10`（2.5e-11 < 1e-9）**接受**；`4.5` 拒绝。
即边界与 ref 的 `rel_tol=abs_tol=1e-9` 一致，绑定方可以明确站在边界哪一侧。

**(c) 漂移检测器敏感性（变异类扫掠）**：7/7；对照基线（只查"名字还在不在"）仅 1/7（漏 `method_removed`、
`param_renamed`、`param_became_required`、`param_kind_widened`、`param_added_required`、`negative_assertion_violated`）。
说明"清点式清单"不足以冻结签名与负向断言。

**(d) 表面覆盖（图 `fig_coverage.png`）**：`IceDEM` 公开成员 11/11 = **100%** 冻结；`IceDEM3D` 20 中冻结 8 = **40%**
（刻意只冻结驱动侧步进面，物理诊断如 `bond_law_report`/`friction_cone_limit` 不进契约，见 ⑥ 的取舍理由）。

**(e) 环境隔离敏感性（confounder 控制）**：3-D 探针若保留默认 `fix_x_edges=True`/`drag=2.0`/`contact_damping=5.0`，
冲量比偏离 1 达 **50.2%**（首轮实测 5.023e-01）——这不是绑定语义错误，而是固定粒子把动量交给支撑、drag 把动量耗散。
隔离样本（无支撑、无阻尼、无 drag、工具抬离）后误差降到 3.56e-16。**教训：动量恒等式类判据必须先把非守恒力关掉，
否则测的是模型配置而不是绑定契约。**

---

## ⑥ 失败模式、局限与工件

### 6.1 本仓 vs OSS（TensorPhysics）差异清单 + 移植取舍

| 维度 | TensorPhysics（ref 做法） | 本仓（移植后） | 取舍理由 |
|---|---|---|---|
| 契约载体 | 散文文档 + 手写元测试（`test_adapter_inventory_meta.py` 43 行）+ 能力矩阵表 | **声明式 JSON 数据** + 内省校验器 + 元测试（20 条） | 数据可被绑定方程序读取/比对，清单与测试不会各自漂移；代价是需维护重钉流程 |
| 冻结手段 | 断言协议成员集合、`assert not hasattr(Solver,"restore")` | 断言 + **规范化 JSON sha256 指纹钉扎** + 负向断言（`explicitly_absent`） | 指纹能把"文档级改字"也纳入评审；代价是每次有意修改契约都要显式重钉 |
| 角色 | 消费者（适配器侧校验 native 库） | **供应方（发布被消费面）** | 双向：供应方冻结 + 消费方校验，边界才闭合 |
| 时钟 | `time_tolerance=1e-10`（绝对），结束校验按 `max(1,|t|)` 缩放，子步整数倍 `rel_tol=abs_tol=1e-9` | 同规则**写进 `clock` 段并复刻验证**（`substep_rule_check`） | 规则在供应侧可见，绑定方可直接对照 |
| 载荷语义 | "瞬时、不静默重映射、每子步起点取载荷" | 语义文本 + **冲量恒等式/不累加/ledger 交叉路径探针** | 语义必须能被证伪，否则只是口号 |
| restart | 适配器 `snapshot()` 返回 `restartable=False` | 2-D 有 `snapshot()/from_snapshot()`（`tensordem.restart/1`），3-D 是 `state_dict()/load_state_dict()`，并**冻结两者不同**（`explicitly_absent: snapshot/from_snapshot/exchange_state` on 3-D） | 与 ref 的"3-D 必须独立适配"结论一致；同时本仓能证明 2-D 重启**逐位可复现**（0 m 漂移），比 ref 的 `restartable=False` 更强，但**不**把 `restore` 加进契约 |
| 3-D `load_time` | 未定义（ref 明说 3-D 需独立适配） | 契约显式冻结"3-D 无 `load_time`"，并验证传 `(N,2)` 载荷被拒、`(N,3)` 生效 | 防止绑定方把 2-D 载荷约定平移到 3-D |
| 边界条件 | 集成门 7 条（力/矩/符号/守恒/对照算例） | 只覆盖**绑定面**（时钟/形状/单位/语义/重启）；物理算例验证留在各算例文档 | 不越界：契约不冒充物理精度验证 |
| 依赖 | 核心零依赖，torch 可缺省 | 契约模块顶层零 torch，行为探针按需 import | 绑定方无 torch 也能读清单 |

**未移植的部分（诚实声明）**：ref 的能力矩阵里 `⚠️ 存在但偏离协议`（`snapshot` 返回 `dict` 而非 `SolverState`）
这类"跨仓一致性"检查需要看到协议定义，属消费方职责，本仓不复制 `Solver` 协议；本单也不接管 ref 的集成门 1–7
（合力/合力矩/能量守恒的算例级验证）。

### 6.2 失败模式（本单实测到的）

1. **动量恒等式被非守恒力污染**：默认配置下 3-D 冲量误差 50.2%（见 ⑤e），若不复核配置就会误判"绑定契约坏了"。
2. **`load_time` 半传即错**：仅对 2-D 有效；3-D 传 `load_time=` 会 `TypeError`——契约已冻结该差异，避免绑定方盲移。
3. **重启命名分叉**：2-D `snapshot/from_snapshot` vs 3-D `state_dict/load_state_dict`；`from_snapshot` 是 classmethod，
   校验器需要单独断言绑定方式（`kind_classmethod`），否则 `inspect.signature` 少一个 `self` 会误判参数漂移。
4. **宽化被误判为破坏**：`load_time` 从 kw-only 变普通参数对关键字调用者源兼容 → 必须分 `additive` 而不是 `breaking`，
   否则日报噪声会让人关掉检测（本单按 additive 处理，并允许 `status=CONFORMS`）。
5. **指纹 churn**：任何契约文字改动都必须重钉；遗漏会让 `test_frozen_fingerprint_is_pinned` 红，属预期摩擦而非 bug。
6. **冻结面覆盖不均**：3-D 公开成员只冻结 40%，未冻结的物理诊断若被绑定方依赖，仍可能静默漂移（已登记为限制）。

### 6.3 工件与复现

| 路径 | 内容 |
|---|---|
| `src/tensordem/binding_contract.py` | 契约数据 + 指纹 + 校验器 + 19 探针 + 变异矩阵 + 子步规则 |
| `tests/test_binding_contract.py` | 20 条测试（含 7 类变异参数化、无 torch 导入、判据阈值） |
| `scripts/xval_library_binding_contract.py` | 产出 `result.json` 与 4 张图（1.96 s） |
| `artifacts/xval_library_binding_contract/result.json` | 全部数字 + 环境（python 3.11.6 / torch 2.7.1）+ 判据表 |
| `artifacts/xval_library_binding_contract/fig_*.png` | 冲量误差-子步数 / 变异检测矩阵 / 覆盖度 / 子步边界 |
| `prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT.json` | 预登记阈值与停止规则 |

复现：`python scripts/xval_library_binding_contract.py && pytest tests/test_binding_contract.py -q`

### 6.4 生态摩擦（如实记录）

1. `knowledge_base` 检索引擎离线（`engine_offline`, `127.0.0.1:8021 Connection refused`），开工三查的
   "同域 capability 检索"退化为空结果；改以仓内 `grep` + `negative-knowledge/search`（返回 0 条）替代。
2. `negative_knowledge_ledger`/`oss_solutions`/`tensor_xval_cases`/`tensor_dev_missions` 对本工作树 DB 账号
   **permission denied**，无法读取本锚点的 official 判据行与既有 XVAL 表；本单判据改由 `prereg/` 自登记（见该文件）。
3. 本机 aarch64 单核即可完成（总墙钟 <15 s），未占用集群资源。

### 6.5 判据总结

8/8 预登记判据达标；数值判据相对误差 ≤3.6e-16（≪15%，且 ≪3%）；漂移检测 7/7（基线 1/7）；
契约指纹 `sha256:0cbfed8c1172f8412dedfc013e87c79f56f0567ca22d782e0387818f0ee4105d` 与冻结面一致。
**未做**：把 `verify()` 接进 TensorPhysics/ship 平台的 CI（跨仓消费需要对方仓改动，超出本仓边界）。
---

## ⑦ 返修轮（m17518）：集成调用方 + 数值 benchmark 工件

原单 m17455 的交付**未提交**（`git status` 全为 `??`），maturity gate 因此只能看到占位/空工件，判 L1。
返修轮把同一批成果**落进分支并补两级缺口**：L3 集成调用方（主求解器真调用）、L4 benchmark（ref vs got 数值对照 + repro_cmd）。

### 7.1 集成调用方（L3：不是只在测试里调用）

| 调用方 | 调用点 | 语义 | 成本 |
|---|---|---|---|
| `src/tensordem/dem.py` `IceDEM.exchange_state()` | 平台面向外导出 `tensordem.exchange/1` 之前 | 绑定面漂移则**导出即失败**，不把漂移态喂给耦合仿真 | 首次 ~1 ms，之后按类缓存 1 次 dict 查表 |
| `src/tensordem/dem3d.py` `IceDEM3D.state_dict()` | 3-D 重启导出（绑定方恢复耦合运行的入口）之前 | 同上（3-D 条目） | 同上 |
| `src/tensordem/cli.py` `binding-contract` 子命令 | `python -m tensordem binding-contract --out dir` | 跑内省校验 + 19 语义探针 + 数值 benchmark 并落 JSON | ~4 s |

缓存策略：**只缓存 CONFORMS**；失败必须重新内省（`refresh=True` 亦可强制），避免"一次异常漂移把后续导出永久钉死/或反之"。
测试 `test_exchange_state_calls_the_frozen_gate` / `test_state_dict_calls_the_frozen_gate` 用
"打破负向断言（给类加 `restore`）→ 导出必须抛 `BindingContractError`"来证明门禁是**真执行**而不是装饰。

### 7.2 数值 benchmark（L4：ref vs got，机读工件）

工件 `artifacts/xval_library_binding_contract/binding_contract_cli.json`（`benchmark.rows`，11 行，全部含
`ref` / `got` / `rel_error` / `tolerance` / `pass`）。判据：owner 线 **15%**；更紧的行另带预登记容差。

| # | 行 | ref（出处） | got | 相对误差 | <15% |
|---|---|---|---|---|---|
| 1 | `impulse_momentum_2d`（n=16） | 解析 `n·dt·F` | `m·ΣΔv` | 0.0（≤1e-16 舍入） | ✅ |
| 2 | `free_flight_vs_symplectic`（t=0.01 s） | 半隐式欧拉闭式 `a·dt²·n(n+1)/2` | 绑定路径位移 | 6.77e-14 | ✅ |
| 3 | `free_flight_vs_continuum_dt_x1` | 连续解 `0.5·a·t²` | 同上 | 4.55e-02 | ✅ |
| 4 | `..._dt_x0.5` | 同上 | 同上 | 2.27e-02 | ✅ |
| 5 | `..._dt_x0.25` | 同上 | 同上 | 1.14e-02 | ✅ |
| 6 | `energy_drift_conservative`（2000 步） | 0（无耗散路径，E(0)=4.051e-3 J） | max\|ΔE\|/E(0) | 1.39e-02 | ✅ |
| 7 | `hertz_reduced_modulus` | `E/(2(1-ν²))`=4.9451e9 Pa | 导出核 | 0.0 | ✅ |
| 8 | `hertz_effective_radius` | `R/2`=0.0125 m | 导出核 | 0.0 | ✅ |
| 9 | `hertz_normal_force`（δ=1e-4 m, E=9 GPa, ν=0.3） | `(4/3)E*√R*δ^1.5`=737.165 N | 导出核 | 0.0 | ✅ |
| 10 | `clock_n_dt`（n=37） | 解析 `n·dt` | `model.time` | 0.0 | ✅ |
| 11 | `restart_roundtrip_drift` | 0 m（逐位） | 最大位置漂移 | 0.0 | ✅ |

**收敛性/敏感性（新增两点）**：
(a) 自由飞行 vs 连续解：dt、dt/2、dt/4 三档误差 4.545e-2 → 2.273e-2 → 1.136e-2，实测收敛阶
`p = log2(e_coarse/e_fine) = 1.000, 1.000`（半隐式欧拉对常加速度的一阶格式，逐位吻合）；
(b) 守恒格点能量漂移随步数：2000 步内 max\|ΔE\|/E(0) = **1.39%**（≈1.4e-2，对称辛格式的有界振荡，非耗散增长）。

判读：11/11 行 `<15%`（最差 4.5%），其中 6 行为机器精度级（≤7e-14 或 0）——它们是**绑定语义恒等式**，不是数值近似；
3 行是真正的一阶离散误差，且被测出正确的收敛阶。

### 7.3 复现命令（repro_cmd）

```bash
# 1) 契约门禁 + 19 语义探针 + 11 行数值 benchmark（写出机读工件）
PYTHONPATH=src python -m tensordem binding-contract \
    --out artifacts/xval_library_binding_contract --benchmark-energy-steps 2000
# 2) 同一入口的模块级复现（无 argparse 依赖）
PYTHONPATH=src python -m tensordem.binding_contract --verify --probes --benchmark \
    --out artifacts/xval_library_binding_contract
# 3) 对标全流程（result.json + 4 图）
PYTHONPATH=src python scripts/xval_library_binding_contract.py
# 4) 测试
PYTHONPATH=src python -m pytest tests/test_binding_contract.py -q     # 29 passed
```
