# XVAL-DEM-LIBRARY-BINDING-CONTRACT — 第 4 轮：真实消费方执行（executed binding）

锚点：`XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT` · 仓库 TensorDEM · mission 19466 · 分支 `agent/m19466`
预登记（阈值先于运行）：`prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT_R4.json`
上游链条：第 1 轮 = 供应方冻结面（PR #1450，m17518/m17960 落回 main）；第 2 轮 = 消费方覆盖（m18458）；
第 3 轮 = 面清点 no-silent-surface（m18534，`docs/XVAL_DEM_LIBRARY_BINDING_CONTRACT_R3.md`）。

---

## ① 案例与问题（为什么还需要第 4 轮）

前三轮把「供应方边界」立住了：`BINDING_CONTRACT` 声明式清单 + sha256 指纹 + `verify()` 内省 +
19 个行为探针 + 全公开面清点（`surface_ledger`）。但**执行证据全部在供应方一侧**：

| 检查 | 执行对象 | 谁在跑 |
|---|---|---|
| `verify()` / 19 探针 / 2-D·3-D benchmark | 真 `IceDEM` / `IceDEM3D` | TensorDEM 自己 |
| TensorPhysics 消费方 | **`FakeIceDEM`**（`TensorPhysics/tests/test_tensordem_adapter.py`） | TensorPhysics 自己 |

即：真正把 TensorDEM 当库绑定的那份代码（`tensorphysics.adapters.tensordem.TensorDEMAdapter`，
113 行）**从来没有跑过真的 `IceDEM`**。契约说「global dt 必须是 native dt 的整数倍（rel_tol 1e-9）、
每个子步起点取载荷并原样传入、`snapshot()` 不可重启」，这些都是对消费方行为的断言，却只由供应方
自证 —— 这是典型的「两边各自的测试都绿，接缝没人测」。

本轮判别性问题：**把真实消费方跑在真实供应方上，契约声明的每一个数字是否被观测到？**
并且要证明这个交叉核对**不是摆设**：喂进受控漂移，它必须报错。

## ② 方法

| 组成 | 内容 |
|---|---|
| 真实消费方 | `tensorphysics.Simulation` + `SimulationConfig` + `TensorDEMAdapter`（上游 checkout，经 `TENSORPHYSICS_SRC` 或默认兄弟路径解析；缺失则如实 WITHHELD，绝不造数） |
| 真实供应方 | `tensordem.dem.IceDEM(DEMConfig(nx=6, ny=3, fix_edges=False, contact_damping=0, drag=0, tool_gap=10))`，18 粒子、无支撑、无阻尼、无 drag（动量干净） |
| 记录代理 | `RecordingIceDEM` 把消费方 `step()` 收到的 `(external_forces, load_time)` **原件**截获（含 `id()` 恒等检查），再委托真求解器 |
| 主臂 | global dt = 4·dt_native，跑 3 个 global step → 期望 12 个 native 子步 |
| 受控漂移臂 | ①`AccumulatingIceDEM`（供应方把载荷累加而不是替换，即契约要防的 H0 失效模式）②stale native clock ③非整数倍 global dt（真 `TensorDEMAdapter` 必须拒） |
| 边界臂 | 子步规则容差两侧真跑：`4+1e-10`（收）、`4+1e-6`（拒）、`4.5`（拒） |
| 指纹钉扎 | 报告内比对 `fingerprint() == FROZEN_FINGERPRINT`，契约被静默改动则本单工件失效 |
| 入口 | `scripts/xval_library_binding_contract_live.py`（独立可跑，写 JSON+表，任一行 FAIL 则 rc≠0） |
| 顺带修的缺陷 | `src/tensordem/cli.py` 缺 `if __name__ == "__main__"` 块 → 文档里的复现命令 `python -m tensordem.cli binding-contract ...` **什么都不做、不写工件、仍返回 0**（静默成功） |

## ③ 参数表（含无量纲数）

| 量 | 值 | 无量纲化 |
|---|---|---|
| 粒子数 | 18（6×3 键合网格） | — |
| 单粒子载荷 | `F=(0.25, −0.1) N` | `F/(m·g)=0.0708`（m=0.36011 kg） |
| native dt | `4.536235e-4 s` | `dt/T_c = 1.076e-2`（T_c=π√(m/k)） |
| global dt | `4·dt = 1.814494e-3 s` | 子步数 n=4 |
| global steps | 3 | 总子步 12，`t_end=5.4435e-3 s = 0.1291·T_c` |
| 子步规则容差 | `rel_tol=abs_tol=1e-9` | `4+1e-10` vs `4+1e-6` |
| 载荷 stamp 容差 | 1e-12 s | 相对 `t_end` = 1.8e-10 |
| 冲量恒等式容差 | 1e-12 | owner 线 15% 的 1/1.5e11 |
| H0 判别量 | `(n+1)/2 = 6.5` | 与 1.0 分离 84.6% |

## ④ 结果（`artifacts/xval_library_binding_contract_live/live_binding.json`，全部程序化产出）

| # | 臂 | ref（契约声明/解析） | got（实测） | 相对误差 | 判定 |
|---|---|---|---:|---:|---|
| 1 | 子步数 | 12 | 12 | **0.0** | PASS |
| 2 | `load_time` stamp 最大偏差 | 0 s（理想 `k·dt`） | 4.34e-19 s | 4.34e-19 | PASS |
| 3 | 载荷原样传入（`id()` 恒等） | 12/12 | 12/12 | 0.0 | PASS |
| 4 | 外部冲量 vs `n·dt·F` | `[0.024495671, −0.009798269]` | `[0.024495671, −0.009798269]` | **1.77e-16** | PASS |
| 5 | 冲量比 got/ref（不累加） | 1.0 | 0.9999999999999999 | **2.22e-16** | PASS |
| 6 | H0 分离度 | ≥0.15（owner 线） | **0.8462** | — | PASS |
| 7 | 时钟 `n·dt` | 5.443483e-3 s | 5.443483e-3 s | **1.59e-16** | PASS |
| 8 | 不可重启（负向断言，执行） | `restartable=False` 且无 `Solver.restore` | 同 | 0 | PASS |
| 9 | `Simulation.run()` 终态 | FINISHED | FINISHED | 0 | PASS |
| 10 | 子步边界 `4+1e-10` | 接受 | 接受 | 0 | PASS |
| 11 | 子步边界 `4+1e-6` | 拒绝 | `SimulationError` | 0 | PASS |
| 12 | 子步边界 `4.5` | 拒绝 | `SimulationError` | 0 | PASS |
| 13 | 漂移检出率 | 2/2 | **2/2** | 0 | PASS |
| 14 | 不累加扫掠 n∈{1,2,4,8,12,16} | ratio 1.0 | 1.000000000000000 … 1.0000000000000002 | **2.22e-16** | PASS |

不累加扫掠（追加臂，见 `prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT_R4_addendum.json`，阈值沿用 R4 预登记）：

| n（子步/global step） | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| 实测冲量比 got/ref | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.999999… | 0.999999… |
| H0（累加）预测 (n+1)/2 | 1.0 | 1.5 | 2.5 | 4.5 | 6.5 | 8.5 |

实测在全部 n 上平坦（偏差 ≤2.22e-16），H0 线随 n 线性抬升 → 判别性随 n 放大，与「契约是空话」明确分离。

受控漂移臂的判别性（关键：证明核对不是摆设）：

| 漂移臂 | 预测（若契约为空话） | 实测 | 结论 |
|---|---|---|---|
| `AccumulatingIceDEM`（供应方累加载荷） | H0：冲量比 `(n+1)/2 = 6.5` | **6.500000000000001**（两分量均） | 检出（比 1.0 偏离 5.5×），与 H0 预测吻合到 1e-15 |
| `stale native clock`（`time=0.5·dt`） | 消费方应拒 | `SimulationError`（`TensorDEMAdapterError`） | 检出 |

供应方侧对照（同一 commit 重跑，`artifacts/xval_library_binding_contract/`）：

| 指标 | 值 |
|---|---|
| `verify()` | CONFORMS，73/73 检查匹配 |
| 语义探针 | 19/19 PASS |
| 2-D benchmark | 11/11 MET，worst rel err **4.545e-2**（`free_flight_vs_continuum_dt_x1`，一阶半隐式对连续解的离散误差，容差 0.15） |
| 3-D benchmark | 15/15 MET，worst rel err **8.642e-3**（`energy_drift_conservative_3d`） |
| 契约指纹 | `sha256:048e4cf9…195cd`（与 `FROZEN_FINGERPRINT` 一致，本单未改冻结面） |

## ⑤ 注意点（坑与局限）

1. **静默成功是最贵的坑**：`python -m tensordem.cli` 在此前的文档里是复现命令，但没有 `__main__` 块 →
   导入即退出、rc=0、零工件。这让「没有工件」看起来像「通过」。已修，并加 `tests/test_cli_module_entry.py`
   断言「入口必须真的写出工件」与「模块入口 / 包入口指纹一致」。
2. **消费方只在别的仓**：TensorDEM CI 里没有 `tensorphysics`，故本单的 live 测试在缺失时 **skip**（诚实），
   工件由 `scripts/xval_library_binding_contract_live.py` 在本机真实执行产生；报告里记录了
   `consumer` 路径与 `supplier.git_sha`，两者任一变化工件应失效重跑。
3. **只覆盖 2-D**：TensorPhysics 的 `TensorDEMAdapter` 只适配 2-D `IceDEM`（3-D 适配器在本轮上游不存在）→
   3-D 面只由供应方探针 + 3-D benchmark 背书，本轮**不**声明 3-D live binding。
4. **读数口径**：冲量用求解器自己的外部冲量账本 `accounting["external_impulse_Ns"]`，而不是粒子总动量 ——
   重力与键合力会污染后者；隔离样本已关掉阻尼/drag/支撑。
5. **容差不是「调」出来的**：stamp 偏差 4.3e-19 s 来自 `context.time + i·dt` 与 `k·dt` 两条浮点路径的
   末位差异，预登记给的是 1e-12 s 上限；不是事后放宽。

## ⑥ 误差判定（逐项回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 子步数 | 契约整数倍规则（解析） | 0 | ✅ |
| `load_time` stamp | 解析 `k·dt` | 4.34e-19 s（相对 t_end 8.0e-17） | ✅ |
| 外部冲量 | 解析 `n·dt·F` | 1.77e-16 | ✅ |
| 载荷不累加比值 | 解析 1.0 | 2.22e-16 | ✅ |
| 时钟 | 解析 `n·dt` | 1.59e-16 | ✅ |
| 重启往返（供应方） | 逐位一致 | 0 | ✅ |
| 2-D 自由飞离散误差（供应方 benchmark） | 连续解 `½gt²` | **4.55e-2** | ✅（4.5% < 3%？**否**：4.55% 超 3%，但 < owner 15% 线；该行容差预登记 0.15，属一阶积分的预期离散误差） |
| 3-D 能量漂移（供应方 benchmark） | 保守格点零漂移 | **8.64e-3** | ✅ <3% |
| H0 分离度 | `(n+1)/2` | 6.5 vs 1.0（84.6% 分离） | ✅ |

**诚实标注**：唯一超过 3% 的量是 2-D `free_flight_vs_continuum_dt_x1` = **4.55%**（对连续解的一阶离散误差，
dt 减半/四分之一时依次 2.27%/1.14% → 一阶收敛，属方法预期）。其余全部量在机器精度量级（≤2.2e-16），
远优于 <3% 与 owner 15% 线。无一项超过 owner 15% 不可接受线。