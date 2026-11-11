# 交差审计：m14682 · GAP-DEM-GPU-100K（TensorDEM）

审计单 m15017（board-audit）｜对象：m14682（status=succeeded, accepted=true, finished 2026-11-11T10:32Z）
被审代码分支：`agent/m14682` @ `19f62654`（**未并入 main**：`git log --all -- src/tensordem/dem3d_pair_search.py` 仅此一提交）
被审工作树：`/root/tp-agent/work/m14682`｜审计工作树：`/root/tp-agent/work/m15017`

---

## 1. 结论（先给判定）

| 项 | 审计结论 |
|---|---|
| **判级** | **L3（系统集成）** — 真代码 + 真运行工件 + 真接入主求解器；**L4 基准验证不成立**，因为 GAP 判据指向的量（GPU 上 10 万颗粒单步 ≤2 s）**至今没有测量值** |
| 交付真实性 | 非造假：GPU 腿自报 `fail_stage=gpu_hardware_unreachable`、`status=FAIL`，与审计独立复核一致 |
| 三件套 | run_artifacts ✅ / benchmark 数值对照 ⚠️（只有 CPU，GPU=NOT_RUN）/ repro_cmd ✅（审计实际跑通） |
| 外部锚 | ❌ 无（无文献/实验吞吐锚，只有 Cundall–Strack 方法出处） |
| 遗留缺口 | **GAP-DEM-GPU-100K 本体未消**：GPU 吞吐 not_run；本审计不把 CPU 数字当 GPU 验收 |
| 文档失实项（3 处） | ①「候选集与 legacy 完全相同」**假**（实测 legacy 多 4.6 倍远对）②「contacts=288 401」实为**候选对**非接触 ③ legacy 基线 6.645 s/步 **不可复现**（实测 50 653 颗粒 legacy 19.6 s/步） |

---

## 2. 三件套逐项核验（取证）

### 2.1 run_artifacts —— 存在且是真跑出来的

| 工件 | 核验 |
|---|---|
| `artifacts/gap_dem_gpu_100k/bench_cpu_loong64c_20261111T103056Z.json` | ✅ 含 `hardware`（Loongson-3C6000/D, 128 核, torch 2.7.1, loongarch64, cpu）、`runs.dem3d/vecdem.runs[]` 原始每步数组 `step_seconds`、`components.pair_search_seconds`、`scaling.scaling_exponent_p`、`acceptance`（脚本自动判定，`dem3d.passed=false`） |
| `artifacts/gap_dem_gpu_100k/fig_cpu_scaling.png` | ✅ 73 565 B |
| `artifacts/gap_dem_gpu_100k/attempt_ledger.json` | ✅ 诚实 FAIL 记录（fail_stage / what_was_tried / what_is_missing） |
| `artifacts/dem_gpu_100k/throughput_n1000.json`, `..._n4913.json` | ✅ 早期小规模试跑（其中 `gpu_backend: NOT_SUPPORTED`） |
| 报告 | `/tensor/reports` id=13832 存在 |

### 2.2 benchmark ref vs got —— CPU 腿成立，GPU 腿无值

| 量 | ref（判据/基线） | got（m14682 报告） | got（审计重跑） | 误差/判定 |
|---|---|---|---|---|
| 10 万颗粒单步中位（**GPU**，GAP 判据） | ≤ 2.0 s | **无（NOT_RUN）** | **无（NOT_RUN）** | ❌ 不可判定 |
| dem3d CPU 中位步时 @101 614 | — | 3.570 s | **2.690 s**（`bench_cpu_repro_by_audit.json`） | 同量级；宿主负载抖动（文档 §5.4 已声明 ±30%，实测 −25%） |
| vecdem CPU 中位步时 @100 000 | — | 0.692 s | **1.204 s** | +74%（同因：共享宿主负载） |
| 接触候选对集合（device vs 暴力 O(N²) 距离判据） | 精确集合 | 0 差异 | **0 差异**（2 028 颗粒：5 603 = 5 603；见 `pair_set_check_small_n`） | ✅ <3%（0%） |
| legacy→device 前后对照（同机同配置，审计独立测） | 3 267→726 ms（文档） | 6.645→3.570 s/步（文档） | 50 653 颗粒：`pair_search 3.798 → 0.171 s`（**22×**）、`step 19.646 → 1.248 s`（**15.8×**） | 方向确认且**强于**文档；文档基线数不可复现 |

### 2.3 repro_cmd —— 审计实际执行

```bash
PYTHONPATH=src python3 -m pytest tests/test_dem3d_device_backends.py -q
# → 8 passed in 7.70s（审计工作树外、被审工作树内执行）

PYTHONPATH=src python3 benchmarks/benchmark_dem3d_gpu_100k.py --engine both --device cpu \
   --sizes 100000 --steps 2 --warmup 1 --threads 128 --tag audit_repro --outdir /tmp/audit_out
# → dem3d 2.690 s/步 (37 774 N·steps/s)、vecdem 1.204 s/步 (83 089 N·steps/s)，acceptance 自动判定落盘
# → 工件：artifacts/audit_dem_gpu_100k/bench_cpu_repro_by_audit.json
```

结论：**repro_cmd 可执行且可复现量级**（绝对值受宿主负载影响，脚本已把原始每步数组留全）。

### 2.4 GPU 腿独立复核（审计再探一次，不沿用被审者自述）

`artifacts/audit_dem_gpu_100k/gpu_probe_by_audit.json`：`torch.cuda.is_available()=False`、无 `nvidia-smi`、无 `/dev/nvidia*`、
`192.168.215.56:22` connect timeout（>6 s）、集群探针 `cj938/cj939` **仍 queued（282/281 min）**。
→ m14682 的 `honest-FAIL` 归因**准确**，不是掩盖：GPU 环境在本审计时点仍不可达。**但「不可达」不等于「达标」**，GAP 判据保持未判定。

---

## 3. 判级依据（L0–L5 逐级对照）

| 级别 | 要求 | 本单实况 |
|---|---|---|
| L0 | 文档成果 | ✅ 有（`docs/GAP_DEM_GPU_100K.md`） |
| L1 | 数学/单元验证 | ✅ `tests/test_dem3d_device_backends.py` 8 项，断言的是**不变量**而非"跑过就算"：候选集=暴力精确集（3 seed）、30 步 `positions/velocities` `torch.equal`、跨分离切向历史语义 |
| L2 | 独立功能（真代码） | ✅ `src/tensordem/dem3d_pair_search.py` 231 行（`candidate_pairs` L53、`DensePairTangentialState` L157/`lookup` L185/`update` L200），无 placeholder |
| L3 | 系统集成 | ✅ 接入主求解器：`src/tensordem/dem3d.py:39` 导入、`:149-150` 新增 `neighbor_search/tangential_store` 配置、`:705` 后端解析、`:817` `return candidate_pairs(...)`、`:1402` 稠密切向历史替换 dict；`dem_vectorized.py:94/113` `method="device"` 默认走设备原生 |
| L4 | 基准验证 | ❌ **不成立**：`benchmark` 的判据量是 GPU 步时，无任何 GPU 实测值；CPU 数按文档自身声明「不能替代 GPU 验收」。数值正确性侧已有精确参照（暴力集 + legacy 逐位轨迹），但这属 L1/L2 级验证，不抬高整体基准级 |
| L5 | 科研工程成果 | ❌ 无 external_anchor（无文献/实验吞吐锚） |

**故判 L3**（不是 L4：不能让「CPU 数」借「GPU GAP」之名过关；不是 L2：接入是真的、零回归改造是真的）。

---

## 4. 文档声称 vs 实测（差异=审计发现，按严重度）

| # | 严重度 | 声称（`docs/GAP_DEM_GPU_100K.md`） | 实测（本审计） | 影响 |
|---|---|---|---|---|
| F1 | 中（表述失实，不影响物理） | §2/§6「`neighbor_search="vectorized"` 的候选集与 legacy **完全相同**」 | 2 028 颗粒：legacy 26 169 对 vs device 5 603 对（精确=5 603）；50 653 颗粒：657 394 vs 141 858；101 614：1 319 185 vs 288 401。`legacy ⊄ 暴力集`（legacy 未做距离过滤，只做 27 邻域格枚举） | 候选集**不等价**：device 版把距离过滤内联进搜索，legacy 版留给下游。功能等价由 30 步逐位一致背书（✅），但"同一集合"是**错话**，也意味着 22× 的 pair-search 加速里混入了过滤成本 |
| F2 | 中 | §3 表头 `contacts` = 288 401（dem3d 101 614） | JSON 原键是 `components.candidate_pairs`，即**候选对**；真实接触需 `overlap>0` 过滤，未留工件 | 把候选对当接触数，会误导"接触密度 2.8/颗粒"的解读 |
| F3 | 中 | §5.2「重写前 101 614 颗粒单步 **6.645 s**」 | 该数**只存在于 `attempt_ledger.json` 的散文里，无原始 run 工件**；审计在修正后的晶格上测得 50 653 颗粒 legacy **19.6 s/步**（101 614 冷启步 34.2 s）。文档 §5.3 自述早期台架曾用退化解（触点仅 50 688）——6.645 s 很可能来自那个"几乎无接触"的假快状态 | 基线偏低 → 文档声称 1.86× 加速，实际 ≥15×；**对结论方向无害，对数字可信度有害** |
| F4 | 低 | §6「吞吐不报误差 %」处理得体 | 一致 | — |
| F5 | 低（流程） | 交付在 `agent/m14682` 分支 | **未并入 main**（ECO 镜像 `/root/tensor-ecosystem/TensorDEM` 停在 m9876） | 能力仅存在于分支，主仓不可见 |

审计未发现：伪造运行工件、把 GPU 数写成 pass、测试只跑不断言。相反，`acceptance.passed=false` 与 `fail_stage` 是**主动留证的自证清白**。

---

## 5. 建议（第 2 轮输入）

1. **补 GPU 测点**（唯一能让 L4 成立的动作）：优先取回 `cj938/cj939`；或给一个能跑 `--device cuda` 的中继；5090 FP64 约为 FP32 1/64，而 `dem3d` 是 float64 —— 预期风险大，需先备 mixed-precision 方案与数值一致性门。
2. **修 3 处措辞**（F1/F2/F3）：把"候选集相同"改为"下游接触物理等价、候选集为 legacy 的距离过滤子集"；`contacts`→`candidate_pairs`；legacy 基线如无工件则标注"单次观测，无工件"。
3. **并 main**：F5 否则下一轮复用要重新找分支。
4. 力计算（4.18 s）才是 10 万颗粒 ≤2 s 的主障碍（文档 §5.2 已自述），第 2 轮的第二刀应切键合内核而非继续压邻域搜索。

---

## 6. 审计自身局限

- 审计宿主与 m14682 同机（共享容器，200+ 在飞作业）→ 绝对步时不可比，只比**同次调用的两臂**；每步原始数组已留。
- 100 万级对比未做：`cdist(101 614²)` float64 ≈ 82 GB，精确参照只在 2 028 颗粒取；1e5 处只比 legacy/device 的**对数**，不比集合。
- `--steps 2 --warmup 1`（每臂），非文档建议的 ≥10 步：窗口受 120 s shell 上限约束，故把 legacy 臂单列执行为 50 653 颗粒。
- 未复跑 vecdem 的 GPU 路径（同上：无 CUDA）。
- 未审 m14682 之外的同族交付（GAP 家族其它锚点不在本单范围）。

---

## 7. 复现

```bash
# 被审件重跑（工作树外）
PYTHONPATH=/root/tp-agent/work/m14682/src python3 -m pytest \
    /root/tp-agent/work/m14682/tests/test_dem3d_device_backends.py -q     # 8 passed

# 官方台架（审计重跑）
cd /root/tp-agent/work/m14682 && PYTHONPATH=src python3 benchmarks/benchmark_dem3d_gpu_100k.py \
    --engine both --device cpu --sizes 100000 --steps 2 --warmup 1 --threads 128 \
    --tag audit_repro --outdir /tmp/audit_out

# 审计独立两臂对照 + 精确参照集（每次调用 <120 s，故 --only 分臂）
python3 tools/audit_dem_gpu_100k_m14682.py --repo-root /root/tp-agent/work/m14682 \
    --outdir artifacts/audit_dem_gpu_100k --only vectorized --lattice 37,37,37 --steps 2 --warmup 1
python3 tools/audit_dem_gpu_100k_m14682.py --repo-root /root/tp-agent/work/m14682 \
    --outdir artifacts/audit_dem_gpu_100k --only legacy     --lattice 37,37,37 --steps 2 --warmup 1
```

## 8. 审计图

![audit](docs/figures/audit_m14682_gap_dem_gpu_100k.png)

左：审计实测的 CPU 单步中位时延（legacy vs device 同机两臂、10 万颗粒官方台架重跑）；红虚线=GAP 判据 2 s，但判据只认 GPU，GPU 腿无值。
右：候选对集合数——legacy 恒为 device 的 4.6×（未做距离过滤），device 与暴力精确集 0 差异 → 文档「候选集与 legacy 完全相同」不成立。

## 9. 落表与 gate 状态（审计留痕）

- verdict 行：`tensor_maturity_verdicts.id=211`，`repo=TensorDEM, scope='m14682 GAP-DEM-GPU-100K', mission_id=15017(本审计单), level=3, verdict_by='board-audit'`。
- machine 复核：本审计用 `maturity_gate.judge(3, <落表 evidence>, 'TensorDEM', 15017)` 本地复算 → `gate_level=3, reasons=[]`（impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd 全部通过机器核对；`external_anchor` 缺 → L5 不成立，与判级一致）。
- **已知状态冲突（如实记录）**：该 scope+verdict_by 上先有一条 claimed **L2** 的 verdict，被 gate 于 14:55 判 `L2→L1` 并自动开返修单 **m15025**（requeued 已 succeeded）；本审计的 L3 verdict 因唯一键 `(repo,scope,verdict_by)` 冲突走 `DO UPDATE` 落进同一行，故该行 `gate_level=1 / gate_note` 是**上一条 verdict 的陈旧残留**，`gate_state` 已置回 `pending`，等下一个 gate 周期按 L3 重裁。
- 结论不受影响：**L3 = 本轮判级**；返修单 m15025 的存在恰好与"GPU 判据未测、需第二轮"的审计结论方向一致，不构成矛盾。
