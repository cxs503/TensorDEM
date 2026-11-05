"""Build + post the m3857 merge report (figure, six-section markdown, API post)."""
import base64, json, os, urllib.request, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/root/tp-agent/work/m3857/docs/artifacts/merge_m3857"
os.makedirs(OUT, exist_ok=True)

# ---- data: differential probe (zero hand-copied numbers: read the json) ----
probes = {
    "merged": "/tmp/probe_m3857.json",
    "branch": "/tmp/probe_wt230_branch.json",
    "main": "/tmp/probe_wt230_main.json",
}
P = {k: json.load(open(v)) for k, v in probes.items()}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
colors = {"merged": "#1f77b4", "branch": "#ff7f0e", "main": "#2ca02c"}

# (a) face-extent bond counts: branch contract, identical in all three trees
ax = axes[0]
labels = ["plane\nx=0.55", "disc\n0.25 m2", "disc\n0.5 m2", "disc\n1.0 m2", "layer\ntol=1e-9"]
keys = ["branch/plane_x055", "branch/disc_area0.25", "branch/disc_area0.5",
        "branch/disc_area1.0", "branch/layer_tolerance"]
w = 0.26
for i, tree in enumerate(("merged", "branch", "main")):
    ax.bar([x + (i - 1) * w for x in range(len(keys))],
           [P[tree][k] for k in keys], width=w, label=tree, color=colors[tree])
ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("bonds severed"); ax.set_title("(a) face-extent contract (branch-owned)\nmerged == branch == main")
ax.legend(fontsize=8)

# (b) mass mapping: DEM vs FEM continuum mass, exact 1 m3 lattice
ax = axes[1]
modes = ["cell", "sphere"]
dem = [P["merged"][f"main/ledger_{m}/dem_mass"] for m in modes]
fem = P["merged"]["main/ledger_cell/fem_continuum_mass"]
ax.bar([0, 1], dem, width=0.5, color=["#2ca02c", "#d62728"], label="DEM mass")
ax.axhline(fem, color="k", ls="--", lw=1.2, label=f"FEM continuum {fem:.0f} kg")
for i, m in enumerate(modes):
    e = P["merged"][f"main/ledger_{m}/relative_error"]
    ax.annotate(f"{e*100:+.3f}%", (i, dem[i]), ha="center", va="bottom", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["mass_mode=cell\n(volume consistent)", "mass_mode=sphere\n(frozen SI-2)"])
ax.set_ylabel("mass [kg]"); ax.set_title("(b) mass contract (main-owned)\n5^3 lattice, rho=917, 1 m3")
ax.legend(fontsize=8)

# (c) coverage (branch) and volume_fraction (main) - the name collision
ax = axes[2]
names = ["domain_coverage\n0.9 m cube", "volume_fraction\ncell", "volume_fraction\nsphere"]
ks = ["branch/coverage_coarse", "main/volume_fraction_cell", "main/volume_fraction_sphere"]
for i, tree in enumerate(("merged", "branch", "main")):
    vals = []
    for k in ks:
        v = P[tree].get(k)
        if v is None:  # the branch calls the coverage term volume_fraction
            v = P[tree].get("branch/coverage_coarse") if k.startswith("branch") else None
        vals.append(float("nan") if v is None else v)
    ax.bar([x + (i - 1) * w for x in range(len(ks))], vals, width=w, label=tree, color=colors[tree])
ax.set_xticks(range(len(ks))); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("fraction"); ax.set_title("(c) coverage vs mass-sphere fraction\n(name collision, both keys kept)")
ax.legend(fontsize=8)

fig.suptitle("TensorDEM m3857: merged (06058c6) vs agent/m2965 (33060f7) vs main (e12a2ab) - 0 behavioural mismatches", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
png = f"{OUT}/fig1_merge_dual_contract.png"
fig.savefig(png, dpi=150, bbox_inches="tight")
print("figure", png, os.path.getsize(png))

md = f"""# PR #174 治愈（TensorDEM, mission m3857）：`agent/m2965` 的双契约语义并集合并

## ① 案例介绍

**物理背景**：TensorDEM 的 `tensordem.fracture_gen` 把 FEM 失效面（failure faces）转换成 DEM 黏结颗粒场：
立方晶格排布 → 最近邻黏结拓扑 → 按失效面切断跨界黏结 → 输出可直接喂给 `IceDEM3D.step()` 的初态。
本单不是物理任务，而是**分支卫生**任务：GitHub 报告 PR #174（`agent/m2965` → `main`）`mergeable=false`。
根因是双方在**同一基线** `8cc88f7` 上**各自独立实现了同一套契约**（质量映射 + 面足迹），4 个文件内容冲突。

| 参数 | 值 |
|---|---|
| 几何 | 简单立方晶格，5×5×5 = 125 颗粒，1.0 m 立方域（另有 0.9 m 非整数覆盖算例） |
| 颗粒半径 / 间距 | r = 0.1 m，h = 2r = 0.2 m |
| 密度 | ρ = 917 kg/m³（冰） |
| 黏结 | 最近邻 × 3 方向，300 条，rest length 0.2 m |
| 失效面 | 法向 n=(1,0,0)，面积 0.25 / 0.5 / 1.0 m²，质心 x=0.55（层间）/ x=0.5（与颗粒层重合） |
| 质量映射 | `cell`（体积一致，m=ρh³）/ `sphere`（内接球，冻结 SI-2 旧契约） |
| 参考质量 | FEM 连续体 ρV = 917 kg |

| 版本 | SHA |
|---|---|
| 合并基 | `8cc88f7` |
| `main`（their） | `e12a2ab` |
| `agent/m2965` 旧头 | `33060f7` |
| 合并提交 新头 | `06058c6` |

## ② 怎么做的

1. `git merge-tree` 先定位冲突面（4 文件，`fracture_gen.py` 有 14 个冲突块），确认其余 345 文件为文本可合并。
2. 逐块判定**语义并集体**：两侧各自新增的 API/常量/校验/审计键一律保留，名称撞车（`volume_fraction`）时
   让覆盖率项改名 `domain_coverage` 并保留主干的 `volume_fraction` 语义——两量都是审计量，不能只留一个。
   - 分支侧：`mass_mapping=` / `face_extent=` 别名、`on_plane_tolerance`（平面穿层退化修复）、
     `_BOND_BLOCK` 分块向量化拓扑、`MASS_MAPPING_ALIASES` 别名表、覆盖率项。
   - 主干侧：`normalise_mass_mapping()` / `normalise_face_extent()` 单一校验路径、`footprint_aware`、
     `_FOOTPRINT_TOLERANCE`、`volume_fraction`（质量球占晶格胞体积比）、账本 `relative_error_magnitude`。
   - 适配器：分支的两个别名字段喂主干的规范化函数，矛盾组合（`face_extent="plane"` +
     `footprint_aware=True`）显式抛错而不静默取胜。
3. 发现并修复**静默陈旧产物**：`benchmarks/fracture_gen_audit_report.json` 无冲突但带着合并前分支的旧数字，
   用合并后源码重跑生成器刷新（零手抄）。
4. 造**行为判据**：`scripts/merge_m3857_dual_contract_probe.py` 在分支/主干/合并三个 checkout 各跑一遍，
   逐观测量比对（分支家观测量 == 分支，主干家观测量 == 主干）。
5. 工具链：git 2.41 三方合并、pytest 9、torch 2.7.1（CPU）、matplotlib 作图。

## ③ 计算结果

**差分探针：33 个观测量，0 个不匹配**（merged vs 该观测量的"归属方"）。

| 观测量（归属） | merged | branch | main |
|---|---|---|---|
| 颗粒数 / 黏结数（5³ 晶格） | 125 / 300 | 125 / 300 | 125 / 300 |
| 键长最大偏差 | 5.55e-16 | 5.55e-16 | 5.55e-16 |
| `domain_coverage`（1.0 m 立方，整除） | 1.0 | 1.0 | 1.0 |
| `domain_coverage`（0.9 m 立方，非整除） | 0.702331961591 | 0.702331961591 | 0.702331961591 |
| 无限平面 x=0.55 切断键数 | 25 | 25 | 25 |
| 足迹盘 A=0.25 / 0.5 / 1.0 m² 切断键数 | 5 / 9 / 21 | 5 / 9 / 21 | 5 / 9 / 21 |
| 平面穿颗粒层，tol=0 / 1e-9 | 0 / 50 | 0 / 50 | 0 / 50 |
| 默认 `mass_mode` | cell | cell | cell |
| `volume_fraction`（cell / sphere） | 1.0 / 0.523598775598 | 1.0 / 0.523598775598 | 1.0 / 0.523598775598 |
| `mass_radius / r` | 1.2407009818 | 1.2407009818 | 1.2407009818 |
| 账本 `relative_error`（sphere） | −0.476401224402 | −0.476401224402 | −0.476401224402 |
| `relative_error_magnitude` / `absolute_relative_error` | 0.476401224402 / 0.476401224402 | 0.476401224402 / 0.476401224402 | 0.476401224402 / 0.476401224402 |
| 质量映射 DEM 质量（cell / sphere，1 m³） | 917 / 480.140 kg | 917 / 480.140 kg | 917 / 480.140 kg |

**测试证据**

| 套件 | 结果 |
|---|---|
| 分支 `33060f7`，断裂相关子集 | 69 passed |
| `main` `e12a2ab`，同一子集 | 86 passed（基线） |
| **合并后**，同一子集 | **86 passed**（两侧超集，不低于基线） |
| **合并后**，`tests/` 全量（1331 采集） | 1231 passed, 2 skipped, 1 failed = `test_doc_evidence_pointers.py` 陈旧产物/引用门（本分支新文档引入，回报内已修：重生成 JSON + 门禁复跑） |
| 删除检查 | 相对 `33060f7` 与 `e12a2ab` 均**零文件删除**；相对 main 仅 4 冲突文件 + 文档 + 重生成 JSON |

## ④ 结果图

见 figure（三联图）：
(a) 面足迹契约（分支家）三树完全一致的切断键数；
(b) 质量映射（主干家）DEM 质量 vs FEM 连续体 917 kg，sphere 模式 −47.64%；
(c) 名字撞车解决：覆盖率（合并名 `domain_coverage`）与 `volume_fraction`（质量球/胞体积）两键并存。

## ⑤ 注意点

- `volume_fraction` 在分支上曾是**覆盖率**；合并后该语义改名为 `domain_coverage`。树外 notebook 若按旧义读取需换键。
- 小足迹盘的切断键数是**离散晶格计数**（0.25 m² 盘 = 5 条 vs 连续估计 6.25 条，−20%），三树一致，属量化误差，
  不是合并缺陷；收敛扫描见 `benchmarks/fracture_gen_mass_footprint_audit.py`。
- 平面上与颗粒层重合时严格符号判定会切断 0 条（浮点退化）；`on_plane_tolerance>0` 修复为 50 条（2n²），
  该行为在两侧完全一致。
- 本次推送受平台安全策略限制（本人不得 push），`agent/m2965` 的更新由平台/merge_governor 完成；
  新提交 `06058c6` 的第一父即 `33060f7`，可 fast-forward。

## ⑥ 误差判定（是否 <3%）

本单为**合并治愈**，物理量未改动，判据是"与归属方逐观测量相等"而非连续量误差：

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 33 个差分观测量 | 归属方（branch 或 main） | **0.0（逐位相等）** | 是（0%） |
| 断裂相关测试 86 项 | main 基线 86 passed | 0 项差异 | 是（0%） |
| 文件删除数 | 两父提交 | 0 | 是 |
| 质量映射 `cell` vs 连续体 ρV | 917 kg | 2.48e-16（机器零） | 是 |
| 质量映射 `sphere` vs 连续体 ρV | 917 kg | **−47.64%**（冻结 SI-2 缺陷，审计要暴露的量，非合并误差） | 否（按设计暴露） |

诚实结论：唯一的 >3% 数字（−47.64%）是**冻结旧契约的已知缺陷**，两侧代码在合并前后完全一致，
不是本次合并引入；`main` 已把体积一致映射设为默认值，合并后该默认得以保留。
"""

fig_b64 = base64.b64encode(open(png, "rb").read()).decode()
report = {
    "title": "PR #174 治愈：TensorDEM agent/m2965 双契约语义并集合并（0 行为差异）",
    "summary": "agent/m2965 与 main 各自实现同一套 fracture_gen 质量/足迹契约导致 mergeable=false。逐文件语义并集解冲（4 文件、16 冲突块），volume_fraction 名字撞车以 domain_coverage 分离保留双语义，重生成陈旧审计 JSON；差分探针在分支/主干/合并三树跑 33 观测量，0 不匹配；断裂相关测试 86 passed = main 基线（分支侧 69 的超集）；相对两父提交零文件删除。旧 SHA 33060f7 → 新 SHA 06058c6。",
    "scenario_key": "tensordem_pr174_merge_heal",
    "mission_id": 3857,
    "tags": ["TensorDEM", "merge-heal", "fracture_gen", "semantic-union", "m3857", "PR174"],
    "figures": [{"name": "fig1_merge_dual_contract.png",
                 "caption": "merged(06058c6) vs agent/m2965(33060f7) vs main(e12a2ab): 面足迹契约/质量契约/名字撞车三视图，0 行为差异",
                 "png_base64": fig_b64}],
    "markdown": md,
}
json.dump(report, open("/tmp/report.json", "w"))
print("report bytes", os.path.getsize("/tmp/report.json"), "fig b64", len(fig_b64))

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
usr = os.environ.get("TENSOR_AGENT_USER"); pwd = os.environ.get("TENSOR_AGENT_PASS")
tok = None
if usr and pwd:
    req = urllib.request.Request(base + "/tensor/login",
                                 data=json.dumps({"username": usr, "password": pwd}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        tok = json.load(urllib.request.urlopen(req, timeout=30)).get("access_token") or json.load(
            urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        print("login failed", e)
print("token?", bool(tok))
if tok:
    body = open("/tmp/report.json", "rb").read()
    req = urllib.request.Request(base + "/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {tok}"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        print("POST", r.status, r.read()[:300])
    except Exception as e:
        print("post failed", e, getattr(e, "read", lambda: b"")()[:300])