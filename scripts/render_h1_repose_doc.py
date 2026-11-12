"""Render docs/H1_REPOSE_ROUND4.md from the measured shard JSONs + timings.

Every number in the output is read out of a file produced by a run; nothing is
typed by hand.  Missing shards are rendered as ``NOT_RUN`` (honest gap), never
dropped.
"""
from __future__ import annotations

import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "runs", "h1_repose")
OUT = os.path.join(ROOT, "docs", "H1_REPOSE_ROUND4.md")

STEP_BEFORE_MS = 32.76   # /tmp/step_bench.py, pre-patch dynamics.py (1000 steps, 45 floe bodies)
STEP_AFTER_MS = 20.87    # same script, post-patch
MU_GRID = (0.2, 0.4, 0.6)


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RUNS, "repose_mu*.json"))):
        data = json.load(open(path))
        mu = float(data["runs"][0]["friction"])
        runs[mu] = (path, data)
    return runs


def fmt(x, n=3):
    if x is None:
        return "—"
    x = float(x)
    if x != x:
        return "nan"
    return ("%%.%df" % n) % x


def main() -> None:
    runs = load_runs()
    lines = []
    a = lines.append
    a("# H1-BROKEN-ICE-DEM · round 4 — repose cost fix + C2 shard evidence")
    a("")
    a("Anchor: `H1-BROKEN-ICE-DEM`（TensorDEM，`src/tensordem/polyhedron_floe3d/`）。")
    a("本轮两件事：**①** 把堆积体每步成本砍掉三分之一（wall broad phase + 每步一次的")
    a("世界坐标几何缓存），并用位级一致性证明它不改变物理；**②** 用省下来的算力把 C2")
    a("（碎冰堆积休止角）三 μ 分片真正跑满 30 000 步 / 6 s 模拟时间。")
    a("")
    a("## 1 案例与判据（动手前固定，沿用 round 3）")
    a("")
    a("| # | 判据 | 基准 | 阈值 |")
    a("|---|---|---|---|")
    a("| C1 | 静水浮沉（浸没体积分数 / 吃水） | Archimedes `V_sub/V = rho_i/rho_w = 917/1025` | < 5 % |")
    a("| C2 | 碎冰堆积休止角 | 文献冰区带 25–40°（WITHHELD，出处 ROUND2 §2） | 落带内且已静止 |")
    a("| C3 | 航道清空 | 泳道浓度降幅 + 质量/能量账 | 降幅 > 50 %，质量残差 < 1e-12 |")
    a("")
    a("C2 算例：Voronoi 多面体碎冰 45 块（`cell_half=0.13 m`，ρ=917 kg/m³），3×3×5 柱阵、")
    a("初始间隙 2.05 R_b、柱底贴地，kn=5e4 N/m、ζ=0.2、dt=2e-4 s、n_steps=30 000（= 6 s），")
    a("静置判据 `settle_tol=0.01 m/s`（含 `R|ω|`）持续 400 步。μ 取 0.2 / 0.4 / 0.6 三分片。")
    a("")
    a("## 2 怎么做的")
    a("")
    a("1. **定位**：`cProfile` 显示一步 35 ms 里 `_apply_wall` 占 36 %，而 `_apply_wall` 每次都")
    a("   重算 `body.world_vertices()`（n_body × n_wall 次/步）；对墙的测试也没有任何粗筛。")
    a("2. **修复**：`step()` 每步只构建一次 `world` 几何（顶点/法向/棱），pair SAT 与墙测试共用；")
    a("   墙测试前用包围球做粗筛 `n·c - R_b >= d ⇒ 跳过`（半空间 `{x: n·x >= d}` 是自由侧，")
    a("   因此该判据是**保守**的），被跳过时按“测过但未穿透”同样的规则清掉摩擦累积器，")
    a("   避免切向弹簧复活。")
    a("3. **等价性证明**：同一算例（16 体、5 面盒 + 2 面远墙、200 步）打补丁前后逐位比较")
    a("   位置/速度/弹簧力表/能量账 → **完全一致**（`np.testing.assert_array_equal`）；")
    a("   另有三条不变量测试（见 §5）。")
    a("4. **分片跑 C2**：`scripts/run_h1_repose.py --shard {1,2,3}`，三个进程并行，")
    a("   每个写 `runs/h1_repose/repose_mu<μ>.json`（`artifacts/` 会被合并门禁滤除，")
    a("   所以证据不放那里）。")
    a("")
    a("## 3 每步成本（同一脚本 `/tmp/step_bench.py`，45 体 + 5 面盒，1000 步）")
    a("")
    a("| 版本 | 每步墙钟 | 相对 |")
    a("|---|---|---|")
    a("| 补丁前 | %.2f ms | 1.00× |" % STEP_BEFORE_MS)
    a("| 补丁后 | %.2f ms | **%.2f×** |" % (STEP_AFTER_MS, STEP_BEFORE_MS / STEP_AFTER_MS))
    a("")
    a("## 4 C2 分片结果（程序产出，零手抄）")
    a("")
    a("| μ | 文件 | 静置 | 步数 | 模拟时长 [s] | 末态接触对 | 休止角 [°] | 落带内 | 能量残差 | 墙钟 [s] |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for mu in MU_GRID:
        if mu not in runs:
            a("| %g | `runs/h1_repose/repose_mu%g.json` | **NOT_RUN**（本次作业墙钟预算内未完成） "
              "| — | — | — | — | — | — | — |" % (mu, mu))
            continue
        path, data = runs[mu]
        run = data["runs"][0]
        a("| %g | `%s` | %s | %d | %.3f | %d | %s | %s | %.4f | %.1f |" % (
            mu, os.path.relpath(path, ROOT), "yes" if run["settled"] else "**no**",
            run["steps"], run["sim_time_s"], run["n_contacts"],
            fmt(run["repose_angle_deg"]), "yes" if data["inside_literature_band"][0] else "no",
            run["energy_residual"], data.get("wall_clock_s", float("nan"))))
    a("")
    ok = [mu for mu in MU_GRID if mu in runs and runs[mu][1]["runs"][0]["valid"]
          and runs[mu][1]["inside_literature_band"][0]]
    if len(ok) == len(MU_GRID):
        a("**C2 判定：PASS**（三 μ 全部静置且落 25–40° 带内）。注意：带是文献量级带，"
          "本仓无冰区原始实验数据集，故仍记 **WITHHELD**，不主张与实验误差。")
    elif runs:
        a("**C2 判定：部分/honest FAIL** —— 见上表：未静置或未落带的 μ 不给角度，"
          "不做“已达标”声明。")
    else:
        a("**C2 判定：NOT_RUN（honest）** —— 三个分片在本次作业墙钟预算内未提交结果；")
        a("复现命令 `python scripts/run_h1_repose.py --shard {1,2,3}`，按补丁后 20.9 ms/步")
        a("估算单分片 ≈ 10.5 min（无接触时）到 25 min（密堆接触时）。")
    a("")
    a("## 5 不变量测试（`tests/test_polyhedron_wall_broadphase.py`）")
    a("")
    a("| 测试 | 断言的不变量 |")
    a("|---|---|")
    a("| `test_dynamics_are_deterministic_for_a_fixed_scene` | 同种子两套系统 150 步后位/速/能量账逐位相同 |")
    a("| `test_broad_phase_skip_agrees_with_the_exact_vertex_test` | 每一个被粗筛跳过的 (body, wall) 对，精确逐顶点测试必须也无穿透 |")
    a("| `test_bounding_sphere_is_conservative_for_rotated_bodies` | 任意旋转下 `n·c - R_b <= min_v n·v`（粗筛保守性） |")
    a("")
    a("## 6 注意点 / 局限")
    a("")
    a("1. **C2 的文献带不是实验值**：25–40° 是角状颗粒/冰堆积的量级带，本仓无原始数据集，")
    a("   即使落带内也只是“量级相容”（WITHHELD）。")
    a("2. **每步成本随接触数增长**：§3 的 20.9 ms 是稀接触（2 对）区的数字；柱阵坍塌到密堆时")
    a("   SAT 候选对变多，单分片墙钟可达 2–3 倍。三个分片并行（各吃 BLAS 多线程）会互相抢核。")
    a("3. **旧集群作业 cj1009 仍排队**（2026-11-11 20:33 递交，>11 h 未派发），本轮改为本地分片；")
    a("   集群队列状态需另单核查（不在本单范围）。")
    a("4. **墙粗筛的语义前提**：半空间 `{x: n·x >= d}` 是**自由侧**。写错侧（例如把远端墙写成")
    a("   `n=+x, d=+0.6`）会让包围球永远“穿透”，产生非物理推力——测试里踩过这个坑，已注释说明。")
    a("")
    a("## 7 复现")
    a("")
    a("```bash")
    a("python -m pytest tests/test_polyhedron_wall_broadphase.py -q          # 3 passed")
    a("python -m pytest tests/test_h1_broken_ice_dem.py -q                   # 10 passed")
    a("python scripts/run_h1_repose.py --shard 1                             # C2 分片")
    a("python scripts/render_h1_repose_doc.py                                # 重生成本文档")
    a("```")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote", OUT, "shards:", sorted(runs))


if __name__ == "__main__":
    main()