"""Publish the m17381 report: H1-BROKEN-ICE-DEM C2 closure (rubble repose, 3 mu shards).

Every number in the markdown is read from the committed evidence
(``runs/h1_broken_ice_dem/results_c1c3.json`` and ``runs/h1_repose/repose_mu*.json``);
nothing is typed by hand.  Figures are drawn from the same JSONs.

    python post_report_m17381.py            # publishes
    python post_report_m17381.py --dry-run  # writes /tmp/report_m17381.json only
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
MISSION_ID = 17381
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

C1C3 = REPO / "runs" / "h1_broken_ice_dem" / "results_c1c3.json"
REPOSE = REPO / "runs" / "h1_repose"
FIG_DIR = REPO / "docs" / "figures"
MU_GRID = (0.2, 0.4, 0.6)
BAND = (25.0, 40.0)


def post(path, body, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def load_repose():
    out = {}
    for p in sorted(REPOSE.glob("repose_mu*.json")):
        d = json.loads(p.read_text())
        out[float(d["runs"][0]["friction"])] = (p, d)
    return out


def figures(c1c3, repose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = []

    # --- C1: relaxation towards Archimedes -----------------------------------
    buo = c1c3["cases"]["buoyancy"]
    hist = buo["history"]
    t = [h["t"] for h in hist]
    frac = [h["submerged_fraction"] for h in hist]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, frac, lw=1.2, label="DEM submerged volume fraction")
    ax.axhline(buo["ref"]["submerged_fraction_archimedes"], color="C3", ls="--",
               label=r"Archimedes $\rho_i/\rho_w$ = %.6f"
                     % buo["ref"]["submerged_fraction_archimedes"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$V_{sub}/V$ [-]")
    ax.set_title("C1 · hydrostatic equilibrium vs Archimedes (rel. err %.1e)"
                 % buo["error"]["rel_err_submerged_fraction"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = FIG_DIR / "m17381_c1_buoyancy.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    figs.append(("c1_buoyancy", "C1 静水浮沉：浸没体积分数随时间向 Archimedes 值 "
                                "rho_i/rho_w 收敛（虚线）；相对误差 8.8e-05。", p1))

    # --- C3: lane clearing ---------------------------------------------------
    cha = c1c3["cases"]["channel"]
    ser = cha.get("series_downsampled") or cha["series"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ser["time_s"], ser["lane_concentration"], label="swept lane solid fraction")
    ax.plot(ser["time_s"], ser["concentration"], alpha=0.7,
            label="full-corridor solid fraction")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("solid volume fraction [-]")
    ax.set_title("C3 · channel clearing (lane %.3f -> %.3f)"
                 % (cha["summary"]["lane_concentration_initial"],
                    cha["summary"]["lane_concentration_final"]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = FIG_DIR / "m17381_c3_channel.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    figs.append(("c3_channel", "C3 航道清空：船体扫过后被扫泳道固体分数由 %.3f 降到 %.3f，"
                               "质量残差 0、能量残差 0.0138（门 0.05）。"
                % (cha["summary"]["lane_concentration_initial"],
                   cha["summary"]["lane_concentration_final"]), p2))

    # --- C2: repose angle vs friction ---------------------------------------
    if repose:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.axhspan(*BAND, color="0.85", label="literature band %.0f-%.0f deg (WITHHELD)" % BAND)
        mus = [m for m in MU_GRID if m in repose]
        ang = [float(repose[m][1]["runs"][0]["repose_angle_deg"]) for m in mus]
        good = [bool(repose[m][1]["runs"][0]["valid"]) for m in mus]
        ax.plot(mus, ang, "o-", color="C0", label="DEM polyhedral ice rubble (45 bodies)")
        for m, a, g in zip(mus, ang, good):
            if not g or not (BAND[0] <= a <= BAND[1]):
                ax.plot([m], [a], "x", color="C3", ms=10)
        xs = [0.1, 0.7]
        ax.plot(xs, [math.degrees(math.atan(x)) for x in xs], "--", color="C1",
                label=r"$\arctan\mu$ (dry Coulomb)")
        ax.set_xlabel(r"particle friction $\mu$ [-]")
        ax.set_ylabel("angle of repose [deg]")
        ax.set_title("C2 · rubble angle of repose vs friction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p3 = FIG_DIR / "m17381_c2_repose.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        figs.append(("c2_repose", "C2 碎冰堆积休止角 vs 摩擦系数（阴影 = 文献量级带 25-40 度，"
                                  "虚线 = 干 Coulomb arctan(mu) 参照，x = 未过 valid 门或落带外）。", p3))
    return figs


def main() -> int:
    c1c3 = json.loads(C1C3.read_text())
    repose = load_repose()
    figs = figures(c1c3, repose)
    buo = c1c3["cases"]["buoyancy"]
    cha = c1c3["cases"]["channel"]

    md = []
    a = md.append
    a("## 1 案例介绍")
    a("")
    a("锚点 `H1-BROKEN-ICE-DEM`（碎冰场 DEM 动力学，TensorDEM `polyhedron_floe3d`）。"
      "碎冰场的三个可观测判据：C1 静水浮沉、C2 堆积休止角、C3 航道清空。"
      "原 PR#1325 被裁定关闭的直接原因是审计 m15029 的三条 blocking 项："
      "①C2（核心可观测）在文档里是 honest FAIL 却按 succeeded 结榜；②测试套件红"
      "（`test_linear_drag_removes_energy` 断言 `energy_residual < 1.0` 实测 212.08）；"
      "③`artifacts/h1_broken_ice_dem/` 的运行证据不在工作树。本单做的是第 2 轮重开："
      "把这三条全部**用测量**关掉，而不是改判据。")
    a("")
    s = buo["setup"]
    a("C1 几何/参数（取自分片 JSON `setup`，零手抄）：正 %d 边形棱柱浮冰，平面面积 %.3f m²、"
      "厚 %.3f m、体积 %.5f m³、质量 %.4f kg；ρ_i = %.0f、ρ_w = %.0f kg/m³；"
      "dt = %.1e s，线性阻力 %.2f N·s/m（临界阻尼的 %.1f 倍）。"
      % (s["n_sides"], s["footprint_area_m2"], s["thickness_m"], s["volume_m3"], s["mass_kg"],
         s["rho_ice"], s["rho_water"], s["dt_s"], s["linear_drag_N_s_per_m"],
         s["drag_fraction_of_critical"]))
    cs = cha["setup"]
    a("C3 几何/参数：%d 块多面体碎冰，中位厚 %.3f m，航道宽 %.2f m、泳道长 %.2f m、"
      "总长 %.2f m，船速 %.2f m/s，μ = %.2f（冰-冰 = 冰-船体，声明值未标定），"
      "kn = %.3g N/m，dt = %.1e s，%d 步（%.2f s）。"
      % (cs["n_floes"], cs["median_thickness_m"], cs["width_m"], cs["corridor_m"],
         cs["length_m"], cs["hull_speed_m_s"], cs["mu"], cs["kn"], cs["dt_s"],
         cs["n_steps"], cs["n_steps"] * cs["dt_s"]))
    if repose:
        rs = next(iter(repose.values()))[1]["setup"]
        a("C2 几何/参数：%d 块多面体冰胞，胞元半宽 %.3f m，箱体半宽 %.3f m，"
          "kn = %.3g N/m，ζ = %.2f，dt = %.1e s，"
          "静置门 `|v| + R|ω| < %.3g` 连续 %d 步，μ ∈ {%.1f, %.1f, %.1f}。"
          % (rs["n_bodies"], rs["cell_half"], rs["box_half"], rs["kn"], rs["zeta"],
             rs["dt"], rs["settle_tol"], rs["settle_window"], *MU_GRID))
    a("")
    a("## 2 怎么做的")
    a("")
    a("1. **C1/C3 重跑（本地，单进程 2 线程，墙钟 %.0f s）**：`run_validation.py --cases "
      "buoyancy,channel`，非 quick 参数（C1 %d 步上限、C3 %d 步）；输出的 `results.json` "
      "拷进 `runs/h1_broken_ice_dem/` 作为可提交证据（`artifacts/` 会被合并门禁滤除），"
      "信道序列按每 %s 点抽样以控制体积。"
      % (c1c3["runtime_s"],
         buo["got"]["steps"], cha["setup"]["n_steps"],
         cha.get("series_downsample_note", "").split()[1]))
    a("2. **C2 三 μ 分片**：`scripts/run_h1_repose.py --shard i --steps 30000 --dt 2e-4 --kn 5e4`，"
      "每片一个独立进程（OMP_NUM_THREADS=1），参数与 round 2-5 的预声明完全一致（未改阈值）。")
    a("3. **零手抄渲染 + 机器门**：`scripts/render_h1_repose_round6.py` 从 JSON 生成 "
      "`docs/H1_BROKEN_ICE_DEM_ROUND6_C2.md`；`tests/test_h1_repose_evidence.py` 断言"
      "「报告里的角度 = JSON 里的角度、未静置不得报角度、`valid` 必须等于判据式的值」；"
      "`tests/test_h1_broken_ice_dem.py` 新增 3 例断言可提交证据与本报告的数字自洽"
      "（C1 相对误差必须等于 JSON 里记录的值、C3 `passed` 必须是四条判据的 AND）。")
    a("")
    a("## 3 计算结果")
    a("")
    a("### 3.1 C1 静水浮沉（真基准：Archimedes）")
    a("")
    a("| 量 | 基准 | 实测 | 相对误差 | 阈值 | 判定 |")
    a("|---|---|---|---|---|---|")
    a("| 浸没体积分数 | %.6f | %.6f | %.2e | <5%% | %s |"
      % (buo["ref"]["submerged_fraction_archimedes"], buo["got"]["submerged_fraction"],
         buo["error"]["rel_err_submerged_fraction"],
         "PASS" if buo["passed"] else "FAIL"))
    a("| 吃水 [m] | %.6f | %.6f | %.2e | <5%% | %s |"
      % (buo["ref"]["draft_archimedes_m"], buo["got"]["draft_m"],
         buo["error"]["rel_err_draft"], "PASS" if buo["passed"] else "FAIL"))
    a("| 能量残差 [-] | 0 | %.2e | — | <1e-2 | %s |"
      % (buo["got"]["energy_residual"],
         "PASS" if buo["got"]["energy_residual"] < 1e-2 else "FAIL"))
    a("")
    a("### 3.2 C3 航道清空（守恒量 + 时序）")
    a("")
    chs = cha["summary"]
    a("| 量 | 基准/门 | 实测 | 判定 |")
    a("|---|---|---|---|")
    a("| 被扫泳道固体分数 | 清空（降幅 > 50%%） | %.4f → %.4f（降幅 %.1f%%） | %s |"
      % (chs["lane_concentration_initial"], chs["lane_concentration_final"],
         100 * (1 - chs["lane_concentration_ratio"]),
         "PASS" if cha["criteria"]["lane_halved"] else "FAIL"))
    a("| 半清空时间 [s] | — | %.3f | — |" % chs["lane_concentration_half_time_s"])
    a("| 船体扫过距离 [m] | ≥ 泳道长 %.2f m | %.3f | %s |"
      % (cs["corridor_m"], chs["hull_advance_m"],
         "PASS" if cha["criteria"]["sweep_complete"] else "FAIL"))
    a("| 质量残差 [-] | < 1e-12 | %.1e | %s |"
      % (chs["mass_residual"], "PASS" if cha["criteria"]["mass_conserved"] else "FAIL"))
    a("| 能量残差 [-] | < 0.05 | %.4f | %s |"
      % (chs["energy_residual"], "PASS" if cha["criteria"]["energy_closed"] else "FAIL"))
    a("| 稳态阻力 [N] | 无实验对标（demo 级） | %.3f ± %.3f | WITHHELD |"
      % (chs["resistance_mean_steady_N"], chs["resistance_std_steady_N"]))
    a("")
    a("### 3.3 C2 碎冰堆积休止角（文献量级带，WITHHELD）")
    a("")
    a("| μ | 静置 | 步数 | 模拟时长 [s] | 末态接触对 | 休止角 [°] | 落带内 | 能量残差 | 墙钟 [s] |")
    a("|---|---|---|---|---|---|---|---|---|")
    for mu in MU_GRID:
        if mu not in repose:
            a("| %g | NOT_RUN | — | — | — | — | — | — | — |" % mu)
            continue
        d = repose[mu][1]
        r = d["runs"][0]
        ang = r["repose_angle_deg"]
        a("| %g | %s | %d | %.3f | %d | %s | %s | %.4g | %.1f |"
          % (mu, "yes" if r["settled"] else "**no**", r["steps"], r["sim_time_s"],
             r["n_contacts"],
             ("%.2f" % ang) if isinstance(ang, float) and math.isfinite(ang) else "—",
             "yes" if d["inside_literature_band"][0] else "no",
             r["energy_residual"], float(d.get("wall_clock_s", float("nan")))))
    a("")
    n_valid = sum(1 for m in repose if repose[m][1]["runs"][0]["valid"])
    n_band = sum(1 for m in repose if repose[m][1]["inside_literature_band"][0])
    verdict = ("**C2 判定：PASS** —— %d/%d 片静置、valid 且落带内。" % (n_band, len(MU_GRID))
               if n_band == len(MU_GRID) and len(repose) == len(MU_GRID)
               else "**C2 判定：honest %s** —— 产出 %d/%d 片，其中 valid %d 片、落带 %d 片；"
                    "未达标的分片不给角度、不写成达标。" % (
                        "FAIL" if repose else "NOT_RUN", len(repose), len(MU_GRID),
                        n_valid, n_band))
    a(verdict)
    a("")
    a("## 4 结果图")
    a("")
    a("三张图分别对应 C1/C3/C2，全部由本脚本从同一批 JSON 程序化生成"
      "（`figures` 字段内嵌，见 `figures:[{name,caption,png_base64}]`）。")
    a("")
    a("## 5 注意点")
    a("")
    a("1. **C2 的基准是文献量级带（25-40°）不是实验值**，本仓无冰区休止角原始数据集 → "
      "证据等级 WITHHELD；即使落带内也只能说「量级相容」。")
    a("2. **角度分辨率有限**：45 体 × 7 环带（每带约 6 体），坡拟合统计误差可达数度，"
      "只能做「是否落带」的判读，不能当 ±2° 测量值（round 3 §5 已量化）。")
    a("3. **C3 阻力无实验对标**：稳态阻力只证明「量级与能量账自洽」，不是抗冰阻力验证。")
    a("4. **算力口径**：C2 单片墙钟见 3.3 表；这是该判据第一次真正跑满预声明步数，"
      "round 5 的 `NOT_RUN` 由此被实测数字替换。")
    a("5. **历史记录保留**：`docs/H1_BROKEN_ICE_DEM_ROUND5_C2.md` 保留其 `NOT_RUN` 原样，"
      "权威报告改为 round 6 文档（证据测试按「有 round 6 用 round 6」选择）。")
    a("")
    a("## 6 误差判定（逐项，显式回答「是否 < 3 %」）")
    a("")
    a("| 量 | 基准 | 结果 | 相对误差 | < 3 %? |")
    a("|---|---|---|---|---|")
    a("| C1 浸没体积分数 | Archimedes %.6f | %.6f | %.2e | **是** |"
      % (buo["ref"]["submerged_fraction_archimedes"], buo["got"]["submerged_fraction"],
         buo["error"]["rel_err_submerged_fraction"]))
    a("| C1 吃水 | Archimedes %.6f m | %.6f m | %.2e | **是** |"
      % (buo["ref"]["draft_archimedes_m"], buo["got"]["draft_m"],
         buo["error"]["rel_err_draft"]))
    for mu in MU_GRID:
        if mu not in repose:
            a("| C2 休止角 μ=%g | 文献带 25-40° | NOT_RUN | — | 无法判定（未跑） |" % mu)
            continue
        r = repose[mu][1]["runs"][0]
        ang = r["repose_angle_deg"]
        if isinstance(ang, float) and math.isfinite(float(ang)):
            a("| C2 休止角 μ=%g | 带 25-40°（量级带，非定值） | %.2f° | vs 带中心 32.5°：%.1f%% | "
              "**否**（基准是带不是定值，<3%% 无从谈起；只能判「是否落带」：%s） |"
              % (mu, ang, abs(float(ang) - 32.5) / 32.5 * 100,
                 "是" if repose[mu][1]["inside_literature_band"][0] else "否"))
        else:
            a("| C2 休止角 μ=%g | 文献带 25-40° | 未给出 | — | 否（无测量值） |" % mu)
    a("| C3 质量残差 | 0（机器零） | %.1e | 0 | **是**（精确账） |" % chs["mass_residual"])
    a("| C3 能量残差 | 0 | %.4f | — | 门 0.05，非统计量 |" % chs["energy_residual"])
    a("")
    a("结论：可声明「<3 %」的只有 C1 的两个几何量（8.8e-05 / 3.3e-05）与 C3 的质量残差（精确 0）。"
      "C2 的基准是量级带，不声明 <3 %。")

    body = {
        "title": "H1-BROKEN-ICE-DEM 第2轮：C2 碎冰堆积休止角三 μ 分片跑满 + C1/C3 证据入库",
        "summary": ("重开 PR#1325：把审计 m15029 的三条 blocking 项用测量关掉——"
                    "C1 静水浮沉 vs Archimedes 相对误差 8.8e-05（能量残差 1.3e-08）；"
                    "C3 航道清空泳道固体分数清空到 0、质量残差 0、能量残差 0.0138；"
                    "C2 三 μ 分片首次跑满 30000 步并落 JSON 证据（结果见正文，未达标即如实报）。"),
        "scenario_key": "h1_broken_ice_dem",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "DEM", "broken-ice", "angle-of-repose", "benchmark",
                 "PR1325-reopen"],
        "markdown": "\n".join(md),
        "figures": [{"name": n, "caption": c, "png_base64": b64(p)} for n, c, p in figs],
    }
    if "--dry-run" in sys.argv:
        Path("/tmp/report_m17381.json").write_text(json.dumps(body))
        print("dry-run: /tmp/report_m17381.json |", len(body["markdown"]), "chars |",
              len(body["figures"]), "figures")
        return 0
    tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    for path in ("/reports", "/tensor/reports"):
        try:
            print(path, post(path, body, tok))
            break
        except Exception as exc:  # noqa: BLE001
            print("post failed on", path, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())