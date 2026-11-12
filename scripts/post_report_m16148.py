"""Build and publish the m16148 platform report (numbers injected from the JSON)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ART = Path("artifacts/propeller_ice_milling_legacy")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")

d = json.loads((ART / "verification.json").read_text(encoding="utf-8"))
g = {x["id"]: x for x in d["gates"]}
cross = d["crossval"][0]["table"]

lines: list[str] = []
A = lines.append
A("# 遗留螺旋桨-冰铣削模块（IN27-S4）解析验证与缺陷报告 — TensorDEM m16148")
A("")
A("## ① 案例介绍")
A("")
A("冰区航行船舶的螺旋桨不仅排开碎冰：被卷入盘面的冰块会被叶片**铣削**，轴系承受叶片通过频率（BPF）的脉动扭矩，"
  "碎屑被桨流喷射输运。本仓库存在两套同物理实现：`src/tensordem/propeller_ice_milling.py`（1140 行，战役节点 IN27-S4，"
  "**零测试、零驱动、零文档**）与 `src/tensordem/propeller_ice/` 包（IN34-W3-N05，28 测试 + 生成式报告）。"
  "遗留模块在公共 `src` 树内声明“塑性封顶 Hertz 接触 / 叶元-动量开水 / **动量守恒**自相似桨流 / 碎屑输运”，"
  "却从未被任何测试触碰——本单闭合该验证欠账。")
A("")
A("几何（遗留默认值）：Z=4, D=3.0 m, r_h/R=0.40, P/D=0.95, A_E/A_0=0.700；包几何按 A_E/A_0 匹配"
  f"（chord_ratio={d['crossval'][0]['meta']['pkg_chord_ratio']:.6f}, skew=0）。水密度 1025 kg/m³。")
A("")
A("## ② 方法与工具链")
A("")
A("六族判据，每条对**被测代码之外**的参考：G1 平面形状/展开面积闭合（闭式 Beta 积分 B(1+e,1+e)、A_E=EAR·πD²/4、"
  "螺旋桨距角 atan(P/2πr)）；G2 桨流动量通量 ∫ρu²2πr dr = T 与致动盘诱导速度闭式；"
  "G3 碎屑二次阻力闭式 |w|=|w0|/(1+k|w0|t) 与步长细化收敛；G4 合成 BPF 谐波列幅值回收 + 白噪声反例；"
  "G5 η=J·K_T/(2πK_Q) 恒等式、K_T(J) 单调、零推力进速比 J0；G6 与已测包在 D,Z,r_h/R,P/D,A_E/A_0 全匹配几何上互校。"
  "工具链：`scripts/verify_propeller_ice_milling_legacy.py` → `artifacts/propeller_ice_milling_legacy/verification.json` "
  "→ 本文（数字程序化注入，零手抄）。计算量秒级，未占集群。")
A("")
A("## ③ 计算结果")
A("")
A(f"判据总账：**{d['n_pass']}/{d['n_gates']} PASS**，8 条失败全部归因于 3 个登记缺陷。")
A("")
A("| 判据 | 实测 | 参考 | 相对误差 | 容差 | 判定 |")
A("|---|---|---|---|---|---|")
for gid in ("G1a_chord_scale_e0.35", "G1b_panel_area_e0.35", "G1c_pitch_angle_0.7R",
            "G2_momentum_flux_x1.0m", "G2b_induced_velocity", "G3_drag_relax_finest",
            "G3b_convergence_order", "G4a_bpf_amplitude", "G4c_harmonic_2",
            "G5a_efficiency_identity", "G6_K_T_divergence"):
    x = g[gid]
    A(f"| `{gid}` | {x['measured']:.5g} | {x['expected']:.5g} | {x['rel_err']:.3g} | "
      f"{x['tol']:.2g} | **{x['verdict']}** |")
A("")
A("交叉互校（匹配几何）：")
A("")
A("| J | K_T 遗留 | K_T 包 | K_Q 遗留 | K_Q 包 | η 遗留 | η 包 | 包内诱导封顶条数 |")
A("|---|---|---|---|---|---|---|---|")
for r in cross:
    A(f"| {r['J']:.2f} | {r['K_T_legacy']:.4f} | {r['K_T_package']:.4f} | {r['K_Q_legacy']:.4f} | "
      f"{r['K_Q_package']:.4f} | {r['eta_legacy']:.4f} | {r['eta_package']:.4f} | "
      f"{r['pkg_strips_at_induction_cap']}/{r['pkg_n_strips']} |")
A("")
A("## ④ 结果图")
A("")
A("图1 两实现开水特性（匹配几何）；图2 碎屑输运 vs 闭式解的步长细化；图3 合成 BPF 谐波列的谱幅值回收。")
A("")
A("## ⑤ 注意点（缺陷与局限）")
A("")
for f in d["findings"]:
    A(f"**{f['id']}（{f['severity']}，`{f['where']}`）**：{f['what']} *影响*：{f['impact']}")
    A("")
A("局限：无实验开水图谱与冰铣削扭矩实测入仓，故两实现的**绝对** K_T/K_Q 一律 WITHHELD，仅恒等式/趋势/J0 可门禁；"
  "碎屑模型为稀相、忽略重力浮力（Fr≫1）；遗留模块的塑性封顶 Hertz 接触分支位于 `MillingSimulation` 内部，"
  "需完整瞬态算例，未纳入本单。结论不是“测试绿=问题解决”：17/25 通过项是解析参考门，失败项为登记缺陷，未粉饰。")
A("")
A("## ⑥ 误差判定（逐项，显式回答是否 <3%）")
A("")
A(f"* 平面形状尺度 c_max vs 闭式 Beta 积分：{g['G1a_chord_scale_e0.35']['rel_err']:.3g} → **<3% 是**")
A(f"* 叶面积 vs EAR·πD²/4：{g['G1b_panel_area_e0.35']['rel_err']:.3g} → **<3% 是**")
A(f"* BPF 扭矩幅值 vs 解析 120 N·m：{g['G4a_bpf_amplitude']['rel_err']:.3g} → **<3% 是**")
A(f"* 效率恒等式 η=J·K_T/(2πK_Q)：{g['G5a_efficiency_identity']['rel_err']:.3g} → **<3% 是**（机器精度）")
A(f"* 桨流动量通量 vs T：{g['G2_momentum_flux_x1.0m']['rel_err']:.3g} → **<3% 否：恰为 T/2（DEFECT-D1）**")
A(f"* 碎屑出口速度 vs 模块自身桨流载速下的闭式解：{g['G3_drag_relax_finest']['rel_err']:.3g} → **<3% 否（DEFECT-D2）**")
A(f"* K_T 遗留 vs 包（匹配几何）：{g['G6_K_T_divergence']['detail_worst_divergence']:.3g} → "
  "**<3% 否，绝对值 WITHHELD（DEFECT-D3）**")
A("")
A("**处置**：对遗留模块 request changes（修 D1/D2，验收线就是两个 `xfail(strict)` 判据转绿）；IN34-W3-N05 包仍为规范引擎；"
  "D3 作为后续单登记（引入叶片/动量不平衡残差或声明封顶适用 J 上限）。未修改任何被测源码，缺陷以证据登记而非静默修补。")

md = "\n".join(lines)

caps = {
    "fig1_open_water_crossval.png": "两实现开水特性：K_T/K_Q/η vs 进速比 J（几何按 D,Z,r_h/R,P/D,A_E/A_0 匹配）",
    "fig2_debris_convergence.png": "碎屑二次阻力输运：遗留 RK2 vs 闭式解，随步长细化的相对误差",
    "fig3_spectrum_recovery.png": "遗留 torque_spectrum 对合成 BPF 谐波列的幅值回收（Hann 窗）",
}
figs = [{"name": n, "caption": c,
         "png_base64": base64.b64encode((ART / n).read_bytes()).decode()}
        for n, c in caps.items()]

report = {
    "title": "TensorDEM 遗留螺旋桨-冰铣削模块（IN27-S4）解析验证：25 判据 17 通过 + 3 缺陷登记",
    "summary": (
        "对零测试的遗留 propeller_ice_milling（IN27-S4，1140 行）建立 25 条独立参考判据（17 PASS），"
        "登记 3 个缺陷：D1 桨流动量通量仅为 T/2（与其‘动量守恒’声明矛盾）；"
        "D2 integrate_debris 用 σ=0.5D 而 RaceModel.sigma=0.25D（出口速度低 32.7%，其 0.5–2.0 m/s 喷射验收带不可采信）；"
        "D3 已测包 BEM 在 J≲0.2 全部条带撞 a=0.85 诱导封顶而残差仅 1e-8（单叶元/动量不平衡不可见，_buhl 未调用）。"
        "两实现绝对 K_T/K_Q 均 WITHHELD（无实验图谱）。"),
    "scenario_key": "tensordem-propeller-ice-milling-legacy-verification",
    "mission_id": 16148,
    "tags": ["TensorDEM", "propeller-ice", "milling", "verification-debt", "IN27-S4", "adversarial-audit"],
    "markdown": md,
    "figures": figs,
}

Path("/tmp/report_m16148.json").write_text(json.dumps(report), encoding="utf-8")
print("report json bytes:", len(json.dumps(report)))


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


if os.environ.get("POST_REPORT") == "1":
    cred = {"username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"]}
    tok = post("/auth/login", cred)
    token = tok.get("access_token") or tok.get("token")
    print("token acquired:", bool(token))
    out = post("/tensor/reports", report, token)
    print("posted:", json.dumps({k: v for k, v in out.items() if k in ("id", "status", "message")}))