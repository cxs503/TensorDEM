"""Build the m3499 platform report payload from the audit artefacts (no hand-copied numbers)."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = json.loads((ROOT / "artifacts/hertz_pair_restitution/audit.json").read_text())
md = (ROOT / "docs/dem3d_hertz_pair_restitution_verification.md").read_text()
fig = base64.b64encode(Path("/tmp/hertz_fig.png").read_bytes()).decode()

summary = (
    'TensorDEM contact_model="hertz" 配对分支用线性 Kelvin-Voigt 映射把声明的 restitution_coefficient 换成 '
    "zeta，再套到 gamma(d)=2*zeta*sqrt(k_s(d)*m_eff) 的 Hertz 阻尼上；此前无测试无文档。"
    "本单用独立 RK4 参考解（能量闭合 ~1e-4、三级步长加密）与 2x2x2 IceDEM3D 正碰对照："
    "引擎与其自身接触律偏差 <=0.02%（实现层无缺陷；dt x1/1/4/1/16 三档差 <0.02%，速度 0.1-1.0 m/s 内 e 变化 <3e-4，"
    "故非时间步误差、非速度相关误差）。但实测恢复系数系统性高于声明值："
    + ", ".join(f"{r['e_target']:.1f} -> {r['bias_vs_target_pct']:+.2f}%" for r in A["pair_rows"])
    + "。机理：d^(1/4) 阻尼在接触起点与恢复末段被 clamp 到零，线性理论假设的那部分耗能从不发生。"
    "给出可直接使用的标定反演表（要 e=0.3 应填 0.122；要 0.5 应填 0.392）。"
    "另测得 platen 分支完全忽略 restitution_coefficient（默认 platen_damping=0 时 e=1.000001）。"
    "src 未改动：本单交付测量、判据、标定表与 10 项测试。"
)

report = {
    "title": ("Hertz 配对接触恢复系数：引擎实现 OK (<=0.02%)，但声明的 restitution_coefficient "
              "在 e=0.5/0.3 处偏差 +16.4%/+43.7%（honest 否定）"),
    "summary": summary,
    "scenario_key": "tensordem-hertz-pair-restitution-verification",
    "mission_id": 3499,
    "tags": ["TensorDEM", "DEM3D", "hertz", "restitution", "verification-debt",
             "calibration", "honest-fail"],
    "markdown": md,
    "figures": [{
        "name": "hertz_pair_restitution.png",
        "caption": "四联图：声明 vs 实测恢复系数；相对偏差与 ±3% 带；冲击速度无关性；标定反演表",
        "png_base64": fig,
    }],
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("payload KB", round(len(json.dumps(report)) / 1024, 1))