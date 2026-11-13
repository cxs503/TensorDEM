#!/usr/bin/env python3
"""Build /tmp/report.json for mission m17063 (RCP steady-state audit)."""
import base64
import json
import os

doc = open('docs/AUDIT_m17063_RCP_STEADY_STATE.md').read()
summ = json.load(open('benchmarks/case_rcp_deposition/runs/steady_state_audit.json'))
gates = summ['gates']
lines = ["", "## 附：闸门 G20–G26", ""]
for k, v in gates.items():
    lines.append("- `%s`: %s" % (k, "PASS" if v['pass'] else "FAIL"))
md = doc + "\n".join(lines) + "\n"

figs = []
for name, cap in [
        ("steady_state_plateau", "phi(t) of the six production runs with plateau onset"),
        ("steady_state_residual", "unbalanced force / mean contact force vs Coulomb bound"),
        ("steady_state_friction", "bulk z versus mu at fixed size: approaches zc=d+1=4")]:
    path = 'benchmarks/case_rcp_deposition/figures/%s.png' % name
    figs.append({"name": name + ".png", "caption": cap,
                 "png_base64": base64.b64encode(open(path, 'rb').read()).decode()})

payload = {
    "title": "TensorDEM m17063 · RCP 沉积堆积的「稳态」与等静性核验（工件级复核，零新仿真）",
    "summary": ("用已落盘的 10000 球沉积末态独立复核验收词「稳态孔隙率」：phi(t) 平台长度 7.0-11.0 个"
                "时间单位、峰峰值 <=1.9e-3（G20）；无摩擦档静止力平衡残差约 1% 平均接触力（G21）；"
                "有摩擦档残差落在 Coulomb 上界内（G22）；配位数 z 与 Maxwell 等静值 6/4 对照并给出"
                "接触阈值带 ±0.12-0.16（G23/G24/G26）；刚度梯级的 z 不收敛（G25 honest FAIL）。"),
    "scenario_key": "tensordem/rcp_deposition_steady_state",
    "mission_id": 17063,
    "tags": ["TensorDEM", "RCP", "packing", "steady-state", "isostaticity",
             "verification-debt"],
    "markdown": md,
    "figures": figs,
}
with open('/tmp/report.json', 'w') as fh:
    json.dump(payload, fh)
print('payload bytes', os.path.getsize('/tmp/report.json'), 'gates', len(gates))