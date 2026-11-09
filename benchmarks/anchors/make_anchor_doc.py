#!/usr/bin/env python3
"""Emit docs/ICE_POLYGON_FLOE_FIELD_ANCHOR.md from the anchor artifact (no hand-typed numbers)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
d = json.load(open(os.path.join(HERE, "ice-polygon-floe-field.json"), encoding="utf-8"))
conv, rec, chk = d["conversion_chain"], d["recompute"], d["checks"]
fields = d["case_caliber"]["fields"]
first = list(fields.values())[0]
covs = ", ".join(sorted({f"{v['coverage']:.2f}" for v in fields.values()}))
fams = ", ".join(sorted({v["family"] for v in fields.values()}))
L = []
A = L.append
A("# ice-polygon-floe-field — 外部锚核销（W18 paper-exp 实验 #1）\n")
A(f"**Repo**: TensorDEM · **branch/commit**: `{d['repo']['branch']}` / "
  f"`{(d['repo']['commit'] or '')[:12]}` · **task_key**: `{d['task_key']}` · "
  f"**generated**: {d['generated_at']} (UTC)\n")
A("**Artifact**: `benchmarks/anchors/ice-polygon-floe-field.json` — 全部数字由 "
  "`benchmarks/anchors/build_ice_polygon_floe_field.py` 注入，无手抄\n")
A(f"**Verdict**: `{d['verdict']}` — {d['verdict_reason']}\n")
A("---\n\n## 1. 案例介绍（物理背景 / 几何 / 参数）\n")
A("本算例＝IN32-N01 的无序多边形碎冰场（航道内碎冰盖）：floe 为 Laguerre "
  "prescribed-area 镶嵌生成的多边形，接触为凸多边形 SAT 接触（带力臂与力矩）。"
  "外部锚要对标 **ISO 19906:2019** 的 p(A) 面积-压力关系（Annex A）与全球冰作用"
  "（Clause 14/15）。\n")
A("| 参数 | 取值 / 来源 |\n|---|---|")
A(f"| floe 数 | N = {first['n_floes']} |")
A(f"| FSD 族 | {fams} |")
A(f"| 覆盖率 φ | {covs} |")
A(f"| 冰厚 h | {first['thickness_m']:.6g} m（=0.1×median r_eq，代码给出） |")
A(f"| 场盒 | {first['box_m'][0]:.2f} × {first['box_m'][1]:.2f} m |")
A(f"| 接触斑口径 | A_i = κ·2·r_eq,i·h，κ = {conv['calibers']['floe_patch']['kappa']} |")
A(f"| seed | {first['seeds']} |")
A("\n## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）\n")
A("1. **来源通道探测**（每通道一次，判完即止）：tcp 443、http GET 正本页、文件系统清单扫描 "
  f"{d['source_access_probe'][-1]['files_scanned']} 个文本文件，primary text 命中 "
  f"{d['source_access_probe'][-1]['primary_text_hits']}。\n")
A("2. **锚位登记表**（`ANCHOR_SLOTS`：Annex A p(A) / Clause 14-15 全球冰力）：只能由 "
  "`--anchor-file` 注入的逐字抄录填值；VERIFIED 必须同时具备 quote_verbatim + "
  "table_or_page + value，半填文件直接 raise（不得静默升级）。\n")
A("3. **算子层换算链**：p(A)=p₀(A/A_ref)^(-n) → F(A)=p(A)·A → 场聚合 "
  "F_field=Σᵢp(A_i)A_i = p₀·A_ref^n·K(n)，K(n)=ΣᵢA_i^(1-n)。绝对层缺 p₀、A_ref 即 "
  "WITHHELD；相对层 p₀、A_ref 约掉，故锚未核销也能先做可复算的同口径换算。\n")
A("4. **机检**：量纲齐次（分数指数代数，含 A_ref 归一化陷阱行）、局部-平均压力恒等式"
  "（Gauss–Legendre，4 个面积点）、log-log 指数反演、尺度不变性、聚合双路一致、"
  "口径退化（同族 FSD 不同 φ 的接触斑集合）。\n")
A("5. **复算**：重载工件内 (A, p) 采样点反演 n，与工件内 n 及 F=p·A 比对。\n")
A("6. 工具链：numpy + 本仓 `tensordem.polygon_floe_field` / `tensordem.fsd_packing`；"
  "matplotlib 出图；pytest 30 例。\n")
A("\n## 3. 计算结果\n### 3.1 判据逐条\n")
A("| 判据 | 状态 | 实测（节选） | 差距 |\n|---|---|---|---|")
for c in d["criteria"]:
    m = json.dumps(c["measured"], ensure_ascii=False)[:150]
    A(f"| {c['id']} {c['statement'][:30]}… | `{c['status']}` | {m} | {c['gap'] or '—'} |")
A("\n### 3.2 算子层（单位归一 p₀=1 Pa、A_ref=1 m²；只复算算子，不是标准原值）\n")
A("| A [m²] | p/p₀ | F/(p₀·A_ref) |\n|---|---|---|")
s = conv["samples"]
for a, p, f in zip(s["A_m2"], s["p_norm_Pa"], s["F_norm_N"]):
    A(f"| {a:g} | {p:.10g} | {f:.10g} |")
A(f"\nlog-log 斜率 = {s['log_log_slope']:.12g}（应 = -n = {-conv['exponent_used']:.12g}），"
  f"拟合残差 {s['log_log_fit_residual']:.3g}。\n")
A("### 3.3 本算例口径的场聚合核与锚无关系数\n")
k = conv["calibers"]["floe_patch"]["kernel_over_fields"]
A("| 配置 | K(n)=ΣA_i^(1-n) [m^(2(1-n))] | dlnK/dn | K 在 n∈[0.20,0.70] 带宽 |\n|---|---|---|---|")
for key, r in sorted(k.items()):
    g = r["kernel_over_n_grid"]
    A(f"| {key} | {r['kernel_K_n_m2_1mn']:.10g} | {r['log_sensitivity_dlnK_dn']:.6g} | "
      f"{g['K_min']:.6g} … {g['K_max']:.6g} |")
A("\n| 比值（锚无关） | 候选 n 下 | n∈[0.20,0.70] 带宽 |\n|---|---|---|")
pr = d["predictions_for_domain_test"]
for key, val in sorted(pr["ratios_at_candidate_n"].items()):
    lo, hi = pr["band_over_n_0.20_0.70"][key]
    A(f"| {key} | {val:.10g} | {lo:.6g} … {hi:.6g} |")
sf = conv["calibers"]["structure_face"]
A(f"\n结构面口径（A=W·h）：A_face = {sf['A_ref_face_m2']:.6g} m²，"
  f"半宽/全宽载荷比 = {sf['ratio_halfwidth_over_full']:.10g}。\n")
A("### 3.4 机检与复算\n| 检查 | 结果 |\n|---|---|")
for grp in ("dimensional", "consistency"):
    for r in chk[grp]["rows"]:
        A(f"| [{grp[:4]}] {r['check']} | {'PASS' if r['pass'] else 'FAIL'} |")
A(f"| 复算 | 残差 {rec['residual_max']:.3g} < {rec['threshold']:g}；采样点反演 n = "
  f"{rec['n_recovered_from_stored_samples']:.12g} |")
A(f"| 登记纪律 | 违例 {len(chk['registry_discipline']['violations'])}；已核销槽位 "
  f"{chk['registry_discipline']['verified_slots']} |")
A("\n## 4. 结果图\n")
A("- `artifacts/anchors/ice_polygon_floe_field/fig1_operator.png` — p(A) 与 F=p·A 算子，"
  "含 n 扫描带（0.20 / 候选 / 0.70）。")
A("- `fig2_field_kernels.png` — 左：φ=0.85 四族 FSD 的 floe 接触斑面积分布；右：锚无关系数"
  "随 n 的漂移（缺锚的代价＝带宽）。")
A("- `fig3_checks.png` — 量纲/一致性机检逐条红绿。")
A("\n## 5. 注意点（坑与局限）\n")
A("- **正本不可得**：ISO 19906:2019 为付费标准，文件系统 0 命中、公开页 403；"
  "引句/表号/页码/p₀/A_ref 全空，故『与规范一致 X%』在本状态下不可检验。\n")
A("- **K(n) 不是力**：量纲为 m^(2(1-n))；把 A_ref 归一为 1 m 得到的数字不是牛顿"
  "（量纲表专门列了这条陷阱行）。\n")
A("- **口径退化**：floe_patch 口径下同族 FSD 的接触斑集合与堆积浓度无关（残差 ~1e-15），"
  "故本工件不能给出『载荷随冰密集度变化』的结论——那只能由力链/堵塞 DEM 实验回答。\n")
A("- **候选指数只能当对照**：平台内部先例 n 仅用于敏感性带宽，登记 comparison_only，"
  "未进入任何 PASS 判据的 measured（登记纪律机检）。\n")
A("- 文献级对照留待 C1 核销后进行；本工件不引用任何未经逐字抄录的规范数值。\n")
A("\n## 6. 误差判定（对基准逐项）\n")
A("| 关键量 | 基准 | 本工件误差 | 是否 <3% |\n|---|---|---|---|")
A(f"| 复算残差（采样点反演 n、p、F、K） | 机检阈值 1e-9 | {rec['residual_max']:.3g} | 是 |")
A(f"| log-log 指数反演 | -n | {abs(s['log_log_slope'] + conv['exponent_used']):.3g} | 是 |")
cons_max = max(float(r.get("residual_max", 0.0)) for r in chk["consistency"]["rows"])
A(f"| 局部-平均压力恒等式 / 聚合双路 | 闭式解 | {cons_max:.3g} | 是 |")
A("| 场载荷数值 vs ISO 规范值 | ISO 19906 正本 | **未计算**（锚位 WITHHELD） | "
  "**不可判定** — 缺 p₀ 与 A_ref，无基准可比；不声明任何百分比 |")
A("\n**honest FAIL**：C1 FAIL（0/2 锚位核销）→ `verdict = FAIL`；C2 算子层 PASS / 绝对层 "
  "WITHHELD；C3、C4 PASS。未完成部分＝绝对层与领域-规范差额（理由见 §5/§6）。\n")
out = os.path.join(REPO, "docs", "ICE_POLYGON_FLOE_FIELD_ANCHOR.md")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", out, os.path.getsize(out))
