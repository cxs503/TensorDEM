import json, os
ART="/root/tp-agent/work/m11157/artifacts/dem_auto_calibration"
recs=[json.load(open(os.path.join(ART,f"{m}.json"))) for m in ("glass_beads","sand","gravel")]
doc=open("/root/tp-agent/work/m11157/docs/DEM_AUTO_CALIBRATION.md").read()
head="""## 摘要

本节点交付 **DEM 接触参数自动标定反演闭环**（`tensordem.dem_calibration`）：正演为雨淋堆积
DEM 试验（240 球/3 类材料，接触律复用已验证的 `granular_column` 核），宏观响应取休止角 θ_r 与
堆积固体分数；反演为 Matern-5/2 高斯过程代理 + 期望改进贝叶斯优化（18 次正演/材料），
后验用随机游走 Metropolis（代理误差入方差），并在**换种子**与**换工况**两条独立路径上验收。

**结果（honest FAIL）**：判据 (d) 不确定性、(e) 可复现性通过；(a) 精度、(b) 收敛、(c) 泛化未通过，
且原因已定位并量化：本系统规模下的休止角响应尚不具备物理一致性
（玻璃珠 4.09°、砾石 18.11°，而其 μ_pp 分别为 0.34/0.68，量级与文献带 20–30°/35–42° 不符），
说明该响应被恢复系数主导的弹跳/铺展而非摩擦主导——三类材料的 θ 排序因此与 μ 排序脱钩。
报告同时记录了两条被否定的释放类协议（退壁提升、过陡锥松弛）的实测数据与失败机理。

**pytest**：`tests/test_dem_calibration.py` 22 passed / 3 failed——
失败的 3 项正是上述生产验收判据（`test_calibration_accuracy`、`test_calibration_convergence`、
`test_generalization`），未被覆盖或粉饰。

"""
body={"title":"DEM 接触参数自动标定反演：GP-贝叶斯优化 + 后验不确定性（休止角锚定）——诚实 FAIL 与已量化的物理缺口",
 "summary":"TensorDEM 新增 dem_calibration 包：雨淋堆积 DEM 正演 + GP/EI 贝叶斯优化 + MH 后验。三类材料各 18 次正演；可复现性与后验覆盖率通过，精度/收敛/泛化诚实 FAIL（休止角响应被恢复系数主导，与 μ 排序脱钩）。",
 "markdown": head + doc}
json.dump(body, open(os.path.join(ART,"report_body.json"),"w"), ensure_ascii=False)
print("ok", len(body["markdown"]))
