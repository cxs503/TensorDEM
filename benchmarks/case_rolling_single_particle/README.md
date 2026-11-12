# CASE-DEM-ROLLING-SINGLE-PARTICLE

算例池条目：单颗实心球在 15° 斜面上的**无滑移纯滚动**，对标经典刚体动力学
`a = g sin(theta)/(1 + I/(m R^2))`、`alpha = a/R`（实心球 `= (5/7) g sin(theta)`），
容差 2 %。次级参考：Ai et al. 2011 (Powder Technol. 206:269) 滚动阻力模型的
`a = g (sin t - mu_r cos t)/(1 + I/(mR^2))` 与临界角 `arctan(mu_r)`。

复用能力（不另起炉灶）：`tensordem.rolling_resistance.simulate_sphere_on_incline`。

| 文件 | 作用 |
|---|---|
| `run_case.py` | 算例驱动：基准点 + 负控 + Δt 收敛 + k_n 敏感性 + 窗长敏感性 → `artifacts/.../result.json` |
| `figures.py` | 四联图（§ README 顶部报告的图注见 docs） |
| `build_report.py` | 从 result.json **程序化**生成 `docs/CASE-DEM-ROLLING-SINGLE-PARTICLE.md` 与平台报告 payload |
| `../../prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json` | 预注册（判据/负控/停止规则先于生产运行） |
| `../../tests/test_case_rolling_single_particle.py` | CI 守卫（6 项，含两个负控） |

一键复现：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python benchmarks/case_rolling_single_particle/run_case.py \
    --out artifacts/case_rolling_single_particle/result.json
PYTHONPATH=src python benchmarks/case_rolling_single_particle/figures.py
PYTHONPATH=src python benchmarks/case_rolling_single_particle/build_report.py
PYTHONPATH=src python -m pytest tests/test_case_rolling_single_particle.py -q
```

单核本地总耗时 < 1 分钟（最贵点 Δt=5e-6 用时 ~15 s），无需集群。
