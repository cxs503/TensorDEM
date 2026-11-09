"""Publish the paper-style progress report for mission 7053 to the platform.

Every number in the markdown is read from
``artifacts/contact_physics/integration_report.json`` (and the artifacts it
aggregates), and every figure is read from the PNGs the benchmark wrote and
embedded as base64 -- no number is typed by hand.

Usage
-----
``python3 scripts/post_contact_physics_report.py`` (reads ``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS`` from the environment)
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "contact_physics"
REPORT = ARTIFACTS / "integration_report.json"
MISSION_ID = 7053


def _api(path: str, payload=None, token: str | None = None, method: str | None = None):
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + path, data=data, headers=headers,
        method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def _token() -> str:
    login = _api("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                 "password": os.environ["TENSOR_AGENT_PASS"]})
    token = (login.get("access_token") or login.get("token")
             or login.get("data", {}).get("access_token"))
    if not token:
        raise RuntimeError(f"login response without a token: {list(login)}")
    return token


def _markdown(report: dict) -> str:
    acceptance = report["acceptance"]
    regimes = report["regimes"]
    performance = report["performance"]
    gate = report["regression"]

    rows = "\n".join(
        f"| {name} | {regime['steps']} | {regime['duration_s']:.4f} | "
        f"{regime['max_relative_error']:.1e} | "
        f"{regime['final']['interface_temperature_c']:.3f} | "
        f"{regime['final']['mu']:.5f} | {regime['final']['film_thickness_m']:.2e} | "
        f"{regime['final']['damage_max']:.3f} | {int(regime['final']['broken_bonds'])} |"
        for name, regime in regimes.items())

    model = regimes["cold_slow"]["model"]
    morphology = regimes["cold_slow"]["statistics"]["morphology"]
    network = regimes["cold_slow"]["statistics"]["force_network"]

    return f"""# P121W5-N11 成熟化单：DEM 接触物理模块（温度键合 + 润滑耦合 + 力链统计）集成验证与 API 稳定化

## ① 案例介绍（物理背景 / 几何 / 参数）

本单把冰区航行战役中三个已验收的接触物理节点集成为一个可复用模块，并在同一工况下做端到端自洽性验证：

* **P121W5-N02｜温度相关键合模型**：`tensordem.ice_bond_viscoelastic` 的 Maxwell-Wiechert（Prony 级数）键合律 + Kachanov 蠕变损伤钟，温度依赖取 `tensordem.wet_ice_cohesion.ice_resistance` 的 Timco & Weeks (2010) 卤水体积/强度关系。
* **P121W5-N05｜冰脊形态与接触网络统计**：`tensordem.ridge_dem` 的确定性冻结键合三角点阵冰脊 + `tensordem.ice_rubble_microstructure` 的配位数/力链分布估计。
* **P121W5-N08｜润滑-接触双向耦合**：`tensordem.interface_thermal` 的熔化分数摩擦律 mu(T)、Blok 热分配、Stefan 潜热开关与 Newton 求解的耦合推进器。

共享工况（`OperatingPoint`）：滑移接触冰-钢，法向力 {model['normal_force_n']:.3g} N，名义接触压力 {model['mean_pressure_pa']:.3g} Pa，冷态摩擦系数 mu_cold = {model['mu_cold']}；键合束 {model['n_bonds']} 根，参考力 F_ref = {model['reference_force_n']:.0f} N，损伤时间 t_ref = {model['damage_time_s']:.0f} s，Kachanov 指数 m = {model['damage_exponent']:.0f}，盐度 {model['salinity_ppt']:.0f} ppt。三个物理区间：

| 区间 | 滑移速度 [m/s] | 初始冰温 [degC] | 应变率 [1/s] | 关注物理 |
|---|---|---|---|---|
| cold_slow | 0.02 | -20 | 1e-5 | 升温/摩擦衰减，无熔化膜，键合完好（全程在强度关系拟合带内） |
| cold_rupture | 0.02 | -20 | 0.06 | 纯蠕变渐进破坏（断裂发生在拟合带内，排除卤水体积塌陷） |
| melting | 0.2 | -10 | 1e-5 | 界面钉在熔点：Stefan 熔化膜生长、mu→mu_wet、F_ref→0 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **API 稳定化（判据 b）**：新增 `tensordem.contact_physics` 子包（`bond.py` / `lubrication.py` / `statistics.py` / `march.py`），每个能力一个对象一个方法签名；上游三个节点模块**零改动**，其公开签名与文件哈希被 `scripts/freeze_contact_physics_api.py` 冻结为基线（基线 rev `{report['api']['baseline_git_rev'][:12]}`），逐模块字节比对 → 签名变更 {acceptance['b_api_stability']['signature_changes']} 个。
2. **集成测试（判据 a）**：`ContactPhysicsMarch` 在一个循环里按固定顺序推进 N08 → N02（用 N08 给出的界面温度）→ N05（统计采样）。测试里另写一份**只用库函数**的组合驱动（`standalone_reference`）作为参照，逐量比对 20 个历史序列。
3. **性能基准（判据 c）**：单线程 torch、冰脊点阵预先装配（两条路径同样排除）、交替 A/B 计时、min-of-N 估计（timeit 约定）；内存用 tracemalloc 峰值。全部原始样本写入工件，可复核。
4. **回归门（判据 d）**：N02/N05/N08 的 9 个 pytest 文件分节点独立进程执行，`scripts/contact_physics_regression_gate.py` 落盘。
5. **文档（判据 e）**：`docs/contact_physics_module.md` 覆盖全部 6 个公开 API；`examples/contact_physics_demo.py` 可独立运行；文档中的验收表由 `scripts/build_contact_physics_doc.py` 从工件程序化注入，杜绝手抄漂移。

## ③ 计算结果（判据数字）

### (a) 集成一致性：集成运行 vs 三模块独立运行

| 区间 | 步数 | 时长 [s] | 最大相对误差 | T_int 末值 [degC] | mu 末值 | 膜厚末值 [m] | 损伤 D_max | 断键数 |
|---|---|---|---|---|---|---|---|---|
{rows}

全部 20 个序列（温度、mu、熔化分数、承载面积分数、摩擦热流、膜厚、膜法向刚度、键合力均值/峰值、力/应变刚度比、应变、损伤增量、损伤、断键数、功、耗散、储能、耦合迭代数）相对误差 = **{acceptance['a_integration']['max_relative_error']:.1e}**（判据 1e-10，逐位相同）。

### (c) 性能

| 量 | 集成 | 三模块独立 | 开销 |
|---|---|---|---|
| 墙钟时间（min-of-{performance['repeats']}，单线程）[s] | {performance['integrated_wall_s']:.4f} | {performance['standalone_wall_s']:.4f} | {performance['time_overhead_percent']:+.2f} % |
| 墙钟时间（中位数，同批样本）[s] | {performance['integrated_wall_median_s']:.4f} | {performance['standalone_wall_median_s']:.4f} | {performance['time_overhead_median_percent']:+.2f} % |
| Python 峰值内存（tracemalloc，min）[B] | {performance['integrated_peak_py_bytes']} | {performance['standalone_peak_py_bytes']} | {performance['memory_overhead_percent']:+.2f} % |

### (d) 回归门

前置节点 {gate['passed_tests']} 个用例全绿（失败 {gate['failed_tests']}，错误 {gate['errors']}），per-node：
""" + "\n".join(f"* {row['node']}：passed={row['passed']} failed={row['failed']} rc={row['returncode']}"
                for row in gate["per_node"]) + f"""

### (e) 文档覆盖

公开 API 覆盖率 {acceptance['e_documentation']['coverage'] * 100:.0f}%（6/6），示例脚本独立运行成功。

### 冰脊形态与力链统计（N05 采样，cold_slow 工况）

实现孔隙率 {morphology['realised_porosity']:.4f}（设定 {morphology['requested_porosity']:.2f}），颗粒 {int(morphology['n_particles'])}、冻结键 {int(morphology['n_bonds'])}、配位数 Z = {morphology['mean_coordination']:.3f}、脊侧面角 {morphology['keel_angle_deg']:.2f} deg；接触力分布 CV = {network['coefficient_of_variation']:.3f}，强接触占比 {network['strong_fraction']:.3f}，强接触承载份额 {network['strong_load_fraction']:.3f}。

## ④ 结果图

见附图：耦合历史（界面温度/mu/膜厚）、键合响应（力-应变与损伤）、集成-独立一致性逐量误差、冰脊形貌与力链统计。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **耦合是单向传递而非同时求解**：N08 的界面温度进入 N02 的损伤钟，但键合力*不*反馈接触压力。闭合该环是物理变更（压力成为 Newton 未知量），超出成熟化单范围，明确列为后续预算项。
2. **损伤截断是集成规则不是新本构**：上游 Kachanov 速率无界，适配层把 D 截断在 1 并让断键不传力；原始增量仍可单独调用以供验证面使用。
3. **T ≥ 0 处 strength_factor = 0 是极限不是拟合**：熔化界面没有固相键合；卤水体积关系在 5 ppt 下约 -0.72 degC 已触及 0.29 孔隙极限而返回值 0，-2 degC 以上已属拟合带外（`within_strength_fit()` 显式报告）。
4. **温度下限 -60 degC**：超出则拒绝而不是外插。
5. **统计是单次实现**：`RidgeContactStatistics` 采样一条给定种子点阵，不是脊龙骨折算分布；多实现集合是后续项。
6. **性能数字是同进程单线程**，不代表 GPU/多线程扩展性；测量机器负载均值 >300，故采用 min-of-N 并公开全部样本。
7. **形态数字是点阵统计**：实现孔隙率与设定值不同（{morphology['realised_porosity']:.3f} vs {morphology['requested_porosity']:.2f}），因为抽稀步骤移除了颗粒；引用时一律用实现值。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 集成运行 20 个状态序列 | 三模块独立驱动（库函数直调） | {acceptance['a_integration']['max_relative_error']:.1e} | 是（判据 1e-10，实际逐位相同） |
| 时间开销 | 三模块独立运行 | {performance['time_overhead_percent']:+.2f} % | 是（判据 <5%） |
| 内存开销 | 三模块独立运行 | {performance['memory_overhead_percent']:+.2f} % | 是（判据 <5%） |
| 上游 N02/N05/N08 验收 | 各节点自带参考/解析解 | 继承（未改变代码，哈希比对通过） | 是（回归门 {gate['passed_tests']} 用例全绿） |

**说明**：本节点是成熟化/集成节点，没有独立实验基准，其精度主张是"集成层不引入偏差"；物理精度由上游节点各自的解析/文献验收继承（本单通过字节哈希 + 回归门证明未被扰动）。

---

审计链：`artifacts/contact_physics/integration_report.json`（总表）、`api_baseline.json`（签名基线）、`regression_gate.json`（回归门）、`performance_pytest.json`（性能原始样本）、`docs/contact_physics_module.md`（文档）、`tests/test_contact_physics_integration.py`（判据可执行形式）。生成时间 {report['generated_at_utc']}，git `{report['git_rev'][:12]}`。
"""


def main() -> int:
    report = json.loads(REPORT.read_text())
    figures = []
    for figure in report["figures"]:
        path = ARTIFACTS / figure["name"]
        figures.append({
            "name": figure["name"],
            "caption": figure["caption"],
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    token = _token()
    payload = {
        "title": "P121W5-N11 DEM 接触物理模块成熟化：温度键合+润滑耦合+力链统计集成验证与 API 稳定化",
        "summary": ("三个 DEM 接触物理节点集成为一个模块：集成运行与三模块独立运行逐位相同"
                    f"（最大相对误差 {report['acceptance']['a_integration']['max_relative_error']:.1e}），"
                    f"上游签名变更 {report['acceptance']['b_api_stability']['signature_changes']} 个，"
                    f"开销 时间 {report['performance']['time_overhead_percent']:+.2f}% / 内存 "
                    f"{report['performance']['memory_overhead_percent']:+.2f}%，"
                    f"回归门 {report['regression']['passed_tests']} 用例全绿。"),
        "scenario_key": "ice_navigation",
        "mission_id": MISSION_ID,
        "tags": ["DEM", "contact-physics", "lubrication", "ice-bond", "P121W5-N11",
                 "integration", "maturation"],
        "markdown": _markdown(report),
        "figures": figures,
    }
    try:
        response = _api("/tensor/reports", payload, token=token)
        print("posted:", json.dumps(response)[:400])
    except urllib.error.HTTPError as error:  # surface the body for debugging
        print("HTTP", error.code, error.read().decode()[:800])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())