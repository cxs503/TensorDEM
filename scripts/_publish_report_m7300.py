"""Publish the P70-N03 report to the platform (m7300)."""
import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]


def post(path, body, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req) as r:
        return r.status, json.load(r)


_, d = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                            "password": os.environ["TENSOR_AGENT_PASS"]})
tok = d.get("access_token") or d.get("token")


def fig(name, caption):
    with open(os.path.join("artifacts/p70n03", name), "rb") as fh:
        return {"name": name, "caption": caption,
                "png_base64": base64.b64encode(fh.read()).decode()}


MD = r"""
# P70-N03 二元颗粒床分凝动力学验证面（TensorDEM）— 第 2 次尝试

**结论先行（honest FAIL）**：内核与稳性传导映射已建成并通过单元测试，
但**物理验证未通过**——①请求规模（2×10⁴–1×10⁵ 颗粒）下内核在沉降阶段产生非有限壁面接触力
（已定位到壁面切向状态，未修完）；②可用规模（4×10³）下粗颗粒不是上浮而是下沉
（S<0），且 S 对 Γ 完全无依赖（驱动振幅 A/d≈0.1 太小、初态偏析被冻结）。
判据 (a)–(f) 均**不主张满足**。以下按六要素如实记录。

## ① 案例介绍

二元球床（尺寸比 d_l/d_s = 1/1.5/2/3/4，粗颗粒体积分数 0.2/0.4，ρ=2500 kg/m³，
μ=0.4、e=0.3、μ_r=0.02）在三维盒（0.10–0.15 m 方形截面）与准二维单层球板
（深度 1.15 d_l）中，由竖直振动地板驱动：Γ = Aω²/g ∈ {1, 1.5, 2, 2.5, 3}，
f ∈ {15, 25, 40, 60} Hz，30 个周期；另有顶部喷嘴冻结注入的重力倾倒建堆构型。
仓库原有内核 `granular3d_core` 是单分散的（`Material3D.radius` 单值），无法表达二元床，
因此本节点新增逐颗粒半径内核。

| 参数 | 振动战役 | 说明 |
|---|---|---|
| 粒径 d_s | 6–8 mm | 软球刚度 k_n = 900–3200 N/m，由 2 % 静载侵入量反推 |
| dt | 7.5×10⁻⁵ s | 0.06 t_c |
| 颗粒数 | 4×10³ 可用 / 2×10⁴ 失败 | 请求 2×10⁴–1×10⁵ |
| 周期数 | 30 | 1.2 s |

## ② 怎么做的

* **新内核** `tensordem.bidisperse_segregation`：法向线性弹簧-阻尼
  `F_n = max(k_n δ − γ v_n, 0)`、切向 Cundall–Strack 增量弹簧（饱和 μF_n）、
  速率受限滚动阻力、地板+四侧壁+顶盖摩擦接触；逐对切向/滚动状态用
  **排序键 + searchsorted** 向量化（不用 dict 循环）；邻域按半径分类用各自截止半径，
  候选对数量与真实接触数同量级；两种驱动（振动/喷嘴倾倒）。
* **可测量而非注入**的分凝指数：S = (顶层粗颗粒体积分数 − 名义值)/名义值，
  顶层 = 自由面下 20 % 床高；Brazil-nut 上升速度 v_up 取粗颗粒质心相对全床质心
  在末段 40 % 时间窗的最小二乘斜率。
* **稳性传导** `tensordem.segregation_stability`：分区质量矩 → M、KG_c、LCG_c、ρ_b、
  自由液面惯性 I_FS = A_FS B²/12、ΔGM = −ΔKG_c·M/Δ − λ ΔI_FS/Δ（λ=0.12，
  谷物规则 grain factor，CALIBRATED）、Δtrim = −ΔLCG_c/L_hold；每个闭式量都与
  逐颗粒独立求和/分片求和**对拍**（审计误差落 JSON）；底板压力按 x 分条给峰值、
  峰值/静压比与总载荷平衡。
* **工具链**：单元测试 `tests/test_bidisperse_segregation.py`（9 项，5.4 s）；
  战役驱动 `scripts/run_bidisperse_segregation.py`（59 例，`xargs -P 48` 撒 loong64 双节点）；
  判据与图 `scripts/analyze_bidisperse_segregation.py`；失稳诊断 4 个 `scripts/_diag_*_m7300.py`。

## ③ 计算结果（程序化注入，源文件 runs/p70n03/*.json）

尺寸比扫描（Γ=2, f=25 Hz, φ=0.2, box3d, N=4×10³, 30 周期）：

| d_l/d_s | seed1 | seed2 | seed3 | 均值 |
|---|---|---|---|---|
| 1.0（负控） | −0.0729 | −0.0196 | −0.0013 | −0.031 |
| 1.5 | −0.2725 | −0.3891 | −0.3290 | −0.330 |
| 2.0 | −0.5575 | −1.0000 | −0.6739 | −0.744 |
| 3.0 | +0.4706 | −0.3494 | +0.2084 | +0.110 |
| 4.0 | +0.3223 | +0.0573 | +0.0684 | +0.149 |

Γ 扫描（s=2, f=25 Hz）：S = −0.692, −0.618, −0.697, −0.618, −0.618（Γ=1,1.5,2,2.5,3）；
频率扫描（Γ=2, s=2）：S = −0.558(f=15), −0.558(25), −0.557(40), −0.558(60)——**同一 seed 下 4 位小数相同**。

传导量（示例 s=2, Γ=2, seed1）：ΔVCG_cargo / ΔGM 在多数算例为 NaN，原因是末态床体
出现非有限坐标（见 ⑤）；分区质量/质心审计在单元测试床上 < 1×10⁻¹²，
质量守恒相对误差 4.6×10⁻¹⁵。底板载荷平衡误差 0.69，峰值/静压比 1.03。

判据汇总：`artifacts/p70n03/acceptance.json`（`all_criteria_pass: false`）。

## ④ 结果图

图 1：S 对尺寸比（含 ±0.02 负控带）——S 随尺寸比**非单调**，s=2 强负（粗沉）。
图 2：S 对 Γ——**无依赖**，说明振动未改组床体（驱动器太弱）。

## ⑤ 注意点（踩坑与局限）

1. **规模失稳（阻断）**：N≥10⁴ 时沉降阶段出现非有限位置，非有限值优先出现在
   壁面接触力（`_wall_contacts` 的 f_n/f_t），此时 max_overlap=0、v_max≈0.26 m/s；
   与摩擦/滚动无关（μ=0 复现），与 dt 弱相关（dt/5 只把失稳步数从 200 推到 1500），
   与刚度强相关（k_n=800 稳定 2500 步，k_n=3206 约 200 步失稳），**同一配置不同节点
   失稳步数不同**（200/600/1200）→ 指向读取未初始化内存/内容相关分支，而非纯数值不稳定。
   最可疑处：`wall_tang` 的逐颗粒增量状态在"某墙无活动接触时整体清零"与逐墙 cap 的失配。
2. **驱动太弱**：Γ=2、25 Hz → A = 0.8 mm，A/d_s ≈ 0.1（文献/实验常用 0.5–2），
   床体基本无膨胀-渗流循环，S(t) 被沉降期初态冻结。
3. **初态偏析**：沉降阶段重的大颗粒先压实到底部（S(0)≈−0.6），主判据应改用
   ΔS = S(end) − S(0) 并联报绝对 S。
4. **统计噪声**：4×10³ 颗粒时顶层粗组仅 ~80 颗，σ(S)≈0.1，负控判据 (a) 需要
   ≥2×10⁴ 颗粒 + 时间/种子平均才有判别力——这使修好规模问题成为前置条件。
5. **软球 DEM**：k_n 由 2 % 侵入量反推，接触力峰值/声速类量不可用于定量结论；
   无量纲参数（Γ、A/d、尺寸比、体积分数）保持物理一致。
6. 文献锚（Γ_c≈1–2、v_up ∝ (d_l/d_s−1)^p 的 p≈1）本轮**未使用**：既无 Γ 依赖也无
   可用 v_up（NaN），报告不引用未核实数值。

## ⑥ 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 质量守恒（Σm 恒定） | 初态总质量 | 4.6×10⁻¹⁵（相对） | 是 |
| 分区质心 KG_c 闭式 vs 逐颗粒 | 独立直接求和 | <1×10⁻¹² | 是 |
| 分区 LCG_c 闭式 vs 逐颗粒 | 独立直接求和 | <1×10⁻¹² | 是 |
| 自由液面惯性 I_FS 闭式 | 解析水平面 A·B²/12 | <1×10⁻¹² | 是 |
| 界面对拍（单元测试） | — | 9/9 通过 | — |
| **分凝指数 S vs 文献标度** | 巴西坚果方向/Γ 依赖 | **方向相反、无 Γ 依赖** | **否（FAIL）** |
| **v_up ∝ (d_l/d_s−1)^p** | 文献 p≈1 | v_up 全部 NaN，无法比较 | **否（FAIL）** |
| **Γ_c ±30 %** | 文献 Γ_c≈1–2 | 无 Γ 依赖，无法拟合 | **否（FAIL）** |
| **ΔTCG/ΔGM 战役算例** | 分片求和 | 末态非有限 → NaN | **否（FAIL）** |

即：**数值账目类误差全部远小于 3 %；物理标度类判据全部未通过**，且失败原因是
可定位的实现/配置问题（壁面状态失稳 + 驱动太弱 + 初态偏析），不是对分凝物理的否证。

## 待办（下一轮）

1. 重写 `_wall_contacts` 的逐墙切向状态（不再整体清零），`np.seterr(all="raise")` 全开
   跑 N=2×10⁴ 沉降回归（目标 5×10³ 步无非有限值），在 `_diag_mode_m7300.py` 上二分；
2. 驱动重设：f = 10–15 Hz、A/d_s = 0.5–1（同 Γ）、30–120 周期；
3. 反偏析初始化 + 以 ΔS 为主判据、负控三 seed×时间平均评估；
4. 规模修好后用同一脚本重跑 `runs/p70n03`（`--list` 已给出全部算例名）。
"""


report = {
    "title": "P70-N03 二元颗粒床分凝动力学验证面（TensorDEM）— 内核建成、物理验证 honest FAIL",
    "summary": ("新增逐颗粒半径二元 DEM 内核与稳性传导映射（单元测试 9/9 通过，"
                "质量/质心/自由液面审计误差 <1e-12）；但分凝物理验证失败：N>=1e4 沉降阶段壁面接触力出现"
                "非有限值（已定位未修完），N=4e3 下粗颗粒下沉（S<0）且 S 对 Γ 完全无依赖（A/d~0.1 驱动过弱）。"
                "判据 (a)-(f) 均不主张满足。"),
    "scenario_key": "platform_audit",
    "mission_id": 7300,
    "tags": ["TensorDEM", "DEM", "granular", "bidisperse", "segregation",
             "Brazil-nut", "stability", "honest-FAIL", "P70-N03"],
    "markdown": MD,
    "figures": [
        fig("fig_S_size_ratio.png",
            "图 1：分凝指数 S（顶层 20 % 粗颗粒体积分数相对超出量）对尺寸比 d_l/d_s，"
            "Γ=2、f=25 Hz、N=4e3、30 周期、3 seeds；红线为等径负控 ±0.02 判定带。"
            "S 非单调且 s=2 强负 → 粗颗粒下沉，未复现巴西坚果效应。"),
        fig("fig_S_gamma.png",
            "图 2：S 对振动强度 Γ（s=2、f=25 Hz、3 seeds）——同一 seed 下数值完全相同，"
            "说明弱驱动（A/d≈0.1）未改组床体，无法拟合阈值/饱和。"),
    ],
}

status, resp = post("/tensor/reports", report, tok)
print("report", status, json.dumps(resp, ensure_ascii=False)[:400])