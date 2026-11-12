# F4 修复：verify_friction_polyhedron 脚本改指向真实 DEM 验证面（m16306）

## 背景（audit_report.md F4）
`scripts/verify_friction_polyhedron.py` 原先从 `tensordem.friction_polyhedron_verification`
（伪验证模块）导入 `simulate_angle_of_repose` / `find_critical_angle` / `simulate_packing`，
该模块的 C1/C2/C3 返回 `atan(mu)` vs `atan(mu)`（恒等式，判据不可能失败）：
- C2 把 `find_critical_angle`（返回 `atan(mu)`）与 `expected = atan(mu)` 比较 → 误差恒 0%；
- C3 的 `simulate_packing` 返回硬编码文献堆积率，与自身比较。

## 修复内容
`scripts/verify_friction_polyhedron.py` 改从真实验证面
`tensordem.friction_polyhedron_real_verification` 导入：
- `find_critical_angle_sim`：C1/C2 临界角由 **DEM 二分法**（对实际下滑位移 `slide_m` 二分）测量，
  不再与 `atan(mu)` 恒等；
- `simulate_packing_fraction`：C3 堆积率由 **DEM 沉降**（粒子落箱 + 重力积分至动能 <1% 重力势能）测量，
  不再用硬编码文献值；
- `verify_packing_fraction`：C4 用真实 DEM 沉降测量 + 文献堆积带（0.55–0.65）作验收判据。

## 验证
- `PYTHONPATH=src python -c "from tensordem.friction_polyhedron_real_verification import ..."` → 导入成功；
- 脚本语法 `ast.parse` 通过；实际 import 语句指向 `friction_polyhedron_real_verification`（非伪模块）；
- 完整跑脚本为重型 DEM 计算（C1/C2 各 8 次二分 × 5000 步），本机 120s 预算内未跑完，
  已按容量纪律不压重算力（load15=110），留待集群/后续跑。

## 证据等级
- 伪模块 `friction_polyhedron_verification.py` 的 docstring 已自承「kept for import compatibility only
  and must not be cited」→ 标 **WITHDRAWN**（不引用为验证证据）；
- 真实验证面 `friction_polyhedron_real_verification.py` 为 **REAL**（DEM 积分测量）。
