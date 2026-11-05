# PR #235 CI修复总结

## 任务目标
修复TensorDEM PR #235（分支agent/m3731）的CI失败，使unit-tests作业变绿。

## 问题诊断
**失败测试**：`tests/test_dem3d_contact_invariants.py::KernelInvariantTests::test_mixed_topology_all_invariants_pass`

**错误类型**：AttributeError
```
AttributeError: module 'tensordem.contact_invariants' has no attribute 'translation'
AttributeError: module 'tensordem.contact_invariants' has no attribute 'rotation'
AttributeError: module 'tensordem.contact_invariants' has no attribute 'brute_force'
AttributeError: module 'tensordem.contact_invariants' has no attribute 'determinism'
```

**根因**：测试使用getattr动态调用函数时使用了错误的短名称，而实际函数名为：
- `translation_invariance`（需3参数）
- `rotation_equivariance`（需2参数）
- `brute_force_agreement`（需2参数）
- `determinism_repeatability`（需2参数）

## 修复方案
在`tests/test_dem3d_contact_invariants.py`中建立显式函数字典映射：

```python
invariant_funcs = {
    "momentum_closure": lambda case, seed: momentum_closure(case, seed),
    "translation": lambda case, seed: translation_invariance(case, seed, (1.5, -7.25, 3.5)),
    "rotation": lambda case, seed: rotation_equivariance(case, seed),
    "brute_force": lambda case, seed: brute_force_agreement(case, seed),
    "determinism": lambda case, seed: determinism_repeatability(case, seed),
    "load_superposition": lambda case, seed: load_superposition(case, seed),
}
```

## 验证结果
**修复前**：8个子测试失败（4个不变量×2个种子）
**修复后**：15个测试全部通过，56个子测试全部通过

```bash
$ python -m pytest tests/test_dem3d_contact_invariants.py -v
============= 15 passed, 2 warnings, 56 subtests passed in 21.65s ==============
```

## 改动文件
- `tests/test_dem3d_contact_invariants.py`：修复test_mixed_topology_all_invariants_pass方法

## 提交记录
```
6d9d138 fix: correct function name mapping in mixed topology invariant test
```

## 状态
✅ 本地验证通过
⏳ 等待平台自动推送至GitHub（安全策略禁止直接push）
