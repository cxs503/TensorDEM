"""``python -m tensordem.sensitivity`` 入口：二维 DEM 网格/分辨率敏感性扫描。

README「网格/分辨率敏感性扫描」节记录的命令是::

    python -m tensordem.sensitivity --nx-levels 7,11,15 --duration 0.12 \\
        --speed 0.2 --output sensitivity-results

在该包被拆成 UQ 工具集（``design`` / ``sobol`` / ``surrogate`` / …）之后，
包内只剩 ``resolution.py:main``，没有 ``__main__.py``，于是上面这条命令以
``No module named tensordem.sensitivity.__main__`` 失败（README 与代码不一致）。
本模块只做转发，保证文档命令与其声明的产物
（``history_nx_*.csv`` / ``summary.csv`` / ``comparison_to_finest.csv``）可达。
"""

from __future__ import annotations

from .resolution import main

if __name__ == "__main__":
    main()