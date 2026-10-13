# -*- coding: utf-8 -*-
"""``prize_rules`` 的适配入口（原为独立重复实现）。

背景（PLAN §2 风险 4）
----------------------
本文件原先自带一套与根目录 ``prize_rules.py`` **重复**的计奖实现
（时期划分、奖级→条件映射、基础档/上浮档、官方快照解析）。
两套实现一旦漂移就会产生无法察觉的计奖差异。

处理
----
`tests/test_profit_accounting.py::test_duplicate_prize_implementations_agree`
先对两套实现逐期逐奖级做了一致性核对（含 25001 / 26001 / 26014 / 26050 /
26051 / 26116 六个代表期，full 与 no_bonus 两种口径）。
**一致性测试通过后**，本文件改为 ``prize_rules`` 的**薄适配层**：
所有规则常量与奖金表都来自 ``prize_rules``，这里只保留历史调用方需要的
少量接口形状（``load_official(run_dir)`` 返回带 ``.draws`` / ``.pool_before`` 字典
的对象；``issue_table(issue, bonus_mode)`` 使用旧参数名）。

保留这些形状是为了不动 ``tools/five_plus12_math.py``、
``tools/analysis_math_baseline.py`` 与既有测试；**不再新增任何规则逻辑**。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as _pr                       # noqa: E402

# --- 规则常量：直接引用唯一实现 -------------------------------------------
LEVEL_CONDITIONS = _pr.LEVEL_CONDITIONS
BASE_AMOUNT = _pr.BASIC_AMOUNT
ERA3_POOL_THRESHOLD = _pr.ERA3_POOL_THRESHOLD
ERA3_HIGH_AMOUNT = _pr.ERA3_HIGH_AMOUNT
ERA_BOUNDS = _pr.ERA_BOUNDS

era_of = _pr.era_of


def base_table(issue: int, pool_before=None) -> Dict:
    """规则基础档（不含派奖）。委托 ``prize_rules.base_table``。"""
    return _pr.base_table(issue, pool_before=pool_before)


# ---------------------------------------------------------------------------
# 官方快照适配
# ---------------------------------------------------------------------------

class OfficialPrizeData:
    """``prize_rules.OfficialSnapshot`` 的适配器。

    额外提供旧调用方需要的 ``.draws``（DataFrame）与 ``.pool_before``（**字典**）。
    注意：``prize_rules.OfficialSnapshot.pool_before`` 是**方法**，此处是**字典**，
    这是历史参数形状，别混用。
    """

    _BONUS_MODE = {"full": "official",
                   "expected": "official_expected_bonus",
                   "none": "official_no_bonus"}

    def __init__(self, draws_csv: str, levels_csv: str):
        import pandas as pd
        self._pd = pd
        self.draws = pd.read_csv(draws_csv)
        self.pool_before: Dict[int, Optional[float]] = {}
        prev = None
        for r in self.draws.itertuples():
            self.pool_before[int(r.issue)] = prev
            if pd.notna(r.pool_after):
                prev = float(r.pool_after)

    @property
    def issues(self):
        return sorted(self.pool_before)

    def has(self, issue) -> bool:
        return int(issue) in self.pool_before

    def amount(self, issue, level):
        snap = _pr.get_snapshot()
        return snap.amount(issue, level)

    def issue_table(self, issue, bonus_mode="expected") -> Dict:
        """旧参数名 ``bonus_mode`` → ``prize_rules`` 的 mode。"""
        mode = self._BONUS_MODE.get(bonus_mode, "official")
        return _pr.get_prize_table(issue, mode=mode)


def load_official(run_dir: str) -> OfficialPrizeData:
    return OfficialPrizeData(
        os.path.join(run_dir, "data", "official_dlt_draws.csv"),
        os.path.join(run_dir, "data", "official_dlt_prize_levels.csv"),
    )
