# -*- coding: utf-8 -*-
"""超级大乐透奖级/奖金规则（官方核验版，供回测与推理共用）。

**为什么需要这个模块**

``config.py`` 原先用两张静态奖金表覆盖 2007–2026：
``_DLT_PRIZE_TABLE_PRE_2026`` 与 ``_DLT_PRIZE_TABLE_2026``。
用官方开奖数据（2934 期，7001–26116）逐期核对后确认该表存在多处错误，
详见 ``docs/project_notes/P5_PRIZE_RULE_VERIFICATION.md``：

1. 2019-02-20（第 19019 期）起，``3+1`` 与 ``2+2`` 的单注奖金是 **15 元**
   （八等奖），而旧表写 10 元；由此旧表把"前区中 3 个"的单组回收算成 625 元。
2. 2026-02-02（第 26014 期）起奖级由 9 个精简为 7 个，且**固定奖金额不再固定**：
   当期开奖前奖池 >= 8 亿元时启用上浮档。旧表把 (3,2) 记 200 元、
   (3,1) 记 15 元、(3,0) 记 5 元，都只是 2019–2026 旧规则的一部分，
   在 2026 新规则下"前区中 3 个"的单组回收实际是 **675 元**（基础档）
   或 **875 元**（奖池 >= 8 亿的上浮档），而不是 725 元。
3. 旧表完全没有建模一年一度的**固定奖派奖活动**（2021–2026 共 6 次，
   每次约 17~33 期），这类活动会显著抬高小奖回收。
4. 旧表用 1000 万 / 20 万 作为一、二等奖的固定值，与真实浮动奖金差距很大。

**本模块的三层信息**

A. ``LEVEL_CONDITIONS``：每个规则时期各奖级对应的 (前区命中, 后区命中) 集合。
   由官方逐期中奖注数反推（相对误差 < 4%），并与规则文本交叉验证。
B. 官方逐期奖金表（``data/dlt/official_dlt_prize_levels.csv``，由
   ``tools/fetch_official_dlt.py`` 生成）：直接使用逐期公布的单注奖金，
   并按"全额派奖"计入派奖（5+12 单票满足派奖参与门槛）。适用于已开奖期号。
C. ``BASIC_AMOUNT`` 基础档：官方数据未覆盖的未来期次使用，
   必须标注为"基础档估计"。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

Condition = Tuple[int, int]

# 独立运行目录之外的公共快照位置（data/ 已被 .gitignore 忽略，需用
# tools/fetch_official_dlt.py 重新生成）
DEFAULT_SNAPSHOT_DIR = os.path.join("data", "dlt")
DRAWS_CSV = "official_dlt_draws.csv"
LEVELS_CSV = "official_dlt_prize_levels.csv"

ERA_BOUNDS = [
    (7001, 9120, "era1a"),
    (9121, 14051, "era1b"),
    (14052, 19018, "era1c"),
    (19019, 26013, "era2"),
    (26014, 10 ** 9, "era3"),
]

ERA_LABEL = {
    "era1a": "2007-05-30 ~ 2009-10-14",
    "era1b": "2009-10-17 ~ 2014-05-05",
    "era1c": "2014-05-07 ~ 2019-02-18",
    "era2": "2019-02-20 ~ 2026-01-31（9 个奖级）",
    "era3": "2026-02-02 起（奖级精简为 7 个）",
}

LEVEL_CONDITIONS: Dict[str, Dict[str, set]] = {
    "era1a": {
        "一等奖": {(5, 2)}, "二等奖": {(5, 1)}, "三等奖": {(5, 0)},
        "四等奖": {(4, 2)}, "五等奖": {(4, 1)}, "六等奖": {(4, 0), (3, 2)},
    },
    "era1b": {
        "一等奖": {(5, 2)}, "二等奖": {(5, 1)}, "三等奖": {(5, 0)},
        "四等奖": {(4, 2)}, "五等奖": {(4, 1)}, "六等奖": {(4, 0), (3, 2)},
    },
    "era1c": {
        "一等奖": {(5, 2)}, "二等奖": {(5, 1)}, "三等奖": {(5, 0), (4, 2)},
        "四等奖": {(4, 1), (3, 2)}, "五等奖": {(1, 2)},
        "六等奖": {(4, 0), (3, 1), (2, 2), (3, 0), (2, 1), (0, 2)},
    },
    "era2": {
        "一等奖": {(5, 2)}, "二等奖": {(5, 1)}, "三等奖": {(5, 0)},
        "四等奖": {(4, 2)}, "五等奖": {(4, 1)}, "六等奖": {(3, 2)},
        "七等奖": {(4, 0)}, "八等奖": {(3, 1), (2, 2)},
        "九等奖": {(3, 0), (2, 1), (1, 2), (0, 2)},
    },
    "era3": {
        "一等奖": {(5, 2)}, "二等奖": {(5, 1)},
        "三等奖": {(5, 0), (4, 2)}, "四等奖": {(4, 1)},
        "五等奖": {(4, 0), (3, 2)}, "六等奖": {(3, 1), (2, 2)},
        "七等奖": {(3, 0), (2, 1), (1, 2), (0, 2)},
    },
}

# 固定奖基础档（元）。None 表示该奖级是浮动奖。
BASIC_AMOUNT: Dict[str, Dict[str, Optional[int]]] = {
    "era1a": {"三等奖": None, "四等奖": 3000, "五等奖": 500, "六等奖": 200},
    "era1b": {"三等奖": None, "四等奖": 3000, "五等奖": 600, "六等奖": 100},
    "era1c": {"三等奖": None, "四等奖": 200, "五等奖": 10, "六等奖": 5},
    "era2": {"三等奖": 10000, "四等奖": 3000, "五等奖": 300, "六等奖": 200,
             "七等奖": 100, "八等奖": 15, "九等奖": 5},
    "era3": {"三等奖": 5000, "四等奖": 300, "五等奖": 150,
             "六等奖": 15, "七等奖": 5},
}

# era3：当期开奖前奖池 >= 8 亿元时的上浮档（官方逐期数据实证，0 例外）
ERA3_POOL_THRESHOLD = 800_000_000
ERA3_HIGH_AMOUNT = {"三等奖": 6666, "四等奖": 380, "五等奖": 200,
                    "六等奖": 18, "七等奖": 7}

# 派奖（加奖）对应的官方奖级行名
BONUS_LEVELS = {
    "一等奖": "一等奖基本派奖",
    "二等奖": "二等奖基本派奖",
    "三等奖": "三等奖派奖",
    "四等奖": "四等奖派奖",
    "五等奖": "五等奖派奖",
    "六等奖": "六等奖派奖",
    "七等奖": "七等奖派奖",
    "八等奖": "八等奖派奖",
    "九等奖": "九等奖派奖",
}

# 无官方数据时的一、二等奖替代值（保守，低于 2019 年以来的实测均值）
FLOATING_FALLBACK = {"一等奖": 9_000_000, "二等奖": 180_000}

# 已知的派奖活动期（用于文档与前瞻记录）
BONUS_CAMPAIGNS = [
    ("21039", "21057"), ("22040", "22068"), ("23039", "23059"),
    ("24030", "24062"), ("25038", "25055"), ("26050", "26066"),
]


def era_of(issue: int) -> str:
    issue = int(issue)
    for lo, hi, name in ERA_BOUNDS:
        if lo <= issue <= hi:
            return name
    raise ValueError("期号 {} 不在已知范围内".format(issue))


def level_of(issue: int, front_hit: int, back_hit: int) -> Optional[str]:
    for level, conds in LEVEL_CONDITIONS[era_of(issue)].items():
        if (front_hit, back_hit) in conds:
            return level
    return None


# ---------------------------------------------------------------------------
# 官方快照
# ---------------------------------------------------------------------------

class OfficialSnapshot:
    """官方逐期奖金快照（可选）。缺失时自动降级为基础档。"""

    def __init__(self, draws_csv=None, levels_csv=None):
        self.ok = False
        self.levels: Dict[Tuple[int, str], Tuple[Optional[float], Optional[float]]] = {}
        self.pool_after: Dict[int, float] = {}
        self.dates: Dict[int, str] = {}
        self.draws = None
        self.issues = []
        if not draws_csv:
            draws_csv = os.path.join(DEFAULT_SNAPSHOT_DIR, DRAWS_CSV)
            levels_csv = os.path.join(DEFAULT_SNAPSHOT_DIR, LEVELS_CSV)
        if not (os.path.exists(draws_csv) and os.path.exists(levels_csv)):
            return
        import pandas as pd
        d = pd.read_csv(draws_csv)
        if "amt_一等奖" not in d.columns:
            return
        self.draws = d
        self.dates: Dict[int, str] = {}
        for r in d.itertuples():
            self.pool_after[int(r.issue)] = None if pd.isna(r.pool_after) else float(r.pool_after)
            try:
                self.dates[int(r.issue)] = str(r.draw_date)
            except Exception:
                pass
        self.issues = sorted(self.pool_after)
        lv = pd.read_csv(levels_csv)
        for r in lv.itertuples():
            amt = None if pd.isna(r.stake_amount) else float(r.stake_amount)
            cnt = None if pd.isna(r.stake_count) else float(r.stake_count)
            tot = None if pd.isna(r.total_amount) else float(r.total_amount)
            # 数据质量修正：官方接口在 23039–23059（2023 年派奖期）把派奖各奖级的
            # stakeAmount 返回 0（stakeAmountFormat=-1），此时用
            # totalPrizeamount / stakeCount 反推单注派奖额。
            if (amt is None or amt == 0) and cnt and tot:
                amt = tot / cnt
            self.levels[(int(r.issue), r.level)] = (amt, cnt)
        self._issue_set = set(self.issues)
        self.ok = True

    def has(self, issue) -> bool:
        return self.ok and int(issue) in self._issue_set

    def pool_before(self, issue):
        """当期开奖前奖池 = 上一期的开奖后奖池。

        对**尚未开奖**的期次（期号大于快照中最新一期）同样成立：
        其开奖前奖池就是快照最新一期的开奖后奖池，该信息已公开，
        可用于开奖前的档位判定（不构成前视）。
        """
        if not self.ok:
            return None
        iss = self.issues
        i = int(issue)
        if i > iss[-1]:
            return self.pool_after.get(iss[-1])
        try:
            idx = iss.index(i)
        except ValueError:
            return None
        if idx == 0:
            return None
        return self.pool_after.get(iss[idx - 1])

    def amount(self, issue, level):
        return self.levels.get((int(issue), level), (None, None))

    def issue_table(self, issue, bonus_mode="expected") -> Dict[Condition, int]:
        issue = int(issue)
        conds = LEVEL_CONDITIONS[era_of(issue)]
        table: Dict[Condition, int] = {}
        for level, cset in conds.items():
            amt, cnt = self.amount(issue, level)
            if amt is None or cnt is None or cnt <= 0:
                continue
            eff = float(amt)
            if bonus_mode != "none":
                b_amt, b_cnt = self.amount(issue, BONUS_LEVELS.get(level, ""))
                if b_amt and b_cnt:
                    eff += b_amt if bonus_mode == "full" else b_amt * (b_cnt / cnt)
            for c in cset:
                table[c] = int(round(eff))
        return table


_SNAPSHOT: Optional[OfficialSnapshot] = None
_SNAPSHOT_TRIED = False


def get_snapshot(reload=False) -> OfficialSnapshot:
    global _SNAPSHOT, _SNAPSHOT_TRIED
    if _SNAPSHOT_TRIED and not reload:
        return _SNAPSHOT
    _SNAPSHOT = OfficialSnapshot()
    _SNAPSHOT_TRIED = True
    return _SNAPSHOT


# ---------------------------------------------------------------------------
# 面向回测/推理的统一入口
# ---------------------------------------------------------------------------

PRIZE_MODES = ("official", "official_full_bonus", "official_expected_bonus",
               "official_no_bonus", "base", "legacy_config")

# 派奖参与门槛：官方公告为"单票单期满 18 元"。一组 5+12 = 66 注 × 2 元 = 132 元，
# 是**单张票、单期**投注，满足门槛，因此中奖注可拿到全额派奖（50%；
# 七等奖为 100%）。未达门槛的是单注小额票，官方数据里体现为
# "派奖注数 < 中奖注数"（约 55%~78%）。
# 因此本项目默认口径 official == official_full_bonus。
_BONUS_MODE_ALIASES = {
    "official": "full",
    "official_full_bonus": "full",
    "official_expected_bonus": "expected",
    "official_no_bonus": "none",
}


def get_prize_table(issue: int, mode: str = "official",
                    snapshot: Optional[OfficialSnapshot] = None
                    ) -> Dict[Condition, int]:
    """返回 {(前区命中, 后区命中): 单注奖金（元）}。

    mode:
      official            官方逐期奖金 + **全额派奖**（默认）。
                          5+12 单票 132 元满足"单票单期满 18 元"的派奖门槛，
                          故本项目的中奖注按全额派奖计。
      official_full_bonus 同 official（显式名）
      official_expected_bonus 派奖按"该奖级平均每中奖注"计（保守下界）
      official_no_bonus   官方逐期奖金，不计派奖（更保守的下界）
      base                规则基础档（不含派奖、不含浮动；era3 默认基础档）
      legacy_config       旧的 config 静态表（仅供复现历史结论，已知有误）
    """
    issue = int(issue)
    if mode == "legacy_config":
        return _legacy_config_table(issue)
    snap = snapshot if snapshot is not None else get_snapshot()
    if mode in _BONUS_MODE_ALIASES and snap.has(issue):
        bm = _BONUS_MODE_ALIASES[mode]
        t = snap.issue_table(issue, bm)
        if t:
            return t
    return base_table(issue, pool_before=(snap.pool_before(issue) if snap.ok else None))


def base_table(issue: int, pool_before=None) -> Dict[Condition, int]:
    era = era_of(issue)
    amounts = dict(BASIC_AMOUNT[era])
    if era == "era3" and pool_before is not None and pool_before >= ERA3_POOL_THRESHOLD:
        amounts.update(ERA3_HIGH_AMOUNT)
    table: Dict[Condition, int] = {}
    for level, cset in LEVEL_CONDITIONS[era].items():
        amt = amounts.get(level)
        if amt is None:
            amt = FLOATING_FALLBACK.get(level)
        if amt is None:
            continue
        for c in cset:
            table[c] = int(amt)
    return table


# 旧的静态表，保留以便复现历史结论 / 做对照实验
_LEGACY_PRE_2026 = {
    (5, 2): 10_000_000, (5, 1): 200_000, (5, 0): 10_000, (4, 2): 3_000,
    (4, 1): 300, (4, 0): 100, (3, 2): 200, (3, 1): 10, (2, 2): 10,
    (3, 0): 5, (2, 1): 5, (1, 2): 5, (0, 2): 5,
}
_LEGACY_2026 = dict(_LEGACY_PRE_2026)
_LEGACY_2026.update({(4, 0): 100, (3, 1): 15, (2, 2): 15})
_LEGACY_REFORM_ISSUE = 26014


def _legacy_config_table(issue: int) -> Dict[Condition, int]:
    if issue is not None and int(issue) < _LEGACY_REFORM_ISSUE:
        return dict(_LEGACY_PRE_2026)
    return dict(_LEGACY_2026)


# ---------------------------------------------------------------------------
# 5+12 单组回收
# ---------------------------------------------------------------------------

BACK_HIT_TICKETS = {2: 1, 1: 20, 0: 45}
COST_PER_GROUP = 132


def group_payout(table: Dict[Condition, int], front_hit: int) -> int:
    """一组 5+12（前区 5 码 + 后区全包 66 注）的总奖金。"""
    return sum(n * table.get((front_hit, bh), 0) for bh, n in BACK_HIT_TICKETS.items())


def payout_by_front_hit(table: Dict[Condition, int]) -> Dict[int, int]:
    return {f: group_payout(table, f) for f in range(6)}
