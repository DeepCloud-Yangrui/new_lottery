# -*- coding: utf-8 -*-
"""亏损约束与预算占用（V2 T4 / D5 的前置件）。

PLAN §5 的口径
--------------
* 累计亏损默认按 **500 元** 模拟，1000 元只做敏感性对照，**不自动放宽**；
* 不每日、每期、也不换策略重置；
* 参与前须满足：**当前累计亏损 + 未结算暴露 + 拟投入 ≤ 限额**；
* 最坏新增损失**保守按全部票款**计，**不预支**可能回收的小奖。

本模块只做约束判断，**不构成购彩授权**，也不决定"应该买几组"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def loss_to_date_from(cumulative_net: int) -> int:
    """把"累计净利润"换算成"累计亏损"（正数）。

    累计净利润为正时记 0（没有亏损），为负时取绝对值。
    该定义天然满足"不按期/不按策略重置"。
    """
    return max(0, -int(cumulative_net))


def can_allocate(loss_to_date: int, unsettled_cost: int,
                 proposed_cost: int, loss_limit: int) -> bool:
    """保守口径：`已实现亏损 + 未结算暴露 + 拟投入 ≤ 限额`。

    Args:
        loss_to_date: 当前累计亏损（正数；已实现盈利不作为可动用额度）
        unsettled_cost: 已投注但尚未开奖的票款
        proposed_cost: 本次拟投入的全部票款（最坏情况按全额损失计）
        loss_limit: 亏损硬上限

    Returns:
        bool
    """
    total = (max(0, int(loss_to_date)) + max(0, int(unsettled_cost))
             + max(0, int(proposed_cost)))
    return total <= int(loss_limit)


def max_affordable(loss_to_date: int, unsettled_cost: int,
                   loss_limit: int, unit_cost: int) -> int:
    """在限额内还能买多少个 `unit_cost` 的单元（向下取整，不少于 0）。"""
    if unit_cost <= 0:
        return 0
    free = int(loss_limit) - max(0, int(loss_to_date)) - max(0, int(unsettled_cost))
    return max(0, free // int(unit_cost))


@dataclass
class BudgetTracker:
    """跨策略、跨期持续的亏损预算追踪器。

    * 换策略**不重置**：内部只维护一个累计净收益；
    * 未结算暴露单独占用额度，结算后释放。
    """

    loss_limit: int = 500
    groups_per_period: int = 1
    unit_cost: int = 132
    realized_net: int = 0                     # 已结算的累计净利润（可为负）
    unsettled: Dict[str, dict] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)

    # -- 状态 ----------------------------------------------------------
    @property
    def loss_to_date(self) -> int:
        return loss_to_date_from(self.realized_net)

    @property
    def unsettled_cost(self) -> int:
        return sum(int(v["cost"]) for v in self.unsettled.values())

    @property
    def committed(self) -> int:
        """已实现亏损 + 未结算暴露。"""
        return self.loss_to_date + self.unsettled_cost

    def remaining(self) -> int:
        return int(self.loss_limit) - self.committed

    # -- 决策 ----------------------------------------------------------
    def can_allocate(self, proposed_cost: int) -> bool:
        return can_allocate(self.loss_to_date, self.unsettled_cost,
                            proposed_cost, self.loss_limit)

    def max_affordable_groups(self) -> int:
        return max_affordable(self.loss_to_date, self.unsettled_cost,
                              self.loss_limit, self.unit_cost)

    # -- 记账 ----------------------------------------------------------
    def reserve(self, ticket_id: str, cost: int, note: str = "") -> bool:
        """登记一笔**未结算**投注（占用额度）。超限则拒绝并返回 False。"""
        if not self.can_allocate(cost):
            self.history.append({"action": "reserve_rejected", "ticket_id": ticket_id,
                                 "cost": int(cost), "note": note})
            return False
        self.unsettled[ticket_id] = {"cost": int(cost), "note": note}
        self.history.append({"action": "reserve", "ticket_id": ticket_id,
                             "cost": int(cost), "note": note})
        return True

    def settle(self, ticket_id: str, net_profit: int) -> Optional[int]:
        """结算一笔投注：释放占用并计入累计净收益。"""
        if ticket_id not in self.unsettled:
            return None
        self.unsettled.pop(ticket_id)
        self.realized_net += int(net_profit)
        self.history.append({"action": "settle", "ticket_id": ticket_id,
                             "net_profit": int(net_profit),
                             "realized_net": self.realized_net})
        return self.realized_net

    def summary(self) -> dict:
        return {
            "loss_limit": self.loss_limit,
            "realized_net": self.realized_net,
            "loss_to_date": self.loss_to_date,
            "unsettled_cost": self.unsettled_cost,
            "committed": self.committed,
            "remaining": self.remaining(),
            "max_affordable_groups": self.max_affordable_groups(),
            "limit_breached": self.committed > self.loss_limit,
            "note": "换策略不重置；最坏新增损失按全部票款计，不预支小奖回收。",
        }
