# -*- coding: utf-8 -*-
"""
Author: BigCat
"""
import os

ball_name = [
    ("红球", "red"),
    ("蓝球", "blue")
]

data_file_name = "data.csv"

name_path = {
    "ssq": {
        "name": "双色球",
        "path": "data/ssq/"
    },
    "dlt": {
        "name": "大乐透",
        "path": "data/dlt/"
    }
}

model_path = os.getcwd() + "/model/"

model_args = {
    "ssq": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "sequence_len": 6,
            "red_n_class": 33,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_n_class": 16,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/ssq/red_ball_model/",
            "blue": model_path + "/ssq/blue_ball_model/"
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 5,
            "red_n_class": 35,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 2,
            "blue_n_class": 12,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/dlt/red_ball_model/",
            "blue": model_path + "/dlt/blue_ball_model/"
        }
    }
}

# 模型名
pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"

# --- 大乐透 5+12 策略参数 ---
plus_strategy = {
    "dlt": {
        # 前区 Top-N 候选池
        "top_n_front": 10,
        # 候选池排序后保留的前区组合上限
        "max_front_combos": 20,
        # 实际每期下注的前区组合数（每组固定配后区66注）
        "play_front_combos": 1,
        # 前区过滤规则
        "rule_filters": {
            "odd_min": 1,
            "odd_max": 4,
            "big_min": 1,
            "big_max": 4,
            "sum_min": 55,
            "sum_max": 125,
            "max_overlap_with_last": 2
        },
        # 候选号码评分窗口
        "score_windows": {
            "short": 10,
            "mid": 30,
            "long": 100
        },
        # LSTM + RF + 统计分 融合权重
        "ensemble_weights": {
            "lstm": 0.35,
            "rf": 0.45,
            "stat": 0.20
        },
        # 回测参数
        "backtest": {
            "start_issue": None,
            "end_issue": None,
            "rolling_train_size": 500
        },
        # ⚠️ 已废弃（2026-10-13）：下面这组"奖金近似"从未被回测代码使用，
        # 且数值已被官方核验证伪（2019-02-20 起 (3,1) 应为 15 元而非 10 元）。
        # 计奖请使用 prize_rules.get_prize_table(issue)。
        "payouts": {
            "3+2": 200,
            "3+1": 10,
            "3+0": 5
        }
    }
}

# --- 大乐透奖金表 ---
#
# ⚠️ 更正说明（2026-10-13）
# 下面两张静态表（_DLT_PRIZE_TABLE_PRE_2026 / _DLT_PRIZE_TABLE_2026）经官方开奖
# 数据（2934 期，7001–26116）逐期核对后被确认有误，仅作为历史产物的存档保留，
# 不再作为默认计奖依据。已确认的主要错误：
#   1. 2019-02-20（第 19019 期）起 (3,1)/(2,2) 的单注奖金是 15 元（八等奖），
#      旧表写 10 元 → 旧表把"前区中 3 个"的单组回收算成 625 元，实际是 725 元。
#   2. 2026-02-02（第 26014 期）起奖级由 9 个精简为 7 个，且固定奖金额不再固定：
#      当期开奖前奖池 >= 8 亿元时启用上浮档。旧表把 (3,2) 记 200、(3,1) 记 15、
#      (3,0) 记 5，在 26014 之后都不成立：前区中 3 个的单组回收实际为 675 元
#      （基础档）或 875 元（奖池 >= 8 亿上浮档）。
#   3. 旧表未建模历次固定奖派奖活动（2021–2026 共 6 次）。
#   4. 旧表把一、二等奖当作固定 1000 万 / 20 万，与真实浮动奖金差距很大。
#
# 现行计奖请使用 prize_rules.get_prize_table()：
#   - 已开奖期号：官方逐期公布的单注奖金 + 全额派奖
#     （5+12 单票 132 元满足"单张单期 >= 18 元"的派奖参与门槛）
#   - 未来期号  ：规则基础档（era3 基础档 / 奖池 >= 8 亿时的上浮档）
# 详见 docs/project_notes/P5_PRIZE_RULE_VERIFICATION.md。

# 2026年改革前（issue < 26014）——历史存档，已知有误，勿用于新实验
_DLT_PRIZE_TABLE_PRE_2026 = {
    (5, 2): 10_000_000,  # 一等奖（浮动，近似值）
    (5, 1): 200_000,     # 二等奖（浮动，近似值）
    (5, 0): 10_000,      # 三等奖
    (4, 2): 3_000,       # 四等奖
    (4, 1): 300,         # 五等奖
    (4, 0): 100,         # 六等奖
    (3, 2): 200,         # 六等奖
    (3, 1): 10,          # 七等奖（旧规则）
    (2, 2): 10,          # 七等奖（旧规则）
    (3, 0): 5,           # 八等奖
    (2, 1): 5,           # 八等奖
    (1, 2): 5,           # 八等奖
    (0, 2): 5,           # 八等奖
}

# 2026年改革后（issue >= 26014）——历史存档，已知有误，勿用于新实验
_DLT_PRIZE_TABLE_2026 = {
    (5, 2): 10_000_000,  # 一等奖（浮动，近似值）
    (5, 1): 200_000,     # 二等奖（浮动，近似值）
    (5, 0): 10_000,      # 三等奖
    (4, 2): 3_000,       # 四等奖
    (4, 1): 300,         # 五等奖
    (4, 0): 100,         # 七等奖
    (3, 2): 200,         # 六等奖
    (3, 1): 15,          # 八等奖（2026新规则上调）
    (2, 2): 15,          # 八等奖（2026新规则上调）
    (3, 0): 5,           # 九等奖
    (2, 1): 5,           # 九等奖
    (1, 2): 5,           # 九等奖
    (0, 2): 5,           # 九等奖
}

# 改革分界线期号（含）：从此期起使用 2026 新规则
_DLT_REFORM_ISSUE = 26014


def get_dlt_prize_table(issue, mode="official"):
    """根据期号返回适用的单注奖金表 {(前区命中, 后区命中): 奖金元}。

    本函数已改为委托 ``prize_rules.get_prize_table``（官方核验版）。
    若要复现历史（2026-10-13 之前）的回测结论，请显式传入
    ``mode="legacy_config"``。

    Args:
        issue: 大乐透期号（int）
        mode: "official"（默认）| "official_no_bonus" | "official_full_bonus"
              | "base" | "legacy_config"
    """
    from prize_rules import get_prize_table as _get
    return _get(issue, mode=mode)



# RF 训练参数与存储路径
rf_args = {
    "dlt": {
        "front_model_path": model_path + "/dlt/rf_front_model.pkl",
        "front_meta_path": model_path + "/dlt/rf_front_meta.json",
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "random_state": 42
    }
}
