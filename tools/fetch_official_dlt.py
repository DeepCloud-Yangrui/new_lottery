# -*- coding: utf-8 -*-
"""抓取官方（中国体育彩票）超级大乐透历史开奖与奖级明细。

数据源：https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry
        gameNo=85（超级大乐透）

用途：为 5+12 计奖核验提供可追溯的官方奖级/奖金/奖池快照。
本脚本只读官方接口，不修改仓库生产数据 data/dlt/data.csv。

用法：
    python tools/fetch_official_dlt.py --out outputs/<run_dir> [--pages 0]

输出：
    <out>/sources/official_api_page_<n>.json   原始分页响应（可追溯）
    <out>/data/official_dlt_draws.csv          每期一行（宽表，含各奖级单注奖金）
    <out>/data/official_dlt_prize_levels.csv   长表：每期每奖级一行
    <out>/data/official_fetch_meta.json        抓取元信息（时间、URL、条数、sha256）
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import requests

API = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
GAME_NO = "85"  # 超级大乐透

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.sporttery.cn/",
    "Origin": "https://www.sporttery.cn",
}

LEVEL_KEYS = [
    "一等奖", "二等奖", "三等奖", "四等奖", "五等奖",
    "六等奖", "七等奖", "八等奖", "九等奖",
]
ADD_LEVEL_KEYS = [k + "(追加)" for k in LEVEL_KEYS]
ALL_LEVEL_KEYS = []
for _k in LEVEL_KEYS:
    ALL_LEVEL_KEYS.append(_k)
    ALL_LEVEL_KEYS.append(_k + "(追加)")


def _num(s):
    """把 '10,000,000' / '7.50' / '---' / '' 解析为 **float** 或 None。

    ⚠️ 修复记录（2026-10-14）：原实现是 ``int(float(s))``，会把接口以字符串
    公布的半元金额**截断**。例如 2026 派奖期的 ``六等奖派奖`` 原始值是
    ``stakeAmount='7.50'``（``totalPrizeamount='4,521,202.50'``，恰等于
    602,827 × 7.50），旧实现入库成 7，导致 ``prize_rules`` 的 official 口径
    在这些期次把一组 5+12 的 R(3) 少算成 1115（应为 1125）。
    现在统一保留小数，不再截断。
    """
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not s or s in {"---", "-", "--"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def fetch_page(page_no, page_size, timeout=30, retries=4):
    params = {
        "gameNo": GAME_NO,
        "provinceId": "0",
        "pageSize": str(page_size),
        "isVerify": "1",
        "pageNo": str(page_no),
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("fetch page {} failed: {}".format(page_no, last_err))


def fetch_all(page_size=100, max_pages=0, sleep_s=0.4, verbose=True):
    """翻页抓取所有历史开奖，返回 (records, raw_pages, total)。"""
    first = fetch_page(1, page_size)
    pages_total = int(first["value"]["pages"])
    total = int(first["value"]["total"])
    raw_pages = {1: first}
    if max_pages and pages_total > max_pages:
        pages_total = max_pages
    if verbose:
        print("官方接口: 共 {} 期, {} 页 (pageSize={}), 本次抓取 {} 页".format(
            total, pages_total, page_size, pages_total))

    records = list(first["value"]["list"])
    for p in range(2, pages_total + 1):
        if verbose and (p % 10 == 0 or p == pages_total):
            print("  ... page {}/{}".format(p, pages_total))
        j = fetch_page(p, page_size)
        raw_pages[p] = j
        records.extend(j["value"]["list"])
        time.sleep(sleep_s)
    return records, raw_pages, total


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def to_rows(records):
    """转为 (draw_rows, level_rows)。"""
    draw_rows, level_rows = [], []
    for it in records:
        issue = str(it.get("lotteryDrawNum", "")).strip()
        result = str(it.get("lotteryDrawResult", "")).strip()
        nums = result.split()
        if len(nums) != 7:
            continue
        row = {
            "issue": int(issue) if issue.isdigit() else issue,
            "draw_date": it.get("lotteryDrawTime", ""),
            "front_1": int(nums[0]), "front_2": int(nums[1]), "front_3": int(nums[2]),
            "front_4": int(nums[3]), "front_5": int(nums[4]),
            "back_1": int(nums[5]), "back_2": int(nums[6]),
            "pool_after": _num(it.get("poolBalanceAfterdraw")),
            "sale": _num(it.get("totalSaleAmount")),
        }
        by_level = {}
        for pl in it.get("prizeLevelList", []) or []:
            name = str(pl.get("prizeLevel", "")).strip()
            amt = _num(pl.get("stakeAmount"))
            cnt = _num(pl.get("stakeCount"))
            tot = _num(pl.get("totalPrizeamount"))
            by_level[name] = (amt, cnt, tot)
            level_rows.append({
                "issue": row["issue"], "level": name,
                "stake_amount": amt, "stake_count": cnt, "total_amount": tot,
            })
        for key in ALL_LEVEL_KEYS:
            amt, cnt, _tot = by_level.get(key, (None, None, None))
            row["amt_" + key] = amt
            row["cnt_" + key] = cnt
        draw_rows.append(row)
    draw_rows.sort(key=lambda r: r["issue"] if isinstance(r["issue"], int) else 0)
    level_rows.sort(key=lambda r: (r["issue"] if isinstance(r["issue"], int) else 0, r["level"]))
    return draw_rows, level_rows


def write_csv(path, rows, fieldnames):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录（独立运行目录）")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=0, help="0=全部")
    ap.add_argument("--fetch", type=int, default=1,
                    help="1=联网抓取；0=复用已有 sources/ 快照重新解析")
    args = ap.parse_args()

    out = args.out
    src_dir = os.path.join(out, "sources")
    data_dir = os.path.join(out, "data")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if args.fetch:
        records, raw_pages, total = fetch_all(page_size=args.page_size,
                                              max_pages=args.max_pages)
        for p, j in raw_pages.items():
            with open(os.path.join(src_dir, "official_api_page_{:04d}.json".format(p)),
                      "w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False, indent=1)
        raw_total = total
    else:
        records = []
        for fn in sorted(os.listdir(src_dir)):
            m = re.match(r"official_api_page_(\d+)\.json$", fn)
            if not m:
                continue
            with open(os.path.join(src_dir, fn), "r", encoding="utf-8") as f:
                records.extend(json.load(f)["value"]["list"])
        raw_total = len(records)

    draw_rows, level_rows = to_rows(records)
    draw_path = os.path.join(data_dir, "official_dlt_draws.csv")
    lvl_path = os.path.join(data_dir, "official_dlt_prize_levels.csv")
    write_csv(draw_path, draw_rows, list(draw_rows[0].keys()))
    write_csv(lvl_path, level_rows, ["issue", "level", "stake_amount", "stake_count", "total_amount"])

    meta = {
        "source": "中国体育彩票官方开奖接口 webapi.sporttery.cn (gameNo=85 超级大乐透)",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "api": API,
        "n_draws_parsed": len(draw_rows),
        "n_draws_reported_by_api": raw_total,
        "issue_min": draw_rows[0]["issue"],
        "issue_max": draw_rows[-1]["issue"],
        "files": {
            "official_dlt_draws.csv": _sha256(draw_path),
            "official_dlt_prize_levels.csv": _sha256(lvl_path),
        },
    }
    with open(os.path.join(data_dir, "official_fetch_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
