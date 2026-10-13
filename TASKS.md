# TASKS — 盈利目标重规划 V2

> **2026-10-14交付核对**：已收到V3 E0—E4报告，本次全量测试152 passed、0 skipped；接受有限研究交付，预算按E3跳过。当前阶段进入代码/数据交接，服务器F1尚未执行。权威勘误与新旧快照用途见 `docs/project_notes/V3_HANDOFF_REVIEW_2026-10-14.md`；旧“仅设计/等待V3报告”状态为历史记录。

## 当前任务：服务器接管与F1五候选比较

执行依据：[服务器交接计划](docs/project_notes/SERVER_F1_HANDOFF_PLAN.md)。以下为当前状态，下方旧阶段记录不可作为本次已执行证据。

- [x] 按用户决定制定服务器统一研究计划，加入固定小型MLP与三个主比较的统计调整。
- [x] S0：收到本地V3 E0—E4报告，核对交付，全量152 passed；文档残留通过交接勘误说明。
- [ ] S1：显式暂存相关成果，提交推送origin；单独交接冻结数据/来源证据包及SHA256。
- [ ] S2：服务器Claude Code探测环境、拉取同一SHA、核对输入、建立独立研究环境并冒烟。
- [ ] F1-0/1：在服务器冻结配置，实现U/F/L/R/N与防泄漏、早停、种子平均测试。
- [ ] F1-2：服务器完成一次冻结历史评估，记录全部候选/种子与失败项。
- [ ] F1-3：报告概率、命中、收益及局限，提交小型成果；关闭本轮或交付冻结候选。

本次未修改实现代码、未训练、未提交推送、未访问服务器。用户已指定服务器Claude Code执行，无需向Codex提供SSH信息。

## F1 前区概率预测（方向已批准，设计完成）

- [x] 编写 `docs/project_notes/FRONT_PROBABILITY_RESEARCH_PLAN.md`。
- [ ] F1-0：协议、数据快照、4个时间块和参数冻结。
- [ ] F1-1：U/F/L/R概率模型适配与防泄漏测试。
- [ ] F1-2：一次冻结历史评估，保存逐期35号概率和Top-5。
- [ ] F1-3：概率/命中/收益分层报告及本轮关闭。

独立于V3；不运行GPU训练、不覆盖生产模型、不自动启用后续前瞻。

## 新阶段 V3：派奖叠加有限实验（仅设计完成）

- [x] 根据用户选择设计有限模拟实验，见 `docs/project_notes/PROFIT_V3_BONUS_FEASIBILITY_PLAN.md`。
- [ ] E0：冻结规则、四格协议、输入证据及独立输出目录。
- [ ] E1：实际派奖生效与金额精度回归测试、最小修复。
- [ ] E2：精确四格收益与浮动奖盈亏平衡边界。
- [ ] E3：仅候选有依据时做500/1000元预算模拟。
- [ ] E4：定向验证、结果报告与关闭本轮。

本阶段未实施代码或实验，不自动恢复前瞻/投注/服务器部署。以下保留V2任务记录。

2026-10-13。执行依据：[PLAN.md](PLAN.md)。不将旧报告的完成标记当作纠错已验收。

## 本轮已完成：规划

- [x] 阅读当前文档、P5结果与关键源码。
- [x] 确认包含一、二等奖的主盈利目标。
- [x] 确认一周研究周期与500—1000元亏损区间。
- [x] 保存原AGENTS/ROADMAP/TASKS归档。
- [x] 编写整体设计、执行任务、文件边界与验收门槛。
- [x] D1—D3 执行与关卡（见下）。

## 结论（统一表述）

> **当前未发现可信盈利候选，既有历史表现和预算约束不支持进入实际投注，
> 因此结束本轮研究。**
>
> **这不是对所有 5+12 策略永远不可能盈利的证明。**

## Claude Code执行队列

- [x] **T1 / D1：P5统计勘误、盈亏平衡公式及定向测试。**
  产物：`docs/project_notes/P5_STATISTICAL_ERRATA.md`；
  `tests/test_profit_math_v2.py`（20 项，实测通过）；
  派生结果 `outputs/2026-10-13_5plus12_correction_r2/corrections/`。
  修正：`tools/five_plus12_math.py`（盈亏平衡重写 + 精确多组矩）、
  `tools/analysis_summary.py`（撤回 z≈−9~−11）、`tools/eval_gate_frozen.py`（z 改名）。
  新增 `tools/analysis_pool_conditional.py`（条件/无条件分离）。
- [x] **T2 / D2：完整奖、同一期合并计税、费用与估算标签。**
  产物：`tools/profit_accounting.py`（`settle_period` + `SimAccount`）、
  `tests/test_profit_accounting.py`（18 项，实测通过）、
  账本样例 `corrections/ledger_s1_oos.json`。
  另有：`tools/dlt_prize_rules.py` 在一致性测试通过后改为适配层（消除重复实现）；
  `tests/test_profit_accounting.py::test_duplicate_prize_implementations_agree`。
  税务来源已一手核验并存档（公告2024年第12号，2024-09-01 起执行）。
- [x] **T3 / D3：开奖前机会地图、浮动奖敏感性、盈利缺口。**
  产物：`tools/profit_opportunity.py`（`estimate_opportunity`）、
  `tests/test_profit_opportunity.py`（15 项，实测通过）、
  `tools/build_opportunity_map.py` → `corrections/opportunity_map_26117.json`；
  报告 `docs/project_notes/PROFIT_V2_D2_D3_LEDGER_AND_OPPORTUNITY.md`。
  **关卡结论：全部 8 个开奖前情景均为负 → 停止"仅靠奖金条件盈利"这一分支。**
- [x] **T4 / D4：最多3族×2变体的冻结复核，复用缓存。**
  产物：`docs/project_notes/PROFIT_V2_D4_PROTOCOL.md`（先冻结）、
  `tools/eval_profit_v2.py`、`corrections/eval_profit_v2.json`、
  报告 `docs/project_notes/PROFIT_V2_D4_RESULTS.md`。
  **结论：四个政策税后 ROI −73.5% ~ −78.9%，无一优于同成本随机对照；
  该窗口可检出 11.8 元/期效应，而盈亏平衡需 66 元/期 → 排除"现有策略在此窗口
  达到税后盈亏平衡"。同预算下 1 组优于 3 组。**
- [x] T4 / D5（完整）：`tools/profit_budget.py`（`can_allocate` + `BudgetTracker`）+
  `tests/test_profit_budget.py`（17 项，实测通过）；0/1/2/3 组与 500 元限额已覆盖。
  另新增 `tools/sim_budget_v2.py`（预登记规则下的逐期预算模拟）与
  `tools/analysis_power_v2.py`（完整分布功效）。
  **500 / 1000 元均已实跑**，产物 `corrections/budget_v2.json`、
  `budget_start_sensitivity.json`、`power_v2.json`；
  报告 `docs/project_notes/PROFIT_V2_D5_BUDGET.md`。
  **结论：500 元平均只能参与 4.06 期、1000 元 9.37 期；450 个起点无一为正；
  三个指标（额度不足停止／触及上限／超限）全部分开报告，超限恒为 0。**
- [ ] T5 / D6：**未执行** —— 按 PLAN 为"仅有候选时修复"；本轮无候选。
  `tools/forward_log.py` 的已知缺陷（仅凭本地 CSV 判断开奖状态、`latest+1` 推目标期、
  按期号去重会遮蔽同期多策略、缺 as_of/hash/版本）**仍然存在**，未改动。
- [x] **T6 / D7：真实测试结果、证据索引、本周继续/停止裁决。**
  产物：`docs/project_notes/PROFIT_V2_WEEK1_VERDICT.md`。
  测试实测：**112 passed**（6 个文件）。
  **结论：无可信候选 → 结束本轮盈利研究，保留工具，不自动续期。**

## 本轮修正记录（用户定向要求）

- [x] **撤回 D4 功效结论**：不再用"未出现大奖的 500 期样本 sd"推导完整收益功效；
      改为**完整分布**计算（含大奖、税、分摊假设），结论为"**功效未确定**"。
- [x] **d3 机会地图修正**：实施真正的 `as_of` 校验（`published_at` + 来源期次）；
      修正高奖池反事实的**字典键类型错误**（原 `update(ERA3_HIGH_AMOUNT)`
      只塞了字符串键，未改变任何 (前区,后区) 奖金）；奖级分解改用与机会估计
      **同一套浮动奖情景**；区分"已测情景全为负"与"所有奖金条件都不可能盈利"；
      多组分布明确标注"随机组 ≠ 固定策略的相关结构"。
- [x] **D5 真正执行**：500/1000 元 × 0/1/2/3 组；先检查额度再参与；
      三个指标分开；减仓规则事先固定为**不允许**（另附标注为违反规则的敏感性）。
- [x] **证据维护**：原始 `SUMMARY.json` **不可恢复**，已如实记录（含 sha256）；
      后续产物只写新目录；修正"6,471 元回撤 = 全部票款"的错误表述；
      核对测试数量（112）。
- [x] **新增回归测试** `tests/test_v2_corrections.py`（本轮 16 项 → 勘误后 21 项），逐条覆盖上述错误。

## 最后一轮有限勘误（r3，2026-10-13）

- [x] **修正 5+12 一等奖概率**：一组 5+12 **后区 1~12 全包**，
      因此 `P(一等奖) = P(前区 5 中) = 1/C(35,5)`，**不得再除以 66**。
      旧写法（除以 66）把 500 期内至少一次一等奖算成 0.0023%，
      正确值为 **0.1539%**（相差 66 倍）。
      受影响字段已重新生成到**新目录**
      `outputs/2026-10-13_5plus12_correction_r3/corrections/power_v3.json`。
- [x] **删除把"约 46 万期""可排除 2000 元/期"当作确定结论的表述**：
      改为显式标注的**正态近似量级示例**（字段名前缀
      `NORMAL_APPROX_EXAMPLE_ONLY_`，`status = 示例，不是结论`），
      并新增 `reliable_power.status = "未确定"`。
- [x] **清理残留**：ROADMAP / TASKS / D4 / D5 / WEEK1_VERDICT 中
      已撤回功效结论的残留表述全部改写；
      **450 个起点明确标注为"高度重叠，不是独立验证"**。
- [x] **统一最终结论**（见本文件顶部「结论（统一表述）」）。
- [x] 新增 5 项回归测试（概率口径、0.1539% 断言、字段标注、示例字段命名、文档残留检查）。

**未做**：不扩展实验、不全量重跑、不新增策略、不自动续期、不启用前瞻或实际投注、不提交推送。

## V3 派奖叠加实验（2026-10-13，仅模拟）

- [x] **E0 冻结与保护**：新建 `outputs/2026-10-13_5plus12_v3_bonus/`，
      记录代码/数据 hash、依赖版本、git 状态、四格设计与**数据暴露声明**；旧目录未动。
- [x] **E1 派奖与金额精度**：证实并修复两处缺口（派奖只返回标记从未加钱；
      整数元截断半元）；改用 `Decimal`（`prec=60`）禁止逐注截断；
      四格目标值验证 A=675、B=875、**C=1125**、D=1470。
- [x] **E2 精确四格与门槛**：`tools/bonus_feasibility_v3.py`
      （`evaluate_rule_cell` / `solve_break_even_region`）；
      浮动奖必须带 `historical_pair`/`assumed_pair`/`rule_envelope` 标签，拒绝无标签 fallback。
- [x] **E3 门槛检验**：103 期（26014–26116）**情景期望为正的期数 = 0**；
      最近 718 期 `a+20b` 从未达到 D 格门槛 → **按 PLAN E3 直接收尾，未做预算扩展**。
- [x] **E4 交付**：`tests/test_bonus_feasibility_v3.py`（31 passed）；
      受影响的现有测试 56 passed；全量 **148 passed**；
      报告 `docs/project_notes/PROFIT_V3_BONUS_RESULTS.md`。

- [x] **对抗性复核（3 镜头 + 3 裁判，全部 UPHELD）**：复核者独立重写脚本
      （未 import 被检验模块）逐位复现四格收益、四个盈亏平衡门槛、103 期逐期结果。
      **裁判另发现并已修复一个仓库真实缺陷**：`tools/fetch_official_dlt.py::_num()`
      用 `int(float(s))` 把官方以字符串公布的半元金额 `'7.50'` 截断成 7
      （全历史受影响 **202 行**：九等奖派奖 2.0→2.5、八等奖派奖 7.0→7.5、
      六等奖派奖 7.0→7.5）。已改为保留 `float` 并用**既有原始响应**重新解析
      （不重新抓网），修正快照写入 V3 目录、**未覆盖**旧快照。
      对 D4/D5 的 500 期窗口影响 = **+0.2223 元/期**（51 期受影响），
      对 V3 四格影响 = **0**。新增 3 项回归测试。

**未做**：预算模拟、前瞻、实际投注、模型训练、全量回测、部署、提交推送。

## 待确定

- [ ] 硬亏损限额500还是1000元；当前500元仅模拟。
- [ ] 研发现金预算；默认不新增付费资源。
- [ ] 未来如要求实际试验，再明确本金、回撤及授权；本轮不执行。

## 已知未修复缺陷（D6 未执行）

`tools/forward_log.py`（上一轮产物，本轮未改动）：

- 仅用本地 `data/dlt/data.csv` 判断"是否已开奖"，**过期 CSV 会让未开奖的期次被当成已开奖**；
- 以 `latest + 1` 推目标期，**跨年/休市时会错**；
- 结算去重只按 `target_issue`，**同一期多策略会互相遮蔽**；
- 记录里缺 `as_of` / 时区 / 数据与模型 hash / `protocol_id`，无法证明"先于开奖登记"。

以上在启用前瞻记录**之前**必须修复（T5/D6）。

## 本轮不做

真实投注、自动调度、生产数据更新、新模型、全量重跑、P3/P4、无边界搜索、Git提交/推送。必要时仅重算受纠错影响的派生结果。

历史任务账本保存在 [历史TASKS](docs/project_notes/archive/TASKS_before_profit_plan_v2_2026-10-13.md)。原测试和完成状态是上一轮执行者报告，本次规划未重跑验收。
