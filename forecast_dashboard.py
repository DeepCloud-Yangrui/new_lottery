"""Read-only Streamlit view of on-demand 5+12 forecasts."""
import json
import inspect
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "outputs" / "next_5plus12"
MODELS = {"N": "神经网络 · 主模型", "F": "平滑频次", "R": "随机森林", "U": "均匀基线"}


def balls(values, limit, count):
    if len(values) != count or any(type(v) is not int or not 1 <= v <= limit for v in values) or len(set(values)) != count:
        raise ValueError("号码范围或个数不符")
    return "　".join(f"{v:02d}" for v in sorted(values))


def read_issue(root, issue):
    record = json.loads((root / f"{issue}.json").read_text(encoding="utf-8"))
    if record["issue"] != issue or record["train_end"] >= issue:
        raise ValueError("期号或训练截止期不符")
    front = balls(record["front"], 35, 5)
    balls(record["back"], 12, 12)
    matrix = {}
    for key, name in MODELS.items():
        r = record["records"][key]
        if r["issue"] != issue or (key == "N" and balls(r["top5"], 35, 5) != front):
            raise ValueError("模型记录与主预测不一致")
        p = r["p"]
        if len(p) != 35 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p)-5)>1e-6:
            raise ValueError("边际概率无效")
        matrix[name] = p
    return record, pd.DataFrame(matrix, index=pd.Index(range(1,36),name="前区号码"))


def render(root=DATA):
    st.caption("超级大乐透　/　预测研究")
    st.title("5+12 预测看板")
    issues = sorted((int(p.stem) for p in root.glob("*.json") if p.stem.isdecimal()), reverse=True)
    if not issues:
        st.info("还没有已保存的预测。请先在研究服务器运行预测入口，再刷新本页。")
        return
    issue = st.selectbox("预测期号", issues, format_func=lambda i:f"第 {i} 期")
    try:
        r,probs=read_issue(root,issue)
        close=datetime.fromisoformat(r["sale_after"]["row"]["lotterySaleEndtime"]).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        created=datetime.fromisoformat(r["created"]).astimezone(close.tzinfo)
    except (OSError,KeyError,TypeError,ValueError) as exc:
        st.error(f"本期记录无法核验：{exc}")
        return
    st.caption(f"生成于 {created:%Y-%m-%d %H:%M:%S}（北京）　·　训练截至第 {r['train_end']} 期")
    with st.container(**({"border":True} if "border" in inspect.signature(st.container).parameters else {})):
        st.caption("主模型预测　/　前区五个号码")
        st.header(balls(r["front"],35,5))
        st.write("后区 **01–12 全包**　·　66 注　·　单组成本 **132 元**")
    state="已过计划停售时间" if datetime.now(close.tzinfo)>=close else "尚未到计划停售时间"
    st.caption(f"{state}　·　{close:%Y-%m-%d %H:%M}（北京）　·　非销售终端实时状态")
    a,b,c=st.tabs(["号码概率","开奖对照","记录来源"])
    with a:
        names=st.multiselect("比较策略",list(MODELS.values()),default=[MODELS["N"],MODELS["U"]])
        if names:
            axes = {"x_label":"前区号码","y_label":"入选概率（%）"} if "x_label" in inspect.signature(st.line_chart).parameters else {}
            st.line_chart(probs[names]*100,**axes)
        st.caption("每个点是一个号码入选前区的边际概率，不是五个号码同时命中的概率。历史比较尚未证实主模型优于均匀基线。")
    with b:
        result_file=root/f"{issue}.result.json"
        if not result_file.exists():
            st.info("尚未导入本期官方开奖结果。")
        else:
            try:
                result=json.loads(result_file.read_text(encoding="utf-8"))
                if result["issue"]!=issue:raise ValueError("结果期号不符")
                st.write("实际前区：**"+balls(result["front"],35,5)+"**")
                rows=[{"策略":name,"命中个数":result["scores"][k]["hits"],
                       "概率评分（Brier）":result["scores"][k]["brier"]} for k,name in MODELS.items()]
                st.dataframe(pd.DataFrame(rows),hide_index=True)
                st.caption("Brier 越低越好；单期结果不能证明长期优势。")
            except (OSError,KeyError,TypeError,ValueError) as exc:
                st.error(f"无法读取本期评分：{exc}")
    with c:
        st.write("代码版本"); st.code(r["code"],language=None)
        st.write("训练快照 SHA256");st.code(r["snapshot_hash"],language=None)
        for s in r["sale_after"].get("sources",[]):
            st.write(s["url"])


if __name__=="__main__":
    st.set_page_config(page_title="5+12 预测看板",page_icon="🎯",layout="centered")
    render()
