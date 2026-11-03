"""On-demand 5+12 research forecast. Run from the repository root on Linux."""
import argparse
import fcntl
import hashlib
import html
import json
import os
import pickle
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from feature_engineering import (FRONT_COLS, BACK_COLS, build_feature_for_next_issue,
                                 build_front_training_dataset, load_dlt_history)
from tools.front_probability_models import fit_candidate, predict_candidate

CST = timezone(timedelta(hours=8))
API = 'https://webapi.sporttery.cn/gateway/lottery/'
DIGITAL = API + 'getDigitalDrawInfoV1.qry?param=85,0&isVerify=1'
SCHEDULE = [
    ('https://www.wztc.cn/system/2026/10/30/105863888.shtml','gb18030',
     '自11月2日晚开售的第26126期开始，至11月30日晚停售的第26137期结束，共12期'),
    ('https://www.js-lottery.com/tzgg/tzgg/cms/post-148036.html','utf8',
     '第26120期（2026年10月19日21:10开售）至第26135期（2026年11月25日21:00止售）'),
    ('https://www.js-lottery.com/tzgg/tzgg/cms/post-137115.html','utf8','游戏销售截止时间由当期开奖日20:00调整为21:00'),
    ('https://m.lottery.gov.cn/ksjz/m/yxgz_dlt/','utf8','超级大乐透每周一、三、六开奖（休市日除外）'),
]
ROOT = Path('outputs/next_5plus12')
CFG = Path('configs/front_probability_f1.json')
OFFICIAL = {'webapi.sporttery.cn', 'www.lottery.gov.cn', 'm.lottery.gov.cn'}


def now():
    return datetime.now(CST)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      default=lambda x: x.tolist(), allow_nan=False).encode()


def get(url):
    u = urlparse(url)
    if u.scheme != 'https' or u.hostname not in OFFICIAL or u.username or u.password or u.port:
        raise ValueError('需要白名单内的官方 HTTPS 接口')
    # No redirect: never assign official provenance to another host.
    raw = subprocess.check_output(['curl', '-fsS', '--max-time', '20', '--retry', '2', url])
    return json.loads(raw)


def check_balls(balls):
    if len(balls) != 7 or any(type(x) is not int for x in balls):
        raise ValueError('开奖号码不是7个整数')
    for group, limit in ((balls[:5], 35), (balls[5:], 12)):
        if len(set(group)) != len(group) or not all(1 <= x <= limit for x in group):
            raise ValueError('号码重复或越界')
    return sorted(balls[:5]) + sorted(balls[5:])


def continuous(issues):
    for x, y in zip(issues, issues[1:]):
        if not (y == x+1 or (y//1000 == x//1000+1 and y%1000 == 1)):
            raise ValueError(f'缺期或重复: {x} -> {y}')


def merge(history, rows):
    if history['期数'].duplicated().any():
        raise ValueError('冻结输入重复期')
    cols = FRONT_COLS + BACK_COLS
    data = {int(i): check_balls([int(x) for x in v]) for i, v in
            zip(history['期数'], history[cols].values)}
    observed = {}
    for r in rows:
        if str(r['lotteryGameNum']) != '85':
            raise ValueError('彩种不符')
        i = int(r['lotteryDrawNum'])
        balls = check_balls(list(map(int, r['lotteryDrawResult'].split())))
        if i in observed or (i in data and data[i] != balls):
            raise ValueError(f'重复或冲突的官方记录: {i}')
        observed[i] = balls
    if not data.keys() & observed.keys():
        raise ValueError('缺少冻结快照重叠记录')
    data.update(observed)
    ids = sorted(data)
    continuous(ids)
    if ids[-1] != max(observed):
        raise ValueError('官方数据早于本地')
    return pd.DataFrame([[i]+data[i] for i in ids], columns=['期数']+cols)


def snapshot(folder):
    hist = load_dlt_history('data/dlt/data.csv')
    if sha(Path('data/dlt/data.csv').read_bytes()) != json.loads(CFG.read_bytes())['data']['sha256']:
        raise ValueError('基础快照 SHA 不匹配')
    rows = []
    for page in range(1, 101):
        url = API + f'getHistoryPageListV1.qry?gameNo=85&pageSize=100&pageNo={page}&provinceId=0&isVerify=1'
        d = get(url)
        if not d.get('success') or int(d['value']['pageNo']) != page:
            raise ValueError('官方分页请求失败')
        page_rows = d['value']['list']
        (folder/f'page{page}.json').write_bytes(dump(dict(url=url, retrieved=now().isoformat(), response=d)))
        rows.extend(page_rows)
        if min(int(r['lotteryDrawNum']) for r in page_rows) <= hist['期数'].max():
            break
    data = merge(hist, rows)
    last = get(DIGITAL)['value']['dlt']
    if int(last['lotteryDrawNum']) != data.iloc[-1,0] or check_balls(list(map(int,last['lotteryDrawResult'].split()))) != list(data.iloc[-1,1:]):
        raise ValueError('独立官方最新开奖核对失败')
    (folder/'latest.json').write_bytes(dump(dict(url=DIGITAL,retrieved=now().isoformat(),response=last)))
    data.to_csv(folder/'history.csv', index=False)
    return data


def sale(d, last, when, issue=None):
    """Require an actual official unsatisfied draw row, not term arithmetic."""
    if str(d.get('lotteryGameNum')) != '85' or d.get('lotteryDrawResult') not in ('', None):
        raise ValueError('不是可核验的未开奖大乐透销售记录')
    target = int(d['lotteryDrawNum'])
    start,end = [datetime.strptime(d[k], '%Y-%m-%d %H:%M:%S').replace(tzinfo=CST)
                 for k in ('lotterySaleBeginTime','lotterySaleEndtime')]
    continuous([last,target])
    if target <= last or (issue is not None and target != issue) or not start <= when < end:
        raise ValueError('期号不符、未开售或已经停售')
    return target,end


def scheduled_slot(last, when, issue):
    # Only the intersection of two published, independently anchored schedules.
    # Beyond this window a new official schedule or live sales row is required.
    def enumerate_window(start, end, first, final):
        ds=pd.date_range(start,end)
        dates=[d.date() for d in ds if d.weekday() in (0,2,5)]
        if len(dates)!=final-first+1:raise ValueError('官方期数与日历不符')
        return dict(zip(range(first,final+1),dates))
    js=enumerate_window('2026-10-20','2026-11-25',26120,26135)
    zj=enumerate_window('2026-11-03','2026-11-30',26126,26137)
    for t,day in zj.items():
        if js.get(t)!=day:continue
        prev=js[t-1]
        # Conservative publication gate: do not publish on opening evening.
        start=datetime.combine(prev+timedelta(days=1),datetime.min.time(),CST)
        end=datetime.combine(day,datetime.min.time(),CST).replace(hour=21)
        if start<=when<end and t-1==last and issue in (None,t):
            return t,start,end
    raise ValueError('无覆盖当前时刻且与最新开奖一致的双源公告日程')


def status(url, last, issue):
    obj = get(url)
    def visit(v):
        if isinstance(v, dict):
            if 'lotterySaleBeginTime' in v:
                yield v
            for x in v.values():
                yield from visit(x)
        elif isinstance(v,list):
            for x in v:
                yield from visit(x)
    good = {}
    for row in visit(obj):
        try:
            target,end = sale(row,last,now(),issue)
        except (KeyError, TypeError, ValueError):
            continue
        good[(target,end.isoformat(),row['lotterySaleBeginTime'])] = row
    if not good and url==DIGITAL:
        latest=obj['value']['dlt']
        if int(latest['lotteryDrawNum'])!=last or not latest.get('lotteryDrawResult'):
            raise ValueError('公告日程与最新开奖状态不一致')
        target,start,end=scheduled_slot(last,now(),issue)
        evidence=[]
        for src,enc,fragment in SCHEDULE:
            raw=subprocess.check_output(['curl','-fsSL','--proto-redir','=https','--max-time','20',src])
            plain=re.sub(r'\s+','',html.unescape(re.sub('<[^>]+>','',raw.decode(enc))))
            if fragment not in plain:raise ValueError('官方销售公告内容不符: '+src)
            evidence.append(dict(url=src,sha256=sha(raw),text=fragment))
        row=dict(lotteryGameNum='85',lotteryDrawNum=target,lotteryDrawResult='',
                 lotterySaleBeginTime=start.strftime('%Y-%m-%d %H:%M:%S'),
                 lotterySaleEndtime=end.strftime('%Y-%m-%d %H:%M:%S'))
        sale(row,last,now(),target)
        return target,dict(method='published_schedule_not_live_terminal',url=url,
                           latest_draw=last,retrieved=now().isoformat(),sources=evidence,row=row,
                           start_note='公告开售次日零点起才发布，并非实际开售时刻')
    if len(good) != 1:
        raise ValueError('未取得唯一官方在售期及其停售时间；不发布号码')
    key,row = good.popitem()
    return key[0],dict(url=url,retrieved=now().isoformat(),row=row)


def train(data, folder, cfg, device):
    from tools.front_probability_mlp import save_mlp_candidate
    cfg=json.loads(json.dumps(cfg))
    cfg['resources']['device']=device
    X,y,ids=build_front_training_dataset(data)
    assert X.shape[1]==cfg['features']['expected_dim'] and (y.sum(axis=1)==5).all()
    models={}
    for mid in ('N','F','R','U'):
        fitted=fit_candidate(mid,X,y,ids,cfg)
        if mid=='N':
            if len(fitted['seeds'])!=3:
                raise ValueError('MLP种子不完整')
            save_mlp_candidate(fitted,str(folder/'N.pt'))
        else:
            with (folder/f'{mid}.pkl').open('wb') as f:
                pickle.dump(fitted,f)
        models[mid]=fitted
    return models


def locked_run(args):
    from tools.front_probability_mlp import load_mlp_candidate
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.issue and (ROOT/f'{args.issue}.json').exists():
            return json.loads((ROOT/f'{args.issue}.json').read_bytes())
        folder=Path(tempfile.mkdtemp(prefix='run-',dir=ROOT))
        data=snapshot(folder)
        last=int(data['期数'].iloc[-1])
        if not args.train_only:
            target,before=status(args.sale_url,last,args.issue)
            dest=ROOT/f'{target}.json'
            if dest.exists():
                return json.loads(dest.read_bytes())
        gitsha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        cfg=json.loads(CFG.read_bytes())
        key=sha(dump([sha((folder/'history.csv').read_bytes()),sha(CFG.read_bytes()),gitsha,args.device]))
        cache=ROOT/('models-'+key)
        if not cache.exists():
            tmp=Path(tempfile.mkdtemp(prefix='fitting-',dir=ROOT))
            train(data,tmp,cfg,args.device)
            (tmp/'source.json').write_bytes(dump(dict(snapshot=str(folder),code=gitsha,config=cfg,device=args.device,last=last)))
            tmp.rename(cache)
        if args.train_only:
            return dict(models=str(cache),train_end=last,rows=len(data))
        X=build_feature_for_next_issue(data)
        records={}
        for mid in ('N','F','R','U'):
            m=load_mlp_candidate(str(cache/'N.pt'),args.device) if mid=='N' else pickle.loads((cache/f'{mid}.pkl').read_bytes())
            r=predict_candidate(m,X,[target],data)[0]
            if len(r['p'])!=35 or not np.isfinite(r['p']).all() or abs(sum(r['p'])-5)>1e-6 or len(set(r['top5']))!=5:
                raise ValueError('模型输出不合法')
            records[mid]=r
        target2,after=status(args.sale_url,last,target)
        if after['row']['lotterySaleEndtime']!=before['row']['lotterySaleEndtime']:
            raise ValueError('停售时刻已变，请重新核验')
        out=dict(issue=target2,train_end=last,created=now().isoformat(),front=sorted(records['N']['top5']),
                 back=list(range(1,13)),cost=132,tickets=66,models=str(cache),snapshot=str(folder),
                 snapshot_hash=sha((folder/'history.csv').read_bytes()),config_hash=sha(CFG.read_bytes()),
                 code=gitsha,sale_before=before,sale_after=after,records=records)
        temp=folder/'prediction.json'
        temp.write_bytes(dump(out))
        sale(after['row'],last,now(),target)
        os.link(temp,dest)
        return out


def evaluate(issue):
    pred=json.loads((ROOT/f'{issue}.json').read_bytes())
    folder=Path(tempfile.mkdtemp(prefix='result-',dir=ROOT))
    d=snapshot(folder)
    r=d.loc[d['期数']==issue,FRONT_COLS]
    if len(r)!=1:raise ValueError('官方尚无该期开奖结果')
    front=list(map(int,r.iloc[0]))
    y=np.isin(np.arange(1,36),front)
    result=dict(issue=issue,front=front,snapshot=str(folder),scores={
        m:dict(hits=len(set(x['top5'])&set(front)),brier=float(np.mean((np.array(x['p'])-y)**2)))
        for m,x in pred['records'].items()})
    tmp=folder/'result.json';tmp.write_bytes(dump(result));dest=ROOT/f'{issue}.result.json'
    try:os.link(tmp,dest)
    except FileExistsError:return json.loads(dest.read_bytes())
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--issue',type=int);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--sale-url',default=DIGITAL);ap.add_argument('--train-only',action='store_true')
    ap.add_argument('--evaluate',type=int)
    args=ap.parse_args()
    try:
        result=evaluate(args.evaluate) if args.evaluate else locked_run(args)
        print(dump({k:v for k,v in result.items() if k not in ('records','sale_before','sale_after')}).decode())
    except (ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as e:
        ap.exit(2,f'{e}\n')
