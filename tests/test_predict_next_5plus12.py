from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools import predict_next_5plus12 as n


def hist():
    return pd.DataFrame([[25150,1,2,3,4,5,1,2]],columns=['期数']+n.FRONT_COLS+n.BACK_COLS)


def draw(i,balls='01 02 03 04 05 01 02'):
    return dict(lotteryGameNum='85',lotteryDrawNum=i,lotteryDrawResult=balls)


def test_merge_checks_overlap_ranges_cross_year():
    h=hist()
    assert list(n.merge(h,[draw(26001),draw(25150)])['期数'])==[25150,26001]
    for rows in ([draw(26001)], [draw(25150),draw(26002)], [draw(25150),draw(25150)],
                 [draw(25150,'1 2 3 4 6 1 2')],[draw(25150),draw(26001,'1 1 3 4 5 1 2')]):
        with pytest.raises(ValueError): n.merge(h,rows)


def test_schedule_not_just_increment():
    t=datetime(2026,11,3,12,tzinfo=n.CST)
    assert n.scheduled_slot(26125,t,None)[0]==26126
    for last,dt in [(26124,t),(26125,t.replace(day=4,hour=21)),(26125,t.replace(month=12))]:
        with pytest.raises(ValueError): n.scheduled_slot(last,dt,None)


def test_live_and_closed_status():
    row=dict(draw(26126,''),lotterySaleBeginTime='2026-11-02 21:10:00',
             lotterySaleEndtime='2026-11-04 21:00:00')
    t=datetime(2026,11,3,12,tzinfo=n.CST)
    assert n.sale(row,26125,t,None)[0]==26126
    for other in [dict(row,lotteryDrawResult='01'),dict(row,lotteryGameNum=1),
                  dict(row,lotterySaleEndtime='2026-11-03 12:00:00')]:
        with pytest.raises(ValueError):n.sale(other,26125,t,None)


def test_publish_failure_then_concurrent_repeat(tmp_path,monkeypatch):
    monkeypatch.setattr(n,'ROOT',tmp_path)
    import tools.front_probability_mlp as mlp
    monkeypatch.setattr(n,'snapshot',lambda f:(hist().to_csv(f/'history.csv'),hist())[1])
    t=datetime(2026,11,3,12,tzinfo=n.CST)
    monkeypatch.setattr(n,'now',lambda:t)
    row=dict(draw(26126,''),lotterySaleBeginTime='2026-11-02 21:10:00',lotterySaleEndtime='2026-11-04 21:00:00')
    monkeypatch.setattr(n,'status',lambda *a:(26126,dict(row=row)))
    h=hist();h.iloc[0,0]=26125
    monkeypatch.setattr(n,'snapshot',lambda f:(h.to_csv(f/'history.csv'),h)[1])
    args=SimpleNamespace(issue=26126,sale_url=n.DIGITAL,device='cpu',train_only=False)
    def fail(*a):raise RuntimeError('MLP failed')
    monkeypatch.setattr(n,'train',fail)
    with pytest.raises(RuntimeError):n.locked_run(args)
    assert not (tmp_path/'26126.json').exists()
    def train(data,folder,*a):
        for m in ('U','F','R'):(folder/(m+'.pkl')).write_bytes(n.pickle.dumps({}))
    monkeypatch.setattr(n,'train',train)
    monkeypatch.setattr(mlp,'load_mlp_candidate',lambda *a:{})
    monkeypatch.setattr(n,'build_feature_for_next_issue',lambda *a:None)
    monkeypatch.setattr(n,'predict_candidate',lambda *a:[dict(p=np.ones(35)/7,top5=[1,2,3,4,5])])
    with ThreadPoolExecutor(2) as pool:res=list(pool.map(lambda _: n.locked_run(args),range(2)))
    assert res[0]['front']==res[1]['front'] and res[1]['tickets']==66
    saved=(tmp_path/'26126.json').read_bytes()
    monkeypatch.setattr(n,'snapshot',fail)
    assert n.locked_run(args)==json.loads(saved)
