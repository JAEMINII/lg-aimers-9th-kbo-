# -*- coding: utf-8 -*-
"""팀별 라벨 계단 스캔 — 월별 (팀관여 R행 성공률 − 리그) 시계열에서 지속 단절 탐지."""
import sys, numpy as np, pandas as pd
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
tr=pd.read_csv("open (1)/data/train.csv",encoding="utf-8-sig",
    usecols=["season","game_month","game_type","pitcher_team_id","batter_team_id","control_success"])
tr=tr[tr.game_type=="R"]
tr["ym"]=tr.season*100+tr.game_month
lg=tr.groupby("ym")["control_success"].mean()
months=sorted(lg.index)
teams=sorted(set(tr.pitcher_team_id)|set(tr.batter_team_id))
P("팀 | 최대 지속단절 (전후 각>=5개월) | 최근 12개월 내 시작된 단절 후보")
for T in teams:
    inv=tr[(tr.pitcher_team_id==T)|(tr.batter_team_id==T)]
    s=inv.groupby("ym")["control_success"].agg(["mean","size"])
    s=s[s["size"]>=800]
    d=(s["mean"]-lg.reindex(s.index)).dropna()
    if len(d)<12: continue
    vals=d.to_numpy(); idx=list(d.index)
    best=None
    for k in range(5,len(vals)-4):
        pre=vals[max(0,k-10):k].mean(); post=vals[k:k+10].mean()
        gap=post-pre
        if best is None or abs(gap)>abs(best[0]): best=(gap,idx[k])
    # 최근 단절: 2024-01 이후 시작, 전 8개월 vs 후 전부(>=3개월)
    rec=None
    for k in range(len(vals)-3,4,-1):
        if idx[k]<202401: break
        pre=vals[max(0,k-8):k].mean(); post=vals[k:].mean()
        if len(vals)-k>=3:
            gap=post-pre
            if rec is None or abs(gap)>abs(rec[0]): rec=(gap,idx[k],len(vals)-k)
    o=f"  팀{int(T):2d}  최대 {100*best[0]:+.2f}pp @{best[1]}"
    if rec: o+=f"   | 최근 {100*rec[0]:+.2f}pp @{rec[1]} (후속 {rec[2]}개월)"
    P(o)
P("")
P("== 13팀 대조 (검증): 2023-05 부근이 최대로 나와야 함")
